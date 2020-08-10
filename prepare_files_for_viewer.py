#!/usr/bin/env python3
#
# This file will export the metadata into a format that the TV viewer can use.
#
# Please review: docs/TVNEWS_VIEWER_README.md for usage instruction

import argparse
from collections import defaultdict
import json
import csv
import os
from pathlib import Path
import shutil
from subprocess import check_call
from multiprocessing import Pool
import tempfile
from tqdm import tqdm
from typing import NamedTuple, List, Tuple, Dict, Set, Optional

from rs_intervalset.writer import (
    IntervalListMappingWriter, IntervalSetMappingWriter)

from util.consts import (FILE_BBOXES,
                         FILE_METADATA,
                         FILE_GENDERS,
                         FILE_CAPTIONS_ORIG,
                         FILE_CAPTIONS,
                         FILE_COMMERCIALS,
                         FILE_IDENTITIES,
                         FILE_IDENTITIES_PROP)
from util.utils import load_json, save_json


class Video(NamedTuple):
    id: int
    name: str
    num_frames: int
    fps: float
    width: int
    height: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path',
                        help='path to directory of ingest pipeline outputs.')
    parser.add_argument('out_path',
                        help='path to directory to output results to.')
    parser.add_argument('--index-dir', help=('path to directory where the '
                        'index should be located if different than out_path.'))
    parser.add_argument('--bbox-dir', help=('path to directory where the face '
                        'bboxes should be located if different than out_path'))
    overwrite_behavior = parser.add_mutually_exclusive_group()
    overwrite_behavior.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Overwrite existing output files.')
    overwrite_behavior.add_argument(
        '-u', '--update', action='store_true',
        help='Update existing files in place.')

    parser.add_argument('--host-file', type=str,
                        help='File containing list of hosts')

    parser.add_argument('--face-sample-rate', type=int, default=1,
                        help='Number of seconds per sample')
    return parser.parse_args()


def main(in_path, out_path, index_dir, bbox_dir, overwrite, update, host_file,
         face_sample_rate):
    if os.path.exists(out_path):
        if overwrite:
            shutil.rmtree(out_path)
        elif not update:
            raise FileExistsError('Output directory exists: {}'.format(out_path))
    elif update:
        raise FileNotFoundError('No existing files to update: {}'.format(out_path))

    os.makedirs(out_path, exist_ok=update)

    def get_out_path(*args):
        return os.path.join(out_path, *args)

    new_videos = load_videos(in_path)
    print('Found data for {} videos'.format(len(new_videos)))

    if update:
        all_videos, canonical_show_map = load_existing_video_metadata(get_out_path('videos.json'))

        # Check for duplicate videos
        all_video_names = set(x.name for x in all_videos)
        for v in new_videos:
            assert v.name not in all_video_names, v.name + ' already exists!'

        # Re-id the new videos
        next_video_id = max(v.id for v in all_videos) + 1
        print('Starting video ids from', next_video_id)
        new_videos = [v._replace(id=v.id + next_video_id) for v in new_videos]
        all_videos.extend(new_videos)
    else:
        print('Starting video ids from 0')
        all_videos = new_videos
        canonical_show_map = {}

    # Task 1: Write out video metadata file
    #
    # This is a JSON file with schema:
    # [
    #     [
    #         'video id', 'video name', 'show', 'channel', '# of frames',
    #         'fps', 'width', 'height'
    #     ],
    #     ...
    # ]
    print('Saving video metadata')
    save_json([get_video_metadata(v, canonical_show_map) for v in all_videos],
               get_out_path('videos.json'))

    # Task 2: Write commercials intervals file
    print('Saving commercial intervals')
    with IntervalSetMappingWriter(
        get_out_path('commercials.iset.bin'), append=update
    ) as writer:
        for video in tqdm(new_videos):
            comm_intervals = get_commercial_intervals(in_path, video)
            if comm_intervals:
                writer.write(video.id, comm_intervals)

    # Task 3: Write out identitity metadata file
    #
    # This is a JSON file with schema:
    # {
    #     "<person name>": ["journalist", "politician", ...],
    #     ...
    # }
    # FIXME: this is an empty dictionary for now (no updates supported either)
    if not update:
        print('Saving person metadata')
        save_json({}, get_out_path('people.metadata.json'))

    # Task 4: Write out face bounding boxes
    #
    # This is to render bounding boxes on the frames, with information such as
    # gender and identity.
    print('Saving face bounding boxes')
    face_bbox_dir = bbox_dir if bbox_dir else get_out_path('face-bboxes')
    os.makedirs(face_bbox_dir, exist_ok=update)
    with Pool() as workers:
        bbox_tasks = [(in_path, video, face_sample_rate, face_bbox_dir)
                      for video in new_videos]
        for _ in tqdm(workers.imap_unordered(
            save_bboxes_for_video, bbox_tasks
        ), total=len(bbox_tasks)):
            pass

    # Task 5 & 6: Write out intervals for all faces on screen and separate
    # files for identities
    print('Saving face intervals')
    if not host_file:
        print('No host file specified: the host flag will not be set')
        host_dict = {}
    else:
        print('Host file:', host_file)
        host_dict = read_host_csv(host_file)

    people_ilist_dir = get_out_path('people')
    os.makedirs(people_ilist_dir, exist_ok=update)
    person_ilist_writers = {}
    with Pool() as workers, IntervalListMappingWriter(
        get_out_path('faces.ilist.bin'),
        1,   # 1 byte of binary payload
        append=update
    ) as writer:
        face_ilist_tasks = [(in_path, video, face_sample_rate, host_dict)
                            for video in new_videos]
        for video, all_face_intervals, person_face_intervals in tqdm(
            workers.imap(get_face_intervals_for_video, face_ilist_tasks),
            total=len(face_ilist_tasks)
        ):
            if len(all_face_intervals) > 0:
                writer.write(video.id, all_face_intervals)
            for person_name, person_intervals in person_face_intervals.items():
                if person_name not in person_ilist_writers:
                    person_ilist_path = os.path.join(
                        people_ilist_dir,
                        '{}.ilist.bin'.format(person_name.lower()))
                    if update and not os.path.isfile(person_ilist_path):
                        # Skip the person, since their file does not exist
                        continue
                    person_ilist_writers[person_name] = IntervalListMappingWriter(
                        person_ilist_path,
                        1,   # 1 byte of binary payload
                        append=update
                    )
                person_ilist_writers[person_name].write(
                    video.id, person_intervals)
    # Close all of the identity writers
    for writer in person_ilist_writers.values():
        writer.close()

    # Task 7: index the captions
    print('Indexing the captions')
    index_dir = index_dir if index_dir else get_out_path('index')
    tmp_dir = None
    try:
        tmp_dir = collect_caption_files(in_path, new_videos)
        if len(os.listdir(tmp_dir)) == 0:
            print('No captions files exist!')

        else:
            if update:
                cmd = [
                    os.path.dirname(os.path.realpath(__file__))
                    + '/deps/caption-index/scripts/update_index.py',
                    '--skip-existing-names',
                    '-d', tmp_dir,
                    index_dir
                ]
            else:
                cmd = [
                    os.path.dirname(os.path.realpath(__file__))
                    + '/deps/caption-index/scripts/build_index.py',
                    '-d', tmp_dir,
                    '-o', index_dir
                ]
            check_call(cmd)

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    print('Done! Output written to: {}'.format(out_path))


# Bits 0-1 are gender
# Bit 2 is host
# Bits 3-7 are face height
def encode_face_interval_payload(gender, is_host, height):
    ret = 0
    ret |= gender
    if is_host:
        ret |= 1 << 2
    ret |= height << 3
    return ret


def load_videos(video_dir: str):
    videos = []
    # Start video IDs at 1 to ensure the ID passes boolean check
    for i, video_name in enumerate(os.listdir(video_dir), 1):
        meta_path = os.path.join(video_dir, video_name, FILE_METADATA)
        meta_dict = load_json(meta_path)
        videos.append(Video(
            id=i, name=video_name, num_frames=meta_dict['frames'],
            fps=meta_dict['fps'], width=meta_dict['width'],
            height=meta_dict['height']))
    return videos


def load_existing_video_metadata(fpath: str):
    existing_canonical_show_map = {}
    videos = []
    with open(fpath) as fp:
        for v in json.load(fp):
            vid, name, channel, canonical_show, num_frames, fps, width, height = v
            _, raw_show = get_channel_show(name)

            # Infer the canonical show mapping
            existing_canonical_show_map[
                (channel.lower(), raw_show.lower())
            ] = canonical_show

            videos.append(Video(vid, name, num_frames, fps, width, height))
    return videos, existing_canonical_show_map


def get_video_metadata(video: Video, canonical_show_map: Dict) -> Tuple:
    channel, raw_show = get_channel_show(video.name)
    canonical_show = canonical_show_map.get((channel.lower(), raw_show.lower()))
    return (
        video.id,
        video.name,
        raw_show if canonical_show is None else canonical_show,
        channel,
        video.num_frames,
        video.fps,
        video.width,
        video.height
    )


def get_commercial_intervals(video_dir: str, video: Video):
    fpath = os.path.join(video_dir, video.name, FILE_COMMERCIALS)
    if not os.path.exists(fpath):
        return None

    def format_commercial(interval):
        start_ms = int(interval[0] / video.fps * 1000)
        end_ms = int(interval[1] / video.fps * 1000)
        return (start_ms, end_ms)

    with open(fpath) as fp:
        commercials = json.load(fp)
        return [format_commercial(c) for c in commercials]


def get_face_intervals(video_dir: str, video: Video, face_sample_rate: int,
                       host_dict: Dict[str, Set[str]]):
    channel, _ = get_channel_show(video.name)

    def get_is_host(identity: Optional[str]):
        return (identity and channel in host_dict
                and identity.lower() in host_dict[channel])

    def load_file(name: str, second: str = '', optional: bool = False):
        fpath = os.path.join(video_dir, video.name, name)
        secondary = os.path.join(video_dir, video.name, second)
        if os.path.exists(fpath):
            return load_json(fpath)
        elif second and os.path.exists(secondary):
            return load_json(secondary)
        elif optional:
            return []
        else:
            raise FileNotFoundError(fpath)

    genders = {
        a: b
        for a, b, _ in load_file(FILE_GENDERS)
    }
    gender_dict = {'f': 0, 'm': 1, 'o': 2}
    identities = {
        a: b.lower()
        for a, b, _ in load_file(FILE_IDENTITIES_PROP, FILE_IDENTITIES, optional=True)
    }
    face_intervals = []
    person_face_intervals = defaultdict(list)
    for face_id, face_meta in load_file(FILE_BBOXES):
        start_time = face_meta['frame_num'] / video.fps
        start_ms = int(start_time * 1000)
        end_ms = start_ms + int(1000 * face_sample_rate)
        face_height = face_meta['bbox']['y2'] - face_meta['bbox']['y1']
        face_identity = identities.get(face_id)
        face_interval = (start_ms, end_ms, encode_face_interval_payload(
            gender_dict[genders.get(face_id, 'O').lower()],
            get_is_host(face_identity),
            min(round(face_height * 31), 31)    # 5-bits
        ))
        face_intervals.append(face_interval)
        if face_identity:
            person_face_intervals[face_identity].append(face_interval)

    return face_intervals, person_face_intervals


def get_face_intervals_for_video(args):
    in_path, video, face_sample_rate, host_dict = args
    return (
        video,
        *get_face_intervals(in_path, video, face_sample_rate, host_dict))


def get_channel_show(video_name: str):
    try:
        channel, _, _, show = video_name.split('_', 3)
    except ValueError:
        #print("For the TV News Viewer, video names must follow the format "
        #      "'CHANNEL_YYYYMMDD_hhmmss_SHOW'.")
        #exit()
        channel = video_name.split('_', 1)[0]
        show = ''

    if channel[-1] == 'W':
        channel = channel[:-1]
    return channel, show


def format_bbox_file_data(video_dir: str, video: Video, face_sample_rate: int):

    def load_file(name: str, second: str = '', optional: bool = False):
        fpath = os.path.join(video_dir, video.name, name)
        secondary = os.path.join(video_dir, video.name, second)
        if os.path.exists(fpath):
            return load_json(fpath)
        elif second and os.path.exists(secondary):
            return load_json(secondary)
        elif optional:
            return []
        else:
            raise FileNotFoundError(fpath)

    genders = {
        a: b
        for a, b, _ in load_file(FILE_GENDERS)
    }
    identities = {
        a: b.lower()
        for a, b, _ in load_file(FILE_IDENTITIES_PROP, FILE_IDENTITIES, optional=True)
    }
    # person IDs start at 1 to ensure ID passes any boolean check
    identity_to_id = {
        x: i
        for i, x in enumerate(sorted({x for x in identities.values()}), 1)
    }
    face_bboxes = []
    for face_id, face_meta in load_file(FILE_BBOXES):
        start_time = face_meta['frame_num'] / video.fps
        face_bbox = {
            't': [start_time, start_time + 1 * face_sample_rate],
            'b': [
                round(face_meta['bbox']['x1'], 2),
                round(face_meta['bbox']['y1'], 2),
                round(face_meta['bbox']['x2'], 2),
                round(face_meta['bbox']['y2'], 2)
            ]
        }
        if genders:
            gender = genders.get(face_id)
            if gender:
                face_bbox['g'] = gender.lower()
        if identities:
            identity = identities.get(face_id)
            if identity:
                face_bbox['i'] = identity_to_id[identity]
        face_bboxes.append(face_bbox)
    return {'faces': face_bboxes, 'ids': list(identity_to_id.items())}


def save_bboxes_for_video(args):
    in_path, video, face_sample_rate, face_bbox_dir = args
    try:
        save_json(
            format_bbox_file_data(in_path, video, face_sample_rate),
            os.path.join(face_bbox_dir, '{}.json'.format(video.id))
        )
    except Exception as e:
        print('Failed to write bboxes for {}: {}'.format(video.name, str(e)))


def collect_caption_files(video_dir: str, videos: List[Video]):
    tmp_dir = tempfile.mkdtemp()
    for video in videos:
        captions_file = FILE_CAPTIONS
        if not os.path.exists(os.path.join(video_dir, video.name, FILE_CAPTIONS)):
            if not os.path.exists(os.path.join(video_dir, video.name, FILE_CAPTIONS_ORIG)):
                print("No captions exist for video '{}'".format(video.name))
                continue

            captions_file = FILE_CAPTIONS_ORIG

        src_path = os.path.join(video_dir, video.name, captions_file)
        dst_path = os.path.join(tmp_dir, '{}.srt'.format(video.name))
        shutil.copyfile(src_path, dst_path)

    return tmp_dir


def read_host_csv(host_file):
    hosts = defaultdict(set)
    with open(host_file) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            hosts[row['channel']].add(row['name'].lower())
    return hosts


if __name__ == '__main__':
    main(**vars(get_args()))
