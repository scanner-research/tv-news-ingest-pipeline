#!/usr/bin/env python3
#
# This file will export the metadata into a format that the TV viewer can use.
#
# Please review: docs/TVNEWS_VIEWER_README.md for usage instruction

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from subprocess import check_call
import tempfile
from typing import NamedTuple, List, Tuple

from rs_intervalset.writer import (
    IntervalListMappingWriter, IntervalSetMappingWriter)

from util.consts import (FILE_BBOXES,
                         FILE_METADATA,
                         FILE_GENDERS,
                         FILE_CAPTIONS,
                         FILE_COMMERCIALS,
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
                        help='path todirectory to output results to.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite existing output files.')
    parser.add_argument('--face-sample-rate', type=int, default=1,
                        help='Number of samples per second')
    return parser.parse_args()


def main(in_path, out_path, force, face_sample_rate):
    if os.path.exists(out_path):
        if force:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError('Output directory exists: {}'.format(out_path))
    os.makedirs(out_path)

    videos = load_videos(in_path)
    print('Found data for {} videos'.format(len(videos)))

    def get_out_path(*args):
        return os.path.join(out_path, *args)

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
    save_json([get_video_metadata(v) for v in videos],
              get_out_path('videos.json'))

    # Task 2: Write commercials intervals file
    print('Saving commercial intervals')
    with IntervalSetMappingWriter(
        get_out_path('commercials.iset.bin')
    ) as writer:
        for video in videos:
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
    # FIXME: this is an empty dictionary for now
    print('Saving person metadata')
    save_json({}, get_out_path('people.metadata.json'))

    # Task 4: Write out face bounding boxes
    #
    # This is to render bounding boxes on the frames, with information such as
    # gender and identity.
    print('Saving face bounding boxes')
    face_bbox_dir = get_out_path('face-bboxes')
    os.makedirs(face_bbox_dir)
    for video in videos:
        try:
            save_json(
                format_bbox_file_data(in_path, video, face_sample_rate),
                os.path.join(face_bbox_dir, '{}.json'.format(video.id))
            )
        except Exception as e:
            print('Failed to write bboxes for {}: {}'.format(video.name, str(e)))

    # Task 5 & 6: Write out intervals for all faces on screen and separate
    # files for identities
    print('Saving face intervals')
    people_ilist_dir = get_out_path('people')
    os.makedirs(people_ilist_dir)
    person_ilist_writers = {}
    with IntervalListMappingWriter(
        get_out_path('faces.ilist.bin'),
        1   # 1 byte of binary payload
    ) as writer:
        for video in videos:
            all_face_intervals, person_face_intervals = get_face_intervals(
                in_path, video, face_sample_rate)
            if len(all_face_intervals) > 0:
                writer.write(video.id, all_face_intervals)
            for person_name, person_intervals in person_face_intervals.items():
                if person_name not in person_ilist_writers:
                    person_ilist_writers[person_name] = IntervalListMappingWriter(
                        os.path.join(
                            people_ilist_dir,
                            '{}.ilist.bin'.format(person_name)),
                        1   # 1 byte of binary payload
                    )
                    person_ilist_writers[person_name].write(
                        video.id, person_intervals)
    # Close all of the identity writers
    for writer in person_ilist_writers.values():
        writer.close()

    # Task 7: index the captions
    print('Indexing the captions')
    index_dir = get_out_path('index')
    tmp_dir = None
    try:
        tmp_dir = collect_caption_files(in_path, videos)
        check_call([
            os.path.dirname(os.path.realpath(__file__))
            + '/deps/caption-index/scripts/build_index.py',
            tmp_dir,
            '-o', index_dir
        ])
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    print('Done! Output written to: {}'.format(out_path))



# Bit 0 is gender
# Bit 1 is host
# Bits 2-7 are face height
def encode_face_interval_payload(gender, is_host, height):
    ret = 0
    ret |= gender
    if is_host:
        ret |= 1 << 2
    ret |= height << 3
    return ret


def load_videos(video_dir: str):
    videos = []
    for i, video_name in enumerate(os.listdir(video_dir)):
        meta_path = os.path.join(video_dir, video_name, FILE_METADATA)
        meta_dict = load_json(meta_path)
        videos.append(Video(
            id=i, name=video_name, num_frames=meta_dict['frames'],
            fps=meta_dict['fps'], width=meta_dict['width'],
            height=meta_dict['height']))
    return videos


def get_video_metadata(video: Video) -> Tuple:
    try:
        channel, _, _, show = video.name.split('_', 3)
    except ValueError:
        print("For the TV News Viewer, video names must follow the format "
              "'CHANNEL_YYYYMMDD_hhmmss_SHOW'. Exiting.")
        exit()

    return (
        video.id,
        video.name,
        show,
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


def get_face_intervals(video_dir: str, video: Video, face_sample_rate: int):

    def load_file(name: str, optional: bool = False):
        fpath = os.path.join(video_dir, video.name, name)
        if os.path.exists(fpath):
            return load_json(fpath)
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
        a: b
        for a, b, _ in load_file(FILE_IDENTITIES_PROP, optional=True)
    }
    face_intervals = []
    person_face_intervals = defaultdict(list)
    for face_id, face_meta in load_file(FILE_BBOXES):
        start_time = face_meta['frame_num'] / video.fps
        start_ms = int(start_time * 1000)
        end_ms = start_ms + int(1000 / face_sample_rate)
        face_height = face_meta['bbox']['y2'] - face_meta['bbox']['y1']
        face_interval = (start_ms, end_ms, encode_face_interval_payload(
            gender_dict[genders.get(face_id, 'O').lower()],
            False,
            min(round(face_height * 31), 31)    # 5-bits
        ))
        face_intervals.append(face_interval)
        identity = identities.get(face_id)
        if identity:
            person_face_intervals[identity].append(face_interval)
    return face_intervals, person_face_intervals


def format_bbox_file_data(video_dir: str, video: Video, face_sample_rate: int):

    def load_file(name: str, optional: bool = False):
        fpath = os.path.join(video_dir, video.name, name)
        if os.path.exists(fpath):
            return load_json(fpath)
        elif optional:
            return []
        else:
            raise FileNotFoundError(fpath)

    genders = {
        a: b
        for a, b, _ in load_file(FILE_GENDERS)
    }
    identities = {
        a: b
        for a, b, _ in load_file(FILE_IDENTITIES_PROP,  optional=True)
    }
    identity_to_id = {
        x: i
        for i, x in enumerate(sorted({x for x in identities.values()}))
    }
    face_bboxes = []
    for face_id, face_meta in load_file(FILE_BBOXES):
        start_time = face_meta['frame_num'] / video.fps
        face_bbox = {
            't': [start_time, start_time + 1 / face_sample_rate],
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
            identity = identity_to_id.get(face_id)
            if identity:
                face_bbox['i'] = identity_to_id[identity]
        face_bboxes.append(face_bbox)
    return {'faces': face_bboxes, 'ids': list(identity_to_id.items())}


def collect_caption_files(video_dir: str, videos: List[Video]):
    tmp_dir = tempfile.mkdtemp()
    for video in videos:
        src_path = os.path.join(video_dir, video.name, FILE_CAPTIONS)
        dst_path = os.path.join(tmp_dir, '{}.srt'.format(video.name))
        shutil.copyfile(src_path, dst_path)
    return tmp_dir


if __name__ == '__main__':
    main(**vars(get_args()))
