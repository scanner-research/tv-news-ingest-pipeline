#!/usr/bin/env python3

import argparse
import re
import os
import json
import shutil
from subprocess import check_output, check_call
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='year', type=int, required=True, help='The year for which to download videos.')
    parser.add_argument('--subs', dest='subs_dir', type=str, default='subs',
                        help='Directory to save captions to')
    parser.add_argument('--list', dest='list_file', type=str,
                        help='File to write the list of downloaded videos')
    parser.add_argument('--gcs-video-path', dest='gcs_video_path', type=str, default='gs://esper/tvnews/videos',
                        help='The path in Google cloud to which videos should be uploaded.')
    parser.add_argument('--gcs-caption-path', dest='gcs_caption_path', type=str, default=None,
                        help='The path in Google cloud to which captions/subtitles should be uploaded. If not specified, they are not uploaded')
    return parser.parse_args()


def parse_ia_identifier(s):
    """Split off the last"""
    return os.path.splitext(s.split('/')[-1])[0]


def list_downloaded_videos(year, gcs_video_path):
    """List the videos in the bucket"""
    output = check_output(
        ['gsutil', 'ls', '{}/*_{}*'.format(gcs_video_path, year)]).decode()
    videos = [x for x in output.split('\n') if x.strip()]
    return {parse_ia_identifier(x) for x in videos}


def list_ia_videos(year):
    prefixes = ['MSNBC', 'MSNBCW', 'CNN', 'CNNW', 'FOXNEWS', 'FOXNEWSW']
    identifiers = []
    identifier_re = re.compile(r'^[A-Z]+_[0-9]{8}_', re.IGNORECASE)
    for p in prefixes:
        query_string = '{}_{}'.format(p, year)
        output = check_output(['ia', 'search', query_string]).decode()
        for line in output.split('\n'):
            line = line.strip()
            if line:
                identifier = json.loads(line)['identifier']
                if identifier.startswith(query_string):
                    identifiers.append(identifier)
    return identifiers


def download_video_and_subs(identifier, subs_dir, gcs_video_path, gcs_caption_path):
    try:
        check_call(['ia', 'download', '--glob=*.mp4', identifier])
        check_call(['ia', 'download', '--glob=*.srt', identifier])
    except Exception as e:
        print("Error while downloading from internet archive", e)

    if os.path.exists(identifier):
        for fname in os.listdir(identifier):
            if fname.endswith('.mp4'):
                local_path = os.path.join(identifier, fname)
                cloud_path = gcs_video_path + '/' + fname
                check_call(['gsutil', 'cp', '-n', local_path, cloud_path])
            if fname.endswith('.srt'):
                local_path = os.path.join(identifier, fname)
                subs_path = os.path.join(subs_dir, fname)
                shutil.copyfile(local_path, subs_path)
                if gcs_caption_path is not None:
                    gcs_subs_path = os.path.join(gcs_caption_path, fname)
                    check_call(['gsutil', 'cp', '-n', local_path, gcs_subs_path])
        # FIXME: probably want to keep the video files around locally
        shutil.rmtree(identifier)


def main(year, subs_dir, list_file, gcs_video_path, gcs_caption_path):
    if not os.path.exists(subs_dir):
        os.makedirs(subs_dir)

    print('Listing downloaded videos')
    downloaded = list_downloaded_videos(year, gcs_video_path)

    print('Listing available videos')
    available = list_ia_videos(year)

    to_download = [x for x in available if x not in downloaded]
    if list_file:
        with open(list_file, 'w') as f:
            for identifier in to_download:
                f.write(identifier)
                f.write('\n')

    print('Downloading')
    for identifier in tqdm(to_download):
        download_video_and_subs(identifier, subs_dir, gcs_video_path, gcs_caption_path)


if __name__ == '__main__':
    main(**vars(get_args()))
