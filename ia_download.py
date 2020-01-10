#!/usr/bin/env python3

import argparse
import re
import os
import json
import shutil
from subprocess import check_output, check_call
from tqdm import tqdm


GCS_VIDEO_PATH = 'gs://esper/tvnews/videos'
GCS_VIDEO_OUTPUT_PATH = 'gs://esper/tvnews/videos'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='year', type=int, required=True)
    parser.add_argument('--subs', dest='subs_dir', type=str, default='subs',
                        help='Directory to save captions to')
    parser.add_argument('--list', dest='list_file', type=str,
                        help='File to write the list of downloaded videos')
    return parser.parse_args()


def parse_ia_identifier(s):
    """Split off the last"""
    return os.path.splitext(s.split('/')[-1])[0]


def list_downloaded_videos(year):
    """List the videos in the bucket"""
    output = check_output(
        ['gsutil', 'ls', '{}/*{}*'.format(GCS_VIDEO_PATH, year)]).decode()
    videos = [x for x in output.split('\n') if x.strip()]
    return {parse_ia_identifier(x) for x in videos}


def list_ia_videos(year):
    prefixes = ['MSNBC', 'MSNBCW', 'CNN', 'CNNW', 'FOXNEWS', 'FOXNEWSW']
    identifiers = []
    identifier_re = re.compile(r'^[A-Z]+_[0-9]{8}_', re.IGNORECASE)
    for p in prefixes:
        output = check_output(
            ['ia', 'search', '{}_{}'.format(p, year)]).decode()
        for line in output.split('\n'):
            line = line.strip()
            if line:
                identifier = json.loads(line)['identifier']
                if identifier.startswith(p) and identifier_re.match(identifier):
                    identifiers.append(identifier)
    return identifiers


def download_video_and_subs(identifier, subs_dir):
    try:
        check_call(['ia', 'download', '--glob=*.mp4', identifier])
        check_call(['ia', 'download', '--glob=*.srt', identifier])
    except:
        pass
    if os.path.exists(identifier):
        for fname in os.listdir(identifier):
            if fname.endswith('.mp4'):
                local_path = os.path.join(identifier, fname)
                cloud_path = GCS_VIDEO_OUTPUT_PATH + '/' + fname
                check_call(['gsutil', 'cp', '-n', local_path, cloud_path])
            if fname.endswith('.srt'):
                local_path = os.path.join(identifier, fname)
                subs_path = os.path.join(subs_dir, fname)
                shutil.copyfile(local_path, subs_path)
        # FIXME: probably want to keep the video files around locally
        shutil.rmtree(identifier)


def main(year, subs_dir, list_file):
    if not os.path.exists(subs_dir):
        os.makedirs(subs_dir)

    print('Listing downloaded videos')
    downloaded = list_downloaded_videos(year)

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
        download_video_and_subs(identifier, subs_dir)


if __name__ == '__main__':
    main(**vars(get_args()))
