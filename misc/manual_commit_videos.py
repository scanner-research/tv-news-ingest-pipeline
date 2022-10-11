#!/usr/bin/env python3

import os
import argparse
import tempfile
import subprocess
from tqdm import tqdm


GCS_VIDEOS_DIR = 'gs://tvnews-ingest/videos'
OLD_GCS_VIDEO_DIR = 'gs://esper/tvnews/videos'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list')
    parser.add_argument('-e', '--execute', action='store_true')
    return parser.parse_args()


def main(file_list, execute):
    print('Reading:', file_list)
    files = []
    with open(file_list) as f:
        for l in f:
            l = l.strip()
            if len(l) == 0:
                continue
            files.append(l)

    prefix = files[0]
    assert prefix == OLD_GCS_VIDEO_DIR + '/', prefix
    files = files[1:]
    for fname in files:
        assert fname.startswith(prefix), fname

    print('Found {} files'.format(len(files)))
    with tempfile.NamedTemporaryFile() as empty_file:
        for fpath in tqdm(files):
            fname = fpath[len(prefix):]
            cloud_path = os.path.join(GCS_VIDEOS_DIR, fname)
            print('Commit:', fname)
            if execute:
                subprocess.check_call([
                    '/snap/bin/gsutil', 'cp', '-n',
                    empty_file.name, cloud_path])

    print('Commited {} files'.format(len(files)))


if __name__ == '__main__':
    main(**vars(get_args()))