#!/usr/bin/env python3

import os
import argparse
import tempfile
import subprocess
from multiprocessing import Pool
from tqdm import tqdm


GCS_VIDEOS_DIR = 'gs://tvnews-ingest/videos'
OLD_GCS_VIDEO_DIR = 'gs://esper/tvnews/videos'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list')
    parser.add_argument('-e', '--execute', action='store_true')
    return parser.parse_args()


def commit(args):
    fname, tmp_path, execute = args
    cloud_path = os.path.join(GCS_VIDEOS_DIR, fname)
    print('Commit:', fname, cloud_path)
    if execute:
        subprocess.check_call([
            '/snap/bin/gsutil', 'cp', '-n',
            tmp_path, cloud_path])
            

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

    files = [fpath[len(prefix):] for fpath in files]

    print('Found {} files'.format(len(files)))
    with tempfile.NamedTemporaryFile() as empty_file, Pool() as workers:
        tmp_path = empty_file.name
        worker_args = [(fname, tmp_path, execute) for fname in files]

        for _ in tqdm(
                workers.imap_unordered(commit, worker_args), total=len(files)
        ):
            pass

    print('Commited {} files'.format(len(files)))


if __name__ == '__main__':
    main(**vars(get_args()))