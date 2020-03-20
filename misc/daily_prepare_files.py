#!/usr/bin/env python3

import argparse
import datetime
import errno
import fcntl
import json
from multiprocessing import Pool
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


GCS_OUTPUT_DIR = 'gs://esper/tvnews/ingest-pipeline/outputs'

APP_DATA_PATH = '../esper-tv-widget/data/'
INDEX_PATH = '../esper-tv-widget/index'

LOCAL_OUTPUT_PATH = '/tmp/pipeline_outputs'

PREFIXES = ['MSNBC', 'MSNBCW', 'CNN', 'CNNW', 'FOXNEWS', 'FOXNEWSW']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='year', type=int, default=None,
                        help=('the year for which to download videos. If not '
                              'specified, defaults to year it was yesterday.'))
    parser.add_argument('--local-out-path', default=LOCAL_OUTPUT_PATH,
                        help='directory to save video outputs to')
    parser.add_argument('--gcs-output-path', default=GCS_OUTPUT_DIR,
                        help=('the pipeline output directory'))
    parser.add_argument('--num-processes', dest='num_processes', type=int,
                        default=1, help=('the number of parallel workers to '
                                         'run the downloads on.'))
    return parser.parse_args()


def main(year, local_out_path, gcs_output_path, num_processes):
    # Make sure this script isn't already running
    if not lock_script():
        print('This script is already running. Exiting.')
        return

    downloaded = download_unprepared_outputs(year, local_out_path, gcs_output_path, num_processes)

    if len(downloaded) == 0:
        print('There are no video outputs to prepare at this time. Exiting.')
        return

    cmd = ['/usr/bin/python3', 'prepare_files_for_viewer.py', '-u', LOCAL_OUTPUT_PATH, APP_DATA_PATH, '--index-dir', INDEX_PATH]
    subprocess.check_call(cmd)

    os.chdir('../esper-tv-widget')
    subprocess.check_call(['/usr/bin/python3', 'derive_data.py', '-i'])

    print('Cleaning up local files.')
    shutil.rmtree(LOCAL_OUTPUT_PATH)

    print('Restarting server.')
    subprocess.check_call(['sudo', 'service', 'tv-viewer', 'restart'])
    subprocess.check_call(['sudo', 'rm', '-rf', '/tmp/nginx-cache'])
    subprocess.check_call(['sudo', 'service', 'nginx', 'restart'])


def download_unprepared_outputs(year, local_out_path, gcs_output_path, num_processes):
    os.makedirs(local_out_path, exist_ok=True)
    if year is None:
        year = (datetime.datetime.now() - datetime.timedelta(days=1)).year

    available_outputs = list_pipeline_outputs(year, gcs_output_path)

    processed_outputs = list_processed_outputs()

    to_download = available_outputs - processed_outputs
    if not to_download:
        return []

    print('Downloading {} video outputs on {} threads'.format(len(to_download), num_processes))

    orig_path = os.getcwd()
    os.chdir(local_out_path)
    pool = Pool(num_processes)
    num_done = 0
    start_time = time.time()
    for _ in pool.imap_unordered(download_pipeline_output, [(i, gcs_output_path, local_out_path) for i in to_download]):
        num_done+=1
        print("Finished downloading {} of {} in {} seconds".format(num_done, len(to_download), time.time() - start_time))

    os.chdir(orig_path)
    return to_download


def download_pipeline_output(args):
    identifier, gcs_output_path, local_out_path = args
    subprocess.check_call(['/snap/bin/gsutil', '-m', 'cp', '-nr', os.path.join(gcs_output_path, identifier), './'])


def list_processed_outputs():
    with open(os.path.join(APP_DATA_PATH, 'videos.json'), 'r') as f:
        videos = json.load(f)

    videos = set(x[1] for x in videos)
    return videos


def list_pipeline_outputs(year, gcs_output_path):
    videos = set()

    for prefix in PREFIXES:
        try:
            output = subprocess.check_output(
                ['/snap/bin/gsutil', 'ls', '-d', '{}/{}_{}*'.format(gcs_output_path, prefix, year)]
            ).decode()

            videos |= {parse_identifier(x) for x in output.split('\n') if x.strip()}
        except subprocess.CalledProcessError as e:
            pass

    return videos


def lock_script() -> bool:
    """
    Locks a file pertaining to this script so that it cannot be run simultaneously.

    Since the lock is automatically released when this script ends, there is no
    need for an unlock function for this use case.

    Returns:
        True if the lock was acquired, False otherwise.

    """

    global lockfile
    lockfile = open('/tmp/{}.lock'.format(Path(__file__).name), 'w')

    try:
        # Try to grab an exclusive lock on the file, raise error otherwise
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    except OSError as e:
        if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
            return False
        raise

    else:
        return True


def parse_identifier(s):
    """Split off the last"""
    parts = s.split('/')
    if parts[-1] == '':
        return parts[-2]
    else:
        return parts[-1]


if __name__ == '__main__':
    main(**vars(get_args()))
