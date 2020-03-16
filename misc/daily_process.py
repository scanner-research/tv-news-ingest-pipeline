#!/usr/bin/env python3

"""
File: daily_process.py
----------------------
This script is for downloading and processing new videos from the Internet
Archive, meant to be run on a daily basis.

First, we determine which videos are available that have not already been
processed.

Then, we download those videos and transcripts, and run them as a batch
through the pipeline.

The outputs of the pipeline (not including face crops) are then uploaded to a
cloud bucket to be downloaded at the server, which will perform the
incremental update.

Finally, all local files are erased after being uploaded.

"""

import argparse
import datetime
import json
from multiprocessing import Pool
import os
import re
import shutil
import subprocess
import time
from tqdm import tqdm

from util.utils import lock_script


WORKING_DIR = '/tmp/daily_process'
DOWNLOAD_DIR = os.path.join(WORKING_DIR, 'downloads')
BATCH_VIDEOS_PATH = os.path.join(WORKING_DIR, 'batch_videos.txt')
BATCH_CAPTIONS_PATH = os.path.join(WORKING_DIR, 'batch_captions.txt')
PIPELINE_OUTPUT_DIR =  os.path.join(WORKING_DIR, 'pipeline_output')

GCS_VIDEOS_DIR = 'gs://esper/tvnews/videos'
GCS_CAPTIONS_DIR = 'gs://esper/tvnews/subs'
GCS_OUTPUT_DIR = 'gs://esper/tvnews/ingest-pipeline/tmp'  # pipeline output

PREFIXES = ['MSNBC', 'MSNBCW', 'CNN', 'CNNW', 'FOXNEWS', 'FOXNEWSW']

MAX_VIDEO_DOWNLOADS = 72


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='year', type=int, default=None,
                        help=('The year for which to download videos. If not '
                              'specified, defaults to year it was yesterday.'))
    parser.add_argument('--local-out-path', default=DOWNLOAD_DIR,
                        help='Directory to save videos and captions to. ')
    parser.add_argument('--gcs-video-path', default=GCS_VIDEOS_DIR,
                        help=('The path in Google cloud to which videos should '
                              'be uploaded.'))
    parser.add_argument('--gcs-caption-path', default=GCS_CAPTIONS_DIR,
                        help=('The path in Google cloud to which '
                              'captions/subtitles should be uploaded.'))
    parser.add_argument('--num-processes', dest='num_processes', type=int,
                        default=1, help=('The number of parallel workers to '
                                         'run the downloads on.'))
    return parser.parse_args()


def main(year, local_out_path, gcs_video_path, gcs_caption_path, num_processes):

    # Make sure this is not currently running
    if not lock_script():
        print('This script is already running. Exiting.')
        return

    downloaded = download_unprocessed_videos(year, local_out_path,
                                             gcs_video_path, num_processes)

    create_batch_files(local_out_path, downloaded)

    cmd = ['python3', 'pipeline.py', BATCH_VIDEOS_PATH, '--captions',
           BATCH_CAPTIONS_PATH, PIPELINE_OUTPUT_DIR]
    subprocess.check_call(cmd)

    upload_all_pipeline_outputs_to_cloud(PIPELINE_OUTPUT_DIR, downloaded,
                                         num_processes, GCS_OUTPUT_DIR)
    upload_processed_videos_to_cloud(local_out_path, downloaded, num_processes,
                                     gcs_video_path, gcs_caption_path)

    # Clean up
    shutil.rmtree(WORKING_DIR)
    cmd = ['sudo', 'docker', '--host', '127.0.0.1:2375', 'container', 'prune',
           '-a']
    subprocess.check_call(cmd)


def upload_all_pipeline_outputs_to_cloud(out_path, downloaded, num_processes, gcs_output_path):
    print('Uploading {} video outputs on {} threads'.format(len(downloaded),
            num_processes))

    orig_path = os.getcwd()
    os.chdir(out_path)
    pool = Pool(num_processes)
    num_done = 0
    start_time = time.time()
    for _ in pool.imap_unordered(upload_pipeline_output_to_cloud,
        [(i, gcs_output_path) for i in downloaded]
    ):
        num_done += 1
        print('Finished uploading {} of {} in {} seconds'.format(num_done,
                len(downloaded), time.time() - start_time))

    os.chdir(orig_path)


def upload_pipeline_output_to_cloud(args):
    identifier, gcs_output_path = args

    if os.path.exists(identifier):
        # does not upload crops
        cmd = ['gsutil', 'cp', '-n', os.path.join(identifier, '*'), gcs_output_path]
        print('COMMAND', cmd)
        subprocess.check_call(cmd)

        cmd = ['sudo', 'rm', '-rf', identifier]
        print('COMMAND', cmd)
        subprocess.check_call(cmd)


def upload_processed_videos_to_cloud(local_out_path, downloaded, num_processes, gcs_video_path, gcs_caption_path):
    print('Uploading {} videos on {} threads'.format(len(downloaded), num_processes))
    # Change the current working directory so we download all files into the
    # local_out_path
    orig_path = os.getcwd()
    os.chdir(local_out_path)
    pool = Pool(processes = num_processes)
    num_done = 0
    start_time = time.time()
    for _ in pool.imap_unordered(upload_video_and_subs_to_cloud,
        [(i, gcs_video_path, gcs_caption_path) for i in downloaded]
    ):
        num_done+=1
        print("Finished uploading {} of {} in {} seconds".format(num_done, len(downloaded), time.time() - start_time))

    os.chdir(orig_path)


def download_unprocessed_videos(year, local_out_path, gcs_video_path,
                                num_processes):
    # Make sure output directory exists
    os.makedirs(local_out_path, exist_ok=True)

    if year is None:
        year = (datetime.datetime.now() - datetime.timedelta(days=1)).year
        print("Year not specified. Downloading data for {}.".format(year))

    print('Listing downloaded videos...')
    downloaded = list_downloaded_videos(year, gcs_video_path)

    print('Listing available videos...')
    available = list_ia_videos(year)

    # Exclude videos we have already downloaded and videos which have no available mp4
    to_download = []
    for video in available:
        if video in downloaded:
            continue
        # This prints results directly to the terminal
        status = subprocess.call(['ia', 'list', video, '--glob=*.mp4'])
        # If there is no video, the list will return a nonzero status
        if status == 0:
            to_download.append(video)
            if len(to_download) >= MAX_VIDEO_DOWNLOADS:
                break

    print('Downloading {} videos on {} threads'.format(len(to_download), num_processes))

    # Change the current working directory so we download all files into the
    # local_out_path
    orig_path = os.getcwd()
    os.chdir(local_out_path)
    pool = Pool(processes = num_processes)
    num_done = 0
    start_time = time.time()
    for _ in pool.imap_unordered(download_video_and_subs, to_download):
        num_done+=1
        print("Finished downloading {} of {} in {} seconds".format(num_done, len(to_download), time.time() - start_time))

    os.chdir(orig_path)
    return to_download


def create_batch_files(local_out_path, downloaded):
    # Batch videos file
    with open(BATCH_VIDEOS_PATH, 'w') as f:
        lines = [os.path.join(local_out_path, i, i + '.mp4') for i in downloaded]
        f.write('\n'.join(lines))

    # Rename transcripts
    for identifier in downloaded:
        id_path = os.path.join(local_out_path, identifier)
        files = os.listdir(id_path)
        captions = list(filter(lambda x: x.endswith('.srt'), files))
        captions = sorted(list(filter(lambda x: '.cc' in x, captions)))
        os.rename(os.path.join(id_path, captions[0]),
                  os.path.join(id_path, identifier + '.srt'))

    # Batch captions file
    with open(BATCH_CAPTIONS_PATH, 'w') as f:
        lines = [os.path.join(local_out_path, i, i + '.srt') for i in downloaded]
        f.write('\n'.join(lines))


def parse_ia_identifier(s):
    """Split off the last"""
    return os.path.splitext(s.split('/')[-1])[0]


def list_downloaded_videos(year, gcs_video_path):
    """List the videos in the bucket"""
    videos = set()
    # GCS is prefix indexed, so iterating over the prefixes rather than having
    # a leading star is way faster
    for prefix in PREFIXES:
        try:
            output = subprocess.check_output(
                ['gsutil', 'ls', '{}/{}_{}*'.format(gcs_video_path, prefix, year)]).decode()
            videos |= {parse_ia_identifier(x) for x in output.split('\n') if x.strip()}
        except subprocess.CalledProcessError as e:
            # It's probably just no matches, which is fine
            pass
    return videos


def list_ia_videos(year):
    identifiers = []
    identifier_re = re.compile(r'^[A-Z]+_[0-9]{8}_', re.IGNORECASE)
    for p in PREFIXES:
        query_string = '{}_{}'.format(p, year)
        output = subprocess.check_output(['ia', 'search', query_string]).decode()
        for line in output.split('\n'):
            line = line.strip()
            if line:
                identifier = json.loads(line)['identifier']
                if identifier.startswith(query_string):
                    identifiers.append(identifier)

    return identifiers

def download_video_and_subs(identifier):
    try:
        subprocess.check_call(['ia', 'download', '--glob=*.mp4', identifier])
        subprocess.check_call(['ia', 'download', '--glob=*.srt', identifier])
    except Exception as e:
        print("Error while downloading from internet archive", e)


def upload_video_and_subs_to_cloud(args):
    identifier, gcs_video_path, gcs_caption_path = args

    if os.path.exists(identifier):
        for fname in os.listdir(identifier):
            if fname.endswith('.mp4'):
                local_path = os.path.join(identifier, fname)
                cloud_path = os.path.join(gcs_video_path, fname)
                subprocess.check_call(['gsutil', 'cp', '-n', local_path, cloud_path])
            if fname.endswith('.srt') and gcs_caption_path is not None:
                local_path = os.path.join(identifier, fname)
                cloud_path = os.path.join(gcs_caption_path, fname)
                subprocess.check_call(['gsutil', 'cp', '-n', local_path, cloud_path])

        # FIXME: probably want to keep the video files around locally
        # shutil.rmtree(identifier)


if __name__ == '__main__':
    main(**vars(get_args()))
