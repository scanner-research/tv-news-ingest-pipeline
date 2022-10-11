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
import errno
import fcntl
import json
import tempfile
from multiprocessing import Pool
import os
import sys
from pathlib import Path
import re
import shutil
import subprocess
import time

FILE_BBOXES = 'bboxes.json'
FILE_EMBEDS = 'embeddings.json'
FILE_BLACK_FRAMES = 'black_frames.json'
FILE_COMMERCIALS = 'commercials.json'
FILE_METADATA = 'metadata.json'
FILE_GENDERS = 'genders.json'
FILE_IDENTITIES = 'identities.json'
FILE_IDENTITIES_PROP = 'identities_propogated.json'
FILE_CAPTIONS = 'captions.srt'
FILE_CAPTIONS_ORIG = 'captions_orig.srt'
FILE_ALIGNMENT_STATS = 'alignment_stats.json'
DIR_CROPS = 'crops'

ALL_OUTPUTS = [
    FILE_BBOXES,
    FILE_EMBEDS,
    FILE_METADATA,
    FILE_GENDERS,
    FILE_IDENTITIES,
    FILE_IDENTITIES_PROP,
    FILE_CAPTIONS,
    FILE_CAPTIONS_ORIG,
    FILE_ALIGNMENT_STATS,
    DIR_CROPS
]

# Do not place inside of /tmp so that partial outputs remain if the machine
# goes down
WORKING_DIR = 'daily_process_tmp'
DOWNLOAD_DIR = os.path.join(WORKING_DIR, 'downloads')
BATCH_VIDEOS_PATH = os.path.join(WORKING_DIR, 'batch_videos.txt')
BATCH_CAPTIONS_PATH = os.path.join(WORKING_DIR, 'batch_captions.txt')
PIPELINE_OUTPUT_DIR =  os.path.join(WORKING_DIR, 'pipeline_output')

GCS_VIDEOS_DIR = 'gs://tvnews-ingest/videos'
GCS_OUTPUT_DIR = 'gs://tvnews-ingest/pipeline-outputs'  # pipeline output
GCS_SYNC_DIR = 'gs://tvnews-ingest/pipeline-sync'

PREFIXES = ['MSNBC', 'MSNBCW', 'CNN', 'CNNW', 'FOXNEWS', 'FOXNEWSW']

MAX_VIDEO_DOWNLOADS = 50


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', dest='year', type=int, default=None,
                        help=('The year for which to download videos. If not '
                              'specified, defaults to year it was yesterday.'))
    parser.add_argument('--local-out-path', default=DOWNLOAD_DIR,
                        help='Directory to save videos and captions to.')
    parser.add_argument('--gcs-video-path', default=GCS_VIDEOS_DIR,
                        help=('The path in Google cloud to which videos should '
                              'be uploaded.'))
    parser.add_argument('--num-processes', dest='num_processes', type=int,
                        default=1, help=('The number of parallel workers to '
                                         'run the downloads on.'))
    return parser.parse_args()


def main(year, local_out_path, gcs_video_path, num_processes):

    # Make sure this is not currently running
    if not lock_script():
        print('This script is already running. Exiting.')
        sys.stdout.flush()
        return

    downloaded = download_unprocessed_videos(year, local_out_path,
                                             gcs_video_path, num_processes)
    if len(downloaded) == 0:
        print('No videos to process.')
        sys.stdout.flush()
        return

    create_batch_files(local_out_path, downloaded)

    cmd = ['python3', 'pipeline.py', '-p', BATCH_VIDEOS_PATH, '--captions',
           BATCH_CAPTIONS_PATH, PIPELINE_OUTPUT_DIR]
    subprocess.check_call(cmd)

    if not upload_all_pipeline_outputs_to_cloud(PIPELINE_OUTPUT_DIR, downloaded,
            num_processes, GCS_OUTPUT_DIR):
        print('Upload failed. Exiting.')
        sys.stdout.flush()
        exit()

    commit_processed_videos_to_cloud(local_out_path, downloaded, gcs_video_path)

    # Clean up
    print('Cleaning up files.')
    sys.stdout.flush()
    shutil.rmtree(WORKING_DIR)

    print('Done.')
    sys.stdout.flush()


def upload_all_pipeline_outputs_to_cloud(out_path, downloaded, num_processes,
                                         gcs_output_path):
    print('Uploading {} video outputs on {} threads'.format(len(downloaded),
            num_processes))
    sys.stdout.flush()

    orig_path = os.getcwd()
    os.chdir(out_path)

    # Confirm that all pipeline outputs are present
    for i in downloaded[:]:
        if not all(os.path.exists(os.path.join(i, f)) for f in ALL_OUTPUTS):
            print('Missing outputs for', i)
            sys.stdout.flush()
            downloaded.remove(i)

    sync_with_server()

    pool = Pool(num_processes)
    num_done = 0
    start_time = time.time()
    for _ in pool.imap_unordered(upload_pipeline_output_to_cloud,
        [(i, gcs_output_path) for i in downloaded]
    ):
        num_done += 1
        print('Finished uploading {} of {} in {:0.2f} seconds'.format(num_done,
                len(downloaded), time.time() - start_time))
        sys.stdout.flush()

    os.chdir(orig_path)

    unsync_with_server()

    return True


def upload_pipeline_output_to_cloud(args):
    identifier, gcs_output_path = args

    if os.path.exists(identifier):
        # does not upload crops
        cmd = ['gsutil', '-m', 'cp', '-n', os.path.join(identifier, '*'),
               os.path.join(gcs_output_path, identifier)]
        print('Command:', cmd)
        sys.stdout.flush()
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

        cmd = ['rm', '-rf', identifier]
        print('Command:', cmd)
        sys.stdout.flush()
        subprocess.check_call(cmd)


def commit_processed_videos_to_cloud(local_out_path, downloaded,
                                     gcs_video_path):
    print('Marking {} videos as done'.format(len(downloaded)))
    sys.stdout.flush()
    orig_path = os.getcwd()
    os.chdir(local_out_path)

    with tempfile.NamedTemporaryFile() as empty_file:
        count = 0
        for identifier in downloaded:
            if os.path.exists(identifier):
                for fname in os.listdir(identifier):
                    if fname.endswith('.mp4'):
                        cloud_path = os.path.join(gcs_video_path, fname)
                        cmd = ['/snap/bin/gsutil', 'cp', '-n',
                               empty_file.name, cloud_path]
                        print('Command:', cmd)
                        sys.stdout.flush()
                        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                count += 1
    print('Finished marking {} videos as done'.format(count))
    sys.stdout.flush()

    os.chdir(orig_path)


def download_unprocessed_videos(year, local_out_path, gcs_video_path,
                                num_processes):
    # Make sure output directory exists
    os.makedirs(local_out_path, exist_ok=True)

    if year is None:
        year = (datetime.datetime.now() - datetime.timedelta(days=1)).year
        print("Year not specified. Downloading data for {}.".format(year))
        sys.stdout.flush()

    print('Listing downloaded videos...')
    sys.stdout.flush()
    downloaded = list_downloaded_videos(year, gcs_video_path)

    print('Listing available videos...')
    sys.stdout.flush()
    available = list_ia_videos(year)
    available_by_date = sorted(available, key=lambda x: x.split('_')[1], reverse=True)

    # Exclude videos we have already downloaded and videos which have no available mp4
    to_download = []
    for video in available_by_date:
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
    sys.stdout.flush()

    # Change the current working directory so we download all files into the
    # local_out_path
    orig_path = os.getcwd()
    os.chdir(local_out_path)
    pool = Pool(processes = num_processes)
    num_done = 0
    start_time = time.time()
    for identifier in pool.imap_unordered(download_video_and_subs, to_download[:]):
        if identifier:
            to_download.remove(identifier)
        else:
            num_done+=1
            print("Finished downloading {} of {} in {:0.2f} seconds".format(num_done, len(to_download), time.time() - start_time))
            sys.stdout.flush()

    os.chdir(orig_path)
    return to_download


def sync_with_server():
    while True:
        cmd = ['gsutil', 'mv', GCS_SYNC_DIR + '/.placeholder',
               GCS_SYNC_DIR + '/.uploading']
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return

        cmd = ['gsutil', 'ls', GCS_SYNC_DIR + '/.downloading']
        proc = subprocess.run(cmd)

        if proc.returncode != 0:
            return

        print('Server is downloading. Sleeping for 60 seconds.')
        sys.stdout.flush()
        time.sleep(60)


def unsync_with_server():
    cmd = ['gsutil', 'mv', GCS_SYNC_DIR + '/.uploading',
           GCS_SYNC_DIR + '/.placeholder']
    subprocess.run(cmd, check=True)


def create_batch_files(local_out_path, downloaded):
    # Rename transcripts
    for identifier in downloaded[:]:
        id_path = os.path.join(local_out_path, identifier)
        files = os.listdir(id_path)
        captions = list(filter(lambda x: x.endswith('.srt'), files))
        captions = sorted(list(filter(lambda x: '.cc' in x, captions)))
        if not captions:
            print('No captions for ', identifier)
            sys.stdout.flush()
            downloaded.remove(identifier)
        else:
            os.rename(os.path.join(id_path, captions[0]),
                      os.path.join(id_path, identifier + '.srt'))

    # Batch videos file
    with open(BATCH_VIDEOS_PATH, 'w') as f:
        lines = [os.path.join(local_out_path, i, i + '.mp4') for i in downloaded]
        f.write('\n'.join(lines))

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
    # identifier_re = re.compile(r'^[A-Z]+_[0-9]{8}_', re.IGNORECASE)
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
    print('Download:', identifier)
    sys.stdout.flush()
    try:
        subprocess.check_call([
            'ia', 'download', '-q', '--glob=*.mp4', identifier])
        subprocess.check_call([
            'ia', 'download', '-q', '--glob=*.srt', identifier])
    except Exception as e:
        print("Error while downloading from internet archive", identifier, e)
        sys.stdout.flush()
        return identifier


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


if __name__ == '__main__':
    main(**vars(get_args()))
