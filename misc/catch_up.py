#!/usr/bin/env python3

"""
File: catch_up.py
-----------------
This script is for downloading and processing videos that have been collecting
in gs://esper/tvnews/videos, in order to catch up.


"""

import argparse
import errno
import fcntl
from multiprocessing import Pool
import os
from pathlib import Path
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

WORKING_DIR = '.catch_up_tmp'
DOWNLOAD_DIR = os.path.join(WORKING_DIR, 'downloads')
BATCH_VIDEOS_PATH = os.path.join(WORKING_DIR, 'batch_videos.txt')
BATCH_CAPTIONS_PATH = os.path.join(WORKING_DIR, 'batch_captions.txt')
PIPELINE_OUTPUT_DIR = os.path.join(WORKING_DIR, 'pipeline_output')

NUM_PROCS = os.cpu_count() if os.cpu_count() else 1
GCS_VIDEOS_DIR = 'gs://esper/tvnews/videos'
GCS_CAPTIONS_DIR = 'gs://esper/tvnews/subs'
GCS_OUTPUT_DIR = 'gs://esper/tvnews/ingest-pipeline/outputs'

PREFIXES = ['MSNBC', 'MSNBCW', 'CNN', 'CNNW', 'FOXNEWS', 'FOXNEWSW']

MAX_VIDEO_DOWNLOADS = 100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date-prefix', help='the date prefix to search.')
    parser.add_argument('--local-out-path', default=DOWNLOAD_DIR,
                        help='Directory to save videos and captions to.')
    parser.add_argument('--gcs-video-path', default=GCS_VIDEOS_DIR,
                        help='The path in Google cloud where the videos are stored.')
    parser.add_argument('--gcs-caption-path', default=GCS_CAPTIONS_DIR,
                        help=('The path in Google cloud where the '
                              'captions/subtitles are stored.'))
    parser.add_argument('--gcs-output-path', default=GCS_OUTPUT_DIR,
                        help=('The path in Google cloud where the pipeline '
                              'outputs should be stored.'))
    parser.add_argument('--num-processes', dest='num_processes', type=int,
                        default=NUM_PROCS, help=('The number of parallel workers '
                                                 'to run the downloads on.'))
    return parser.parse_args()


def main(date_prefix, local_out_path, gcs_video_path, gcs_caption_path,
         gcs_output_path, num_processes):

    # Make sure this is not currently running
    if not lock_script():
        print('This script is already running. Exiting.')
        return

    local_out_path = Path(local_out_path)

    if not date_prefix:
        date_prefix = ''

    downloaded = download_unprocessed_videos(date_prefix, local_out_path,
                                             gcs_video_path, gcs_caption_path,
                                             gcs_output_path)

    if not downloaded:
        print('No videos to download for date-prefix=' + date_prefix)
        return

    create_batch_files(local_out_path, downloaded)

    cmd = ['python3', 'pipeline.py', '-p', BATCH_VIDEOS_PATH, '--captions',
           BATCH_CAPTIONS_PATH, PIPELINE_OUTPUT_DIR]
    subprocess.run(cmd, check=True)

    if not upload_all_pipeline_outputs_to_cloud(PIPELINE_OUTPUT_DIR, downloaded,
            num_processes, GCS_OUTPUT_DIR):
        print('Upload failed. Exiting.')
        exit()

    # Clean up
    shutil.rmtree(WORKING_DIR)


def download_unprocessed_videos(date_prefix, local_out_path, gcs_video_path,
                                gcs_caption_path, gcs_output_path):
    local_out_path.mkdir(parents=True, exist_ok=True)

    available = list_available_videos(date_prefix, gcs_video_path,
                                      gcs_caption_path)

    processed = list_processed_videos(date_prefix, gcs_output_path)

    to_download = list(available - processed)
    if not to_download:
        return []

    if len(to_download) > MAX_VIDEO_DOWNLOADS:
        to_download = to_download[:MAX_VIDEO_DOWNLOADS]

    print('Downloading {} videos'.format(len(to_download)))

    orig_path = os.getcwd()
    os.chdir(str(local_out_path))

    cmd = ['gsutil', '-m', 'cp', '-n']
    for identifier in to_download:
        cmd.append(os.path.join(gcs_video_path, identifier) + '.mp4')
        cmd.append(os.path.join(gcs_caption_path, identifier + '*.srt'))
    cmd.append('./')

    subprocess.run(cmd, check=True)

    os.chdir(orig_path)
    return to_download


def list_available_videos(date_prefix, gcs_video_path, gcs_caption_path):
    """
    Lists the videos with captions downloaded to GCS.

    Args:
        date_prefix (str): the date prefix to search.
        gcs_video_path (Path): the GCS path to the videos.
        gcs_caption_path (Path): the GCS path to the captions.

    Returns:
        List[str]: the identifiers of available videos.

    """

    videos = set()
    captions = set()

    for prefix in PREFIXES:
        try:
            cmd = ['gsutil', 'ls',
                   '{}/{}_{}*'.format(gcs_video_path, prefix, date_prefix)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
            output = proc.stdout.decode()
            videos |= {Path(x).stem for x in output.split('\n') if x.strip()}

            cmd = ['gsutil', 'ls',
                   '{}/{}_{}*'.format(gcs_caption_path, prefix, date_prefix)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
            output = proc.stdout.decode()
            captions |= {Path(Path(x).stem).stem for x in output.split('\n') if x.strip()}

        except subprocess.CalledProcessError as e:
            pass  # no matches

    return videos & captions


def list_processed_videos(date_prefix, gcs_output_path):
    videos = set()

    for prefix in PREFIXES:
        try:
            cmd = ['gsutil', 'ls', '-d',
                   '{}/{}_{}*'.format(gcs_output_path, prefix, date_prefix)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
            output = proc.stdout.decode()
            videos |= {Path(x).name for x in output.split('\n') if x.strip()}
        except subprocess.CalledProcessError as e:
            pass  # no matches

    return videos


def create_batch_files(local_out_path, downloaded):
    # Batch videos file
    with open(BATCH_VIDEOS_PATH, 'w') as f:
        lines = [os.path.join(local_out_path, i + '.mp4') for i in downloaded]
        f.write('\n'.join(lines))

    # Rename transcripts
    for identifier in downloaded:
        files = os.listdir(local_out_path)
        if (identifier + '.srt') in files:
            continue

        files = list(filter(lambda x: x.startswith(identifier), files))
        captions = list(filter(lambda x: x.endswith('.srt'), files))
        captions = sorted(list(filter(lambda x: '.cc' in x, captions)))
        os.rename(os.path.join(local_out_path, captions[0]),
                  os.path.join(local_out_path, identifier + '.srt'))

    # Batch captions file
    with open(BATCH_CAPTIONS_PATH, 'w') as f:
        lines = [os.path.join(local_out_path, i + '.srt') for i in downloaded]
        f.write('\n'.join(lines))


def upload_all_pipeline_outputs_to_cloud(out_path, downloaded, num_processes,
                                         gcs_output_path):
    print('Uploading {} video outputs on {} threads'.format(len(downloaded),
            num_processes))

    orig_path = os.getcwd()
    os.chdir(out_path)

    # Confirm that all pipeline outputs are present
    missing = False
    for i in downloaded:
        if not all(os.path.exists(os.path.join(i, f)) for f in ALL_OUTPUTS):
            print('Missing outputs for', i)
            missing = True

    if missing:
        print('Some outputs are missing. Stopping upload')
        os.chdir(orig_path)
        return False

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
    return True

def upload_pipeline_output_to_cloud(args):
    identifier, gcs_output_path = args

    if os.path.exists(identifier):
        # does not upload crops
        cmd = ['gsutil', 'cp', '-n', os.path.join(identifier, '*'),
               os.path.join(gcs_output_path, identifier)]
        subprocess.check_call(cmd)

        cmd = ['sudo', 'rm', '-rf', identifier]
        subprocess.check_call(cmd)


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
