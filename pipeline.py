#!/usr/bin/env python3

"""
File: pipeline.py
-----------------
This script is the interface to the TVNews video processing pipeline.

Given a video filepath or textfile containing a list of video filepaths, this 
script takes the video(s) through the following stages:

    - scanner component (scanner_component.py)
        - face detection
        - face embeddings
        - face image crops

    - black frame detection (black_frame_detection.py)

    - face identification (identify_faces_with_aws.py)

    - gender classification (classify_gender.py)


Sample output directory after pipeline completion:

    output_dir/
    ├── video_name1
    │   ├── bboxes.json
    │   ├── black_frames.json
    │   ├── embeddings.json
    │   ├── genders.json
    │   ├── identities.json
    │   ├── metadata.json
    │   └── montages
    │       ├── 0.json
    │       └── 0.png
    ├── video_name2
    │   └── ...
    └── ... 

"""
import time
import argparse
import glob
from multiprocessing import Pool
import os
import subprocess

from tqdm import tqdm

from docker_compose_api import (container_up, container_down, pull_container,
                                run_command_in_container, DEFAULT_HOST,
                                DEFAULT_SERVICE)
import classify_gender
import identify_faces_with_aws
from utils import get_base_name, update_pbar
from consts import (OUTFILE_EMBEDS, OUTFILE_GENDERS, OUTDIR_MONTAGES,
                    OUTFILE_IDENTITIES, SCANNER_COMPONENT_OUTPUTS)

NAMED_COMPONENTS = [
    'face_detection',   # <
    'face_embeddings',  # < all within scanner_component.py
    'face_crops',       # <
    'scanner_component',
    'black_frames',
    'identities',
    'genders'
]

class PipelineException(Exception):
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('-r', '--resilient', action='store_true',
                        help='leave docker container up after execution')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, 
                        help='docker host IP:port')
    parser.add_argument('--service', type=str, default=DEFAULT_SERVICE,
                        help='docker compose service for scanner',
                        choices=['cpu', 'gpu'])
    parser.add_argument('-i', '--init-run', action='store_true',
                        help='running on videos for the first time')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force rerun for all videos')
    parser.add_argument('-d', '--disable', nargs='+', choices=NAMED_COMPONENTS,
                        help='list of named components to disable')
    return parser.parse_args()


def main(in_path, out_path, resilient=False, host=DEFAULT_HOST,
         service=DEFAULT_SERVICE, init_run=False, force=False,
         disable=None):

    if disable is None:
        disable = []

    print(f'Creating output directories at "{out_path}"...')
    output_dirs = create_output_dirs(in_path, out_path)

    if 'scanner_component' not in disable:
        start = time.time()
        run_scanner_component(in_path, out_path, disable, init_run, force,
                              host=host, service=service)
        end = time.time()
        print(f'took {end - start} seconds to run scanner component')
  
    if 'black_frames' not in disable:
        start = time.time()
        run_black_frame_detection(in_path, out_path, init_run, force,
                docker_up=('scanner_component' in disable),
                docker_down=(not resilient), host=host, service=service)
        end = time.time()
        print(f'took {end - start} seconds to run black frame')

    if 'identities' not in disable:
        start = time.time()
        identify_faces_with_aws.main(out_path, out_path, force=force)
        end = time.time()
        print(f'took {end - start} seconds to identify')

    if 'genders' not in disable:
        start = time.time()
        classify_gender.main(out_path, out_path, force=force)
        end = time.time()
        print(f'took {end - start} seconds to classify')


def create_output_dirs(video_path: str, output_path: str) -> list:
    """
    Creates output subdirectories for each video being processed.
    Necessary due to docker container processes being owned by root.

    Args:
        video_path: path to the video file or textfile containing filepaths.
        output_path: path to the output directory.

    Returns:
        a list of output directory paths.

    """

    if not video_path.endswith('.mp4'):
        with open(video_path, 'r') as f:
            video_paths = [l.strip() for l in f if l.strip()]

        out_paths = [os.path.join(output_path, get_base_name(v))
                     for v in video_paths]
        for out in out_paths:
            if not os.path.exists(out):
                os.makedirs(out)

        return out_paths

    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return [output_path]
        

def run_scanner_component(in_path, out_path, disable=None, init_run=False,
                          force_rerun=False, docker_up=True, docker_down=False,
                          host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    if docker_up:
        prepare_docker_container(host, service)

    cmd = build_scanner_component_command(in_path, out_path, init_run,
                                          force_rerun)
    run_command_in_container(cmd, host, service)

    if docker_down:
        print('Shutting down docker container...')
        container_down(host=host)


def run_black_frame_detection(in_path, out_path, init_run=False,
        force_rerun=False, docker_up=False, docker_down=False,
        host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    if docker_up:
        prepare_docker_container(host, service)

    cmd = build_black_frame_detection_command(in_path, out_path, init_run,
                                              force_rerun)
    run_command_in_container(cmd, host, service)

    if docker_down:
        print('Shutting down docker container...')
        container_down(host=host)


def prepare_docker_container(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    try:
        print('Getting docker container ready...')
        pull_container(host, service)
        container_up(host, service)
    except subprocess.CalledProcessError as err:
        raise PipelineException(
            f'Could not connect to docker daemon at http://{host}.'
             'Try running to following command: '
            f'`sudo dockerd -H tcp://{host} --log-level error &`'
        )


def build_scanner_component_command(in_path, out_path, init_run=False,
                                    force_rerun=False):
    cmd = ['python3', 'scanner_component.py', in_path, out_path]
    if init_run:
        cmd.append('-i')
    if force_rerun:
        cmd.append('-f')

    return ' '.join(cmd)


def build_black_frame_detection_command(in_path, out_path, init_run=False,
                                        force_rerun=False):
    cmd = ['python3', 'black_frame_detection.py', in_path, out_path]
    if init_run:
        cmd.append('-i')
    if force_rerun:
        cmd.append('-f')

    return ' '.join(cmd)


def verify_scanner_component_output(output_paths: list) -> bool:
    """
    Verifies whether all expected outputs of the scanner component are present.

    Args:
        output_paths: the list of output directory paths.

    Returns:
        True if all expected outputs are present, False otherwise.

    """
    
    return all(os.path.exists(os.path.join(path, expected))
               for path in output_paths
               for expected in SCANNER_COMPONENT_OUTPUTS)


if __name__ == '__main__':
    main(**vars(get_args()))
