#!/usr/bin/env python3

"""
File: pipeline.py
-----------------
This script is the interface to the TV-News video processing pipeline.

Given a video filepath or textfile containing a list of video filepaths, this 
script takes the video(s) through the following stages:

    - scanner component (scanner_component.py)
        - face detection
        - face embeddings
        - face image crops

    - black frame detection (black_frame_detection.py)

    - face identification (identify_faces_with_aws.py)

    - gender classification (classify_gender.py)

    - copies captions


Sample output directory after pipeline completion:

    output_dir/
    ├── video_name1
    │   ├── bboxes.json
    │   ├── black_frames.json
    │   ├── embeddings.json
    │   ├── genders.json
    │   ├── identities.json
    │   ├── metadata.json
    │   ├── captions.srt
    │   └── crops
    │       ├── 0.png
    │       └── 1.png
    ├── video_name2
    │   └── ...
    └── ... 

"""

import argparse
import glob
from multiprocessing import Pool
import os
import shutil
import subprocess

from tqdm import tqdm

from components import classify_gender, identify_faces_with_aws
from util.consts import (OUTFILE_EMBEDS, OUTFILE_GENDERS, OUTDIR_MONTAGES,
                         OUTFILE_IDENTITIES, OUTFILE_CAPTIONS)
from util.docker_compose_api import (container_up, container_down,
                                     pull_container, run_command_in_container,
                                     DEFAULT_HOST, DEFAULT_SERVICE)
from util.utils import get_base_name, update_pbar

NAMED_COMPONENTS = [
    'face_detection',   # <
    'face_embeddings',  # < all within scanner_component.py
    'face_crops',       # <
    'scanner_component',
    'black_frames',
    'identities',
    'genders',
    'captions'
]


class PipelineException(Exception):
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('--captions', help=('path to srt or to a text file '
                                            'containing srt filepaths'))
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


def main(in_path, captions, out_path, resilient=False, host=DEFAULT_HOST,
         service=DEFAULT_SERVICE, init_run=False, force=False,
         disable=None):

    if disable is None:
        disable = []

    single = in_path.endswith('.mp4')

    print('Creating output directories at "{out_path}"...'.format(out_path=out_path)
    output_dirs = create_output_dirs(in_path, out_path)

    if 'scanner_component' not in disable:
        run_scanner_component(in_path, out_path, disable, init_run, force,
                              host=host, service=service)
  
    if 'black_frames' not in disable:
        run_black_frame_detection(in_path, out_path, init_run, force,
                docker_up=('scanner_component' in disable),
                docker_down=(not resilient), host=host, service=service)

    if 'identities' not in disable:
        identify_faces_with_aws.main(out_path, out_path, force=force,
                                     single=single)

    if 'genders' not in disable:
        classify_gender.main(out_path, out_path, force=force)

    if 'captions' not in disable and captions is not None:
        copy_captions(captions, out_path)
    

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

    cmd = build_scanner_component_command(in_path, out_path, disable, init_run,
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
            ('Could not connect to docker daemon at http://{host}.'
            'Try running to following command: '
            '`sudo dockerd -H tcp://{host} --log-level error &`').format(host=host)
        )


def build_scanner_component_command(in_path, out_path, disable=None,
                                    init_run=False, force_rerun=False):
    cmd = ['python3', 'components/scanner_component.py', in_path, out_path]
    if disable:
        cmd.append('-d')
        for d in disable:
            cmd.append(d)
    if init_run:
        cmd.append('-i')
    if force_rerun:
        cmd.append('-f')

    return ' '.join(cmd)


def build_black_frame_detection_command(in_path, out_path, init_run=False,
                                        force_rerun=False):
    cmd = ['python3', 'components/black_frame_detection.py', in_path, out_path]
    if init_run:
        cmd.append('-i')
    if force_rerun:
        cmd.append('-f')

    return ' '.join(cmd)


def copy_captions(in_path, out_dir):
    if in_path.endswith('.srt'):
        pbar = tqdm(total=1, desc='Copying captions', unit='video')
        out_path = os.path.join(out_dir, OUTFILE_CAPTIONS)
        if not os.path.exists(out_path):
            shutil.copy(in_path, out_path)
        pbar.update()
    else:
        with open(in_path, 'r') as f:
            paths = [l.strip() for l in f if l.strip()]
        video_names = [get_base_name(p) for p in paths]
        out_paths = [os.path.join(out_dir, v, OUTFILE_CAPTIONS)
                     for v in video_names]
        for captions, out_path in zip(
            tqdm(paths, desc='Copying captions', unit='video'), out_paths
        ):
            if not os.path.exists(out_path):
                shutil.copy(captions, out_path)


if __name__ == '__main__':
    main(**vars(get_args()))
