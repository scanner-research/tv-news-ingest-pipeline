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
        - montage face images
        - black frame detection

    - face identification (identify_faces_with_aws.py)

    - gender classification (classify_gender.py)

"""

import argparse
import glob
from multiprocessing import Pool
import os
import subprocess

from tqdm import tqdm

from docker_compose_api import (container_up, container_down,
                                run_command_in_container,
                                DEFAULT_HOST, DEFAULT_SERVICE)
import classify_gender
from utils import get_base_name
from consts import (OUTFILE_EMBEDS, OUTFILE_GENDERS)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path',
        help='the path to the video file or a text file containing a list of '
             'video filepaths')
    parser.add_argument('output_path',
        help='the path to the directory in which to place the output files.')
    parser.add_argument('-c', '--use-cloud', action='store_true')
    parser.add_argument('-r', '--resilient', action='store_true',
        help='leave docker container up after execution')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, 
        help='docker host IP:port')
    parser.add_argument('--service', type=str, default=DEFAULT_SERVICE,
        choices=['cpu', 'gpu'], help='docker compose service for scanner')
    return parser.parse_args()


def main(video_path, output_path, use_cloud, resilient, host, service):
    create_output_dirs(video_path, output_path)

    # Run scanner component
    container_up(host=host, service=service)

    cmd = build_scanner_component_command(video_path, output_path, use_cloud)
    run_command_in_container(cmd, host=host, service=service)

    if not resilient:
        container_down(host=host)

    video_dirpaths = glob.glob(os.path.join(output_path, '*'))

    # Face identification
    cmd = build_identify_faces_command(output_path, output_path)
    subprocess.run(cmd, shell=True)

    # Gender classification
    for video_dirpath in tqdm(video_dirpaths, desc='Classifying genders'):
        in_file = os.path.join(video_dirpath, OUTFILE_EMBEDS)
        out_file = os.path.join(video_dirpath, OUTFILE_GENDERS) 
        classify_gender.process_single(in_file, out_file)


def create_output_dirs(video_path, output_path):
    if not video_path.endswith('.mp4'):
        with open(video_path, 'r') as f:
            video_paths = [l.strip() for l in f if l.strip()]
            out_paths = [os.path.join(output_path, get_base_name(v))
                         for v in video_paths]
            for out in out_paths:
                if not os.path.exists(out):
                    os.makedirs(out)


def build_scanner_component_command(video_path, output_path, use_cloud):
    cmd = ['python3', 'scanner_component.py', video_path, output_path]
    if use_cloud:
        cmd.append('-c')

    return ' '.join(cmd)


def build_identify_faces_command(input_path, output_path):
    cmd = ['python3', 'identify_faces_with_aws.py', input_path, output_path]
    return ' '.join(cmd)


def build_classify_genders_command(input_path, output_path):
    cmd = ['python3', 'classify_gender.py', input_path, output_path]
    return ' '.join(cmd)


if __name__ == '__main__':
    main(**vars(get_args()))
