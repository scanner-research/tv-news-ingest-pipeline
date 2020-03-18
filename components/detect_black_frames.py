#!/usr/bin/env python3

"""
File: black_frame_detection.py
------------------------------
Script for detecting black frames in videos using Scanner.

Notes: Must be run inside Scanner's docker container.

When running on a batch of videos for the first time, use the '--init-run'
flag to skip a potentially costly check to see if the outputs already exist.

If docker is running as root (which it most commonly will be), then all
directories and files created will be owned by root.


Example
-------

    in_path:  batch.txt
    out_path: output_dir

    where 'batch.txt' looks like:

        path/to/video1.mp4
        different/path/to/video2.mp4

    outputs

        output_dir/
        ├── video1
        │   └── black_frames.json
        └── video2
            └── black_frames.json

"""

import argparse
from functools import partial
import itertools
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import subprocess
from collections import deque
import os
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

from util.consts import FILE_BLACK_FRAMES
from util.utils import json_is_valid

NUM_THREADS = os.cpu_count()
BINARY_PATH = 'deps/detect-black-frames/detect_black_frames'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('-i', '--init-run', action='store_true',
                        help='running on videos for the first time')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force rerun for all videos')
    return parser.parse_args()


def main(in_path, out_path, init_run=False, force=False):
    if in_path.endswith('.mp4'):
        video_paths = [Path(in_path)]
    else:  # batch text file
        video_paths = [Path(l.strip()) for l in open(in_path, 'r') if l.strip()]

    out_paths = [Path(out_path, p.stem) for p in video_paths]

    assert len(video_paths) == len(out_paths), 'Mismatch between video and ' \
                                               'output paths'

    video_names = get_videos_to_process(video_paths, out_paths, init_run or force)

    # Make sure there are videos to process
    if len(video_names) == 0:
        print('All videos have existing black frame outputs.')
        return

    for video_path, out_path in zip(tqdm(
        video_paths, desc='Detecting black frames', unit='video'
    ), out_paths):
        cmd = [BINARY_PATH, '-n', '1', '-j', str(NUM_THREADS)]
        path_str = '{} {}'.format(video_path, out_path/FILE_BLACK_FRAMES)
        subprocess.run(cmd, input=path_str.encode('utf-8'), check=True)


def get_videos_to_process(video_paths, out_paths, skip=False):
    """
    Gets names of the videos to process, ignoring videos with existing outputs.

    Modifies video_paths and out_paths by removing already processed videos.

    Args:
        video_paths (List[Path]): a list of video file paths.
        out_paths (List[Path]): a list of output directory paths.

    Returns:
       List[str]: a list of video names (without extension) to be processed.

    """

    video_names = [p.stem for p in video_paths]
    if not skip:
        for i in range(len(video_names) - 1, -1, -1):
            print(out_paths[i]/FILE_BLACK_FRAMES)
            if json_is_valid(out_paths[i]/FILE_BLACK_FRAMES):
                video_names.pop(i)
                out_paths.pop(i)

    return video_names


if __name__ == '__main__':
    main(**vars(get_args()))
