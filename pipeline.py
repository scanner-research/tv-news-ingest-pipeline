#!/usr/bin/env python3

"""
File: pipeline.py
-----------------
This script is the interface to the TV-News video processing pipeline.

Given a video filepath or textfile containing a list of video filepaths, this
script takes the video(s) through the following stages:

    - scanner component (scanner_component.py)
        - detect faces
        - compute face embeddings
        - extract face image crops

    - detect black frames (black_frame_detection.py)

    - identify faces (identify_faces_with_aws.py)

    - propagate identities to unlabeled faces (identity_propagation.py)

    - classify gender (classify_gender.py)

    - copy original captions

    - time align captions (caption_alignment.py)

    - detect commercials (commercial_detection.py)


Sample output directory after pipeline completion:

    output_dir/
    ├── video1
    │   ├── alignment_stats.json
    │   ├── bboxes.json
    │   ├── black_frames.json
    │   ├── embeddings.json
    │   ├── genders.json
    │   ├── identities.json
    │   ├── identities_propagated.json
    │   ├── metadata.json
    │   ├── captions.srt
    │   ├── captions_orig.srt
    │   ├── commercials.json
    │   └── crops
    │       ├── 0.png
    │       └── 1.png
    ├── video2
    │   └── ...
    └── ...

"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from tqdm import tqdm

from util import config
from util.consts import FILE_CAPTIONS_ORIG
from util.docker_compose_api import run_command_in_container
from util.utils import get_base_name

NAMED_COMPONENTS = [
    'face_detection',   # <
    'face_embeddings',  # < all within scanner_component.py
    'face_crops',       # <
    'scanner_component',
    'black_frames',
    'identities',
    'identity_propagation',
    'genders',
    'captions_copy',
    'caption_alignment',
    'commercials'
]


class PipelineException(Exception):
    pass


def get_args():
    """Get command line arguments."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('--captions', help=('path to srt or to a text file '
                                            'containing srt filepaths'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('--host', help='docker host IP:port')
    parser.add_argument('-i', '--init-run', action='store_true',
                        help='running on videos for the first time')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force rerun for all videos')
    parser.add_argument('-d', '--disable', nargs='+', choices=NAMED_COMPONENTS,
                        help='list of named components to disable')
    parser.add_argument('-s', '--script', choices=NAMED_COMPONENTS[3:],
                        help='run a single component of the pipeline as a script')
    return parser.parse_args()


def main(in_path, captions, out_path, host=None, init_run=False, force=False,
         disable=None, script=None):
    """
    The entrypoint for the pipeline.

    Args:
        in_path (str): the path to the video file or batch file.
        captions (str): the path to the captions file or batch captions file.
        out_path (str): the path to the output directory.
        host (Optional[str]): the host to connect to Docker on. Default None.
        init_run (bool): whether this is the first time processing the videos.
                         Default False.
        force (bool): whether to overwrite existing outputs. Default False.
        disable (Optional[List]): a list of components to disable.
                                  Default None.
        script (Optional[str]): a single component to run. Default None.
    
    """

    start = time.time()

    # Configuration settings
    if disable is None:
        disable = config.DISABLE if config.DISABLE else []

    if host is None:
        host = config.HOST

    # Validate file formats
    single = not in_path.endswith('.txt') and not os.path.isdir(in_path)
    if single and not in_path.endswith('.mp4'):
        print('Only the mp4 video format is supported. Exiting.')
        return

    if single and captions is not None and not captions.endswith('.srt'):
        print('Only the srt captions format is supported. Exiting.')
        return

    print('Creating output directories at "{}"...'.format(out_path))
    video_paths, output_dirs = create_output_dirs(in_path, out_path, single)
    video_dirpaths = [str(Path(p).parent) for p in video_paths]

    # Step through each pipeline component
    if (script and script == 'scanner_component') \
            or (not script and 'scanner_component' not in disable):
        run_scanner_component(in_path, out_path, video_dirpaths, disable,
                              init_run, force, host=host)

    if (script and script == 'black_frames') \
            or (not script and 'black_frames' not in disable):
        from components import detect_black_frames
        detect_black_frames.main(in_path, out_path, init_run=init_run,
                                 force=force)

    if (script and script == 'identities') \
            or (not script and 'identities' not in disable):
        from components import identify_faces_with_aws
        identify_faces_with_aws.main(out_path, out_path, force=force)

    if (script and script == 'identity_propagation') \
            or (not script and 'identity_propagation' not in disable):
        from components import identity_propagation
        identity_propagation.main(out_path, out_path, force=force)

    if (script and script == 'genders') \
            or (not script and 'genders' not in disable):
        from components import classify_gender
        classify_gender.main(out_path, out_path, force=force)

    if captions is not None:
        if (script and script == 'captions_copy') \
                or (not script and 'captions_copy' not in disable):
            from components import copy_captions
            copy_captions.main(captions, out_path)

        if (script and script == 'caption_alignment') \
                or (not script and 'caption_alignment' not in disable):
            from components import caption_alignment
            caption_alignment.main(in_path, captions, out_path, force=force)

    if (script and script == 'commercials') \
            or (not script and 'commercials' not in disable):
        from components import commercial_detection
        commercial_detection.main(out_path, out_path, force=force)

    if not script:
        end = time.time()
        print('Pipeline completed in {:.2f} seconds.'.format(end - start))


def create_output_dirs(in_path, out_path, single):
    """
    Creates output subdirectories for each video being processed.
    Necessary due to docker container processes being owned by root.

    Args:
        in_path (str): path to a video file or batch text file containing
                       filepaths
        out_path (str): path to the output directory.
        single (bool): whether the in_path is a single file or a batch.

    Returns:
        List[Path], List[Path]: a list of video filepaths and a list of output 
            directory paths.

    """

    if single:
        video_paths = [in_path]

    else:  # in_path is a batch text file
        video_paths = [l.strip() for l in open(in_path, 'r') if l.strip()]

    out_paths = [os.path.join(out_path, get_base_name(v)) for v in video_paths]
    for out in out_paths:
        os.makedirs(out, exist_ok=True)

    return video_paths, out_paths


def run_scanner_component(in_path, out_path, video_dirpaths,
                          disable, init_run, force, host):
    """
    Runs scanner_component.py in a docker container.

    Args:
        in_path (pathlib.Path): the path to the input file or batch file.
        out_path (pathlib.Path): the path to the output directory.
        video_dirpaths (List[pathlib.Path]): the paths to each videos parent 
            directory.
        disable (List[str]): a list of named components to disable.
        init_run (bool): whether this is the first time processing the videos.
        force (bool): whether existing outputs should be recomputed.
        host (str): the IP/port that the Docker daemon is listening on.

    """

    cmd = build_scanner_component_command(in_path, out_path, disable, init_run,
                                          force)
    run_command_in_container(cmd, in_path, out_path, video_dirpaths, host)


def build_scanner_component_command(in_path, out_path, disable,
                                    init_run, force):
    """
    Builds a command to run scanner_component.py within a Docker container.
    Args:
        in_path (pathlib.Path): the path to the input file or batch file.
        out_path (pathlib.Path): the path to the output directory.
        disable (List[str]): a list of named components to disable.
        init_run (bool): whether this is the first time processing the videos.
        force (bool): whether existing outputs should be recomputed.
    Returns:
        str: the command to run.
    """

    cmd = ['python3', 'components/scanner_component.py', in_path, out_path]
    if disable:
        scanner_parts = [
            x for x in disable
            if x in ['face_detection', 'face_embeddings', 'face_crops']
        ]
        if scanner_parts:
            cmd.append('-d')
            for d in scanner_parts:
                cmd.append(d)

    if init_run:
        cmd.append('-i')

    if force:
        cmd.append('-f')

    return ' '.join(cmd)


if __name__ == '__main__':
    main(**vars(get_args()))
