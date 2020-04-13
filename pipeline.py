#!/usr/bin/env python3

"""
File: pipeline.py
-----------------
This script is the interface to the TV-News video processing pipeline.

Given a video filepath or textfile containing a list of video filepaths, this
script takes the video(s) through the following stages:

    - detect faces (detect_faces_and_compute_embeddings.py)

    - compute face embeddings (detect_faces_and_compute_embeddings.py)

    - extract face image crops (detect_faces_and_compute_embeddings.py)

    - detect black frames (black_frame_detection.py)

    - identify faces (identify_faces_with_aws.py)

    - propagate identities to unlabeled faces (identity_propagation.py)

    - classify gender (classify_gender.py)

    - copy original captions (copy_captions.py)

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
import multiprocessing as mp
import os
from pathlib import Path
import time

from util import config

NAMED_COMPONENTS = [
    'face_component',
    'black_frames',
    'identities',
    'identity_propagation',
    'genders',
    'captions_copy',
    'caption_alignment',
    'commercials'
]


class PipelineError(Exception):
    """Base class for Pipeline errors."""
    pass


class FileTypeNotSupportedError(PipelineError):
    """For unsupported file types."""

    def __init__(self, type, format, supported):
        self.type = type
        self.format = format
        self.supported = supported

        self.message = f'The {format} {type} format is not supported. ' \
                       f'Try one of these instead: {supported}.'

        super().__init__(self.message)


def get_args():
    """Get command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('--captions', help=('path to srt or to a text file '
                                            'containing srt filepaths'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('-i', '--init-run', action='store_true',
                        help='running on videos for the first time')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force rerun for all videos')
    parser.add_argument('-d', '--disable', nargs='+', choices=NAMED_COMPONENTS,
                        help='list of named components to disable')
    parser.add_argument('-s', '--script', choices=NAMED_COMPONENTS,
                        help='run a single component of the pipeline as a script')
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='run two branches of components in parallel')
    return parser.parse_args()


def main(in_path, captions, out_path, init_run=False, force=False,
         disable=None, script=None, parallel=False):
    """
    The entrypoint for the pipeline.

    Args:
        in_path (str): the path to the video file or batch file.
        captions (str): the path to the captions file or batch captions file.
        out_path (str): the path to the output directory.
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

    # Validate file formats
    single = not in_path.endswith('.txt') and not os.path.isdir(in_path)
    if single and not in_path.endswith('.mp4'):
        raise FileTypeNotSupportedError(
            'video', Path(in_path).suffix.strip('.'), ['mp4']
        )

    if single and captions is not None and not captions.endswith('.srt'):
        raise FileTypeNotSupportedError(
            'captions', Path(captions).suffix.strip('.'), ['srt']
        )

    n_videos = 1 if single else len([l for l in open(in_path, 'r') if l.strip()])

    # Step through each pipeline component
    should_run = lambda c: script and script == c or (not script and c not in disable)

    if should_run('face_component'):
        # Import component only when necessary, in case deps aren't installed
        from components import detect_faces_and_compute_embeddings
        detect_faces_and_compute_embeddings.main(in_path, out_path, init_run,
                                                 force)

    # Computation that relies on the outputs of the face component
    # Separated to allow for parallel branches with `-p` flag
    def faces_path():
        if should_run('identities'):
            from components import identify_faces_with_aws
            identify_faces_with_aws.main(out_path, out_path, force=force)

        if should_run('identity_propagation'):
            from components import identity_propagation
            identity_propagation.main(out_path, out_path, force=force)

        if should_run('genders'):
            from components import classify_gender
            classify_gender.main(out_path, out_path, force=force)

    if parallel:
        proc = mp.Process(target=faces_path)
        proc.start()
    else:
        faces_path()

    if should_run('black_frames'):
        from components import detect_black_frames
        detect_black_frames.main(in_path, out_path, init_run, force)

    # Captions are optional so make sure they are provided
    if captions is not None:
        if should_run('captions_copy'):
            from components import copy_captions
            copy_captions.main(captions, out_path)

        if should_run('caption_alignment'):
            from components import caption_alignment
            caption_alignment.main(in_path, captions, out_path, force=force)

    if should_run('commercials'):
        from components import commercial_detection
        commercial_detection.main(out_path, out_path, force=force)

    if parallel:
        proc.join()

    if not script:
        end = time.time()
        print(f'{"Script" if script else "Pipeline"} completed over {n_videos} '
              f'videos in {end - start:.2f} seconds.')


if __name__ == '__main__':
    main(**vars(get_args()))
