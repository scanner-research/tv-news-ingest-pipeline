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


Example #1: Single
------------------

        in_path:  my_video.mp4
        out_path: my_output_dir

    outputs

        my_output_dir/
        └── black_frames.json


Example #2: Batch
-----------------

        in_path:  my_batch.txt
        out_path: my_output_dir
        
    where 'my_batch.txt' looks like:

        path/to/my_video1.mp4
        different/path/to/my_video2.mp4

    outputs 

        my_output_dir/
        ├── my_video1
        │   └── black_frames.json
        └── my_video2
            └── black_frames.json

"""

import argparse
import json
from multiprocessing import Pool
import os
import struct
from typing import Sequence

from tqdm import tqdm

import scannerpy as sp
from scannerpy.storage import NamedStorage
from scannerpy.types import Histogram
import scannertools.imgproc  # for op Histogram

from util.config import NUM_PIPELINES
from util.consts import OUTFILE_BLACK_FRAMES
from util.utils import (get_base_name, get_batch_io_paths, init_scanner_config,
                   json_is_valid, remove_unfinished_outputs, save_json)

# Twice as manner for black frames as for face detection.
NUM_PIPELINES = NUM_PIPELINES * 2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('-i', '--init-run', action='store_true',
                        help='running on videos for the first time')
    parser.add_argument('-f', '--force-rerun', action='store_true',
                        help='force rerun for all videos')
    parser.add_argument('-p', '--pipelines', type=int, default=NUM_PIPELINES, 
                        help='number of pipelines for scanner')
    return parser.parse_args()


def main(in_path, out_path, init_run=False, force_rerun=False,
         pipelines=NUM_PIPELINES):
    init_scanner_config()

    if not in_path.endswith('.mp4'):
        video_paths, out_paths = get_batch_io_paths(in_path, out_path)
    else:
        video_paths = [in_path]
        out_paths = [out_path]

    process_videos(video_paths, out_paths, init_run, force_rerun, pipelines)


def process_videos(video_paths, out_paths, init_run=False, rerun=False,
                   pipelines=NUM_PIPELINES):
    
    assert len(video_paths) == len(out_paths), ('Mismatch between video and '
                                                'output paths')

    cl = sp.Client(enable_watchdog=False)

    video_names = [get_base_name(path) for path in video_paths]
    if not init_run and not rerun:
        for i in range(len(video_names) - 1, -1, -1):
            if json_is_valid(os.path.join(out_paths[i], OUTFILE_BLACK_FRAMES)):
                video_names.pop(i)
                out_paths.pop(i)

    if not video_names:
        print('All videos have existing black frame outputs')
        return

    # Don't reingest videos with existing output
    videos = [sp.NamedVideoStream(cl, a, path=b, inplace=True)
              for a, b in zip(video_names, video_paths)]

    frames = cl.io.Input(videos)
    histograms = cl.ops.Histogram(frame=frames)
    black_frames = cl.ops.BlackFrames(hists=histograms)

    all_black_frame_outputs = [sp.NamedStream(cl, 'black_frames:' + v)
                               for v in video_names]

    if not init_run or rerun:
        remove_unfinished_outputs(cl, video_names, [all_black_frame_outputs],
                del_fn=lambda c, o: NamedStorage().delete(c, o), clean=rerun)

    output_op = cl.io.Output(black_frames, all_black_frame_outputs)

    print('Running graph')
    cl.run(output_op,
           sp.PerfParams.estimate(pipeline_instances_per_node=pipelines),
           cache_mode=sp.CacheMode.Ignore)

    with Pool() as workers, tqdm(
        total=len(video_names), desc='Collecting output', unit='video'
    ) as pbar:
        for video_name, out_path, output_black_frames in zip(
            video_names, out_paths, all_black_frame_outputs
        ):
            if output_black_frames.committed():
                video_black_frames = list(output_black_frames.load())
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                black_frames_outpath = os.path.join(out_path,
                                                    OUTFILE_BLACK_FRAMES)
                workers.apply_async(
                    get_black_frames_results,
                    args=(video_black_frames,),
                    callback=(lambda x: save_json(x, black_frames_outpath)
                                        or pbar.update())
                )

        workers.close()
        workers.join()

        
@sp.register_python_op(name='BlackFrames', batch=1024)
def black_frames(config, hists: Sequence[Histogram]) -> Sequence[bytes]:
    output = []
    for h in hists:
        threshold = 0.99 * sum(h[0])
        is_black = (h[0][0] > threshold and h[1][0] > threshold 
                    and h[2][0] > threshold)
        output.append(struct.pack('B', 1 if is_black else 0))

    return output


def get_black_frames_results(video_black_frames):
    return [i for i, b in enumerate(video_black_frames) if ord(b) > 0]


if __name__ == '__main__':
    main(**vars(get_args()))
