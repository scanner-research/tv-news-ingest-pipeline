#!/usr/bin/env python3

import argparse
import os
import sys
import math
import json
import shutil
import socket
import struct
import random
from tqdm import tqdm
import numpy as np
from typing import Any, Sequence

import storehouse
import scannerpy as sp
from scannerpy import FrameType, protobufs
from scannerpy.types import NumpyArrayFloat32, Histogram
from scannerpy.storage import NamedVideoStorage, NamedStorage
import scannertools.imgproc
import scannertools.vis


DB_NAME = open('db_name').read().strip()
assert DB_NAME is not None, 'Set the scanner db name!'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('out_path')
    parser.add_argument('-c', '--use-cloud', action='store_true')
    parser.add_argument('-i', '--init-run', action='store_true')
    parser.add_argument('-s', '--shuffle', action='store_true')
    parser.add_argument('-p', '--pipelines', type=int,
                        default=os.cpu_count() * 2)
    parser.add_argument('--no-run', action='store_true')
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


@sp.register_python_op(name='BlackFrames', batch=1024)
def black_frames(config, hists: Sequence[Histogram]) -> Sequence[bytes]:
    output = []
    for h in hists:
        threshold = 0.99 * sum(h[0])
        is_black = h[0][0] > threshold and h[1][0] > threshold and h[2][0] > threshold
        output.append(struct.pack('B', 1 if is_black else 0))
    return output


def get_video_name(x):
    return os.path.splitext(x.split('/')[-1])[0]


def build_storage_config():
    with open('key.json') as f:
        credentials = json.load(f)
    for k, v in credentials.items():
        os.environ[k] = v
    storage_config = storehouse.StorageConfig.make_gcs_config('esper')
    return storage_config


def process_videos(video_paths, out_paths, use_cloud, init_run, no_run,
                   pipelines):
    print('Processing {} videos'.format(len(video_paths)))
    assert len(video_paths) == len(out_paths)

    video_kwargs = {}
    storage = None
    if use_cloud:
        storage = NamedVideoStorage(storage_config=build_storage_config())
        video_kwargs['storage'] = storage

    cl = sp.Client(enable_watchdog=False)

    video_names = [get_video_name(video_path) for video_path in video_paths]

    videos = [sp.NamedVideoStream(cl, a, path=b, inplace=True, **video_kwargs)
              for a, b in zip(video_names, video_paths)]

    if storage and not no_run:
        storage.ingest(cl, videos)
        videos_with_paths = videos
        videos = [sp.NamedVideoStream(cl, v, inplace=True, **video_kwargs)
                  for v in video_names]
    print('Ingested video')

    frames = cl.io.Input(videos)
    histograms = cl.ops.Histogram(frame=frames)
    black_frames = cl.ops.BlackFrames(hists=histograms)

    all_black_frame_outputs = [sp.NamedStream(cl, 'black-frames:' + v)
                               for v in video_names]

    if not init_run and not no_run:
        to_delete = []
        for video_name, black_frame_output in zip(
            video_names, all_black_frame_outputs
        ):
            print(video_name, 'commit_black_frames={}'.format(
                  black_frame_output.committed()))
            # Force a rerun
            if not black_frame_output.committed():
                to_delete.append(black_frame_output)
        NamedStorage().delete(cl, to_delete)

    output_op = cl.io.Output(black_frames, all_black_frame_outputs)

    print('Running graph')
    if not no_run:
        cl.run(output_op,
               sp.PerfParams.estimate(pipeline_instances_per_node=pipelines),
               cache_mode=sp.CacheMode.Ignore)

    print('Collecting results')
    for video_name, out_path, black_frame_output in tqdm(
        list(zip(video_names, out_paths, all_black_frame_outputs))
    ):
        if black_frame_output.committed():
            print('Writing:', video_name)
            video_black_frames = black_frame_output.load()
            if not os.path.exists(out_path):
                with open(out_path, 'w') as f:
                    json.dump([
                        i for i, b in enumerate(video_black_frames)
                        if ord(b) > 0
                    ], f)
        else:
            print('Not committed:', video_name)
    print('Done')


GCS_TOML = """
# Scanner config
# Copy this to ~/.scanner/config.toml

[storage]
type = "gcs"
bucket = "esper"
db_path = "tvnews/scanner-james/{}"
[network]
worker_port = "5002"
master = "localhost"
master_port = "5001"
""".format(DB_NAME)

LOCAL_TOML = """
# Scanner config
# Copy this to ~/.scanner/config.toml

[storage]
type = "posix"
db_path = "/root/.scanner/db"
[network]
worker_port = "5002"
master = "localhost"
master_port = "5001"
"""


def init_scanner_config(use_cloud):
    scanner_config_dir = '/root/.scanner'
    if not os.path.exists(scanner_config_dir):
        os.makedirs(scanner_config_dir)
    with open(os.path.join(scanner_config_dir, 'config.toml'), 'w') as f:
        f.write(GCS_TOML if use_cloud else LOCAL_TOML)


def main(video_path, out_path, use_cloud, init_run, shuffle, no_run, pipelines):
    init_scanner_config(use_cloud)
    if not video_path.endswith('.mp4'):
        if not out_path.startswith('gs://') and not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(video_path, 'r') as f:
            video_paths = [l.strip() for l in f if l.strip()]
            if shuffle:
                random.shuffle(video_paths)
            out_paths = [os.path.join(out_path, get_video_name(v) + '.black_frames.json')
                         for v in video_paths]
    else:
        video_paths = [video_path]
        out_paths = [out_path]
    process_videos(video_paths, out_paths, use_cloud, init_run, no_run,
                   pipelines)


if __name__ == "__main__":
    main(**vars(get_args()))
