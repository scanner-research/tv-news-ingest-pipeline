#!/usr/bin/env python3

"""
File: scanner_component.py
--------------------------
This file handles all video processing that uses scanner, in one location to 
reduce decode overhead.

Input:

    - Video filepath (e.g., '/path/to/video.mp4' or relative path 'video.mp4')

        OR

    - Text filepath containing video filepaths (e.g., '/path/to/batch.txt', 
      
      batch.txt:
          /path/to/video1.mp4
          /path/to/video2.mp4
          ...


Applies the following:

    - face detection
    - face embeddings
    - montage face images
    - black frame detection


This script creates the following outputs rooted in a given output directory:

    <output_directory>/
        - <video1>/
            - bboxes.json
            - embeddings.json
            - black_frames.json
            - metadata.json
            - montages/
                - 0.png
                - 0.json
                - 1.png
                - 1.json
                  ...
        - <video2>/
          ...

"""

import argparse
import os
import sys
import math
import cv2
import json
import subprocess
import shutil
import socket
import struct
import random
from tqdm import tqdm
from subprocess import check_call
from multiprocessing import Pool
from collections import namedtuple
from PIL import Image
import pickle
import numpy as np
from typing import Any, Sequence

import storehouse
import scannerpy as sp
from scannerpy import FrameType, protobufs
from scannerpy.types import BboxList, UniformList, NumpyArrayFloat32, Histogram
from scannerpy.storage import NamedVideoStorage, NamedStorage
import scannertools.face_detection
import scannertools.face_embedding
from scannertools.face_embedding import FacenetEmbeddings
import scannertools.vis
import scannertools.imgproc # for op Histogram


from black_frame_detection import black_frames, get_black_frames_results
from face_detection_and_embeddings import (dilate_bboxes,
                                           get_face_bboxes_results,
                                           get_face_embeddings_results)
from montage_face_images import (crop_faces, get_face_crops_results, 
                                 save_montage_results)
from consts import (OUTFILE_BBOXES, OUTFILE_EMBEDS, OUTFILE_BLACK_FRAMES,
                    OUTFILE_METADATA)
from utils import get_base_name, save_json


# Register scanner ops
black_frames = sp.register_python_op(
    name='BlackFrames', batch=1024)(black_frames)
crop_faces = sp.register_python_op(name='CropFaces')(crop_faces)
dilate_bboxes = sp.register_python_op(name='DilateBboxes')(dilate_bboxes)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path',
        help='path to video file or to text file containing multiple video '
             'paths')
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('-c', '--use-cloud', action='store_true',
        help='run with the videos stored in GCS paths')
    parser.add_argument('-i', '--init-run', action='store_true',
        help='running on videos for the first time (skips check for '
             'incomplete results)')
    parser.add_argument('-s', '--shuffle', action='store_true',
        help='shuffle the order of the videos')
    parser.add_argument('-p', '--pipelines', type=int, default=24)
    parser.add_argument('--interval', type=int, default=3,
        help='interval length in seconds')

    return parser.parse_args()


def main(video_path, out_path, use_cloud, init_run, shuffle, pipelines,
         interval):

    init_scanner_config(use_cloud)

    if not video_path.endswith('.mp4'):
        if not out_path.startswith('gs://') and not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(video_path, 'r') as f:
            video_paths = [l.strip() for l in f if l.strip()]
            if shuffle:
                random.shuffle(video_paths)
            out_paths = [os.path.join(out_path, get_base_name(v))
                         for v in video_paths]

    else:
        video_paths = [video_path]
        out_paths = [out_path]

    process_videos(video_paths, out_paths, use_cloud, init_run, pipelines,
                   interval)


DB_NAME = open('db_name').read().strip()
assert DB_NAME is not None, 'Set the scanner db name! (Create "db_name" file)'


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


def process_videos(video_paths, out_paths, use_cloud, init_run, pipelines,
                   interval):
    print('Processing {} videos'.format(len(video_paths)))

    video_kwargs = {}
    storage = None
    if use_cloud:
        storage = NamedVideoStorage(storage_config=build_storage_config())
        video_kwargs['storage'] = storage

    cl = sp.Client(enable_watchdog=False)

    video_names = [get_base_name(vid) for vid in video_paths]
    videos = [sp.NamedVideoStream(cl, a, path=b, inplace=True, **video_kwargs)
              for a, b in zip(video_names, video_paths)]
    
    all_strides = [
        get_video_stride(video_name, v, interval)
        for video_name, v in zip(
            tqdm(video_names, desc='Computing stride'), videos
        )
    ]

    all_metadata = [
        get_video_metadata(video_name, v)
        for video_name, v in zip(
            tqdm(video_names, desc='Collecting video metadata'), videos
        )
    ]

    if storage:
        storage.ingest(cl, videos)
        videos = [sp.NamedVideoStream(cl, v, inplace=True, **video_kwargs)
                  for v in video_names]

    frames = cl.io.Input(videos)
    strided_frames = cl.streams.Stride(frames, all_strides)
    faces = cl.ops.MTCNNDetectFaces(frame=strided_frames)
    dilated_faces = cl.ops.DilateBboxes(bboxes=faces)
    embeddings = cl.ops.EmbedFaces(frame=strided_frames, bboxes=dilated_faces)
    face_crops = cl.ops.CropFaces(frame=strided_frames, bboxes=dilated_faces)
    histograms = cl.ops.Histogram(frame=frames)
    black_frames = cl.ops.BlackFrames(hists=histograms)

    all_output_faces = [sp.NamedStream(cl, 'face_bboxes_' + v)
                        for v in video_names]
    all_output_embeddings = [sp.NamedStream(cl, 'face_embeddings_' + v)
                             for v in video_names]
    all_output_crops = [sp.NamedStream(cl, 'face_crops_' + v)
                        for v in video_names]
    all_output_black_frames = [sp.NamedStream(cl, 'black_frames_' + v)
                               for v in video_names]

    if not init_run:
        remove_unfinished_outputs(
            cl, video_names, all_output_faces, all_output_embeddings, 
            all_output_crops, all_output_black_frames, clean=False
        )
   
    output_op_faces = cl.io.Output(faces, all_output_faces)
    output_op_embeddings = cl.io.Output(embeddings, all_output_embeddings)
    output_op_crops = cl.io.Output(face_crops, all_output_crops)
    output_op_black_frames = cl.io.Output(black_frames, all_output_black_frames)

    print('Running graph')
    cl.run([output_op_crops, output_op_embeddings, output_op_faces],
           sp.PerfParams.estimate(pipeline_instances_per_node=pipelines),
           cache_mode=sp.CacheMode.Ignore)
    
    cl.run(output_op_black_frames,
           sp.PerfParams.estimate(pipeline_instances_per_node=pipelines),
           cache_mode=sp.CacheMode.Ignore)


    pool = Pool()
    results = []
    tmp_dir = '/tmp/face_crops'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    pbar = tqdm(total=len(video_names) * 5, desc='Collecting output')
    update_pbar = lambda x: pbar.update(1)

    for (video_name, output_path, stride, meta, output_faces, output_embeddings,
         output_crops, output_black_frames) in zip(
            video_names, out_paths, all_strides, all_metadata, 
            all_output_faces, all_output_embeddings, all_output_crops,
            all_output_black_frames
    ):
        if all([out.committed() for out in [
            output_faces, output_embeddings, output_crops, output_black_frames
        ]]):
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            detected_faces = list(output_faces.load(ty=BboxList))
            embedded_faces = list(output_embeddings.load(ty=FacenetEmbeddings))
            cropped_faces = list(output_crops.load())
            video_black_frames = output_black_frames.load()

            # Save metadata
            metadata_outpath = os.path.join(output_path, OUTFILE_METADATA)
            save_json(meta, metadata_outpath)
            update_pbar(None)

            # Save bboxes
            bbox_outpath = os.path.join(output_path, OUTFILE_BBOXES)
            bboxes = get_face_bboxes_results(detected_faces, stride) 
            save_json(bboxes, bbox_outpath)
            update_pbar(None)

            # Save embeddings
            embed_outpath = os.path.join(output_path, OUTFILE_EMBEDS)
            embeds = get_face_embeddings_results(embedded_faces)
            save_json(embeds, embed_outpath)
            update_pbar(None)

            # Save montages
            crops = get_face_crops_results(cropped_faces)
            tmp_path = os.path.join(tmp_dir, '{}.pkl'.format(video_name))
            with open(tmp_path, 'wb') as f:
                pickle.dump(crops, f)

            results.append(pool.apply_async(
                save_montage_results,
                args=(video_name, output_path, bboxes, tmp_path),
                callback=update_pbar
            ))

            # Save black frames
            black_frames_outpath = os.path.join(output_path,
                                                OUTFILE_BLACK_FRAMES)
            black_frames_results = get_black_frames_results(video_black_frames)
            save_json(black_frames_results, black_frames_outpath)
            update_pbar(None)

        else:
            print('Missing results for {}: faces={}, embs={}, crops={}'.format(
                video_name, output_faces.committed(),
                output_embeddings.committed(), output_crops.committed()
            ))

    pool.close()
    for r in results:
        r.get()
    pool.join()


def build_storage_config():
    with open('key.json') as f:
        credentials = json.load(f)
    for k, v in credentials.items():
        os.environ[k] = v
    storage_config = storehouse.StorageConfig.make_gcs_config('esper')
    return storage_config


def get_video_stride(video_name, video, interval, cache_dir='/tmp/stride'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = os.path.join(cache_dir, video_name)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return int(f.read())
    
    fps = video.as_hwang().video_index.fps()
    stride = math.ceil(interval * fps)  # default 3 second intervals
    with open(cache_path, 'w') as f:
        f.write(str(stride))

    return stride


def get_video_metadata(video_name, video):
    return {
        'name': video_name,
        'fps': video.as_hwang().video_index.fps(),
        'frames': video.as_hwang().video_index.frames(),
        'width': video.as_hwang().video_index.frame_width(), 
        'height': video.as_hwang().video_index.frame_height()
    }


def remove_unfinished_outputs(cl, video_names, faces, embeddings, crops,
                              black_frames, clean=False):
    for collection in zip(video_names, faces, embeddings, crops, black_frames):
        video_name = collection[0]
        outputs = collection[1:]
        commits = [out.committed() for out in outputs]

        print(video_name,
            'commits: faces={}, embeddings={}, crops={}, black_frames={}'.format(
                *commits
            )
        )

        # Force a rerun
        if clean or not all(commits):
            print('Rerunning:', video_name)
            NamedStorage().delete(cl, outputs)


def get_video_name(x):
    return os.path.splitext(x.split('/')[-1])[0]


if __name__ == "__main__":
    main(**vars(get_args()))
