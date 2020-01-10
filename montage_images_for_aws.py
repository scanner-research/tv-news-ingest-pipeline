#!/usr/bin/env python3

import argparse
import os
import sys
import math
import cv2
import json
import subprocess
import shutil
import socket
import random
from tqdm import tqdm
from subprocess import check_call
from multiprocessing import Pool
from collections import namedtuple
from PIL import Image
import pickle
import numpy as np
from typing import Any

import storehouse
import scannerpy as sp
from scannerpy import FrameType, protobufs
from scannerpy.types import BboxList, UniformList, NumpyArrayFloat32
from scannerpy.storage import NamedVideoStorage, NamedStorage
import scannertools.face_detection
import scannertools.face_embedding
from scannertools.storage.python import PythonStream
import scannertools.vis


DB_NAME = open('db_name').read().strip()
assert DB_NAME is not None, 'Set the scanner db name!'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('face_meta_path')
    parser.add_argument('out_path')
    parser.add_argument('-c', '--use-cloud', action='store_true')
    parser.add_argument('-i', '--init-run', action='store_true')
    parser.add_argument('-s', '--shuffle', action='store_true')
    parser.add_argument('-p', '--pipelines', type=int, default=12)
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def crop_bbox(img, bbox, expand=0.1):
    y1 = max(bbox['y1'] - expand, 0)
    y2 = min(bbox['y2'] + expand, 1)
    x1 = max(bbox['x1'] - expand, 0)
    x2 = min(bbox['x2'] + expand, 1)
    [h, w] = img.shape[:2]
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


@sp.register_python_op(name='CropFaces')
def crop_faces(config, frame: FrameType, bboxes: Any) -> Any:
    assert len(bboxes) > 0
    return [crop_bbox(frame, bbox) for bbox in bboxes]


IMG_SIZE = 200
BLOCK_SIZE = 250
HALF_BLOCK_SIZE = BLOCK_SIZE / 2
PADDING = int((BLOCK_SIZE - IMG_SIZE) / 2)
assert IMG_SIZE + 2 * PADDING == BLOCK_SIZE, \
    'Bad padding: {}'.format(IMG_SIZE + 2 * PADDING)

BLANK_IMAGE = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), dtype=np.uint8)

NUM_ROWS = 6
NUM_COLS = 10


def convert_image(im):
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, dsize=(IMG_SIZE, IMG_SIZE))
    im = cv2.copyMakeBorder(
        im, PADDING, PADDING, PADDING, PADDING,
        cv2.BORDER_CONSTANT, value=0)
    return im


def montage_faces(face_info, face_crops):
    flat_faces_and_crops = []
    for (frame_num, frame_info), frame_face_crops in zip(face_info, face_crops):
        assert len(frame_info) == len(frame_face_crops)
        for info, img in zip(frame_info, frame_face_crops):
            flat_faces_and_crops.append((frame_num, info, img))

    stacked_rows = []
    for i in range(0, len(flat_faces_and_crops), NUM_COLS):
        row_imgs = [
            convert_image(im)
            for _, _, im in flat_faces_and_crops[i:i + NUM_COLS]
        ]
        while len(row_imgs) < NUM_COLS:
            row_imgs.append(BLANK_IMAGE)
        stacked_row = np.hstack(row_imgs)
        stacked_rows.append(stacked_row)

    stacked_imgs = []
    for i in range(0, len(stacked_rows), NUM_ROWS):
        row_imgs = stacked_rows[i:i + NUM_ROWS]
        if len(row_imgs) > 1:
            stacked_imgs.append(np.vstack(row_imgs))
        else:
            stacked_imgs.append(row_imgs[0])

    stacked_img_meta = []
    num_per_stack = NUM_ROWS * NUM_COLS
    for i in range(0, len(flat_faces_and_crops), num_per_stack):
        img_meta = [(x[0], x[1]) for x in flat_faces_and_crops[i:i + num_per_stack]]
        stacked_img_meta.append({
            'rows': math.ceil(len(img_meta) / NUM_COLS),
            'cols': NUM_COLS,
            'block_dim': BLOCK_SIZE,
            'content': img_meta
        })

    return zip(stacked_imgs, stacked_img_meta)


def get_video_name(x):
    return os.path.splitext(x.split('/')[-1])[0]


def build_storage_config():
    with open('key.json') as f:
        credentials = json.load(f)
    for k, v in credentials.items():
        os.environ[k] = v
    storage_config = storehouse.StorageConfig.make_gcs_config('esper')
    return storage_config


def get_face_bboxes(metadata):
    all_bboxes = []
    for _, frame_meta in metadata:
        frame_bboxes = [x['bbox'] for x in frame_meta]
        assert len(frame_bboxes) > 0
        all_bboxes.append(frame_bboxes)
    return all_bboxes


def get_face_frames(metadata):
    all_frames = []
    prev = -1
    for i, _ in metadata:
        all_frames.append(i)
        assert i > prev, '{} <= {}'.format(i, prev)
        prev = i
    # assert len(all_frames) > 0
    return all_frames


def save_results(video_name, out_path, metadata, face_crops_file):
    with open(face_crops_file, 'rb') as in_f:
        face_crops = pickle.load(in_f)
        os.remove(face_crops_file)

    results = montage_faces(metadata, face_crops)

    tmp_dir = os.path.join('/tmp', video_name)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for i, result in enumerate(results):
        img, meta = result
        Image.fromarray(img).save('{}/{}.png'.format(tmp_dir, i),
                                  optimize=True)
        with open('{}/{}.json'.format(tmp_dir, i), 'w') as f:
            json.dump(meta, f)

    if out_path.startswith('gs://'):
        cmd = ['gsutil', '-m', 'cp', '-r', tmp_dir, out_path]
        print('Copying:', cmd)
        check_call(cmd)
        shutil.rmtree(tmp_dir)
    else:
        shutil.move(tmp_dir, out_path)


def process_videos(video_paths, meta_paths, out_paths, use_cloud, init_run,
                   pipelines):
    print('Processing {} videos'.format(len(video_paths)))
    assert len(meta_paths) == len(video_paths)

    all_metadata = []
    meta_suffix_len = len('.faces')
    for a, b in tqdm(list(zip(video_paths, meta_paths))):
        a_name = get_video_name(a)
        b_name = get_video_name(b)[:-meta_suffix_len]
        assert a_name == b_name, '{} != {}'.format(a_name, b_name)
        all_metadata.append(sorted(load_json(b)))
    print('Loaded metadata')

    video_kwargs = {}
    storage = None
    if use_cloud:
        storage = NamedVideoStorage(storage_config=build_storage_config())
        video_kwargs['storage'] = storage

    cl = sp.Client(enable_watchdog=False)

    video_names = [get_video_name(video_path) for video_path in video_paths]

    videos = [sp.NamedVideoStream(cl, a, path=b, inplace=True, **video_kwargs)
              for a, b in zip(video_names, video_paths)]

    if storage:
        storage.ingest(cl, videos)
        videos_with_paths = videos
        videos = [sp.NamedVideoStream(cl, v, inplace=True, **video_kwargs)
                  for v in video_names]
    print('Ingested video')

    frames = cl.io.Input(videos)
    selected_frames = cl.streams.Gather(frames, [
        get_face_frames(metadata) for metadata in all_metadata
    ])
    all_face_bboxes = cl.io.Input([PythonStream(get_face_bboxes(metadata))
                                   for metadata in all_metadata])

    face_crops = cl.ops.CropFaces(frame=selected_frames,
                                  bboxes=all_face_bboxes)

    all_output_crops = [sp.NamedStream(cl, 'face-crops:' + v)
                        for v in video_names]

    if not init_run:
        to_delete = []
        for video_name, output_crops in zip(video_names, all_output_crops):
            print(video_name, 'commit_crops={}'.format(output_crops.committed()))
            # Force a rerun
            if not output_crops.committed():
                to_delete.append(output_crops)
        NamedStorage().delete(cl, to_delete)

    output_op = cl.io.Output(face_crops, all_output_crops)

    print('Running graph')
    cl.run(output_op,
           sp.PerfParams.estimate(pipeline_instances_per_node=pipelines),
           cache_mode=sp.CacheMode.Ignore)

    tmp_dir = '/tmp/face_crops'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    with Pool() as pool, tqdm(total=len(out_paths), desc='Stitching') as pbar:
        def update_pbar(x):
            pbar.update(1)
        results = []
        for video_name, out_path, metadata, output_crop in zip(
            video_names, out_paths, all_metadata, all_output_crops
        ):
            if output_crop.committed():
                tmp_path = os.path.join(tmp_dir, '{}.pkl'.format(video_name))
                with open(tmp_path, 'wb') as f:
                    pickle.dump(list(output_crop.load()), f)
                results.append(pool.apply_async(
                    save_results,
                    args=(video_name, out_path, metadata, tmp_path),
                    callback=update_pbar))
            else:
                print('Failed:', video_name)
        pool.close()
        for r in results:
            r.get()
        pool.join()


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


def main(video_path, out_path, face_meta_path, use_cloud, init_run, shuffle,
         pipelines):
    init_scanner_config(use_cloud)
    if not video_path.endswith('.mp4'):
        if not out_path.startswith('gs://') and not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(video_path, 'r') as f:
            video_paths = [l.strip() for l in f if l.strip()]
            if shuffle:
                random.shuffle(video_paths)
            out_paths = [os.path.join(out_path, get_video_name(v))
                         for v in video_paths]
            meta_paths = [os.path.join(face_meta_path,
                                       get_video_name(v) + '.faces.json')
                          for v in video_paths]
    else:
        video_paths = [video_path]
        out_paths = [out_path]
        meta_paths = [face_meta_path]
    process_videos(video_paths, meta_paths, out_paths, use_cloud, init_run,
                   pipelines)


if __name__ == "__main__":
    main(**vars(get_args()))
