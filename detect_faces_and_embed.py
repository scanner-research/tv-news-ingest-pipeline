#!/usr/bin/env python3

import argparse
import os
import sys
import math
import json
import subprocess
import shutil
import socket
import random
from tqdm import tqdm

import storehouse
import scannerpy as sp
from scannerpy import FrameType, protobufs
from scannerpy.types import BboxList
from scannerpy.storage import NamedVideoStorage, NamedStorage
import scannertools.face_detection
import scannertools.face_embedding
from scannertools.face_embedding import FacenetEmbeddings
import scannertools.vis


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('out_path')

    # Run with the videos stored in GCS paths
    parser.add_argument('-c', '--use-cloud', action='store_true')

    # Shuffle the order of the videos so that progress gets made if there
    # bad videos causing crashes
    parser.add_argument('-s', '--shuffle', action='store_true')

    # Pass this when running the first time (save's the costly check for
    # incomplete results)
    parser.add_argument('-i', '--init-run', action='store_true')
    return parser.parse_args()


def bbox_to_dict(b):
    return {'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2, 'score': b.score}


def get_video_name(x):
    return os.path.splitext(x.split('/')[-1])[0]


def zip_results(stride, detected_faces, embedded_faces):
    assert isinstance(stride, int)
    assert len(detected_faces) == len(embedded_faces)
    result = []
    frame_num = 0
    for faces, embeddings in zip(detected_faces, embedded_faces):
        if len(faces) > 0:
            assert len(faces) == len(embeddings)
            faces_in_frame = []
            for face, embedding in zip(faces, embeddings):
                faces_in_frame.append(
                    {'bbox': bbox_to_dict(face), 'emb': embedding.tolist()})
            result.append((frame_num, faces_in_frame))
        else:
            assert isinstance(embeddings, sp.storage.NullElement)
        frame_num += stride
    return result


DILATE_AMOUNT = 1.05


@sp.register_python_op(name='DilateBboxes')
def dilate_bboxes(config, bboxes: BboxList) -> BboxList:
    return [
        protobufs.BoundingBox(
            x1=bb.x1 * (2. - DILATE_AMOUNT),
            x2=bb.x2 * DILATE_AMOUNT,
            y1=bb.y1 * (2. - DILATE_AMOUNT),
            y2=bb.y2 * DILATE_AMOUNT,
            score=bb.score
        ) for bb in bboxes
    ]


def build_storage_config():
    with open('key.json') as f:
        credentials = json.load(f)
    for k, v in credentials.items():
        os.environ[k] = v
    storage_config = storehouse.StorageConfig.make_gcs_config('esper')
    return storage_config


def process_videos(video_paths, out_paths, use_cloud, init_run):
    print('Processing {} videos'.format(len(video_paths)))
    video_kwargs = {}
    storage = None
    if use_cloud:
        storage = NamedVideoStorage(storage_config=build_storage_config())
        video_kwargs['storage'] = storage

    cl = sp.Client(enable_watchdog=False)

    video_names = [get_video_name(video_path) for video_path in video_paths]

    videos = [sp.NamedVideoStream(cl, a, path=b, inplace=True, **video_kwargs)
              for a, b in zip(video_names, video_paths)]

    def get_stride(video_name, v, cache_dir='/tmp/stride'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = os.path.join(cache_dir, video_name)
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                stride = int(f.read())
        else:
            fps = v.as_hwang().video_index.fps()
            stride = math.ceil(3 * fps)
            with open(cache_path, 'w') as f:
                f.write(str(stride))
        return stride

    all_strides = [
        get_stride(video_name, v)
        for video_name, v in zip(
            video_names, tqdm(videos, desc='Computing stride')
        )]

    if storage:
        storage.ingest(cl, videos)
        videos_with_paths = videos
        videos = [sp.NamedVideoStream(cl, v, inplace=True, **video_kwargs)
                  for v in video_names]

    frames = cl.io.Input(videos)
    strided_frames = cl.streams.Stride(frames, all_strides)

    faces = cl.ops.MTCNNDetectFaces(frame=strided_frames)
    dilated_faces = cl.ops.DilateBboxes(bboxes=faces)
    embeddings = cl.ops.EmbedFaces(frame=strided_frames, bboxes=dilated_faces)

    all_output_faces = [sp.NamedStream(cl, 'face_bboxes_' + v)
                        for v in video_names]
    all_output_embeddings = [sp.NamedStream(cl, 'face_embeddings_' + v)
                             for v in video_names]

    if not init_run:
        for video_name, output_faces, output_embeddings in zip(
            video_names, all_output_faces, all_output_embeddings
        ):
            print(video_name, 'commit_faces={}'.format(output_faces.committed()),
                  'commit_embs={}'.format(output_embeddings.committed()))
            # Force a rerun
            if not output_faces.committed() or not output_embeddings.committed():
                print('Rerunning:', video_name)
                NamedStorage().delete(cl, [output_faces, output_embeddings])

    output_op1 = cl.io.Output(faces, all_output_faces)
    output_op2 = cl.io.Output(embeddings, all_output_embeddings)

    print('Running graph')
    cl.run([output_op1, output_op2],
           sp.PerfParams.estimate(pipeline_instances_per_node=24),
           cache_mode=sp.CacheMode.Ignore)

    print('Collecting output')
    # exp_num_frames = math.ceil(video_info.num_frames / stride)
    for video_name, out_path, stride, output_faces, output_embeddings in tqdm(
        list(zip(video_names, out_paths, all_strides, all_output_faces,
                 all_output_embeddings))
    ):
        if output_faces.committed() and output_embeddings.committed():
            detected_faces = list(output_faces.load())
            # assert len(detected_faces) == exp_num_frames
            embedded_faces = list(output_embeddings.load(ty=FacenetEmbeddings))
            # assert len(embedded_faces) == exp_num_frames
            results = zip_results(stride, detected_faces, embedded_faces)
            with open(out_path + '.json', 'w') as f:
                json.dump(results, f)
        else:
            print('Missing results for {}. Faces={} Embs={}'.format(
                  video_name, output_faces.committed(),
                  output_embeddings.committed()))


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
""".format(socket.gethostname())

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


def main(video_path, out_path, use_cloud, init_run, shuffle):
    init_scanner_config(use_cloud)
    if not video_path.endswith('.mp4'):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(video_path, 'r') as f:
            video_paths = [x for x in (l.strip() for l in f)
                           if x and x[0] != '#']
            if shuffle:
                print('Shuffling the video list.')
                random.shuffle(video_paths)
            out_paths = [os.path.join(out_path, get_video_name(v))
                         for v in video_paths]
    else:
        video_paths = [video_path]
        out_paths = [out_path]
    process_videos(video_paths, out_paths, use_cloud, init_run)


if __name__ == "__main__":
    main(**vars(get_args()))
