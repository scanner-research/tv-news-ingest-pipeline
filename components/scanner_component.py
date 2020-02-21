#!/usr/bin/env python3

"""
File: scanner_component.py
--------------------------
Script for detecting faces, computing face embeddings, and extracting cropped 
images of the faces using Scanner. All these outputs are computed together to 
reduce frame decode overhead.

Since all Scanner outputs for a single run require the same dimensionality, 
black frame detection cannot be run together with these outputs that are 
computed on strided frames.

Be careful when modifying the number of pipelines: you never need more 
pipelines than videos, and a ~4GB TensorFlow model will be loaded per 
pipeline, so if you run into any errors related to memory, try decreasing the 
number of pipelines.


Example #1: Single Video
------------------------

        in_path:  my_video.mp4
        out_path: my_output_dir

    outputs

        my_output_dir/
        ├── bboxes.json
        ├── crops
        │   ├── 0.png
        │   ├── 1.png
        │   └── 2.png
        ├── embeddings.json
        └── metadata.json
        

Example #2: Batch Video
-----------------------

        in_path:  my_batch.txt
        out_path: my_output_dir

    where 'my_batch.txt' looks like:

        path/to/my_video1.mp4
        different/path/to/my_video2.mp4

    outputs

        my_output_dir/
        ├── my_video1/
        │   ├── bboxes.json
        │   ├── crops
        │   │   ├── 0.png
        │   │   ├── 1.png
        │   │   └── 2.png
        │   ├── embeddings.json
        │   └── metadata.json
        └── my_video2/
            ├── bboxes.json
            ├── crops
            │   ├── 0.png
            │   ├── 1.png
            │   └── 2.png
            ├── embeddings.json
            └── metadata.json

"""

import argparse
from functools import partial
import os
import sys
import math
import json
import pickle
import tqdm
from tqdm import tqdm
from multiprocessing import Pool

import scannerpy as sp
from scannerpy import FrameType, protobufs
from scannerpy.storage import NamedStorage
from scannerpy.types import BboxList
import scannertools.face_detection
import scannertools.face_embedding
from scannertools.face_embedding import FacenetEmbeddings

from components.face_detection_and_embeddings import (dilate_bboxes, crop_faces,
                                           get_face_bboxes_results,
                                           get_face_embeddings_results,
                                           handle_face_crops_results)
from util.config import NUM_PIPELINES, STRIDE
from util.consts import (OUTFILE_BBOXES, OUTFILE_EMBEDS, OUTFILE_METADATA,
                    OUTDIR_CROPS, SCANNER_COMPONENT_OUTPUTS)
from util.utils import (get_base_name, get_batch_io_paths, init_scanner_config,
                   json_is_valid, remove_unfinished_outputs, save_json)

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NAMED_COMPONENTS = [
    'face_detection',
    'face_embeddings',
    'face_crops'
]

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
    parser.add_argument('--interval', type=int, default=STRIDE,
                        help='interval length in seconds')
    parser.add_argument('-d', '--disable', nargs='+', choices=NAMED_COMPONENTS,
                        help='list of named components to disable')
    return parser.parse_args()


def main(in_path, out_path, init_run=False, force_rerun=False, 
        pipelines=NUM_PIPELINES, interval=STRIDE, disable=None):

    init_scanner_config()

    if not in_path.endswith('.mp4'):
        video_paths, out_paths = get_batch_io_paths(in_path, out_path)
    else:
        video_paths = [in_path]
        out_paths = [out_path]

    process_videos(video_paths, out_paths, init_run, force_rerun, pipelines, 
                   interval, disable)


def process_videos(video_paths, out_paths, init_run=False, rerun=False,
                   pipelines=NUM_PIPELINES, interval=STRIDE, disable=None):

    assert len(video_paths) == len(out_paths), ('Mismatch between video and '
                                                'output paths')

    if disable is None:
        disable = []

    cl = sp.Client(enable_watchdog=False)

    video_names = [get_base_name(vid) for vid in video_paths]
    if not init_run and not rerun:
        for i in range(len(video_names) - 1, -1, -1):
            if (('face_detection' in disable
                or json_is_valid(os.path.join(out_paths[i], OUTFILE_BBOXES)))
                and ('face_embeddings' in disable
                    or json_is_valid(os.path.join(out_paths[i], OUTFILE_EMBEDS)))
                and json_is_valid(os.path.join(out_paths[i], OUTFILE_METADATA))
                and ('face_crops' in disable
                    or os.path.isdir(os.path.join(out_paths[i], OUTDIR_CROPS)))
            ):
                video_names.pop(i)
                out_paths.pop(i)

    if not video_names:
        print('All videos have existing scanner outputs')
        return
    
    # Don't reingest videos with existing output
    videos = [sp.NamedVideoStream(cl, a, path=b, inplace=True)
              for a, b in zip(video_names, video_paths)]
    
    all_strides = [
        get_video_stride(video_name, v, interval)
        for video_name, v in zip(
            tqdm(video_names, desc='Computing stride', unit='video'), videos
        )
    ]

    all_metadata = [
        get_video_metadata(video_name, v)
        for video_name, v in zip(
            tqdm(video_names, desc='Collecting metadata', unit='video'), videos
        )
    ]

    frames = cl.io.Input(videos)
    strided_frames = cl.streams.Stride(frames, all_strides)
    faces = cl.ops.MTCNNDetectFaces(frame=strided_frames)
    dilated_faces = cl.ops.DilateBboxes(bboxes=faces)
    embeddings = cl.ops.EmbedFaces(frame=strided_frames, bboxes=dilated_faces)
    face_crops = cl.ops.CropFaces(frame=strided_frames, bboxes=dilated_faces)

    all_outputs = []
    output_ops = []
    if 'face_crops' not in disable:
        all_output_crops = [sp.NamedStream(cl, 'face_crops:' + v)
                            for v in video_names]
        all_outputs.append(all_output_crops)
        output_ops.append(cl.io.Output(face_crops, all_output_crops))
    else:
        all_output_crops = [None] * len(video_names)

    if 'face_embeddings' not in disable:
        all_output_embeddings = [sp.NamedStream(cl, 'face_embeddings:' + v)
                                 for v in video_names]
        all_outputs.append(all_output_embeddings)
        output_ops.append(cl.io.Output(embeddings, all_output_embeddings))
    else:
        all_output_embeddings = [None] * len(video_names)

    if 'face_detection' not in disable:
        all_output_faces = [sp.NamedStream(cl, 'face_bboxes:' + v)
                            for v in video_names]
        all_outputs.append(all_output_faces)
        output_ops.append(cl.io.Output(faces, all_output_faces))
    else:
        all_output_faces = [None] * len(video_names)

    if not init_run or rerun:
        remove_unfinished_outputs(
            cl, video_names,
            all_outputs,
            del_fn=lambda c, o: NamedStorage().delete(c, o),
            clean=rerun
        )
  
    print('Running graph')
    cl.run(output_ops,
           sp.PerfParams.estimate(pipeline_instances_per_node=pipelines),
           cache_mode=sp.CacheMode.Ignore)
  
    # Async callback function
    def callback_fn(data=None, path=None, save_fn=None, pbar=None):
        if save_fn:
            save_fn(data, path)
        if pbar:
            pbar.update()

    tmp_dir = '/tmp/face_crops'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    with Pool() as workers, tqdm(
        total=len(video_names) * (len(SCANNER_COMPONENT_OUTPUTS) - len(disable)),
        desc='Collecting output', unit='output'
    ) as pbar:
        for (video_name, out_path, stride, meta, output_faces, 
             output_embeddings, output_crops) in zip(
                video_names, out_paths, all_strides, all_metadata, 
                all_output_faces, all_output_embeddings, all_output_crops
        ):
            if all(out is None or out.committed() for out in 
                   [output_faces, output_embeddings, output_crops]):
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)

                if output_faces is not None:
                    detected_faces = list(output_faces.load(ty=BboxList))
                if output_embeddings is not None:
                    embedded_faces = list(
                        output_embeddings.load(ty=FacenetEmbeddings)
                    )
                if output_crops is not None:
                    cropped_faces = list(output_crops.load())

                # Save metadata
                metadata_outpath = os.path.join(out_path, OUTFILE_METADATA)
                save_json(meta, metadata_outpath)
                pbar.update()
                
                # Save bboxes
                if output_faces is not None:
                    bbox_outpath = os.path.join(out_path, OUTFILE_BBOXES)
                    callback = partial(callback_fn, path=bbox_outpath,
                                       save_fn=save_json, pbar=pbar)
                    workers.apply_async(
                        get_face_bboxes_results,
                        args=(detected_faces, stride),
                        callback=partial(callback_fn, path=bbox_outpath,
                                         save_fn=save_json, pbar=pbar)
                    )

                # Save embeddings
                if output_embeddings is not None:
                    embed_outpath = os.path.join(out_path, OUTFILE_EMBEDS)
                    workers.apply_async(
                        get_face_embeddings_results,
                        args=(embedded_faces,),
                        callback=partial(callback_fn, path=embed_outpath,
                                         save_fn=save_json, pbar=pbar)
                    )

                # Save crops
                if output_crops is not None:
                    tmp_path = os.path.join(tmp_dir, '{}.pkl'.format(video_name))
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(cropped_faces, f)
                    crops_outpath = os.path.join(out_path, OUTDIR_CROPS)
                    result = workers.apply_async(
                        handle_face_crops_results, 
                        args=(tmp_path, crops_outpath),
                        callback=partial(callback_fn, pbar=pbar)
                    )
                
            else:
                tqdm.write(('Missing results for {}: faces={}, embs={}, '
                       'crops={}').format(
                    video_name, output_faces.committed(),
                    output_embeddings.committed(), output_crops.committed()
                ))

                                    
        workers.close()
        workers.join()


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


if __name__ == "__main__":
    main(**vars(get_args()))
