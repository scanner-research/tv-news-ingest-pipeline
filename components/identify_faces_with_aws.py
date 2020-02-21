#!/usr/bin/env python3

"""
File: identify_faces_with_aws.py
--------------------------------
Identifies faces from face montages through AWS.

Input:

    - Path to directory containing one subdirectory per video:
    
      "path/to/<input_dir>" where <input_dir> looks like

        <input_dir>/
            - <video1>/
            - <video2>/
            ...

This script creates the following outputs within the video subdirectories:

    input_dir/
        - video1/
            - identities.json
        - video2/
            - identities.json
        ...

or for single file:
    
    output_dir/
        - identities.json

where there is one JSON file per video containing a list of (face_id, identity)
tuples.

"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from multiprocessing import Pool
import os
from PIL import Image, ImageDraw
import time

import tqdm
from tqdm import tqdm
import boto3

from util.consts import OUTFILE_IDENTITIES, OUTDIR_CROPS
from util.config import AWS_CREDENTIALS_FILE, MONTAGE_WIDTH, MONTAGE_HEIGHT
from components.montage_face_images import create_montage_bytes
from util.utils import get_base_name, load_json, save_json

SAVE_DEBUG_IMAGES = False
CLIENT = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to directory for a single '
            'video or to a directory with subdirectories for each video'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('--credential-file', type=str,
                        default=AWS_CREDENTIALS_FILE)
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwrite existing output')
    return parser.parse_args()


def main(in_path, out_path, credential_file=AWS_CREDENTIALS_FILE, force=False,
         single=False):
    # Check if input is for a batch or single video
    if single:  # single
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        pbar = tqdm(total=1, desc='Identifying faces', unit='video')
        crops_path = os.path.join(in_path, OUTDIR_CROPS)
        if not os.path.exists(crops_path):
            print('No face crops available, skipping face identification')
            pbar.update()
            return
        
        identities_outpath = os.path.join(output_idr, OUTFILE_IDENTITIES)
        if force or not os.path.exists(identities_outpath):
            process_video(crops_path, identities_outpath, 100)
            pbar.update()

        return
    else:  # batch
        video_names = list(os.listdir(in_path))
        out_paths = [os.path.join(out_path, name) for name in video_names]

    for p in out_paths:
        if not os.path.isdir(p):
            os.makedirs(p)

    num_workers = min(len(video_names), 5)
    num_threads_per_worker = 100 // num_workers  # prevent throttling

    with Pool(num_workers) as workers, tqdm(
        total=len(video_names), desc='Identifying faces', unit='video'
    ) as pbar:
        for video_name, output_dir in zip(video_names, out_paths):
            crops_path = os.path.join(in_path, video_name, OUTDIR_CROPS)
            if not os.path.exists(crops_path):
                print('Skipping face identification for {}'.format(video_name))
                pbar.update()
                continue

            identities_outpath = os.path.join(output_dir, OUTFILE_IDENTITIES)
            if force or not os.path.exists(identities_outpath):
                workers.apply_async(
                    process_video,
                    args=(crops_path, identities_outpath,
                          num_threads_per_worker),
                    callback=lambda x: pbar.update())
            else:
                pbar.update()

        workers.close()
        workers.join()
        

def process_video(crops_path, identities_outpath, max_threads=100):
    assert os.path.isdir(crops_path)
    
    img_files = [img for img in sorted(os.listdir(crops_path),
                 key=lambda x: int(get_base_name(x)))]

    video_labels = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = []
        for i in range(0, len(img_files), MONTAGE_WIDTH * MONTAGE_HEIGHT):
            img_span = img_files[i:i + MONTAGE_WIDTH * MONTAGE_HEIGHT]
            futures.append(executor.submit(
                submit_images_for_labeling, crops_path, img_span
            ))

    for future in futures:
        if future.done() and not future.cancelled():
            video_labels.extend(future.result())

    save_json(video_labels, identities_outpath)


def submit_images_for_labeling(crops_path, img_files):
    client = load_client(AWS_CREDENTIALS_FILE)
    img_filepaths = [os.path.join(crops_path, f) for f in img_files]
    montage_bytes, meta = create_montage_bytes(img_filepaths)
    img_ids = [int(get_base_name(x)) for x in img_files]
   
    res = search_aws(montage_bytes, client)
    return process_labeling_results(meta['cols'], meta['block_dim'],
                                    img_ids, res)


def search_aws(img_data, client):
    # Supported image formats: JPEG, PNG, GIF, BMP.
    # Image dimensions must be at least 50 x 50.
    # Image file size must be less than 5MB.
    assert len(img_data) < 5e6, 'File too large: {}'.format(len(img_data))
    for i in range(10):
        try:
            resp = client.recognize_celebrities(Image={'Bytes': img_data})
            return resp
        except Exception as e:
            delay = 2 ** i
            tqdm.write('Error (retry in {}s): {}'.format(delay, e))
            time.sleep(delay)
    raise Exception('Too many timeouts: {}'.format(e))


def process_labeling_results(
    n_cols, block_size, img_ids, aws_response, img_draw=None
):
    half_block_size = int(block_size / 2)
    sixth_block_size = int(block_size / 6)
    width = n_cols * block_size
    height = math.ceil(len(img_ids) / n_cols) * block_size

    labels = {}
    for face in aws_response['CelebrityFaces']:
        x0 = face['Face']['BoundingBox']['Left']
        y0 = face['Face']['BoundingBox']['Top']
        x1 = x0 + face['Face']['BoundingBox']['Width']
        y1 = y0 + face['Face']['BoundingBox']['Height']

        if img_draw:
            img_draw.rectangle(
                (x0 * width, y0 * height, x1 * width, y1 * height),
                outline='red')
            text = '{}, {}'.format(
                face['Name'].encode('ascii', 'ignore'), face['MatchConfidence'])
            img_draw.text((x0 * width, y0 * height), text, fill='red')

        # Reverse map the index
        center_x = (x0 + x1) / 2 * width
        center_y = (y0 + y1) / 2 * height
        residual_x = abs(center_x % block_size - half_block_size)
        residual_y = abs(center_y % block_size - half_block_size)

        # Center must be in middle third of the image
        if (residual_x < sixth_block_size and residual_y < sixth_block_size):
            grid_x = math.floor(center_x / block_size)
            grid_y = math.floor(center_y / block_size)
            idx = grid_y * n_cols + grid_x
            if idx >= 0 and idx < len(img_ids):
                face_id = img_ids[idx]
                l1_dist = residual_x + residual_y
                face_label = (face['Name'], face['MatchConfidence'], l1_dist,
                              grid_x, grid_y)
                if face_id in labels:
                    if labels[face_id][2] > l1_dist:
                        labels[face_id] = face_label
                else:
                    labels[face_id] = face_label

    for face in aws_response['UnrecognizedFaces']:
        x0 = face['BoundingBox']['Left']
        y0 = face['BoundingBox']['Top']
        x1 = x0 + face['BoundingBox']['Width']
        y1 = y0 + face['BoundingBox']['Height']

        if img_draw:
            img_draw.rectangle(
                (x0 * width, y0 * height, x1 * width, y1 * height),
                outline='blue')

        # Reverse map the index
        center_x = (x0 + x1) / 2 * width
        center_y = (y0 + y1) / 2 * height
        residual_x = abs(center_x % block_size - half_block_size)
        residual_y = abs(center_y % block_size - half_block_size)

        # Center must be in middle of the image
        if (residual_x < half_block_size and residual_y < half_block_size):
            grid_x = math.floor(center_x / block_size)
            grid_y = math.floor(center_y / block_size)
            idx = grid_y * n_cols + grid_x
            if idx >= 0 and idx < len(img_ids):
                face_id = img_ids[idx]
                l1_dist = residual_x + residual_y
                if face_id in labels:
                    if labels[face_id][2] > l1_dist:
                        del labels[face_id]

    if img_draw:
        for label_meta in labels.values():
            name, _, _, grid_x, grid_y = label_meta
            img_draw.text((grid_x * block_size, grid_y * block_size),
                          name.encode('ascii', 'ignore'), fill='red')


    return [(k, v[0], v[1]) for k, v in labels.items()]


def load_client(credential_file):
    with open(credential_file) as f:
        f.readline()
        _, _, key_id, key_secret, _ = f.readline().split(',')

    session = boto3.session.Session()
    return session.client('rekognition', aws_access_key_id=key_id,
                          aws_secret_access_key=key_secret,
                          region_name='us-west-1')


if __name__ == '__main__':
    main(**vars(get_args()))
