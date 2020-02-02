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

    <input_dir>/
        - <video1>/
            - identities.json
        - <video2>/
            - identities.json
        ...

where there is one JSON file per video containing a list of (face_id, identity)
tuples.

"""

import argparse
import os
import json
import glob
import boto3
import math
import shutil
import time
from subprocess import check_call
from PIL import Image, ImageDraw
from tqdm import tqdm
from multiprocessing import Pool

from utils import load_json, save_json
from consts import OUTFILE_IDENTITIES

SAVE_DEBUG_IMAGES = False
CREDENTIAL_FILE = None
CLIENT = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str,
        help='path to directory containing montage folders')
    parser.add_argument('output_dir', type=str, help='path to output directory')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--debug-dir', type=str, default='debug')

    parser.add_argument('--credential-file', type=str,
                        default='aws-credentials.csv')

    # Save debugging images with bbox overlays
    parser.add_argument('--save-img', action='store_true')

    # Max parallel requests (100+ to one region causes throttling)
    parser.add_argument('-n', '--parallelism', type=int, default=64)
    return parser.parse_args()


def main(input_dir, output_dir, debug_dir, credential_file, limit, save_img,
         parallelism):

    global SAVE_DEBUG_IMAGES, CREDENTIAL_FILE
    SAVE_DEBUG_IMAGES = save_img
    CREDENTIAL_FILE = credential_file

    video_names = list(os.listdir(input_dir))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with Pool(parallelism) as workers, \
            tqdm(total=len(video_names), desc='Identifying faces') as pbar:
        def update_pbar(x):
            pbar.update(1)

        results = []

        for video_name in video_names:
            in_path = os.path.join(input_dir, video_name, 'montages')
            out_path = os.path.join(input_dir, video_name, OUTFILE_IDENTITIES)
            if not os.path.exists(out_path):
                if debug_dir:
                    debug_path = os.path.join(debug_dir, video_name)
                else:
                    debug_path = None

                result = workers.apply_async(
                    process_video,
                    args=(video_name, in_path, out_path, debug_path),
                    callback=update_pbar)
                results.append(result)
            else:
                update_pbar(None)

        for result in results:
            result.get()
        workers.close()
        workers.join()


def process_video(video_name, in_path, out_path, debug_dir):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    remove_on_done = False
    if in_path.startswith('gs://'):
        try:
            input_dir = load_from_gcs(in_path, video_name)
        except Exception as e:
            print('Failed:', video_name, '-', e)
            return
        remove_on_done = True
    else:
        assert os.path.isdir(in_path)
        input_dir = in_path

    img_files = [img for img in sorted(os.listdir(input_dir))
                 if img.endswith('.png')]
    # print('Processing video:', video_name, '({} chunks)'.format(len(img_files)))
    video_labels = []
    for img_file in sorted(img_files, key=lambda x: int(x.split('.')[0])):
        img_prefix = os.path.splitext(img_file)[0]
        img_path = os.path.join(input_dir, img_file)
        meta_path = os.path.join(input_dir, img_prefix + '.json')
        if os.path.exists(meta_path):
            if debug_dir:
                debug_prefix = os.path.join(debug_dir, img_prefix)
            else:
                debug_prefix = None
            group_labels = submit_image_for_labeling(
                img_path, meta_path, debug_prefix)
            video_labels.extend(group_labels)
        else:
            print('Missing metdata for:', img_path)

    save_json(video_labels, out_path)

    if remove_on_done:
        print('Removing:', input_dir)
        shutil.rmtree(input_dir)


def read_img(img):
    with open(img, 'rb') as f:
        return f.read()


def load_client(credential_file):
    with open(credential_file) as f:
        f.readline()
        _, _, key_id, key_secret, _ = f.readline().split(',')

    return boto3.client(
        'rekognition', aws_access_key_id=key_id,
        aws_secret_access_key=key_secret,
        region_name='us-west-1')


def search_aws(img_data):
    global CLIENT
    if CLIENT is None:
        CLIENT = load_client(CREDENTIAL_FILE)
    # Supported image formats: JPEG, PNG, GIF, BMP.
    # Image dimensions must be at least 50 x 50.
    # Image file size must be less than 5MB.
    assert len(img_data) < 5e6, 'File too large: {}'.format(len(img_data))
    for i in range(10):
        try:
            resp = CLIENT.recognize_celebrities(Image={'Bytes': img_data})
            return resp
        except Exception as e:
            delay = 2 ** i
            print('Error (retry in {}s):'.format(delay), e)
            time.sleep(delay)
    raise Exception('Too many timeouts: {}'.format(e))


def split_list(L, n):
    assert type(L) is list, "L is not a list"
    for i in range(0, len(L), n):
        yield L[i:i+n]


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


def submit_image_for_labeling(img_path, meta_path, debug_prefix):
    stacked_img_data = read_img(img_path)
    img_meta = load_json(meta_path)
    cols = img_meta['cols']
    block_size = img_meta['block_dim']
    img_ids = img_meta['content']

    if debug_prefix:
        debug_path = debug_prefix + '.response.json'
        if os.path.exists(debug_path):
            with open(debug_path) as f:
                resp = json.load(f)['response']
        else:
            resp = search_aws(stacked_img_data)
            with open(debug_path, 'w') as f:
                json.dump({
                    'config': {
                        'cols': cols,
                        'image_size': block_size
                    },
                    'ids': img_ids,
                    'response': resp
                }, f)
    else:
        resp = search_aws(stacked_img_data)

    if debug_prefix and SAVE_DEBUG_IMAGES:
        stacked_img = Image.open(img_path)
        stacked_img_draw = ImageDraw.Draw(stacked_img)
    else:
        stacked_img = None
        stacked_img_draw = None

    results = process_labeling_results(cols, block_size, img_ids, resp,
                                       stacked_img_draw)

    if stacked_img:
        stacked_img.save(debug_prefix + '.annotated.png', optimize=True)
    return results


def load_from_gcs(gcs_path, video_name):
    tmp_path = '/tmp/{}'.format(video_name)
    check_call(['gsutil', '-q', '-m', 'cp', '-r', gcs_path, '/tmp/'])
    return tmp_path


if __name__ == '__main__':
    main(**vars(get_args()))
