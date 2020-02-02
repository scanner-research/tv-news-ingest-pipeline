import cv2
import json
import math
import os
import pickle
from PIL import Image
import shutil
from subprocess import check_call
from typing import Any

import numpy as np

from scannerpy.types import FrameType, BboxList


OUTDIR_MONTAGES = 'montages'


def crop_faces(config, frame: FrameType, bboxes: BboxList) -> Any:
    return [crop_bbox(frame, bbox) for bbox in bboxes]


def crop_bbox(img, bbox, expand=0.1):
    y1 = max(bbox.y1 - expand, 0)
    y2 = min(bbox.y2 + expand, 1)
    x1 = max(bbox.x1 - expand, 0)
    x2 = min(bbox.x2 + expand, 1)
    [h, w] = img.shape[:2]
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def get_face_crops_results(face_crops):
    result = []  # [(<face_id>, <crop>)]
    for crops in face_crops:
        faces_in_frame = [
            (face_id, img) for face_id, img in enumerate(crops, len(result))
        ]

        result += faces_in_frame

    return result


def save_montage_results(video_name, output_path, metadata, face_crops_file):
    with open(face_crops_file, 'rb') as in_f:
        face_crops = pickle.load(in_f)
        os.remove(face_crops_file)

    results = montage_faces(metadata, face_crops)

    tmp_dir = os.path.join('/tmp', video_name, OUTDIR_MONTAGES)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for i, result in enumerate(results):
        img, meta = result
        Image.fromarray(img).save('{}/{}.png'.format(tmp_dir, i),
                                  optimize=True)

        with open('{}/{}.json'.format(tmp_dir, i), 'w') as f:
            json.dump(meta, f)

        if output_path.startswith('gs://'):
            cmd = ['gsutil', '-m', 'cp', '-r', tmp_dir, output_path]
            print('Copying:', cmd)
            check_call(cmd)
            shutil.rmtree(tmp_dir)
        else:
            if os.path.isdir(os.path.join(output_path, OUTDIR_MONTAGES)):
                shutil.rmtree(os.path.join(output_path, OUTDIR_MONTAGES))
                
            shutil.move(tmp_dir, output_path)


IMG_SIZE = 200
BLOCK_SIZE = 250
HALF_BLOCK_SIZE = BLOCK_SIZE / 2
PADDING = int((BLOCK_SIZE - IMG_SIZE) / 2)
assert IMG_SIZE + 2 * PADDING == BLOCK_SIZE, \
    'Bad padding: {}'.format(IMG_SIZE + 2 * PADDING)

BLANK_IMAGE = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), dtype=np.uint8)

NUM_ROWS = 6
NUM_COLS = 10


def montage_faces(face_info, face_crops):
    stacked_rows = []
    for i in range(0, len(face_crops), NUM_COLS):
        row_imgs = [
            convert_image(im)
            for _, im in face_crops[i:i + NUM_COLS]
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
    for i in range(0, len(face_crops), num_per_stack):
        ids = [x[0] for x in face_crops[i:i + num_per_stack]]
        stacked_img_meta.append({
            'rows': math.ceil(len(ids) / NUM_COLS),
            'cols': NUM_COLS,
            'block_dim': BLOCK_SIZE,
            'content': ids
        })

    return zip(stacked_imgs, stacked_img_meta)


def convert_image(im):
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, dsize=(IMG_SIZE, IMG_SIZE))
    im = cv2.copyMakeBorder(
        im, PADDING, PADDING, PADDING, PADDING,
        cv2.BORDER_CONSTANT, value=0)
    return im

