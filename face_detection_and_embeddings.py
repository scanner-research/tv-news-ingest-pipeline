"""
File: face_detection_and_embeddings.py
--------------------------------------
This module contains functions related to detecting faces, extracting bounding 
boxes and crops, and computing face_embeddings.

Used in scanner_component.py.

"""

from concurrent.futures import ThreadPoolExecutor
import os
import pickle
from PIL import Image
from typing import Any

import scannerpy as sp
from scannerpy import protobufs
from scannerpy.storage import NullElement
from scannerpy.types import BboxList, FrameType


DILATE_AMOUNT = 1.05
@sp.register_python_op(name='DilateBboxes')
def dilate_bboxes(config, bboxes: BboxList) -> BboxList:
    """
    Scanner operation for changing the size of bounding boxes.

    Args:
        config: the scanner config object.
        bboxes: a list bounding boxes found (passed in per frame)

    Returns:
        a list of the modified bounding boxes.

    """

    return [
        protobufs.BoundingBox(
            x1=bb.x1 * (2. - DILATE_AMOUNT),
            x2=bb.x2 * DILATE_AMOUNT,
            y1=bb.y1 * (2. - DILATE_AMOUNT),
            y2=bb.y2 * DILATE_AMOUNT,
            score=bb.score
        ) for bb in bboxes
    ]


@sp.register_python_op(name='CropFaces')
def crop_faces(config, frame: FrameType, bboxes: BboxList) -> Any:
    return [crop_bbox(frame, bbox) for bbox in bboxes]


def get_face_bboxes_results(detected_faces, stride: int):
    assert isinstance(stride, int)

    result = []  # [(<face_id>, {'frame_num': <n>, 'bbox': <bbox_dict>}), ...]
    frame_num = 0
    for faces in detected_faces:
        faces_in_frame = [
            (face_id, {'frame_num': frame_num, 'bbox': bbox_to_dict(face)})
            for face_id, face in enumerate(faces, len(result))
        ]

        result += faces_in_frame
        frame_num += stride

    return result


def get_face_embeddings_results(face_embeddings):
    result = []  # [(<face_id>, <embedding>), ...]
    for embeddings in face_embeddings:
        if isinstance(embeddings, NullElement):
            continue

        faces_in_frame = [
            (face_id, embed.tolist())
            for face_id, embed in enumerate(embeddings, len(result))
        ]

        result += faces_in_frame

    return result


def handle_face_crops_results(face_crops_path, out_dirpath):
    # Results are too large to transmit
    results = get_face_crops_results(face_crops_path)
    save_face_crops(results, out_dirpath)


def get_face_crops_results(face_crops_path):
    with open(face_crops_path, 'rb') as f:
        face_crops = pickle.load(f)
    os.remove(face_crops_path)

    result = []  # [(<face_id>, <crop>)]
    for crops in face_crops:
        faces_in_frame = [
            (face_id, img) for face_id, img in enumerate(crops, len(result))
        ]

        result += faces_in_frame

    return result


def bbox_to_dict(b):
    return {'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2, 'score': b.score}


def crop_bbox(img, bbox, expand=0.1):
    y1 = max(bbox.y1 - expand, 0)
    y2 = min(bbox.y2 + expand, 1)
    x1 = max(bbox.x1 - expand, 0)
    x2 = min(bbox.x2 + expand, 1)
    [h, w] = img.shape[:2]
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def save_face_crops(face_crops, out_dirpath: str):
    if not os.path.isdir(out_dirpath):
        os.makedirs(out_dirpath)

    def save_img(img, fp):
        Image.fromarray(img).save(fp, optimize=True)

    with ThreadPoolExecutor() as executor:
        for face_id, img in face_crops:
            img_filepath = os.path.join(out_dirpath, str(face_id) + '.png')
            executor.submit(save_img, img, img_filepath)
