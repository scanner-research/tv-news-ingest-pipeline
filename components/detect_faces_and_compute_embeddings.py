from concurrent.futures import ThreadPoolExecutor
from functools import partial
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from PIL import Image
from threading import Thread

import cv2
from tqdm import tqdm
from tqdm import trange

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import components.models.facenet as facenet
import components.models.mtcnn as mtcnn
from util import config
from util.consts import (
    FILE_BBOXES,
    FILE_EMBEDS,
    FILE_METADATA,
    DIR_CROPS,
    SCANNER_COMPONENT_OUTPUTS
)
from util.utils import json_is_valid, save_json

MODELS_DIR = 'components/data'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help=('path to mp4 or to a text file '
                                         'containing video filepaths'))
    parser.add_argument('out_path', help='path to output directory')
    parser.add_argument('-i', '--init-run', action='store_true',
                        help='running on videos for the first time')
    parser.add_argument('-f', '--force-rerun', action='store_true',
                        help='force rerun for all videos')
    parser.add_argument('--interval', type=int, default=config.STRIDE,
                        help='interval length in seconds')
    parser.add_argument('-d', '--disable', nargs='+', choices=NAMED_COMPONENTS,
                        help='list of named components to disable')
    return parser.parse_args()


def main(in_path, out_path, init_run=False, force=False, interval=config.STRIDE,
         disable=None):
    if in_path.endswith('.mp4'):
        video_paths = [Path(in_path)]
    else:
        video_paths = [Path(l.strip()) for l in open(in_path, 'r') if l.strip()]

    out_paths = [Path(out_path)/p.stem for p in video_paths]
    process_videos(video_paths, out_paths, init_run, force, interval, disable)


def process_videos(video_paths, out_paths, init_run=False, force=False,
                   interval=config.STRIDE, disable=None):
    assert len(video_paths) == len(out_paths), ('Mismatch between video and '
                                                'output paths')

    if disable is None:
        disable = []

    # Don't reingest videos with existing outputs
    if not init_run and not force:
        for i in range(len(video_paths) - 1, -1, -1):
            if (('face_detection' in disable
                or json_is_valid(os.path.join(out_paths[i], FILE_BBOXES)))
                and ('face_embeddings' in disable
                    or json_is_valid(os.path.join(out_paths[i], FILE_EMBEDS)))
                and json_is_valid(os.path.join(out_paths[i], FILE_METADATA))
                and ('face_crops' in disable
                    or os.path.isdir(os.path.join(out_paths[i], DIR_CROPS)))
            ):
                video_paths.pop(i)
                out_paths.pop(i)

    video_names = [vid.stem for vid in video_paths]
    if not video_names:
        print('All videos have existing outputs.')
        return

    face_embedder, face_detector = load_models(MODELS_DIR)

    all_metadata = [
        get_video_metadata(video_names[i], video_paths[i])
        for i in trange(len(video_names), desc='Collecting metadata', unit='video')
    ]

    all_bboxes = [[] for _ in range(len(video_names))]
    all_crops = [[] for _ in range(len(video_names))]
    all_embeddings = [[] for _ in range(len(video_names))]

    n_threads = os.cpu_count() if os.cpu_count() else 1

    total_sec = int(sum(math.floor(m['frames'] / m['fps'] / interval) for m in all_metadata))

    pbar = tqdm(total=total_sec, desc='Processing videos', unit='frame')
    for vid_id in range(len(video_names)):
        path = video_paths[vid_id]
        meta = all_metadata[vid_id]

        pbar.set_description('Processing videos: {}'.format(meta['name']))
        thread_bboxes = [[] for _ in range(n_threads)]
        thread_crops = [[] for _ in range(n_threads)]
        thread_embeddings = [[] for _ in range(n_threads)]

        threads = [Thread(
            target=thread_task,
            args=(str(path), meta, interval, n_threads, i, thread_bboxes, thread_crops,
                  thread_embeddings, face_embedder, face_detector, pbar)
        ) for i in range(n_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        for i in range(n_threads):
            all_bboxes[vid_id] += thread_bboxes[i]
            all_crops[vid_id] += thread_crops[i]
            all_embeddings[vid_id] += thread_embeddings[i]

    pbar.close()

    # Async callback function
    def callback_fn(data=None, path=None, save_fn=None, pbar=None):
        if save_fn:
            save_fn(data, path)
        if pbar:
            pbar.update(1)

    tmp_dir = '/tmp/face_crops'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    with mp.Pool() as workers, tqdm(
        total=len(video_names) * (len(SCANNER_COMPONENT_OUTPUTS) - len(disable)),
        desc='Collecting output', unit='output'
    ) as pbar:
        for (video_name, out_path, meta, output_faces, output_embeddings,
             output_crops) in zip(
                video_names, out_paths, all_metadata, all_bboxes,
                all_embeddings, all_crops
        ):
            target_sec = math.floor(m['frames'] / m['fps'] / interval)
            if any(len(x) != target_sec for x in [output_faces, output_embeddings, output_crops]):
                # Error decoding video
                print('There was an error decoding video \'{}\'. Skipping.'.format(meta['name']))
                continue

            metadata_outpath = out_path/FILE_METADATA
            save_json(meta, str(metadata_outpath))
            pbar.update(1)

            bbox_outpath = out_path/FILE_BBOXES
            workers.apply_async(
                handle_face_bboxes_results,
                args=(output_faces, meta['fps'] * interval, str(bbox_outpath)),
                callback=partial(callback_fn, pbar=pbar)
            )

            embed_outpath = out_path/FILE_EMBEDS
            workers.apply_async(
                handle_face_embeddings_results,
                args=(output_embeddings, str(embed_outpath)),
                callback=partial(callback_fn, pbar=pbar)
            )

            tmp_path = os.path.join(tmp_dir, '{}.pkl'.format(video_name))
            with open(tmp_path, 'wb') as f:
                pickle.dump(output_crops, f)
            crops_outpath = out_path/DIR_CROPS
            workers.apply_async(
                handle_face_crops_results,
                args=(tmp_path, str(crops_outpath)),
                callback=partial(callback_fn, pbar=pbar)
            )

        workers.close()
        workers.join()


def load_models(models_dir):
    face_embedder = facenet.FaceNetEmbed(os.path.join(models_dir, 'facenet'))
    face_detector = mtcnn.MTCNN(os.path.join(models_dir, 'align'))
    return face_embedder, face_detector


def get_video_metadata(video_name: str, video_path: Path):
    video = cv2.VideoCapture(str(video_path))
    return {
        'name': video_name,
        'fps': video.get(cv2.CAP_PROP_FPS),
        'frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }


BATCH_SIZE = 16
def thread_task(in_path, metadata, interval, n_threads, thread_id,
                thread_bboxes, thread_crops, thread_embeddings, face_embedder,
                face_detector, pbar):

    video = cv2.VideoCapture(in_path)

    if not video.isOpened():
        print('Error opening video file.')
        return

    n_sec = math.floor(metadata['frames'] / metadata['fps'] / interval)
    chunk_size_sec = math.floor(n_sec / n_threads)
    start_sec = chunk_size_sec * thread_id
    if thread_id == n_threads - 1:
        chunk_size_sec += n_sec % n_threads
    end_sec = start_sec + chunk_size_sec

    for sec in range(start_sec, end_sec, BATCH_SIZE):
        frames = []

        for i in range(sec, min(sec + BATCH_SIZE, end_sec)):
            frame_num = math.ceil(i * metadata['fps'] * interval)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = video.read()
            if not success:
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        detected_faces = face_detector.face_detect(frames)
        dilated_bboxes = [dilate_bboxes(x) if x else [] for x in detected_faces]
        crops = [[crop_bbox(frame, bb) for bb in x] if x else [] for x in dilated_bboxes]
        embeddings = [face_embedder.embed(c) if c else [] for c in crops]

        thread_bboxes[thread_id].extend(detected_faces)
        thread_crops[thread_id].extend(crops)
        thread_embeddings[thread_id].extend(embeddings)
        pbar.update(len(frames))


DILATE_AMOUNT = 1.05
def dilate_bboxes(detected_faces):
    return [{
        'x1': bbox['x1'] * (2 - DILATE_AMOUNT),
        'x2': bbox['x2'] * DILATE_AMOUNT,
        'y1': bbox['y1'] * (2 - DILATE_AMOUNT),
        'y2': bbox['y2'] * DILATE_AMOUNT
    } for bbox in detected_faces]


def crop_bbox(img, bbox, expand=0.1):
    y1 = max(bbox['y1'] - expand, 0)
    y2 = min(bbox['y2'] + expand, 1)
    x1 = max(bbox['x1'] - expand, 0)
    x2 = min(bbox['x2'] + expand, 1)
    h, w = img.shape[:2]
    cropped = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w), :]

    # Crop largest square
    if cropped.shape[0] > cropped.shape[1]:
        target_height = cropped.shape[1]
        diff = target_height // 2
        center = cropped.shape[0] // 2
        square = cropped[center - diff:center + (target_height - diff), :, :]
    else:
        target_width = cropped.shape[0]
        diff = target_width // 2
        center = cropped.shape[1] // 2
        square = cropped[:, center - diff:center + (target_width - diff), :]

    return square


def handle_face_bboxes_results(detected_faces, stride, outpath: str):
    result = []  # [(<face_id>, {'frame_num': <n>, 'bbox': <bbox_dict>}), ...]
    for i, faces in enumerate(detected_faces):
        faces_in_frame = [
            (face_id, {'frame_num': math.ceil(i * stride), 'bbox': face})
            for face_id, face in enumerate(faces, len(result))
        ]

        result += faces_in_frame

    save_json(result, outpath)


def handle_face_embeddings_results(face_embeddings, outpath):
    result = []  # [(<face_id>, <embedding>), ...]
    for embeddings in face_embeddings:
        faces_in_frame = [
            (face_id, [float(x) for x in embed])
            for face_id, embed in enumerate(embeddings, len(result))
        ]

        result += faces_in_frame

    save_json(result, outpath)


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


def save_face_crops(face_crops, out_dirpath: str):
    if not os.path.isdir(out_dirpath):
        os.makedirs(out_dirpath)

    def save_img(img, fp):
        Image.fromarray(img).save(fp, optimize=True)

    with ThreadPoolExecutor() as executor:
        for face_id, img in face_crops:
            img_filepath = os.path.join(out_dirpath, str(face_id) + '.png')
            executor.submit(save_img, img, img_filepath)
