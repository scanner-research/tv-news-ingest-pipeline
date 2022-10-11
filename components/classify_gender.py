#!/usr/bin/env python3

"""
File: classify_gender.py
------------------------
Script for classifying gender using face embeddings.

Example
-------

    in_path:  output_dir
    out_path: output_dir

    where 'output_dir' contains video output subdirectories

    outputs

        output_dir/
        ├── video1
        │   └── genders.json
        └── video2
            └── genders.json

"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append('.')

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from util.utils import save_json, load_json, format_hmmss
from util.consts import FILE_EMBEDS, FILE_GENDERS

GENDER_TRAIN_X_FILE = 'components/gender_model/train_X.npy'
GENDER_TRAIN_Y_FILE = 'components/gender_model/train_y.npy'
KNN_K = 7

train_X = np.load(GENDER_TRAIN_X_FILE)
train_y = np.load(GENDER_TRAIN_Y_FILE)
clf = KNeighborsClassifier(KNN_K)
clf.fit(train_X, train_y)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str,
                        help='path to directory of video outputs')
    parser.add_argument('out_path', type=str,
                        help='path to output directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwrite existing output')
    return parser.parse_args()


def main(in_path, out_path, force=False):
    video_names = list(os.listdir(in_path))
    out_paths = [Path(out_path)/name for name in video_names]
    in_path = Path(in_path)

    for p in out_paths:
        p.mkdir(parents=True, exist_ok=True)

    # Prune videos that should not be run
    msg = []
    for i in range(len(video_names) - 1, -1, -1):
        embeds_path = in_path/video_names[i]/FILE_EMBEDS
        if not embeds_path.exists():
            msg.append("Skipping classifying gender for video '{}': "
                       "'{}' does not exist.".format(video_names[i], embeds_path))
            video_names.pop(i)
            out_paths.pop(i)
            continue

        genders_outpath = out_paths[i]/FILE_GENDERS
        if not force and genders_outpath.exists():
            msg.append("Skipping classifying gender for video '{}': '{}' "
                       "already exists.".format(video_names[i], genders_outpath))
            video_names.pop(i)
            out_paths.pop(i)

    if not video_names:
        print('All videos have existing gender classifications.')
        sys.stdout.flush()
        return

    if msg:
        print(*msg, sep='\n')
        sys.stdout.flush()

    start_time = time.time()
    for i in range(len(video_names)):
        print('Classifying gender ({:0.1f} % done, {} elapsed)'.format(
            i / len(video_names) * 100, format_hmmss(time.time() - start_time)))
        sys.stdout.flush()
        embeds_path = in_path/video_names[i]/FILE_EMBEDS
        genders_outpath = out_paths[i]/FILE_GENDERS
        process_single(str(embeds_path), str(genders_outpath))

    print('Done classifying gender. {} elapsed'.format(
        format_hmmss(time.time() - start_time)))
    sys.stdout.flush()


# def process_single(in_file, out_file):
#     # Load the detected faces and embeddings and run the classifier
#     result = [(face_id, *predict_gender_and_score(embed))
#               for face_id, embed in load_json(in_file)]
#     save_json(result, out_file)


# def predict_gender_and_score(x):
#     p = clf.predict_proba([x])[0]
#     return 'F' if p[1] > p[0] else 'M', max(p)


def process_single(in_file, out_file):
    face_ids, embs = zip(*load_json(in_file))
    pr = clf.predict_proba(embs)
    score = np.max(pr, axis=1)

    assert pr.shape == (len(face_ids), 2)
    assert score.shape == (len(face_ids),)
    result = [(face_id, 'F' if pr[i, 1] > pr[i, 0] else 'M', score[i].item()) 
              for i, face_id in enumerate(face_ids)]
    save_json(result, out_file)


if __name__ == '__main__':
    main(**vars(get_args()))
