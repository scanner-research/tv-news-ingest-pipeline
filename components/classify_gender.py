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
        │   └── genders.json
        └── video2
            └── genders.json

"""

import argparse
from multiprocessing import Pool
import os

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

from util.utils import save_json, load_json, get_base_name
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
    out_paths = [os.path.join(out_path, name) for name in video_names]

    for p in out_paths:
        os.makedirs(p, exist_ok=True)

    with Pool() as workers, tqdm(
        total=len(video_names), desc='Classifying genders', unit='video'
    ) as pbar:
        for video_name, output_dir in zip(video_names, out_paths):
            embeds_path = os.path.join(in_path, video_name, FILE_EMBEDS)
            genders_outpath = os.path.join(output_dir, FILE_GENDERS)
            if force or not os.path.exists(genders_outpath):
                workers.apply_async(
                    process_single,
                    args=(embeds_path, genders_outpath),
                    callback=lambda x: pbar.update())
            else:
                tqdm.write("Skipping classifying gender for video '{}': '{}' "
                      "already exists!".format(video_name, FILE_GENDERS))
                pbar.update()

        workers.close()
        workers.join()


def process_single(in_file, out_file):
    # Load the detected faces and embeddings and run the classifier
    result = [(face_id, predict_gender(embed), predict_gender_score(embed))
              for face_id, embed in load_json(in_file)]

    save_json(result, out_file)


def predict_gender(x):
    return 'F' if clf.predict([x]) == 1 else 'M'


def predict_gender_score(x):
    # FIXME: this was not tested. Need to check if this is sane
    return max(clf.predict_proba([x])[0])


if __name__ == '__main__':
    main(**vars(get_args()))
