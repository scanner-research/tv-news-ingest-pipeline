#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

from utils import save_json, load_json, get_base_name
from consts import OUTFILE_GENDERS

GENDER_TRAIN_X_FILE = 'gender_model/train_X.npy'
GENDER_TRAIN_Y_FILE = 'gender_model/train_y.npy'
KNN_K = 7

train_X = np.load(GENDER_TRAIN_X_FILE)
train_y = np.load(GENDER_TRAIN_Y_FILE)
clf = KNeighborsClassifier(KNN_K)
clf.fit(train_X, train_y)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str,
                        help='file containing face embeddings')
    parser.add_argument('output_path', type=str,
                        help='file to output results to')
    return parser.parse_args()


def main(input_path: str, output_path: str):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    process_single(input_path, output_path)

    if os.path.isdir(in_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        input_files = glob.glob(os.path.join(in_path, '*_embeddings.json'))
        for in_file in tqdm(input_files):
            video_name = get_base_name(in_file)[:-len('_embeddings')]
            out_file = os.path.join(out_path, video_name + '.json')
            process_single(in_file, out_file)

    #else:
    #    # Single video case
    #    if os.path.isdir(out_path):
    #        out_path = os.path.join(out_path,
    #                                get_video_name(in_path) + '.json')
    #    process_single(in_path, out_path)


def predict_gender(x):
    return 'F' if clf.predict([x]) == 1 else 'M'


def predict_gender_score(x):
    # FIXME: this was not tested. Need to check if this is sane
    return max(clf.predict_proba([x])[0])


def process_single(in_file, out_file):
    # Load the detected faces and embeddings and run the classifier
    result = [
        (face_id, predict_gender(embed), predict_gender_score(embed))
        for face_id, embed in load_json(in_file)
    ]

    save_json(result, out_file)


if __name__ == '__main__':
    main(**vars(get_args()))
