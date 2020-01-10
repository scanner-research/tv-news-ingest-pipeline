#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier


GENDER_TRAIN_X_FILE = 'gender_model/train_X.npy'
GENDER_TRAIN_Y_FILE = 'gender_model/train_y.npy'
KNN_K = 7

train_X = np.load(GENDER_TRAIN_X_FILE)
train_y = np.load(GENDER_TRAIN_Y_FILE)
clf = KNeighborsClassifier(KNN_K)
clf.fit(train_X, train_y)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str,
                        help='Directory or file containing face embeddings')
    parser.add_argument('out_path', type=str,
                        help='Directory or file to output results to')
    return parser.parse_args()


def load_json(fname):
    with open(fname) as f:
        return json.load(f)


def predict_gender(x):
    return 'F' if clf.predict([x]) == 1 else 'M'


def predict_gender_score(x):
    # FIXME: this was not tested. Need to check if this is sane
    return max(clf.predict_proba([x])[0])


def process_single(in_file, out_file):
    # Load the detected faces and embeddings and run the classifier
    print('Classifying gender:', in_file)
    result = []
    for frame_num, faces in load_json(in_file):
        face_emb = f['emb']
        result.append((
            frame_num, [{
                'bbox': f['bbox'],
                'gender': predict_gender(face_emb),
                'gender_score': predict_gender_score(face_emb)
            } for f in faces]
        ))
    with open(out_file, 'w') as f:
        json.dump(result, f)
    print('Wrote labels:', out_file)


def get_video_name(x):
    return os.splitext(x.split('/')[-1])[0]


def main(in_path: str, out_path: str):
    if os.path.isdir(in_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for video_in_file in tqdm(os.listdir(in_path)):
            video_in_path = os.path.join(in_path, video_in_file)
            video_out_file = get_video_name(video_in_file) + '.json'
            video_out_path = os.path.join(out_path, video_out_file)
            process_single(video_in_path, video_out_path)
    else:
        # Single video case
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path,
                                    get_video_name(in_path) + '.json')
        process_single(in_path, out_path)


if __name__ == '__main__':
    main(**vars(get_args()))
