#!/usr/bin/env python3

import argparse
from multiprocessing import Pool
import os

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

from utils import save_json, load_json, get_base_name
from consts import OUTFILE_EMBEDS, OUTFILE_GENDERS

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
                        help='file containing face embeddings')
    parser.add_argument('out_path', type=str,
                        help='path to output directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwrite existing output')
    return parser.parse_args()


def main(in_path: str, out_path: str, force: bool = False):
    # Check whether input is single or batch
    if OUTFILE_EMBEDS in list(os.listdir(in_path)):
        return  # TODO: implement single
    else:
        video_names = list(os.listdir(in_path))
        out_paths = [os.path.join(out_path, name) for name in video_names]
    
    for p in out_paths:
        if not os.path.isdir(p):
            os.makedirs(p)
    
    with Pool() as workers, tqdm(
        total=len(video_names), desc='Classifying genders', unit='video'
    ) as pbar:
        for video_name, output_dir in zip(video_names, out_paths):
            embeds_path = os.path.join(in_path, video_name, OUTFILE_EMBEDS)
            genders_outpath = os.path.join(output_dir, OUTFILE_GENDERS)
            if force or not os.path.exists(genders_outpath):
                workers.apply_async(
                    process_single,
                    args=(embeds_path, genders_outpath),
                    callback=lambda x: pbar.update())
            else:
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
