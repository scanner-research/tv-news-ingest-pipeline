from collections import Counter
import json
from multiprocessing import Pool
import os

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from util.consts import (FILE_IDENTITIES, FILE_EMBEDS,
                         FILE_IDENTITIES_PROP)
from util.utils import load_json, save_json

PROB_THRESH = 0.9
MIN_LABEL_THRESH = 5
L2_THRESH = 0.7


def main(in_path, out_path, force=False):
    video_names = list(os.listdir(in_path))
    out_paths = [os.path.join(out_path, name) for name in video_names]

    for p in out_paths:
        if not os.path.isdir(p):
            os.makedirs(p)

    with Pool() as workers, tqdm(
        total=len(video_names), desc='Propogating identities', unit='video'
    ) as pbar:
        for video_name, output_dir in zip(video_names, out_paths):
            identities_path = os.path.join(in_path, video_name,
                                           FILE_IDENTITIES)
            embeds_path = os.path.join(in_path, video_name, FILE_EMBEDS)
            propogated_ids_outpath = os.path.join(output_dir,
                                                  FILE_IDENTITIES_PROP)
            if force or not os.path.exists(propogated_ids_outpath):
                workers.apply_async(
                    process_single,
                    args=(identities_path, embeds_path, propogated_ids_outpath),
                    callback=lambda x: pbar.update())
            else:
                pbar.update()

        workers.close()
        workers.join()


def process_single(identities_path, embeds_path, propogated_ids_outpath):
    identities = load_json(identities_path)
    embeddings = load_json(embeds_path)
    
    labeled_face_ids = set(x[0] for x in identities)
    counts = Counter(x[1] for x in identities)
    names_to_propogate = set(x for x in counts if counts[x] > MIN_LABEL_THRESH)
    face_id_to_identity = {
        x[0]: x[1] for x in identities if x[1] in names_to_propogate
    }
    face_ids_to_propogate = set(face_id_to_identity.keys()) 
    name_to_face_ids = {name: [] for name in names_to_propogate}
    for face_id, name in face_id_to_identity.items():
        name_to_face_ids[name].append(face_id)

    face_id_to_embed = {
        x[0]: x[1] for x in embeddings if x[0] not in labeled_face_ids
    }
    face_id_to_embed_prop = {
        x[0]: x[1] for x in embeddings if x[0] in face_ids_to_propogate
    }

    unlabeled_array = np.array([x for x in face_id_to_embed.values()])
    best_so_far = [(None, 0)] * unlabeled_array.shape[0]
    for name, ids in name_to_face_ids.items():
        labeled_array = np.array([face_id_to_embed_prop[i] for i in ids])
        dists = euclidean_distances(unlabeled_array, labeled_array)
        dists = (dists < L2_THRESH).astype(int)
        votes = dists.sum(axis=1)
        for i in range(votes.shape[0]):
            if votes[i] > best_so_far[i][1]:
                best_so_far[i] = (name, votes[i])

        for i, face_id in enumerate(face_id_to_embed.keys()):
            if best_so_far[i][0] is not None:
                identities.append((face_id, best_so_far[i][0], 50.0))

    save_json(identities, propogated_ids_outpath)

