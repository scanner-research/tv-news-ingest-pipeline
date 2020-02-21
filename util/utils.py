#!/usr/bin/env python3

"""
File: utils.py
--------------
Contains general functions used across multiple parts of the pipeline.

"""

import json
import os
from pathlib import Path

from util.consts import LOCAL_TOML


def save_json(data, fname: str):
    """
    Saves a given object in JSON format to the specified file.

    Args:
        data: the object to save.
        fname: the name of the file to save to.

    """

    with open(fname, 'w') as f:
        json.dump(data, f)


def load_json(fname: str):
    """
    Loads an object from a JSON file.

    Args:
        fname: the name of the file to load from.

    Returns:
        the JSON object.

    """

    with open(fname) as f:
        return json.load(f)


def get_base_name(path: str) -> str:
    """
    Gets the name of the file without the extension from a path.

    Args:
        path: the path from which to extract the base name.

    Returns:
        the base file name.

    """

    return os.path.splitext(Path(path).name)[0]


def json_is_valid(path: str) -> bool:
    try:
        json.load(open(path, 'r'))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        return False

    return True


def get_batch_io_paths(in_path, out_path):
    with open(in_path, 'r') as f:
        in_paths = [l.strip() for l in f if l.strip()]
        out_paths = [os.path.join(out_path, get_base_name(p)) for p in in_paths]

    return in_paths, out_paths


def update_pbar(bar):
    def update(x):
        bar.update()

    return update


########### Scanner ##########


def init_scanner_config():
    scanner_config_dir = '/root/.scanner'
    if not os.path.exists(scanner_config_dir):
        os.makedirs(scanner_config_dir)

    with open(os.path.join(scanner_config_dir, 'config.toml'), 'w') as f:
        f.write(LOCAL_TOML)


def remove_unfinished_outputs(cl, video_names, all_outputs, del_fn, clean=False):
    for collection in zip(video_names, *all_outputs):
        video_name = collection[0]
        outputs = collection[1:]
        if clean or not all(out is None or out.committed() for out in outputs):
            del_fn(cl, list(filter(lambda x: x is not None, outputs)))
        else:
            print('Using cached results for', video_name)

