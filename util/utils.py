"""
File: utils.py
--------------
Contains general functions used across multiple parts of the pipeline.

"""

import errno
import fcntl
import json
import os
from pathlib import Path
from typing import Callable, List


def save_json(data, fname: str) -> None:
    """
    Saves a given object in JSON format to the specified file.

    Args:
        data: the object to save.
        fname: the name of the file to save to.

    """

    json.dump(data, open(fname, 'w'))


def load_json(fname: str):
    """
    Loads an object from a JSON file.

    Args:
        fname: the name of the file to load from.

    Returns:
        the JSON object.

    """

    return json.load(open(fname, 'r'))


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
    """
    Checks whether a specified JSON file is valid.

    Args:
        path: the path to the JSON file

    Returns:
        False if the file does not exist or is not properly formatted, True
        otherwise.

    """

    try:
        json.load(open(path, 'r'))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        return False

    return True


def format_hmmss(s):
    h = int(s / 3600)
    s -= h * 3600
    m = int(s / 60)
    s -= m * 60
    return '{:d}h {:02d}m {:02d}s'.format(h, m, int(s))