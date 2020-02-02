"""
File: utils.py
--------------
Contains utility functions used across the pipeline.

"""

import json
import os
from pathlib import Path


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
