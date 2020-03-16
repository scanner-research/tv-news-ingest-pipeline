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


def lock_script() -> bool:
    """
    Locks a file pertaining to this script so that it cannot be run simultaneously.

    Since the lock is automatically released when this script ends, there is no
    need for an unlock function for this use case.

    Returns:
        True if the lock was acquired, False otherwise.

    """

    lockfile = open('/tmp/{}.lock'.format(__file__), 'w')

    try:
        # Try to grab an exclusive lock on the file, raise error otherwise
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    except OSError as e:
        if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
            return False
        raise

    else:
        return True


########## Scanner ##########
# These are used in both scanner_component.py and black_frame_detection.py.

LOCAL_TOML = """
# Scanner config
# Copy this to ~/.scanner/config.toml

[storage]
type = "posix"
db_path = "/root/.scanner/db"
[network]
worker_port = "5002"
master = "localhost"
master_port = "5001"
"""

def init_scanner_config():
    """Initializes scanner configuration file."""

    scanner_config_dir = '/root/.scanner'
    if not os.path.exists(scanner_config_dir):
        os.makedirs(scanner_config_dir)

    with open(os.path.join(scanner_config_dir, 'config.toml'), 'w') as f:
        f.write(LOCAL_TOML)


def remove_unfinished_outputs(cl, video_names: List[str], all_outputs,
                              del_fn: Callable, clean: bool = False):
    """
    Removes any outputs for videos that don't have all their outputs complete.

    Args:
        cl (scannerpy.Client): the scannerpy Client object.
        video_names: the list of video names.
        all_outputs (List[List[scannerpy.NamedStream]]): a list of all the
            scanner output sinks.
        del_fn: the function used to delete outputs.
        clean: whether to force delete all existing outputs.

    """

    for collection in zip(video_names, *all_outputs):
        video_name = collection[0]
        outputs = collection[1:]
        if clean or not all(out is None or out.committed() for out in outputs):
            del_fn(cl, list(filter(lambda x: x is not None, outputs)))
        else:
            print('Using cached results for', video_name)
