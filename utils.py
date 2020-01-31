import json
import os
from pathlib import Path


def save_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)


def load_json(fname):
    with open(fname) as f:
        return json.load(f)


def get_base_name(path):
    return os.path.splitext(Path(path).name)[0]
