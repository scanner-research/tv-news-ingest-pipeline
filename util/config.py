#!/usr/bin/env python3

"""
File: config.py
---------------
Contains config variables for the pipeline, read from the 'config' file.

"""

import os

import yaml

CONFIG_FILE = 'config.yml'

CONFIG_KEYS = {
    'num_pipelines': (os.cpu_count() // 2 if os.cpu_count() else 1),
    'stride': 1,
    'montage_width': 10,
    'montage_height': 6,
    'aws_access_key_id': None,
    'aws_secret_access_key': None,
    'disable': [],
    'host': '127.0.0.1:2375'
}

# Set config variables defaults
for key, value in CONFIG_KEYS.items():
    vars()[key.upper()] = value

# Read values from config file
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        keys = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in keys.items():
            assert key in CONFIG_KEYS, 'Invalid configuration key.'
            vars()[key.upper()] = value
