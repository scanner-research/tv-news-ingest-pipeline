"""
File: config.py
---------------
Contains config variables for the pipeline, read from the 'config.yml' file.
Defaults are defined here.

"""

import os

import yaml

CONFIG_FILE = 'config.yml'

CONFIG_KEYS = {
    # General
    'disable': [],

    # Face component
    'interval': 1, # seconds/sample

    # Face identification with AWS
    'montage_width': 10, # number of columns of images
    'montage_height': 6, # number of rows of images
    'aws_access_key_id': None,
    'aws_secret_access_key': None,
    'aws_region': 'us-west-1',
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
