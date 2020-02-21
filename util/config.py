#!/usr/bin/env python3

"""
File: config.py
---------------

"""

import os

CONFIG_FILE = 'config'

CONFIG_KEYS = {
    'montage_height': 6,
    'montage_width': 10,
    'stride': 3,
    'num_pipelines': (os.cpu_count() // 2 if os.cpu_count() else 1),
    'aws_credentials_file': 'aws-credentials.csv',
}

# Set config variables defaults
for key, value in CONFIG_KEYS.items():
    vars()[key.upper()] = value

# Read values from config file
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue

            key, value = line.split('=')
            assert key in CONFIG_KEYS, 'Invalid configuration key.'

            vars()[key.upper()] = type(CONFIG_KEYS[key])(value)
