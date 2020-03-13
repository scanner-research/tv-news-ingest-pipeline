#!/bin/bash

set -e  # exit on error

git submodule init
git submodule update

sudo pip3 install -r requirements.txt

# Install caption-index
rustup override set nightly-2019-09-01
(cd deps/caption-index && python3 setup.py install --user)

# Install gentle (need to run install as root)
(cd deps/gentle && sudo bash ./install.sh)
