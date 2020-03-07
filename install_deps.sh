#!bin/bash

set -e  # exit on error

git submodule init
git submodule update

sudo pip3 install -r requirements.txt

# Install gentle (need to run install as root)
(cd deps/gentle && sudo bash ./install.sh)

