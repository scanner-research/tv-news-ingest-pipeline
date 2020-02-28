#!bin/bash

set -e  # exit on error

git submodule init
git submodule update

pip3 install -r requirements.txt

docker pull scannerresearch/scannertools:cpu-latest

# Install gentle (need to run install as root)
(cd deps/gentle && sudo bash ./install.sh)

