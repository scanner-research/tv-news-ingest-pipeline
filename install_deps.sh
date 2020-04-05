#!/bin/bash

set -e  # exit on error

git submodule init
git submodule update

pip3 install -r requirements.txt

# Unpack models
(cd components && cat data.tar.bz2.* | tar -xvjf -)

# Install and build detect_black_frames with OpenCV
sudo apt-get update
sudo apt-get install -y unzip
(cd deps/detect-black-frames \
    && wget https://github.com/opencv/opencv/archive/4.2.0.zip \
    && unzip 4.2.0.zip && rm 4.2.0.zip)

sudo apt-get install -y \
    build-essential \
    cmake git pkg-config \
    libgtk2.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev

(cd deps/detect-black-frames/opencv-4.2.0 \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j8 \
    && sudo make install)

(cd deps/detect-black-frames && cmake . && make)

# Install caption-index
rustup override set nightly-2019-09-01
(cd deps/caption-index && python3 setup.py install --user)

# Install gentle (need to run install as root)
(cd deps/gentle && sudo bash ./install.sh)
