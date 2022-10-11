#!/bin/bash

set -e  # exit on error

git submodule init
git submodule update

pip3 install -r requirements.txt

# Unpack models
(cd components && cat data.tar.bz2.* | tar -xvjf -)

# Install and build detect_black_frames with OpenCV
OPENCV_VERSION=4.6.0

sudo apt-get update
sudo apt-get install -y unzip
(cd deps/detect-black-frames \
    && wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip \
    && unzip $OPENCV_VERSION.zip && rm $OPENCV_VERSION.zip)

sudo apt-get install -y \
    build-essential \
    cmake git pkg-config \
    libgtk2.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev

(cd deps/detect-black-frames/opencv-$OPENCV_VERSION \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j8 \
    && sudo make install)

(cd deps/detect-black-frames && cmake . && make)

# Install caption-index
(cd deps/caption-index && python3 setup.py install --user)

# Install gentle (need to run install as root)
(cd deps/gentle && sudo bash ./install.sh)
