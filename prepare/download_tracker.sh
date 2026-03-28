#!/bin/bash
set -e

# Download and extract tracker.tar.xz
echo -e "Downloading tracker.tar.xz from Google Drive...\n"
gdown 1ZBHIWAI4y_O4uuzKKPSC3O1yrh0wnbGv -O tracker.tar.xz

echo -e "Download completed. Extracting to current directory...\n"
tar -xf tracker.tar.xz

echo -e "Cleaning up the archive...\n"
rm tracker.tar.xz

echo -e "The 'tracker' directory is ready.\n"