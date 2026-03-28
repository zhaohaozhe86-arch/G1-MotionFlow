#!/bin/bash
set -e

# Download and extract experiments.tar.xz
echo -e "Downloading experiments.tar.xz from Google Drive...\n"
gdown 1AEMBy0S9_rXPHUJtzbN6j21z3Nq0l_JN -O experiments.tar.xz

echo -e "Download completed. Extracting to current directory...\n"
tar -xf experiments.tar.xz

echo -e "Cleaning up the archive...\n"
rm experiments.tar.xz

echo -e "The 'experiments' directory is ready.\n"