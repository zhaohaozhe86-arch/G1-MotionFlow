#!/bin/bash
set -e

# Download raw motion sequences (tar.xz) without extracting
echo -e "Downloading raw motion sequences from Google Drive...\n"

gdown 1kv71BeT-ZxHfJM58FPKOruizuKtpkCmU -O AMASS.tar.xz

echo -e "Raw motion sequences download completed.\n"