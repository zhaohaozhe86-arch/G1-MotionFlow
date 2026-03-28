#!/bin/bash
set -e

# Download whisper-large-v2 to deps/ directory
mkdir -p deps/
cd deps/

echo -e "Initializing Git LFS...\n"
git lfs install

echo -e "Cloning whisper-large-v2 from HuggingFace...\n"
git clone https://huggingface.co/openai/whisper-large-v2 models--openai--whisper-large-v2

cd ..
echo -e "whisper-large-v2 setup completed successfully.\n"