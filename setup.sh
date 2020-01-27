#!/bin/bash

# Start from directory of script
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p data/ImageNet && ln -s /data/ImageNet/ILSVRC2012/train data/ImageNet/train

# Set up symlinks for the example notebooks
sh notebooks/setup_notebooks.sh