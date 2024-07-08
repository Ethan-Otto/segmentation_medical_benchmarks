#!/bin/bash
ENV_NAME=detectron2

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
source activate $ENV_NAME

echo "Running postBuild script"
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html