#!/bin/bash

# Activate the Anaconda environment
source /opt/anaconda3/bin/activate image-processing

# Run the machine learning module
python -m src.cli ml --task train --dataset data/images_Kimia216 --classifier ensemble --features all --output output/machine_learning_kimia216_ensemble

# Deactivate the environment
conda deactivate
