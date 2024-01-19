#!/bin/bash

BASE_DIR=./

source $BASE_DIR/venv/bin/activate
cd $BASE_DIR



# Training script (see config.py for details)
python3 train.py \
    --data-ratio 0.05 \
    --model \
        FPN \
    --crop relative_2d_max \
    --training-dataset hrf_fusion \
    --fusion-modality slo \
    --version jbhi_rebuttal


# With exactly the same arguments as the training script, plus
# some additional arguments for the test (see the script for details)
python3 validate_ensemble.py \
    --data-ratio 0.05 \
    --model \
        FPN \
    --crop relative_2d_max \
    --training-dataset hrf_fusion \
    --test-dataset hrf_fusion \
    --eval-split hrf_images_with_oct_masks \
    --fusion-modality slo \
    --save-all-outputs \
    --version jbhi_rebuttal
