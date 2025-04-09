#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python recon_mri.py \
    --cfg_scale 2.0 \
    --eta 0.8 \
    --acc_factor 4 \
    --center_fraction 0.08 \
    --mask_type "uniform1d"