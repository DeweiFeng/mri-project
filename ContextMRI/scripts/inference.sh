#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg_scale 2.0 --mri_type "skm-tea"
