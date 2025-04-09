#!/bin/bash
name="ContextMRI"

pretrained_path="./MRI_checkpoint"
mri_metadata_dir_knee="../fastmri-plus/knee/meta/metadata_train.csv" # your metadata for knee 
mri_metadata_dir_brain="../fastmri-plus/brain/meta/metadata_train.csv" # your metadata for brain
output_dir="./output"

accelerate launch --num_processes=8 --num_machines=1 train_mri.py \
  --pretrained_model_name_or_path $pretrained_path  \
  --mri_metadata_dir_knee $mri_metadata_dir_knee \
  --mri_metadata_dir_brain $mri_metadata_dir_brain \
  --output_dir $output_dir \
  --mixed_precision="fp16" \
  --resolution=320 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --checkpointing_steps=1000 \
  --seed="42" \