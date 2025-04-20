#!/bin/bash
name="ContextMRI"

exp_name=$1

data_dir=/data/sls/scratch/hengjui/6S982/data/fastmri_sample
fastmri_dir=${data_dir}/fastmri
pretrained_path=/data/sls/scratch/hengjui/6S982/pretrained/MRI_checkpoint
mri_metadata_dir_brain=${data_dir}/metadata_brain.csv
# mri_metadata_dir_knee="../fastmri-plus/knee/meta/metadata_train.csv" # your metadata for knee 
# mri_metadata_dir_brain="../fastmri-plus/brain/meta/metadata_train.csv" # your metadata for brain
output_dir=/data/sls/scratch/hengjui/6S982/exp/${exp_name}

accelerate launch --num_processes=1 --num_machines=1 train_mri.py \
  --pretrained_model_name_or_path $pretrained_path \
  --pretrained_model_type "fastmri" \
  --data_root ${fastmri_dir} \
  --mri_metadata_dir_brain $mri_metadata_dir_brain \
  --output_dir $output_dir \
  --mixed_precision="fp16" \
  --resolution=320 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --report_to="tensorboard" \
  --lr_scheduler="linear" \
  --lr_warmup_steps=1000 \
  --max_train_steps=10000 \
  --checkpointing_steps=100 \
  --seed="42" \
  --enable_condition_emb \
  --metadata_stats metadata_stats.json \
  --cfg_prob 0.5 \
  --cfg_strategy "independent"

#   --mri_metadata_dir_knee $mri_metadata_dir_knee \
