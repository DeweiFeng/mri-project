#!/bin/bash
name="ContextMRI"

exp_name=$1

data_dir=D:/Research/data
fastmri_dir=${data_dir}/fastmri/fastmri/brain_mvue_320_train
pretrained_path=D:/Research/data/MRI_checkpoint
mri_metadata_dir_brain=${data_dir}/metadata_brain_train.csv
output_dir=D:/Research/data/${exp_name}

accelerate launch --num_processes=1 --num_machines=1 train_mri.py \
  --pretrained_model_name_or_path $pretrained_path \
  --pretrained_model_type "fastmri" \
  --data_root ${fastmri_dir} \
  --mri_metadata_dir_brain $mri_metadata_dir_brain \
  --output_dir $output_dir \
  --mixed_precision="fp16" \
  --resolution=320 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-5 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=1000 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --seed="42" \
  --metadata_stats D:/Research/data/Test_1/metadata_stats.json \
  --cfg_prob 0.5 \
  --cfg_strategy "joint" \
  --text_encoder_type "t5" \
  --t5_model_name_or_path "D:/Research/data/t5"

#   --mri_metadata_dir_knee $mri_metadata_dir_knee \
