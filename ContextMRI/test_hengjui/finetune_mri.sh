#!/bin/bash
name="ContextMRI"

exp_name=$1

# data_dir=/data/sls/scratch/hengjui/6S982/data/fastmri_sample
data_dir=/data/sls/scratch/hengjui/6S982/data/fastmri_0418
fastmri_dir=${data_dir}/fastmri
pretrained_path=/data/sls/scratch/hengjui/6S982/pretrained/MRI_checkpoint
# mri_metadata_dir_brain=${data_dir}/metadata_brain.csv
mri_metadata_dir_brain=${data_dir}/metadata_brain_train.csv
# mri_metadata_dir_knee="../fastmri-plus/knee/meta/metadata_train.csv" # your metadata for knee 
# mri_metadata_dir_brain="../fastmri-plus/brain/meta/metadata_train.csv" # your metadata for brain
metadata_stats=test_hengjui/metadata_stats.json

lr=5e-5
cfg_prob=0.5
cfg_strategy=independent
use_lora=True

additional_args=""
if [ "$use_lora" = True ]; then
  additional_args="--use_lora"
fi

exp_name=${exp_name}_lr${lr}_cp${cfg_prob}_cs${cfg_strategy}_lora${use_lora}
output_dir=/data/sls/scratch/hengjui/6S982/exp/${exp_name}

accelerate launch --num_processes=4 --num_machines=1 train_mri.py \
  --pretrained_model_name_or_path $pretrained_path \
  --pretrained_model_type "fastmri" \
  --data_root ${fastmri_dir} \
  --mri_metadata_dir_brain $mri_metadata_dir_brain \
  --output_dir $output_dir \
  --mixed_precision="fp16" \
  --resolution=320 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=5 \
  --learning_rate=${lr} \
  --report_to="tensorboard" \
  --lr_scheduler="linear" \
  --lr_warmup_steps=500 \
  --max_train_steps=2500 \
  --checkpointing_steps=500 \
  --seed="42" \
  --enable_condition_emb \
  --metadata_stats ${metadata_stats} \
  --cfg_prob ${cfg_prob} \
  --cfg_strategy ${cfg_strategy} \
  ${additional_args}

#   --mri_metadata_dir_knee $mri_metadata_dir_knee \
