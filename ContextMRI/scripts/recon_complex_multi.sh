#!/bin/bash
index=0

for cfg_scale in 0.0 1.0 2.0 3.0; do

    echo cfg scale = ${cfg_scale}
    echo stage = ${stage}
    for eta in 0.8; do
        echo eta = ${eta}
        remainder=$((index % 8))

        CUDA_VISIBLE_DEVICES=$remainder python recon_complex_multi.py \
            --cfg_scale ${cfg_scale} \
            --eta ${eta} \
            --acc_factor 4 \
            --center_fraction 0.08 \
            --mask_type "uniform1d" &
        ((index++))
    done

done

wait

echo "All tasks completed."