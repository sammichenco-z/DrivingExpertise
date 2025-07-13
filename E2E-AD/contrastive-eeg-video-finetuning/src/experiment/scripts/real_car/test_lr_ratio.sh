#!/bin/bash

# Define the dropout values to iterate over
learning_rate=2e-5
learning_rate_ratio_values=(1 1e-1 1e-2 1e-3 1e-4 10)
dropout=0.01
weight_decay=1e-5
fine_tuning_policy="all_trainable"
GPU_id=4
debug_id=19

# Iterate over each learning rate value
for lr_ratio in "${learning_rate_ratio_values[@]}"; do
    # Define a unique identifier for this experiment
    extra_identifier="debug_${debug_id}_bs_16+epoch_120+lr_${learning_rate}+dropout_${dropout}+weight_decay_${weight_decay}+eeg_normalized+full_tuning+lr_ratio_${lr_ratio}"

    # Run the training script with the current learning rate
    CUDA_VISIBLE_DEVICES="$GPU_id" torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 10000)) \
        /opt/nvme0/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py \
        --extra_identifier "$extra_identifier" \
        --dropout "$dropout" \
        --learning_rate "$learning_rate" \
        --weight_decay "$weight_decay" \
        --fine_tuning_policy "$fine_tuning_policy" \
        --lr_ratio "$lr_ratio"

    # Optional: Add a short delay between runs to avoid conflicts
    sleep 5
done