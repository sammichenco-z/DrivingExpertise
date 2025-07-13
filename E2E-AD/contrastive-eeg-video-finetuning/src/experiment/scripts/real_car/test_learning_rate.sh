#!/bin/bash

# Define the dropout values to iterate over
learning_rate_values_smaller=(5e-6 1e-6 5e-7 1e-7)
learning_rate_values_larger=(6e-5 1e-4 5e-4 1e-3)
dropout=0.01
weight_decay=1e-5
fine_tuning_policy="all_trainable"
GPU_id=0
debug_id=17

# Iterate over each learning rate value
for learning_rate in "${learning_rate_values_larger[@]}"; do
    # Define a unique identifier for this experiment
    extra_identifier="debug_${debug_id}_bs_16+epoch_120+lr_${learning_rate}+dropout_${dropout}+weight_decay_${weight_decay}+full_tuning"

    # Run the training script with the current learning rate
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 10000)) \
        /opt/nvme0/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py \
        --extra_identifier "$extra_identifier" \
        --dropout "$dropout" \
        --learning_rate "$learning_rate" \
        --weight_decay "$weight_decay" \
        --fine_tuning_policy "$fine_tuning_policy"

    # Optional: Add a short delay between runs to avoid conflicts
    sleep 5
done