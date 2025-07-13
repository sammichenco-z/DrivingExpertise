#!/bin/bash

# Define the dropout values to iterate over
dropout_values=(0.02 0.03 0.04 0.05)
learning_rate=2e-5
weight_decay=1e-5
GPU_id=0


# Iterate over each dropout value
for dropout in "${dropout_values[@]}"; do
    # Define a unique identifier for this experiment
    extra_identifier="debug_15+bs_16+epoch_120+lr_${learning_rate}+dropout_${dropout}+weight_decay_${weight_decay}+add_param"

    # Run the training script with the current dropout value
    CUDA_VISIBLE_DEVICES="$GPU_id" torchrun --nproc_per_node=1 --master_port=$((29000 + RANDOM % 100)) \
        /opt/nvme0/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py \
        --extra_identifier "$extra_identifier" \
        --dropout "$dropout" \
        --learning_rate "$learning_rate" \
        --weight_decay "$weight_decay"

    # Optional: Add a short delay between runs to avoid conflicts
    sleep 5
done