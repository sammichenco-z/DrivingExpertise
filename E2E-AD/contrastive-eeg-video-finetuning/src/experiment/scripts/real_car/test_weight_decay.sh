#!/bin/bash

# Define the weight_decay values to iterate over
# weight_decay_values=(1e-5 1.5e-5 2e-5 3e-5 1e-4 5e-4 1e-3)
weight_decay_values=(1e-2 1e-1 1)

learning_rate=2e-5
dropout=0.01
GPU_id=3

# Iterate over each weight_decay value
for weight_decay in "${weight_decay_values[@]}"; do
    # Define a unique identifier for this experiment
    extra_identifier="debug_13+bs_16+epoch_120+lr_${learning_rate}+dropout_${dropout}+weight_decay_${weight_decay}"

    # Run the training script with the current learning rate, dropout, and weight_decay
    CUDA_VISIBLE_DEVICES="$GPU_id" torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 10000)) \
        /opt/nvme0/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py \
        --extra_identifier "$extra_identifier" \
        --dropout "$dropout" \
        --learning_rate "$learning_rate" \
        --weight_decay "$weight_decay"

    # Optional: Add a short delay between runs to avoid conflicts
    sleep 5
done