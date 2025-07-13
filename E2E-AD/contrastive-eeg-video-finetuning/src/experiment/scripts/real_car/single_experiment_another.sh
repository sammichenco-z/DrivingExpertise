#!/bin/bash

learning_rate=2e-5
learning_rate_ratio=1
dropout=0.01
weight_decay=1e-5
fine_tuning_policy="all_trainable"
GPU_id=6
debug_id=20
config_path="/data/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/cfgs/train_config_2_layer_512.yaml"

extra_identifier="debug_${debug_id}+bs_16+epoch_120+lr_${learning_rate}+dropout_${dropout}+weight_decay_${weight_decay}+eeg_normalized+full_tuning+lr_ratio_${learning_rate_ratio}+2_layer_512"

# Run the training script with the current learning rate, dropout, and weight_decay
CUDA_VISIBLE_DEVICES="$GPU_id" torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 10000)) \
    /data/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py \
    --extra_identifier "$extra_identifier" \
    --dropout "$dropout" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --fine_tuning_policy "$fine_tuning_policy" \
    --lr_ratio "$learning_rate_ratio" \
    --config "$config_path"
