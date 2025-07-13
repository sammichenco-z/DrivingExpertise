#!/bin/bash

# Remote server information
REMOTE_SERVER="10.0.0.251"
REMOTE_USER="zhengxj"
REMOTE_PORT="20122"
REMOTE_FOLDER="/home/aidrive/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor-eeg-exp"

# Source checkpoint paths and their corresponding exp_identifiers
declare -A CKPTS=(
    ["indoor_eeg_dataset_1_all_video+all_subject+eeg_semantic_decision_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_1_all_video+all_subject+eeg_semantic_decision_visual_spatial+opt_hyperparam/video_encoder_epoch_6.safetensors"
    ["indoor_eeg_dataset_4_all_video+expert_subject+eeg_semantic_decision+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_4_all_video+expert_subject+eeg_semantic_decision+opt_hyperparam/video_encoder_epoch_6.safetensors"
    ["indoor_eeg_dataset_5_all_video+novice_subject+eeg_semantic_decision+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_5_all_video+novice_subject+eeg_semantic_decision+opt_hyperparam/video_encoder_epoch_6.safetensors"
    ["indoor_eeg_dataset_6_all_video+expert_subject+eeg_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_6_all_video+expert_subject+eeg_visual_spatial+opt_hyperparam/video_encoder_epoch_4.safetensors"
    ["indoor_eeg_dataset_7_all_video+novice_subject+eeg_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_7_all_video+novice_subject+eeg_visual_spatial+opt_hyperparam/video_encoder_epoch_4.safetensors"
    ["indoor_eeg_dataset_9_real_video+expert_subject+eeg_semantic_decision_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_9_real_video+expert_subject+eeg_semantic_decision_visual_spatial+opt_hyperparam/video_encoder_epoch_7.safetensors"
    ["indoor_eeg_dataset_10_real_video+novice_subject+eeg_semantic_decision_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_10_real_video+novice_subject+eeg_semantic_decision_visual_spatial+opt_hyperparam/video_encoder_epoch_3.safetensors"
    ["indoor_eeg_dataset_11_real_video+expert_subject+eeg_semantic_decision+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_11_real_video+expert_subject+eeg_semantic_decision+opt_hyperparam/video_encoder_epoch_3.safetensors"
    ["indoor_eeg_dataset_12_real_video+novice_subject+eeg_semantic_decision+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_12_real_video+novice_subject+eeg_semantic_decision+opt_hyperparam/video_encoder_epoch_7.safetensors"
    ["indoor_eeg_dataset_13_real_video+expert_subject+eeg_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_13_real_video+expert_subject+eeg_visual_spatial+opt_hyperparam/video_encoder_epoch_3.safetensors"
    ["indoor_eeg_dataset_14_real_video+novice_subject+eeg_visual_spatial+opt_hyperparam"]="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/indoor_eeg_dataset_14_real_video+novice_subject+eeg_visual_spatial+opt_hyperparam/video_encoder_epoch_4.safetensors"
)

# Function to copy checkpoints
for exp_id in "${!CKPTS[@]}"; do
    source_path="${CKPTS[$exp_id]}"
    target_path="${REMOTE_FOLDER}/${exp_id}.safetensors"
    
    echo "Copying $source_path to $REMOTE_SERVER:$target_path"
    scp -P "$REMOTE_PORT" "$source_path" "${REMOTE_USER}@${REMOTE_SERVER}:$target_path"
    
    if [ $? -eq 0 ]; then
        echo "Successfully copied $source_path to $REMOTE_SERVER:$target_path"
    else
        echo "Failed to copy $source_path to $REMOTE_SERVER:$target_path"
    fi
done