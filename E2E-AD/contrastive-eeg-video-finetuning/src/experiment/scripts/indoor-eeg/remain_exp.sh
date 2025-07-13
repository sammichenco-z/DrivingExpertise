#!/bin/bash

exp_id_list=(2 3 8)
learning_rate=5e-5
learning_rate_ratio=0.2
dropout=0.1
weight_decay=8e-5
fine_tuning_policy="all_trainable"
video_type_list=("all" "all" "real")
subject_type_list=("expert" "novice" "all")
eeg_area="[\"semantic\",\"decision\",\"visual\",\"spatial\"]"

GPU_id="0,1,2"
debug_id=3

config_path="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/cfgs/train_config_indoor_eeg.yaml"

# Loop through experiments
for i in {0..2}; do
    exp_id="${exp_id_list[$i]}"
    video_type="${video_type_list[$i]}"
    subject_type="${subject_type_list[$i]}"
    
    # Extra identifier for the experiment
    extra_identifier="${exp_id}+opt_hyperparam"

    # Run the training script with the specified parameters
    CUDA_VISIBLE_DEVICES="$GPU_id" torchrun --nproc_per_node=3 --master_port=$((20000 + RANDOM % 10000)) \
        /home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp_indoor_dataset.py \
        --extra_identifier "$extra_identifier" \
        --dropout "$dropout" \
        --learning_rate "$learning_rate" \
        --weight_decay "$weight_decay" \
        --fine_tuning_policy "$fine_tuning_policy" \
        --lr_ratio "$learning_rate_ratio" \
        --config "$config_path" \
        --video_type "$video_type" \
        --subject_type "$subject_type" \
        --eeg_area "$eeg_area"

    # Pause before the next experiment
    sleep 10
done
