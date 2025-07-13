#!/bin/bash

learning_rate=5e-5
learning_rate_ratio=0.2
dropout=0.1
weight_decay=8e-5
fine_tuning_policy="all_trainable"
video_type="real"
subject_type_list=("expert" "novice")
eeg_area_list=("[\"semantic\",\"decision\"]" "[\"visual\",\"spatial\"]" "[\"semantic\",\"decision\",\"visual\",\"spatial\"]")

GPU_id="0,1,2"
debug_id=3

config_path="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/cfgs/train_config_indoor_eeg.yaml"

for subject_type in "${subject_type_list[@]}"; do
    for eeg_area in "${eeg_area_list[@]}"; do
        # extra_identifier="try_${debug_id}+dropout_${dropout}+weight_decay_${weight_decay}+lr_${learning_rate}+lr_ratio_${learning_rate_ratio}_pad_video+2_stage_ft"
        extra_identifier="opt_hyperparam"
        # Run the training script with the current learning rate, dropout, and weight_decay
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

        sleep 10
    done
done


extra_identifier="opt_hyperparam"
# Run the training script with the current learning rate, dropout, and weight_decay
CUDA_VISIBLE_DEVICES="$GPU_id" torchrun --nproc_per_node=3 --master_port=$((20000 + RANDOM % 10000)) \
    /home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp_indoor_dataset.py \
    --extra_identifier "$extra_identifier" \
    --dropout "$dropout" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --fine_tuning_policy "$fine_tuning_policy" \
    --lr_ratio "$learning_rate_ratio" \
    --config "$config_path" \
    --video_type "all" \
    --subject_type "all" \
    --eeg_area "[\"semantic\",\"decision\",\"visual\",\"spatial\"]"