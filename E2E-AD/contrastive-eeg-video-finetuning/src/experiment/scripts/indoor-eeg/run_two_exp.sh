#!/bin/bash

learning_rate_list=(5e-5)
learning_rate_ratio_list=(0.2)
dropout=0.1
weight_decay=8e-5
fine_tuning_policy="all_trainable"
video_type="all"
subject_type="expert"
eeg_area="[\"visual\",\"spatial\",\"semantic\",\"decision\"]"

GPU_id="0,1,2"
debug_id=3

config_path="/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/cfgs/train_config_indoor_eeg.yaml"

for idx in 0 1; do

    learning_rate=${learning_rate_list[$idx]}
    learning_rate_ratio=${learning_rate_ratio_list[$idx]}

    extra_identifier="try_${debug_id}+dropout_${dropout}+weight_decay_${weight_decay}+lr_${learning_rate}+lr_ratio_${learning_rate_ratio}_pad_video+2_stage_ft"

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