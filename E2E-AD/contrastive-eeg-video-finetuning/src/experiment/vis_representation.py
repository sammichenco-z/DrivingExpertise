import os
import sys
import mne
import cv2
import json
import timm
import torch
import numpy as np
from torch import nn
from einops import rearrange
from tqdm import tqdm
exp_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(exp_dir)
model_dir = os.path.join(src_dir, "models")
sys.path.append(model_dir)

project_dir = "/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning"
dataset_dir = "/home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/datasets/real_car"
fig_save_dir = os.path.join(exp_dir, "vis")

import eeg_encoder.labram
from eeg_encoder.utils import get_input_chans

from torchvision.models.video.swin_transformer import swin3d_b, Swin3D_B_Weights

def obtain_representation(dataset_dir, vis_datset_identifier, vis_participant_identifier):
    def create_data_pairs(clip_info, participant_identifier):
        data_pairs = []
        for key, clip_num in clip_info.items():
            if participant_identifier in key:
                for clip_id in range(clip_num):
                    eeg_file = os.path.join(dataset_dir, vis_datset_identifier, "eeg", f"{key}_clip_{clip_id}.raw.fif")
                    video_file = os.path.join(dataset_dir, vis_datset_identifier, "video", f"{key}_clip_{clip_id}.mp4")
                    driving_condition = key.split("_")[0]
                    data_pairs.append((eeg_file, video_file, driving_condition))
        print(f"Participant Identifier: {participant_identifier}\nNumber of data pairs: {len(data_pairs)}")
        return data_pairs
    def get_item(data_pairs, idx):
        eeg_file, video_file, driving_condition = data_pairs[idx]
        # Load EEG data
        eeg_data = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        eeg_data.drop_channels(['HEO', 'VEO'])
        eeg_data = eeg_data.get_data()
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Reshape eeg_data from (60, t * 200) to (60, t, 200)
        n_channel, n_patch_points = eeg_data.shape
        duration = n_patch_points // 200
        eeg_data = rearrange(eeg_data, 'c (p t) -> c p t', p=duration, t=200)
        
        # Load video data
        video_cap = cv2.VideoCapture(video_file)
        frames = []
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1))
        video_cap.release()
        
        video_data = torch.stack(frames, dim=0)  # Shape: (num_frames, C, H, W)
        
        return eeg_data, video_data, driving_condition

    def obtain_standard_eeg_channel_indexes():
        eeg_dir = os.path.join(dataset_dir, vis_datset_identifier, "eeg")
        eeg_data_filename = os.listdir(eeg_dir)[0]
        eeg_data_path = os.path.join(eeg_dir, eeg_data_filename)
        eeg_data = mne.io.read_raw_fif(eeg_data_path, preload=True, verbose=False)
        eeg_data.drop_channels(['HEO', 'VEO'])
        
        ch_names = eeg_data.ch_names
        input_chans = get_input_chans(ch_names)
        
        return input_chans

    clip_info_path = os.path.join(dataset_dir, vis_datset_identifier, f"clip_info.json")
    with open(clip_info_path, 'r') as f:
        clip_info = json.load(f)
        
    data_pairs = create_data_pairs(clip_info, vis_participant_identifier)
    # load models
    eeg_encoder = timm.create_model("labram_base_patch200_200_no_head", pretrained=True)
    eeg_encoder_missing_keys, eeg_encoder_unexpected_keys = eeg_encoder.load_ckpts(fine_tuned=False, config={"ckpt_load_path": "ckpts/eeg_encoder_base.pth"}, project_dir=project_dir)
    print(f"EEG Encoder Missing Keys: {eeg_encoder_missing_keys}")
    print(f"EEG Encoder Unexpected Keys: {eeg_encoder_unexpected_keys}")
    
    standard_eeg_channels = obtain_standard_eeg_channel_indexes()
    
    video_encoder = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
    video_encoder.head = nn.Identity()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_encoder.to(device)
    video_encoder.to(device)
    # otain representation and condition list
    eeg_representation_list = []
    video_representation_list = []
    driving_condition_list = []
    
    for idx in tqdm(range(len(data_pairs))):
        eeg_data, video_data, driving_condition = get_item(data_pairs, idx)
        eeg_data = eeg_data.to(device)
        video_data = video_data.to(device)
        
        eeg_data = eeg_data.unsqueeze(0)  # Shape: (1, C, P, T)
        video_data = video_data.unsqueeze(0)  # Shape: (1, num_frames, C, H, W)
        
        with torch.no_grad():
            eeg_representation = eeg_encoder(eeg_data, input_chans=standard_eeg_channels)
            video_representation = video_encoder(video_data.permute(0, 2, 1, 3, 4))
        
        eeg_representation_list.append(eeg_representation.cpu().numpy())
        video_representation_list.append(video_representation.cpu().numpy())
        driving_condition_list.append(driving_condition)
        
    eeg_representation = np.concatenate(eeg_representation_list, axis=0)
    video_representation = np.concatenate(video_representation_list, axis=0)
    driving_condition = np.array(driving_condition_list)
    print(f"EEG Representation Shape: {eeg_representation.shape}")
    print(f"Video Representation Shape: {video_representation.shape}")
    print(f"Driving Condition Shape: {driving_condition.shape}")
    return eeg_representation, video_representation, driving_condition
   

def visualize_representation(eeg_representation, video_representation, driving_condition):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Use a colormap with enough unique colors
    cmap = plt.get_cmap('tab20')  # 20 unique colors
    
    driving_condition_list = np.array(list(set(driving_condition)))
    n_driving_conditions = len(driving_condition_list)
    
    figure_eeg = plt.figure(figsize=(10, 8))
    tsne_eeg = TSNE(n_components=2, perplexity=30, n_iter=300)
    eeg_feature_2d = tsne_eeg.fit_transform(eeg_representation)
    ax_eeg = figure_eeg.add_subplot(111)
    for i, condition in enumerate(driving_condition_list):
        indices = np.where(driving_condition == condition)[0]
        ax_eeg.scatter(eeg_feature_2d[indices, 0], eeg_feature_2d[indices, 1], label=condition, color=cmap(i / n_driving_conditions), alpha=0.7)
    ax_eeg.set_title("EEG Representation")
    ax_eeg.legend()
    plt.savefig(os.path.join(fig_save_dir, "eeg_representation.png"))
    
    figure_video = plt.figure(figsize=(10, 8))
    tsne_video = TSNE(n_components=2, perplexity=30, n_iter=300)
    video_feature_2d = tsne_video.fit_transform(video_representation)
    ax_video = figure_video.add_subplot(111)
    for i, condition in enumerate(driving_condition_list):
        indices = np.where(driving_condition == condition)[0]
        ax_video.scatter(video_feature_2d[indices, 0], video_feature_2d[indices, 1], label=condition, color=cmap(i / n_driving_conditions), alpha=0.7)
    ax_video.set_title("Video Representation")
    ax_video.legend()
    plt.savefig(os.path.join(fig_save_dir, "video_representation.png"))
    

def main():
    vis_datset_identifier = "processed_duration_2_video_fps_2"
    vis_participant_identifier = "P06"
    
    eeg_representation, video_representation, driving_condition = obtain_representation(dataset_dir, vis_datset_identifier, vis_participant_identifier)
    visualize_representation(eeg_representation, video_representation, driving_condition)
    

if __name__ == "__main__":
    main()