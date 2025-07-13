import os
import json
import torch
from torch.utils.data import Dataset
import mne
import cv2
import yaml
from einops import rearrange

import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from models.eeg_encoder.utils import get_input_chans

eeg_area_dict = {
    "visual": ["O1", "O2", "OZ", "PO3", "PO4", "PO7", "PO8", "POZ"],
    "spatial": ["CP1", "CP2", "CPZ", "FC1", "FC2"],
    "semantic": ["FT7", "FT8", "T7", "T8", "F5", "F6", "F7", "F8"],
    "decision": ["AF3", "AF4", "F3", "F4", "FP1", "FP2", "FPZ"]
}

class RealCarDataset(Dataset):
    def __init__(self, config, dataset_type):
        assert dataset_type in ['train', 'val', 'test'], f"Invalid dataset type: {dataset_type}"
        
        self.clip_info_path = os.path.join(config["dataset_dir"], f"clip_info_{dataset_type}.json")
        self.eeg_dir = os.path.join(config["dataset_dir"], "eeg")
        self.video_dir = os.path.join(config["dataset_dir"], "video")
        self.data_duration = config['data_duration']
        self.subject_type = config['subject_type']
        self.eeg_area = config['eeg_area']
        self.eeg_channels = []
        for area in self.eeg_area:
            self.eeg_channels += eeg_area_dict[area]
        
        assert self.subject_type in ["expert", "novice", "all"]
        with open(self.clip_info_path, 'r') as f:
            self.clip_info = json.load(f)
        
        self.data_pairs = self._create_data_pairs()

    def _create_data_pairs(self):
        data_pairs = []
        if self.subject_type == "all":
            for key, clip_id_list in self.clip_info.items():
                for clip_id in clip_id_list:
                    eeg_file = os.path.join(self.eeg_dir, f"{key}_clip_{clip_id}.raw.fif")
                    video_file = os.path.join(self.video_dir, f"{key}_clip_{clip_id}.mp4")
                    data_pairs.append((eeg_file, video_file))
        else:
            for key, clip_id_list in self.clip_info.items():
                if self.subject_type in key: # choose expert or novice
                    for clip_id in clip_id_list:
                        eeg_file = os.path.join(self.eeg_dir, f"{key}_clip_{clip_id}.raw.fif")
                        video_file = os.path.join(self.video_dir, f"{key}_clip_{clip_id}.mp4")
                        data_pairs.append((eeg_file, video_file))
        
        print(f"Subject Type: {self.subject_type}\nNumber of data pairs: {len(data_pairs)}")
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        eeg_file, video_file = self.data_pairs[idx]
        # Load EEG data
        eeg_data = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        eeg_data.drop_channels(['HEO', 'VEO'])
        eeg_data.pick(self.eeg_channels)
        eeg_data = eeg_data.get_data()
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        # follow the paper, to preprocess the eeg data
        eeg_data /= 1e-4  # Normalize EEG data to [-1, 1]
        
        # Reshape eeg_data from (60, t * 200) to (60, t, 200) using rearrange
        # b: batch_size
        # c: number of channels
        # p: number of patches
        # t: number of time steps
        n_channel, n_patch_points = eeg_data.shape
        if n_patch_points < 200:
            eeg_data = torch.cat([eeg_data, torch.zeros(n_channel, 200 - n_patch_points)], dim=1)
            self.data_duration = 1
        
        if isinstance(self.data_duration, float):
            self.data_duration = int(self.data_duration)
        eeg_data = rearrange(eeg_data, 'c (p t) -> c p t', p=self.data_duration, t=200)
        
        # Load video data
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # Convert pixel values from [0, 255] to [0, 1]
            frames.append(torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1))  # Convert to tensor and permute dimensions (C, H, W)
        cap.release()
        
        video_data = torch.stack(frames)
        
        return eeg_data, video_data
    
    def obtain_standard_eeg_channel_indexes(self):
        # eeg_data_filename = os.listdir(self.eeg_dir)[0]
        # eeg_data_path = os.path.join(self.eeg_dir, eeg_data_filename)
        # eeg_data = mne.io.read_raw_fif(eeg_data_path, preload=True, verbose=False)
        # eeg_data.drop_channels(['HEO', 'VEO'])
        
        # ch_names = eeg_data.ch_names
        input_chans = get_input_chans(self.eeg_channels)
        
        return input_chans