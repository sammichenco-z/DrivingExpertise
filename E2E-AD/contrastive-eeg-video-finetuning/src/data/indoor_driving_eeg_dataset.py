import os
import cv2
import sys
import json
import torch
import pandas as pd
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from models.eeg_encoder.utils import get_input_chans

# we use several things to config dataset
# 1. mode: train, val, test
# 2. video_type: real | virtual | all
# 3. subject_type: expert | novice | all
# 4. eeg_area: subset of ["visual", "spatial", "semantic", "decision"]
# 5. info_dir: the directory of the info files
# 6. eeg_dir: the directory of the EEG data
# 7. video_dir: the directory of the video data

eeg_area_dict = {
    "visual": ["O1", "O2", "OZ", "PO3", "PO4", "PO7", "PO8", "POZ"],
    "spatial": ["CP1", "CP2", "CPZ", "FC1", "FC2"],
    "semantic": ["FT7", "FT8", "T7", "T8", "F5", "F6", "F7", "F8"],
    "decision": ["AF3", "AF4", "F3", "F4", "FP1", "FP2", "FPZ"]
}

class IndoorDrivingEEGDataset(Dataset):
    def __init__(self, config):
        # we first check the config
        assert config["mode"] in ["training_set", "validation_set", "test_set"], f"Invalid mode: {config['mode']}"
        assert config["video_type"] in ["real", "virtual", "all"], f"Invalid video type: {config['video_type']}"
        assert config["subject_type"] in ["expert", "novice", "all"], f"Invalid subject type: {config['subject_type']}"
        for single_eeg_area in config["eeg_area"]:
            assert single_eeg_area in ["visual", "spatial", "semantic", "decision"], f"Invalid EEG area: {single_eeg_area}"
        assert os.path.exists(config["info_dir"]), f"Invalid info directory: {config['info_dir']}"
        assert os.path.exists(config["eeg_dir"]), f"Invalid EEG directory: {config['eeg_dir']}"
        assert os.path.exists(config["video_dir"]), f"Invalid video directory: {config['video_dir']}"
        self.mode = config["mode"]
        self.video_type = config["video_type"]
        self.subject_type = config["subject_type"]
        self.eeg_area = config["eeg_area"]
        self.info_dir = config["info_dir"]
        self.eeg_dir = config["eeg_dir"]
        self.video_dir = config["video_dir"]
        self.preprocess = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms()
        # load the info files
        info_filename = f"{self.video_type}_video_{self.subject_type}_subject.json"
        info_path = os.path.join(self.info_dir, info_filename)
        assert os.path.exists(info_path), f"Invalid info file: {info_path}"
        with open(info_path, 'r') as f:
            self.info = json.load(f)[self.mode] # only load the mode we need
    
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        item_info = self.info[idx]

        # obtain the used eeg channels
        eeg_channel_list = []
        for single_eeg_area in self.eeg_area:
            eeg_channel_list += eeg_area_dict[single_eeg_area]

        # load the EEG data
        eeg_relative_path = item_info["eeg_path"]
        eeg_path = os.path.join(self.eeg_dir, eeg_relative_path)
        assert os.path.exists(eeg_path), f"Invalid EEG file: {eeg_path}"
        eeg_data = self._load_eeg_data(eeg_path, eeg_channel_list)

        # load the video data
        video_relative_path = item_info["video_path"]
        video_path = os.path.join(self.video_dir, video_relative_path)
        assert os.path.exists(video_path), f"Invalid video file: {video_path}"
        video_data = self._load_video_data(video_path)

        return eeg_data, video_data

    def _load_eeg_data(self, eeg_path, eeg_channel_list):
        eeg_data = pd.read_csv(eeg_path, index_col=0, usecols=lambda col: 'Unnamed' not in col)

        # select the EEG channels we need
        eeg_data = eeg_data.loc[eeg_channel_list]
        # select the duration we need, and resample the data to 200Hz
        eeg_data = eeg_data.iloc[:, 200::5]  # select every 5th sample, starting from the 200th sample
        # convert the data to numpy array
        eeg_data = eeg_data.to_numpy()
        # convert the data to torch tensor
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        # follow the paper, to preprocess the eeg data, divided by 0.1 mV
        eeg_data /= 1e-4  # Normalize EEG data to [-1, 1]
        
        # Reshape eeg_data from (n_channel, t * 200) to (n_channel, t, 200) using rearrange
        # b: batch_size
        # c: number of channels
        # p: number of patches
        # t: number of time steps
        n_channel, n_patch_points = eeg_data.shape
        if n_patch_points < 200:
            eeg_data = torch.cat([torch.zeros(n_channel, 200 - n_patch_points), eeg_data], dim=1)

        eeg_data = rearrange(eeg_data, 'c (p t) -> c p t', p=1, t=200)

        return eeg_data

    def _load_video_data(self, video_path):
        # Load the video data
        video_data = []
        cap = cv2.VideoCapture(video_path)
        # here, we assume all video's fps is 30 (and we checked that almost all videos are 30fps, some are 29.97fps)
        # and we only read the first 18 frames (which equals to 0.6s)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= 18, f"Video {video_path} has less than 18 frames"
        # read the first 18 frames
        for i in range(18):
            ret, frame = cap.read() # shape of frame is (H, W, C)
            if not ret:
                break
            # convert the frame from np.ndarray to torch tensor
            frame = torch.tensor(frame, dtype=torch.float32)  # shape of frame is (H, W, C)
            video_data.append(frame)
        cap.release()

        # here, we assume all videos are 30fps, and we only read the first 18 frames (which equals to 0.6s)
        # for data efficiency, we resample the fps here from 30 to 10
        # so we need to select every 3rd frame
        video_data = video_data[::3]  # select every 3rd frame
        # Convert to tensor
        video_data = torch.stack(video_data)  # shape of video_data is (T, H, W, C)
        # permute the dimensions to (T, C, H, W)
        video_data = video_data.permute(0, 3, 1, 2)
        # (Here, only 6 frames is used, fps = 10, which equals 0.6s)
        # however, in EEG, we use zero to stands the first 0.4 second, here for the video, we should align with that
        _, C, H, W = video_data.shape
        n_resting_frame = 4
        video_data = torch.cat([torch.zeros(n_resting_frame, C, H, W), video_data], dim=0)
        # preprocess the video data
        # after preprocess, shape of video data is (C, T, H, W)
        video_data = video_data / 255.0  # Convert pixel values from [0, 255] to [0, 1]
        video_data = self.preprocess(video_data)
        # After preprocess, shape of video data is (C, T, H, W)
        # But in Video encoder, it will Change shape from (B, T, C, H, W) to (B, C, T, H, W)
        # So we need to swap the dimensions to (T, C, H, W)
        video_data = rearrange(video_data, 'c t h w -> t c h w')

        return video_data

    def get_eeg_input_chans(self):
        # obtain the used eeg channels
        eeg_channel_list = []
        for single_eeg_area in self.eeg_area:
            eeg_channel_list += eeg_area_dict[single_eeg_area]
        standard_eeg_channels = get_input_chans(eeg_channel_list)
        # convert the list to torch tensor
        eeg_input_chans = torch.tensor(standard_eeg_channels, dtype=torch.int32)
        return eeg_input_chans


def main():
    dataset_config = {
        "mode": "training_set",
        "video_type": "all",
        "subject_type": "all",
        "eeg_area": ["semantic", "decision"],
        "info_dir": "/home/aidrive/zhengxj/projects/cognitive-driving-world-model/datasets/indoor-driving-eeg/info",
        "eeg_dir": "/home/aidrive/zhengxj/projects/cognitive-driving-world-model/datasets/indoor-driving-eeg/eeg",
        "video_dir": "/home/aidrive/zhengxj/projects/cognitive-driving-world-model/datasets/indoor-driving-eeg/video"
    }
    dataset = IndoorDrivingEEGDataset(dataset_config)
    print(f"Number of samples: {len(dataset)}")

    eeg_sample, video_sample, eeg_channels = dataset[0]  # Load the first sample
    print(f"EEG sample data: {eeg_sample.shape}")
    print(f"Video sample data: {video_sample.shape}")
    print(f"EEG input channels: {eeg_channels}")
    # save the first video frame
    import matplotlib.pyplot as plt
    plt.imshow(video_sample[:,0,...].permute(1, 2, 0).numpy())
    plt.axis('off')
    # we use server without monitor
    plt.savefig("/home/aidrive/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/data/tmp/test.png", bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()