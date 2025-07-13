import os
import cv2
import sys
import yaml
import torch
from PIL import Image
from torchvision import transforms

proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(proj_dir, 'src')
cur_dir = os.path.join(src_dir, 'run_model')
sys.path.append(src_dir)
from torch import nn
from models.video_encoder.video_encoder import VideoEncoder
from vision15.models.video.swin_transformer import swin3d_b, Swin3D_B_Weights

# read the config file
config_path = os.path.join(proj_dir, 'cfgs', 'video_encoder_inference.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
project_dir = config['project_dir']
config = config['video_encoder']

# 原始帧经过这个进行预处理
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def obtain_example_data(version_identifier):
    video_file = os.path.join(cur_dir, 'example_data', version_identifier)
    cap = cv2.VideoCapture(video_file)
    
    # output some video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f"Video FPS: {fps}")
    print(f"Video duration: {duration} seconds") 
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # (H, W, C)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).numpy() # (H, W, C) -[ToTensor]-> (C, H, W)
        frames.append(torch.tensor(frame, dtype=torch.float32)) # (C, H, W)
    cap.release()
    video_data = torch.stack(frames).unsqueeze(0)  # Add batch dimension (T, C, H, W) -> (B, T, C, H, W)
    
    return video_data

def inference_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("# === Memo === #")
    print("1. Please check that the model is loaded its ckeckpoint correctly.")
    # 测试EEG微调版本 video swin transformer
    # print("===> Test EEG fine-tuned video swin transformer")
    # # 测试 duration = 2, fps = 10
    # print("===> Test duration = 2, fps = 10")
    # config["ckpt_load_path"] = "ckpts/duration_2_fps_10/video_encoder_ft_1.safetensors"
    # cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    # cognitive_video_encoder.load_ckpts()
    # video_data_duration_2_fps_10 = obtain_example_data("duration_2_fps_10.mp4")
    # video_data_duration_2_fps_10 = video_data_duration_2_fps_10.to(device)
    # print(f"shape of video data: {video_data_duration_2_fps_10.shape}")
    # with torch.no_grad():
    #     video_embeddings = cognitive_video_encoder(video_data_duration_2_fps_10)
        
    # print("Shape of video embeddings:", video_embeddings.shape)

    # # 测试 duration = 2, fps = 2
    print("===> Test duration = 2, fps = 2")
    config["ckpt_load_path"] = "ckpts/duration_2_fps_2_all_debug_18+bs_16+epoch_120+lr_2e-5+dropout_0.01+weight_decay_1e-5+eeg_normalized/video_encoder_epoch_119.safetensors"
    cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    cognitive_video_encoder.load_ckpts()
    video_data_duration_2_fps_2 = obtain_example_data("duration_2_fps_2.mp4")
    video_data_duration_2_fps_2 = video_data_duration_2_fps_2.to(device)
    print(f"shape of video data: {video_data_duration_2_fps_2.shape}")
    with torch.no_grad():
        video_embeddings = cognitive_video_encoder(video_data_duration_2_fps_2)
        
    # print("Shape of video embeddings:", video_embeddings.shape)
    
    # # 测试 duration = 0.2, fps = 10
    # print("===> Test duration = 0.2, fps = 10")
    # config["ckpt_load_path"] = "ckpts/duration_0.2_fps_10/video_encoder_ft_1.safetensors"
    # cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    # cognitive_video_encoder.load_ckpts()
    # video_data_duration_0_2_fps_10 = obtain_example_data("duration_0.2_fps_10.mp4")
    # video_data_duration_0_2_fps_10 = video_data_duration_0_2_fps_10.to(device)
    # print(f"shape of video data: {video_data_duration_0_2_fps_10.shape}")
    # with torch.no_grad():
    #     video_embeddings = cognitive_video_encoder(video_data_duration_0_2_fps_10)
        
    # print("Shape of video embeddings:", video_embeddings.shape)
    
    # # 测试 duration = 0.2, fps = 10
    # print("===> Test duration = 0.4, fps = 10")
    # config["ckpt_load_path"] = "ckpts/duration_0.4_fps_10/video_encoder_epoch_26.safetensors"
    # cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    # cognitive_video_encoder.load_ckpts()
    # video_data_duration_0_4_fps_10 = obtain_example_data("duration_0.4_fps_10.mp4")
    # video_data_duration_0_4_fps_10 = video_data_duration_0_4_fps_10.to(device)
    # print(f"shape of video data: {video_data_duration_0_4_fps_10.shape}")
    # with torch.no_grad():
    #     video_embeddings = cognitive_video_encoder(video_data_duration_0_4_fps_10)
        
    # print("Shape of video embeddings:", video_embeddings.shape)
    
    # 测试原始预训练参数版本video swin transformer
    print("===> Test original pre-trained video swin transformer")
    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    video_swin_transformer = swin3d_b(weights=weights).to(device)
    print("check parameter, head:\n ", video_swin_transformer.head.weight)
    video_swin_transformer.head = nn.Identity()  # Remove the classification head
    
    # print("===> Test duration = 2, fps = 10")
    # with torch.no_grad():
    #     video_data_duration_2_fps_10 = video_data_duration_2_fps_10.permute(0, 2, 1, 3, 4)  # Change shape from (B, T, C, H, W) to (B, C, T, H, W)
    #     # test_tensor = torch.rand(1,4,3,736,1280).to(device)
    #     video_embeddings = video_swin_transformer(video_data_duration_2_fps_10)
    # print("Shape of video embeddings:", video_embeddings.shape)
    
    print("===> Test duration = 2, fps = 2")
    with torch.no_grad():
        video_data_duration_2_fps_2 = video_data_duration_2_fps_2.permute(0, 2, 1, 3, 4) # Change shape from (B, T, C, H, W) to (B, C, T, H, W)
        video_embeddings = video_swin_transformer(video_data_duration_2_fps_2)
    print("Shape of video embeddings:", video_embeddings.shape)
    
    # print("===> Test duration = 0.2, fps = 10")
    # with torch.no_grad():
    #     video_data_duration_0_2_fps_10 = video_data_duration_0_2_fps_10.permute(0, 2, 1, 3, 4)
    #     video_embeddings = video_swin_transformer(video_data_duration_0_2_fps_10)
    # print("Shape of video embeddings:", video_embeddings.shape)
    
    # print("===> Test duration = 0.4, fps = 10")
    # with torch.no_grad():
    #     video_data_duration_0_4_fps_10 = video_data_duration_0_4_fps_10.permute(0, 2, 1, 3, 4)
    #     video_embeddings = video_swin_transformer(video_data_duration_0_4_fps_10)
    # print("Shape of video embeddings:", video_embeddings.shape)
    
if __name__ == "__main__":
    inference_example()