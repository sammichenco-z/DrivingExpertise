# This file contains the training loop for the contrastive learning process. It handles data loading, model training, and evaluation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import logging
import os
import yaml
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)
from data.real_car_dataset import RealCarDataset
from models.contrastive_model import ContrastiveModel

cfgs_dir = os.path.join(project_dir, 'cfgs')
with open(os.path.join(cfgs_dir, 'train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

config["model"]["version_identifier"] = config["version_identifier"]

# Ensure correct types
config["model"]["learning_rate"] = float(config["model"]["learning_rate"])
config["model"]["epochs"] = int(config["model"]["epochs"])
config["model"]["batch_size"] = int(config["model"]["batch_size"])

tensorboard_dir = os.path.join(project_dir, 'tensorboard', config['version_identifier'])
os.makedirs(tensorboard_dir, exist_ok=True)

log_dir = os.path.join(project_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"{config['version_identifier']}.log")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])
logger = logging.getLogger()

def evaluate_model(eval_dataset, model, batch_size, standard_eeg_channels):
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        total_loss = 0
        start_time = time.time()
        for i, (eeg_data, video_data) in enumerate(eval_loader):
            eeg_data, video_data = eeg_data.to(device), video_data.to(device)
            eeg_representation, video_representation = model(eeg_data, video_data, standard_eeg_channels)
            loss = model.contrastive_loss(eeg_representation, video_representation)
            total_loss += loss.item()
            
            if i % 20 == 0:
                # Calculate elapsed and remaining time
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (i + 1) * (len(eval_loader) - (i + 1))
                logger.info(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}/{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}] Iteration {i+1}/{len(eval_loader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(eval_loader)
    
    return avg_loss
    

def train_model(epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    train_dataset = RealCarDataset(config["data"], "train")
    standard_eeg_channels = train_dataset.obtain_standard_eeg_channel_indexes()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = RealCarDataset(config["data"], "val")

    # Initialize model, loss function, and optimizer
    model = ContrastiveModel(config["model"]).to(device)
    model.load_parameters()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Use tqdm to create a progress bar
        start_time = time.time()
        for i, (eeg_data, video_data) in enumerate(train_loader):
            eeg_data, video_data = eeg_data.to(device), video_data.to(device)

            # Forward pass
            optimizer.zero_grad()
            eeg_representation, video_representation = model(eeg_data, video_data, standard_eeg_channels)
            loss = model.contrastive_loss(eeg_representation, video_representation)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate elapsed and remaining time
            if i % 20 == 0:
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
                logger.info(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}/{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}] Iteration {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train_iter', loss.item(), epoch * len(train_loader) + i + 1)
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        # Log the average loss to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)
        
        # Evaluate the model on the validation set
        val_loss = evaluate_model(val_dataset, model, batch_size, standard_eeg_channels)
        logger.info(f'Validation Loss in epoch {epoch}: {avg_loss:.4f}')
        writer.add_scalar('Loss/eval_epoch', val_loss, epoch + 1)
        
        # save checkpoints
        model.save_epoch_ckpts(epoch)
        # only keep al most five checkpoints
        model.remove_old_epoch_ckpts(5)
        
    writer.close()
    
    model.save_final_ckpts()


if __name__ == "__main__":
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    learning_rate = config["model"]["learning_rate"]
    train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)