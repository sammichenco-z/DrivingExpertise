# This file contains the training loop for the contrastive learning process. It handles data loading, model training, and evaluation.

import os
import ast  # Add this import at the top of the file
import sys
import time
import yaml
import copy
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)
from models.contrastive_model import ContrastiveModel
from data.indoor_driving_eeg_dataset import IndoorDrivingEEGDataset

cfgs_dir = os.path.join(project_dir, 'cfgs')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(eval_dataset, model, batch_size, standard_eeg_channels, rank, world_size, logger):
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    model.eval()
    total_loss = 0
    n_mini_batches = len(eval_loader)
    with torch.no_grad():
        if rank == 0:
            start_time = time.time()
        for i, (eeg_data, video_data) in enumerate(eval_loader):
            eeg_data, video_data = eeg_data.to(rank), video_data.to(rank)
            eeg_representation, video_representation = model(eeg_data, video_data, standard_eeg_channels)
            loss = model.module.contrastive_loss(eeg_representation, video_representation)
            total_loss += loss.item()
            
            if rank == 0 and i % 10 == 0:
                # Calculate elapsed and remaining time
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (i + 1) * (len(eval_loader) - (i + 1))
                logger.info(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}/{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}] Iteration {i+1}/{len(eval_loader)}, Loss: {total_loss / (i + 1):.4f}")
    
    dist.barrier()
    total_loss_tensor = torch.tensor(total_loss).to(rank)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    n_mini_batches_tensor = torch.tensor(n_mini_batches).to(rank)
    dist.all_reduce(n_mini_batches_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / n_mini_batches_tensor.item()
    return avg_loss

def obtain_optimizer(model, learning_rate, weight_decay, lr_ratio):
    # lr_ratio means the ratio of learning rate between pre-trained part and fine-tuning (new added) part
    # lr of pretrained part = learning_rate * lr_ratio
    # for example:
    # lr_ratio=1 means the same learning rate for both parts
    # lr_ratio=0.1 means the learning rate for pre-trained part is 10 times smaller than the fine-tuning part
    video_encoder_pretrained_modules = ["swin_transformer"]
    video_encoder_head_modules = ["proj_norm", "proj"]
    eeg_encoder_pretrained_modules = ["cls_token", "pos_embed", "time_embed", "patch_embed", "pos_drop", "blocks", "norm"]
    eeg_encoder_head_modules = ["fc_norm", "head"]

    pretrained_params = []
    head_params = []

    # first process the video encoder
    for name, param in model.module.video_encoder.named_parameters():
        if any(name.startswith(module) for module in video_encoder_pretrained_modules):
            pretrained_params.append(param)
        elif any(name.startswith(module) for module in video_encoder_head_modules):
            head_params.append(param)
        else:
            raise ValueError(f"Unknown module in video encoder: {name}")
    # then process the eeg encoder
    for name, param in model.module.eeg_encoder.named_parameters():
        if any(name.startswith(module) for module in eeg_encoder_pretrained_modules):
            pretrained_params.append(param)
        elif any(name.startswith(module) for module in eeg_encoder_head_modules):
            head_params.append(param)
        else:
            raise ValueError(f"Unknown module in eeg encoder: {name}")
    # finally obtain the optimizer
    if weight_decay:
        assert weight_decay > 0, "weight decay should be greater than 0"
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': learning_rate * lr_ratio, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': learning_rate, 'weight_decay': weight_decay}
        ])
    else:
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': learning_rate * lr_ratio},
            {'params': head_params, 'lr': learning_rate}
        ])

    # check the number of parameters controled by the optimizer
    all_params = set(p for name, p in model.module.named_parameters() if name != 'temperature')
    assert set(pretrained_params + head_params) == all_params

    # log the number of parameters
    num_pretrained_params = sum(p.numel() for p in pretrained_params)
    num_head_params = sum(p.numel() for p in head_params)
    message = f"Pre-trained parameters: {num_pretrained_params}, Head parameters: {num_head_params}, ratio: {(num_head_params / num_pretrained_params)*100:.2f}%"

    return optimizer, message


def train_model_ddp(epochs, batch_size, learning_rate, weight_decay):
    assert torch.cuda.is_available(), "CUDA is not available"
    device_count = torch.cuda.device_count()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Start running on rank {rank}, world_size is {world_size}")
    
    
    # Initialize TensorBoard writer and logger
    tensorboard_dir = os.path.join(project_dir, 'tensorboard', config['version_identifier'])
    os.makedirs(tensorboard_dir, exist_ok=True)

    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{config['version_identifier']}.log")
    
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ])
        logger = logging.getLogger()
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        logger = None
    
    # Load dataset
    trainset_config = copy.deepcopy(config["data"])
    trainset_config['mode'] = 'training_set'
    train_dataset = IndoorDrivingEEGDataset(trainset_config)

    val_dataset_config = copy.deepcopy(config["data"])
    val_dataset_config['mode'] = 'validation_set'
    val_dataset = IndoorDrivingEEGDataset(val_dataset_config)

    train_sampler = DistributedSampler(train_dataset, num_replicas=device_count, rank=rank)
    # standard_eeg_channels = train_dataset.obtain_standard_eeg_channel_indexes()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    # obtain the standard eeg channels
    standard_eeg_channels = train_dataset.get_eeg_input_chans()
    standard_eeg_channels = standard_eeg_channels.to(rank)
    # Initialize model, loss function, and optimizer
    model = ContrastiveModel(config["model"]).to(rank)

    # load parameters of model
    # if rank == 0:
    eeg_encoder_missing_keys, eeg_encoder_unexpected_keys, video_encoder_missing_keys, video_encoder_unexpected_keys = model.load_parameters()
    if rank == 0:
        logger.info(f"Missing keys in EEG encoder: {eeg_encoder_missing_keys}")
        logger.info(f"Unexpected keys in EEG encoder: {eeg_encoder_unexpected_keys}")
        logger.info(f"Missing keys in Video encoder: {video_encoder_missing_keys}")
        logger.info(f"Unexpected keys in Video encoder: {video_encoder_unexpected_keys}")
    dist.barrier()
    
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if config["model"]["fine_tuning_policy"] == "all_trainable":
        lr_ratio = config["model"]["lr_ratio"]
        optimizer, message = obtain_optimizer(ddp_model, learning_rate, weight_decay, lr_ratio=lr_ratio)
    elif config["model"]["fine_tuning_policy"] == "missing_trainable":
        if weight_decay:
            assert weight_decay > 0, "weight decay should be greater than 0"
            optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown fine-tuning policy: {config['model']['fine_tuning_policy']}")
    
    if rank == 0:
        logger.info(f"contrastive model: \n {ddp_model.module}")
        logger.info(f"Using weight decay: {weight_decay}")
        logger.info(optimizer)
        ddp_model.module.check_parameters(logger)
        if message:
            logger.info(message)

    # current_policy = None
    for epoch in range(epochs):
        ddp_model.train()
        total_loss = 0
        n_mini_batches = len(train_loader)

        # two-stage fine-tuning
        # if epoch < 2:
        #     new_policy = "missing_trainable"
        # else:
        #     new_policy = "all_trainable"
        
        # if new_policy != current_policy:
        #     ddp_model.module.change_fine_tuning_policy(new_policy)
        #     current_policy = new_policy
        #     if rank == 0:
        #         ddp_model.module.check_parameters(logger)

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            if rank == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {current_lr}")
                writer.add_scalar("Learning Rate", current_lr, epoch)
        
        # Use tqdm to create a progress bar
        if rank == 0:
            start_time = time.time()
        for i, (eeg_data, video_data) in enumerate(train_loader):
            eeg_data, video_data = eeg_data.to(rank), video_data.to(rank)

            # Forward pass
            optimizer.zero_grad()
            eeg_representation, video_representation = ddp_model(eeg_data, video_data, standard_eeg_channels)
            loss = ddp_model.module.contrastive_loss(eeg_representation, video_representation)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if rank == 0:
                # Calculate elapsed and remaining time
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
                logger.info(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}/{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}] Iteration {i+1}/{len(train_loader)}, Loss: {total_loss / (i + 1):.4f}")
                writer.add_scalar('Loss/train_iter', loss.item(), epoch * len(train_loader) + i + 1)
        
        dist.barrier()
        total_loss_tensor = torch.tensor(total_loss).to(rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        n_mini_batches_tensor = torch.tensor(n_mini_batches).to(rank)
        dist.all_reduce(n_mini_batches_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / n_mini_batches_tensor.item()
        
        if rank == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            # Log the average loss to TensorBoard
            writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)

            # save checkpoints
            ddp_model.module.save_epoch_ckpts(epoch)
            # only keep al most five checkpoints
            ddp_model.module.remove_old_epoch_ckpts(5)
        dist.barrier()
        
        # Evaluate the model on the validation set
        eval_loss = evaluate_model(val_dataset, ddp_model, batch_size, standard_eeg_channels, rank, world_size, logger)
        if rank == 0:
            logger.info(f'Validation Loss in epoch {epoch + 1}: {eval_loss:.4f}')
            writer.add_scalar('Loss/eval_epoch', eval_loss, epoch + 1)

    if rank == 0:
        ddp_model.module.save_final_ckpts()
        writer.close()
    dist.barrier()
    
    # Clean up DDP
    dist.destroy_process_group()


def test_ddp_train_model(epochs, batch_size, learning_rate):
    assert torch.cuda.is_available(), "CUDA is not available"
    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    print(f"Start running on rank {rank}")
    
    dist.destroy_process_group()
    print(f"End running on rank {rank}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra_identifier', type=str, default='debug_x_unknown_default', help='the extra identifier for the version identifier of the model')
    parser.add_argument('--dropout', type=float, default=0.1, help='the value of p of dropout layers for your model')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='the learning rate of the model')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay of the model')
    parser.add_argument('--fine_tuning_policy', type=str, choices=["all_trainable", "missing_trainable"], help='the fine-tuning policy of the model')
    parser.add_argument('--lr_ratio', type=float, default=1e-2, help='the lr ratio of the pretrained part and the fine-tuning part')
    parser.add_argument('--config', type=str, default=os.path.join(cfgs_dir, 'train_config.yaml'), help='the config file of the model')
    parser.add_argument('--video_type', type=str, default='all', choices=['real', 'virtual', 'all'], help='the type of video data')
    parser.add_argument('--subject_type', type=str, default='all', choices=['expert', 'novice', 'all'], help='the type of subject data')
    parser.add_argument('--eeg_area', type=str, default="['visual', 'spatial', 'semantic', 'decision']", help='the area of the EEG data (as a Python list-like string)')

    arg = parser.parse_args()

    # Convert the eeg_area string to a Python list
    arg.eeg_area = ast.literal_eval(arg.eeg_area)

    return arg

def initialize_parameters(arg):
    # load the config file
    print(f"Loading config file: {arg.config}")
    with open(arg.config, 'r') as f:
        global config
        config = yaml.safe_load(f)

    config["model"]["extra_identifier"] = arg.extra_identifier
    config["model"]["dropout"] = arg.dropout
    config["model"]["learning_rate"] = arg.learning_rate
    config["model"]["weight_decay"] = arg.weight_decay
    config["model"]["fine_tuning_policy"] = arg.fine_tuning_policy
    config["model"]["lr_ratio"] = arg.lr_ratio
    
    eeg_area_str = "_".join(arg.eeg_area)
    version_identifier = f"indoor_eeg_dataset_{arg.video_type}_video+{arg.subject_type}_subject+eeg_{eeg_area_str}+{config['model']['extra_identifier']}"
    config["version_identifier"] = version_identifier
    config["data"]["video_type"] = arg.video_type
    config["data"]["subject_type"] = arg.subject_type
    config["data"]["eeg_area"] = arg.eeg_area

    # Ensure correct types
    config["model"]["learning_rate"] = float(config["model"]["learning_rate"])
    config["model"]["epochs"] = int(config["model"]["epochs"])
    config["model"]["batch_size"] = int(config["model"]["batch_size"])

    config["model"]["version_identifier"] = version_identifier
    config["model"]["temperature_ckpt_path"] = f"ckpts/{version_identifier}/temperature_ft_1.safetensors"
    config["model"]["eeg_encoder"]["ckpt_save_path"] = f"ckpts/{version_identifier}/eeg_encoder_ft_1.safetensors"
    config["model"]["video_encoder"]["ckpt_save_path"] = f"ckpts/{version_identifier}/video_encoder_ft_1.safetensors"

if __name__ == "__main__":
    arg = parse_arguments()
    initialize_parameters(arg)
    seed = config["seed"]
    set_random_seed(seed)
    
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    learning_rate = config["model"]["learning_rate"]
    weight_decay = config["model"]["weight_decay"]
    
    train_model_ddp(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
    # test_ddp_train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    # print(config["model"]["video_encoder"]["ckpt_save_path"])
    
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29001 /home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py --extra_identifier debug_11_bs_16+epoch_120+lr_1e-5+dropout_0.1 --dropout 0.1