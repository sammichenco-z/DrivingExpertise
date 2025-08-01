# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook,
                         build_runner, )
from mmcv.optims import build_optimizer
from mmcv.utils import build_from_cfg

from mmcv.core import EvalHook

from mmcv.datasets import (build_dataset, replace_ImageToTensor)
from mmcv.utils import get_root_logger, get_dist_info
import time
import os.path as osp
from mmcv.datasets import build_dataloader
from mmcv.core.evaluation.eval_hooks import CustomDistEvalHook
from adzoo.bevformer.apis.test import custom_multi_gpu_test

def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #assert len(dataset)==1s
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        ) for ds in dataset
    ]
    
    # import ipdb
    # ipdb.set_trace()
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = DistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = DataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = DataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, test_fn=custom_multi_gpu_test, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        #runner.load_checkpoint(cfg.load_from,strict=False)
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = 'module.' + k  # 加上前缀
            new_state_dict[new_key] = v

        model_state = runner.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for k in model_state.keys():
            if k in new_state_dict:
                if model_state[k].shape == new_state_dict[k].shape:
                    # 同名且shape相同，复制权重
                    model_state[k].copy_(new_state_dict[k])
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)
            else:
                skipped_keys.append(k)

        print(f"成功加载的参数({len(loaded_keys)}):")
        # for k in loaded_keys:
        #     print(k)
        print(f"\n未加载（形状不匹配或缺失）参数({len(skipped_keys)}):")
        # breakpoint()
        # for k in skipped_keys:
        #     print(k)

        #runner.model.load_state_dict(state_dict, strict=False)
        # msg = runner.model.load_state_dict(new_state_dict, strict=False)
        # model_keys = set(runner.model.state_dict().keys())
        # ckpt_keys = set(new_state_dict.keys())
        # loaded_keys = model_keys.intersection(ckpt_keys).difference(set(msg.missing_keys))
        # print(f"成功加载的参数 ({len(loaded_keys)})")
        # for k in sorted(loaded_keys):
        #     print(k)
        # print("Checkpoint keys:")
        # for k in sorted(ckpt_keys)[:10]:
        #     print(k)
        # print("Model keys:")
        # for k in sorted(model_keys)[:10]:
        #     print(k)
        # print(f"\n缺失未加载的参数 ({len(msg.missing_keys)})：")
        # for k in msg.missing_keys:
        #     print(k)
        # print(f"\n权重文件中多余未匹配的参数 ({len(msg.unexpected_keys)})：")
        # for k in msg.unexpected_keys:
        #     print(k)
    runner.run(data_loaders, cfg.workflow)
        # breakpoint()


