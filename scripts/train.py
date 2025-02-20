"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
Main training script for MapNet
"""
import set_paths

import os.path as osp
import json
import torch

import numpy as np
import argparse
import configparser
from torch import nn
from torchvision import transforms, models

from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import (MapNetWithIMUCriterionSeparate, PoseNetCriterion, MapNetCriterion, MapNetWithIMUCriterion,
                              MapNetOnlineCriterion)
from models.posenet import PoseNet, MapNet, PoseNetWithImuOutput
from dataset_loaders.composite import MF, MFOnline

from dataset_loaders.seven_scenes import SevenScenes
from dataset_loaders.inloc import InLoc


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                                 'MapNet variants')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str,
                        choices=('AirSim', '7Scenes', 'InLoc', 'InLocRes', 'RobotCar'),
                        help='Dataset')
    parser.add_argument('--imu_mode', type=str, default='None',
                        choices=('None', 'Average', 'Separate', 'Position', 'Orientation'),
                        help='imu incorporation')
    parser.add_argument('--noisy_training',  type=str,
                        choices=('None', 'v1', 'v2'),
                        help='Use noisy training set')
    parser.add_argument('--loss',  type=str,
                        choices=('L1', 'SmoothL1', 'MSE'),
                        help='choose loss function')
    parser.add_argument('--average_method', type=str, default='simple',
                        choices=('Simple', 'Interpolate'),
                        help='Method to average predictions for imu_mode "Separate"')
    parser.add_argument('--scene', type=str, help='Scene name')
    parser.add_argument('--config_file', type=str, help='configuration file')
    parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++'),
                        help='Model to train')
    parser.add_argument('--device', type=str, default='0',
                        help='value to be set to $CUDA_VISIBLE_DEVICES')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
                        default=None)
    parser.add_argument('--learn_beta', action='store_true',
                        help='Learn the weight of translation loss')
    parser.add_argument('--learn_gamma', action='store_true',
                        help='Learn the weight of rotation loss')
    parser.add_argument('--resume_optim', action='store_true',
                        help='Resume optimization (only effective if a checkpoint is given')
    parser.add_argument('--suffix', type=str, default='',
                        help='Experiment name suffix (as is)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    settings = configparser.ConfigParser()
    args = parse_arguments()
    with open(args.config_file, 'r') as f:
        settings.read_file(f)
    section = settings['optimization']
    optim_config = {k: json.loads(v) for k, v in section.items() if k != 'opt'}
    opt_method = section['opt']
    lr = optim_config.pop('lr')
    weight_decay = optim_config.pop('weight_decay')

    section = settings['hyperparameters']
    dropout = section.getfloat('dropout')
    color_jitter = section.getfloat('color_jitter', 0)
    sax = 0.0
    saq = section.getfloat('beta')
    if args.model.find('mapnet') >= 0:
        skip = section.getint('skip')
        real = section.getboolean('real')
        variable_skip = section.getboolean('variable_skip')
        srx = 0.0
        srq = section.getfloat('gamma')
        steps = section.getint('steps')
        if args.imu_mode != 'None':
            imu_weight = section.getfloat('imu_loss_weight')
    if args.model.find('++') >= 0:
        vo_lib = section.get('vo_lib', 'orbslam')
        print('Using {:s} VO'.format(vo_lib))

    section = settings['training']
    seed = section.getint('seed')

    # model
    feature_extractor = models.resnet34(pretrained=True)
    if args.imu_mode == "Separate":
        posenet = PoseNetWithImuOutput(feature_extractor, droprate=dropout, pretrained=True,
                                       filter_nans=(args.model == 'mapnet++'))
    else:
        posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                          filter_nans=(args.model == 'mapnet++'))
    if args.model == 'posenet':
        model = posenet
    elif args.model.find('mapnet') >= 0:
        model = MapNet(mapnet=posenet)
    else:
        raise NotImplementedError

    # loss function
    if args.model == 'posenet':
        train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=args.learn_beta)
        val_criterion = PoseNetCriterion()
    elif args.model.find('mapnet') >= 0:
        kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=args.learn_beta,
                      learn_gamma=args.learn_gamma)
        if args.loss == "SmoothL1":
            kwargs = dict(kwargs, t_loss_fn=nn.SmoothL1Loss(), q_loss_fn=nn.SmoothL1Loss())
        elif args.loss == "MSE":
            kwargs = dict(kwargs, t_loss_fn=nn.MSELoss(), q_loss_fn=nn.MSELoss())
        if args.model.find('++') >= 0:
            kwargs = dict(kwargs, gps_mode=(vo_lib == 'gps'))
            train_criterion = MapNetOnlineCriterion(**kwargs)
            val_criterion = MapNetOnlineCriterion()
        else:
            if args.imu_mode == 'None':
                train_criterion = MapNetCriterion(**kwargs)
            elif args.imu_mode == 'Average':
                train_criterion = MapNetWithIMUCriterion(**kwargs, srx_imu= srx, srq_imu= srq, imu_weight=imu_weight)
            elif args.imu_mode == 'Separate':
                train_criterion = MapNetWithIMUCriterionSeparate(**kwargs, srx_imu= srx, srq_imu= srq)
            val_criterion = MapNetCriterion()
    else:
        raise NotImplementedError

    # optimizer
    param_list = [{'params': model.parameters()}]
    if args.learn_beta and hasattr(train_criterion, 'sax') and \
            hasattr(train_criterion, 'saq'):
        param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
    if args.learn_gamma and hasattr(train_criterion, 'srx') and \
            hasattr(train_criterion, 'srq'):
        param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
    if args.learn_gamma and hasattr(train_criterion, 'srx_imu') and \
            hasattr(train_criterion, 'srq_imu'):
        param_list.append({'params': [train_criterion.srx_imu, train_criterion.srq_imu]})
    optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
                          weight_decay=weight_decay, **optim_config)

    data_dir = osp.join('..', 'data', args.dataset)
    stats_file = osp.join(data_dir, args.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)

    crop_size_file = osp.join(data_dir, 'crop_size.txt')
    crop_size = tuple(np.loadtxt(crop_size_file).astype(int))

    # transformers
    tforms = []
    if color_jitter > 0:
        assert color_jitter <= 1.0
        print('Using ColorJitter data augmentation')
        tforms.append(transforms.ColorJitter(brightness=color_jitter,
                                             contrast=color_jitter, saturation=color_jitter, hue=0.5))
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=stats[1]))
    data_transform = transforms.Compose(tforms)
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    # datasets
    data_dir = args.data_dir 
    kwargs = dict(scene=args.scene, data_path=data_dir, transform=data_transform,
                  target_transform=target_transform, seed=seed)
    # if model being tested is posenet
    if args.model == 'posenet':
        if args.dataset == '7Scenes':
            train_set = SevenScenes(train=True, **kwargs)
            val_set = SevenScenes(train=False, **kwargs)
        elif args.dataset == 'RobotCar':
            # Ugly, but requires robotcar_sdk
            from dataset_loaders.robotcar import RobotCar
            train_set = RobotCar(train=True, **kwargs)
            val_set = RobotCar(train=False, **kwargs)
        elif args.dataset == 'InLoc':
            train_set = InLoc(train=True, **kwargs)
            val_set = InLoc(train=False, **kwargs)
        elif args.dataset == 'InLocRes':
            train_set = InLoc(train=True, **kwargs)
            val_set = InLoc(train=False, **kwargs)
        else:
            raise NotImplementedError
    
    # MapNet data loaders managed here
    elif args.model.find('mapnet') >= 0:
        kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
                      variable_skip=variable_skip, simulate_noise=args.noisy_training)
        if args.model.find('++') >= 0:
            train_set = MFOnline(vo_lib=vo_lib, gps_mode=(vo_lib == 'gps'), **kwargs)
            val_set = None
        else:
            if args.imu_mode == 'None':
                train_set = MF(train=True, real=real, **kwargs)
                val_set = MF(train=False, real=real, **kwargs)
            else: 
                train_set = MF(train=True, real=real, include_imu=True, **kwargs)
                val_set = MF(train=False, real=real, include_imu=True, **kwargs)
    else:
        raise NotImplementedError

    # trainer
    config_name = args.config_file.split('/')[-1]
    config_name = config_name.split('.')[0]
    experiment_name = '{:s}_{:s}_{:s}_{:s}_imu_{:s}_noisy_{:s}_skip_{:s}'.format(args.dataset, args.scene,
                                                                                 args.model, config_name, 
                                                                                 args.imu_mode, args.noisy_training,
                                                                                 skip)
    if args.learn_beta:
        experiment_name = '{:s}_learn_beta'.format(experiment_name)
    if args.learn_gamma:
        experiment_name = '{:s}_learn_gamma'.format(experiment_name)
    experiment_name += args.suffix
    trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                      experiment_name, train_set, val_set, device=args.device, 
                      imu_mode=args.imu_mode, average_method=args.average_method,
                      checkpoint_file=args.checkpoint,
                      resume_optim=args.resume_optim, val_criterion=val_criterion,)
    lstm = args.model == 'vidloc'
    trainer.train_val(lstm=lstm)
