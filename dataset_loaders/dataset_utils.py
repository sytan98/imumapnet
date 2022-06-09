import os.path as osp
import numpy as np

import sys
sys.path.append("../")

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data.dataloader import default_collate

from airsim import AirSim

def safe_collate(batch):
    """
    Collate function for DataLoader that filters out None's
    :param batch: minibatch
    :return: minibatch filtered for None's
    """
    # fix to work with python 3
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def get_dataset_mean_std(train : bool):
    seq = 'building_10fps'
    mode = 0
    num_workers = 0

    data_dir = osp.join('..', 'data', 'AirSim')
    stats_file = osp.join(data_dir, 'stats.txt')
    stats = np.loadtxt(stats_file)
    print(stats)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats[0], 
                             std= stats[1])
    ])
    dset = AirSim(seq, 'D:/Imperial/FYP/captured_data/airsim_drone_mode', train, transform,
                       mode=mode, simulate_noise=False)
    print('Loaded AirSim sequence {:s}, length = {:d}'.format(seq,
                                                               len(dset)))

    data_loader = torch.utils.data.DataLoader(dset,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=num_workers, 
                                              pin_memory=True,
                                              collate_fn=safe_collate)

    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    length = 0
    for idx, batch in enumerate(data_loader):
        data = batch[0]
        # print(mini_batch['img1'].shape)
        psum    += data.sum(dim = [0, 2, 3])
        psum_sq += (data ** 2).sum(dim = [0, 2, 3])

        length += data.shape[0]
    # pixel count
    print(length)
    count = length * 224 * 224

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

def resize_image_and_save(train : bool):
    train_or_val = 'train' if train else 'val'
    seq = 'building_10fps'
    mode = 0
    num_workers = 0
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    dset = AirSim(seq, 'D:/Imperial/FYP/captured_data/airsim_drone_mode', train, transform,
                       mode=mode, simulate_noise=False)
    print('Loaded AirSim sequence {:s}, length = {:d}'.format(seq,
                                                               len(dset)))

    data_loader = torch.utils.data.DataLoader(dset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=num_workers, 
                                              pin_memory=True,
                                              collate_fn=safe_collate)


    for idx, batch in enumerate(data_loader):
        data = batch[0][0]
        save_image(data, f'D:/Imperial/FYP/captured_data/airsim_drone_mode/{seq}/{train_or_val}/resized_images/{idx * 10}.png')

if __name__ == '__main__':
    get_dataset_mean_std(True)
    # resize_image_and_save(False)
