"""
pytorch data loader for the synthetic dataset created by AirSim
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data

import sys
sys.path.append("../")
import pickle

from common.vis_utils import show_batch, show_stereo_batch
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from common.pose_utils import process_poses_from_quartenion
from dataset_loaders.utils import load_image


class AirSim(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7, real=False,
                 skip_images=False, simulate_noise='None', **kwargs):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/deepslam_data/7Scenes'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param mode: 0: just color image
        :param real: If True, load poses from SLAM/integration of VO
        :param skip_images: If True, skip loading images and return None instead
        :Param simulate_noise: If True, train_noise.txt is used which simulates bad colmap
        """
        self.skip = kwargs.pop('skip', 1)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        data_dir = osp.join('..', 'data', 'AirSim')

        # decide which sequences to use
        data_root = osp.join(data_path, scene)
        train_or_val = "train" if train else "val"
        if train:
            if simulate_noise == 'None':
                airsim_rec_file = f'{train_or_val}_clean_skip_{self.skip}.txt'
            else:
                airsim_rec_file = f'{train_or_val}_noisy_{simulate_noise}_skip_{self.skip}.txt' 
        else:
            airsim_rec_file = f'{train_or_val}.txt'

        # read poses and collect image names
        self.c_imgs = []
        vo_stats = []
        pairs_txt = osp.join(data_root, train_or_val, airsim_rec_file)

        t_gt, q_gt = [], []
        self.relative_imu = []
        with open(pairs_txt, 'r') as f:
            for line in f.readlines()[1::]:
                chunks = line.rstrip().split(' ')
                self.c_imgs.append(osp.join(data_root, train_or_val, "images", chunks[0]))
                t_gt.append(np.array([float(chunks[1]), float(chunks[2]), float(chunks[3])]))
                q_gt.append(np.array([float(chunks[4]), float(chunks[5]), float(chunks[6]), float(chunks[7])]))
                self.relative_imu.append(torch.tensor([float(chunks[8]), float(chunks[9]), float(chunks[10]), 
                                                       float(chunks[11]), float(chunks[12]),float(chunks[13]),float(chunks[14])]))

        vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1} # No alignment needed
        t_gt = np.vstack(t_gt)
        q_gt = np.vstack(q_gt)
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')

        if train and not real:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename) # Currently not normalising

        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        pss = process_poses_from_quartenion(t_in=t_gt, q_in= q_gt, mean_t=mean_t, std_t=std_t,
                                            align_R=vo_stats['R'], align_t=vo_stats['t'],
                                            align_s=vo_stats['s'])
        self.poses = np.vstack((self.poses, pss))


    def __getitem__(self, index):
        imu_rel_pose = self.relative_imu[index]
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            if self.mode == 0:
                img = None
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img = self.transform(img)

        return img, pose, imu_rel_pose

    def __len__(self):
        return self.poses.shape[0]


def main():
    """
    visualizes the dataset
    """
    seq = 'building_final'
    mode = 0
    num_workers = 0
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dset = AirSim(seq, 'D:/Imperial/FYP/captured_data/airsim_drone_mode', True, transform,
                       mode=mode, simulate_noise=False)
    print('Loaded AirSim sequence {:s}, length = {:d}'.format(seq,
                                                               len(dset)))

    data_loader = data.DataLoader(dset, batch_size=8, shuffle=True,
                                  num_workers=num_workers)

    batch_count = 0
    N = 2
    for batch in data_loader:
        print('Minibatch {:d}'.format(batch_count))
        print(batch[0].shape)
        if mode < 2:
            show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
        elif mode == 2:
            lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
            rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
            show_stereo_batch(lb, rb)

        batch_count += 1
        if batch_count >= N:
            break


if __name__ == '__main__':
    main()
