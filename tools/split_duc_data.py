""" Utility function to split the InLoc DUC data into train and val / test """

import os
import pdb
import glob
import tqdm
import shutil
import random
import argparse


def copy_to_dir(dst_dir, file_list, selected_ids):
    os.makedirs(dst_dir, exist_ok=True)

    for x in tqdm.tqdm(selected_ids):
        color_fname = file_list[x]
        shutil.copy(color_fname, dst_dir)
        depth_fname = color_fname.replace('color.png', 'depth.png')
        shutil.copy(depth_fname, dst_dir)
        pose_fname = color_fname.replace('color.png', 'pose.txt')
        shutil.copy(pose_fname, dst_dir)


if __name__ == '__main__':
    parser =  argparse.ArgumentParser(
        description='Args for splitting the DUC dataset.'
                    'This will copy files and make the train and val folders')
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    file_list = sorted(glob.glob(args.src_dir + '/*color.png'))

    all_indices = set(range(len(file_list)))
    sel_val = set(random.sample(all_indices, int(len(file_list) * 0.1)))
    sel_train = all_indices - sel_val
    
    val_dir = os.path.join(args.dst_dir, 'val')
    copy_to_dir(val_dir, file_list, sel_val)
    train_dir = os.path.join(args.dst_dir, 'train')
    copy_to_dir(train_dir, file_list, sel_train)


        
    
    
    
