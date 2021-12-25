import os
import glob
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_tankstemple_data(basedir,args):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))

    all_poses = []
    all_imgs = []
    i_split = [[], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))

        img = (imageio.imread(rgb_path) / 255.).astype(np.float32)
        dim_tgt = ( img.shape[1]//args.factor, img.shape[0]//args.factor     )
        img_reshape = cv2.resize(img, dim_tgt, interpolation = cv2.INTER_AREA)

        all_imgs.append(img_reshape)
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    i_split.append(i_split[-1])

    path_intrinsics = os.path.join(basedir, 'intrinsics.txt')
    H, W = imgs[0].shape[:2]
    K = np.loadtxt(path_intrinsics)
    K[:2,:3] = K[:2,:3] / args.factor # diminished the images by the factor for resizing
    focal = float(K[0,0])

    path_traj = os.path.join(basedir, 'test_traj.txt')
    if os.path.isfile(path_traj):
        render_poses = torch.Tensor(np.loadtxt(path_traj).reshape(-1,4,4).astype(np.float32))
    else:
        render_poses = poses[i_split[-1]]

    return imgs, poses, render_poses, [H, W, focal], K, i_split

