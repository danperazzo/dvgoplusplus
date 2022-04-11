import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo
from lib.load_data import load_data
from lib.utils import intersect_sphere
from lib.load_llff import imread
import lib.utils as utils
import cv2
import numpy as np




def load_many_images(global_path,image_dic):

    dvg_list = []
    for subdir, dirs, files in os.walk(global_path + image_dic):
        #print(files)
        print(dirs)
        list_for_this = []
        for file in files:
            filepath = subdir + os.sep + file
            #print(filepath)

            if filepath.endswith(".png"):
                image = imread(filepath)/255.
                list_for_this.append(image)
             #   print(image.shape)
              #  print(filepath)
        if  list_for_this:
            dvg_list.append(list_for_this)
    return dvg_list


def get_error_images(loss_func,imgs_dir,target_dir):
    score_alls = []
    for dir_scene_imgs, dir_scene_target in zip(imgs_dir,target_dir):
      score_array = []
      #print(len(dir_scene_target))
      for img_orig,img_tgt in zip(dir_scene_imgs, dir_scene_target):

          #print(img_orig.shape)
          

          score_array.append(loss_func(img_orig,img_tgt))

      score_alls.append(np.array(score_array).mean())
    return np.array(score_alls)

def mask_images(imgs_dir,mask_dir):
    masked_img_all = []
    for dir_scene_imgs, dir_scene_mask in zip(imgs_dir,mask_dir):
      masked_img_scene = []
      for img_orig,img_mask in zip(dir_scene_imgs, dir_scene_mask):

          #print(img_orig.shape)

          masked_img_scene.append(img_orig*img_mask[...,None])

      masked_img_all.append(masked_img_scene)
    return masked_img_all

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def create_mask_from_images(global_path,image_dic):

    dvg_list = []
    for subdir, dirs, files in os.walk(global_path + image_dic):
        #print(files)
        list_for_this = []
        for file in files:
            filepath = subdir + os.sep + file
            #print(filepath)

            if filepath.endswith(".png"):
                image = rgb2gray(imread(filepath)/255.)
                image[image>0.09] = 1.0
                list_for_this.append(image)
             #   print(image.shape)
              #  print(filepath)
        if  list_for_this:
            dvg_list.append(list_for_this)
    return dvg_list


print("Helloooo")

path = "C:/Users/Temp/Desktop/DirectVoxGO/Results_texts/"

dvg_p_mask = create_mask_from_images(path,"HashDirectVoxGo++/llff_foreground/")
plenoxel_dict = create_mask_from_images(path,"Plenoxels/images_fg/")

PSNR = lambda x,y : -10. * np.log10(np.mean(np.square(x - y)))
ssim = lambda x,y :utils.rgb_ssim(x, y, max_val=1)
lpips = lambda x,y :utils.rgb_lpips(np.float32(x), np.float32(y), net_name='vgg', device=torch.device("cpu"))


mask_errors = get_error_images(PSNR,plenoxel_dict,dvg_p_mask)

print("Mask Error:")
print(mask_errors)


mine_dic = load_many_images(path,"HashDirectVoxGo++/llff_image_all/")
gt_dic = load_many_images(path,"Ground_truth/")
plenoxel_dict = load_many_images(path,"Plenoxels/images_all/")
dvg_dic = load_many_images(path,"DirectVoxGo/llff_all_image/")



mask_mine_dic = mask_images(mine_dic,dvg_p_mask)
mask_img_gt = mask_images(gt_dic,dvg_p_mask)
mask_img_plenoxel = mask_images(plenoxel_dict,dvg_p_mask)
mask_img_dvg = mask_images(dvg_dic,dvg_p_mask)


mine_masked_psnr = get_error_images(PSNR,mask_mine_dic,mask_img_gt)
mine_masked_ssim = get_error_images(ssim,mask_mine_dic,mask_img_gt)
mine_masked_lpips = get_error_images(lpips,mask_mine_dic,mask_img_gt)


print("mine:")
print(mine_masked_psnr,mine_masked_psnr.mean())
print(mine_masked_ssim,mine_masked_ssim.mean())
print(mine_masked_lpips,mine_masked_lpips.mean())


print("directvoxgo:")
dvgo_masked_psnr = get_error_images(PSNR,mask_img_dvg,mask_img_gt)
dvgo_masked_ssim = get_error_images(ssim,mask_img_dvg,mask_img_gt)
dvgo_masked_lpips = get_error_images(lpips,mask_img_dvg,mask_img_gt)


print(dvgo_masked_psnr,dvgo_masked_psnr.mean())
print(dvgo_masked_ssim,dvgo_masked_ssim.mean())
print(dvgo_masked_lpips,dvgo_masked_lpips.mean())


plenoxel_masked_psnr = get_error_images(PSNR,mask_img_plenoxel,mask_img_gt)
plenoxel_masked_ssim = get_error_images(ssim,mask_img_plenoxel,mask_img_gt)
plenoxel_masked_lpips = get_error_images(lpips,mask_img_plenoxel,mask_img_gt)

print("plenoxel:")
print(plenoxel_masked_psnr,plenoxel_masked_psnr.mean())
print(plenoxel_masked_ssim,plenoxel_masked_ssim.mean())
print(plenoxel_masked_lpips,plenoxel_masked_lpips.mean())






#print(len(dvg_dic))
#print(len(plenoxel_dict))



