import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_erosion

from utils.parser import set_parser
from models.Unet3Du import UNet3D

def get_boundary_map(volume):
    filter_data = gaussian_filter(volume, 1)
    threshold = 0.4

    binary_mask = filter_data > threshold
 
    eroded_mask = binary_erosion(binary_mask)
    boundary_map = binary_mask.astype(int) - eroded_mask.astype(int)

    return boundary_map

def thresholding(volume, threshold):
    copy_volume = volume.copy()

    copy_volume[copy_volume > threshold] = 1
    copy_volume[copy_volume <= threshold] = 0

    return copy_volume

def dice_coefficient(prediction, target):

    intersection = np.sum(prediction * target)
    dice = (2. * intersection) / (np.sum(prediction) + np.sum(target))
    return dice

def iou_coefficient(prediction, target):
    interesction = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou = np.sum(interesction) / np.sum(union)

    return iou

def get_volume(path):
    volume = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(volume)

    volume = np.transpose(volume, (1,0, 2))
    volume = np.rot90(volume, 2)

    return volume

def get_path():
    test_CT = pd.read_excel(f"F:\\HIPPO\\test.xlsx")['CT']
    test_hippo = pd.read_excel(f"F:\\HIPPO\\test.xlsx")['HIPPO']

    return test_CT, test_hippo

def minmaxnormalize(volume):
    copy_volume = volume.copy()

    s = np.min(copy_volume)
    b = np.max(copy_volume)

    return (copy_volume - s) / (b - s)

def adjust_window(args, volume):
    copy_volume = volume.copy()

    window_min = args.window_min
    window_max = args.window_max

    copy_volume[copy_volume <= window_min] = window_min
    copy_volume[copy_volume >= window_max] = window_max

    return copy_volume 

def get_binary_volume(volume):
    copy_volume = volume.copy()

    copy_volume[copy_volume != 0] = 1

    return copy_volume

def get__centroid(volume):
    non_zero_index = np.transpose(np.nonzero(volume))
    centroid = np.mean(non_zero_index, axis=0)

    return centroid 

def crop__volume(volume, crop_size):
    centroid = get__centroid(volume)

    start_x = int(centroid[0])
    start_y = int(centroid[1])
    start_z = int(centroid[2])

    copy_volume = volume.copy()

    cropped_volume = copy_volume[start_x-crop_size[0]//2:start_x+crop_size[0]//2, start_y-crop_size[1]//2:start_y+crop_size[1]//2, start_z-crop_size[2]//2:start_z+crop_size[2]//2]
    
    return cropped_volume

def get_boundary_map(volume):
    filter_data = gaussian_filter(volume, 1)
    threshold = 0.4

    binary_mask = filter_data > threshold

    eroded_mask = binary_erosion(binary_mask)
    boundary_map = binary_mask.astype(int) - eroded_mask.astype(int)

    return boundary_map

def data_preprocessing(args, ctpath, hippopath):
    ct = get_volume(ctpath)
    hippo = get_volume(hippopath)

    ct = adjust_window(args, ct)
    ct = minmaxnormalize(ct)

    hippo = get_binary_volume(hippo)
    boundary = get_boundary_map(hippo)

    ct = crop__volume(ct, (64,128,128))
    hippo = crop__volume(hippo, (64,128,128))
    boundary = crop__volume(boundary, (64,128,128))

    return torch.from_numpy(ct).unsqueeze(0).unsqueeze(0), torch.from_numpy(hippo).unsqueeze(0).unsqueeze(0), torch.from_numpy(boundary).unsqueeze(0).unsqueeze(0)

if __name__=='__main__':
    args = set_parser()

    device = args.device

    test_CT, test_hippo = get_path()

    datasize = len(test_CT)
    print(f"Test data size : {datasize}")

    model = UNet3D(1, 1).to(device)
    model = nn.DataParallel(model).to(device)
    
    model_save_path = f"F:\\HIPPO\\{args.date}\\model_parameters"
    filename = f"{args.model}_e.pt"
    model.load_state_dict(torch.load(os.path.join(model_save_path, filename)))
    print(f"Model Load Complete!")

    print(f"Start Test!")
    total_dice0, total_iou0 = 0,0
    total_dice1, total_iou1 = 0,0
    total_dice2, total_iou2 = 0,0
    total_dice3, total_iou3 = 0,0
    total_dice4, total_iou4 = 0,0
    total_dice5, total_iou5 = 0,0
    total_dice6, total_iou6 = 0,0
    total_dice7, total_iou7 = 0,0
    total_dice8, total_iou8 = 0,0
    total_dice9, total_iou9 = 0,0

    for i in range(datasize):
        ct, hippo, edge = data_preprocessing(args, test_CT[i], test_hippo[i])
        folder = test_CT[i].split('\\')[3]

        pred, e = model(ct.to(device).float(), edge.to(device).float())

        pred = np.clip(pred.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        # o2 = np.clip(o2.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        # o3 = np.clip(o3.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        # o4 = np.clip(o4.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)

        # o2_b = np.clip(o2_b.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        # o3_b = np.clip(o3_b.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        # o4_b = np.clip(o4_b.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        #b = b.squeeze(0).squeeze(0).detach().cpu().numpy()
        #b = get_boundary_map(b)

        hippovolume = np.clip(hippo.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, 1)
        ctvolume = ct.squeeze(0).squeeze(0).detach().cpu().numpy()

        os.makedirs(f"D:\\HIPPO\\{args.date}\\testu\\{folder}", exist_ok=True)
        os.makedirs(f"D:\\HIPPO\\{args.date}\\testu\\{folder}\\{filename}", exist_ok=True)

        ct = sitk.GetImageFromArray(ctvolume)
        hippo = sitk.GetImageFromArray(hippovolume)

        threshold = 0.9
        boundary = sitk.GetImageFromArray(edge)
        pred1 = sitk.GetImageFromArray(thresholding(pred, threshold))
        # o2 = sitk.GetImageFromArray(thresholding(o2, threshold))
        # o3 = sitk.GetImageFromArray(thresholding(o3, threshold))
        # o4 = sitk.GetImageFromArray(thresholding(o4, threshold))
        # o2_b = sitk.GetImageFromArray(thresholding(o2_b, threshold))
        # o3_b = sitk.GetImageFromArray(thresholding(o3_b, threshold))
        # o4_b = sitk.GetImageFromArray(thresholding(o4_b, threshold))
        #b = sitk.GetImageFromArray(thresholding(b, threshold))

        sitk.WriteImage(ct, f"D:\\HIPPO\\{args.date}\\testu\\{folder}\\{filename}\\ct.nii.gz")
        sitk.WriteImage(hippo, f"D:\\HIPPO\\{args.date}\\testu\\{folder}\\{filename}\\hippo.nii.gz")
        #sitk.WriteImage(edge, f"D:\\HIPPO\\{args.date}\\testu\\{folder}\\{filename}\\boundary.nii.gz")

        sitk.WriteImage(pred1, f"D:\\HIPPO\\{args.date}\\testu\\{folder}\\{filename}\\pred.nii.gz")
        # sitk.WriteImage(o2, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\o2.nii.gz")
        # sitk.WriteImage(o3, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\o3.nii.gz")
        # sitk.WriteImage(o4, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\o4.nii.gz")
        # sitk.WriteImage(o2_b, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\o2_b.nii.gz")
        # sitk.WriteImage(o3_b, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\o3_b.nii.gz")
        # sitk.WriteImage(o4_b, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\o4_b.nii.gz")
        #sitk.WriteImage(b, f"D:\\HIPPO\\{args.date}\\test\\{folder}\\{filename}\\b.nii.gz")

        for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            dice = dice_coefficient(thresholding(pred, threshold), hippovolume)
            iou = iou_coefficient(thresholding(pred, threshold), hippovolume)

            if threshold == 0:
                total_dice0 += dice
                total_iou0 += iou
            elif threshold == 0.1:
                total_dice1 += dice
                total_iou1 += iou  
            elif threshold == 0.2:
                total_dice2 += dice
                total_iou2 += iou 
            elif threshold == 0.3:
                total_dice3 += dice
                total_iou3 += iou 
            elif threshold == 0.4:
                total_dice4 += dice
                total_iou4 += iou 
            elif threshold == 0.5:
                total_dice5 += dice
                total_iou5 += iou 
            elif threshold == 0.6:
                total_dice6 += dice
                total_iou6 += iou 
            elif threshold == 0.7:
                total_dice7 += dice
                total_iou7 += iou 
            elif threshold == 0.8:
                total_dice8 += dice
                total_iou8 += iou 
            elif threshold == 0.9:
                total_dice9 += dice
                total_iou9 += iou 
                     
            print(f"{folder} {threshold}=>  Dice : {dice:>.3f} IOU : {iou:>.3f}")
        print()
    #datasize = datasize-3
    print(f"Mean Dice 0 : {total_dice0/datasize:>.3f}")
    print(f"Mean IOU  0 : {total_iou0/datasize:>.3f}")
    print()
    print(f"Mean Dice 1 : {total_dice1/datasize:>.3f}")
    print(f"Mean IOU  1 : {total_iou1/datasize:>.3f}")
    print()
    print(f"Mean Dice 2 : {total_dice2/datasize:>.3f}")
    print(f"Mean IOU  2 : {total_iou2/datasize:>.3f}")
    print()
    print(f"Mean Dice 3 : {total_dice3/datasize:>.3f}")
    print(f"Mean IOU  3 : {total_iou3/datasize:>.3f}")
    print()
    print(f"Mean Dice 4 : {total_dice4/datasize:>.3f}")
    print(f"Mean IOU  4 : {total_iou4/datasize:>.3f}")
    print()
    print(f"Mean Dice 5 : {total_dice5/datasize:>.3f}")
    print(f"Mean IOU  5 : {total_iou5/datasize:>.3f}")
    print()
    print(f"Mean Dice 6 : {total_dice6/datasize:>.3f}")
    print(f"Mean IOU  6 : {total_iou6/datasize:>.3f}")
    print()
    print(f"Mean Dice 7 : {total_dice7/datasize:>.3f}")
    print(f"Mean IOU  7 : {total_iou7/datasize:>.3f}")
    print()
    print(f"Mean Dice 8 : {total_dice8/datasize:>.3f}")
    print(f"Mean IOU  8 : {total_iou8/datasize:>.3f}")
    print()
    print(f"Mean Dice 9 : {total_dice9/datasize:>.3f}")
    print(f"Mean IOU  9 : {total_iou9/datasize:>.3f}")
    print()