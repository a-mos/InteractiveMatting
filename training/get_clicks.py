import os
from os import path
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean, inv_im_trans
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

import skimage

def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def get_clicks(seg, gt, from_zero=True):
    
    TOLERANCE = 10
    TOLERANCE_SEMITRANS = 15
    # False positive and false negative
    need_negative = ((seg > TOLERANCE) & (gt == 0)).astype(np.uint8)
    need_positive = ((seg < 255 - TOLERANCE) & (gt == 255)).astype(np.uint8)
    need_semitrans = (((seg == 0) | (seg == 255)) & ((gt > TOLERANCE_SEMITRANS) & (gt < 255 - TOLERANCE_SEMITRANS))).astype(np.uint8)
    
    opening_size = np.random.randint(5, 20)
    need_negative = cv2.morphologyEx(need_negative, cv2.MORPH_OPEN, disk_kernel(opening_size))
    need_positive = cv2.morphologyEx(need_positive, cv2.MORPH_OPEN, disk_kernel(opening_size))
    need_semitrans = cv2.morphologyEx(need_semitrans, cv2.MORPH_OPEN, disk_kernel(opening_size))
    
    negative_components = cv2.connectedComponentsWithStats(need_negative)
    positive_components = cv2.connectedComponentsWithStats(need_positive)
    semitrans_components = cv2.connectedComponentsWithStats(need_semitrans)
    
    num_positive = num_negative = num_semitrans = 0
    #print(negative_components[0], positive_components[0], semitrans_components[0])
    if not from_zero:
        for tryes in range(negative_components[0]):
            if np.random.geometric(p=1./6.) == 1 and num_negative <= 3:
                num_negative += 1
            else:
                break
        for tryes in range(positive_components[0]):
            if np.random.geometric(p=1./6.) == 1 and num_positive <= 3:
                num_positive += 1
            else:
                break        

        if num_positive == num_negative == 0:
            pass
        else:
            for tryes in range(semitrans_components[0]):
                if np.random.geometric(p=1./6.) == 1 and num_semitrans <= 3:
                    num_semitrans += 1
                else:
                    break        

    num_points = [num_negative, num_positive, num_semitrans]
    masks = []
    
    for idx, component in enumerate([negative_components, positive_components, semitrans_components]):
        mask = np.zeros_like(seg)
        if not from_zero:
            X = list(component[3][1:][:, 0])
            Y = list(component[3][1:][:, 1])
            AREA = list(component[2][:, -1][1:])
            POINTS = sorted(list(zip(X, Y, AREA)), key=lambda x: -x[-1])[:num_points[idx]]
            for point in POINTS:
                min_r = 1 + np.sqrt(point[2]) // 6
                max_r = np.sqrt(point[2]) // 3
                if min_r >= max_r:
                    max_r = min_r + 1
                RADIUS = min(np.random.randint(min_r, max_r), 15)
                rr, cc = skimage.draw.disk([int(point[1]), int(point[0])], RADIUS, shape=mask.shape)
                mask[rr, cc] = 1
        masks.append(mask)
        
    if from_zero or np.sum(num_points) == 0 or np.random.random() > 0.8:
        negative_area = (gt == 0).astype(np.uint8)
        positive_area = (gt == 255).astype(np.uint8)
        semitrans_area = ((gt > 0) & (gt < 255)).astype(np.uint8)

        opening_size = np.random.randint(5, 20)
        need_negative = cv2.morphologyEx(negative_area, cv2.MORPH_OPEN, disk_kernel(opening_size))
        need_positive = cv2.morphologyEx(positive_area, cv2.MORPH_OPEN, disk_kernel(opening_size))
        need_semitrans = cv2.morphologyEx(semitrans_area, cv2.MORPH_OPEN, disk_kernel(opening_size))

        negative = cv2.erode(need_negative, disk_kernel(opening_size), iterations=1)
        positive = cv2.erode(need_positive, disk_kernel(opening_size), iterations=1)
        semitrans = cv2.erode(need_semitrans, disk_kernel(opening_size), iterations=1)
    
        if from_zero:  
            num_semitrans = num_negative = 0
            num_positive = np.random.randint(1, 3)
        elif np.sum(num_points) == 0:
            num_positive = np.random.randint(1, 3)
            num_negative = np.random.randint(1, 3)
            num_semitrans = np.random.randint(0, 2)
        else:  
            num_positive = np.random.randint(0, 2)
            num_negative = np.random.randint(0, 2)
            num_semitrans = np.random.randint(0, 2)
            
        num_points_from_gt = [num_negative, num_positive, num_semitrans]
        #print(num_points_from_gt)
        for idx, component in enumerate([negative, positive, semitrans]): 
            comp = np.where(component)
            for i in range(num_points_from_gt[idx]):
                if comp[0].shape[0]:
                    selected_point = np.random.choice(range(comp[0].shape[0]))
                    RADIUS = opening_size
                    rr, cc = skimage.draw.disk([comp[0][selected_point], comp[1][selected_point]], RADIUS, shape=masks[idx].shape)
                    masks[idx][rr, cc] = 1
    
    return masks[0].astype(np.uint8), masks[1].astype(np.uint8), masks[2].astype(np.uint8)