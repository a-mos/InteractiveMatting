import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean, inv_im_trans
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

#Mode0 - png - rgb+alpha
#Mode1 - png - rgb+alpha + removebg
#Mode2 - png + jpeg - rgb+alpha separetly
#Mode3 - png - rgb+alpha + separetly + removebg

class StaticTransformDataset(Dataset):
    def __init__(self, root, method=0, bg_path=None):
        self.root = root
        self.method = method
        self.bg_path = bg_path
        
        if method == 0:
            self.im_list = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root))]

        elif method == 1:
            self.im_list = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root))]
            self.bg_list = [os.path.join(self.bg_path, x) for x in sorted(os.listdir(self.bg_path))]

        elif method == 2:
            self.im_list = sorted(os.listdir(os.path.join(self.root, 'images')))
            self.alpha_paths = sorted(os.listdir(os.path.join(self.root, 'masks')))
            assert [x[:-4] for x in self.im_list] == [x[:-4] for x in self.alpha_paths]
            self.im_list = [os.path.join(self.root, 'images', x) for x in self.im_list]
            self.alpha_paths = [os.path.join(self.root, 'masks', x) for x in self.alpha_paths]
            
        elif method == 3:
            self.im_list = sorted(os.listdir(os.path.join(self.root, 'images')))
            self.alpha_paths = sorted(os.listdir(os.path.join(self.root, 'masks')))
            assert [x[:-4] for x in self.im_list] == [x[:-4] for x in self.alpha_paths]
            self.im_list = [os.path.join(self.root, 'images', x) for x in self.im_list]
            self.alpha_paths = [os.path.join(self.root, 'masks', x) for x in self.alpha_paths]
            self.bg_list = [os.path.join(self.bg_path, x) for x in sorted(os.listdir(self.bg_path))]
            
        print('%d images found in %s' % (len(self.im_list), root))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, resample=Image.BILINEAR, fillcolor=im_mean),
            transforms.Resize(256, Image.BILINEAR),
            transforms.RandomCrop((256, 256), pad_if_needed=True, fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, resample=Image.NEAREST, fillcolor=0),
            transforms.Resize(256, Image.NEAREST),
            transforms.RandomCrop((256, 256), pad_if_needed=True, fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    
    def __getitem__(self, idx):
        
        if self.method == 0:
            im = Image.open(self.im_list[idx]).convert('RGBA')
            _,_,_, gt = im.split()
            im = im.convert('RGB')
            
        if self.method == 1:
            im = Image.open(self.im_list[idx]).convert('RGBA')
            _,_,_, gt = im.split()
            
            im = im.convert('RGB')
            alpha = np.array(gt) / 255.
            img = np.array(im) / 255.
            
            bg = Image.open(self.bg_list[(idx + len(self.im_list)) % len(self.bg_list)]).convert('RGB')
            bg = bg.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
            
            bg = np.array(bg) / 255.
            img[:, :, 0] = alpha * img[:, :, 0] + (1 - alpha) * bg[:, :, 0]
            img[:, :, 1] = alpha * img[:, :, 1] + (1 - alpha) * bg[:, :, 1]
            img[:, :, 2] = alpha * img[:, :, 2] + (1 - alpha) * bg[:, :, 2]
            im = Image.fromarray(np.uint8(img * 255))
            
        if self.method == 2:
            im = Image.open(self.im_list[idx]).convert('RGB')
            gt = Image.open(self.alpha_paths[idx]).convert('L')
            
        if self.method == 3:
            im = Image.open(self.im_list[idx]).convert('RGB')
            gt = Image.open(self.alpha_paths[idx]).convert('L')
            
            alpha = np.array(gt) / 255.
            img = np.array(im) / 255.
            
            bg = Image.open(self.bg_list[(idx + len(self.im_list)) % len(self.bg_list)]).convert('RGB')
            bg = bg.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
            
            bg = np.array(bg) / 255.
            img[:, :, 0] = alpha * img[:, :, 0] + (1 - alpha) * bg[:, :, 0]
            img[:, :, 1] = alpha * img[:, :, 1] + (1 - alpha) * bg[:, :, 1]
            img[:, :, 2] = alpha * img[:, :, 2] + (1 - alpha) * bg[:, :, 2]
            im = Image.fromarray(np.uint8(img * 255))
            
            
        sequence_seed = np.random.randint(2147483647)
        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        im = self.im_lone_transform(im)
        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)

        gt_np = np.array(gt)
        if np.random.rand() < 0.33:
            # from_zero - no previous mask
            seg = np.zeros_like(gt_np)
            from_zero = True
        else:
            iou_max = 0.95
            iou_min = 0.4
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            seg = perturb_mask(gt_np, iou_target=iou_target)
            from_zero = False
        
        # Generate scribbles
        p_srb, n_srb, s_srb = get_scribble(seg, gt_np, from_zero=from_zero)

        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)

        p_srb = torch.from_numpy(p_srb)
        n_srb = torch.from_numpy(n_srb)
        s_srb = torch.from_numpy(s_srb)
            
        srb = torch.stack([n_srb, p_srb, s_srb], 0).float()
        seg = self.final_gt_transform(seg)

        info = {}
        info['name'] = self.im_list[idx]

        data = {
            'rgb': im,
            'gt': gt,
            'seg': seg,
            'srb': srb,
            'info': info
        }

        return data


    def __len__(self):
        return len(self.im_list)