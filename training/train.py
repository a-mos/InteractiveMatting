import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader, ConcatDataset, random_split
from PIL import Image
import tqdm
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import skimage

torch.backends.cudnn.benchmark = True
IMG_SIZE = 512

def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from model import Base

cv2.setNumThreads(0)

def get_clicks(seg, gt, from_zero=True):
    cv2.setNumThreads(0)
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
                if np.random.geometric(p=1./5.) == 1 and num_semitrans <= 4:
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

        opening_size = np.random.randint(3, 20)
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
            num_semitrans = np.random.randint(1, 5)
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

#Mode0 - png - rgb+alpha
#Mode1 - png - rgb+alpha + removebg
#Mode2 - png + jpeg - rgb+alpha separetly
#Mode3 - png - rgb+alpha + separetly + removebg

class StaticTransformDataset(Dataset):
    def __init__(self, root, method=0, bg_path=None):
        cv2.setNumThreads(0)
        self.root = root
        self.method = method
        self.bg_path = bg_path
        
        if method == 0:
            self.im_list = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root))]

        elif method == 1:
            self.im_list = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root))]
            self.bg_list = [os.path.join(self.bg_path, x) for x in sorted(os.listdir(self.bg_path))]
            self.bg_list = [x for x in self.bg_list if 'ipynb' not in x]

        elif method == 2:
            self.im_list = sorted(os.listdir(os.path.join(self.root, 'images')))
            self.alpha_paths = sorted(os.listdir(os.path.join(self.root, 'masks')))
            assert [x[:-4] for x in self.im_list] == [x[:-4] for x in self.alpha_paths]
            self.im_list = [os.path.join(self.root, 'images', x) for x in self.im_list]
            self.alpha_paths = [os.path.join(self.root, 'masks', x) for x in self.alpha_paths]
            self.alpha_paths = [x for x in self.alpha_paths if 'ipynb' not in x]
        
        elif method == 3:
            self.im_list = sorted(os.listdir(os.path.join(self.root, 'images')))
            self.alpha_paths = sorted(os.listdir(os.path.join(self.root, 'masks')))
            assert [x[:-4] for x in self.im_list] == [x[:-4] for x in self.alpha_paths]
            self.im_list = [os.path.join(self.root, 'images', x) for x in self.im_list]
            self.alpha_paths = [os.path.join(self.root, 'masks', x) for x in self.alpha_paths]
            self.bg_list = [os.path.join(self.bg_path, x) for x in sorted(os.listdir(self.bg_path))]
            self.bg_list = [x for x in self.bg_list if 'ipynb' not in x]
            self.alpha_paths = [x for x in self.alpha_paths if 'ipynb' not in x]
        
        self.im_list = [x for x in self.im_list if 'ipynb' not in x]

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.05, 0.05, 0.01),
        ])

        self.im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.3), shear=10, resample=Image.BILINEAR, fillcolor=im_mean),
            transforms.Resize(IMG_SIZE, Image.BILINEAR),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE), pad_if_needed=True, fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.3), shear=10, resample=Image.NEAREST, fillcolor=0),
            transforms.Resize(IMG_SIZE, Image.NEAREST),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE), pad_if_needed=True, fill=0),
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
        try:
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

                bg = Image.open(np.random.choice(self.bg_list, size=1)[0]).convert('RGB')
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
                iou_min = 0.7
                iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
                seg = perturb_mask(gt_np, iou_target=iou_target)
                from_zero = False

            # Generate scribbles
            n_srb, p_srb, s_srb = get_clicks(seg, gt_np, from_zero=from_zero)

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
        except:
            print(idx, len(self.im_list))
            print(self.im_list[idx])
            return self.__getitem__((idx+1)%len(self.im_list))

    def __len__(self):
        return len(self.im_list)


class ValidationStaticTransformDataset(Dataset):
    def __init__(self, root, method=0, bg_path=None):
        cv2.setNumThreads(0)
        self.root = root
        self.method = method
        self.bg_path = bg_path
        
        if method == 0:
            self.im_list = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root))]

        elif method == 1:
            self.im_list = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root))]
            self.bg_list = [os.path.join(self.bg_path, x) for x in sorted(os.listdir(self.bg_path))]
            self.bg_list = [x for x in self.bg_list if 'ipynb' not in x]

        elif method == 2:
            self.im_list = sorted(os.listdir(os.path.join(self.root, 'images')))
            self.alpha_paths = sorted(os.listdir(os.path.join(self.root, 'masks')))
            assert [x[:-4] for x in self.im_list] == [x[:-4] for x in self.alpha_paths]
            self.im_list = [os.path.join(self.root, 'images', x) for x in self.im_list]
            self.alpha_paths = [os.path.join(self.root, 'masks', x) for x in self.alpha_paths]
            self.alpha_paths = [x for x in self.alpha_paths if 'ipynb' not in x]
        
        elif method == 3:
            self.im_list = sorted(os.listdir(os.path.join(self.root, 'images')))
            self.alpha_paths = sorted(os.listdir(os.path.join(self.root, 'masks')))
            assert [x[:-4] for x in self.im_list] == [x[:-4] for x in self.alpha_paths]
            self.im_list = [os.path.join(self.root, 'images', x) for x in self.im_list]
            self.alpha_paths = [os.path.join(self.root, 'masks', x) for x in self.alpha_paths]
            self.bg_list = [os.path.join(self.bg_path, x) for x in sorted(os.listdir(self.bg_path))]
            self.bg_list = [x for x in self.bg_list if 'ipynb' not in x]
            self.alpha_paths = [x for x in self.alpha_paths if 'ipynb' not in x]
        
        self.im_list = [x for x in self.im_list if 'ipynb' not in x]

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8,1.2), shear=0, resample=Image.NEAREST, fillcolor=0),
            transforms.Resize(IMG_SIZE, Image.BILINEAR),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE), pad_if_needed=True, fill=im_mean),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8,1.2), shear=0, resample=Image.NEAREST, fillcolor=0),
            transforms.Resize(IMG_SIZE, Image.NEAREST),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE), pad_if_needed=True, fill=0),
        ])

    def __getitem__(self, idx):
        try:
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

            reseed(idx)
            im = self.im_dual_transform(im)
            reseed(idx)
            gt = self.gt_dual_transform(gt)
            gt_np = np.array(gt)

            np.random.seed(idx)
            if np.random.rand() < 0.33:
                # from_zero - no previous mask
                seg = np.zeros_like(gt_np)
                from_zero = True
            else:
                iou_max = 0.95
                iou_min = 0.7
                iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
                seg = perturb_mask(gt_np, iou_target=iou_target)
                from_zero = False

            # Generate scribbles
            n_srb, p_srb, s_srb = get_clicks(seg, gt_np, from_zero=from_zero)

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
        except:
            print(idx, len(self.im_list))
            print(self.im_list[idx])
            return self.__getitem__((idx+1)%len(self.im_list))

    def __len__(self):
        return len(self.im_list)


if __name__ == '__main__':

    static_root = '/mnt/ssd3/datasets/Compositions1K'
    adobe = StaticTransformDataset(path.join(static_root, 'Training_merge/Adobe-licensed images'), method=3, bg_path=os.path.join(static_root, 'train2014'))
    val_adobe = ValidationStaticTransformDataset(path.join(static_root, 'Training_merge/Adobe-licensed images'), method=3, bg_path=os.path.join(static_root, 'train2014'))
    
    static_dataset = adobe
    val_static_dataset = val_adobe

    train_size = int(len(static_dataset) * 0.95)
    test_size = len(static_dataset) - train_size
    train_dataset, test_dataset = random_split(static_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_indexes = sorted(train_dataset.indices)
    test_indexes = sorted(test_dataset.indices)

    train_dataset = torch.utils.data.Subset(static_dataset, train_indexes)
    test_dataset = torch.utils.data.Subset(val_static_dataset, test_indexes)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=8, drop_last=True, pin_memory=False, shuffle=True, timeout=600)
    valid_loader = DataLoader(test_dataset, batch_size=16, num_workers=8, drop_last=True, pin_memory=False, shuffle=False, generator=torch.Generator().manual_seed(42), timeout=300)

    print("LEN TRAIN: ", len(train_loader))
    print("LEN VAL: ", len(valid_loader))
    
    model = Base("resnet50", 7, 1)
    model.to('cuda')
    criterion = nn.L1Loss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    #scaler = torch.cuda.amp.GradScaler()
    tb = SummaryWriter(log_dir='runs/matting_compositions1k')

    train_losses = {}
    valid_losses = {}
    best_loss = 99999

    optimizer.zero_grad()
    optimizer.step()
    
    for epoch in range(0, 5000):
        model.train()
        valid_losses[epoch] = []
        train_losses[epoch] = []
        running_loss = 0
    
        for batch_num, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = torch.cat((batch['rgb'], batch['seg'], batch['srb']), dim=1).cuda()
            outputs = model(inputs)
            loss = criterion(outputs, batch['gt'].cuda())

            if batch_num % 20 == 19:
                rgb = inv_im_trans(batch['rgb'])
                prev_mask = batch['seg']
                scribbles = batch['srb']
                gt = batch['gt']
                grid_rgb = torchvision.utils.make_grid(torch.from_numpy(np.where(scribbles, scribbles, rgb)), nrow=batch['rgb'].shape[0])
                grid_seg = torchvision.utils.make_grid(prev_mask, nrow=batch['rgb'].shape[0])
                grid_cur_segm = torchvision.utils.make_grid(outputs, nrow=batch['rgb'].shape[0])
                grid_gt = torchvision.utils.make_grid(gt, nrow=batch['gt'].shape[0])
                tb.add_image("Train", torch.cat((grid_rgb, grid_cur_segm.cpu(), grid_gt, grid_seg), dim = 1), global_step=epoch * len(train_loader) + batch_num)
                tb.flush()

            loss.backward()
            optimizer.step()
            
            if batch_num % 10 == 9:
                tb.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + batch_num)
                tb.flush()
        
        model.eval()
        with torch.no_grad():

            for batch_num, batch in enumerate(valid_loader):

                #with torch.cuda.amp.autocast(enabled=False):
                inputs = torch.cat((batch['rgb'], batch['seg'], batch['srb']), dim=1).cuda()
                outputs = model(inputs)
                loss = criterion(outputs, batch['gt'].cuda())
                
                if batch_num % 2 == 1:
                    rgb = inv_im_trans(batch['rgb'])
                    prev_mask = batch['seg']
                    scribbles = batch['srb']
                    gt = batch['gt']
                    grid_rgb = torchvision.utils.make_grid(torch.from_numpy(np.where(scribbles, scribbles, rgb)), nrow=batch['rgb'].shape[0])
                    grid_seg = torchvision.utils.make_grid(prev_mask, nrow=batch['rgb'].shape[0])
                    grid_cur_segm = torchvision.utils.make_grid(outputs, nrow=batch['rgb'].shape[0])
                    grid_gt = torchvision.utils.make_grid(gt, nrow=batch['gt'].shape[0])
                    tb.add_image("Valid", torch.cat((grid_rgb, grid_cur_segm.cpu(), grid_gt, grid_seg), dim = 1), global_step=epoch * len(train_loader) + batch_num)
                    tb.flush()

                valid_losses[epoch].append(loss.item())

            tb.add_scalar('Valid loss', np.mean(valid_losses[epoch]), epoch)
            tb.flush()
            
        print(" Valid loss: ", np.mean(valid_losses[epoch]))
        if np.mean(valid_losses[epoch]) < best_loss and np.mean(valid_losses[epoch]) < 0.09:
            best_loss = np.mean(valid_losses[epoch])
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(valid_losses[epoch])},
                 os.path.join(r'./saves', str(best_loss)))