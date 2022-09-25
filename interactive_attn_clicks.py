import os
from os import path
from argparse import ArgumentParser
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
from model.network import deeplabv3plus_resnet50 as S2M
from model.aggregate import aggregate_wbg_channel as aggregate
from dataset.range_transform import im_normalization
from util.tensor_util import pad_divide_by
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from torchsummary import summary
from torch import nn
from torch.nn import functional as F
import math
from model_base import Base
from typing import List
torch.backends.cudnn.benchmark = True


class InteractiveManager:
    def __init__(self, model, image, mask):
        self.model = model

        self.image = im_normalization(TF.to_tensor(image)).unsqueeze(0).cuda()
        self.mask = TF.to_tensor(mask).unsqueeze(0).cuda()

        h, w = self.image.shape[-2:]
        self.image, self.pad = pad_divide_by(self.image, 16)
        self.mask, _ = pad_divide_by(self.mask, 16)
        self.last_mask = None

        # Positive and negative scribbles
        self.p_srb = np.zeros((h, w), dtype=np.uint8)
        self.n_srb = np.zeros((h, w), dtype=np.uint8)
        self.s_srb = np.zeros((h, w), dtype=np.uint8)

        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.mode = 0
        self.need_update = True

    def mouse_down(self, ex, ey):
        self.last_ex = ex
        self.last_ey = ey
        self.pressed = True
        if self.mode == 0:
            cv2.circle(self.p_srb, (ex, ey), radius=15, color=(1), thickness=-1)
        elif self.mode == 1:
            cv2.circle(self.n_srb, (ex, ey), radius=15, color=(1), thickness=-1)
        elif self.mode == 2:
            cv2.circle(self.s_srb, (ex, ey), radius=15, color=(1), thickness=-1)
        self.need_update = True

    def mouse_move(self, ex, ey):
        if not self.pressed:
            return
        if self.mode == 0:
            cv2.line(self.p_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=15)
        elif self.mode == 1:
            cv2.line(self.n_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=15)
        elif self.mode == 2:
            cv2.line(self.s_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=15)
        self.need_update = True
        self.last_ex = ex
        self.last_ey = ey

    def mouse_up(self):
        self.pressed = False

    first = True

    def run_s2m(self):
        # Convert scribbles to tensors
        Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        Rss = torch.from_numpy(self.s_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        Rs = torch.cat([Rsn, Rsp, Rss], 1)
        Rs, _ = pad_divide_by(Rs, 16)

        # Use the network to do stuff
        inputs = torch.cat([self.image, self.mask, Rs], 1)
        mask = net(inputs)
        #inputs = torch.cat([self.image, mask, Rs], 1)
        #mask = net(inputs)
        #print(mask)
        # We don't overwrite current mask until commit
        if self.first:
            mask.zero_()
            self.first = False
            
        self.last_mask = mask
        np_mask = (mask.detach().cpu().numpy()[0,0] * 255).astype(np.uint8)

        if self.pad[2]+self.pad[3] > 0:
            np_mask = np_mask[self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            np_mask = np_mask[:,self.pad[0]:-self.pad[1]]

        return np_mask

    def commit(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        self.s_srb.fill(0)
        if self.last_mask is not None:
            self.mask = self.last_mask

    def clean_up(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        self.s_srb.fill(0)
        self.mask.zero_()
        self.last_mask = None



parser = ArgumentParser()
parser.add_argument('--image', default=r'./antique-honiton-lace-1182740_1920_0.png')
parser.add_argument('--model', default=r'./saves/0.009609096754474689')
parser.add_argument('--mask', default=None)
args = parser.parse_args()



def limit_longest_size(image, max_side_size):
    if image.shape[0] > image.shape[1]:
        target_h = min(image.shape[0], max_side_size)
        target_w = int(image.shape[1] / image.shape[0] * target_h + 0.5)
    else:
        target_w = min(image.shape[1], max_side_size)
        target_h = int(image.shape[0] / image.shape[1] * target_w + 0.5)
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return image


net = Base("resnet50", 7, 1)
net.load_state_dict(torch.load(args.model)['model_state_dict'])
net = net.cuda().eval()
torch.set_grad_enabled(False)

# Reading stuff
image = cv2.imread(args.image, cv2.IMREAD_COLOR)
image = limit_longest_size(image, 1000)
h, w = image.shape[:2]
if args.mask is None:
    mask = np.zeros((h, w), dtype=np.uint8)
else:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

manager = InteractiveManager(net, image, mask)

def mouse_callback(event, x, y, *args):
    if event == cv2.EVENT_LBUTTONDOWN:
        manager.mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        manager.mouse_up()
    elif event == cv2.EVENT_MBUTTONDOWN:
        manager.mode = (manager.mode + 1) % 3
        if manager.mode == 0:
            print('Entering positive scribble mode.')
        elif manager.mode == 1:
            print('Entering negative scribble mode.')
        elif manager.mode == 2:
            print('Entering transparent scribble mode.')

    # Draw
    if event == cv2.EVENT_MOUSEMOVE:
        manager.mouse_move(x, y)

def comp_image(image, mask, p_srb, n_srb, s_srb):
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:,:,2] = 1
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    comp = (image*0.5 + color_mask*mask*0.5).astype(np.uint8)
    comp[p_srb>0.5, :] = np.array([0, 255, 0], dtype=np.uint8)
    comp[s_srb>0.5, :] = np.array([255, 0, 0], dtype=np.uint8)
    comp[n_srb>0.5, :] = np.array([0, 0, 255], dtype=np.uint8)
    return comp

# OpenCV setup
cv2.namedWindow('Test')
cv2.setMouseCallback('Test', mouse_callback)

print('Usage: python interactive.py --image <image> --model <model> [Optional: --mask initial_mask]')
print('This GUI is rudimentary; the network is naively designed.')
print('Mouse Left - Draw scribbles')
print('Mouse middle key - Switch positive/negative')
print('Key f - Commit changes, clear scribbles')
print('Key r - Clear everything')
print('Key d - Switch between overlay/mask view')
print('Key s - Save masks into a temporary output folder (./output/)')

display_comp = True
while 1:
    if manager.need_update:
        np_mask = manager.run_s2m()
        if display_comp:
            display = comp_image(image, np_mask, manager.p_srb, manager.n_srb, manager.s_srb)
        else:
            display = np_mask
        manager.need_update = False

    cv2.imshow('Test', display)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('f'):
        manager.commit()
        manager.need_update = True
    elif k == ord('s'):
        print('saved')
        os.makedirs('output', exist_ok=True)
        cv2.imwrite('output/%s' % path.basename(args.mask), mask)
    elif k == ord('d'):
        display_comp = not display_comp
        manager.need_update = True
    elif k == ord('r'):
        manager.clean_up()
        manager.first = True
        manager.need_update = True
    elif k == 27:
        break

cv2.destroyAllWindows()
