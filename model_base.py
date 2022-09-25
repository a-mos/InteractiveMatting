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
from typing import List
from resnet import ResNetEncoder
from decoder import Decoder

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class SelfAttention(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_proj = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)
 
    def forward(self, x):  
        key, value = self.key_proj(x), self.val_proj(x)
        key = key.flatten(start_dim=2)
        a = key.pow(2).sum(1).unsqueeze(2)
        b = 2 * (key.transpose(1, 2) @ key)
        c = key.pow(2).sum(1).unsqueeze(1)
        affinity = (-a+b-c) / math.sqrt(key.shape[1])   # B, THW, HW
        affinity = F.softmax(affinity, dim=1)
        self_attn = torch.bmm(value.flatten(start_dim=2), affinity).view(x.shape[0], -1, x.shape[2], x.shape[3])
        out = torch.cat((value, self_attn), dim=1)
        return out
    
    
class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = []
        
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(SelfAttention(2048, keydim=128, valdim=256))    

        modules.append(ASPPPooling(in_channels, out_channels))
        

        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(1792, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
            
        #_res.append(self.kv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

#from torchvision.models.segmentation.deeplabv3 import ASPP

class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """
    
    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([512, 256, 128, 64, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts).clip(0., 1.)
        return x