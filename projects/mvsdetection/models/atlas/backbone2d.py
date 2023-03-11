import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class FPNFeature(nn.Module):
    """ Converts feature pyrimid to singe feature map (from Detectron2)"""
    
    def __init__(self, feature_strides, feature_channels, output_dim=32, output_stride=4):
        super().__init__()
        self.feature_strides = feature_strides
        self.feature_channels = feature_channels
        self.scale_heads = nn.ModuleList()
        for i in range(len(self.feature_channels)):
            head_ops = nn.ModuleList()
            head_length = max(
                1, int(np.log2(feature_strides[i]) - np.log2(output_stride))
            )
            for k in range(head_length):
                batch_normalization = nn.BatchNorm2d(output_dim)
                conv = nn.Conv2d(in_channels=feature_channels[i] if k == 0 else output_dim,
                              out_channels=output_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
                net = nn.Sequential(
                    conv,
                    batch_normalization,
                    nn.ReLU()
                )
                head_ops.append(net)
                if feature_strides[i] != output_stride:
                    head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads.append(nn.Sequential(*head_ops))


    def forward(self, features):
        for i in range(len(self.feature_channels)):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        return x
