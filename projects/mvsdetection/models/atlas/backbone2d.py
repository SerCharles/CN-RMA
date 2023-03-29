import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models import BACKBONES
from mmcv.runner import auto_fp16
from projects.mvsdetection.models.atlas.detectron_base import (
    c2_xavier_fill, 
    c2_msra_fill, 
    get_norm,
    Backbone,
    CNNBlockBase,
    Conv2d,
    ShapeSpec,
    FrozenBatchNorm2d 
)


@BACKBONES.register_module()
class AtlasFPNFeature(nn.Module):
    """ Converts feature pyrimid to singe feature map (from Detectron2)"""
    
    def __init__(self, feature_strides, feature_channels, output_dim=32, output_stride=4, norm='BN'):
        super().__init__()
        self.fp16_enabled = False
        self.in_features = ["p2", "p3", "p4", "p5"]
        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(output_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else output_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, output_dim),
                    activation=F.relu,
                )
                c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != output_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

    @auto_fp16() 
    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        return x