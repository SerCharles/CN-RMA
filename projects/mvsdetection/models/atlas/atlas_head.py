# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models import HEADS 

@HEADS.register_module
class AtlasTSDFHead(nn.Module):
    """ Main head that regresses the TSDF"""

    def __init__(self, input_channels, n_scales, voxel_size, label_smoothing, sparse_threshold):
        super().__init__()
        self.input_channels = input_channels
        self.n_scales = n_scales
        self.voxel_size = voxel_size
        self.label_smoothing = label_smoothing
        self.sparse_threshold = sparse_threshold
        self.voxel_sizes = [self.voxel_size * (2 ** i) for i in range(n_scales)][::-1] #0.16, 0.08, 0.04
        self.keys = [str(int(voxel_size * 100)).zfill(3) for voxel_size in self.voxel_sizes] #016, 008, 004
        decoders = [nn.Conv3d(c, 1, 1, bias=False) 
                    for c in self.input_channels][::-1] 
        self.decoders = nn.ModuleList(decoders)


    def forward(self, xs, targets=None):
        output = {}
        losses = {}
        mask_surface_pred = []

        for i, (decoder, x) in enumerate(zip(self.decoders, xs)):
            # regress the TSDF
            tsdf = torch.tanh(decoder(x)) * self.label_smoothing
            
            # use previous scale to sparsify current scale
            if i > 0:
                previous_key = 'scene_tsdf_' + self.keys[i - 1]
                tsdf_prev = output[previous_key]
                tsdf_prev = F.interpolate(tsdf_prev, scale_factor=2)
                tsdf_prev = tsdf_prev.type_as(tsdf)
                mask_surface_pred_prev = tsdf_prev.abs()<self.sparse_threshold[i-1]
                tsdf[~mask_surface_pred_prev] = tsdf_prev[~mask_surface_pred_prev].sign()*.999
                mask_surface_pred.append(mask_surface_pred_prev)
            tsdf_key = 'scene_tsdf_' + self.keys[i]
            output[tsdf_key] = tsdf

        # compute losses
        if targets is not None:
            for i in range(self.n_scales):
                loss = torch.Tensor(np.array([0]))[0].cuda()
                loss_key = 'tsdf_loss_' + self.keys[i]
                losses.update({loss_key: loss})
            for i, voxel_size in enumerate(self.voxel_sizes):
                tsdf_key = 'scene_tsdf_' + self.keys[i]
                gt_key = 'tsdf_gt_' + self.keys[i]
                pred = output[tsdf_key]
                trgt = targets[gt_key]
                mask_observed = trgt<1
                mask_outside  = (trgt==1).all(-1, keepdim=True)
                pred = log_transform(pred, 1.0)
                trgt = log_transform(trgt, 1.0)
                loss = F.l1_loss(pred, trgt, reduction='none')
                loss_key = 'tsdf_loss_' + self.keys[i]
                if i==0:
                    # no sparsifing mask for first resolution
                    losses[loss_key] = loss[mask_observed | mask_outside].mean()
                else:
                    mask = mask_surface_pred[i-1] & (mask_observed | mask_outside)
                    if mask.sum()>0:
                        losses[loss_key] = loss[mask].mean()
                    else:
                        losses[loss_key] = 0 * loss.sum()
        return output, losses

def log_transform(x, shift=1):
    """ rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return x.sign() * (1 + x.abs() / shift).log()

