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

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from mmdet.models import NECKS

@NECKS.register_module()
class Neck3D(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512], layers=[1, 2, 3, 4]):
        super(Neck3D, self).__init__()
        self.block = BasicBlock
        self.channels = channels
        self.up_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        for i in range(0, len(channels) - 1):
            self.skip_connections.append(self._make_skip_connection_block(channels[i]))

            self.up_layers.append(self._make_up_block(channels[i + 1], channels[i]))
            self.conv_layers.append(nn.Sequential(*[
                BasicBlock(channels[i], channels[i], stride=1, dilation=1, dimension=3) 
                for _ in range(layers[i])]))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
    
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiRELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiRELU()
        )
    
    def _make_skip_connection_block(channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(channels, channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(channels),
            ME.MinkowskiRELU()
        )
    
    def forward(self, xs):
        outs = []
        x = xs[len(xs) - 1]#4
        outs.append(x)
        for i in range(1, len(xs) - 1): #1,2,3,4
            idx = len(xs) - 1 - i #3,2,1,0
            x = self.up_layers[idx](x) 
            
            old_x = xs[idx]
            y = self.skip_connections[idx](old_x)
            x = x + y 
            
            x = self.conv_layers[idx](x)            
            outs.append(x)
        outs = outs[::-1]
        return outs
        