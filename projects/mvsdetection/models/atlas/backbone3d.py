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
from mmdet.models import BACKBONES

@BACKBONES.register_module()
class Backbone3D(nn.Module):
    def __init__(self, in_channels=32, out_channels=[32, 64, 128, 256, 512], layers=[1, 3, 4, 6, 3]):
        super(Backbone3D, self).__init__()
        self.block = BasicBlock
        self.out_channels = out_channels
        self.n_outs = len(out_channels)
        self.inplanes = in_channels
        self.layers = nn.ModuleList()
        
        layer_0 = self._make_layer(self.block, self.out_channels[0], layers[0], stride=1)
        self.layers.append(layer_0)
        
        for i in range(1, self.n_outs):
            layer = self._make_layer(self.block, self.out_channels[i], layers[i], stride=2)
            self.layers.append(layer)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=3,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=3))

        return nn.Sequential(*layers)
    
    def forward(self, x: ME.SparseTensor):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs
