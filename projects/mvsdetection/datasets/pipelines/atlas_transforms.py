# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie and Jiaming Sun.

# Original header:
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

from PIL import Image, ImageOps
import numpy as np
import torch
import mmcv 
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from projects.mvsdetection.datasets.tsdf import TSDF





@PIPELINES.register_module()
class AtlasToTensor(object):
    def __call__(self, data):
        data['imgs'] = torch.Tensor(np.stack(data['imgs']).transpose([0, 3, 1, 2])) #N * C * H * w
        data['intrinsics'] = torch.Tensor(np.stack(data['intrinsics'])) #N * 3 * 3
        data['extrinsics'] = torch.Tensor(np.stack(data['extrinsics'])) #N * 4 * 4
        if 'ann_info' in data.keys():
            data['gt_bboxes_3d'] = data['ann_info']['gt_bboxes_3d']
            data['gt_labels_3d'] = torch.Tensor(data['ann_info']['gt_labels_3d']).long()
            data['axis_align_matrix'] = torch.Tensor(data['ann_info']['axis_align_matrix'])
            data.pop('ann_info')
        return data

@PIPELINES.register_module()
class AtlasCollectData(object):
    def __call__(self, data):
        result = {}
        result['imgs'] = DC(data['imgs'])
        result['projection'] = DC(data['projection'])
        result['tsdf_dict'] = DC(data['tsdf_dict'], cpu_only=True)
        result['scene'] = DC(data['scene'], cpu_only=True)
        if 'offset' in data.keys():
            result['offset'] = DC(data['offset'])
        if 'gt_bboxes_3d' in data.keys():
            result['gt_bboxes_3d'] = DC(data['gt_bboxes_3d'], cpu_only=True)
            result['gt_labels_3d'] = DC(data['gt_labels_3d'])
            result['axis_align_matrix'] = DC(data['axis_align_matrix'])
        return result



def pad_scannet(img, intrinsics):
    """ Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    """

    w, h = img.size
    if w == 1296 and h == 968:
        img = ImageOps.expand(img, border=(0, 2))
        intrinsics[1, 2] += 2
    return img, intrinsics

@PIPELINES.register_module()
class AtlasResizeImage(object):
    """ Resize everything to given size.

    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        for i in range(len(data['imgs'])):
            im, intrinsics = pad_scannet(data['imgs'][i], data['intrinsics'][i])
            w, h = im.size
            im = im.resize(self.size, Image.BILINEAR)
            intrinsics[0, :] /= (w / self.size[0])
            intrinsics[1, :] /= (h / self.size[1])

            data['imgs'][i] = np.array(im, dtype=np.float32)
            data['intrinsics'][i] = intrinsics

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    

@PIPELINES.register_module()
class AtlasIntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix"""
    def __call__(self, data):
        data['projection'] = []
        for i in range(len(data['intrinsics'])):
            intrinsic = data['intrinsics'][i]
            extrinsic = data['extrinsics'][i]
            projection = intrinsic @ extrinsic.inverse()[:3, :]
            data['projection'].append(projection)
        data['projection'] = torch.stack(data['projection']) #N * 3 * 4
        data.pop('intrinsics')
        data.pop('extrinsics')
        return data
    


def transform_space(data, transform, voxel_dim, origin):
    """ Apply a 3x4 linear transform to the world coordinate system.

    This affects pose as well as TSDFs.
    """
    for i in range(len(data['extrinsics'])):
        data['extrinsics'][i] = transform.inverse() @ data['extrinsics'][i]

    voxel_sizes = [int(key[8:]) for key in data['tsdf_dict']] #4, 8, 16

    for voxel_size in voxel_sizes:
        scale = voxel_size / min(voxel_sizes)
        vd = [int(vd / scale) for vd in voxel_dim]
        key = 'tsdf_gt_' + str(voxel_size).zfill(3)
        data['tsdf_dict'][key] = data['tsdf_dict'][key].transform(transform, vd, origin)
    return data

@PIPELINES.register_module()
class AtlasRandomTransformSpace(object):
    """ Apply a random 3x4 linear transform to the world coordinate system."""

    def __init__(self, voxel_dim, random_rotation=True, random_translation=True,
                 paddingXY=1.5, paddingZ=.25, origin=[0,0,0]):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying 
                the size of the output volume
            random_rotation: wheater or not to apply a random rotation
            random_translation: wheater or not to apply a random translation
            paddingXY: amount to allow croping beyond maximum extent of TSDF
            paddingZ: amount to allow croping beyond maximum extent of TSDF
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        """

        self.voxel_dim = voxel_dim
        self.origin = origin
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.padding_start = torch.tensor([paddingXY, paddingXY, paddingZ])
        # no need to pad above (bias towards floor in volume)
        self.padding_end = torch.tensor([paddingXY, paddingXY, 0])

    def __call__(self, data):
        tsdf = data['tsdf_dict']['tsdf_gt_004']

        # construct rotaion matrix about z axis
        if self.random_rotation:
            r = torch.rand(1) * 2 * np.pi
        else:
            r = 0
        # first construct it in 2d so we can rotate bounding corners in the plane
        R = torch.tensor([[np.cos(r), -np.sin(r)],
                          [np.sin(r), np.cos(r)]], dtype=torch.float32)

        # get corners of bounding volume
        voxel_dim = torch.tensor(tsdf.tsdf_vol.shape) * tsdf.voxel_size
        xmin, ymin, zmin = tsdf.origin[0]
        xmax, ymax, zmax = tsdf.origin[0] + voxel_dim
        corners2d = torch.tensor([[xmin, xmin, xmax, xmax],
                                  [ymin, ymax, ymin, ymax]], dtype=torch.float32)

        # rotate corners in plane
        corners2d = R @ corners2d

        # get new bounding volume (add padding for data augmentation)
        xmin = corners2d[0].min()
        xmax = corners2d[0].max()
        ymin = corners2d[1].min()
        ymax = corners2d[1].max()
        zmin = zmin
        zmax = zmax

        # randomly sample a crop
        start = torch.tensor([xmin, ymin, zmin]) - self.padding_start
        end = (-torch.as_tensor(self.voxel_dim) * tsdf.voxel_size +
                torch.tensor([xmax, ymax, zmax]) + self.padding_end)
        if self.random_translation:
            t = torch.rand(3)
        else:
            t = .5
        t = t*start + (1-t) * end
            
        T = torch.eye(4)
        T[:2,:2] = R
        T[:3,3] = -t

        data['offset'] = -t
        return transform_space(data, T.inverse(), self.voxel_dim, self.origin)

    def __repr__(self):
        return self.__class__.__name__
    


@PIPELINES.register_module()
class AtlasTestTransformSpace(object):
    """ See transform_space"""

    def __init__(self, voxel_dim, origin):
        self.voxel_dim = voxel_dim
        self.origin = origin

    def __call__(self, data):
        T = torch.eye(4)
        voxel_size = data['tsdf_dict']['tsdf_gt_004'].voxel_size 
        origin = data['tsdf_dict']['tsdf_gt_004'].origin
        shift = torch.tensor([.5, .5, .5]) // voxel_size
        offset = origin - shift * voxel_size
        T[:3, 3] = offset
        data['offset'] = offset
        return transform_space(data, T, self.voxel_dim, self.origin)

    def __repr__(self):
        return self.__class__.__name__
    