# Modified from
# https://github.com/magicleap/Atlas/blob/master/atlas/transforms.py
# Copyright (c) MagicLeap, Inc. and its affiliates.
"""Transforms for ARKit data used in data processing.
"""


import numpy as np
import torch



class Compose(object):
    """ Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class ToTensor(object):
    """ Convert to torch tensors"""
    def __call__(self, data):
        for frame in data['frames']:
            image = np.array(frame['image'])
            frame['image'] = torch.as_tensor(image).float().permute(2, 0, 1)
            frame['intrinsics'] = torch.as_tensor(frame['intrinsics'])
            frame['pose'] = torch.as_tensor(frame['pose'])

            if 'depth' in frame:
                frame['depth'] = torch.as_tensor(np.array(frame['depth']))

            if 'instance' in frame:
                instance = np.array(frame['instance'])
                frame['instance'] = torch.as_tensor(instance).long()
        return data

class IntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix"""
    def __call__(self, data):
        for frame in data['frames']:
            intrinsics = frame.pop('intrinsics')
            pose = frame.pop('pose')
            frame['projection'] = intrinsics @ pose.inverse()[:3,:]
        return data

