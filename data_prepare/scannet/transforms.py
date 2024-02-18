# Modified from
# https://github.com/magicleap/Atlas/blob/master/atlas/transforms.py
# Copyright (c) MagicLeap, Inc. and its affiliates.
"""Transforms for ScanNet data used in data processing.
"""


from PIL import Image, ImageOps
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


def pad_scannet(frame):
    """ Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    """

    w,h = frame['image'].size
    if w==1296 and h==968:
        frame['image'] = ImageOps.expand(frame['image'], border=(0,2))
        frame['intrinsics'][1, 2] += 2
        if 'instance' in frame and frame['instance'] is not None:
            frame['instance'] = ImageOps.expand(frame['instance'], border=(0,2))
    return frame


class ResizeImage(object):
    """ Resize everything to given size.

    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        for frame in data['frames']:
            pad_scannet(frame)
            w,h = frame['image'].size
            frame['image'] = frame['image'].resize(self.size, Image.BILINEAR)
            frame['intrinsics'][0, :] /= (w / self.size[0])
            frame['intrinsics'][1, :] /= (h / self.size[1])

            if 'depth' in frame:
                frame['depth'] = frame['depth'].resize(self.size, Image.NEAREST)
            
            if 'instance' in frame and frame['instance'] is not None:
                frame['instance'] = frame['instance'].resize(self.size, Image.NEAREST)
            #if 'semseg' in frame:
            #    frame['semseg'] = frame['semseg'].resize(self.size, Image.NEAREST)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

