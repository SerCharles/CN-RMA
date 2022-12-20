import torch
import numpy as np
import os
import cv2
import PIL.Image as Image
from math import *
from data_utils import *


def collate_fn(list_data):
    intrinsic, extrinsic, color_image, depth_image, valid = list_data
    return intrinsic, extrinsic, color_image, depth_image, valid


class ScanNetDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, data_path, scene_id, max_depth=inf):
        """
        Args:
            data_path [str]: [our dataset posed image path] 
            scene_id [str]: [the scene id]
            max_depth [float]: [mask out large depth values since they are noisy]
        """
        n_images = len([_ for _ in os.listdir(os.path.join(data_path, scene_id)) if _.endswith(".jpg")])
        self.n_images = n_images
        self.scene_id = scene_id
        self.data_path = data_path
        self.max_depth = max_depth
        self.id_list = [i for i in range(n_images)]


    def __len__(self):
        return self.n_images

    def __getitem__(self, id):
        """
        Args:
            id [int]: [the index]
        
        Returns:
            intrinsic [torch float array], [3 * 3]: [the intrinsic]
            extrinsic [torch float array], [4 * 4: [the extrinsic]
            color_image [torch float array], [H * W * 3]: [the image]
            depth_image [torch float array], [H * W]: [the depth image]
            valid [bool]: [whether the group is valid]
        """
        idx = str(self.id_list[id]).zfill(5)
        intrinsic_path = os.path.join(self.data_path, self.scene_id, "intrinsic_depth.txt")
        extrinsic_path = os.path.join(self.data_path, self.scene_id, idx + ".txt")
        image_path = os.path.join(self.data_path, self.scene_id, idx + ".jpg")
        depth_path = os.path.join(self.data_path, self.scene_id, idx + ".png")
        
        intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
        extrinsic = np.loadtxt(extrinsic_path)
        depth_image = cv2.imread(depth_path, -1).astype(np.float32)
        depth_image /= 1000.0 
        depth_image[depth_image > self.max_depth] = 0
        color_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)
        if extrinsic[0][0] == np.inf or extrinsic[0][0] == -np.inf or extrinsic[0][0] == np.nan:
            valid = False 
        else: 
            valid = True
        return intrinsic, extrinsic, color_image, depth_image, valid
