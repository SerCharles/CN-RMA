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

import os
import numpy as np
from PIL import Image
import torch


class RScanSimpleDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, data_path, scene_id, transform=None):
        """
        Args:
            info_file: path to json file (format described in datasets/README)
            transform: transform object to preprocess data
            frame_types: which images to load (ex: depth, semseg, etc)
            voxel_types: list of voxel attributes to load with the TSDF
            voxel_sizes: list of voxel sizes to load
            num_frames: number of evenly spaced frames to use (-1 for all)
        """

        self.data_path = data_path
        self.scene_id = scene_id
        self.transform = transform
        self.n_images = len([_ for _ in os.listdir(os.path.join(data_path, scene_id, 'sequence')) if _.endswith(".jpg")])


    def __len__(self):
        return self.n_images

    def __getitem__(self, i):
        """
        Returns:
            dict of meta data and images for a single frame
        """

        frame = self.map_frame(i)

        # put data in common format so we can apply transforms
        data = {'frames': [frame]}
        if self.transform is not None:
            data = self.transform(data)
        # remove data from common format and return the single frame
        data = data['frames'][0]
        return data


    def map_frame(self, i):
        """ Load images and metadata for a single frame.

        Given an info json we use this to load the images, etc for a single frame

        Args:
            frame: dict with metadata and paths to image files
            (see datasets/README)

        Returns:
            dict containg metadata plus the loaded image
        """
        idx = str(i).zfill(6)
        intrinsic_path = os.path.join(self.data_path, self.scene_id, 'sequence', '_info.txt')
        extrinsic_path = os.path.join(self.data_path, self.scene_id, 'sequence', 'frame-' + idx + ".pose.txt")
        image_path = os.path.join(self.data_path, self.scene_id, 'sequence', 'frame-' + idx + ".color.jpg")
        depth_path = os.path.join(self.data_path, self.scene_id, 'sequence', 'frame-' + idx + ".depth.pgm")
        
        data = {}
        data['id'] = i
        data['image'] = Image.open(image_path)
        data['intrinsics_color'] = self.read_intrinsic(intrinsic_path, 'm_calibrationColorIntrinsic')[:3, :3]
        data['intrinsics_depth'] = self.read_intrinsic(intrinsic_path, 'm_calibrationDepthIntrinsic')[:3, :3]
        data['pose'] = np.loadtxt(extrinsic_path, dtype=np.float32)
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32) / 1000.0
        data['depth'] = Image.fromarray(depth)
                
        if not np.all(np.isfinite(data['pose'])):
            data['valid'] = False
            print(self.scene_id + '/image_' + str(data['id']) + ' is invalid, removed!')
        else:
            data['valid'] = True  
            
        return data
    
    def read_intrinsic(self, data_path, string):
        intrinsic = np.eye(4)
        if os.path.exists(data_path):
            lines = open(data_path).readlines()
            for line in lines:
                if string in line:
                    intrinsic = [
                        float(x)
                        for x in line.rstrip().strip(string + ' = ').split(' ')
                    ]
                    break    
        intrinsic = np.array(intrinsic, dtype=np.float32).reshape((4, 4))
        return intrinsic