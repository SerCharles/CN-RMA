# Modified from
# https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
# Copyright (c) MagicLeap, Inc. and its affiliates.
"""Dataset for ARKitScenes used in data preparing
"""

import os
import numpy as np
from PIL import Image

import copy
import glob
import numpy as np
import os

from rotation import convert_angle_axis_to_matrix3


def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


class ARKitSimpleLoader(object):
    def __init__(self, data_path, scene_id, transform, split='Training'):
        assert split in ['Training', 'Validation']
        self.root_path = os.path.join(data_path, '3dod', split, scene_id, scene_id + '_frames')
        self.transform = transform
        
        depth_folder = os.path.join(self.root_path, "lowres_depth")

        depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
        self.frame_ids = [os.path.basename(x) for x in depth_images]
        self.frame_ids = [x.split(".png")[0].split("_")[1] for x in self.frame_ids]
        self.video_id = depth_folder.split('/')[-3]
        self.frame_ids = [x for x in self.frame_ids]
        self.frame_ids.sort()
        self.intrinsics = {}

        traj_file = os.path.join(self.root_path, 'lowres_wide.traj')
        with open(traj_file) as f:
            self.traj = f.readlines()
        # convert traj to json dict
        poses_from_traj = {}
        for line in self.traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1].tolist()

        if os.path.exists(traj_file):
            # self.poses = json.load(open(traj_file))
            self.poses = poses_from_traj
        else:
            self.poses = {}

        # get intrinsics
        for frame_id in self.frame_ids:
            intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics", f"{self.video_id}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("frame_id", frame_id)
                print(intrinsic_fn)
            self.intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)
        


    def __len__(self):
        return len(self.frame_ids)

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
        
    def map_frame(self, idx):
        """
        Returns:
            frame: a dict
                {id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["id"] = frame_id
        fname = "{}_{}.png".format(self.video_id, frame_id)
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "lowres_depth", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path)

        image_path = os.path.join(self.root_path, "lowres_wide", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path, "does not exist")
        
        frame['image'] = Image.open(image_path)
        depth = Image.open(depth_image_path)
        depth = np.array(depth, dtype=np.float32) / 1000.0
        frame['depth'] = Image.fromarray(depth)

        '''
        frame["depth"] = cv2.imread(depth_image_path, -1)
        frame["image"] = cv2.imread(image_path)
        depth_height, depth_width = frame["depth"].shape
        im_height, im_width, im_channels = frame["image"].shape
        if depth_height != im_height:
            frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
            frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.imread(image_path)
        frame["depth"] = frame["depth"].astype(np.float32) / 1000.0
        '''

        frame["intrinsics"] = copy.deepcopy(self.intrinsics[frame_id]).astype(np.float32)
        if str(frame_id) in self.poses.keys():
            frame_pose = np.array(self.poses[str(frame_id)])
        else:
            for my_key in list(self.poses.keys()):
                if abs(float(frame_id) - float(my_key)) < 0.005:
                    frame_pose = np.array(self.poses[str(my_key)])
        frame["pose"] = copy.deepcopy(frame_pose).astype(np.float32)
        



        if not np.all(np.isfinite(frame['pose'])):
            frame['valid'] = False
            print(image_path + '  is invalid, removed!')
        else:
            frame['valid'] = True

        return frame
    
