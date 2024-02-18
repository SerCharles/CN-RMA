# Modified from
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/scannet_dataset.py
# Copyright (c) OpenMMLab, Inc. and its affiliates.
"""ARKitScenes Dataset
"""


import numpy as np 
import os 
import torch 
import cv2 
import pickle 
import copy
import warnings
import random
from PIL import Image 
from torch.utils.data import Dataset 
from mmdet.datasets import DATASETS 
from mmdet3d.datasets import Custom3DDataset 
from mmdet3d.core.bbox import DepthInstance3DBoxes 
from projects.mvsdetection.datasets.tsdf import TSDF

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

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



@DATASETS.register_module()
class AtlasARKitDataset(Custom3DDataset):
    def __init__(self, data_root, ann_file, pipeline=None, classes=None, test_mode=False, num_frames=50, voxel_size=0.04, select_type='random'):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes,
                         modality={'use_depth':True, 'use_camera':True}, box_type_3d='Depth', filter_empty_gt=False, test_mode=test_mode)
        self.num_frames = num_frames
        self.voxel_size = voxel_size
        self.data_infos = sorted(self.data_infos, key=lambda x: x['scene'])
        self.select_type = select_type

    def read_scene_volumes(self, data_path, scene, voxel_size):
        full_tsdf_dict = {}
        for i in range(3):
            current_voxel_size = voxel_size * (2 ** i)
            current_file_key = 'tsdf_gt_' + str(int(current_voxel_size * 100)).zfill(3) #004, 008, 016
            old_key = 'tsdf_' + str(int(current_voxel_size * 100)).zfill(2) + '.npz' #04 08 16
            raw_tsdf = np.load(os.path.join(data_path, scene, old_key), allow_pickle=True)
            vol_origin = torch.as_tensor(raw_tsdf['origin']).view(1, 3)
            tsdf_vol = torch.as_tensor(raw_tsdf['tsdf'])
            full_tsdf = TSDF(current_voxel_size, vol_origin, tsdf_vol) 
            full_tsdf_dict[current_file_key] = full_tsdf
        return full_tsdf_dict
    
            
    def get_data_info(self, index):
        info = self.data_infos[index]
        imgs = [] 
        extrinsics = []
        intrinsics = []
        
        scene = info['scene']
        split = info['split']

        
        tsdf_dict = self.read_scene_volumes(os.path.join(self.data_root, 'atlas_tsdf'), scene, self.voxel_size)
        annos = self.get_ann_info(index)
        
        
        #select type: 
        #unit:use per n pictures
        #random:use random n pictures
        total_image_ids = info['total_image_ids']
        if self.num_frames <= 0 or self.num_frames > len(total_image_ids):
            image_ids = total_image_ids
        elif self.select_type == 'random':
            image_ids = random.sample(total_image_ids, self.num_frames)
        elif self.select_type == 'unit':
            m = len(total_image_ids)
            n = self.num_frames
            image_ids = []
            k = (m - 1) // (n - 1)
            image_ids = []
            for i in range(n):
                image_ids.append(total_image_ids[i * k])
        image_ids.sort()
        
        if not 'image_paths' in info.keys(): 
            data_path = os.path.join(self.data_root, split, scene, scene + '_frames')
            #read extrinsics
            traj_file = os.path.join(data_path, 'lowres_wide.traj')
            with open(traj_file) as f:
                self.traj = f.readlines()
            # convert traj to json dict
            poses_from_traj = {}
            for line in self.traj:
                traj_timestamp = line.split(" ")[0]
                poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1].tolist()

            # get intrinsics
            intrinsics_from_traj = {}
            for frame_id in image_ids:
                intrinsic_fn = os.path.join(data_path, "lowres_wide_intrinsics", f"{scene}_{frame_id}.pincam")
                if not os.path.exists(intrinsic_fn):
                    intrinsic_fn = os.path.join(data_path, "lowres_wide_intrinsics",
                                            f"{scene}_{float(frame_id) - 0.001:.3f}.pincam")
                if not os.path.exists(intrinsic_fn):
                    intrinsic_fn = os.path.join(data_path, "lowres_wide_intrinsics",
                                            f"{scene}_{float(frame_id) + 0.001:.3f}.pincam")
                if not os.path.exists(intrinsic_fn):
                    print("frame_id", frame_id)
                    print(intrinsic_fn)
                intrinsics_from_traj[frame_id] = st2_camera_intrinsics(intrinsic_fn)
        
        
            for i, vid in enumerate(image_ids):            
                img_path = os.path.join(data_path, 'lowres_wide', scene + '_' + vid + '.png')
                img = Image.open(img_path)
                '''
                depth_path = os.path.join(data_path, 'lowres_depth', scene + '_' + vid + '.png')
                depth = Image.open(depth_path)
                depth = np.array(depth, dtype=np.float32) / 1000.0
                '''

                intrinsic = copy.deepcopy(intrinsics_from_traj[str(vid)]).astype(np.float32)
                if str(vid) in poses_from_traj.keys():
                    frame_pose = np.array(poses_from_traj[str(vid)])
                else:
                    for my_key in list(poses_from_traj.keys()):
                        if abs(float(vid) - float(my_key)) < 0.005:
                            frame_pose = np.array(poses_from_traj[str(my_key)])
                extrinsic = copy.deepcopy(frame_pose).astype(np.float32)
            
                if not np.isfinite(extrinsic).all():
                    print(scene, vid, 'is invalid!')
                    raise ValueError
            
                imgs.append(img)
                intrinsics.append(intrinsic)
                extrinsics.append(extrinsic)
        else:
            for i, vid in enumerate(image_ids):
                img_path = os.path.join(self.data_root, info['image_paths'][vid])
                img = Image.open(img_path)
                
                '''
                depth_path = os.path.join(self.data_root, info['depth_paths'][vid])
                depth = Image.open(depth_path)
                depth = np.array(depth, dtype=np.float32) / 1000.0
                '''
                intrinsic = info['intrinsics'][vid].astype(np.float32)
                extrinsic = info['extrinsics'][vid].astype(np.float32)
                
                imgs.append(img)
                intrinsics.append(intrinsic)
                extrinsics.append(extrinsic)

                
        
        items = {
            'split': split,
            'scene': scene, 
            'image_ids': image_ids,
            'imgs': imgs, 
            'intrinsics': intrinsics,
            'extrinsics': extrinsics, 
            'tsdf_dict': tsdf_dict,        
        }
        items['ann_info'] = annos 
        return items
            
    def get_ann_info(self, index):
        info = self.data_infos[index]
        
        if 'annos' not in info.keys():
            return None
        
        if 'axis_align_matrix' in info['annos'].keys():
            axis_align_matrix = info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            axis_align_matrix = np.eye(4).astype(np.float32)
            warnings.warn('Axis align matrix is invalid, set to I')
        
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(np.float32) #K * 7
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)
        gt_bboxes_3d = DepthInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], with_yaw=True, 
                                            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        ann_results = dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, axis_align_matrix=axis_align_matrix)
        return ann_results
    
    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example 

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example 
    
    def evaluate(self, outputs, voxel_size=0.04, save_path='./work_dir', logger=None):
        return {}
