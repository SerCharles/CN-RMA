# Modified from
# https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/scannet_data_utils.py
# Copyright (c) OpenMMLab, Inc. and its affiliates.
"""Aggregate ARKitScenes data into a pkl file for mmlab usage
Usage example: python ./aggregate_data.py
"""


import json 
import mmcv
import numpy as np
import os
import cv2
import copy
import glob
import argparse
from concurrent import futures as futures


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

class ARKitData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = os.path.join(root_path)
        self.classes = [
            "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
            "sink", "washer", "toilet", "bathtub", "oven", # 5..10
            "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
            "tv_monitor", "sofa", # 15..17
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val']
        if split == 'train':
            self.sample_id_list = [scene for scene in os.listdir(os.path.join(root_path, 'Training'))]
            self.split = 'Training'
        else:
            self.sample_id_list = [scene for scene in os.listdir(os.path.join(root_path, 'Validation'))]
            self.split = 'Validation'
    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = os.path.join(self.root_dir, 'arkit_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = os.path.join(self.root_dir, 'arkit_instance_data',
                            f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = os.path.join(self.root_dir, 'arkit_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)


    def read_2d_info(self, scene):
        data_path = os.path.join(self.root_dir, self.split, scene, scene + '_frames')
        
        #get image ids
        depth_folder = os.path.join(data_path, "lowres_depth")
        depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
        frame_ids = [os.path.basename(x) for x in depth_images]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        frame_ids = [x for x in frame_ids]
        frame_ids.sort()
        
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
        for frame_id in frame_ids:
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
        
        image_paths = {}
        depth_paths = {}
        extrinsics = {}
        intrinsics = {}
        total_image_ids = []

        for i, vid in enumerate(frame_ids):            
            intrinsic = copy.deepcopy(intrinsics_from_traj[str(vid)]).astype(np.float32)
            if str(vid) in poses_from_traj.keys():
                frame_pose = np.array(poses_from_traj[str(vid)])
            else:
                for my_key in list(poses_from_traj.keys()):
                    if abs(float(vid) - float(my_key)) < 0.005:
                        frame_pose = np.array(poses_from_traj[str(my_key)])
            extrinsic = copy.deepcopy(frame_pose).astype(np.float32)
            img_path = os.path.join(self.split, scene, scene + '_frames', 'lowres_wide', scene + '_' + vid + '.png')
            depth_path = os.path.join(self.split, scene, scene + '_frames', 'lowres_depth', scene + '_' + vid + '.png')
            if np.all(np.isfinite(extrinsic)):
                total_image_ids.append(vid)
                image_paths[vid] = img_path
                intrinsics[vid] = intrinsic
                extrinsics[vid] = extrinsic
                depth_paths[vid] = depth_path
            else:
                print(f'invalid extrinsic for {scene}_{vid}')
        
        return total_image_ids, image_paths, depth_paths, intrinsics, extrinsics

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            tsdf_path = os.path.join(self.root_dir, 'atlas_tsdf', sample_idx)
            info_path = os.path.join(tsdf_path, 'info.json')
            with open(info_path) as f:
                info = json.load(f)

            info['split'] = self.split
            total_image_ids, image_paths, depth_paths, intrinsics, extrinsics = self.read_2d_info(sample_idx)
            info['total_image_ids'] = total_image_ids
            info['image_paths'] = image_paths
            info['depth_paths'] = depth_paths
            info['intrinsics'] = intrinsics 
            info['extrinsics'] = extrinsics
                
            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 7
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                    axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                    annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                    info['annos'] = annotations
                else:
                    print('-' * 100)
                    print(info['split'] + '/' + info['scene'] + ' has no gt bbox, pass!')
                    print('-' * 100)
                    info = None
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        results = []
        for sample_idx in sample_id_list:
            info = process_single_scene(sample_idx)
            if info != None:
                results.append(info)
        return results

def create_indoor_info_file(data_path, save_path, workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str): Prefix of the pkl to be saved. Default: 'sunrgbd'.
        save_path (str): Path of the pkl to be saved. Default: None.
        use_v1 (bool): Whether to use v1. Default: False.
        workers (int): Number of threads to be used. Default: 4.
    """

    train_filename = os.path.join(save_path, 'arkit_infos_train.pkl')
    val_filename = os.path.join(save_path, 'arkit_infos_val.pkl')
    train_dataset = ARKitData(root_path=data_path, split='train')
    val_dataset = ARKitData(root_path=data_path, split='val')
    
    infos_train = train_dataset.get_infos(num_workers=workers, has_label=True)
    mmcv.dump(infos_train, train_filename, 'pkl')
    print(f'arkit info train file is saved to {train_filename}')
    infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
    mmcv.dump(infos_val, val_filename, 'pkl')
    print(f'arkit info val file is saved to {val_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/arkit', help='specify the root path of dataset')
    parser.add_argument('--save_path', type=str, default='./data/arkit', help='name of info pkl')
    #parser.add_argument('--data_path', type=str, default='/data1/sgl/ARKit', help='specify the root path of dataset')
    #parser.add_argument('--save_path', type=str, default='/data1/sgl/ARKit', help='name of info pkl')
    
    parser.add_argument('--workers', type=int, default=4, help='number of threads to be used')
    args = parser.parse_args()
    create_indoor_info_file(data_path=args.data_path, save_path=args.save_path, workers=args.workers)

