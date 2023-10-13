import numpy as np 
import os 
import torch 
import cv2 
import pickle 
import warnings
import random
from PIL import Image 
from torch.utils.data import Dataset 
from mmdet.datasets import DATASETS 
from mmdet3d.datasets import Custom3DDataset 
from mmdet3d.core.bbox import DepthInstance3DBoxes 
from projects.mvsdetection.utils import save_results
from projects.mvsdetection.datasets.tsdf import TSDF

@DATASETS.register_module()
class AtlasRScanDataset(Custom3DDataset):
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

    def read_intrinsic(self, data_path):
        intrinsic = np.eye(4)
        if os.path.exists(data_path):
            lines = open(data_path).readlines()
            for line in lines:
                if 'm_calibrationColorIntrinsic' in line:
                    intrinsic = [
                        float(x)
                        for x in line.rstrip().strip('m_calibrationColorIntrinsic = ').split(' ')
                    ]
                    break    
        intrinsic = np.array(intrinsic, dtype=np.float32).reshape((4, 4))
        return intrinsic
            
    def get_data_info(self, index):
        info = self.data_infos[index]
        imgs = [] 
        extrinsics = []
        intrinsics = []
        
        scene = info['scene']
        
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
        
        tsdf_dict = self.read_scene_volumes(os.path.join(self.data_root, 'atlas_tsdf'), scene, self.voxel_size)
        annos = self.get_ann_info(index)
        
        
        for i, vid in enumerate(image_ids):
            vid = str(int(vid)).zfill(6)
            img_path = os.path.join(self.data_root, 'scans', scene, 'sequence', 'frame-' + vid + '.color.jpg')
            extrinsic_path = os.path.join(self.data_root, 'scans', scene, 'sequence', 'frame-' + vid + '.pose.txt')
            intrinsic_path = os.path.join(self.data_root, 'posed_images', scene, 'sequence', '_info.txt')
            
            img = Image.open(img_path)
            intrinsic = self.read_intrinsic(intrinsic_path)[:3, :3]
            intrinsic = intrinsic.astype(np.float32)
            extrinsic = np.loadtxt(extrinsic_path)
            #axis_align_matrix = annos['axis_align_matrix']
            #extrinsic = axis_align_matrix @ extrinsic 
            if not np.isfinite(extrinsic).all():
                print(scene, vid, 'is invalid!')
                raise ValueError
            
            imgs.append(img)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            
        
        items = {
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
        
        if 'axis_align_matrix' in info['annos'].keys():
            axis_align_matrix = info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            axis_align_matrix = np.eye(4).astype(np.float32)
            warnings.warn('Axis align matrix is invalid, set to I')
        
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(np.float32) #K * 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)
        gt_bboxes_3d = DepthInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], with_yaw=False, 
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
