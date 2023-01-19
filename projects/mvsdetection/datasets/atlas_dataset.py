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
class AtlasScanNetDataset(Custom3DDataset):
    def __init__(self, data_root, ann_file, pipeline=None, classes=None, test_mode=False, num_frames=50):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes,
                         modality={'use_depth':True, 'use_camera':True}, box_type_3d='Depth', filter_empty_gt=False, test_mode=test_mode)
        self.num_frames = num_frames
    
    def read_scene_volumes(self, data_path, scene, voxel_size, vol_origin):
        full_tsdf_list = []
        for i in range(3):
            tsdf_path = os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(i))
            current_voxel_size = voxel_size * (2 ** i)
            full_tsdf = TSDF.load(tsdf_path, current_voxel_size, vol_origin)
            full_tsdf_list.append(full_tsdf)
        return full_tsdf_list
            
    def get_data_info(self, index):
        info = self.data_infos[index]
        imgs = [] 
        extrinsics = []
        intrinsics = []
        scene = info['scene']
        voxel_size = info['voxel_size']
        vol_origin = info['vol_origin']
        tsdf_list = self.read_scene_volumes(os.path.join(self.data_root, 'all_tsdf_9'), scene, voxel_size, vol_origin)
        annos = self.get_ann_info(index)
        image_ids = list(range(info['n_images']))
        random.shuffle(image_ids)

        loaded_num = 0
        for i, vid in enumerate(image_ids):
            if self.num_frames > 0 and loaded_num >= self.num_frames:
                break
            
            vid = str(int(vid)).zfill(5)
            img_path = os.path.join(self.data_root, 'posed_images', scene, vid + '.jpg')
            extrinsic_path = os.path.join(self.data_root, 'posed_images', scene, vid + '.txt')
            intrinsic_path = os.path.join(self.data_root, 'posed_images', scene, 'intrinsic.txt')
            
            img = Image.open(img_path)
            intrinsic = np.loadtxt(intrinsic_path, delimiter=' ')[:3, :3]
            intrinsic = intrinsic.astype(np.float32)
            extrinsic = np.loadtxt(extrinsic_path)
            axis_align_matrix = annos['axis_align_matrix']
            
            if extrinsic[0][0] == np.inf or extrinsic[0][0] == -np.inf or extrinsic[0][0] == np.nan:
                continue
    
            extrinsic = axis_align_matrix @ extrinsic 
            imgs.append(img)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            loaded_num += 1 
            
        
        items = {
            'imgs': imgs, 
            'intrinsics': intrinsics,
            'extrinsics': extrinsics, 
            'tsdf_list_full': tsdf_list,
            'vol_origin': info['vol_origin'],
            'scene': scene, 
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
        """
        Evaluate te results 
        Args:
            outputs [array of dicts]:
            {
                loss [torch float]: [total_loss]
                log_vars [dict of losses]: [all the sub losses]
                num_samples [int]: [the batch size]
                results [dict]:
                {
                    scene_name [str]: [scene_name]
                    origin [list, 3]: [origin of the predicted partial volume]
                    scene_tsdf [numpy float list]: [predicted tsdf volume]
                }
            }
        """
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for output in outputs:
            scene_id = output['scene']
            tsdf_pred = output['scene_tsdf']
            mesh_pred = tsdf_pred.get_mesh()
            if not os.path.exists(os.path.join(save_path, scene_id)):
                os.makedirs(os.path.join(save_path, scene_id))
            tsdf_pred.save(os.path.join(save_path, scene_id, scene_id + '.npz'))
            mesh_pred.export(os.path.join(save_path, scene_id, scene_id + '.ply'))
        '''
        return {}
    
    def __len__(self):
        return 4