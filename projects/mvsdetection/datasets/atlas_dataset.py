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
    def __init__(self, data_root, ann_file, pipeline=None, classes=None, test_mode=False, num_frames=50, voxel_size=0.04):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes,
                         modality={'use_depth':True, 'use_camera':True}, box_type_3d='Depth', filter_empty_gt=False, test_mode=test_mode)
        self.num_frames = num_frames
        self.voxel_size = voxel_size
        self.data_infos = sorted(self.data_infos, key=lambda x: x['scene'])
    
    def read_scene_volumes(self, data_path, scene, voxel_size, vol_origin):
        full_tsdf_dict = {}
        for i in range(3):
            current_voxel_size = voxel_size * (2 ** i)
            current_file_key = 'tsdf_gt_' + str(int(current_voxel_size * 100)).zfill(3) #004, 008, 016
            raw_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(i)), allow_pickle=True)
            vol_origin = torch.as_tensor(vol_origin).view(1, 3)
            tsdf_vol = torch.as_tensor(raw_tsdf.f.arr_0)
            full_tsdf = TSDF(voxel_size, vol_origin, tsdf_vol) 
            full_tsdf_dict[current_file_key] = full_tsdf
        return full_tsdf_dict
    
    def read_atlas_volumes(self, data_path, scene, voxel_size):
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
        image_ids = info['image_ids']
        '''
        total_image_ids = info['total_image_ids']
        if self.num_frames > 0:
            image_ids = random.sample(total_image_ids, self.num_frames)
        else:
            image_ids = total_image_ids
        image_ids.sort()
        '''
        
        
        #tsdf_dict = self.read_scene_volumes(os.path.join(self.data_root, 'all_tsdf_9'), scene, info['voxel_size'], info['vol_origin'])
        tsdf_dict = self.read_atlas_volumes(os.path.join(self.data_root, 'atlas'), scene, info['voxel_size'])
        annos = self.get_ann_info(index)
        

        for i, vid in enumerate(image_ids):
            vid = str(int(vid)).zfill(5)
            img_path = os.path.join(self.data_root, 'posed_images', scene, vid + '.jpg')
            extrinsic_path = os.path.join(self.data_root, 'posed_images', scene, vid + '.txt')
            intrinsic_path = os.path.join(self.data_root, 'posed_images', scene, 'intrinsic.txt')
            
            img = Image.open(img_path)
            intrinsic = np.loadtxt(intrinsic_path, delimiter=' ')[:3, :3]
            intrinsic = intrinsic.astype(np.float32)
            extrinsic = np.loadtxt(extrinsic_path)
            axis_align_matrix = annos['axis_align_matrix']
            #extrinsic = axis_align_matrix @ extrinsic 
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
            'voxel_size': info['voxel_size'],
            'vol_origin': info['vol_origin'],            
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
