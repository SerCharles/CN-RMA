import numpy as np 
import os 
import torch 
import cv2 
import pickle 
from PIL import Image 
from torch.utils.data import Dataset 
from mmdet.datasets import DATASETS 
from mmdet3d.datasets import Custom3DDataset 
from mmdet3d.core.bbox import DepthInstance3DBoxes 
from projects.mvsdetection.utils import save_results

@DATASETS.register_module()
class NeuconScanNetDataset(Custom3DDataset):
    def __init__(self, data_root, ann_file, pipeline=None, classes=None, test_mode=False):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes,
                         modality={'use_depth':True, 'use_camera':True}, box_type_3d='Depth', filter_empty_gt=False, test_mode=test_mode)
        self.tsdf_cache = {}
        self.max_cache = 100 
    
    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cache.keys():
            if len(self.tsdf_cache) > self.max_cache:
                self.tsdf_cache = {}
            full_tsdf_list = []
            for l in range(3):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)), allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cache[scene] = full_tsdf_list
        return self.tsdf_cache[scene]
            
    def get_data_info(self, index):
        info = self.data_infos[index]
        imgs = [] 
        depths = []
        extrinsics = []
        intrinsics = []
        scene = info['scene']
        tsdf_list = self.read_scene_volumes(os.path.join(self.data_root, 'all_tsdf_9'), scene)
        annos = self.get_ann_info(index)
        
        for i, vid in enumerate(info['image_ids']):
            img_path = os.path.join(self.data_root, 'posed_images', scene, str(vid).zfill(5) + '.jpg')
            depth_path = os.path.join(self.data_root, 'posed_images', scene, str(vid).zfill(5) + '.png')
            extrinsic_path = os.path.join(self.data_root, 'posed_images', scene, str(vid).zfill(5) + '.txt')
            intrinsic_path = os.path.join(self.data_root, 'posed_images', scene, 'intrinsic.txt')
            
            img = Image.open(img_path)
            depth = cv2.imread(depth_path, -1).astype(np.float32)
            depth /= 1000. 
            depth[depth > 3.0] = 0 
            intrinsic = np.loadtxt(intrinsic_path, delimiter=' ')[:3, :3]
            intrinsic = intrinsic.astype(np.float32)
            extrinsic = np.loadtxt(extrinsic_path)
            axis_align_matrix = annos['axis_align_matrix']
            extrinsic = axis_align_matrix @ extrinsic 
            
            imgs.append(img)
            depths.append(depth)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
        
        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)
        
        items = {
            'imgs': imgs, 
            'depths': depths, 
            'intrinsics': intrinsics,
            'extrinsics': extrinsics, 
            'tsdf_list_full': tsdf_list,
            'vol_origin': info['vol_origin'],
            'scene': scene, 
            'fragment': scene + '_' + str(info['fragment_id'])
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
        for output in outputs:
            if 'scene_name' in output.keys():
                if 'scene_tsdf' in output.keys():
                    save_results(output, voxel_size, save_path)
        return {}
    
