import os 
import argparse 
import numpy as np 
import torch 
from mmdet3d.core.evaluation import indoor_eval
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox import DepthInstance3DBoxes
from glob import glob 

def evaluate_bbox(dataset, data_path, result_path, postfix):
    iou_thr=(0.25, 0.5)
    box_type_3d, box_mode_3d = get_box_type('Depth')
    
    if dataset == 'scannet' or dataset == '3rscan':
        classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                  'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                  'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                  'garbagebin']
        cat2label = {cat: classes.index(cat) for cat in classes}
        label2cat = {cat2label[t]: t for t in cat2label}
        cat_ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        cat_ids2classes = {
            nyu40id: i 
            for i, nyu40id in enumerate(list(cat_ids))
        }
    else:
        classes = [
            "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
            "sink", "washer", "toilet", "bathtub", "oven", # 5..10
            "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
            "tv_monitor", "sofa", # 15..17
        ]
        cat2label = {cat: classes.index(cat) for cat in classes}
        label2cat = {cat2label[t]: t for t in cat2label}
        cat_ids = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        cat_ids2classes = {
            nyu40id: i
            for i, nyu40id in enumerate(list(cat_ids))
        }
    
    scene_files = os.listdir(result_path)
    scene_ids = scene_files
    
    scene_ids.sort()
            
    results = [] 
    for scene_id in scene_ids:
        box_path = os.path.join(result_path, scene_id, scene_id + postfix + '.npz')
        result = {} 
        bbox_data = np.load(box_path)
        bboxes = bbox_data['boxes']
        scores = bbox_data['scores']
        labels = bbox_data['labels']
        bboxes = DepthInstance3DBoxes(
            bboxes,
            box_dim=bboxes.shape[-1],
            with_yaw=False, 
            origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
        result['boxes_3d'] = bboxes
        result['labels_3d'] = torch.Tensor(labels).long()
        result['scores_3d'] = torch.Tensor(scores)
        results.append(result)
    
    gt_annos = [] 
    for scene_id in scene_ids:
        gt_anno = {}
        if dataset == 'scannet':
            box_file = os.path.join(data_path, 'scannet_instance_data', scene_id + '_aligned_bbox.npy')
        elif dataset == '3rscan':
            box_file = os.path.join(data_path, 'recon_instance_data_aabb', scene_id + '_aligned_bbox.npy')
        elif dataset == 'arkit':
            box_file = os.path.join(data_path, 'arkit_instance_data', scene_id + '_aligned_bbox.npy')
        aligned_box_label = np.load(box_file)
        gt_num = aligned_box_label.shape[0]
        gt_anno['gt_num'] = gt_num 
        if gt_num > 0:
            aligned_box = aligned_box_label[:, :-1]
            classes = aligned_box_label[:, -1]
            gt_labels = np.array([
                cat_ids2classes[classes[i]]
                for i in range(gt_num)
            ])
            gt_anno['gt_boxes_upright_depth'] = aligned_box
            gt_anno['class'] = gt_labels 
        gt_annos.append(gt_anno)
    
    ret_dict = indoor_eval(
        gt_annos,
        results,
        iou_thr,
        label2cat,
        logger=None,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='arkit')
    parser.add_argument("--data_path", type=str, default='/data1/sgl/ARKit')
    parser.add_argument("--result_path", type=str, default='/data1/sgl/work_dirs_atlas/arkit_fcaf3d_two_stage/results')
    parser.add_argument("--postfix", type=str, default='_fcaf3d_retrain')
    args = parser.parse_args()
    evaluate_bbox(args.dataset, args.data_path, args.result_path, args.postfix)