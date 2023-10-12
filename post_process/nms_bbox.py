import os
import argparse 
import numpy as np
import torch
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.core import bbox3d2result
from mmdet3d.ops.pcdet_nms import pcdet_nms_gpu, pcdet_nms_normal_gpu

def nms(bboxes, scores, score_thr=0.01, iou_thr=0.5):
    n_classes = scores.shape[1]
    yaw_flag = bboxes.shape[1] == 7
    nms_bboxes, nms_scores, nms_labels = [], [], []
    for i in range(n_classes):
        ids = scores[:, i] > score_thr
        if not ids.any():
            continue

        class_scores = scores[ids, i]
        class_bboxes = bboxes[ids]
        if yaw_flag:
            nms_function = pcdet_nms_gpu
        else:
            class_bboxes = torch.cat((
                class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
            nms_function = pcdet_nms_normal_gpu

        nms_ids, _ = nms_function(class_bboxes, class_scores, iou_thr)
        nms_bboxes.append(class_bboxes[nms_ids])
        nms_scores.append(class_scores[nms_ids])
        nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

    if len(nms_bboxes):
        nms_bboxes = torch.cat(nms_bboxes, dim=0)
        nms_scores = torch.cat(nms_scores, dim=0)
        nms_labels = torch.cat(nms_labels, dim=0)
    else:
        nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
        nms_scores = bboxes.new_zeros((0,))
        nms_labels = bboxes.new_zeros((0,))

    if yaw_flag:
        box_dim = 7
        with_yaw = True
    else:
        box_dim = 6
        with_yaw = False
        nms_bboxes = nms_bboxes[:, :6]
    nms_bboxes = DepthInstance3DBoxes(nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(.5, .5, .5))

    return nms_bboxes, nms_scores, nms_labels

def save_bbox(args, scene_id, bbox_results):
    save_path = os.path.join(args.result_path, scene_id, scene_id + args.postfix)
    bboxes = bbox_results['boxes_3d'].tensor.detach().cpu().numpy()
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 5] / 2
    scores = bbox_results['scores_3d'].detach().cpu().numpy()
    labels = bbox_results['labels_3d'].detach().cpu().numpy()
    np.savez(save_path, boxes=bboxes, scores=scores, labels=labels)
    print('Saved', scene_id)

def nms_bboxes(args):
    scene_files = os.listdir(args.result_path)
    scene_ids = []
    for scene_file in scene_files:
        raw_path = os.path.join(args.result_path, scene_file, scene_file + '_bbox_raw.npz')
        if scene_file[:5] == 'scene' and os.path.exists(raw_path):
                scene_ids.append(scene_file)
    scene_ids.sort()
    
    for scene_id in scene_ids:
        raw_path = os.path.join(args.result_path, scene_id, scene_id + '_bbox_raw.npz')
        data = np.load(raw_path)
        bboxes = torch.tensor(data['bboxes']).cuda()
        scores = torch.tensor(data['scores']).cuda()
        bboxes, scores, labels = nms(bboxes=bboxes, scores=scores)
        bbox_results = bbox3d2result(bboxes, scores, labels)            
        save_bbox(args, scene_id, bbox_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default='/home/sgl/work_dirs_atlas/ray_marching_300_010/results')
    parser.add_argument("--postfix", type=str, default='_atlas_bbox.npz')
    args = parser.parse_args()
    nms_bboxes(args)