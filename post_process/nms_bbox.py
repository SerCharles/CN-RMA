import torch
from torch import nn
from mmdet3d.core.bbox import DepthInstance3DBoxes
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