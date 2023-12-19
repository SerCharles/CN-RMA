#TODO: remove sys.path based import

#!/usr/bin/env python3

import argparse
import numpy as np
import os
import json


from box_utils import compute_box_3d, boxes_to_corners_3d, corners_to_boxes, get_size

class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
    "sink", "washer", "toilet", "bathtub", "oven", # 5..10
    "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
    "tv_monitor", "sofa", # 15..17
]




def extract_gt(gt_fn):
    """extract original label data

    Args:
        gt_fn: str (file name of "annotation.json")
            after loading, we got a dict with keys
                'data', 'stats', 'comment', 'confirm', 'skipped'
            ['data']: a list of dict for bboxes, each dict has keys:
                'uid', 'label', 'modelId', 'children', 'objectId',
                'segments', 'hierarchy', 'isInGroup', 'labelType', 'attributes'
                'label': str
                'segments': dict for boxes
                    'centroid': list of float (x, y, z)?
                    'axesLengths': list of float (x, y, z)?
                    'normalizedAxes': list of float len()=9
                'uid'
            'comments':
            'stats': ...
    Returns:
        skipped: bool
            skipped or not
        boxes_corners: (n, 8, 3) box corners
            **world-coordinate**
        centers: (n, 3)
            **world-coordinate**
        sizes: (n, 3) full-sizes (no halving!)
        labels: list of str
        uids: list of str
    """
    gt = json.load(open(gt_fn, "r"))
    skipped = gt['skipped']
    if len(gt) == 0:
        boxes_corners = np.zeros((0, 8, 3))
        centers = np.zeros((0, 3))
        sizes = np.zeros((0, 3))
        labels, uids = [], []
        return skipped, boxes_corners, centers, sizes, labels, uids

    boxes_corners = []
    centers = []
    sizes = []
    labels = []
    uids = []
    for data in gt['data']:
        l = data["label"]
        for delimiter in [" ", "-", "/"]:
            l = l.replace(delimiter, "_")
        if l not in class_names:
            print("unknown category: %s" % l)
            continue

        rotmat = np.array(data["segments"]["obbAligned"]["normalizedAxes"]).reshape(
            3, 3
        )
        center = np.array(data["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        size = np.array(data["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(size.reshape(3).tolist(), center, rotmat)

        '''
            Box corner order that we return is of the format below:
                6 -------- 7
               /|         /|
              5 -------- 4 .
              | |        | |
              . 2 -------- 3
              |/         |/
              1 -------- 0 
        '''

        boxes_corners.append(box3d.reshape(1, 8, 3))
        size = np.array(get_size(box3d)).reshape(1, 3)
        center = np.mean(box3d, axis=0).reshape(1, 3)

        # boxes_corners.append(box3d.reshape(1, 8, 3))
        centers.append(center)
        sizes.append(size)
        # labels.append(l)
        labels.append(data["label"])
        uids.append(data["uid"])
    centers = np.concatenate(centers, axis=0)
    sizes = np.concatenate(sizes, axis=0)
    boxes_corners = np.concatenate(boxes_corners, axis=0)
    return skipped, boxes_corners, centers, sizes, labels, uids






if __name__ == "__main__":
    align_bbox = np.load('/data1/sgl/ScanNet/scannet_instance_data/scene0000_00_aligned_bbox.npy')
    ins_label = np.load('/data1/sgl/ScanNet/scannet_instance_data/scene0000_00_ins_label.npy')
    sem_label = np.load('/data1/sgl/ScanNet/scannet_instance_data/scene0000_00_sem_label.npy')
    vert = np.load('/data1/sgl/ScanNet/scannet_instance_data/scene0000_00_vert.npy')
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data1/sgl/ARKit/3dod")
    parser.add_argument('--split', default='Validation')
    parser.add_argument("--scene_id", default="41069021")
    parser.add_argument("--output_dir", default="/data1/sgl/ARKit/bbox_gt")

    args = parser.parse_args()
    gt_path = os.path.join(args.data_root, args.split, args.scene_id, args.scene_id + "_3dod_annotation.json")
    output_path = os.path.join(args.output_dir, args.scene_id, args.scene_id + "_gt.npz")
    if not os.path.exists(os.path.join(args.output_dir, args.scene_id)):
        os.makedirs(os.path.join(args.output_dir, args.scene_id))

    #get annotation
    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(gt_path)
    if skipped or boxes_corners.shape[0] == 0:
        exit()
    n_gt = boxes_corners.shape[0]

    #get N*7 bbox
    boxes = corners_to_boxes(boxes_corners)
    corners_recon = boxes_to_corners_3d(boxes)
    
    #switch labels
    label2cls = {}
    cls2label = {}
    for i, cls_ in enumerate(class_names):
        label2cls[i] = cls_
        cls2label[cls_] = i
    num_class = len(class_names)

    label_ids = []
    for l in labels:
        label_ids.append(cls2label[l])
    label_ids = np.array(label_ids)
    scores = np.ones_like(label_ids)
    np.savez(output_path, boxes=boxes, labels=label_ids, scores=scores)
    