# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load Scannet scenes with vertices and ground truth labels for semantic and
instance segmentations."""


import argparse
from math import *
import inspect
import json
import numpy as np
import os
import datetime
from plyfile import PlyData

from box_utils import compute_box_3d, boxes_to_corners_3d, corners_to_boxes, get_size

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))

class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
    "sink", "washer", "toilet", "bathtub", "oven", # 5..10
    "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
    "tv_monitor", "sofa", # 15..17
]
#switch labels
label2cls = {}
cls2label = {}
for i, cls_ in enumerate(class_names):
    label2cls[i] = cls_
    cls2label[cls_] = i


def read_mesh_vertices(filename):
    """Read XYZ and RGB for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        Vertices. Note that RGB values are in 0-255.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices




def extract_bbox_infos(gt_fn):
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


def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    data_path,
                    split='Training'):
    assert split in ['Training', 'Validation']
    mesh_file = os.path.join(data_path, '3dod', split, scan_name, scan_name + '_3dod_mesh.ply')
    bbox_file = os.path.join(data_path, '3dod', split, scan_name, scan_name + "_3dod_annotation.json")

    #get points, along with empty align matrix, instance and semantic labels
    mesh_vertices = read_mesh_vertices(mesh_file)
    axis_align_matrix = np.eye(4)
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]], axis=1)
    semantic_labels = np.zeros_like(mesh_vertices)
    instance_labels = np.zeros_like(mesh_vertices)

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            semantic_labels = semantic_labels[choices]
            instance_labels = instance_labels[choices]
   
    #get N*8 bbox
    skipped, boxes_corners, centers, sizes, labels, uids = extract_bbox_infos(bbox_file)
    if skipped or boxes_corners.shape[0] == 0:
        print('Error! No care instances found!')
        return False
    n_gt = boxes_corners.shape[0]
    boxes = corners_to_boxes(boxes_corners)
    label_ids = []
    for l in labels:
        label_ids.append(cls2label[l])
    label_ids = np.array(label_ids).reshape(-1, 1)
    unaligned_bboxes = np.concatenate([boxes, label_ids], axis=1) #N * 8  
    aligned_bboxes = unaligned_bboxes.copy()
    print(f'Num of care instances: {unaligned_bboxes.shape[0]}')


    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
    np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
    np.save(f'{output_filename_prefix}_unaligned_bbox.npy',unaligned_bboxes)
    np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
    np.save(f'{output_filename_prefix}_axis_align_matrix.npy', axis_align_matrix)
    return True


def batch_export(max_num_point,
                 output_folder,
                 data_dir,
                 save_names_file,
                 split='Training'):
    assert split in ['Training', 'Validation']
    
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = os.listdir(os.path.join(data_dir, '3dod', split))
    scan_names.sort()
    save_names = []
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(output_folder, scan_name)
        if os.path.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        if export_one_scan(scan_name, output_filename_prefix, max_num_point, data_dir, split):
            save_names.append(scan_name)
        print('-' * 20 + 'done')

    with open(save_names_file, 'w') as f:
        for i in range(len(save_names)):
            item = save_names[i]
            if i != len(save_names) - 1:
                f.write("%s\n" % item)
            else:
                f.write("%s" % item)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='/data1/sgl/ARKit/arkit_instance_data')
    parser.add_argument(
        '--data_path', default='/data1/sgl/ARKit')
    parser.add_argument(
        '--train_names_file',
        default='/data1/sgl/ARKit/meta_data/arkit_train.txt',
        help='The path of the file that stores the scan names.')
    parser.add_argument(
        '--val_names_file',
        default='/data1/sgl/ARKit/meta_data/arkit_val.txt',
        help='The path of the file that stores the scan names.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.data_path,
        args.train_names_file,
        'Training')
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.data_path,
        args.val_names_file,
        'Validation')


if __name__ == '__main__':
    main()
