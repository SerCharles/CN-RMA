# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load Scannet scenes with vertices and ground truth labels for semantic and
instance segmentations."""
import argparse
import inspect
import json
import numpy as np
import os
import os.path as osp
import datetime
import scannet_utils

DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId']  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, object_id_to_segs, object_id_to_label_id,
                 instance_ids):
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    instance_bboxes = np.zeros((num_instances, 7))
    keys = list(object_id_to_segs.keys())
    for i in range(len(keys)):
        obj_id = keys[i]
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        xyz_min = np.min(obj_pc, axis=0)
        xyz_max = np.max(obj_pc, axis=0)
        bbox = np.concatenate([(xyz_min + xyz_max) / 2.0, xyz_max - xyz_min,
                               np.array([label_id])])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        #instance_bboxes[obj_id - 1, :] = bbox
        instance_bboxes[i, :] = bbox
    return instance_bboxes


def export(mesh_file,
           agg_file,
           seg_file,
           label_map_file,
           output_file=None):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        agg_file (str): Path of the agg_file.
        seg_file (str): Path of the seg_file.
        label_map_file (str): Path of the label_map_file.
        output_file (str): Path of the output folder.
            Default: None.


    It returns a tuple, which containts the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """

    label_map = scannet_utils.read_label_mapping(
        label_map_file, label_from='Label', label_to='nyu40')
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    axis_align_matrix = np.eye(4)
    # perform global alignment of mesh vertices
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]], axis=1)

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs,
                                        object_id_to_label_id, instance_ids)
    aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs,
                                      object_id_to_label_id, instance_ids)

    if output_file is not None:
        np.save(output_file + '_vert.npy', mesh_vertices)
        np.save(output_file + '_sem_label.npy', label_ids)
        np.save(output_file + '_ins_label.npy', instance_ids)
        np.save(output_file + '_unaligned_bbox.npy', unaligned_bboxes)
        np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)
        np.save(output_file + '_axis_align_matrix.npy', axis_align_matrix)

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, object_id_to_label_id, axis_align_matrix




def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    data_path):
    mesh_file = osp.join(data_path, scan_name, 'labels.instances.annotated.v2.ply')
    agg_file = osp.join(data_path, scan_name, 'semseg.v2.json')
    seg_file = osp.join(data_path, scan_name, 'mesh.refined.0.010000.segs.v2.json')
    # includes axisAlignment info for the train set scans.
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, label_map_file, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask, :]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print(f'Num of instances: {num_instances}')

    bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
    unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
    bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
    aligned_bboxes = aligned_bboxes[bbox_mask, :]
    assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
    print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            semantic_labels = semantic_labels[choices]
            instance_labels = instance_labels[choices]

    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
    np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
    np.save(f'{output_filename_prefix}_unaligned_bbox.npy',unaligned_bboxes)
    np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
    np.save(f'{output_filename_prefix}_axis_align_matrix.npy', axis_align_matrix)


def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 label_map_file,
                 data_dir):
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        #try:
        export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            label_map_file, data_dir)
        #except Exception:
        #    print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='/data1/sgl/3RScan/scannet_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--data_path', default='/data1/sgl/3RScan/scans', help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='/data1/sgl/3RScan/meta_data/3rscan_mapping.tsv',
        help='The path of label map file.')
    parser.add_argument(
        '--train_scan_names_file',
        default='/data1/sgl/3RScan/meta_data/train.txt',
        help='The path of the file that stores the scan names.')
    parser.add_argument(
        '--test_scan_names_file',
        default='/data1/sgl/3RScan/meta_data/val.txt',
        help='The path of the file that stores the scan names.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.data_path)
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.data_path)


if __name__ == '__main__':
    main()
