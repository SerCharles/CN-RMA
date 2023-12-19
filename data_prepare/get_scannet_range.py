# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""
from math import *
import argparse
import datetime
import numpy as np
import os
from load_scannet_data import export
from os import path as osp
from plyfile import PlyData


def read_mesh_vertices(filename):
    """Read XYZ for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        ndarray: Vertices.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices


def process_one_scan(data_path, scene_id):
    mesh_file = osp.join(data_path, 'scans', scene_id, scene_id + '_vh_clean_2.ply')
    meta_file = osp.join(data_path, 'scans', scene_id, f'{scene_id}.txt')

    mesh_vertices = read_mesh_vertices(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # perform global alignment of mesh vertices
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = pts[:, 0:3]
    
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    min_z = np.min(pts[:, 2])
    max_z = np.max(pts[:, 2])
    return min_x, max_x, min_y, max_y, min_z, max_z


def get_range(data_path):
    scan_names = os.listdir(os.path.join(data_path, 'scans'))
    scan_names.sort()
    MIN_X = inf 
    MAX_X = -inf 
    MIN_Y = inf 
    MAX_Y = -inf
    MIN_Z = inf 
    MAX_Z = -inf
    
    
    for scan_name in scan_names:
        print('-' * 20)
        min_x, max_x, min_y, max_y, min_z, max_z = process_one_scan(data_path, scan_name)
        print('scene id:{}, range:({:.3f}, {:.3f}, {:.3f}) to ({:.3f}, {:.3f}, {:.3f})'.format(scan_name, min_x, min_y, min_z, max_x, max_y, max_z))
        
        MIN_X = min(MIN_X, min_x)
        MAX_X = max(MAX_X, max_x)
        MIN_Y = min(MIN_Y, min_y)
        MAX_Y = max(MAX_Y, max_y)
        MIN_Z = min(MIN_Z, min_z)
        MAX_Z = max(MAX_Z, max_z) 
        print('total range:({:.3f}, {:.3f}, {:.3f}) to ({:.3f}, {:.3f}, {:.3f})'.format(MIN_X, MIN_Y, MIN_Z, MAX_X, MAX_Y, MAX_Z))
 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scannet_dir', default='/data1/sgl/ScanNet', help='scannet data directory.')
    args = parser.parse_args()
    get_range(args.scannet_dir)


if __name__ == '__main__':
    main()
