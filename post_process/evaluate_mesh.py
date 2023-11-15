# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import json
import os
import numpy as np
import torch
import trimesh
import open3d as o3d



def parse_args():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")
    parser.add_argument('--dataset', type=str, default='3rscan')
    parser.add_argument("--data_path", type=str, default='/data1/sgl/3RScan')
    parser.add_argument("--result_path", type=str, default='/home/sgl/work_dirs_atlas/3rscan_atlas_recon/results')
    parser.add_argument("--axis_align", type=int, default=1)

    return parser.parse_args()


args = parse_args()

def eval_mesh(pcd_pred, pcd_trgt, threshold=.05, down_sample=.02):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        pcd_pred: open3d pointcloud of prediction mesh
        pcd_trgt: open3d pointcloud of gt mesh
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),
               'dist2': np.mean(dist1),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    return metrics


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def read_axis_align_matrix(data_path):
    axis_align_matrix = np.eye(4)
    if os.path.exists(data_path):
        lines = open(data_path).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [
                    float(x)
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')
                ]
                break    
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix


def display_results(fname):
    key_names = ['dist1', 'dist2', 'prec', 'recal', 'fscore']

    metrics = json.load(open(fname, 'r'))
    metrics = sorted([(scene, metric) for scene, metric in metrics.items()], key=lambda x: x[0])
    scenes = [m[0] for m in metrics]
    metrics = [m[1] for m in metrics]

    keys = metrics[0].keys()
    metrics1 = {m: [] for m in keys}
    for m in metrics:
        for k in keys:
            metrics1[k].append(m[k])

    for k in key_names:
        if k in metrics1:
            v = np.nanmean(np.array(metrics1[k]))
        else:
            v = np.nan
        print('%10s %0.3f' % (k, v))

def process(scene_id):
    if args.dataset == 'scannet':
        if args.axis_align:
            axis_align_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '.txt')
            axis_align_matrix = read_axis_align_matrix(axis_align_path)
        else:
            axis_align_matrix = None
        pred_mesh_path = os.path.join(args.result_path, scene_id, scene_id + '.ply')
        gt_mesh_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '_vh_clean_2.ply')
    elif args.dataset == '3rscan':
        pred_mesh_path = os.path.join(args.result_path, scene_id, scene_id + '.ply')
        gt_mesh_path = os.path.join(args.data_path, 'scans', scene_id, 'labels.instances.annotated.v2.ply')
        axis_align_matrix = None
    
    pcd_pred = o3d.io.read_point_cloud(pred_mesh_path)
    pcd_gt = o3d.io.read_point_cloud(gt_mesh_path)
    if axis_align_matrix is not None:
        pcd_gt.transform(axis_align_matrix)
    metrics = eval_mesh(pcd_pred, pcd_gt)
    print(scene_id, metrics)
    return scene_id, metrics

def process_with_single_worker(info_files):
    metrics = {}
    for i, info_file in enumerate(info_files):
        scene, temp = process(info_file)
        if temp is not None:
            metrics[scene] = temp
    return metrics


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def main():
    if args.dataset == 'scannet':
        scene_names_file = os.path.join(args.data_path, 'meta_data', 'scannetv2_train.txt')
    elif args.dataset == '3rscan':
        scene_names_file = os.path.join(args.data_path, 'meta_data', '3rscan_train.txt')
    scene_names = [line.rstrip() for line in open(scene_names_file)]
    scene_names.sort()

    metrics = process_with_single_worker(scene_names)
    
    rslt_file = os.path.join(args.result_path, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    # display results
    display_results(rslt_file)




if __name__ == "__main__":
    main()
