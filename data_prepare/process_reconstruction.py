#Process the reconstructed pointclouds for FCAF3D

import csv
import numpy as np
import os
import glob
from plyfile import PlyData
import argparse

def read_mesh_vertices(filename):
    """Read XYZ and NX NY NZ for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        Vertices.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['nx']
        vertices[:, 4] = plydata['vertex'].data['ny']
        vertices[:, 5] = plydata['vertex'].data['nz']
    return vertices

def process_reconstruction(data_path, save_path):
    all_files = os.listdir(data_path)
    scene_ids = []
    for file_name in all_files:
        scene_id = file_name.split(os.sep)[-1]
        scene_ids.append(scene_id)
    scene_ids.sort() 
    for scene_id in scene_ids:
        input_path = os.path.join(data_path, scene_id, scene_id + '.ply')
        vertices = read_mesh_vertices(input_path)
        output_path = os.path.join(save_path, scene_id + '_vert.npy')
        np.save(output_path, vertices)
        print('Saved', scene_id)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/home/sgl/MVSDetection/work_dirs/neucon/results')
    parser.add_argument("--save_path", type=str, default='/home/sgl/ScanNet/my_instance_data')
    args = parser.parse_args()
    process_reconstruction(args.data_path, args.save_path)