import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import glob
import trimesh
import argparse
from math import *



def get_voxel_edges(voxel_id, origin, voxel_size):
    """
    Get the edges of the voxel
    
    Args:
        voxel_id [numpy long array], [3]: [the voxel id]
        origin [numpy float array], [3]: [the origin of the voxels]
        voxel_size [float]: [the voxel size]

    Returns:
        edges [numpy float array], [12 * 2 * 3]: [the edges of the bounding boxes]
    """
    
    #init
    corners = []
    for i in range(2):
        corners.append([])
        for j in range(2):
            corners[i].append([])
    edges = []
    translation = origin + voxel_id * voxel_size

        
    #get corners
    for i in range(2):
        for j in range(2):
            for k in range(2):
                dx = i
                dy = j
                dz = k
                d = np.array([dx, dy, dz])
                point = d * voxel_size
                point_global = point + translation 
                corners[i][j].append(point_global)
        
    #get edges 
    for k in range(2):
        edge1 = np.stack([corners[0][0][k], corners[0][1][k]], axis=0) #2 * 3
        edge2 = np.stack([corners[0][0][k], corners[1][0][k]], axis=0) #2 * 3
        edge3 = np.stack([corners[0][1][k], corners[1][1][k]], axis=0) #2 * 3
        edge4 = np.stack([corners[1][0][k], corners[1][1][k]], axis=0) #2 * 3
        edges.append(edge1)
        edges.append(edge2)
        edges.append(edge3)
        edges.append(edge4)
    for i in range(2):
        for j in range(2):
            edge = np.stack([corners[i][j][0], corners[i][j][1]], axis=0) #2 * 3
            edges.append(edge)
    edges = np.stack(edges, axis = 0) #12 * 2 * 3
    return edges

def read_axis_align_matrix(data_path):
    axis_align_matrix = np.eye(4)
    if data_path != None and os.path.exists(data_path):
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

def init_scene(mesh_path, meta_path):
    """Read and init the scene mesh

    Args:
        mesh_path [str]: [the path of the input mesh]
    
    Returns:
        mesh [trimesh.Trimesh]: [the transformed input mesh]
    """
    #read
    axis_align_matrix = read_axis_align_matrix(meta_path)
    mesh = trimesh.load(mesh_path)
    vertexs = np.array(mesh.vertices) #V * 3
    colors = np.array(mesh.visual.vertex_colors)[:, 0:3] #V * 3
    faces = np.array(mesh.faces) #F * 3
    
    #transform
    pts = np.ones((vertexs.shape[0], 4)) #N * 4
    pts[:, 0:3] = vertexs
    pts = np.dot(pts, axis_align_matrix.transpose()) #N * 4
    vertexs = pts[:, 0:3]
    
    #return
    mesh = trimesh.Trimesh(vertices=vertexs.tolist(), vertex_colors=colors.tolist(), faces=faces.tolist())
    return mesh 


def visualize_ray(mesh_path, ray_path, save_path):
    """
    Visualize the boxes and quads in the final pointcloud
    
    Args:
        mesh_path [str]: [the path of the original mesh]
        ray_path [str]: [the path of the ray_info]
        save_path [str]: [the saving path]
    """
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    ], 255).astype(np.uint8).tolist()
    ray_info = np.load(ray_path)
    os = ray_info['o'] #B * 3 * H * W * T
    ds = ray_info['d'] #B * 3 * H * W * T
    voxel_ids = ray_info['voxel_id'] #B * 3 * H * W * T
    valids = ray_info['valid'] #B * H * W * T
    tsdf_results = ray_info['tsdf_results'] #B * H * W * T
    origin = ray_info['origin'][0]
    B, H, W, T = tsdf_results.shape


    b = 0 
    h = H - 1
    w = W - 1

    all_edges = [] 
    all_colors = []    
    o = os[b, :, h, w, 0]
    d = ds[b, :, h, w, 0]
    t_max = sqrt(192 ** 2 + 192 ** 2 + 80 ** 2) * 0.04
    final = o + d * t_max  
    edge = np.stack([o, final], axis=0)[None, :, :]
    all_edges.append(edge)
    all_colors.extend([colors[10]])
    
    for i in range(T):
        voxel_id = voxel_ids[b, :, h, w, i]
        valid = valids[b, h, w, i]
        if valid:
            edges = get_voxel_edges(voxel_id, origin, 0.04)
            all_edges.append(edges)
            all_colors.extend([colors[0]] * 12)

    if len(all_edges) > 0:
        all_edges = np.concatenate(all_edges, axis=0)

    original_trimesh = init_scene(mesh_path, None)
    scene = trimesh.scene.Scene()
    scene.add_geometry(original_trimesh)
    
    rad = 0.005
    res = 16
    for i in range(len(all_edges)):
        source = all_edges[i][0]
        target = all_edges[i][1]
        edge_color = all_colors[i]
        
        # compute line
        vector = target - source 
        M = trimesh.geometry.align_vectors([0,0,1], vector, False)
        vector = target - source # compute again since align_vectors modifies vec in-place!
        M[:3,3] = 0.5 * source + 0.5 * target
        height = np.sqrt(np.dot(vector, vector))
        edge_mesh = trimesh.creation.cylinder(radius=rad, height=height, sections=res, transform=M)
        edge_vertexs = np.array(edge_mesh.vertices).tolist()
        edge_colors = [edge_color] * len(edge_vertexs)
        edge_faces = np.array(edge_mesh.faces).tolist()
        edge_mesh = trimesh.Trimesh(vertices=edge_vertexs, vertex_colors=edge_colors, faces=edge_faces)
        scene.add_geometry(edge_mesh)
    
    scene = trimesh.util.concatenate(scene.dump())
    scene.export(save_path)



if __name__ == "__main__":
    mesh_path = '/home/sgl/work_dirs_atlas/atlas_ray_marching/results/scene0000_00_331/scene0000_00_331.ply'
    ray_path = '/home/sgl/work_dirs_atlas/atlas_ray_marching/results/scene0000_00_331/scene0000_00_331.npz'
    save_path = '/home/sgl/work_dirs_atlas/atlas_ray_marching/results/scene0000_00_331/scene0000_00_331_result.ply'
    visualize_ray(mesh_path, ray_path, save_path)