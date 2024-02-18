# Modified from
# https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/visual_utils.py
# Copyright (c) Apple, Inc. and its affiliates.
"""Visualize Object Detection Results
Usage example: python ./visualize_results.py
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import argparse

def load_scene_ids_scannet(data_path, type):
    """Load the scene ids of train, val and test files

    Args:
        data_path [str]: [The path of our modified ScanNet dataset]
        type [str]: [train, val or test]
    
    Returns:
        scene_ids [list of string]: [the train/val/test scenes]
    """
    if type == 'test':
        directory_name = 'scans_test'
    else:
        directory_name = 'scans'
    path = os.path.join(data_path, 'meta_data', 'scannetv2_' + type + '.txt')
    f = open(path, 'r')
    scenes = f.read().split('\n')
    f.close()
    scene_ids = []
    for scene_id in scenes:
        if 'scene' in scene_id:
            scene_ids.append(scene_id)
    scene_ids.sort()
    return scene_ids

def rotate_points_along_z(points, angle):
    """Rotation clockwise
    Args:
        points: np.array of np.array (B, N, 3 + C) or
            (N, 3 + C) for single batch
        angle: np.array of np.array (B, )
            or (, ) for single batch
            angle along z-axis, angle increases x ==> y
    Returns:
        points_rot:  (B, N, 3 + C) or (N, 3 + C)

    """
    single_batch = len(points.shape) == 2
    if single_batch:
        points = np.expand_dims(points, axis=0)
        angle = np.expand_dims(angle, axis=0)
    cosa = np.expand_dims(np.cos(angle), axis=1)
    sina = np.expand_dims(np.sin(angle), axis=1)
    zeros = np.zeros_like(cosa) # angle.new_zeros(points.shape[0])
    ones = np.ones_like(sina) # angle.new_ones(points.shape[0])

    rot_matrix = (
        np.concatenate((cosa, -sina, zeros, sina, cosa, zeros, zeros, zeros, ones), axis=1)
        .reshape(-1, 3, 3)
    )

    # print(rot_matrix.view(3, 3))
    points_rot = np.matmul(points[:, :, :3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    if single_batch:
        points_rot = points_rot.squeeze(0)

    return points_rot

def boxes_to_edges(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center

    Returns:
        edges: (12, 2, 3)
    """
    template = np.array([[1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1]]
    ) / 2.

    corners3d = np.tile(boxes3d[None, 3:6], (8, 1)) * template
    corners3d = rotate_points_along_z(corners3d.reshape(8, 3), boxes3d[6]).reshape(
        8, 3
    )
    corners3d += boxes3d[None, 0:3]
    
    edges = []
    edges.append([corners3d[0], corners3d[1]])
    edges.append([corners3d[1], corners3d[2]])
    edges.append([corners3d[2], corners3d[3]])
    edges.append([corners3d[3], corners3d[0]])
    edges.append([corners3d[4], corners3d[5]])
    edges.append([corners3d[5], corners3d[6]])
    edges.append([corners3d[6], corners3d[7]])
    edges.append([corners3d[7], corners3d[4]])
    edges.append([corners3d[0], corners3d[4]])
    edges.append([corners3d[1], corners3d[5]])
    edges.append([corners3d[2], corners3d[6]])
    edges.append([corners3d[3], corners3d[7]])
    edges = np.array(edges)
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

def init_pc(mesh_path, meta_path):
    """Read and init the pc mesh

    Args:
        mesh_path [str]: [the path of the input mesh]
    
    Returns:
        mesh [trimesh.Trimesh]: [the transformed input mesh]
    """
    #read
    axis_align_matrix = read_axis_align_matrix(meta_path)
    mesh = trimesh.load(mesh_path)
    vertexs = np.array(mesh.vertices) #V * 3

    
    #transform
    pts = np.ones((vertexs.shape[0], 4)) #N * 4
    pts[:, 0:3] = vertexs
    pts = np.dot(pts, axis_align_matrix.transpose()) #N * 4
    vertexs = pts[:, 0:3]
    
    #return
    mesh = trimesh.Trimesh(vertices=vertexs.tolist())
    return mesh 

def visualize_boxs(mesh_path, meta_path, box_path, save_path, type='mesh'):
    """
    Visualize the boxes and quads in the final pointcloud
    
    Args:
        mesh_path [str]: [the path of the original mesh]
        meta_path [str]: [the path of the meta data]
        box_path [str]: [the path of the bounding boxes]
        save_path [str]: [the saving path]
    """
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    ], 255).astype(np.uint8).tolist()
    bbox_data = np.load(box_path)
    bboxes = bbox_data['boxes']
    scores = bbox_data['scores']
    labels = bbox_data['labels']
    all_edges = [] 
    all_colors = []
    
    for i in range(len(bboxes)):
        if scores[i] < 0.15:
            continue 
        bbox = bboxes[i]
        label = labels[i]
        edges = boxes_to_edges(bbox)
        all_edges.append(edges)
        all_colors.extend([colors[label]] * 12)
    if len(all_edges) > 0:
        all_edges = np.concatenate(all_edges, axis=0)

    if type=='mesh':
        original_trimesh = init_scene(mesh_path, meta_path)
    else:
        original_trimesh = init_pc(mesh_path, meta_path)
    scene = trimesh.scene.Scene()
    scene.add_geometry(original_trimesh)
    
    rad = 0.03
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

def generate_gt(box_path, save_path, dataset='scannet'):
    '''
    Generate gt results 
    
    Args:
        box_path [str]: [the path of the bounding boxes]
        save_path [str]: [the saving path]
    '''
    bboxes_data = np.load(box_path)
    if dataset == 'scannet' or dataset == '3rscan':
        bboxes = bboxes_data[:, 0:6]
        zeros = np.zeros((bboxes_data.shape[0], 1))
        bboxes = np.concatenate((bboxes, zeros), axis=1)
    elif dataset == 'arkit':
        bboxes = bboxes_data[:, 0:7]
    classes = bboxes_data[:, -1].astype(np.int32).tolist()
    
    if dataset == 'scannet' or dataset == '3rscan':
        cat_ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(cat_ids))
        }
        labels = np.array([
            cat_ids2class[classes[i]]
            for i in range(len(classes))
        ])
    elif dataset == 'arkit':
        labels = classes 
    
    scores = np.ones((bboxes_data.shape[0]))
    np.savez(save_path, boxes=bboxes, scores=scores, labels=labels)
    

def main():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")

    parser.add_argument("--dataset", type=str, default='arkit')
    #parser.add_argument("--data_path", type=str, default='/data1/sgl/ScanNet')
    parser.add_argument("--data_path", type=str, default='/data1/sgl/ARKit')

    parser.add_argument("--post_fix", type=str, default='_atlas_bbox')
    parser.add_argument("--save_path", type=str, default='/data1/sgl/CN-RMA_results/arkit_results')

    args = parser.parse_args()
    
    if args.dataset == 'scannet':
        scene_ids = load_scene_ids_scannet(args.data_path, 'val')
    elif args.dataset == 'arkit':
        scene_ids = os.listdir(os.path.join(args.data_path, '3dod', 'Validation'))
    scene_ids.sort()

    for scene_id in scene_ids:
        if args.dataset == 'scannet':
            meta_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '.txt')
            mesh_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '_vh_clean_2.ply')
        elif args.dataset == 'arkit':
            mesh_path = os.path.join(args.data_path, '3dod', 'Validation', scene_id, scene_id + '_3dod_mesh.ply')
            meta_path = None
        
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + args.post_fix + '.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + args.post_fix + '.ply')
        if not os.path.exists(os.path.join(args.save_path, scene_id)):
            continue
        
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='mesh')
        print(scene_id, 'finished!')

if __name__ == "__main__":
    main()