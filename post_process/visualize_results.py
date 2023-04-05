import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import glob
import trimesh
import argparse

def load_scene_ids(data_path, type):
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
            #if os.path.exists(os.path.join(data_path, directory_name, scene_id, 'color', '0.jpg')):
            scene_ids.append(scene_id)
    scene_ids.sort()
    return scene_ids

def heading_to_rotation(heading_angle):
    """Switch the heading angle to the rotation

    Args:
        heading_angle [float]: [the heading angle of the bounding box]

    Returns:
        rotation [numpy float array], [3 * 3]: [the rotation matrix of the bounding box]
    """
    rotation = np.zeros((3,3))
    rotation[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotation[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotation

def get_box_edges(center, lengths, heading_angle):
    """
    Get the edges of the bounding box
    
    Args:
        center [numpy float array], [3]: [the center of the bounding box]
        lengths [numpy float array], [3]: [the lengths of the bounding box]
        heading_angle [float]: [the heading angle of the bounding box]

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
    translation = center
    rotation = heading_to_rotation(heading_angle)
        
    #get corners
    for i in range(2):
        for j in range(2):
            for k in range(2):
                dx = i - 0.5
                dy = j - 0.5
                dz = k - 0.5
                d = np.array([dx, dy, dz])
                point = d * lengths
                point_global = np.dot(rotation, point) + translation 
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
        center = bbox[0:3]
        lengths = bbox[3:6]
        heading_angle = bbox[6]
        edges = get_box_edges(center, lengths, heading_angle)
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

def generate_gt(box_path, save_path):
    '''
    Generate gt results 
    
    Args:
        box_path [str]: [the path of the bounding boxes]
        save_path [str]: [the saving path]
    '''
    bboxes_data = np.load(box_path)
    bboxes = bboxes_data[:, 0:6]
    classes = bboxes_data[:, -1].astype(np.int32).tolist()
    cat_ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    cat_ids2class = {
        nyu40id: i
        for i, nyu40id in enumerate(list(cat_ids))
    }
    labels = np.array([
        cat_ids2class[classes[i]]
        for i in range(len(classes))
    ])
    scores = np.ones((bboxes_data.shape[0]))
    zeros = np.zeros((bboxes_data.shape[0], 1))
    bboxes = np.concatenate((bboxes, zeros), axis=1)
    np.savez(save_path, boxes=bboxes, scores=scores, labels=labels)
    

def main():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")
    parser.add_argument("--data_path", type=str, default='/data/shenguanlin/ScanNet')
    parser.add_argument("--save_path", type=str, default='/data/shenguanlin/atlas_test/results')
    args = parser.parse_args()
    scene_ids = load_scene_ids(args.data_path, 'val')
    #scene_ids = ['scene0011_00', 'scene0011_01', 'scene0015_00', 'scene0019_00', 'scene0019_01']
    scene_ids = ['scene0005_00', 'scene0041_00', 'scene0106_00', 'scene0158_00', 'scene0344_00']
    #print(scene_ids)
    scene_ids.sort()
    for scene_id in scene_ids:
        if not os.path.exists(os.path.join(args.save_path, scene_id)):
            continue
        
        mesh_path = os.path.join(args.save_path, scene_id, scene_id + '_points.ply')
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_test.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + '_pc.ply')
        meta_path = None
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='point')
        mesh_path = os.path.join(args.save_path, scene_id, scene_id + '_features.ply')
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_test.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + '_fc.ply')
        meta_path = None
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='point')
        
        meta_path = None
        mesh_path = os.path.join(args.save_path, scene_id, scene_id + '_gt.ply')
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_gt.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + '_detection.ply')
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='mesh')
        
        '''
        mesh_path = os.path.join(args.save_path, scene_id, scene_id + '_features.ply')
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_gt.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + '_fc.ply')
        meta_path = None
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='point')
        mesh_path = os.path.join(args.save_path, scene_id, scene_id + '.ply')
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_gt.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + '_detection.ply')
        meta_path = None
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='mesh')
        '''
        '''
        meta_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '.txt')
        mesh_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '_vh_clean_2.ply')
        bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_atlas_mine.npz')
        save_path = os.path.join(args.save_path, scene_id, scene_id + '_atlas_mine.ply')
        visualize_boxs(mesh_path, meta_path, bbox_path, save_path, type='mesh')
        '''

        '''
        gt_path = os.path.join(args.data_path, 'scannet_instance_data', scene_id + '_aligned_bbox.npy')
        gt_bbox_path = os.path.join(args.save_path, scene_id, scene_id + '_gt.npz')
        gt_save_path = os.path.join(args.save_path, scene_id, scene_id + '_gt.ply')
        generate_gt(gt_path, gt_bbox_path)
        visualize_boxs(mesh_path,  meta_path, gt_bbox_path, gt_save_path, type='mesh')
        '''
        print('processed ' + scene_id)

if __name__ == "__main__":
    main()