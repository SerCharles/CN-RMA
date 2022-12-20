from math import *
import os
import numpy as np
import PIL.Image as Image 
from plyfile import *

def read_ply(model_path):
    """Load the ply file with points with normal and faces
        V: number of vertexs
        F: number of faces

    Args:
        model_path [string]: [the full path of the model]

    Returns:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
        norms [numpy float array], [V * 3]: [norms of each vertex]
    """
    plydata = PlyData.read(model_path)
    my_vertexs = []
    my_norms = []
    my_faces = []

    vertexs = plydata['vertex']
    faces = plydata['face']
    for i in range(vertexs.count):
        x = float(vertexs[i][0])
        y = float(vertexs[i][1])
        z = float(vertexs[i][2])
        nx = float(vertexs[i][3])
        ny = float(vertexs[i][4])
        nz = float(vertexs[i][5])
        my_vertexs.append([x, y, z])
        my_norms.append([nx, ny, nz])

    for i in range(faces.count):
        a = int(faces[i][0][0])
        b = int(faces[i][0][1])
        c = int(faces[i][0][2])
        my_faces.append([a, b, c])
    
    vertexs = np.array(my_vertexs, dtype='float32')
    faces = np.array(my_faces, dtype='int32')
    norms = np.nan_to_num(np.array(my_norms, dtype='float32'))
    return vertexs, faces, norms

def read_depth(data_path, max_depth=inf):
    """Read a depth image file

    Args:
        data_path [str]: [the full depth path]

    Returns:
        depth_image [numpy float array], [H * W]: [the depth image]       
    """
    depth_image = Image.open(data_path)
    depth_image = np.array(depth_image, dtype=np.float32)        
    depth_image /= 1000.0
    depth_image[depth_image > max_depth] = 0
    return depth_image
    
def read_intrinsic(data_path, type_):
    """Read the the intrinsic of a picture

    Args:
        data_path [str]: [the full path of the intrinsic file]
        type_ [str]: ["depth" or "color", read color intrinsic or depth intrinsic?]
    Returns:
        intrinsic [numpy float array], [3 * 3]: [the intrinsic matrix]
    """
    f = open(data_path, 'r')
    words = f.read().split('\n')
    f.close()
    for word in words:
        word = word.split()
        if len(word) <= 0:
            continue
        if word[0] == 'fx_' + type_:
            fx = float(word[2])
        elif word[0] == 'fy_' + type_:
            fy = float(word[2])        
        elif word[0] == 'mx_' + type_:
            cx = float(word[2]) 
        elif word[0] == 'my_' + type_:
            cy = float(word[2])   
    intrinsic = np.zeros((3, 3), dtype=np.float32)
    intrinsic[0][0] = fx 
    intrinsic[1][1] = fy 
    intrinsic[0][2] = cx 
    intrinsic[1][2] = cy
    return intrinsic
    
def read_scene_basic(data_path, type_):
    """Read the the basic information of a scene

    Args:
        data_path [str]: [the full path of the intrinsic file]
        type_ [str]: ["depth" or "color", read color intrinsic or depth intrinsic?]
    Returns:
        n_frames [int]: [the total number of frames]
        width [int]: [the width of the pictures]
        height [int]: [the height of the pictures]
        fx [float]: [fx]
        fy [float]: [fy]
        cx [float]: [cx]
        cy [float]: [cy]
    """
    f = open(data_path, 'r')
    words = f.read().split('\n')
    f.close()
    for word in words:
        word = word.split()
        if len(word) <= 0:
            continue
        if word[0].lower() == 'num' + type_ + 'frames':
            n_frames = int(word[2])
        elif word[0] == type_ + 'Width':
            width = int(word[2])
        elif word[0] == type_ + 'Height':
            height = int(word[2])
        elif word[0] == 'fx_' + type_:
            fx = float(word[2])
        elif word[0] == 'fy_' + type_:
            fy = float(word[2])        
        elif word[0] == 'mx_' + type_:
            cx = float(word[2]) 
        elif word[0] == 'my_' + type_:
            cy = float(word[2])   

    return n_frames, width, height, fx, fy, cx, cy

def read_extrinsic(data_path):
    """Read an extrinsic file

    Args:
        data_path [str]: [the full extrinsic path]

    Returns:
        extrinsic [numpy float array], [4 * 4]: [the extrinsic matrix]
    """
    extrinsic = np.loadtxt(data_path, delimiter=' ', dtype=np.float32)
    return extrinsic

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
            scene_ids.append(scene_id)
    scene_ids.sort()
    return scene_ids