import os
import torch
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
import cv2


def tsdf2mesh(voxel_size, origin, tsdf_vol):
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    return mesh



def save_results(results, voxel_size, save_path):
    if "scene_name" not in results.keys() or 'scene_tsdf' not in results.keys():
        return
    
    batch_size = len(results['scene_name'])
    for i in range(batch_size):
        scene_id = results['scene_name'][i].replace('/', '-')
        tsdf_volume = results['scene_tsdf'][i].data.cpu().numpy()
        origin = results['origin'][i].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(scene_id))
        else:
            # Marching cubes
            mesh = tsdf2mesh(voxel_size, origin, tsdf_volume)
            # save tsdf volume for atlas evaluation
            data = {'origin': origin,
                    'voxel_size': voxel_size,
                    'tsdf': tsdf_volume}
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(os.path.join(save_path, scene_id)):
                os.makedirs(os.path.join(save_path, scene_id))
            np.savez_compressed(os.path.join(save_path, scene_id, '{}.npz'.format(scene_id)), **data)
            mesh.export(os.path.join(save_path, scene_id, '{}.ply'.format(scene_id)))
