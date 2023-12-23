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

#  Originating Author: Zak Murez (zak.murez.com)

import argparse
import json
import os

import open3d as o3d
import numpy as np
import torch
import trimesh

from arkit_simple_loader import ARKitSimpleLoader
from transforms import *
from tsdf import TSDFFusion, TSDF, coordinates, depth_to_world
from tqdm import tqdm
import ray
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')

    parser.add_argument("--dataset", type=str, default='arkit')    
    parser.add_argument("--data_path", type=str, default='/data1/sgl/ARKit')    
    parser.add_argument("--save_path", type=str, default='/data1/sgl/ARKit/atlas_tsdf')
    
    parser.add_argument('--n_proc', default=2, type=int)
    parser.add_argument('--n_gpu', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--max_depth', default=3., type=float,
        help='mask out large depth values since they are noisy')
    parser.add_argument('--save_mesh', default=1, type=int)
    args = parser.parse_args()
    return args 

args = parse_args()


def fuse_scene(args, scene, frame_list, voxel_size, trunc_ratio=3,
               vol_prcnt=.995, vol_margin=1.5, verbose=2):
    """ Use TSDF fusion with GT depth maps to generate GT TSDFs

    Args:
        path_meta: path to save the TSDFs 
            (we recommend creating a parallel directory structure to save 
            derived data so that we don't modify the original dataset)
        scene: name of scene to process
        trunc_ratio: truncation distance in voxel units
        max_depth: mask out large depth values since they are noisy
        vol_prcnt: for computing the bounding volume of the TSDF... ignore outliers
        vol_margin: padding for computing bounding volume of the TSDF
        fuse_semseg: whether to accumulate semseg images for GT semseg
            (prefered method is to not accumulate and insted transfer labels
            from ground truth labeled mesh)
        device: cpu/ which gpu
        verbose: how much logging to print

    Returns:
        writes a TSDF (.npz) file into path_meta/scene

    Notes: we use a conservative value of max_depth=3 to reduce noise in the 
    ground truth. However, this means some distant data is missing which can
    create artifacts. Nevertheless, we found we acheived the best 2d metrics 
    with the less noisy ground truth.
    """

    if verbose>0:
        print('fusing', scene, 'voxel size', voxel_size)

    # get gpu device for this worker
    device = torch.device('cuda') # gpu for this process


    # find volume bounds and origin by backprojecting depth maps to point clouds
    # use a subset of the frames to save time
    if len(frame_list)<=200:
        inds = np.array(range(len(frame_list))).astype(np.int)
    else:
        inds = np.linspace(0,len(frame_list)-1,200).astype(np.int)

    pts = []
    for id in inds:
        frame = frame_list[id]
        projection = frame['projection'].to(device)
        depth = frame['depth'].to(device)
        depth[depth>args.max_depth]=0
        pts.append( depth_to_world(projection, depth).view(3,-1).T )
    pts = torch.cat(pts)
    pts = pts[torch.isfinite(pts[:,0])].cpu().numpy()
    # use top and bottom vol_prcnt of points plus vol_margin
    origin = torch.as_tensor(np.quantile(pts, 1-vol_prcnt, axis=0)-vol_margin).float()
    vol_max = torch.as_tensor(np.quantile(pts, vol_prcnt, axis=0)+vol_margin).float()
    vol_dim = ((vol_max-origin)/(float(voxel_size)/100)).int().tolist()


    # initialize tsdf
    tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin,
                             trunc_ratio, device, label=False)

    # integrate frames
    for i in range(len(frame_list)):
        frame = frame_list[i]
        if verbose>1 and i%100==0:
            print("{}: Fusing frame {}/{}".format(scene, str(i), str(len(frame_list))))

        projection = frame['projection'].to(device)
        image = frame['image'].to(device)
        depth = frame['depth'].to(device)

        # only use reliable depth
        depth[depth>args.max_depth]=0

        tsdf_fusion.integrate(projection, depth, image)

    # save mesh and tsdf
    file_name_vol = os.path.join(args.save_path, scene, 'tsdf_%02d.npz'%voxel_size)
    tsdf = tsdf_fusion.get_tsdf()
    tsdf.save(file_name_vol)
    
    if args.save_mesh:
        file_name_mesh = os.path.join(args.save_path, scene, 'mesh_%02d.ply'%voxel_size)
        mesh = tsdf.get_mesh()
        mesh.export(file_name_mesh)



#@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def prepare_single(args, scenes, split='Training'):
    assert split in ['Training', 'Validation']
    
    for scene in tqdm(scenes):
        save_path = os.path.join(args.save_path, scene)
        if os.path.exists(os.path.join(save_path, 'tsdf_16.npz')):
            continue
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        transform = Compose([ToTensor(),
                            IntrinsicsPoseToProjection(),
                                  ])
        dataset = ARKitSimpleLoader(args.data_path, scene, transform, split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                             batch_sampler=None, num_workers=args.num_workers)
        
        frame_list = []
        id_list = []
        for i, frame in enumerate(dataloader):
            if i % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(i), str(len(dataset))))
            if frame['valid'] == True:
                frame_list.append(frame)
                id_list.append(frame['id'])
        frame_list = sorted(frame_list, key=lambda x: x['id'])
        id_list.sort()
        
        info = {
            'scene': scene,
            'total_images': len(id_list),
            'total_image_ids': id_list
        }
        info_path = os.path.join(save_path, 'info.json')
        json.dump(info, open(info_path, 'w'))
        
        for voxel_size in [4,8,16]:
        #for voxel_size in [2, 4, 8]:
            fuse_scene(args, scene, frame_list, voxel_size)



def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret

def main(args):
    """ Create all derived data need for the Scannet dataset

    For each scene an info.json file is created containg all meta data required
    by the dataloaders. We also create the ground truth TSDFs by fusing the
    ground truth TSDFs and add semantic labels

    Args:
        path: path to the scannet dataset
        path_meta: path to save all the derived data
            (we recommend creating a parallel directory structure so that 
            we don't modify the original dataset)
        i: process id (used for parallel processing)
            (this process operates on scenes [i::n])
        n: number of processes
        test_only: only prepare the test set (for rapid testing if you dont 
            plan to train)
        max_depth: mask out large depth values since they are noisy

    Returns:
        Writes files to path_meta
    """
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    '''
    all_proc = args.n_proc * args.n_gpu
    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)
    '''
    scenes_training = [scene for scene in os.listdir(os.path.join(args.data_path, '3dod', 'Training'))]
    scenes_training.sort()
    '''
    files = split_list(scenes_training, all_proc)
    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(prepare_single.remote(args, files[w_idx], split='Training'))
    results = ray.get(ray_worker_ids)
    '''
    prepare_single(args, scenes_training, split='Training')

        
    scenes_validation = [scene for scene in os.listdir(os.path.join(args.data_path, '3dod', 'Validation'))]
    scenes_validation.sort()
    '''
    files = split_list(scenes_validation, all_proc)
    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(prepare_single.remote(args, files[w_idx], split='Validation'))
    results = ray.get(ray_worker_ids)
    '''
    prepare_single(args, scenes_validation, split='Validation')




if __name__ == "__main__":
    main(args)
