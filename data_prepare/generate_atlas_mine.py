"""
Generate the TSDF data
"""
import os
import time
from tsdf_fusion import *
import pickle
import argparse
from tqdm import tqdm
import ray
import torch.multiprocessing
from data_utils import *
from scannet_simple_loader import *


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--data_path", type=str, default='/data4/sgl/ScanNet')
    parser.add_argument("--save_name", type=str, default='all_tsdf_9')
    parser.add_argument("--save_mesh", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)

    #arguments used in TSDF Fusion
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--voxel_size', default=0.04, type=float)
    parser.add_argument('--window_size', default=50, type=int)

    parser.add_argument('--n_proc', type=int, default=4, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()

args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)


def save_fragment_pkl(args, scene_id, intrinsic_list, extrinsic_list, image_list, depth_list):
    """Save the meta data of each scene

    Args:
        args [arguments]: [the global arguments]
        scene_id [str]: [the scene id]
        intrinsic_list [the list of torch float array], [3 * 3 each]: [the intrinsics]
        extrinsic_list [the list of torch float array], [4 * 4 each]: [the extrinsics]
        image_list [the list of torch float array], [3 * H * W each]: [the images]
        depth_list [the list of torch float array], [H * W each]: [the depth images]
    """
    print('segment: process scene {}'.format(scene_id))

    id_list = []
    for id in depth_list.keys():
        id_list.append(id)
    id_list.sort()
    
    m = len(id_list)
    n = args.window_size 
    k = (m - 1) // (n - 1)
    image_ids = []
    for i in range(n):
        image_ids.append(id_list[i * k])


    with open(os.path.join(args.save_path, scene_id, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    fragment = {
        'scene': scene_id,
        'total_images': m,
        'total_image_ids': id_list,
        'n_images': n,
        'image_ids': image_ids,
        'vol_origin': tsdf_info['vol_origin'],
        'voxel_size': tsdf_info['voxel_size'],
    }


    with open(os.path.join(args.save_path, scene_id, 'single_fragment_info.pkl'), 'wb') as f:
        pickle.dump(fragment, f)
        



@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_tsdf_data(args, scannet_files):
    """Process the tsdf data

    Args:
        args [arguments]: [the global arguments]
        scannet_files [list]: [list of scene ids]
    """
    image_data_path = os.path.join(args.data_path, 'posed_images')
    for scene_id in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene_id, 'single_fragment_info.pkl')):
            continue
        print('read from disk')

        intrinsic_list = {}
        extrinsic_list = {}
        image_list = {}
        depth_list = {}
        dataset = ScanNetDataset(image_data_path, scene_id, args.max_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=args.num_workers)
        n_images = dataset.__len__()
        axis_align_path = os.path.join(args.data_path, 'scans', scene_id, scene_id + '.txt')
        axis_align_matrix = read_axis_align_matrix(axis_align_path)
        
        for id, (intrinsic, extrinsic, color_image, depth_image, valid) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene_id, str(id), str(n_images)))
            if not valid:
                continue
            extrinsic = axis_align_matrix @ extrinsic
            intrinsic_list.update({id: intrinsic})
            extrinsic_list.update({id: extrinsic})
            depth_list.update({id: depth_image})
            #image_list.update({id: color_image})
            
        save_fragment_pkl(args, scene_id, intrinsic_list, extrinsic_list, image_list, depth_list)

@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def change_tsdf_data(args, scannet_files):
    for scene_id in tqdm(scannet_files):
        with open(os.path.join(args.save_path, scene_id, 'tsdf_info.pkl'), 'rb') as f:
            tsdf_info = pickle.load(f)
        with open(os.path.join(args.save_path, scene_id, 'single_fragment_info.pkl'), 'rb') as f:
            fragment_info = pickle.load(f)
        m = fragment_info['total_images']
        id_list = fragment_info['total_image_ids']
        n = args.window_size 
        k = (m - 1) // (n - 1)
        image_ids = []
        for i in range(n):
            image_ids.append(id_list[i * k])
        fragment = {
            'scene': scene_id,
            'total_images': m,
            'total_image_ids': id_list,
            'n_images': n,
            'image_ids': image_ids,
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        }

        with open(os.path.join(args.save_path, scene_id, 'single_fragment_info.pkl'), 'wb') as f:
            pickle.dump(fragment, f)

def aggregate_pkl(args):
    """Aggregate the meta information of pkl files of all scenes into one pkl

    Args:
        args [arguments]: [the global arguments]
    """
    
    splits = ['train', 'val', 'test']
    for split in splits:
        fragments = []
        scene_ids = load_scene_ids(args.data_path, split)
        for scene_id in scene_ids:
            save_name = os.path.join(args.save_path, scene_id, 'single_fragment_info.pkl')
            if os.path.exists(save_name):
                with open(save_name, 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)
        with open(os.path.join(args.save_path, 'single_fragments_' + split + '.pkl'), 'wb') as f:
            pickle.dump(fragments, f)

def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret

def main(args):
    """The main process of tsdf generation
    """
    
    args.save_path = os.path.join(args.data_path, args.save_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    scenes_train = load_scene_ids(args.data_path, 'train')
    scenes_val = load_scene_ids(args.data_path, 'val')
    scenes_test = load_scene_ids(args.data_path, 'test')
    scene_ids = []
    scene_ids.extend(scenes_train)
    scene_ids.extend(scenes_val)
    scene_ids.extend(scenes_test)
    scene_ids.sort() 
    
    
    all_proc = args.n_proc * args.n_gpu
    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)
    files = split_list(scene_ids, all_proc)
    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_tsdf_data.remote(args, files[w_idx]))

    results = ray.get(ray_worker_ids)
    
    aggregate_pkl(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

