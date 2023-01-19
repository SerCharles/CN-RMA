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
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.04, type=float)
    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)

    parser.add_argument('--n_proc', type=int, default=16, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=2, help='#number of gpus')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()

args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)


def save_tsdf_full(args, scene_id, intrinsic_list, extrinsic_list, image_list, depth_list, save_mesh=False):
    """Save the tsdf results

    Args:
        args [arguments]: [the global arguments]
        scene_id [str]: [the scene id]
        intrinsic_list [the list of torch float array], [3 * 3 each]: [the intrinsics]
        extrinsic_list [the list of torch float array], [4 * 4 each]: [the extrinsics]
        image_list [the list of torch float array], [H * W * 3 each]: [the images]
        depth_list [the list of torch float array], [H * W each]: [the depth images]
        save_mesh [bool, optional]: [whether save mesh of not]. Defaults to False.
    """
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))
    #get the image ids
    n_imgs = len(depth_list.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_ids = np.array(list(depth_list.keys()))[ind]
    else:
        image_ids = depth_list.keys()
    for id in image_ids:
        intrinsic = intrinsic_list[id]
        extrinsic = extrinsic_list[id]
        depth_image = depth_list[id]
        if len(image_list) == 0:
            color_image = None 
        else:
            color_image = image_list[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_image, intrinsic, extrinsic)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for id in depth_list.keys():
        if id % 100 == 0:
            print("{}: Fusing frame {}/{}".format(scene_id, str(id), str(n_imgs)))
        depth_image = depth_list[id]
        extrinsic = extrinsic_list[id]
        if len(image_list) == 0:
            color_image = None
        else:
            color_image = image_list[id]

        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_image, intrinsic, extrinsic, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path, scene_id)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_id, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
        np.savez_compressed(os.path.join(args.save_path, scene_id, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    if save_mesh:
        for l in range(args.num_layers):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()

            meshwrite(os.path.join(args.save_path, scene_id, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms,
                      colors)
            # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            # print("Saving point cloud to pc.ply...")
            # point_cloud = tsdf_vol_list[l].get_point_cloud()
            # pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)

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
    fragments = []
    print('segment: process scene {}'.format(scene_id))

    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        intrinsic = intrinsic_list[id]
        extrinsic = extrinsic_list[id]
        depth_image = depth_list[id]
        

        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = extrinsic
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_image, intrinsic, extrinsic)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(extrinsic[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(extrinsic[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = extrinsic
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_image, intrinsic, extrinsic)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0

    with open(os.path.join(args.save_path, scene_id, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    for i, bnds in enumerate(all_bnds):
        if not os.path.exists(os.path.join(args.save_path, scene_id, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, scene_id, 'fragments', str(i)))
        fragments.append({
            'scene': scene_id,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })

    with open(os.path.join(args.save_path, scene_id, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_tsdf_data(args, scannet_files):
    """Process the tsdf data

    Args:
        args [arguments]: [the global arguments]
        scannet_files [list]: [list of scene ids]
    """
    image_data_path = os.path.join(args.data_path, 'posed_images')
    for scene_id in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene_id, 'fragments.pkl')):
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
            
        save_tsdf_full(args, scene_id, intrinsic_list, extrinsic_list, image_list, depth_list, save_mesh=args.save_mesh)
        save_fragment_pkl(args, scene_id, intrinsic_list, extrinsic_list, image_list, depth_list)

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
            save_name = os.path.join(args.save_path, scene_id, 'fragments.pkl')
            if os.path.exists(save_name):
                with open(save_name, 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)
        with open(os.path.join(args.save_path, 'fragments_' + split + '.pkl'), 'wb') as f:
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

