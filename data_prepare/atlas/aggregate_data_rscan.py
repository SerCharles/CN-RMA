import json 
import mmcv
import numpy as np
import os
import argparse
from concurrent import futures as futures


class RScanData(object):
    """3RScan data.

    Generate 3rscan infos for 3rscan_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = os.path.join(root_path)
        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = os.path.join(self.root_dir, 'meta_data', f'3rscan_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = os.path.join(self.root_dir, '3rscan_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = os.path.join(self.root_dir, '3rscan_instance_data',
                            f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = os.path.join(self.root_dir, '3rscan_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            tsdf_path = os.path.join(self.root_dir, 'atlas_tsdf', sample_idx)
            info_path = os.path.join(tsdf_path, 'info.json')
            with open(info_path) as f:
                info = json.load(f)

            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 6
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                    axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                    annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                    info['annos'] = annotations
                else:
                    info = None
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        results = []
        for sample_idx in sample_id_list:
            info = process_single_scene(sample_idx)
            if info != None:
                results.append(info)
        return results

def create_indoor_info_file(data_path, save_path, workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str): Prefix of the pkl to be saved. Default: 'sunrgbd'.
        save_path (str): Path of the pkl to be saved. Default: None.
        use_v1 (bool): Whether to use v1. Default: False.
        workers (int): Number of threads to be used. Default: 4.
    """

    train_filename = os.path.join(save_path, '3rscan_infos_train.pkl')
    val_filename = os.path.join(save_path, '3rscan_infos_val.pkl')
    train_dataset = RScanData(root_path=data_path, split='train')
    val_dataset = RScanData(root_path=data_path, split='val')
    
    infos_train = train_dataset.get_infos(num_workers=workers, has_label=False)
    mmcv.dump(infos_train, train_filename, 'pkl')
    print(f'3rscan info train file is saved to {train_filename}')
    infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
    mmcv.dump(infos_val, val_filename, 'pkl')
    print(f'3rscan info val file is saved to {val_filename}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('--data_path', type=str, default='./data/3rscan', help='specify the root path of dataset')
    parser.add_argument('--save_path', type=str, default='./data/3rscan', help='name of info pkl')
    parser.add_argument('--workers', type=int, default=4, help='number of threads to be used')
    args = parser.parse_args()
    create_indoor_info_file(data_path=args.data_path, save_path=args.save_path, workers=args.workers)

