'''Checkpoint modification, including modifying the Atlas checkpoint for mmlab usage, 
and combining the Atlas and FCAF3D checkpoints for finetuning
'''

import argparse
import torch


def switch_atlas_ckpt(atlas_path, full_model_path, result_path):
    model_atlas = torch.load(atlas_path)['state_dict']
    model_mine = torch.load(full_model_path)['state_dict']
    
    new_state_dict = model_mine.copy()
    for key in model_mine.keys():
        if 'fpn.bottom_up' in key:
            back_key = key[14:]
            front_key = 'backbone2d.0.bottom_up.'
            atlas_key = front_key + back_key
        elif 'fpn' in key:
            back_key = key[4:]
            front_key = 'backbone2d.0.'
            atlas_key = front_key + back_key
        elif 'feature_2d' in key:
            back_key = key[11:]
            front_key = 'backbone2d.1.'
            atlas_key = front_key + back_key
        elif 'backbone3d' in key:
            atlas_key = key 
        elif 'tsdf_head' in key:
            back_key = key[10:]
            front_key = 'heads3d.heads.0.'
            atlas_key = front_key + back_key
        new_state_dict[key] = model_atlas[atlas_key]
    
    model_mine_full = torch.load(full_model_path)
    model_mine_full['state_dict'] = new_state_dict   
    torch.save(model_mine_full, result_path)
    
def combine_atlas_fcaf3d(atlas_path, fcaf3d_path, full_model_path, result_path):
    if atlas_path != None:
        model_atlas = torch.load(atlas_path)['state_dict']
    if fcaf3d_path != None: 
        model_fcaf3d = torch.load(fcaf3d_path)['state_dict']
    model_mine = torch.load(full_model_path)['state_dict']
    
    if atlas_path == None:
        atlas_mode = 'none'
    elif atlas_path.endswith('.ckpt'):
        atlas_mode = 'ckpt'
    else:
        atlas_mode = 'pth'
        
    if fcaf3d_path == None:
        fcaf3d_mode = 'none'
    else:
        fcaf3d_mode = 'fcaf3d'
    
    new_state_dict = model_mine.copy()
    for key in model_mine.keys():
        atlas_key = None 
        fcaf3d_key = None
        
        if 'fpn.bottom_up' in key:
            if atlas_mode == 'none':
                atlas_key = None
            elif atlas_mode == 'ckpt':
                back_key = key[14:]
                front_key = 'backbone2d.0.bottom_up.'
                atlas_key = front_key + back_key
            else:
                atlas_key = key
        elif 'fpn' in key:
            if atlas_mode == 'none':
                atlas_key = None
            elif atlas_mode == 'ckpt':
                back_key = key[4:]
                front_key = 'backbone2d.0.'
                atlas_key = front_key + back_key
            else:
                atlas_key = key
        elif 'feature_2d' in key:
            if atlas_mode == 'none':
                atlas_key = None
            elif atlas_mode == 'ckpt':
                back_key = key[11:]
                front_key = 'backbone2d.1.'
                atlas_key = front_key + back_key
            else:
                atlas_key = key
        elif 'backbone3d' in key:
            if atlas_mode == 'none':
                atlas_key = None 
            else:
                atlas_key = key 
        elif 'tsdf_head' in key:
            if atlas_mode == 'none':
                atlas_key = None
            elif atlas_mode == 'ckpt':
                back_key = key[10:]
                front_key = 'heads3d.heads.0.'
                atlas_key = front_key + back_key
            else:
                atlas_key = key
        elif 'detection_backbone' in key:
            if fcaf3d_mode == 'none':
                fcaf3d_key = None
            else:
                back_key = key[19:]
                front_key = 'backbone.'
                fcaf3d_key = front_key + back_key
        elif 'detection_head' in key:
            if fcaf3d_mode == 'none':
                fcaf3d_key = None
            else:
                if key == 'detection_head.cls_conv.kernel' or key == 'detection_head.cls_conv.bias':
                    kebab=0
                back_key = key[15:]
                front_key = 'neck_with_head.'
                fcaf3d_key = front_key + back_key
        if atlas_key != None:
            new_state_dict[key] = model_atlas[atlas_key]
        elif fcaf3d_key != None:
            new_state_dict[key] = model_fcaf3d[fcaf3d_key]
        else:
            new_state_dict[key] = model_mine[key]
            
    model_mine_full = torch.load(full_model_path)
    model_mine_full['state_dict'] = new_state_dict   
    torch.save(model_mine_full, result_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--atlas_model_path',
        type=str,
        default='/data1/sgl/work_dirs_atlas/arkit_recon/epoch_80.pth')
    parser.add_argument(
        '--fcaf3d_model_path',
        type=str,
        default='/data1/sgl/work_dirs_atlas/arkit_fcaf3d_two_stage/epoch_12.pth')
    parser.add_argument(
        '--full_model_path',
        type=str,
        default='/data1/sgl/work_dirs_atlas/arkit_atlas_fcaf3d_trial1.pth')
    parser.add_argument(
        '--result_path',
        type=str,
        default='/data1/sgl/work_dirs_atlas/arkit_atlas_fcaf3d_trial1.pth')
    parser.add_argument(
        '--result_type',
        type=str,
        default='full')
    args = parser.parse_args()
    assert args.result_type in ['full', 'atlas']
    
    if args.result_type == 'atlas':
        switch_atlas_ckpt(args.atlas_model_path, args.full_model_path, args.result_path)
    else:
        combine_atlas_fcaf3d(args.atlas_model_path, args.fcaf3d_model_path, args.full_model_path, args.result_path)

if __name__ == "__main__":
    main()