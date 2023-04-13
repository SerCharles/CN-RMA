import torch
import pytorch_lightning
import json
model_atlas = torch.load('/data/shenguanlin/work_dirs_atlas/atlas_mine/final.ckpt')['state_dict']
model_fcaf3d = torch.load('/data1/shenguanlin/middle_test/epoch_12.pth')['state_dict']
model_mine = torch.load('/data/shenguanlin/work_dirs_atlas/atlas_test/test.pth')['state_dict']

atlas = {}
mine = {}
resnet_mine = {}
resnet_atlas = {}
fpn_atlas = {}
fpn_mine = {}
backbone2d_atlas = {}
backbone2d_mine = {}
backbone3d_atlas = {}
backbone3d_mine = {}
head_atlas = {}
head_mine = {}
fcaf3d_backbone = {}
fcaf3d_backbone_mine = {}
fcaf3d_head = {}
fcaf3d_head_mine = {}

for key in model_atlas.keys():
    atlas[key] = model_atlas[key].shape
    if 'backbone2d.0.bottom_up.' in key:
        resnet_atlas[key] = model_atlas[key].shape
    elif 'backbone2d.0.fpn' in key:
        fpn_atlas[key] = model_atlas[key].shape
    elif 'backbone2d.1.' in key:
        backbone2d_atlas[key] = model_atlas[key].shape
    elif 'backbone3d' in key:
        backbone3d_atlas[key] = model_atlas[key].shape
    elif 'heads3d' in key:
        head_atlas[key] = model_atlas[key].shape

for key in model_fcaf3d.keys():
    atlas[key] = model_fcaf3d[key].shape 
    if 'backbone' in key:
        fcaf3d_backbone[key] = model_fcaf3d[key].shape 
    elif 'neck_with_head' in key:
        fcaf3d_head[key] = model_fcaf3d[key].shape
    
for key in model_mine.keys():
    mine[key] = model_mine[key].shape
    if 'fpn.bottom_up' in key:
        resnet_mine[key] = model_mine[key].shape
    elif 'fpn' in key:
        fpn_mine[key] = model_mine[key].shape
    elif 'feature_2d' in key:
        backbone2d_mine[key] = model_mine[key].shape
    elif 'backbone3d' in key:
        backbone3d_mine[key] = model_mine[key].shape
    elif 'tsdf_head' in key:
        head_mine[key] = model_mine[key].shape
    elif 'detection_backbone' in key:
        fcaf3d_backbone_mine[key] = model_mine[key].shape
    elif 'detection_head' in key:
        fcaf3d_head_mine[key] = model_mine[key].shape
        


new_state_dict = model_mine.copy()
for key in model_mine.keys():
    atlas_key = None 
    fcaf3d_key = None
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
    elif 'detection_backbone' in key:
        back_key = key[19:]
        front_key = 'backbone.'
        fcaf3d_key = front_key + back_key
    elif 'detection_head' in key:
        back_key = key[15:]
        front_key = 'neck_with_head.'
        fcaf3d_key = front_key + back_key
    if atlas_key != None:
        new_state_dict[key] = model_atlas[atlas_key]
    elif fcaf3d_key != None:
        new_state_dict[key] = model_fcaf3d[fcaf3d_key]
    

model_mine_full = torch.load('/data/shenguanlin/work_dirs_atlas/atlas_test/test.pth')
model_mine_full['state_dict'] = new_state_dict
torch.save(model_mine_full, '/data/shenguanlin/work_dirs_atlas/atlas_test/test_new.pth')

