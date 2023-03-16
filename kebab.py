import torch
import pytorch_lightning
import json
model_atlas = torch.load('/data/shenguanlin/atlas/final.ckpt')['state_dict']
model_mine = torch.load('/data/shenguanlin/atlas/epoch_1.pth')['state_dict']

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

model_mine_full = torch.load('/data/shenguanlin/atlas/epoch_1.pth')
model_mine_full['state_dict'] = new_state_dict
torch.save(model_mine_full, '/data/shenguanlin/atlas/switch.pth')
'''
with open("/home/sgl/atlas.json","w") as f:
    json.dump(atlas, f)
with open("/home/sgl/mine.json","w") as f:
    json.dump(mine, f)
    
with open("/home/sgl/fpn_atlas.json","w") as f:
    json.dump(fpn_atlas, f)
with open("/home/sgl/fpn_mine.json","w") as f:
    json.dump(fpn_mine, f)

with open("/home/sgl/backbone2d_atlas.json","w") as f:
    json.dump(backbone2d_atlas, f)
with open("/home/sgl/backbone2d_mine.json","w") as f:
    json.dump(backbone2d_mine, f)

with open("/home/sgl/backbone3d_atlas.json","w") as f:
    json.dump(backbone3d_atlas, f)
with open("/home/sgl/backbone3d_mine.json","w") as f:
    json.dump(backbone3d_mine, f)

with open("/home/sgl/head_atlas.json","w") as f:
    json.dump(head_atlas, f)
with open("/home/sgl/head_mine.json","w") as f:
    json.dump(head_mine, f)
'''

