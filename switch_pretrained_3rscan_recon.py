import torch
import pytorch_lightning
import json
model_atlas = torch.load('/home/sgl/work_dirs_atlas/recon_3rscan/best_500.pth')['state_dict']
model_mine = torch.load('/home/sgl/work_dirs_atlas/ray_marching_base_points.pth')['state_dict']

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


new_state_dict = model_mine.copy()
for key in model_atlas.keys():
    new_state_dict[key] = model_atlas[key]

model_mine_full = torch.load('/home/sgl/work_dirs_atlas/ray_marching_base_points.pth')
model_mine_full['state_dict'] = new_state_dict
torch.save(model_mine_full, '/home/sgl/work_dirs_atlas/3rscan_recon.pth')
