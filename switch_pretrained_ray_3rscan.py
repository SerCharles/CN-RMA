import torch
import pytorch_lightning
import json
model_atlas = torch.load('/home/sgl/work_dirs_atlas/3rscan_atlas_recon_old/best_500.pth')['state_dict']
model_ray = torch.load('/home/sgl/work_dirs_atlas/3rscan_stage_2/epoch_11.pth')['state_dict']
model_mine = torch.load('/home/sgl/work_dirs_atlas/pipeline_link.pth')['state_dict']

new_state_dict = model_mine.copy()
for key in model_atlas.keys():
    new_state_dict[key] = model_atlas[key]

#print(model_mine.keys())
#print(model_ray.keys())

for key in model_mine.keys():
    fcaf3d_key = None
    if 'detection_backbone' in key:
        back_key = key[19:]
        front_key = 'backbone.'
        fcaf3d_key = front_key + back_key
    elif 'detection_head' in key:
        back_key = key[15:]
        front_key = 'neck_with_head.'
        fcaf3d_key = front_key + back_key

    if fcaf3d_key != None:
        new_state_dict[key] = model_ray[fcaf3d_key]
    

model_mine_full = torch.load('/home/sgl/work_dirs_atlas/pipeline_link.pth')
model_mine_full['state_dict'] = new_state_dict
torch.save(model_mine_full, '/home/sgl/work_dirs_atlas/3rscan_ray_trial_2.pth')

