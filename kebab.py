import torch
import pytorch_lightning
import json
model_atlas = torch.load('/home/shenguanlin/Atlas/weights/final.ckpt')['state_dict']
model_mine = torch.load('/data4/sgl/mine/atlas/epoch_30.pth')['state_dict']
#torch.save(model_atlas, '/home/shenguanlin/atlas.pt')
#torch.save(model_mine, '/home/shenguanlin/mine.pt')



resnet_atlas_0 = {}
resnet_mine_0 = {}
resnet_atlas_1 = {}
resnet_mine_1 = {}
resnet_atlas_2 = {}
resnet_mine_2 = {}
resnet_atlas_3 = {}
resnet_mine_3 = {}
resnet_atlas_4 = {}
resnet_mine_4 = {}

for key in model_atlas.keys():
    if 'backbone2d.0.bottom_up.res2' in key:
        new_key = key[22:]
        resnet_atlas_1[new_key] = model_atlas[key].shape
        
    elif 'backbone2d.0.bottom_up.res3' in key:
        new_key = key[22:]
        resnet_atlas_2[new_key] = model_atlas[key].shape
        
    elif 'backbone2d.0.bottom_up.res4' in key:
        new_key = key[22:]
        resnet_atlas_3[new_key] = model_atlas[key].shape
        
    elif 'backbone2d.0.bottom_up.res5' in key:
        new_key = key[22:]
        resnet_atlas_4[new_key] = model_atlas[key].shape
    elif 'backbone2d.0.bottom_up.' in key:
        new_key = key[22:]
        resnet_atlas_0[new_key] = model_atlas[key].shape

for key in model_mine.keys():
    if 'resnet.layer1' in key:
        resnet_mine_1[key] = model_mine[key].shape
    elif 'resnet.layer2' in key:
        resnet_mine_2[key] = model_mine[key].shape    
    elif 'resnet.layer3' in key:
        resnet_mine_3[key] = model_mine[key].shape 
    elif 'resnet.layer4' in key:
        resnet_mine_4[key] = model_mine[key].shape     
    elif 'resnet.' in key:
        resnet_mine_0[key] = model_mine[key].shape   
        
with open("/home/shenguanlin/resnet_atlas_0.json","w") as f:
    json.dump(resnet_atlas_0, f)
with open("/home/shenguanlin/resnet_mine_0.json","w") as f:
    json.dump(resnet_mine_0, f)
    
with open("/home/shenguanlin/resnet_atlas_1.json","w") as f:
    json.dump(resnet_atlas_1, f)
with open("/home/shenguanlin/resnet_mine_1.json","w") as f:
    json.dump(resnet_mine_1, f)

with open("/home/shenguanlin/resnet_atlas_2.json","w") as f:
    json.dump(resnet_atlas_2, f)
with open("/home/shenguanlin/resnet_mine_2.json","w") as f:
    json.dump(resnet_mine_2, f)

with open("/home/shenguanlin/resnet_atlas_3.json","w") as f:
    json.dump(resnet_atlas_3, f)
with open("/home/shenguanlin/resnet_mine_3.json","w") as f:
    json.dump(resnet_mine_3, f)

with open("/home/shenguanlin/resnet_atlas_4.json","w") as f:
    json.dump(resnet_atlas_4, f)
with open("/home/shenguanlin/resnet_mine_4.json","w") as f:
    json.dump(resnet_mine_4, f)
    
    

