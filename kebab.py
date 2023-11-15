import os 
data_path = '/data1/sgl/3RScan/atlas_tsdf'
val_path = '/data1/sgl/3RScan/meta_data/val.txt'
test_path = '/data1/sgl/3RScan/meta_data/test.txt'
save_path = '/data1/sgl/3RScan/meta_data/recon_train.txt'
scene_files = os.listdir(data_path)

with open(val_path, 'r') as f:
    val_ids = f.readlines()
val_ids = [val_id.strip() for val_id in val_ids]
with open(test_path, 'r') as f:
    test_ids = f.readlines()
test_ids = [test_id.strip() for test_id in test_ids]

scene_ids = []
for scene in scene_files:
    if (not scene in val_ids) and (not scene in test_ids):
        scene_ids.append(scene)

with open(save_path, 'w') as f:
    f.write('\n'.join(scene_ids))
kebab=0
