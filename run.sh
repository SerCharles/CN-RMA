CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train.sh projects/configs/neucon_pretrain.py 4 --work-dir work_dirs/neucon_pretrain
CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train.sh projects/configs/neucon.py 4 --work-dir work_dirs/neucon