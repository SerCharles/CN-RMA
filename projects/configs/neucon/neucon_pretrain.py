plugin = True
plugin_dir = 'projects/mvsdetection'
class_names = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
               'desk', 'curtain', 'refrigerator', 'showercurtain', 'toilet', 'sink', 'bathtub', 'garbagebin']
classes = len(class_names)

PIXEL_MEAN = [103.53, 116.28, 123.675]
PIXEL_STD = [1.0, 1.0, 1.0]
N_LAYERS = 3
N_VOXELS = [96, 96, 96]
VOXEL_SIZE = 0.04
POS_WEIGHT = 1.5
THRESHOLDS = [0, 0, 0]
N_SCALES = len(THRESHOLDS) - 1
LOSS_WEIGHTS = [1.0, 0.8, 0.64]
SPARSEREG_DROPOUT = False 
USE_GRU_FUSION = False
TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
TEST_NUM_SAMPLE = [4096, 16384, 65536]
PAD_XY_3D = 0.1 
PAD_Z_3D = 0.025

train_pipeline = [
    dict(type='NeuConResizeImage', size=((640, 480))),
    dict(type='NeuConToTensor'),
    dict(type='NeuConRandomTransformSpace', voxel_dim=N_VOXELS, voxel_size=VOXEL_SIZE, 
         random_rotation=True, random_translation=True, paddingXY=PAD_XY_3D, paddingZ=PAD_Z_3D),
    dict(type='NeuConIntrinsicsPoseToProjection', n_views=9, stride=4),
    dict(type='NeuConCollectData')
]

test_pipeline = [
    dict(type='NeuConResizeImage', size=((640, 480))),
    dict(type='NeuConToTensor'),
    dict(type='NeuConRandomTransformSpace', voxel_dim=N_VOXELS, voxel_size=VOXEL_SIZE, 
         random_rotation=False, random_translation=False, paddingXY=0, paddingZ=0),
    dict(type='NeuConIntrinsicsPoseToProjection', n_views=9, stride=4),
    dict(type='NeuConCollectData')
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2, 
    train_dataloader=dict(shuffle=False),
    test_dataloader=dict(shuffle=False),
    train=dict(
        type='NeuconScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_train.pkl',
        classes=class_names, 
        pipeline=train_pipeline, 
        test_mode=False),
    val=dict(
        type='NeuconScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_val.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True),
    test=dict(
        type='NeuconScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_val.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True),
)

model=dict(
    type='NeuralRecon',
    pixel_mean=PIXEL_MEAN,
    pixel_std=PIXEL_STD,
    n_scales=N_SCALES,
    loss_weights=LOSS_WEIGHTS,
    backbone2d=dict(
        type='MnasMulti',
        alpha=1.0
    ),
    head=dict(
        type='NeuConHead',
        alpha=1,
        n_layers=N_LAYERS,
        voxel_size=VOXEL_SIZE,
        n_voxels=N_VOXELS,
        sparsereg_dropout=SPARSEREG_DROPOUT,
        pos_weight=POS_WEIGHT,
        num_sample=TRAIN_NUM_SAMPLE,
        thresholds=THRESHOLDS,
        use_gru_fusion=USE_GRU_FUSION
    ),
    fuse_to_global=dict(
        type='GRUFusion',
        direct_substitute=True,
        thresholds=THRESHOLDS,
        n_layers=N_LAYERS,
        voxel_size=VOXEL_SIZE,
        n_voxels=N_VOXELS,
        use_gru_fusion=USE_GRU_FUSION
    )
)

optimizer = dict(
    type='Adam', 
    lr=1e-3, 
    betas=(0.9, 0.999),
    weight_decay=0.0
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#lr_config = dict(policy='step', warmup=None, step=[12, 24, 48], gamma=0.5)
lr_config = dict(policy='CosineAnnealing',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 min_lr_ratio=1e-3)



find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/neucon_pretrain'
resume_from = None 
load_from = None
workflow = [('train', 1)]
total_epochs = 20 
evaluation = dict(interval=50, voxel_size=VOXEL_SIZE, save_path=work_dir+'/results')
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)