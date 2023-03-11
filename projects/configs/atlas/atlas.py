plugin = True
plugin_dir = 'projects/mvsdetection'
class_names = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
               'desk', 'curtain', 'refrigerator', 'showercurtain', 'toilet', 'sink', 'bathtub', 'garbagebin']
classes = len(class_names)

PIXEL_MEAN = [103.53, 116.28, 123.675]
PIXEL_STD = [1.0, 1.0, 1.0]
VOXEL_SIZE = 0.04
N_SCALES = 3
VOXEL_DIM_TRAIN = [160,160,64]
VOXEL_DIM_TEST = [256,256,96]
#
NUM_FRAMES_TRAIN = 30
NUM_FRAMES_TEST = 30
RANDOM_ROTATION_3D = True
RANDOM_TRANSLATION_3D = True
PAD_XY_3D = 1.0 
PAD_Z_3D = 0.25
LOSS_WEIGHT_TSDF = 1.0

optimizer = dict(
    type='Adam', 
    lr=5e-4
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#lr_config = dict(policy='step', warmup=None, step=[300], gamma=0.1)
lr_config = dict(policy='CosineAnnealing',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 min_lr_ratio=1e-3)

#find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/atlas'
load_from = None
resume_from = None
workflow = [('train', 1)]
total_epochs = 200
evaluation = dict(interval=3000, voxel_size=VOXEL_SIZE, save_path=work_dir+'/results')
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ]
)


train_pipeline = [
    dict(type='ResizeImage', size=((640, 480))),
    dict(type='AtlasToTensor'),
    dict(type='RandomTransformSpace', voxel_dim=VOXEL_DIM_TRAIN, 
        random_rotation=True, random_translation=True, paddingXY=PAD_XY_3D, paddingZ=PAD_Z_3D),
        #random_rotation=False, random_translation=False, paddingXY=0.0, paddingZ=0.0),
    dict(type='IntrinsicsPoseToProjection'),
    dict(type='AtlasCollectData')
]

test_pipeline = [
    dict(type='ResizeImage', size=((640, 480))),
    dict(type='AtlasToTensor'),
    dict(type='TestTransformSpace', voxel_dim=VOXEL_DIM_TEST, origin=[0, 0, 0]),
    dict(type='IntrinsicsPoseToProjection'),
    dict(type='AtlasCollectData')
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1, 
    train_dataloader=dict(shuffle=True),
    test_dataloader=dict(shuffle=False),
    train=dict(
        type='AtlasScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_train.pkl',
        classes=class_names, 
        pipeline=train_pipeline, 
        test_mode=False,
        num_frames=NUM_FRAMES_TRAIN,
        voxel_size=VOXEL_SIZE),
    val=dict(
        type='AtlasScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_val.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True,
        num_frames=NUM_FRAMES_TEST,
        voxel_size=VOXEL_SIZE),
    test=dict(
        type='AtlasScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_train.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True,
        num_frames=NUM_FRAMES_TEST,
        voxel_size=VOXEL_SIZE)
)


model = dict(
    type='Atlas',
    pixel_mean=PIXEL_MEAN,
    pixel_std=PIXEL_STD,
    voxel_size=VOXEL_SIZE,
    n_scales=N_SCALES,
    voxel_dim_train=VOXEL_DIM_TRAIN,
    voxel_dim_test=VOXEL_DIM_TEST,
    origin=[0,0,0],
    backbone2d_stride=4,
    resnet=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        pretrained='torchvision://resnet50',
        style='pytorch'
    ),
    fpn=dict(
        type='FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4
    ),
    feature_2d=dict(
        type='FPNFeature',
        feature_strides=[4, 8, 16, 32],
        feature_channels=[256, 256, 256, 256],
        output_dim=32,
        output_stride=4
    ),
    backbone_3d=dict(
        type='Backbone3D',
        channels=[32, 64, 128, 256],
        layers_down=[1, 2, 3, 4],
        layers_up=[3, 2, 1],
        drop=0.0, 
        zero_init_residual=True,
        cond_proj=False
    ),
    tsdf_head=dict(
        type='TSDFHead',
        input_channels=[32, 64, 128],
        n_scales=3,
        voxel_size=VOXEL_SIZE,
        loss_weight=LOSS_WEIGHT_TSDF,
        label_smoothing=1.05,
        sparse_threshold = [0.99, 0.99, 0.99]
    )
)