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
NUM_FRAMES_TRAIN = 50
NUM_FRAMES_TEST = 500
RANDOM_ROTATION_3D = True
RANDOM_TRANSLATION_3D = True
PAD_XY_3D = 1.0
PAD_Z_3D = 0.25
fp16 = dict(loss_scale=512.)

optimizer = dict(
    type='Adam', 
    lr=5e-4
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[300], gamma=0.1)
'''lr_config = dict(policy='CosineAnnealing',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 min_lr_ratio=1e-3)'''

#find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/atlas'
load_from = '/data/shenguanlin/atlas_mine/switch.pth'
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
    ]
)


train_pipeline = [
    dict(type='AtlasResizeImage', size=((640, 480))),
    dict(type='AtlasToTensor'),
    dict(type='AtlasRandomTransformSpaceRecon', voxel_dim=VOXEL_DIM_TRAIN, 
        random_rotation=True, random_translation=True, paddingXY=PAD_XY_3D, paddingZ=PAD_Z_3D),
    dict(type='AtlasIntrinsicsPoseToProjection'),
    dict(type='AtlasCollectData')
]

test_pipeline = [
    dict(type='AtlasResizeImage', size=((640, 480))),
    dict(type='AtlasToTensor'),
    dict(type='AtlasTestTransformSpace', voxel_dim=VOXEL_DIM_TEST, origin=[0, 0, 0]),
    dict(type='AtlasIntrinsicsPoseToProjection'),
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
        voxel_size=VOXEL_SIZE,
        select_type='random'),
    val=dict(
        type='AtlasScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_val.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True,
        num_frames=NUM_FRAMES_TEST,
        voxel_size=VOXEL_SIZE,
        select_type='random'),
    test=dict(
        type='AtlasScanNetDataset',
        data_root='./data/scannet',
        ann_file='./data/scannet/scannet_infos_val.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True,
        num_frames=NUM_FRAMES_TEST,
        voxel_size=VOXEL_SIZE,
        select_type='random')
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
    backbone2d=dict(
        type='FPNDetectron',
        bottom_up_cfg=dict(
            input_channels=3,
            norm='BN',
            depth=50,
            out_features=["res2", "res3", "res4", "res5"],
            num_groups=1,
            width_per_group=64, 
            stride_in_1x1=True, 
            res5_dilation=1, 
            res2_out_channels=256, 
            stem_out_channels=64,
            freeze_at=2
        ),
        in_features = ["res2", "res3", "res4", "res5"],
        out_channels=256,
        norm='BN',
        fuse_type='sum',
        pretrained='/data/shenguanlin/atlas_mine/R-50.pth'
    ),
    feature_2d=dict(
        type='AtlasFPNFeature',
        feature_strides={'p2':4, 'p3':8, 'p4':16, 'p5':32, 'p6':64},
        feature_channels={'p2':256, 'p3':256, 'p4':256, 'p5':256, 'p6':256},
        output_dim=32, 
        output_stride=4, 
        norm='BN'
    ),
    backbone_3d=dict(
        type='AtlasBackbone3D',
        channels=[32, 64, 128, 256],
        layers_down=[1, 2, 3, 4],
        layers_up=[3, 2, 1],
        drop=0.0, 
        zero_init_residual=True,
        cond_proj=False,
        norm='BN'
    ),
    tsdf_head=dict(
        type='AtlasTSDFHead',
        input_channels=[32, 64, 128],
        n_scales=3,
        voxel_size=VOXEL_SIZE,
        label_smoothing=1.05,
        sparse_threshold = [0.99, 0.99, 0.99]
    )
)