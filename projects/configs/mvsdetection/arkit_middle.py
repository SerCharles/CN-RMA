'''Given trained Atlas reconstruction, get point cloud with features based on ray marching method for pretraining FCAF3D detection model
'''

plugin = True
plugin_dir = 'projects/mvsdetection'

class_names = ["cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool", "chair", "table", "tv_monitor", "sofa"]
classes = len(class_names)

PIXEL_MEAN = [103.53, 116.28, 123.675]
PIXEL_STD = [1.0, 1.0, 1.0]
VOXEL_SIZE = 0.04
VOXEL_SIZE_FCAF3D = 0.01
N_SCALES = 3
VOXEL_DIM_TRAIN = [192, 192, 80]
VOXEL_DIM_TEST = [192, 192, 80]
NUM_FRAMES_TRAIN = 40
#NUM_FRAMES_TEST = 500
NUM_FRAMES_TEST = 40
USE_BATCHNORM_TRAIN = True
USE_BATCHNORM_TEST = True
LOSS_WEIGHT_RECON = 0.5
LOSS_WEIGHT_DETECTION = 1.0
#fp16 = dict(loss_scale=512.)

#ray marching utils
RAY_MARCHING_TYPE = 'neus'
NEUS_THRESHOLD = 0.05
DEPTH_POINTS = None

#middle data save utils
MIDDLE_SAVE_PATH = '/data1/sgl/ARKit/atlas_middle_data_16016064'
MIDDLE_VISUALIZE_PATH = None


optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[27, 36])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/data1/sgl/work_dirs_atlas/test_2'
R50_path = '/home/sgl/work_dirs_atlas/R-50.pth'
save_path = work_dir + '/results'
load_from = None
resume_from = None




workflow = [('train', 1)]
total_epochs = 360
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
    dict(type='AtlasTransformSpaceDetection', voxel_dim=VOXEL_DIM_TRAIN, 
         origin=[0, 0, 0], test=False, mode='middle'),
    dict(type='AtlasIntrinsicsPoseToProjection'),
    dict(type='AtlasCollectData')
]

test_pipeline = [
    dict(type='AtlasResizeImage', size=((640, 480))),
    dict(type='AtlasToTensor'),
    dict(type='AtlasTransformSpaceDetection', voxel_dim=VOXEL_DIM_TEST, 
         origin=[0, 0, 0], test=True, mode='middle'),    
    dict(type='AtlasIntrinsicsPoseToProjection'),
    dict(type='AtlasCollectData')
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1, 
    train_dataloader=dict(shuffle=True),
    test_dataloader=dict(shuffle=False),
    train=dict(
        type='AtlasARKitDataset',
        data_root='./data/arkit',
        ann_file='./data/arkit/arkit_infos_train.pkl',
        classes=class_names, 
        pipeline=train_pipeline, 
        test_mode=False,
        num_frames=NUM_FRAMES_TRAIN,
        voxel_size=VOXEL_SIZE,
        select_type='random'),
    val=dict(
        type='AtlasARKitDataset',
        data_root='./data/arkit',
        ann_file='./data/arkit/arkit_infos_val.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True,
        num_frames=NUM_FRAMES_TEST,
        voxel_size=VOXEL_SIZE,
        select_type='random'),
    test=dict(
        type='AtlasARKitDataset',
        data_root='./data/arkit',
        ann_file='./data/arkit/arkit_infos_train.pkl',
        classes=class_names, 
        pipeline=test_pipeline, 
        test_mode=True,
        num_frames=NUM_FRAMES_TEST,
        voxel_size=VOXEL_SIZE,
        select_type='random')
)


model = dict(
    type='RayMarching',
    pixel_mean=PIXEL_MEAN,
    pixel_std=PIXEL_STD,
    voxel_size=VOXEL_SIZE,
    n_scales=N_SCALES,
    voxel_dim_train=VOXEL_DIM_TRAIN,
    voxel_dim_test=VOXEL_DIM_TEST,
    origin=[0,0,0],
    backbone2d_stride=4,
    loss_weight_detection=LOSS_WEIGHT_DETECTION, 
    loss_weight_recon=LOSS_WEIGHT_RECON,
    voxel_size_fcaf3d=VOXEL_SIZE_FCAF3D,
    use_batchnorm_train=USE_BATCHNORM_TRAIN,
    use_batchnorm_test=USE_BATCHNORM_TEST,
    save_path=save_path,
    ray_marching_type=RAY_MARCHING_TYPE,
    neus_threshold=NEUS_THRESHOLD,
    depth_points=DEPTH_POINTS, 
    middle_save_path=MIDDLE_SAVE_PATH,
    middle_visualize_path=MIDDLE_VISUALIZE_PATH, 
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
        pretrained=R50_path
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
    ),
    detection_backbone=dict(
        type='FCAF3DBackbone',
        in_channels=32,
        depth=34),
    detection_head=dict(
        type='FCAF3DHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        pts_threshold=200000,
        n_classes=classes,
        n_reg_outs=8,
        voxel_size=VOXEL_SIZE_FCAF3D,
        assigner=dict(
            type='FCAF3DAssigner',
            limit=27,
            topk=18,
            n_scales=4),
        loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0, with_yaw=True),
        train_cfg=dict(),
        test_cfg=dict(
            nms_pre=1000,
            iou_thr=.5,
            score_thr=.01)),
        use_feature_transform=True,
        feature_transform=dict(
            flip_ratio_horizontal=0.5,
            flip_ratio_vertical=0.5,
            rot_range=[-0.087266, 0.087266],
            scale_ratio_range=[.9, 1.1],
            translation_std=[.1, .1, .1]),
        max_points=500000)
