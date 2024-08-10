checkpoint_config = dict(by_epoch=True, interval=1)
crop_size = (
    1024,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=(238.97, ),
    pad_val=0,
    seg_pad_val=0,
    size=(
        1024,
        512,
    ),
    std=(37.08, ),
    type='SegDataPreProcessor')
data_root = '/'
dataset_type = 'MyDataset2'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mDice',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    605,
    700,
)
load_from = '../checkpoints/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_stare_20211210_201825-21db614c.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=3,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        conv_type=dict(
            in_channels=1,
            kernel_size=9,
            limit=0.2551,
            loss_weight=1.0,
            mean=[
                0.0,
                0.0,
            ],
            out_channels=6,
            sigma=[
                1.0,
                1.0,
            ]),
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=7,
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='DSAUNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=(238.97, ),
        pad_val=0,
        seg_pad_val=0,
        size=(
            1024,
            512,
        ),
        std=(37.08, ),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=16,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=[
            dict(loss_name='loss_dice', loss_weight=3.0, type='DiceLoss'),
            dict(
                loss_name='loss_ce', loss_weight=1.0, type='CrossEntropyLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=3,
        type='ASPPHead'),
    loss_kernel=False,
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EmbeddinSegmentors')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.0001,
    type='AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(begin=0, by_epoch=True, end=50, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=50,
        by_epoch=True,
        end=1000,
        eta_min_ratio=1e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=0)
resume = False
scale_size = (
    2048,
    1024,
)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        './splits/val_new.txt',
        data_prefix=dict(
            img_path='../wbs/data/img_reg',
            seg_map_path='../wbs/data/mask'),
        data_root='/',
        pipeline=[
            dict(type='LoadImageFromNPYFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MyDataset2'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromNPYFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=1000, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        './splits/train_new.txt',
        data_prefix=dict(
            img_path='../wbs/data/img_reg',
            seg_map_path='../wbs/data/mask'),
        data_root='/',
        pipeline=[
            dict(type='LoadImageFromNPYFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                resize_type='Resize',
                scale=(
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    1024,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='MyDataset2'),
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromNPYFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        resize_type='Resize',
        scale=(
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        1024,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        './splits/val_new.txt',
        data_prefix=dict(
            img_path='../wbs/data/img_reg',
            seg_map_path='../wbs/data/mask'),
        data_root='/',
        pipeline=[
            dict(type='LoadImageFromNPYFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MyDataset2'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        init_kwargs=dict(
            group='aWBS',
            job_type='Unet',
            name='gussion6+loss0+limit0.2551',
            project='papar'),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            init_kwargs=dict(
                group='aWBS',
                job_type='Unet',
                name='gussion6+loss0+limit0.2551',
                project='papar'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dirs/aWBS/Unet/gussion6+loss0+limit0.2551'
