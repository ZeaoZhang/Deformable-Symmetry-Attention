auto_scale_lr = dict(base_batch_size=16, enable=False)
checkpoint_config = dict(by_epoch=True, interval=1)
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=(239.38, ),
    pad_val=0,
    seg_pad_val=0,
    size=(
        512,
        512,
    ),
    std=(37.3, ),
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = '/'
dataset_type = 'MyDataset_checkXray2'
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
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
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
load_from = '../checkpoints/mask2former_r101_8xb2-90k_cityscapes-512x1024_20221130_031628-43e68666.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    backbone=dict(
        conv_type=dict(
            in_channels=1,
            kernel_size=9,
            limit=0.2551,
            loss_weight=1.0,
            mean=[
                0.0,
                0.0,
            ],
            out_channels=4,
            sigma=[
                1.0,
                1.0,
            ]),
        deep_stem=False,
        depth=101,
        frozen_stages=-1,
        in_channels=5,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='DSAResNet5'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=(239.38, ),
        pad_val=0,
        seg_pad_val=0,
        size=(
            512,
            512,
        ),
        std=(37.3, ),
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        loss_cls=dict(
            class_weight=(
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ),
            loss_weight=2.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=4,
        num_queries=100,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='mmdet.MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=5.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    loss_kernel=False,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EmbeddinSegmentors2')
norm_cfg = dict(requires_grad=True, type='BN')
num_classes = 4
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
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=1.0),
            level_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_feat=dict(decay_mult=0.0, lr_mult=1.0)),
        norm_decay_mult=0.0),
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
    dict(begin=0, by_epoch=True, end=3, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=3,
        by_epoch=True,
        end=50,
        eta_min_ratio=1e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=0)
resume = False
scale_size = (
    1024,
    1024,
)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        './splilts/val_reg.txt',
        data_prefix=dict(
            img_path=
            '../data/chexmask/ssim+grid_1+nj_1e-7',
            seg_map_path=
            '../data/chexmask/masks'),
        data_root='/',
        pipeline=[
            dict(type='LoadImageFromNPYFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MyDataset_checkXray2'),
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
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        './splilts/train_reg.txt',
        data_prefix=dict(
            img_path=
            '../data/chexmask/ssim+grid_1+nj_1e-7',
            seg_map_path=
            '../data/chexmask/masks'),
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
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='MyContrastDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='MyDataset_checkXray2'),
    drop_last=True,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromNPYFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='MyContrastDistortion'),
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
        './splilts/val_reg.txt',
        data_prefix=dict(
            img_path=
            '../data/chexmask/ssim+grid_1+nj_1e-7',
            seg_map_path=
            '../data/chexmask/masks'),
        data_root='/',
        pipeline=[
            dict(type='LoadImageFromNPYFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MyDataset_checkXray2'),
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
            group='chexray',
            job_type='mask2former',
            name='3-freesom+gussion4+loss0+limit0.2551',
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
                group='chexray',
                job_type='mask2former',
                name='3-freesom+gussion4+loss0+limit0.2551',
                project='papar'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dirs/checkXray/mask2former/3-freesom+gussion4+loss0+limit0.2551'
