_base_ = [
    '../_base_/datasets/multisource_tif_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

data_root = 'data/OS-CDW'
classes = ('background', 'garbage')
optical_channels = 3
sar_channels = 3
img_norm_cfg = dict(
    mean=[
        1.4063484959761848e-05,
        1.2122119876419955e-05,
        9.797758427178388e-06,
        0.1405809003496627,
        0.024873966822098307,
        1.2069092949081635,
    ],
    std=[
        5.561451493375649e-06,
        4.823169442867821e-06,
        4.88905831993842e-06,
        1.608902579556359,
        0.05450258921209587,
        0.9202422888483104,
    ],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadMultiSourceImageFromFile', use_tifffile=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadMultiSourceImageFromFile', use_tifffile=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(
        data_root=data_root,
        optical_dir='optical/training',
        sar_dir='sar/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        data_root=data_root,
        optical_dir='optical/validation',
        sar_dir='sar/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        data_root=data_root,
        optical_dir='optical/validation',
        sar_dir='sar/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        classes=classes),
    samples_per_gpu=2,
    workers_per_gpu=4)

optical_in_channels = 3
sar_in_channels = 3
num_classes = 2
fusion = 'concat'

model = dict(
    type='DualBranchEncoderDecoder_CFI_RS_V2_BAM_Pyramid_4Scale',
    optical_in_channels=optical_in_channels,
    sar_in_channels=sar_in_channels,
    fusion=fusion,
    cfi_rs=dict(
        level_indices=[1, 2],
        pool_kernel=3,
        gate_kernel=3),
    ms_bam=dict(
        level_indices=[0, 1, 2, 3],
        edge_kernel=3,
        refine_kernel=3),
    auto_infer_decode_channels=False,
    infer_input_size=(256, 256),
    backbone_optical=dict(
        type='ExternalFlashInternImage',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/flash_intern_image_t_1k_224.pth'),
        core_op='DCNv4',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3)),
    backbone_sar=dict(
        type='ExternalVMambaBackbone',
        out_indices=(0, 1, 2, 3),
        pretrained='pretrained/vmamba_vssm1_tiny_0230s_epoch264.pth',
        in_chans=sar_in_channels,
        dims=96,
        depths=(2, 2, 8, 2),
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=1.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.2,
        norm_layer='ln2d'),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[160, 320, 640, 1280],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        decoder_params=dict(embed_dim=256),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=20)
evaluation = dict(
    interval=1,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    rule='greater')
workflow = [('train', 1), ('val', 1)]

work_dir = 'work_dirs/CDWNet'
cudnn_benchmark = True
