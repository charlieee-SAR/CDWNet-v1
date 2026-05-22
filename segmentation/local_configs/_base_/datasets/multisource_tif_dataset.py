# Dataset template for paired Optical/SAR segmentation.

dataset_type = 'MultiSourceTifDataset'
data_root = 'data/your_multisource_dataset'

# Total channels = optical_channels + sar_channels
optical_channels = 3
sar_channels = 2

img_norm_cfg = dict(
    mean=[0.0] * (optical_channels + sar_channels),
    std=[1.0] * (optical_channels + sar_channels),
    to_rgb=False)

img_scale = (256, 256)
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadMultiSourceImageFromFile', use_tifffile=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadMultiSourceImageFromFile', use_tifffile=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
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
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        optical_dir='optical/training',
        sar_dir='sar/training',
        ann_dir='annotations/training',
        optical_suffix='.tif',
        sar_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        classes=('background', 'target')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        optical_dir='optical/validation',
        sar_dir='sar/validation',
        ann_dir='annotations/validation',
        optical_suffix='.tif',
        sar_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'target')),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        optical_dir='optical/validation',
        sar_dir='sar/validation',
        ann_dir='annotations/validation',
        optical_suffix='.tif',
        sar_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'target')))
