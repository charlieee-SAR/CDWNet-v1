# segmentation/configs/_base_/datasets/custom_dataset.py

# dataset settings
dataset_type = 'CustomDataset'  # 使用MMSeg内置的CustomDataset
data_root = 'data/dataset1118_bands5_AnLiao'  # 【修改】你的数据集路径

# 6波段图像的归一化参数（需要根据你的数据统计）
img_norm_cfg = dict(
    # mean=[8.579, 3.580, 10.183, 3.263, 0.271, 1.410],  # 6个通道的均值
    # std=[6.251, 4.508, 6.099, 4.256, 0.175, 0.537],        # 6个通道的标准差
    mean = [11.741517, 5.3139, 14.216368, 3.354613, 1.271325],
    std = [8.133328, 6.439671, 8.195287, 3.953424, 0.515281],
    to_rgb=False)  # 设置为False，因为不是RGB

# 【重要】根据你的图像尺寸调整
img_scale = (256, 256)  # (width, height)
crop_size = (256, 256)   # 训练时的裁剪尺寸

# 训练数据增强pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', use_tifffile=True),  # 使用tifffile
    dict(type='LoadAnnotations', reduce_zero_label=False),  # reduce_zero_label=False保持标签不变
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# 测试pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # 多尺度测试(可选)
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 数据加载配置
data = dict(
    samples_per_gpu=2,  # batch size per GPU
    workers_per_gpu=4,   # 数据加载线程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',       # 【修改】对应你的训练图像目录
        ann_dir='annotations/training',   # 【修改】对应你的训练标注目录
        img_suffix='.tif',               # 【修改】图像后缀
        seg_map_suffix='.png',           # 【修改】标注后缀
        pipeline=train_pipeline,
        classes=('background', 'rice'),  # 可选:显式指定类别
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'rice'),
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=('background', 'rice'),
    )
)