model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b2'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1408,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, )))
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=260),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=260),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='/media/duanbo/hd_2/mmlab/mmpretrain/data',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='EfficientNetRandomCrop', scale=260),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='/media/duanbo/hd_2/mmlab/mmpretrain/data',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='EfficientNetCenterCrop', crop_size=260),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='/media/duanbo/hd_2/mmlab/mmpretrain/data',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='EfficientNetCenterCrop', crop_size=260),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, ))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=256)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = '/media/duanbo/hd_2/mmlab/mmpretrain/tools/work_dirs/efficientnet-b2_8xb32-01norm_in1k/efficientnet-b2_3rdparty-ra-noisystudent_in1k_20221103-301ed299.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
launcher = 'none'
work_dir = './work_dirs/efficientnet-b2_8xb32-01norm_in1k'
