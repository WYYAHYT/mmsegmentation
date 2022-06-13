_base_ = [
    '../_base_/models/cgnet.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        by_epoch=False,
        begin=0,
        end=60000)
]
# runtime settings
total_iters = 60000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iters, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=4000))

crop_size = (680, 680)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
