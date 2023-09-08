_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/sport-fields_320x320.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (320, 320)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
