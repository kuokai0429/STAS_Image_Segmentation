''' Code for training - Cascade Mask R-CNN with Dual Swin Transformer ( DB-Swin-S )
    Command: python train_stas_seg.py --weight_dir cascade_mask_rcnn_dual_swin_s_2_089_Weights --checkpoint_file cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth --config_files cascade_mask_rcnn_swin_small.py cascade_mask_rcnn_cbv2_swin_small.py
'''

print("Running ....")
print("Ignore: 'apex is not installed'")

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataset
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from mmdet.apis import train_detector

import os
import os.path as osp
from pathlib import Path
import shutil
import torch
import argparse

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--weight_dir', required=True, type=str, help="Pretrained weights directory to load. Please put under /Weights folder")
parser.add_argument('--checkpoint_file', required=True, type=str, help="Pretrained checkpoint weights directory to load. Please put under /Weights folder")
parser.add_argument('--config_files', required=True, type=str, nargs='+', help="Configuration files to load. Please put under /Configs folder")
args = parser.parse_args()


####### Loading directories path

current_dir_root = os.getcwd()
weight_dir_root = args.weight_dir
checkpoint_file_root = args.checkpoint_file
config_file_root = args.config_files
# weight_dir_root = "cascade_mask_rcnn_dual_swin_s_2_089_Weights"
# checkpoint_file_root = "cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth"
# config_file_root = ["cascade_mask_rcnn_swin_small.py", "cascade_mask_rcnn_cbv2_swin_small.py"]


####### Prepare model config

# Modify the pretrained weights
pretrained_weights  = torch.load(current_dir_root + '/Weights/' + checkpoint_file_root)

num_class = 1
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_reg.bias'].resize_(num_class*4)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_reg.bias'].resize_(num_class*4)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_reg.bias'].resize_(num_class*4)
pretrained_weights['state_dict']['roi_head.mask_head.0.conv_logits.weight'].resize_(num_class, 256, 1, 1)
pretrained_weights['state_dict']['roi_head.mask_head.0.conv_logits.bias'].resize_(num_class)
pretrained_weights['state_dict']['roi_head.mask_head.1.conv_logits.weight'].resize_(num_class, 256, 1, 1)
pretrained_weights['state_dict']['roi_head.mask_head.1.conv_logits.bias'].resize_(num_class)
pretrained_weights['state_dict']['roi_head.mask_head.2.conv_logits.weight'].resize_(num_class, 256, 1, 1)
pretrained_weights['state_dict']['roi_head.mask_head.2.conv_logits.bias'].resize_(num_class)

torch.save(pretrained_weights, current_dir_root + "/Weights/cascade_mask_rcnn_cbv2_swin_small.pth")

# Create work directory
dirpath = Path(current_dir_root + "/CBNetV2/runs")

if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
    
os.mkdir(current_dir_root + "/CBNetV2/runs")

# Prepare the train/valid coco json annotation file
shutil.copytree(current_dir_root + '/Weights/' + weight_dir_root + '/labelme2coco', current_dir_root + "/CBNetV2/runs/labelme2coco")

# The new config inherits a base config to highlight the necessary modification
shutil.copy(current_dir_root + '/Configs/' + config_file_root[0], current_dir_root + '/CBNetV2/configs/swin/' + config_file_root[0])
shutil.copy(current_dir_root + '/Configs/' + config_file_root[1], current_dir_root + '/CBNetV2/configs/cbnet/' + config_file_root[1])
cfg = Config.fromfile(current_dir_root + '/CBNetV2/configs/cbnet/' + config_file_root[1])

# Load weigths from pretrained model
cfg.load_from = current_dir_root + '/Weights/cascade_mask_rcnn_cbv2_swin_small.pth'

# Set up working dir to save files and logs.
cfg.work_dir = current_dir_root + '/CBNetV2/runs'

# Fixing Issue: " 'ConfigDict' object has no attribute 'device' "
cfg.device = 'cuda' 

# Modify num classes of the model in box head
cfg.model.roi_head.bbox_head[0].num_classes = 1
cfg.model.roi_head.bbox_head[1].num_classes = 1
cfg.model.roi_head.bbox_head[2].num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Modify dataset related settings
cfg.dataset_type = 'CocoDataset'
cfg.classes = current_dir_root + '/CBNetV2/runs/labelme2coco/labels.txt'
cfg.data_root = current_dir_root + '/CBNetV2/runs/labelme2coco'

cfg.data.test.type = 'CocoDataset'
cfg.data.test.classes = current_dir_root + '/CBNetV2/runs/labelme2coco/labels.txt'
cfg.data.test.data_root = current_dir_root + '/CBNetV2/runs/labelme2coco'
cfg.data.test.ann_file = current_dir_root + '/CBNetV2/runs/labelme2coco/val.json'
cfg.data.test.img_prefix = current_dir_root + '/Datasets/STAS_Train_Datasets'

cfg.data.train.type = 'CocoDataset'
cfg.data.train.classes = current_dir_root + '/CBNetV2/runs/labelme2coco/labels.txt'
cfg.data.train.data_root = current_dir_root + '/CBNetV2/runs/labelme2coco'
cfg.data.train.ann_file = current_dir_root + '/CBNetV2/runs/labelme2coco/train.json'
cfg.data.train.img_prefix = current_dir_root + '/Datasets/STAS_Train_Datasets'

cfg.data.val.type = 'CocoDataset'
cfg.data.val.classes = current_dir_root + '/CBNetV2/runs/labelme2coco/labels.txt'
cfg.data.val.data_root = current_dir_root + '/CBNetV2/runs/labelme2coco'
cfg.data.val.ann_file = current_dir_root + '/CBNetV2/runs/labelme2coco/val.json'
cfg.data.val.img_prefix = current_dir_root + '/Datasets/STAS_Train_Datasets'

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.interval = 10
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# Modify learning rate config
cfg.lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', 
    warmup_iters= 1000, 
    warmup_ratio= 1.0/10,
    min_lr=1e-07)

# Modify evaluation related settings
cfg.evaluation.interval = 10
cfg.evaluation.save_best = 'auto'

# Set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10

# Set the runner type
cfg.runner.type = 'EpochBasedRunner'

meta = dict()
meta['config'] = cfg.pretty_text


####### Train the model

total_training_epochs = 50
batch_size = 2

# Total training epochs
cfg.runner.max_epochs = total_training_epochs

# Batch size
cfg.data.samples_per_gpu = batch_size
cfg.data.workers_per_gpu = 1

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector "without loading checkpoints"
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Train model
train_detector(model, datasets, cfg, distributed=False, validate=True, meta=meta)



