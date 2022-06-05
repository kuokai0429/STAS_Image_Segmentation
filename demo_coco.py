''' Segmention COCO Objects on demo.jpg from CBNetV2
    Command: python demo_coco.py --weight_file cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth
'''

print("Running ....")
print("Ignore: 'apex is not installed'")

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import os
import argparse

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--weight_file', required=True, type=str, help="Pretrained weights to load. Please put under /Weights folder.")
args = parser.parse_args()

# Get current directory path
current_dir_root = os.getcwd()

# Load the config
config = current_dir_root + '/CBNetV2/configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py'
config = mmcv.Config.fromfile(config)
config.model.pretrained = None

# Setup a checkpoint file to load
# checkpoint = current_dir_root + '/Weights/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth'
checkpoint = current_dir_root + '/Weights/' + args.weight_file

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location='cuda:0')

# Set the models for inference
model.CLASSES = checkpoint['meta']['CLASSES']
model.cfg = config
model.to('cuda:0')

# Convert the model to evaluation mode
model.eval()

# Use the detector to do inference
img = current_dir_root + '/CBNetV2/demo/demo.jpg'
result = inference_detector(model, img)

# Parse the result
print("\n>>>>>>>> BndBox")
bndbox = []
for i in range(len(result[0])):
    for j in result[0][i]:
      if float(j[-1]) >= 0.05:
        temp = list(map(int, j[:-1]))
        temp.append(round(float(j[-1]), 5))
        print(temp)

# Save image with result
mmcv.mkdir_or_exist(current_dir_root + "/Output")
model.show_result(img, result, out_file=current_dir_root + "/Output/result_coco.jpg")

