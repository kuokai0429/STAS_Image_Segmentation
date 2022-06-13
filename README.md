# STAS_Image_Segmentation

## About

T-brain AI實戰吧 - 肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 II：運用影像分割作法於切割STAS輪廓 <br>

## Prerequisites

***TWCC Container:***
> - pytorch-21.06-py3:latest

## Installation

Step 0. Git clone the project folder.
```
git clone https://github.com/kuokai0429/STAS_Image_Segmentation.git
cd STAS_Image_Segmentation
```

Step 1. Create pipenv environment under current project folder and Install project dependencies.
```
pip3 install pipenv
pipenv --python 3.8
pipenv shell
pipenv install --skip-lock
```

Step 2. Clone and Install CBNetV2 under current project folder.
```
git clone https://github.com/VDIGPKU/CBNetV2.git
cd CBNetV2
python setup.py develop
```

Step 3. Install additional tools for mmcv and mmdetection.
```
cd ..
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

Step 4. Download Configurations, Pretrained Weights and Datasets from shared Google Drive.
> ***Source:*** <br> https://drive.google.com/file/d/1bGVrEcgjQf_aSbh8qBVh3fGkSV1qRKmL/view?usp=sharing
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bGVrEcgjQf_aSbh8qBVh3fGkSV1qRKmL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bGVrEcgjQf_aSbh8qBVh3fGkSV1qRKmL" -O STAS_SEG_Data.zip && rm -rf /tmp/cookies.txt
```

Step 5. Move /Configs, /Weights and /Datasets from /STAS_SEG_Data to current project folder.
```
rm -r Configs Weights Datasets
unzip STAS_SEG_Data.zip
mv STAS_SEG_Data/Configs .
mv STAS_SEG_Data/Weights .
mv STAS_SEG_Data/Datasets .
rmdir STAS_SEG_Data
```

Step 6. Modify the version limitation in ```__init__.py```
```
sed -i "s/mmcv_maximum_version = '1.4.0'/mmcv_maximum_version = '1.5.2'/g" CBNetV2/mmdet/__init__.py
```

## Inference

1. Draw COCO Object segmentations on demo.jpg from mmdetection.
```
python demo_coco.py --weight_file cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth
```

2. Draw STAS segmentations on Public and Private image Dataset.
```
python demo_stas_seg.py --weight_dir cascade_mask_rcnn_dual_swin_s_2_089_Weights --testdata_dirs Public_Image Private_Image/Image
```

## Training

```
python train_stas_seg.py --weight_dir cascade_mask_rcnn_dual_swin_s_2_089_Weights --checkpoint_file cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth --config_files cascade_mask_rcnn_swin_small.py cascade_mask_rcnn_cbv2_swin_small.py
```
