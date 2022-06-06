# STAS_Image_Segmentation

## About:

T-brain AI實戰吧 - 肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 II：運用影像分割作法於切割STAS輪廓 <br>

## Environment (TWCC Container): 

pytorch-21.06-py3:latest <br>

## Command: 

```
git clone https://github.com/kuokai0429/STAS_Image_Segmentation.git
cd STAS_Image_Segmentation

pip3 install pipenv
pipenv --python 3.8
pipenv shell
pipenv install --skip-lock

git clone https://github.com/VDIGPKU/CBNetV2.git
cd CBNetV2
python setup.py develop

cd ..
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bGVrEcgjQf_aSbh8qBVh3fGkSV1qRKmL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bGVrEcgjQf_aSbh8qBVh3fGkSV1qRKmL" -O STAS_SEG_Data.zip && rm -rf /tmp/cookies.txt

rm -r Configs Weights Datasets
unzip STAS_SEG_Data.zip
mv STAS_SEG_Data/Configs .
mv STAS_SEG_Data/Weights .
mv STAS_SEG_Data/Datasets .
rmdir STAS_SEG_Data

sed -i "s/mmcv_maximum_version = '1.4.0'/mmcv_maximum_version = '1.5.2'/g" CBNetV2/mmdet/__init__.py

python demo_coco.py --weight_file cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth
python demo_stas_seg.py --weight_dir cascade_mask_rcnn_dual_swin_s_2_089_Weights --testdata_dirs Public_Image Private_Image/Image
python train_stas_seg.py --weight_dir cascade_mask_rcnn_dual_swin_s_2_089_Weights --checkpoint_file cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth --config_files cascade_mask_rcnn_swin_small.py cascade_mask_rcnn_cbv2_swin_small.py
```
