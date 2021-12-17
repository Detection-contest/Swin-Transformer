# Detection-contest README.md
- 환경 설정  
 0. 요구사항
  - NVIDIA Geforce GTX 1080 Ti
  - cuda 10.1
  
 1. 가상환경 만들기
 ```bash
 conda create -n 가상환경명 python=3.7 -y
 ```

 2. PyTorch 라이브러리 설치(가상환경 활성화 한 후에 수행)
 ```bash
 conda activate 가상환경명
 conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
 ```

 3. Swin-Transformer-Object-Detection폴더에서 mmcv-full 라이브러리 다운로드
 ```bash
 cd Swin-Transformer-Object-Detection
 python -m pip install mmcv_full-1.4.0-cp37-cp37m-manylinux1_x86_64.whl
 ```
  
 4. mmdetection git clone후 필요한 라이브러리들 설치
 ```bash
 git clone https://github.com/open-mmlab/mmdetection.git
 cd mmdetection
 pip install -r requirements/build.txt
 pip install -v -e .
 pip install timm
 pip uninstall pycocotools
 pip install mmpycocotools
 pip install pandas
 pip install tqdm
 ```
  
 5. Swin-Transformer-Object-Detection폴더에서 setup.py로 환경 설정
 ```bash
 cd ../Swin-Transformer-Object-Detection
 python setup.py develop
 ```
</br>
- 데이터 전처리  

data 폴더에 데이터를 업로드한 후 아래의 예시와 같이 수행

  python data2voc.py
  cd Swin-Transformer-Object-Detection/tools/dataset_converters
  python pascal_voc.py

</br>
- 데이터 후처리

인퍼런스 후에 아래의 예시와 같이 수행

1. 테스트 결과를 xml 파일로 저장

  python coco2voc.py


2. 시각화 이미지 저장

  python visualize.py


</br>
- 학습 수행  

Swin-Transformer-Object-Detection폴더에서 수행, Pretrained모델은 아래에서 다운로드 받아서 수행함, Swin-T-IN1K모델을 다운로드 받은 후 아래 예시와 같이 수행  


1. Pretrained models on ImageNet-1K ([Swin-T-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth), [Swin-S-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth), [Swin-B-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)) and ImageNet-22K ([Swin-B-IN22K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth), [Swin-L-IN22K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)) are provided.
2. The supported code and models for ImageNet-1K image classification, COCO object detection and ADE20K semantic segmentation are provided.
3. The cuda kernel implementation for the [local relation layer](https://arxiv.org/pdf/1904.11491.pdf) is provided in branch [LR-Net](https://github.com/microsoft/Swin-Transformer/tree/LR-Net).  


```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
예를 들어, 저희 모델을 GPU 1개인 장치에서 학습시키려면(편의를 위해 pretrained된 모델 파일을 Swin-Transformer-Object-Detection폴더 내에 넣어서 수행),
```
cd Swin-Transformer-Object-Detection
python tools/train.py work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py --cfg-options model.pretrained=swin_tiny_patch4_window7_224.pth
```
</br>
- 인퍼런스 수행  

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```  

예를 들어, 저희 모델을 GPU 1개인 장치에서 인퍼런스를 수행하라면,  

```
python tools/test.py work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/latest.pth --eval bbox
```
