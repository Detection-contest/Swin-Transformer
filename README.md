# Detection-contest README.md
- 환경 설정  
 0. 요구사항
  - NVIDIA Geforce GTX 1080 Ti
  - cuda 10.2
  
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
  
 5. mmdetection폴더에서 setup.py로 환경 설정
 ```bash
 python setup.py develop
 ```

- 데이터 전처리


- 학습 수행(Swin-Transformer-Object-Detection폴더에서 수행)
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
예를 들어, 저희 모델을 GPU 1개인 장치에서 학습시키려면,
```

```

- 인퍼런스 수행
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```
