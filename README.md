# Detection-contest README.md
- 환경 설정  
  0. 요구사항
  <ul>NVIDIA Geforce GTX 1080 Ti</ul>
  <ul>cuda 10.2</ul>
  
  1. 가상환경 만들기
  ```bash
  conda create -n 가상환경명 python=3.7 -y
  ```

  2. PyTorch 라이브러리 설치(가상환경 활성화 한 후에 수행)
  ```bash
  conda activate 가상환경명
  conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
  ```

  3. Swin-Transformer-Object-Detection폴더에서 mmcv-full 라이브러리 다운로드(mmcv-full==1.4.0, mmdet==
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
  
- 학습 수행(Swin-Transformer-Object-Detection폴더에서 수행)
```bash
cd Swin-Transformer-Object-Detection
```
  <ul>Single GPU</ul>
  python tools/train.py 
- 인퍼런스 수행
