Car numberplate recognition


Mask RCNN model
===============
Download trained model: https://github.com/jlelwtf/car_number_recognize/blob/master/models/data/mask-rcnn.pth

Or install git-lfs and run into repo:
```bash
git lfs fetch --all
```

Install libraries
=================
### Install pytorch
to install for CUDA 10.1
```bash
pip install torch torchvision
```

to install for CUDA 10.0
```bash
pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```
to install for CUDA 9.2
```bash
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
to install for CPU only
```bash
pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Install other libs

```bash
pip instal -r requirements.txt
```

Datasets
========
From https://nomeroff.net.ua

downloading:
```
python download_datasets.py
```
