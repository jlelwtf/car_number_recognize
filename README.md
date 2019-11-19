#README

###mask rcnn

trained model: https://github.com/jlelwtf/car_number_recognize/blob/master/models/data/mask-rcnn.pth

or install 
```
git-lfs 
```
and run into repo
``` 
git lfs fetch --all 
```

#### Install pytorch
for CUDA 10.0
```
pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```
for CUDA 9.2
```
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
for CPU only
```
pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### Datasets
From https://nomeroff.net.ua

downloading:
```
python download_datasets.py
```
