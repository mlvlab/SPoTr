# Self-positioning Point-based Transformer for Point Cloud Understanding

Official pytorch implementation of "Self-positioning Point-based Transformer for Point Cloud Understanding" (CVPR 2023).

![Figure](./assets/main_figure.png)

> Jinyoung Park <sup> 1* </sup>, Sanghyeok Lee<sup> 1* </sup>, Sihyeon Kim <sup> 1 </sup>, Yunyang Xiong <sup> 2 </sup>, Hyunwoo J. Kim <sup> 1† </sup>  
> <sup> 1 </sup> Korea University, <sup> 2 </sup> Meta Reality Labs

## Setup
- Clone repository 
```
git clone https://github.com/mlvlab/SPoTr.git
cd SPoTr
```
- Install packages with a setup file
```
bash install.sh
```
- Dataset
```
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
tar -xvf s3disfull.tar
```

## Run Experiments
- Train
```
CUDA_VISIBLE_DEVICES='0' python examples/segmentation/main.py  --cfg cfgs/s3dis/spotr.yaml
```
- Inference
```
CUDA_VISIBLE_DEVICES='0' python examples/segmentation/main.py  --cfg cfgs/s3dis/spotr.yaml  mode=test --pretrained_path ckpt/S3DIS/ckpt_best.pth
```


## Acknowledgement
This repo is built upon OpenPoints.
```
https://github.com/guochengqian/openpoints
```


