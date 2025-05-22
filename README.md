## Introduction
This repository implements Relevance-driven Input Dropout (RelDrop), a novel data augmentation technique that improves model generalization through selective occlusion of relevant input features. The implementation supports both 2D image classification and 3D point cloud classification tasks.

## Repository Structure
```
./
├── 2D_Images/              # 2D image classification implementation
├── 3D_Pointclouds/         # 3D point cloud classification implementation
├── data/                   # Recommended to save and load all the datasets from this folder
├── data/                   # Recommended to save and load all the pre-trained models from this folder
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation
1. Create and activate a virtual environment:
```
python -m venv reldrop_env
source reldrop_env/bin/activate  # Linux/Mac
# or
.\reldrop_env\Scripts\activate   # Windows
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Hardware Requirements
- GPU with minimum 16GB memory (tested on NVIDIA RTX 3090 and NVIDIA A100)
- CPU: Intel Xeon or equivalent
- RAM: 32GB minimum recommended

## Reproduce the experiments in the paper
The current config files already have the hyperparameters set to the best values of the respective experiments

### 2D Images Finetuning (Table-1, Blue columns, Page-6)
```
cd ./2D_Images/run_scripts
python cifar_imagenet.py --config_file training_config.json
```

### 2D Images Zero Shot (Table-1, Green columns, Page-8)
#### ImageNet-A/O
```
cd ./2D_Images/zero_shot_eval
python eval_imagenet_a_o.py --config_file eval_config.json 
```
#### ImageNet-R
```
cd ./2D_Images/zero_shot_eval
python eval_imagenet_r.py --config_file eval_config.json
```

### 3D Pointclouds (Table-2, page-8)
```
cd ./3D_Pointclouds
python train_classification.py --config_file ./config/training_config.yaml
```

## Dataset Downloads
### 2D Image Datasets
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-1k (2012)](https://www.image-net.org/download.php)
- [ImageNet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [ImageNet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)
- [ImageNet-O](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar)

### 3D Point Cloud Datasets
- [ModelNet40](https://modelnet.cs.princeton.edu/)
- [ShapeNet](https://www.kaggle.com/datasets/mitkir/shapenet/data)

## Pre-trained Models
- [ResNet Models (MIT License)](https://huggingface.co/edadaltocg)
- [ImageNet Pre-trained Models (Apache 2.0)](https://huggingface.co/docs/hub/en/timm)