### ASIAF-Net
****
Official PyTorch implementation for paper __Adaptive Short-Temporal Induced Aware Fusion Network for Predicting Attention Regions like a Driver__.

ASIAF-Net is a novel driver attention prediction framework that performs self-adaptive short-temporal feature extraction. And for object-level driver attention prediction, we designed an extra object saliency estimation branch to find objects which should be focused. 

![Model2](https://user-images.githubusercontent.com/63184678/158103047-2589ae94-98ff-4f0b-989c-986c827aedd4.jpg)

## Installation
This code was developed with Python 3.6 on Ubuntu 16.04. The main Python requirements:

pytorch==1.4 <br />
opencv-python <br />
[apex](https://github.com/NVIDIA/apex) <br />

## Datasets
1. Download and extract [DADA-2000 Dataset](https://github.com/JWFangit/LOTVS-DADA).
2. Download the [Ground Truth of Object Saliency Estimation](https://pan.baidu.com/s/1NLm0caqW4O5dK2z6SxM2nA?pwd=enmq) (password:enmq) and extract it.
3. Put all the samples of the DADA-2000 datasets in the same folder, as follows
```
DIR
|- json_file
|    └─train_file
|    └─val_file
|    └─test_file
|- DADA_dataset
|    └─1
|      └─001
|         └─fixation
|         └─images
|         └─maps
|      |
|      └─027
|         └─fixation
|         └─images
|         └─maps
|    |
|    └─54
|      └─001
|         └─fixation
|         └─images
|         └─maps
|      |
|      └─020
|         └─fixation
|         └─images
|         └─maps
|- OSE


```
## News 🎉
  March 2022 Update - the ground truth of object salieny detection 
  March 2022 - Release the part of code
  
## Code
Coming Soon after publication.
