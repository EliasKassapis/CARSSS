# Calibrated Adversarial Refinement for Multimodal Semantic Segmentation

PyTorch implementation of the Calibrated Adversarial Refinement models described in the paper "Calibrated Adversarial Refinement for Multimodal Semantic Segmentation".

## Getting Started
### Setup virtual environment
To install the requirements for this code run:
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

### Directory tree
```
.
├── data
│   └── datasets
│       ├── lidc
│       └── cityscapes
│ 
├── models
│   ├── discriminators
│   ├── general
│   ├── generators
│   │   └── calibration_nets
│   └── losses
│        
├── results
│        └── output
│        
├── testing
│        
├── training
│        
└── utils
```

## Datasets
###Download the LIDC dataset
The pre-processed 180x180 2D crops for the Lung Image Database Consortium (LIDC) image collection dataset 
([LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI))
, as described in 
[A Hierarchical Probabilistic U-Net for Modeling
Multi-Scale Ambiguities (2019)](https://arxiv.org/abs/1905.13077) and used in this work is made publicly available from Khol et. al, and can be downloaded from 
([here](https://console.cloud.google.com/storage/browser/hpunet-data/lidc_crops/)).

After downloading the dataset, please extract each file under *./data/datasets/lidc/*. This should give three folders under the said directory named: *lidc_crops_test*, *lidc_crops_train*, and *lidc_crops_test*.

###Download the pre-processed Cityscapes dataset with the black-box predictions
As described in our paper, we integrate our model on top of a black-box segmentation network. We used a pre-trained *DeepLabV3+(Xception65+ASPP)* model publically available 
[here](https://github.com/nyoki-mtl/pytorch-segmentation)
. We found that this model obtains a mIoU score of 0.79 on the official test-set of the Cityscapes dataset
([Cityscapes](https://www.cityscapes-dataset.com/)).

We have processed the official Cityscapes dataset, downscaling it to a spatial resolution of 256x512 and saved predictions from the black-box segmentation network for every image in the training and validation set. The pre-processed dataset can be downloaded from here
## Train your own models

## Evaluation
### Download pre-trained models

## License
The code in this repository is published under the [Apache License Version 2.0](LICENSE).