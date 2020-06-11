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

### Download the LIDC dataset

The pre-processed 180x180 2D crops for the Lung Image Database Consortium (LIDC) image collection dataset 
([LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI))
, as described in 
[A Hierarchical Probabilistic U-Net for Modeling
Multi-Scale Ambiguities (2019)](https://arxiv.org/abs/1905.13077) and used in this work is made publicly available from Khol et. al, and can be downloaded from 
([here](https://console.cloud.google.com/storage/browser/hpunet-data/lidc_crops/)).

After downloading the dataset, extract each file under `./data/datasets/lidc/`. This should give three folders under the said directory named: `lidc_crops_test`, `lidc_crops_train`, and `lidc_crops_test`.

Please note that the official repository of the 
[Hierarchical Probabilistic U-Net](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_probabilistic_unet)
, the version of the dataset linked above containts 8843 images for training, 1993 for validation and 1980 for testing rather than 8882, 1996 and 1992 images as used in our experiments, however, the score remains the same.

### Download the pre-processed Cityscapes dataset with the black-box predictions

As described in our paper, we integrate our model on top of a black-box segmentation network. We used a pre-trained *DeepLabV3+(Xception65+ASPP)* model publically available 
[here](https://github.com/nyoki-mtl/pytorch-segmentation)
. We found that this model obtains a mIoU score of 0.79 on the official test-set of the Cityscapes dataset
([Cityscapes](https://www.cityscapes-dataset.com/)).

To download the official Cityscapes dataset:

1) Visit the Cityscapes [website](https://www.cityscapes-dataset.com/) and create an account
2) Download the [images](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1) 
3) Extract the files and move the folders *gtFine* and *leftImg8bit* in a new directory for the raw data i.e. `./data/datasets/cityscapes/raw_data`.
4) Create the 19-class labels by following [this issue](https://github.com/junfu1115/DANet/issues/12).
5) Configure your data directories in `./data/datasets/cityscapes/preprocessing_config.py` . 
6) Run `./data/datasets/cityscapes/preprocessing.py` to pre-process the data in downscaled numpy arrays and save under `./data/datasets/cityscapes/processed`.

Subsequently [download](https://drive.google.com/file/d/1EkJD1PUe7J5f5oc_VvUj-7a7XTT-I-Gc/view?usp=sharing) the black-box predictions under `./data/datasets/cityscapes/`, and extract by running:

```
tar -zxvf cityscapes_bb_preds.tar.gz
```
Finally, move the black-box predictions in the processed 

## Train your own models

To train you own calibrated adversarial refinement (CAR) model on the LIDC dataset, run:

```
python main.py --mode train --calibration_net SegNetCalNet --zdim 8 --dataset LIDC --class_flip False
```


To train you own CAR model using the black-box predictions on the modified Cityscapes dataset, run:

```
python main.py --mode train --calibration_net ToyCalNet --zdim 32 --dataset CITYSCAPES19 --class_flip True
```

Launching a run in train mode will create a new directory with the date and time of the start of your run under `./results/output/`, where plots documenting the progress of the training and are saved and models are checkpointed. For example, a run launched on 12:00:00 on 1/1/2020 will create a new folder
 `./results/output/2020-01-01_12:00:00/` .


## Evaluation

To evaluate a pre-trained model or your own trained model 

### Download pre-trained models
A pre-trained CAR model on LIDC can be downloaded from
[here](https://drive.google.com/file/d/1FxsvJjcRt3CsXmokQ4L8cfPVwu3teiTn/view?usp=sharing).


A pre-trained CAR model on the modified Cityscapes dataset can be downloaded from
[here](https://drive.google.com/file/d/1MJzZbByAU7MjNUH1TCuOoA4fwNvK1XF9/view?usp=sharing).


## License
The code in this repository is published under the [Apache License Version 2.0](LICENSE).