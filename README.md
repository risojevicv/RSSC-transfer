# RSSC-transfer
Repository with the source code and pre-trained models for the paper [Do we still need ImageNet pre-training in remote sensing scene classification?](https://arxiv.org/abs/2111.03690).

## Datasets

+ MLRSNet: https://github.com/cugbrs/MLRSNet
+ NWPU-RESISC45: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
+ AID: https://captain-whu.github.io/AID/
+ PatternNet: https://sites.google.com/view/zhouwx/dataset
+ RSI-CB: https://github.com/lehaifeng/RSI-CB
+ UCM: http://weegee.vision.ucmerced.edu/datasets/landuse.html

Dataset splits can be downloaded at https://drive.google.com/drive/folders/1utNOUsQP3bWd-36jwVz9gQ-QZRQKhfvq?usp=sharing. The filenames contain the dataset name and proportion of training images. For example, `MLRSNet80-split.pkl` is the MLRSNet split with 80% training and 20% test images.

## Models
The pre-trained models can be downloaded at: https://drive.google.com/drive/folders/1kMpZEPKs7S8XlNMKK-A7fe8_O_SzlMi3?usp=sharing

Models in `scratch` are trained from scratch on HRRS datasets, models in `imagenet` are pre-trained on ImageNet-1k and fine-tuned on HRRS datasets, and in `ssl` are the original SwAV model as well as the models fine-tuned on HRRS datasets. 

## Installation
Download the data splits from https://drive.google.com/drive/folders/1utNOUsQP3bWd-36jwVz9gQ-QZRQKhfvq?usp=sharing and put them into `data_splits` directory.

Download the pre-trained models from https://drive.google.com/drive/folders/1kMpZEPKs7S8XlNMKK-A7fe8_O_SzlMi3?usp=sharing and put them into `models` directory.

## Training/Fine-tuning
To train/fine-tune a ResNet50 model on a single-label dataset use:

```
python train.py --path /path/to/dataset/ \
                --data-split DATASET_SPLIT \
                 --name MODEL_NAME 
                 --lr MAXIMAL_LR 
                 [--model imagenet|/path/to/model]
                 [--batch-size BATCH_SIZE]
```

To train/fine-tune a ResNet50 model on a multi-label dataset use:

```
python train_multilabel.py --path /path/to/dataset/ \
                           --data-split DATASET_SPLIT \
                           --name MODEL_NAME \
                           --lr MAXIMAL_LR \
                           [--model imagenet|/path/to/model]
                           [--batch-size BATCH_SIZE]
```

Flags:
+ `--path`: path to the images.
+ `--data-split`: one of the splits in `data_splits` directory, e.g. `MLRSNet20`.
+ `--name`: the name of the trained model.
+ `--lr`: the learning rate is linearly increased up to this value and then decreased by a factor of 0.2 in the 50th, 70th, and 90th epochs.
+ `--model` (optional): 
  - absent: the model is trained from scratch, 
  - `imagenet`: a model pre-trained on ImageNet-1k is fine-tuned, 
  - `/path/to/model`: the specified model is fine-tuned.
+ `--batch-size` (optional): the batch size (default: 100)

## Feature extraction
To run feature extraction on a single-label dataset use:

```
python extract_features.py --path /path/to/dataset/ \
                           --data-split DATASET_SPLIT \
                           --features /path/to/features/ \
                           --model imagenet|/path/to/model
```                           

To run feature extraction on a multi-label dataset use:

```
python extract_features_multilabel.py --path /path/to/dataset/ \
                                      --data-split DATASET_SPLIT \
                                      --features /path/to/features/ \ 
                                      --model imagenet|/path/to/model
```                                      

Flags:
+ `--path`: path to the images.
+ `--data-split`: one of the splits in `data_splits` directory, e.g. `MLRSNet20`.
+ `--model`: 
  - `imagenet`: a model pre-trained on ImageNet-1k is used, 
  - `/path/to/model`: the specified model is used as a feature extractor.
+ `--batch-size` (optional): the batch size (default: 100)

## Linear classifier

To train a linear classifier on extracted features use:

`python classify.py /path/to/features/`

for single-label tasks, or

`python classify_multilabel.py /path/to/features/`

for multi-label tasks.

# Citation

If you find this repository useful in your research, please cite:

```
@Article{isprs-archives-XLIII-B3-2022-1399-2022,
AUTHOR = {Risojevi\'c, V. and Stojni\'c, V.},
TITLE = {DO WE STILL NEED IMAGENET PRE-TRAINING IN REMOTE SENSING SCENE CLASSIFICATION?},
JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {XLIII-B3-2022},
YEAR = {2022},
PAGES = {1399--1406},
URL = {https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B3-2022/1399/2022/},
DOI = {10.5194/isprs-archives-XLIII-B3-2022-1399-2022}
}
```