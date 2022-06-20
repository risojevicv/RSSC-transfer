# Self-supervised models
PyTorch code for experiments with self-supervised models.

## Fine-tuning
To fine-tune a ResNet50 model on a single-label dataset use:

```
python fine_tune.py --path /path/to/dataset/ \
                    --data-split DATASET_SPLIT \ 
                    --lr MAXIMAL_LR \
                    --model swav|/path/to/model \
                    [--name MODEL_NAME]
                    [--batch-size BATCH_SIZE]
```

To fine-tune a ResNet50 model on a multi-label dataset use:

```
python fine_tune_multilabel.py --path /path/to/dataset/ \
                               --data-split DATASET_SPLIT \
                               --lr MAXIMAL_LR \
                               --model swav|/path/to/model \
                               [--name MODEL_NAME]
                               [--batch-size BATCH_SIZE]
```

Flags:
+ `--path`: path to the images.
+ `--data-split`: one of the splits in `data_splits` directory, e.g. `MLRSNet20`.
+ `--lr`: the learning rate is linearly increased up to this value and then decreased by a factor of 0.2 in the 50th, 70th, and 90th epochs.
+ `--model`: 
  - `swav`: a model pre-trained using SwAV is fine-tuned, 
  - `/path/to/model`: the specified model is fine-tuned.
+ `--name` (optional): the name of the fine-tuned model.
+ `--batch-size` (optional): the batch size (default: 100)

## Feature extraction
To run feature extraction on a single-label dataset use:

```
python extract_features.py --path /path/to/dataset/ \
                           --data-split DATASET_SPLIT \
                           --features /path/to/features/ \
                           --model swav|/path/to/model
                           [--batch-size BATCH_SIZE]
```                           

To run feature extraction on a multi-label dataset use:

```
python extract_features_multilabel.py --path /path/to/dataset/ \
                                      --data-split DATASET_SPLIT \
                                      --features /path/to/features/ \ 
                                      --model swav|/path/to/model
                                      [--batch-size BATCH_SIZE]
```                                      

Flags:
+ `--path`: path to the images.
+ `--data-split`: one of the splits in `data_splits` directory, e.g. `MLRSNet20`.
+ `--model`: 
  - `swav`: a model pre-trained on ImageNet-1k is used, 
  - `/path/to/model`: the specified model is used as a feature extractor.
+ `--batch-size` (optional): the batch size (default: 100)

## Linear classifier

To train a linear classifier on extracted features use `classify.py` or `classify_multilabel.py` from the home directory.
