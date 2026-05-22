# User Guide

## Inputs

### Training

- the released config `segmentation/local_configs/multisource/CDWNet_config.py`
- tiled optical TIFF files
- tiled SAR TIFF files
- PNG annotation masks
- the OS-CDW folder structure expected by the configs

### Evaluation

- the same config used for training
- a trained checkpoint
- or the released checkpoint downloaded from the Baidu Netdisk package
- validation data in the expected tiled structure

### Inference

- the released config `segmentation/local_configs/multisource/CDWNet_config.py`
- a trained checkpoint
- or the released checkpoint downloaded from the Baidu Netdisk package
- one optical TIFF directory
- one SAR TIFF directory
- matching filenames between the two modalities

## Outputs

### Training

Written under `work_dirs/<experiment_name>/`:

- logs
- checkpoints
- optional curves and figures

### Evaluation

- `metrics_summary.json`
- `class_metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.npy`

### Inference

- predicted segmentation rasters in the chosen output directory

## Main Scripts

### `tools/train.py`

Use this to:

- start training
- reuse the fixed normalization statistics already stored in `CDWNet_config.py`

### `tools/eval_multisource_full.py`

Use this to:

- evaluate a checkpoint
- export confusion matrices
- export per-class and summary metrics

### `tools/predict_multisource_tif.py`

Use this to:

- run inference on paired optical-SAR TIFFs
- write prediction TIFFs
- directly test the released checkpoint on new paired scenes

### `tools/metrics_from_confusion.py`

Use this to:

- compute IoU, Dice, F-score, Precision, Recall, Specificity, NPV, and Kappa from an existing confusion matrix

## Expected Behavior

- optical, SAR, and annotation files should share the same stem
- channel counts in the config must match the actual TIFF files
- the released checkpoint should be read from the Baidu Netdisk `checkpoints` folder
- the backbone pretrained weights should be read from the Baidu Netdisk `pretrained` folder
- the released config already contains the normalization statistics from the original final training run
