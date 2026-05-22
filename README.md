<<<<<<< HEAD
<<<<<<< HEAD
# CDWNet

CDWNet is an open-source optical-SAR semantic segmentation repository for construction and demolition waste detection. This public package is a compact release prepared for manuscript submission and review.

## What Is Included

- source code under `segmentation/mmseg/`
- third-party backbone code under `segmentation/third_party/`
- one released model config under `segmentation/local_configs/multisource/CDWNet_config.py`
- training, evaluation, and inference tools under `segmentation/tools/`
- dataset tiling utility: `cut_multisource_tiles.py`
- environment and dependency files: `environment.yml`, `requirements.txt`


## Released Checkpoint

The released CDWNet checkpoint and pretrained backbone weights for OS-CDW are provided through the Baidu Netdisk link listed in [DATA_ACCESS.md](DATA_ACCESS.md):

- Matching config: `segmentation/local_configs/multisource/CDWNet_config.py`



## Installation

```bash
conda env create -f environment.yml
conda activate cdwnet
cd segmentation
pip install -e .
```

## Dependencies

Main tested stack:

- Python 3.10
- PyTorch 2.1.2
- TorchVision 0.16.2
- CUDA 12.1
- MMCV 1.7.2

See `environment.yml` and `requirements.txt` for the full environment.

## Compute Requirements

Recommended for manuscript-scale training:

- 1 NVIDIA GPU with at least 11 GB VRAM
- 32 GB RAM
- enough disk space for dataset tiles, logs, and checkpoints

## Dataset

The repository expects the tiled OS-CDW dataset at:

```text
segmentation/data/OS-CDW/
├── optical/
│   ├── training/
│   └── validation/
├── sar/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

Dataset distribution is described in [DATA_ACCESS.md](DATA_ACCESS.md).


## Released Config

- `segmentation/local_configs/multisource/CDWNet_config.py`

## Typical Use

### 1. Train

```bash
cd segmentation
PYTHONUNBUFFERED=1 python -u tools/train.py \
  local_configs/multisource/CDWNet_config.py \
  --work-dir work_dirs/CDWNet
```

If the dataset is not placed under `segmentation/data/OS-CDW`, override the data root from the command line:

```bash
cd segmentation
PYTHONUNBUFFERED=1 python -u tools/train.py \
  local_configs/multisource/CDWNet_config.py \
  --work-dir work_dirs/CDWNet \
  --cfg-options \
  data.train.data_root=/absolute/path/to/OS-CDW \
  data.val.data_root=/absolute/path/to/OS-CDW \
  data.test.data_root=/absolute/path/to/OS-CDW \
  data.workers_per_gpu=0 \
  data.samples_per_gpu=1
```

### 2. Evaluate

```bash
cd segmentation
python tools/eval_multisource_full.py \
  local_configs/multisource/CDWNet_config.py \
  /path/to/CDWNet_OSCDW_best.pth \
  --out-dir work_dirs/CDWNet_eval
```

### 3. Predict on paired TIFFs

```bash
cd segmentation
python tools/predict_multisource_tif.py \
  local_configs/multisource/CDWNet_config.py \
  /path/to/CDWNet_OSCDW_best.pth \
  --optical-dir /path/to/optical_tifs \
  --sar-dir /path/to/sar_tifs \
  --out-dir /path/to/output_tifs
```

## Reproducibility

- `train-data` is the tiled OS-CDW dataset used directly for training and validation.
- `original-data` is the uncropped OS-CDW source imagery used to generate tiles and full-scene examples.
- The released config reuses the original final-run normalization statistics so training behavior matches the released model setup.

## User Documentation

See [USER_GUIDE.md](USER_GUIDE.md).

## License

Apache 2.0. See `LICENSE`.
=======
# CDWNet
>>>>>>> 851ec9dd87b579bf7cdd16f646522a7d308fbdca
=======
# CDWNet
>>>>>>> 6c2c8329e449b6fbee645a569fc98d1d15062fec
