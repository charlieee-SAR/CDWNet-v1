# CDWNet

CDWNet is an optical-SAR semantic segmentation repository for construction and demolition waste detection.

## Installation

```bash
conda env create -f environment.yml
conda activate cdwnet
cd segmentation
pip install -e .
```

## Dependencies

- Python 3.10
- PyTorch 2.1.2
- TorchVision 0.16.2
- CUDA 12.1
- MMCV 1.7.2

See `environment.yml` and `requirements.txt` for the full environment.

## Dataset Layout

Place the tiled dataset under `segmentation/data/OS-CDW/`:

```text
segmentation/data/OS-CDW/
├── optical/
│   ├── training/
│   └── test/
├── sar/
│   ├── training/
│   └── test/
└── annotations/
    ├── training/
    └── test/
```

The training command automatically splits `training` into a training subset and an internal validation subset. The `test` folder is used only for standalone evaluation.

Dataset access is described in [DATA_ACCESS.md](DATA_ACCESS.md).

## Config

- `segmentation/local_configs/multisource/CDWNet_config.py`

## Train

```bash
cd segmentation
PYTHONUNBUFFERED=1 python -u tools/train.py \
  local_configs/multisource/CDWNet_config.py \
  --work-dir work_dirs/CDWNet
```

If the dataset is not placed under `segmentation/data/OS-CDW`, override the data root:

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

## Evaluate

```bash
cd segmentation
python tools/eval_multisource_full.py \
  local_configs/multisource/CDWNet_config.py \
  /path/to/checkpoint.pth \
  --out-dir work_dirs/CDWNet_eval
```

## Predict

```bash
cd segmentation
python tools/predict_multisource_tif.py \
  local_configs/multisource/CDWNet_config.py \
  /path/to/checkpoint.pth \
  --optical-dir /path/to/optical_tifs \
  --sar-dir /path/to/sar_tifs \
  --out-dir /path/to/output_tifs
```

## License

Apache 2.0. See `LICENSE`.
