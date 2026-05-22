#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

import numpy as np
from mmcv import Config
from mmcv.utils import DictAction

try:
    import tifffile
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False


def ensure_hwc(arr):
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        if arr.shape[0] <= 32 and arr.shape[0] < arr.shape[-1]:
            return np.transpose(arr, (1, 2, 0))
        return arr
    raise ValueError(f'Unsupported array shape: {arr.shape}')


def load_tif(path):
    if HAS_TIFFFILE:
        try:
            return ensure_hwc(tifffile.imread(str(path)))
        except Exception:
            pass
    if HAS_GDAL:
        ds = gdal.Open(str(path))
        if ds is None:
            raise RuntimeError(f'GDAL cannot open file: {path}')
        arr = ds.ReadAsArray()
        ds = None
        return ensure_hwc(arr)
    raise RuntimeError(
        f'Failed to read {path}. Please install tifffile+imagecodecs or gdal.')


def iter_files_with_suffix(folder, suffix):
    suffix = suffix.lower()
    for p in sorted(Path(folder).rglob(f'*{suffix}')):
        if p.is_file():
            yield p


def compute_mean_std_multisource(train_cfg):
    data_root = train_cfg.get('data_root', '')
    optical_dir = Path(data_root) / train_cfg['optical_dir']
    sar_dir = Path(data_root) / train_cfg['sar_dir']
    optical_suffix = train_cfg.get('optical_suffix', '.tif')
    sar_suffix = train_cfg.get('sar_suffix', '.tif')

    opt_files = list(iter_files_with_suffix(optical_dir, optical_suffix))
    sar_files = list(iter_files_with_suffix(sar_dir, sar_suffix))
    opt_map = {p.stem: p for p in opt_files}
    sar_map = {p.stem: p for p in sar_files}
    stems = sorted(set(opt_map) & set(sar_map))
    if not stems:
        raise RuntimeError(
            f'No paired files found under {optical_dir} and {sar_dir}')

    sum_opt = None
    sq_opt = None
    sum_sar = None
    sq_sar = None
    count = 0

    for stem in stems:
        o = load_tif(opt_map[stem]).astype(np.float64)
        s = load_tif(sar_map[stem]).astype(np.float64)
        if o.shape[:2] != s.shape[:2]:
            raise ValueError(
                f'Shape mismatch for stem={stem}: optical={o.shape}, sar={s.shape}')
        o2 = o.reshape(-1, o.shape[-1])
        s2 = s.reshape(-1, s.shape[-1])
        n = o2.shape[0]
        if s2.shape[0] != n:
            raise ValueError(f'Pixel count mismatch for stem={stem}')
        if sum_opt is None:
            sum_opt = np.zeros(o2.shape[1], dtype=np.float64)
            sq_opt = np.zeros(o2.shape[1], dtype=np.float64)
            sum_sar = np.zeros(s2.shape[1], dtype=np.float64)
            sq_sar = np.zeros(s2.shape[1], dtype=np.float64)
        sum_opt += o2.sum(axis=0)
        sq_opt += np.square(o2).sum(axis=0)
        sum_sar += s2.sum(axis=0)
        sq_sar += np.square(s2).sum(axis=0)
        count += n

    mean_opt = sum_opt / count
    var_opt = np.maximum(sq_opt / count - np.square(mean_opt), 1e-12)
    std_opt = np.sqrt(var_opt)

    mean_sar = sum_sar / count
    var_sar = np.maximum(sq_sar / count - np.square(mean_sar), 1e-12)
    std_sar = np.sqrt(var_sar)

    mean = np.concatenate([mean_opt, mean_sar]).tolist()
    std = np.concatenate([std_opt, std_sar]).tolist()
    return mean, std, len(stems), count


def compute_mean_std_single_source(train_cfg):
    data_root = train_cfg.get('data_root', '')
    img_dir = Path(data_root) / train_cfg['img_dir']
    img_suffix = train_cfg.get('img_suffix', '.tif')
    files = list(iter_files_with_suffix(img_dir, img_suffix))
    if not files:
        raise RuntimeError(f'No files found in {img_dir} with suffix {img_suffix}')

    ch_sum = None
    ch_sq = None
    count = 0
    for p in files:
        arr = load_tif(p).astype(np.float64)
        arr2 = arr.reshape(-1, arr.shape[-1])
        n = arr2.shape[0]
        if ch_sum is None:
            ch_sum = np.zeros(arr2.shape[1], dtype=np.float64)
            ch_sq = np.zeros(arr2.shape[1], dtype=np.float64)
        ch_sum += arr2.sum(axis=0)
        ch_sq += np.square(arr2).sum(axis=0)
        count += n

    mean = ch_sum / count
    var = np.maximum(ch_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean.tolist(), std.tolist(), len(files), count


def patch_pipeline_norm(pipeline, mean, std):
    if isinstance(pipeline, list):
        for t in pipeline:
            patch_pipeline_norm(t, mean, std)
    elif isinstance(pipeline, dict):
        if pipeline.get('type') == 'Normalize':
            pipeline['mean'] = mean
            pipeline['std'] = std
            pipeline['to_rgb'] = False
        for v in pipeline.values():
            if isinstance(v, (list, dict)):
                patch_pipeline_norm(v, mean, std)


def main():
    parser = argparse.ArgumentParser(
        description='Auto compute mean/std from training set and launch mmseg training.'
    )
    parser.add_argument('config', help='Config path')
    parser.add_argument('--work-dir', default=None, help='Work dir for training')
    parser.add_argument(
        '--stats-out',
        default=None,
        help='Path to save computed stats json (default: <work_dir>/auto_norm_stats.json)')
    parser.add_argument(
        '--train-script',
        default='tools/train.py',
        help='Training script to launch after config patch')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options before computing normalization statistics.')

    args, passthrough = parser.parse_known_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    os.makedirs(cfg.work_dir, exist_ok=True)

    train_cfg = cfg.data.train
    dataset_type = train_cfg.get('type', '')

    if dataset_type == 'MultiSourceTifDataset':
        mean, std, sample_num, pixel_num = compute_mean_std_multisource(train_cfg)
    else:
        mean, std, sample_num, pixel_num = compute_mean_std_single_source(train_cfg)

    # patch pipelines
    patch_pipeline_norm(cfg.data.train.pipeline, mean, std)
    patch_pipeline_norm(cfg.data.val.pipeline, mean, std)
    patch_pipeline_norm(cfg.data.test.pipeline, mean, std)

    # patch shared img_norm_cfg if exists
    if 'img_norm_cfg' in cfg:
        cfg.img_norm_cfg.mean = mean
        cfg.img_norm_cfg.std = std
        cfg.img_norm_cfg.to_rgb = False

    stats = {
        'dataset_type': dataset_type,
        'num_samples': int(sample_num),
        'num_pixels': int(pixel_num),
        'mean': [float(x) for x in mean],
        'std': [float(x) for x in std],
    }

    stats_out = args.stats_out or osp.join(cfg.work_dir, 'auto_norm_stats.json')
    with open(stats_out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    patched_cfg_path = osp.join(
        cfg.work_dir,
        f"auto_norm_{osp.basename(args.config)}")
    cfg.dump(patched_cfg_path)

    print('[AUTO_NORM] mean =', stats['mean'])
    print('[AUTO_NORM] std  =', stats['std'])
    print('[AUTO_NORM] stats saved to', stats_out)
    print('[AUTO_NORM] patched config saved to', patched_cfg_path)

    # launch original train script
    cmd = [sys.executable, args.train_script, patched_cfg_path]
    cmd.extend(passthrough)
    print('[AUTO_NORM] launch:', ' '.join(cmd))
    # Force subprocess to import mmseg from CURRENT project first,
    # avoiding stale editable installs from other folders.
    project_root = Path(__file__).resolve().parents[1]  # .../segmentation
    env = os.environ.copy()
    old_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (
        f"{str(project_root)}:{old_pythonpath}"
        if old_pythonpath else str(project_root)
    )
    print('[AUTO_NORM] PYTHONPATH=', env['PYTHONPATH'])

    subprocess.run(cmd, check=True, cwd=str(project_root), env=env)


if __name__ == '__main__':
    main()
