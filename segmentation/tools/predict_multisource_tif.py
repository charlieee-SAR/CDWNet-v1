#!/usr/bin/env python3
import argparse
import copy
import json
import os
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
import tifffile
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils import DictAction

# Ensure local project mmseg has highest priority in sys.path.
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def _patch_normalize_in_pipeline(pipeline, mean, std):
    for t in pipeline:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "Normalize":
            t["mean"] = list(mean)
            t["std"] = list(std)
            t["to_rgb"] = False
        if "transforms" in t and isinstance(t["transforms"], list):
            _patch_normalize_in_pipeline(t["transforms"], mean, std)


def _patch_img_scale_in_pipeline(pipeline, img_scale):
    for t in pipeline:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "MultiScaleFlipAug":
            t["img_scale"] = tuple(img_scale)
            if "transforms" in t and isinstance(t["transforms"], list):
                _patch_img_scale_in_pipeline(t["transforms"], img_scale)
        elif t.get("type") == "Resize":
            t["img_scale"] = tuple(img_scale)
        if "transforms" in t and isinstance(t["transforms"], list):
            _patch_img_scale_in_pipeline(t["transforms"], img_scale)


def _infer_expected_channels(cfg):
    m = cfg.model
    opt_c = int(m.get("optical_in_channels", cfg.get("optical_channels", 3)))
    sar_c = int(m.get("sar_in_channels", cfg.get("sar_channels", 2)))
    return opt_c, sar_c, opt_c + sar_c


def _load_auto_norm_stats(stats_path):
    if not stats_path or not osp.isfile(stats_path):
        return None, None
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mean = list(data.get("mean", []))
        std = list(data.get("std", []))
        if len(mean) > 0 and len(mean) == len(std):
            return mean, std
    except Exception:
        pass
    return None, None


def _fix_test_norm_channels(cfg, stats_path=None, identity_norm=False):
    opt_c, sar_c, total_c = _infer_expected_channels(cfg)
    cfg.optical_channels = opt_c
    cfg.sar_channels = sar_c

    if identity_norm:
        mean = [0.0] * total_c
        std = [1.0] * total_c
        cfg.img_norm_cfg = dict(mean=mean, std=std, to_rgb=False)
        if "data" in cfg and "test" in cfg.data and "pipeline" in cfg.data.test:
            _patch_normalize_in_pipeline(cfg.data.test.pipeline, mean, std)
        return

    mean = None
    std = None
    if "img_norm_cfg" in cfg:
        old_mean = list(cfg.img_norm_cfg.get("mean", []))
        old_std = list(cfg.img_norm_cfg.get("std", []))
        if len(old_mean) == total_c and len(old_std) == total_c:
            mean, std = old_mean, old_std
    if mean is None or std is None:
        stat_mean, stat_std = _load_auto_norm_stats(stats_path)
        if stat_mean is not None and len(stat_mean) == total_c:
            mean, std = stat_mean, stat_std

    if mean is None or std is None:
        mean = [0.0] * total_c
        std = [1.0] * total_c
    cfg.img_norm_cfg = dict(mean=mean, std=std, to_rgb=False)
    if "data" in cfg and "test" in cfg.data and "pipeline" in cfg.data.test:
        _patch_normalize_in_pipeline(cfg.data.test.pipeline, mean, std)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch predict for paired optical/SAR tif directories")
    parser.add_argument("config", help="config path")
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument("--optical-dir", required=True, help="optical tif dir")
    parser.add_argument("--sar-dir", required=True, help="sar tif dir")
    parser.add_argument("--out-dir", required=True, help="output tif dir")
    parser.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cpu")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="force output to binary map (value >0 -> 1)")
    parser.add_argument(
        "--suffix",
        default=".tif",
        help="input filename suffix, default .tif")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options")
    parser.add_argument(
        "--identity-norm",
        action="store_true",
        help="skip training-set normalization and use mean=0,std=1 for all channels")
    parser.add_argument(
        "--native-scale",
        action="store_true",
        help="patch test pipeline img_scale to each input image's original size")
    parser.add_argument(
        "--test-mode",
        choices=["config", "whole", "slide"],
        default="config",
        help="override model.test_cfg.mode for inference")
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="slide mode crop size")
    parser.add_argument(
        "--stride",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="slide mode stride")
    return parser.parse_args()


def stem(path):
    base = osp.basename(path)
    if "." not in base:
        return base
    return ".".join(base.split(".")[:-1])


def collect_pairs(optical_dir, sar_dir, suffix):
    optical_files = sorted(
        [f for f in os.listdir(optical_dir) if f.lower().endswith(suffix.lower())])
    sar_files = sorted(
        [f for f in os.listdir(sar_dir) if f.lower().endswith(suffix.lower())])
    opt_map = {stem(f): f for f in optical_files}
    sar_map = {stem(f): f for f in sar_files}
    common = sorted(set(opt_map.keys()) & set(sar_map.keys()))
    if not common:
        raise RuntimeError("No paired tif found by stem intersection.")
    missing_opt = sorted(set(sar_map.keys()) - set(opt_map.keys()))
    missing_sar = sorted(set(opt_map.keys()) - set(sar_map.keys()))
    return common, opt_map, sar_map, missing_opt, missing_sar


def get_geotiff_extratags(ref_path, out_shape_hw):
    geotiff_tag_codes = (33550, 33922, 34264, 34735, 34736, 34737)
    extratags = []

    with tifffile.TiffFile(ref_path) as tif:
        page = tif.pages[0]
        ref_shape = page.shape
        if len(ref_shape) >= 2:
            ref_h, ref_w = int(ref_shape[0]), int(ref_shape[1])
        else:
            ref_h, ref_w = out_shape_hw

        for code in geotiff_tag_codes:
            tag = page.tags.get(code)
            if tag is None:
                continue
            code_, dtype_, count_, value_, writeonce_ = tag.astuple()

            if code == 33550 and ref_h > 0 and ref_w > 0:
                scale = tag.value
                if isinstance(scale, tuple) and len(scale) >= 3:
                    sx, sy, sz = scale[:3]
                    out_h, out_w = out_shape_hw
                    sx = float(sx) * ref_w / max(out_w, 1)
                    sy = float(sy) * ref_h / max(out_h, 1)
                    value_ = (sx, sy, float(sz))
            elif code == 34264 and ref_h > 0 and ref_w > 0:
                transform = tag.value
                if isinstance(transform, tuple) and len(transform) == 16:
                    out_h, out_w = out_shape_hw
                    sx = ref_w / max(out_w, 1)
                    sy = ref_h / max(out_h, 1)
                    tf = list(transform)
                    tf[0] = float(tf[0]) * sx
                    tf[5] = float(tf[5]) * sy
                    value_ = tuple(tf)

            extratags.append((code_, dtype_, count_, value_, writeonce_))

    return extratags


def build_model(config_path, checkpoint_path, device, cfg_options=None, identity_norm=False):
    cfg = mmcv.Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    stats_path = osp.join(osp.dirname(osp.abspath(checkpoint_path)),
                          "auto_norm_stats.json")
    _fix_test_norm_channels(
        cfg, stats_path=stats_path, identity_norm=identity_norm)
    # Some custom segmentors do not accept top-level `pretrained` in ctor.
    if "pretrained" in cfg.model:
        cfg.model.pop("pretrained")
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = load_checkpoint(model, checkpoint_path, map_location="cpu")
    if "meta" in ckpt and "CLASSES" in ckpt["meta"]:
        model.CLASSES = ckpt["meta"]["CLASSES"]
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def _infer_default_crop_size(cfg):
    if "crop_size" in cfg:
        crop = cfg.crop_size
        if isinstance(crop, (list, tuple)) and len(crop) == 2:
            return int(crop[0]), int(crop[1])
    infer_input_size = cfg.model.get("infer_input_size", None)
    if isinstance(infer_input_size, (list, tuple)) and len(infer_input_size) == 2:
        return int(infer_input_size[0]), int(infer_input_size[1])
    return 256, 256


def override_test_cfg(model, args):
    if args.test_mode == "config":
        return

    if args.test_mode == "whole":
        test_cfg = dict(mode="whole")
    else:
        crop_h, crop_w = args.crop_size if args.crop_size else _infer_default_crop_size(model.cfg)
        if args.stride:
            stride_h, stride_w = args.stride
        else:
            stride_h = max(1, int(crop_h * 2 / 3))
            stride_w = max(1, int(crop_w * 2 / 3))
        test_cfg = dict(
            mode="slide",
            crop_size=(int(crop_h), int(crop_w)),
            stride=(int(stride_h), int(stride_w)))

    model.test_cfg = mmcv.ConfigDict(test_cfg)
    model.cfg.model.test_cfg = mmcv.ConfigDict(test_cfg)


def _get_tif_hw(path):
    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape
    if len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported tif shape: {shape}")


def inference_multisource(model, optical_path, sar_path, native_scale=False):
    cfg = model.cfg
    pipeline_cfg = copy.deepcopy(cfg.data.test.pipeline)
    if native_scale:
        h, w = _get_tif_hw(optical_path)
        _patch_img_scale_in_pipeline(pipeline_cfg, (w, h))
    pipeline = Compose(pipeline_cfg)
    data = dict(
        img_info=dict(filename=optical_path, sar_filename=sar_path),
        img_prefix=None,
        optical_prefix=None,
        sar_prefix=None,
        seg_prefix=None,
        seg_fields=[])
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [next(model.parameters()).device])[0]
    else:
        data["img_metas"] = [i.data[0] for i in data["img_metas"]]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    pred = result[0]
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    return pred.astype(np.uint8)


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    model = build_model(
        args.config,
        args.checkpoint,
        args.device,
        cfg_options=args.cfg_options,
        identity_norm=args.identity_norm)
    override_test_cfg(model, args)

    common, opt_map, sar_map, miss_opt, miss_sar = collect_pairs(
        args.optical_dir, args.sar_dir, args.suffix)
    print(f"[INFO] paired files: {len(common)}")
    if miss_opt:
        print(f"[WARN] missing optical for {len(miss_opt)} stems")
    if miss_sar:
        print(f"[WARN] missing sar for {len(miss_sar)} stems")

    for i, s in enumerate(common, 1):
        op = osp.join(args.optical_dir, opt_map[s])
        sp = osp.join(args.sar_dir, sar_map[s])
        pred = inference_multisource(model, op, sp, native_scale=args.native_scale)
        if args.binary:
            pred = (pred > 0).astype(np.uint8)
        out_path = osp.join(args.out_dir, f"{s}.tif")
        tifffile.imwrite(
            out_path,
            pred,
            extratags=get_geotiff_extratags(op, pred.shape[:2]))
        if i % 100 == 0 or i == len(common):
            print(f"[INFO] {i}/{len(common)} -> {out_path}")

    print(f"[DONE] all predictions saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
