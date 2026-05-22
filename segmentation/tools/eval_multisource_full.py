#!/usr/bin/env python3
import argparse
import csv
import json
import os
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import DictAction

# Ensure local project mmseg has highest priority in sys.path.
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes


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


def _fix_test_norm_channels(cfg, stats_path=None):
    opt_c, sar_c, total_c = _infer_expected_channels(cfg)

    # Keep dataset channel declarations consistent with model input channels.
    cfg.optical_channels = opt_c
    cfg.sar_channels = sar_c

    mean = None
    std = None
    if "img_norm_cfg" in cfg:
        old_mean = list(cfg.img_norm_cfg.get("mean", []))
        old_std = list(cfg.img_norm_cfg.get("std", []))
        if len(old_mean) == total_c and len(old_std) == total_c:
            mean, std = old_mean, old_std
    # If config mean/std is mismatched, try auto_norm_stats.json from run dir.
    if mean is None or std is None:
        stat_mean, stat_std = _load_auto_norm_stats(stats_path)
        if stat_mean is not None and len(stat_mean) == total_c:
            mean, std = stat_mean, stat_std

    if mean is None or std is None:
        mean = [0.0] * total_c
        std = [1.0] * total_c
    cfg.img_norm_cfg = dict(mean=mean, std=std, to_rgb=False)

    # Patch test/val pipelines in-memory only (no config file modifications).
    if "data" in cfg:
        if "test" in cfg.data and "pipeline" in cfg.data.test:
            _patch_normalize_in_pipeline(cfg.data.test.pipeline, mean, std)
        if "val" in cfg.data and "pipeline" in cfg.data.val:
            _patch_normalize_in_pipeline(cfg.data.val.pipeline, mean, std)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multisource checkpoint and export full metrics")
    parser.add_argument("config", help="config path")
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory for confusion matrix and metrics files")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--save-pred-pkl",
        action="store_true",
        help="save raw prediction list to predictions.pkl")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, e.g. data.test.ann_dir=...")
    return parser.parse_args()


def to_label_map(pred, num_classes):
    if isinstance(pred, (list, tuple)):
        if len(pred) == 0:
            return None
        pred = pred[0]
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if not isinstance(pred, np.ndarray):
        pred = np.asarray(pred)
    if pred.ndim == 3 and pred.shape[0] == num_classes:
        pred = pred.argmax(axis=0)
    elif pred.ndim == 3 and pred.shape[-1] == num_classes:
        pred = pred.argmax(axis=-1)
    elif pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    return pred.astype(np.int64)


def update_confusion(cm, gt, pred, num_classes, ignore_index=255):
    if pred.shape != gt.shape:
        if pred.size == gt.size:
            pred = pred.reshape(gt.shape)
        else:
            return
    gt = gt.astype(np.int64).reshape(-1)
    pred = pred.astype(np.int64).reshape(-1)
    valid = (
        (gt != ignore_index)
        & (gt >= 0)
        & (gt < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )
    if not np.any(valid):
        return
    ids = num_classes * gt[valid] + pred[valid]
    cm += np.bincount(ids, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes)


def metrics_from_confusion(cm):
    eps = 1e-10
    total = cm.sum()
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (tp + fp + fn)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    acc_cls = (tp + tn) / (total + eps)
    support = cm.sum(axis=1).astype(np.int64)

    oa = tp.sum() / (total + eps)
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / ((total + eps) ** 2)
    kappa = (oa - pe) / (1 - pe + eps)

    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "iou": float(np.mean(iou)),
        "dice": float(np.mean(dice)),
        "specificity": float(np.mean(specificity)),
        "acc_cls": float(np.mean(acc_cls)),
    }
    weights = support / (support.sum() + eps)
    weighted = {
        "precision": float(np.sum(weights * precision)),
        "recall": float(np.sum(weights * recall)),
        "f1": float(np.sum(weights * f1)),
        "iou": float(np.sum(weights * iou)),
        "dice": float(np.sum(weights * dice)),
    }

    per_class = []
    for i in range(cm.shape[0]):
        per_class.append({
            "class_index": i,
            "support": int(support[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "iou": float(iou[i]),
            "dice": float(dice[i]),
            "specificity": float(specificity[i]),
            "acc_cls": float(acc_cls[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "tn": int(tn[i]),
        })

    summary = {
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "macro_avg": macro,
        "weighted_avg": weighted,
    }
    return per_class, summary


def save_confusion_csv(cm, class_names, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + list(class_names))
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row.tolist())


def save_class_metrics_csv(per_class, class_names, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_name", "class_index", "support", "precision", "recall",
            "f1", "iou", "dice", "specificity", "acc_cls", "tp", "fp", "fn",
            "tn"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_class:
            r = dict(row)
            idx = r["class_index"]
            r["class_name"] = class_names[idx]
            w.writerow(r)


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    stats_path = osp.join(osp.dirname(osp.abspath(args.checkpoint)),
                          "auto_norm_stats.json")
    _fix_test_norm_channels(cfg, stats_path=stats_path)

    # Some custom segmentors (e.g. DualBranchEncoderDecoder variants) do not
    # accept top-level `pretrained` arg in __init__. Remove it if present.
    if "pretrained" in cfg.model:
        cfg.model.pop("pretrained")
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [args.gpu_id]

    dataset = build_dataset(cfg.data.test)
    loader_cfg = dict(
        num_gpus=1,
        dist=False,
        shuffle=False,
        samples_per_gpu=1,
        **cfg.data.get("test_dataloader", {}))
    if args.num_workers is not None:
        loader_cfg["workers_per_gpu"] = args.num_workers
        if args.num_workers == 0:
            loader_cfg["persistent_workers"] = False
    data_loader = build_dataloader(dataset, **loader_cfg)

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "meta" in checkpoint and "CLASSES" in checkpoint["meta"]:
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    model = revert_sync_batchnorm(model)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    model = MMDataParallel(model, device_ids=[args.gpu_id] if "cuda" in device else None)
    model.to(device)

    torch.cuda.empty_cache()
    results = single_gpu_test(
        model,
        data_loader,
        show=False,
        pre_eval=False,
        format_only=False)

    if args.save_pred_pkl:
        mmcv.dump(results, osp.join(args.out_dir, "predictions.pkl"))

    num_classes = len(dataset.CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for idx, pred in enumerate(results):
        gt = dataset.get_gt_seg_map_by_idx(idx)
        pred_map = to_label_map(pred, num_classes)
        if pred_map is None:
            continue
        update_confusion(cm, gt, pred_map, num_classes, ignore_index=255)

    np.save(osp.join(args.out_dir, "confusion_matrix.npy"), cm)
    save_confusion_csv(cm, dataset.CLASSES, osp.join(args.out_dir, "confusion_matrix.csv"))

    per_class, summary = metrics_from_confusion(cm)
    save_class_metrics_csv(
        per_class, dataset.CLASSES, osp.join(args.out_dir, "class_metrics.csv"))

    eval_metrics = {}
    for metric_name in (["mIoU"], ["mDice"], ["mFscore"], ["mIoU", "mDice", "mFscore"]):
        try:
            ret = dataset.evaluate(results, metric=metric_name)
            eval_metrics["+".join(metric_name)] = ret
        except Exception as e:
            eval_metrics["+".join(metric_name)] = {"error": str(e)}

    output = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "classes": list(dataset.CLASSES),
        "summary_from_confusion": summary,
        "eval_from_dataset": eval_metrics,
    }
    with open(osp.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved to: {args.out_dir}")
    print(" - confusion_matrix.npy")
    print(" - confusion_matrix.csv")
    print(" - class_metrics.csv")
    print(" - metrics_summary.json")
    if args.save_pred_pkl:
        print(" - predictions.pkl")


if __name__ == "__main__":
    main()
