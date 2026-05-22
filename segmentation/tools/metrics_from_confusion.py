#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import os.path as osp
from typing import List, Tuple

import numpy as np


EPS = 1e-12

# ========== Inline confusion matrix mode ==========
# If set True, script uses class names + matrix below directly (no CSV reading).
USE_INLINE_CONFUSION = True

# Edit these values directly when USE_INLINE_CONFUSION=True.
INLINE_CLASS_NAMES = ["background", "garbage"]
INLINE_CONFUSION = np.array(
    [
        [71686716, 622466],
        [542157, 4481141],
    ],
    dtype=np.float64,
)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    valid = np.abs(b) > EPS
    out[valid] = a[valid] / b[valid]
    return out


def read_confusion_csv(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Invalid confusion csv format: {path}")

    pred_names = [x.strip() for x in rows[0][1:]]
    gt_names = []
    matrix = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        gt_names.append(r[0].strip())
        matrix.append([float(x) for x in r[1:]])

    cm = np.array(matrix, dtype=np.float64)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got {cm.shape}")

    # Prefer row class names if they match dimension.
    class_names = gt_names if len(gt_names) == cm.shape[0] else pred_names
    if len(class_names) != cm.shape[0]:
        class_names = [f"class_{i}" for i in range(cm.shape[0])]
    return class_names, cm


def kappa_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    if total <= EPS:
        return float("nan")
    po = np.trace(cm) / total
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (total * total)
    denom = 1.0 - pe
    if abs(denom) <= EPS:
        return float("nan")
    return float((po - pe) / denom)


def classwise_ovr_kappa(cm: np.ndarray) -> np.ndarray:
    n = cm.shape[0]
    out = np.full((n,), np.nan, dtype=np.float64)
    total = cm.sum()
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        bin_cm = np.array([[tp, fp], [fn, tn]], dtype=np.float64)
        out[i] = kappa_from_cm(bin_cm)
    return out


def compute_metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)  # gt
    col_sum = cm.sum(axis=0)  # pred
    total = cm.sum()

    fp = col_sum - tp
    fn = row_sum - tp
    tn = total - tp - fp - fn

    recall = safe_div(tp, tp + fn)          # same as class Acc in mmseg table
    precision = safe_div(tp, tp + fp)
    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    fscore = safe_div(2.0 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)

    aacc = float(np.trace(cm) / total) if total > EPS else float("nan")

    metrics = {
        "per_class": {
            "TP": tp.tolist(),
            "FP": fp.tolist(),
            "FN": fn.tolist(),
            "TN": tn.tolist(),
            "IoU": iou.tolist(),
            "Acc": recall.tolist(),
            "Dice": dice.tolist(),
            "Fscore": fscore.tolist(),
            "Precision": precision.tolist(),
            "Recall": recall.tolist(),
            "Specificity": specificity.tolist(),
            "NPV": npv.tolist(),
            "Kappa_ovr": classwise_ovr_kappa(cm).tolist(),
            "Support_GT": row_sum.tolist(),
            "Support_Pred": col_sum.tolist(),
        },
        "summary": {
            # Same naming as your eval script outputs
            "aAcc": aacc,
            "mIoU": float(np.nanmean(iou)),
            "mAcc": float(np.nanmean(recall)),
            "mDice": float(np.nanmean(dice)),
            "mFscore": float(np.nanmean(fscore)),
            "mPrecision": float(np.nanmean(precision)),
            "mRecall": float(np.nanmean(recall)),
            # Extra helpful global indicators
            "Kappa": kappa_from_cm(cm),
            "MacroSpecificity": float(np.nanmean(specificity)),
            "MacroNPV": float(np.nanmean(npv)),
        },
    }
    return metrics


def write_outputs(out_dir: str, class_names: List[str], cm: np.ndarray, metrics: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)

    np.save(osp.join(out_dir, "confusion_matrix.npy"), cm)

    with open(osp.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics["summary"], f, ensure_ascii=False, indent=2)

    # per-class csv
    per_class_path = osp.join(out_dir, "class_metrics.csv")
    fields = [
        "Class", "TP", "FP", "FN", "TN", "IoU", "Acc", "Dice", "Fscore",
        "Precision", "Recall", "Specificity", "NPV", "Kappa_ovr",
        "Support_GT", "Support_Pred"
    ]
    with open(per_class_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, name in enumerate(class_names):
            w.writerow([
                name,
                metrics["per_class"]["TP"][i],
                metrics["per_class"]["FP"][i],
                metrics["per_class"]["FN"][i],
                metrics["per_class"]["TN"][i],
                metrics["per_class"]["IoU"][i],
                metrics["per_class"]["Acc"][i],
                metrics["per_class"]["Dice"][i],
                metrics["per_class"]["Fscore"][i],
                metrics["per_class"]["Precision"][i],
                metrics["per_class"]["Recall"][i],
                metrics["per_class"]["Specificity"][i],
                metrics["per_class"]["NPV"][i],
                metrics["per_class"]["Kappa_ovr"][i],
                metrics["per_class"]["Support_GT"][i],
                metrics["per_class"]["Support_Pred"][i],
            ])


def pretty_print(class_names: List[str], metrics: dict) -> None:
    print("\nPer-class metrics:")
    header = (
        "Class, IoU, Acc, Dice, Fscore, Precision, Recall, "
        "Specificity, NPV, Kappa_ovr"
    )
    print(header)
    for i, name in enumerate(class_names):
        print(
            f"{name}, "
            f"{metrics['per_class']['IoU'][i]:.6f}, "
            f"{metrics['per_class']['Acc'][i]:.6f}, "
            f"{metrics['per_class']['Dice'][i]:.6f}, "
            f"{metrics['per_class']['Fscore'][i]:.6f}, "
            f"{metrics['per_class']['Precision'][i]:.6f}, "
            f"{metrics['per_class']['Recall'][i]:.6f}, "
            f"{metrics['per_class']['Specificity'][i]:.6f}, "
            f"{metrics['per_class']['NPV'][i]:.6f}, "
            f"{metrics['per_class']['Kappa_ovr'][i]:.6f}"
        )

    print("\nSummary metrics:")
    for k, v in metrics["summary"].items():
        print(f"{k}: {v:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute segmentation metrics from a confusion matrix CSV.")
    parser.add_argument(
        "--confusion-csv",
        default=None,
        help="Path to confusion matrix csv, e.g. gt\\pred,background,garbage...")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. If set, writes summary and class csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    if USE_INLINE_CONFUSION:
        class_names = list(INLINE_CLASS_NAMES)
        cm = np.array(INLINE_CONFUSION, dtype=np.float64)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"INLINE_CONFUSION must be square, got {cm.shape}")
        if len(class_names) != cm.shape[0]:
            raise ValueError(
                f"INLINE_CLASS_NAMES length={len(class_names)} "
                f"must equal matrix size={cm.shape[0]}"
            )
    else:
        if not args.confusion_csv:
            raise ValueError("--confusion-csv is required when USE_INLINE_CONFUSION=False")
        class_names, cm = read_confusion_csv(args.confusion_csv)

    metrics = compute_metrics(cm)
    pretty_print(class_names, metrics)
    if args.out_dir:
        write_outputs(args.out_dir, class_names, cm, metrics)
        print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
