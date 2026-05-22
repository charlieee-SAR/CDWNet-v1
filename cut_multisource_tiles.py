#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tile a paired optical-SAR-label dataset into an MMSegmentation-style layout.

Features:
1. Supports arbitrary tile sizes.
2. Supports arbitrary overlap ratios.
3. Filters tiles by the fraction of positive pixels in the label.
4. Writes outputs in a standard MMSegmentation directory structure.

Expected input:
input_root/
  Optical/*.tif
  SAR/*.tif
  label/*.tif or *.png

Generated output:
output_root/
  optical/training/*.tif
  sar/training/*.tif
  annotations/training/*.png
  optical/validation/*.tif
  sar/validation/*.tif
  annotations/validation/*.png
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image

try:
    from osgeo import gdal
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False


def list_files_by_stem(folder: Path, suffixes: Sequence[str]) -> Dict[str, Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            out[p.stem] = p
    return out


def ensure_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        # (C,H,W) -> (H,W,C)
        if arr.shape[0] <= 32 and arr.shape[0] < arr.shape[-1]:
            return np.transpose(arr, (1, 2, 0))
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def load_tif(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(str(path))
        return ensure_hwc(arr)
    except Exception as e:
        # Common case: LZW-compressed TIFF without imagecodecs installed.
        if HAS_GDAL:
            ds = gdal.Open(str(path))
            if ds is None:
                raise RuntimeError(f"GDAL cannot open file: {path}") from e
            arr = ds.ReadAsArray()
            ds = None
            return ensure_hwc(arr)
        if "imagecodecs" in str(e).lower():
            raise RuntimeError(
                f"Failed to read: {path}\n"
                "This TIFF appears to be compressed and tifffile needs imagecodecs.\n"
                "Options:\n"
                "1) Install imagecodecs: pip install imagecodecs\n"
                "2) Install GDAL and use the fallback reader: conda install -c conda-forge gdal"
            ) from e
        raise


def load_label(path: Path) -> np.ndarray:
    if path.suffix.lower() in [".tif", ".tiff"]:
        try:
            arr = tifffile.imread(str(path))
        except Exception as e:
            if HAS_GDAL:
                ds = gdal.Open(str(path))
                if ds is None:
                    raise RuntimeError(f"GDAL cannot open label file: {path}") from e
                arr = ds.ReadAsArray()
                ds = None
            else:
                if "imagecodecs" in str(e).lower():
                    raise RuntimeError(
                        f"Failed to read label: {path}\n"
                        "This TIFF appears to be compressed and tifffile needs imagecodecs.\n"
                        "Options:\n"
                        "1) Install imagecodecs: pip install imagecodecs\n"
                        "2) Install GDAL and use the fallback reader: conda install -c conda-forge gdal"
                    ) from e
                raise
        if arr.ndim == 3:
            if arr.shape[0] <= 32 and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            arr = arr[..., 0]
        return arr
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def save_tif_hwc(arr: np.ndarray, path: Path) -> None:
    # tifffile expects HWC layout for straightforward writing.
    tifffile.imwrite(str(path), arr)


def save_label_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def make_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_one_pair(
    stem: str,
    optical: np.ndarray,
    sar: np.ndarray,
    label: np.ndarray,
    tile_size: int,
    stride: int,
    min_pos_ratio: float,
    positive_value: int = 1,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    h, w = label.shape[:2]
    if optical.shape[:2] != (h, w) or sar.shape[:2] != (h, w):
        raise ValueError(
            f"Shape mismatch: stem={stem}, optical={optical.shape}, sar={sar.shape}, label={label.shape}"
        )

    ys = make_starts(h, tile_size, stride)
    xs = make_starts(w, tile_size, stride)
    kept = []

    for y in ys:
        for x in xs:
            y2 = y + tile_size
            x2 = x + tile_size
            o_patch = optical[y:y2, x:x2, :]
            s_patch = sar[y:y2, x:x2, :]
            l_patch = label[y:y2, x:x2]

            # If the source image is smaller than the tile size, pad to a full tile.
            if o_patch.shape[0] != tile_size or o_patch.shape[1] != tile_size:
                pad_h = tile_size - o_patch.shape[0]
                pad_w = tile_size - o_patch.shape[1]
                o_patch = np.pad(o_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
                s_patch = np.pad(s_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
                l_patch = np.pad(l_patch, ((0, pad_h), (0, pad_w)), mode="constant")

            pos_ratio = float(np.mean(l_patch == positive_value))
            if pos_ratio < min_pos_ratio:
                continue

            patch_name = f"{stem}_{y}_{x}"
            kept.append((patch_name, o_patch, s_patch, l_patch))

    return kept


def split_train_val(items: List[Tuple], val_ratio: float, seed: int) -> Tuple[List[Tuple], List[Tuple]]:
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = int(round(len(items) * val_ratio))
    val_set = set(idx[:n_val])
    train, val = [], []
    for i, it in enumerate(items):
        (val if i in val_set else train).append(it)
    return train, val


def main():
    parser = argparse.ArgumentParser(
        description="Tile paired optical-SAR-label imagery and filter tiles by label coverage."
    )
    parser.add_argument("--input-root", type=str, required=True, help="Input root containing Optical/SAR/label.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root.")
    parser.add_argument("--optical-dirname", type=str, default="Optical")
    parser.add_argument("--sar-dirname", type=str, default="SAR")
    parser.add_argument("--label-dirname", type=str, default="label")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size.")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Tile overlap ratio in [0, 1).")
    parser.add_argument(
        "--min-pos-ratio",
        type=float,
        default=0.01,
        help="Minimum positive-pixel ratio required to keep a tile.",
    )
    parser.add_argument("--positive-value", type=int, default=1, help="Label value treated as foreground.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true", help="Remove the output directory before writing.")
    args = parser.parse_args()

    if not (0 <= args.overlap_ratio < 1):
        raise ValueError("--overlap-ratio must be in [0, 1).")
    if not (0 <= args.min_pos_ratio <= 1):
        raise ValueError("--min-pos-ratio must be in [0, 1].")
    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio must be in [0, 1).")

    stride = max(1, int(round(args.tile_size * (1 - args.overlap_ratio))))
    print(f"[INFO] tile_size={args.tile_size}, overlap_ratio={args.overlap_ratio}, stride={stride}")
    print(f"[INFO] min_pos_ratio={args.min_pos_ratio}, positive_value={args.positive_value}")

    input_root = Path(args.input_root)
    optical_dir = input_root / args.optical_dirname
    sar_dir = input_root / args.sar_dirname
    label_dir = input_root / args.label_dirname
    if not optical_dir.exists() or not sar_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Please check the input directories: {optical_dir}, {sar_dir}, {label_dir}")

    optical_map = list_files_by_stem(optical_dir, [".tif", ".tiff"])
    sar_map = list_files_by_stem(sar_dir, [".tif", ".tiff"])
    label_map = list_files_by_stem(label_dir, [".tif", ".tiff", ".png"])
    common_stems = sorted(set(optical_map) & set(sar_map) & set(label_map))
    if not common_stems:
        raise RuntimeError("No paired Optical/SAR/label files with matching stems were found.")
    print(f"[INFO] Paired scenes: {len(common_stems)}")

    output_root = Path(args.output_root)
    out_dirs = [
        output_root / "optical" / "training",
        output_root / "sar" / "training",
        output_root / "annotations" / "training",
        output_root / "optical" / "validation",
        output_root / "sar" / "validation",
        output_root / "annotations" / "validation",
    ]
    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    all_tiles = []
    for stem in common_stems:
        optical = load_tif(optical_map[stem])
        sar = load_tif(sar_map[stem])
        label = load_label(label_map[stem])
        tiles = tile_one_pair(
            stem=stem,
            optical=optical,
            sar=sar,
            label=label,
            tile_size=args.tile_size,
            stride=stride,
            min_pos_ratio=args.min_pos_ratio,
            positive_value=args.positive_value,
        )
        all_tiles.extend(tiles)

    if not all_tiles:
        raise RuntimeError("No tiles were kept after filtering. Reduce --min-pos-ratio or adjust the tiling setup.")
    print(f"[INFO] Kept tiles: {len(all_tiles)}")

    train_tiles, val_tiles = split_train_val(all_tiles, args.val_ratio, args.seed)
    print(f"[INFO] Training tiles: {len(train_tiles)}, validation tiles: {len(val_tiles)}")

    def dump(split_name: str, items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]):
        for name, o_patch, s_patch, l_patch in items:
            save_tif_hwc(o_patch, output_root / "optical" / split_name / f"{name}.tif")
            save_tif_hwc(s_patch, output_root / "sar" / split_name / f"{name}.tif")
            save_label_png(l_patch, output_root / "annotations" / split_name / f"{name}.png")

    dump("training", train_tiles)
    dump("validation", val_tiles)
    print("[INFO] Done")


if __name__ == "__main__":
    main()
