#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Render a side-by-side parity grid: PyTorch reference vs GGML backends, with
# predicted keypoints + scores overlaid on the query image.
#
# Reads benchmarks/accuracy.json (per-config kps_orig / scores) and the
# corresponding per-mode records produced by scripts/run_all_tests.py.
#
# Usage: python scripts/render_parity.py [--image 2007_007524.jpg]
# Output: benchmarks/parity_<image>.png
# ------------------------------------------------------------------------------
import argparse
import json
import os

import numpy as np  # noqa: E402

import run_all_tests as R_PROMPTS  # noqa: E402  (per-image official prompt sets)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GKD = os.path.dirname(HERE)
IMS = os.path.join(ROOT, "test_real_world", "ims1")

PYTORCH_REF = {  # stock PyTorch outputs on 2007_007524.jpg (full-image ROI)
    "text": {"kps": [111.76, 231.31, 153.34, 174.13, 90.96, 194.92,
                     179.33, 116.95, 90.96, 137.74],
             "scores": [0.9243, 0.9370, 0.9542, 0.8147, 0.8845],
             "labels": ["nose", "left eye", "right eye", "left ear", "right ear"]},
    "visual": {"kps": [153.34, 174.13, 90.96, 194.92, 111.76, 231.31],
               "scores": [0.9311, 0.9484, 0.4230],
               "labels": ["left eye", "right eye", "nose"]},
    "multimodal": {"kps": [153.34, 174.13, 90.96, 194.92, 111.76, 231.31],
                   "scores": [0.9347, 0.9554, 0.8796],
                   "labels": ["left eye", "right eye", "nose"]},
}


def draw_panel(ax, img, kps, scores, labels, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for i in range(len(scores)):
        x, y = kps[2 * i], kps[2 * i + 1]
        c = plt.cm.RdYlGn(min(max(scores[i], 0.0), 1.0))
        ax.scatter(x, y, s=140, facecolors="none", edgecolors=c, linewidths=2.5)
        ax.scatter(x, y, s=12, color=c)
        if labels:
            ax.annotate(f"{labels[i]} {scores[i]:.2f}", (x, y), fontsize=6,
                        textcoords="offset points", xytext=(4, -10), color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="2007_007524.jpg")
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", default=None,
                    help="comma-separated config tags to render; default: per backend "
                         "f32 + q4_K (readable subset - accuracy.json holds the full matrix)")
    ap.add_argument("--grid", action="store_true",
                    help="multi-image parity grid over the official test set "
                         "(reads accuracy_all_images.json); writes "
                         "benchmarks/parity_official_examples.png")
    ap.add_argument("--grid-images", default="cat_dog.jpg,alpaca_150.jpg,adeliepenguin_107.jpg,"
                                            "fish_swim.jpg,pigs_stock_farming.jpg,pet_birds.jpg")
    ap.add_argument("--grid-configs", default="cpu-q4_K,cuda-q4_K,vulkan-q4_K")
    args = ap.parse_args()

    if args.grid:
        render_grid(args)
        return

    acc_path = os.path.join(GKD, "benchmarks", "accuracy.json")
    with open(acc_path) as f:
        acc = json.load(f)

    tags = list(acc)
    if args.configs:
        want = set(args.configs.split(","))
        tags = [t for t in tags if t in want]
    else:
        picked = []
        for backend in ("cpu", "cuda", "vulkan"):
            for dtype in ("f32", "q4_K"):
                tag = f"{backend}-{dtype}"
                if tag in acc:
                    picked.append(tag)
        tags = picked

    img = Image.open(os.path.join(IMS, args.image)).convert("RGB")
    panels = [("PyTorch (reference)", None)]
    panels += [(tag, acc[tag]) for tag in tags]

    # one row per prompt mode
    modes = list(PYTORCH_REF)
    fig, axes = plt.subplots(len(modes), len(panels),
                             figsize=(4.1 * len(panels), 4.1 * len(modes)))
    if len(modes) == 1:
        axes = [axes]

    for mi, mode in enumerate(modes):
        ref = PYTORCH_REF[mode]
        for pi, (title, data) in enumerate(panels):
            ax = axes[mi][pi]
            if data is None:
                kps, scores = ref["kps"], ref["scores"]
            else:
                rec = data.get(mode)
                if rec is None:
                    ax.axis("off")
                    continue
                kps, scores = rec["kps_orig"], rec["scores"]
            t = f"{title}\n{mode}" if mi == 0 else mode
            draw_panel(ax, img, kps, scores, ref["labels"], t)

    out = args.out or os.path.join(GKD, "benchmarks", f"parity_{args.image.split('.')[0]}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def render_grid(args):
    """Rows = official test images, cols = [PyTorch ref, engine configs]; text
    mode with each image's own prompt set. Data: accuracy_all_images.json."""
    data_path = os.path.join(GKD, "benchmarks", "accuracy_all_images.json")
    with open(data_path) as f:
        A = json.load(f)
    images = [im for im in args.grid_images.split(",")
              if im in A["pytorch_ref"]["images"]]
    tags = [t for t in args.grid_configs.split(",") if t in A["configs"]]

    panels = [("PyTorch (reference)", None)] + [(t, t) for t in tags]
    fig, axes = plt.subplots(len(images), len(panels),
                             figsize=(3.6 * len(panels), 3.9 * len(images)))
    axes = np.atleast_2d(axes)
    for ri, im in enumerate(images):
        img = Image.open(os.path.join(IMS, im)).convert("RGB")
        ref = A["pytorch_ref"]["images"][im]
        labels = R_PROMPTS.PROMPTS.get(im, [])
        for ci, (title, tag) in enumerate(panels):
            ax = axes[ri][ci]
            if tag is None:
                kps, scores = ref["kps"], ref["scores"]
            else:
                rec = A["configs"][tag].get(im)
                kps, scores = rec["kps_orig"], rec["scores"]
            head = f"{title} · text\n{im}" if ri == 0 else im
            draw_panel(ax, img, kps, scores, labels, head)
    out = args.out or os.path.join(GKD, "benchmarks", "parity_official_examples.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
