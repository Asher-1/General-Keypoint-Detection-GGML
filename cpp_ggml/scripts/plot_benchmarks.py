#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Generate benchmark charts + tables from benchmarks/*.jsonl (all discovered
# configs, no cherry-picking):
#   - latency_matrix.png      : config x mode latency heatmap (log scale)
#   - speedup_matrix.png      : config x mode speedup heatmap vs PyTorch CUDA
#   - latency_by_config.png   : mean end-to-end latency per config, one panel
#                               per prompt mode
#   - latency_per_image.png   : per-image latency, text mode
#   - stage_breakdown.png     : vision / text / detect stage split (f32, text)
#   - speedup_table.md        : markdown latency + speedup table
#
# Usage: python scripts/plot_benchmarks.py
# ------------------------------------------------------------------------------
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "benchmarks")
MODES = ["text", "visual", "multimodal"]

BACKEND_ORDER = ["cpu", "cuda", "vulkan", "pytorch"]
DTYPE_ORDER = ["f32", "f16", "q8_0", "q4_0", "q4_K", "cuda"]

BACKEND_COLORS = {"cpu": "#4c72b0", "cuda": "#55a868", "vulkan": "#c44e52",
                  "pytorch": "#8172b2"}


def config_sort_key(tag):
    parts = tag.rsplit("-", 1)
    backend = parts[0] if len(parts) == 2 else tag
    dtype = parts[1] if len(parts) == 2 else ""
    return (BACKEND_ORDER.index(backend) if backend in BACKEND_ORDER else 99,
            DTYPE_ORDER.index(dtype) if dtype in DTYPE_ORDER else 99)


def load_all():
    recs = []
    for f in sorted(os.listdir(BENCH)):
        if f.startswith("latency_") and f.endswith(".jsonl"):
            tag = f[len("latency_"):-len(".jsonl")]
            with open(os.path.join(BENCH, f)) as fh:
                for line in fh:
                    r = json.loads(line)
                    r["config"] = tag
                    recs.append(r)
    py = os.path.join(BENCH, "pytorch.jsonl")
    if os.path.exists(py):
        with open(py) as fh:
            for line in fh:
                r = json.loads(line)
                r["config"] = "pytorch-cuda"
                recs.append(r)
    return recs


def main():
    recs = load_all()
    if not recs:
        print("no benchmark records found in", BENCH)
        return
    configs = []
    for r in recs:
        if r["config"] not in configs:
            configs.append(r["config"])
    configs.sort(key=config_sort_key)

    mean = defaultdict(dict)
    for r in recs:
        mean[r["mode"]].setdefault(r["config"], []).append(r["total_ms"])
    # pytorch-cuda is a config inside each mode dict, not a mode
    py_mean = {m: np.mean(v["pytorch-cuda"]) for m, v in mean.items() if "pytorch-cuda" in v}

    # ---------- latency matrix heatmap ----------
    mat = np.full((len(configs), len(MODES)), np.nan)
    for i, c in enumerate(configs):
        for j, mode in enumerate(MODES):
            if mode in mean and c in mean[mode]:
                mat[i, j] = np.mean(mean[mode][c])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(configs) + 2.2))
    finite = mat[np.isfinite(mat)]
    norm = LogNorm(vmin=max(finite.min(), 1.0), vmax=finite.max())
    im = ax.imshow(mat, cmap="viridis", norm=norm, aspect="auto")
    for i in range(len(configs)):
        for j in range(len(MODES)):
            v = mat[i, j]
            if np.isnan(v):
                txt, color = "-", "#777777"
            elif v >= 1000:
                txt, color = f"{v / 1000:.2f} s", "white"
            else:
                txt, color = f"{v:.1f} ms", "white" if norm(v) > 0.35 else "#111111"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_title("GKDT-L end-to-end latency matrix (15 official images, mean)",
                 fontsize=11, pad=28)
    fig.colorbar(im, ax=ax, label="ms (log scale)", shrink=0.8)
    fig.tight_layout()
    p = os.path.join(BENCH, "latency_matrix.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)

    # ---------- speedup matrix heatmap ----------
    if py_mean:
        spd = np.full_like(mat, np.nan)
        for j, mode in enumerate(MODES):
            if mode in py_mean:
                spd[:, j] = py_mean[mode] / mat[:, j]
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(configs) + 2.2))
        vmax = np.nanmax(spd)
        norm = TwoSlopeNorm(vmin=min(np.nanmin(spd), 0.5), vcenter=1.0, vmax=max(vmax, 1.5))
        im = ax.imshow(spd, cmap="RdYlGn", norm=norm, aspect="auto")
        for i in range(len(configs)):
            for j in range(len(MODES)):
                v = spd[i, j]
                txt = "-" if np.isnan(v) else f"{v:.1f}x"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if (np.isfinite(v) and (v > vmax * 0.72 or v < 0.62)) else "#111111")
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels(MODES, fontsize=10)
        ax.xaxis.set_ticks_position("top")
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_title("speedup vs stock PyTorch CUDA (green = faster)", fontsize=11, pad=28)
        fig.colorbar(im, ax=ax, label="speedup (x)", shrink=0.8)
        fig.tight_layout()
        p = os.path.join(BENCH, "speedup_matrix.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print("wrote", p)

    # ---------- grouped bars, one panel per mode ----------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, mode in zip(axes, MODES):
        vals = [np.mean(mean[mode].get(c, [np.nan])) for c in configs]
        cols = [BACKEND_COLORS.get(c.rsplit("-", 1)[0], "#999999") for c in configs]
        bars = ax.bar(range(len(configs)), vals, color=cols)
        for x, v in zip(range(len(configs)), vals):
            if np.isnan(v):
                continue
            label = f"{v / 1000:.2f}s" if v >= 1000 else f"{v:.0f}"
            ax.text(x, v * 1.02, label, ha="center", fontsize=7)
        if py_mean and mode in py_mean:
            ax.axhline(py_mean[mode], color=BACKEND_COLORS["pytorch"], ls="--", lw=1.2)
            ax.text(len(configs) - 0.4, py_mean[mode] * 1.05, "pytorch-cuda",
                    color=BACKEND_COLORS["pytorch"], fontsize=8, ha="right")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("latency (ms)")
        ax.set_title(f"{mode} prompts", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS[b]) for b in BACKEND_ORDER]
    axes[0].legend(handles, BACKEND_ORDER, fontsize=8)
    fig.suptitle("GKDT-L inference latency by config (15 official images)", y=0.995)
    fig.tight_layout()
    p = os.path.join(BENCH, "latency_by_config.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)

    # ---------- per-image latency (text mode) ----------
    fig, ax = plt.subplots(figsize=(13, 5.5))
    images = sorted({r["input"].split("/")[-1] for r in recs if r["mode"] == "text"})
    width = 0.8 / max(len(configs), 1)
    for i, c in enumerate(configs):
        per = {}
        for r in recs:
            if r["config"] == c and r["mode"] == "text":
                per[r["input"].split("/")[-1]] = r["total_ms"]
        vals = [per.get(im, np.nan) for im in images]
        ax.bar(np.arange(len(images)) + (i - len(configs) / 2) * width, vals, width,
               label=c, color=BACKEND_COLORS.get(c.rsplit("-", 1)[0], "#999999"),
               alpha=0.45 + 0.55 * (i % 5) / 4)
    ax.set_xticks(np.arange(len(images)))
    ax.set_xticklabels(images, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.set_title("per-image latency, text prompts (log scale)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(BENCH, "latency_per_image.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)

    # ---------- stage breakdown (f32, text) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = [("preprocess_ms", "#cccccc"), ("vision_ms", "#4c72b0"),
              ("text_ms", "#dd8452"), ("detect_ms", "#55a868"),
              ("decode_ms", "#c44e52")]
    labels = []
    sel = [c for c in configs if c.endswith("-f32")]
    for i, c in enumerate(sel):
        rs = [r for r in recs if r["config"] == c and r["mode"] == "text"]
        vals = [np.mean([r[k] for r in rs]) if rs else 0.0 for k, _ in stages]
        for j, (k, col) in enumerate(stages):
            ax.bar(i, vals[j], 0.6, bottom=sum(vals[:j]), color=col,
                   label=k[:-3] if i == 0 else None)
        labels.append(c)
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("ms")
    ax.set_yscale("log")
    ax.set_title("stage breakdown, text prompts (f32, log scale)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(BENCH, "stage_breakdown.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p)

    # ---------- accuracy over all official images (accuracy_all_images.json) ----------
    acc_all = os.path.join(BENCH, "accuracy_all_images.json")
    if os.path.exists(acc_all):
        with open(acc_all) as f:
            A = json.load(f)
        cfg_tags = [c for c in configs if c in A.get("configs", {})]
        images = sorted(A.get("pytorch_ref", {}).get("images", {}))
        if cfg_tags and images:
            mat = np.full((len(cfg_tags), len(images)), np.nan)
            for i, c in enumerate(cfg_tags):
                for j, im in enumerate(images):
                    if im in A["configs"][c]:
                        mat[i, j] = A["configs"][c][im]["mean_err_px"]
            vmax = np.nanmax(mat)
            fig, ax = plt.subplots(figsize=(1.05 * len(images) + 3.2,
                                            0.42 * len(cfg_tags) + 2.0))
            norm = LogNorm(vmin=max(np.nanmin(mat), 1e-4), vmax=max(vmax, 1e-3))
            im2 = ax.imshow(mat, cmap="viridis_r", norm=norm, aspect="auto")
            for i in range(len(cfg_tags)):
                for j in range(len(images)):
                    v = mat[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.3f}" if v < 1 else f"{v:.1f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if norm(v) < 0.6 else "#111111")
            ax.set_xticks(range(len(images)))
            ax.set_xticklabels(images, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(cfg_tags)))
            ax.set_yticklabels(cfg_tags, fontsize=9)
            ax.set_title("mean keypoint error vs stock PyTorch (px, log scale;\n"
                         "all 15 official images, text prompts)", fontsize=10)
            fig.colorbar(im2, ax=ax, label="px (log)", shrink=0.8)
            fig.tight_layout()
            p = os.path.join(BENCH, "accuracy_by_image.png")
            fig.savefig(p, dpi=140)
            plt.close(fig)
            print("wrote", p)

    # ---------- speedup table ----------
    lines = ["# Speedup table", "",
             "End-to-end single-image latency (ms), mean over the 15 official",
             "test images x warmup/iters protocol in `run_all_tests.py`.",
             "Reference: stock PyTorch GKDT-L on CUDA (RTX 4090).", ""]
    head = [c for c in configs if c != "pytorch-cuda"]
    lines += ["| mode | " + " | ".join(head) + " | pytorch-cuda (ref) |",
              "|---|" + "---|" * (len(head) + 1)]
    for mode in MODES:
        cells = []
        for c in head:
            v = np.mean(mean[mode].get(c, [np.nan]))
            ref = py_mean.get(mode, np.nan)
            sp = f" ({ref / v:.1f}x)" if np.isfinite(v) and np.isfinite(ref) else ""
            cells.append(f"{v:.1f}{sp}" if np.isfinite(v) else "-")
        ref_v = py_mean.get(mode, np.nan)
        lines.append(f"| {mode} | " + " | ".join(cells) +
                     f" | {ref_v:.1f} |" if np.isfinite(ref_v) else
                     f"| {mode} | " + " | ".join(cells) + " | - |")
    p = os.path.join(BENCH, "speedup_table.md")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", p)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
