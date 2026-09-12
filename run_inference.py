#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-click end-to-end INFERENCE of both implementations with rendered output:

    python official PyTorch  (test_real_world demo path, needs torch)
    C++ ggml engine          (cpp_ggml gkd-cli, no Python at runtime)

For every prompt mode (text / visual / multimodal) both sides run the FULL
pipeline - loading, preprocessing, the model, decoding - and their keypoints
are rendered onto the image with the SAME drawing code, plus a side-by-side
comparison chart and machine-readable JSON.

Everything is derived from this script's location (no hardcoded absolute
paths); every dependency is auto-detected and a missing one SKIPS its side
with an actionable message instead of failing the run.

Examples:
    python3 run_inference.py                        # official demo image, all modes
    python3 run_inference.py --image my.jpg --bbox 10 20 300 400
    python3 run_inference.py --backend vulkan --dtype q4_K --modes text
    python3 run_inference.py --skip-python          # C++-only deployment box
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
GKD = os.path.join(ROOT, "cpp_ggml")
IMS = os.path.join(ROOT, "test_real_world", "ims1")
DEMO_IMAGE = os.path.join(IMS, "2007_007524.jpg")
SUPPORT_IMAGE = os.path.join(IMS, "2007_003778.jpg")
SUPPORT_KPS = [343.0, 166.0, 281.0, 158.0, 311.0, 197.0]  # official demo's 1-shot
BACKENDS = ["cuda", "vulkan", "cpu"]
DTYPES = ["f32", "f16", "q8_0", "q4_K"]

CAT_SKELETON = [1, 2, 1, 3, 2, 3, 2, 4, 3, 5]  # official demo skeleton links


def log(msg):
    print(msg, flush=True)


def find_gkd_cli(backend):
    order = [backend] if backend else BACKENDS
    for b in order:
        p = os.path.join(GKD, f"build-{b}", "bin", "gkd-cli")
        if os.path.exists(p):
            return p, b
    return None, None


def find_torch_python():
    """An interpreter whose torch can load the GKDT checkpoint (CUDA torch)."""
    cands = [sys.executable]
    import glob
    for base in ("/home/*/anaconda3", "/home/*/miniconda3", "/home/*/miniforge3",
                 "/opt/conda"):
        cands += sorted(glob.glob(os.path.join(base, "envs", "*", "bin", "python")))
    for py in cands:
        try:
            r = subprocess.run(
                [py, "-c", "import torch; assert torch.cuda.is_available()"],
                capture_output=True, timeout=120)
            if r.returncode == 0:
                return py
        except Exception:
            pass
    return None


def draw_keypoints(image, kps, scores, skeleton, title):
    """Identical drawing for both sides - a fair visual comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * image.size[1] / max(image.size[0], 1)))
    ax.imshow(image)
    ax.set_xlim(0, image.size[0]); ax.set_ylim(image.size[1], 0)
    ax.axis("off")
    cmap = plt.get_cmap("tab10")
    for i, ((x, y), s) in enumerate(zip(kps, scores)):
        c = cmap(i % 10)
        ax.scatter([x], [y], s=40 + 220 * min(max(s, 0), 1), color=c,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(f"{s:.2f}", (x, y), fontsize=6, color=c, zorder=4,
                    xytext=(3, 3), textcoords="offset points")
    for a, b in zip(skeleton[0::2], skeleton[1::2]):
        if a - 1 < len(kps) and b - 1 < len(kps):
            ax.plot([kps[a - 1][0], kps[b - 1][0]], [kps[a - 1][1], kps[b - 1][1]],
                    color="yellow", lw=1.2, alpha=0.75, zorder=2)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def run_python(args, image, modes, out_dir):
    """Official PyTorch end-to-end. Returns {mode: (kps, scores)} or None."""
    try:
        sys.path.insert(0, ROOT)
        import torch  # noqa: E402
        from test_real_world.gkd_inference_lib.gkd_inference import (  # noqa: E402
            GKDInference, demo)
    except Exception as e:  # torch missing / CPU-only torch
        log(f"[python] SKIP - no usable CUDA torch here: {type(e).__name__}: {e}")
        log("[python] run this script with the torch env's interpreter for this side")
        return None
    torch.set_grad_enabled(False)
    try:
        infer = GKDInference(cfg_file=os.path.join(ROOT, "test_real_world", "configs",
                                                   "gkd.yaml"),
                             checkpoint_path=args.checkpoint)
        infer.gkd_model.eval()
    except torch.cuda.OutOfMemoryError:
        log("[python] SKIP - CUDA OOM while loading the official model "
            "(another process holds most of the GPU); rerun when the GPU is "
            "free, or pass --skip-python on a busy box")
        return None

    out = {}
    for mode in modes:
        texts = list(args.kps_texts)
        support_im = ""
        support_kps = []
        if mode in ("visual", "multimodal"):
            support_im = args.support_im
            support_kps = args.support_kps
        if mode == "visual":
            texts = []  # pure visual prompt: the official demo passes no texts
        if mode == "multimodal":
            n_v = len(support_kps) // 2
            if len(texts) != n_v:  # official assert: N_t must equal N_v
                log(f"[python] multimodal: trimming {len(texts)} texts to the "
                    f"{n_v} support keypoints (official N_t == N_v rule)")
                texts = texts[:n_v]
        try:
            preds, scores, _ = demo(infer, image, list(args.bbox), support_im,
                                    support_kps, texts)
        except torch.cuda.OutOfMemoryError:
            log(f"[python] {mode} SKIP - CUDA OOM (GPU busy); the C++ engine "
                "needs only ~1 GB and already completed")
            return out if out else None
        out[mode] = {"kps": preds[0].tolist(), "scores": scores[0].tolist()}
        log(f"[python] {mode}: " +
            ", ".join(f"({x:.1f},{y:.1f})={s:.2f}"
                      for (x, y), s in zip(out[mode]["kps"], out[mode]["scores"])))
    with open(os.path.join(out_dir, "python_official.json"), "w") as f:
        json.dump({"image": image, "modes": out}, f, indent=2)
    return out


def run_cpp(args, image, modes, out_dir):
    """Pure C++ ggml engine end-to-end (gkd-cli). Returns {mode: (...)}."""
    cli, backend = find_gkd_cli(args.backend)
    if not cli:
        log("[cpp] SKIP - no gkd-cli build found; build one via "
            "'cd cpp_ggml && cmake --preset cpu && cmake --build --preset cpu'")
        return None
    model = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{args.dtype}.gguf")
    if not os.path.exists(model):
        log(f"[cpp] SKIP - model missing: {model} (run_e2e.py models stage or "
            "huggingface-cli download Asher-1/GKD_GGUF)")
        return None
    log(f"[cpp] engine: {cli} ({backend}) | model: {os.path.basename(model)}")
    out = {}
    for mode in modes:
        cmd = [cli, "detect", "--model", model, "--input", image,
               "--threads", str(args.threads), "--out",
               os.path.join(out_dir, f"cpp_{mode}_rendered.jpg")]
        if args.bbox:
            for v in args.bbox:
                cmd += ["--bbox", repr(float(v))]
        texts = list(args.kps_texts)
        if mode == "multimodal":
            # official rule N_t == N_v: trim to the support-keypoint count
            n_v = len(args.support_kps) // 2
            if len(texts) != n_v:
                log(f"[cpp] multimodal: trimming {len(texts)} texts to {n_v} "
                    "(official N_t == N_v rule)")
                texts = texts[:n_v]
        if mode == "text":
            for t in texts:
                cmd += ["--kps-texts", t]
        if mode in ("visual", "multimodal"):
            cmd += ["--support-image", args.support_im]
            for v in args.support_kps:
                cmd += ["--support-kps", repr(float(v))]
            for t in ([] if mode == "visual" else texts):
                cmd += ["--kps-texts", t]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            log(f"[cpp] {mode} FAIL:\n{r.stderr[-1200:]}")
            continue
        j = json.loads(next(l for l in r.stdout.splitlines()
                            if l.startswith('{"json":')))["json"]
        n = j["n_prompts"]
        scale, offx, offy = j["trans"]
        kps = [[(j["kps_norm"][2 * i] / 2 + 0.5) * 384 / scale + offx / scale,
                (j["kps_norm"][2 * i + 1] / 2 + 0.5) * 384 / scale + offy / scale]
               for i in range(n)]
        out[mode] = {"kps": kps, "scores": j["scores"]}
        log(f"[cpp] {mode}: " +
            ", ".join(f"({x:.1f},{y:.1f})={s:.2f}"
                      for (x, y), s in zip(kps, j["scores"])))
    with open(os.path.join(out_dir, "cpp_ggml.json"), "w") as f:
        json.dump({"image": image, "backend": backend,
                   "dtype": args.dtype, "modes": out}, f, indent=2)
    return out


def compare(args, image, modes, py, cpp, out_dir):
    """Side-by-side chart + numeric deltas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(image).convert("RGB")
    fig, axes = plt.subplots(len(modes), 2, figsize=(11, 5.4 * len(modes)),
                             squeeze=False)
    for ri, mode in enumerate(modes):
        panels = [("official PyTorch", py and py.get(mode)),
                  (f"C++ ggml ({args.backend or 'auto'}/{args.dtype})",
                   cpp and cpp.get(mode))]
        for ci, (title, res) in enumerate(panels):
            ax = axes[ri][ci]
            ax.imshow(img)
            ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)
            ax.axis("off")
            if res:
                cmap = plt.get_cmap("tab10")
                for i, ((x, y), s) in enumerate(zip(res["kps"], res["scores"])):
                    ax.scatter([x], [y], s=40 + 220 * min(max(s, 0), 1),
                               color=cmap(i % 10), edgecolors="white",
                               linewidths=1.2, zorder=3)
                    ax.annotate(f"{s:.2f}", (x, y), fontsize=6, color=cmap(i % 10),
                                xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"{mode} prompts - {title}"
                         + ("" if res else "\n(SKIPPED)"), fontsize=9)
    fig.suptitle("End-to-end inference: official PyTorch vs C++ ggml engine",
                 y=0.998)
    fig.tight_layout()
    p = os.path.join(out_dir, "python_vs_cpp.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    log(f"[compare] wrote {p}")

    if py and cpp:
        import numpy as np
        for mode in modes:
            if mode in py and mode in cpp:
                d = np.abs(np.asarray(py[mode]["kps"], np.float64) -
                           np.asarray(cpp[mode]["kps"], np.float64))
                ds = np.abs(np.asarray(py[mode]["scores"], np.float64) -
                            np.asarray(cpp[mode]["scores"], np.float64))
                log(f"[compare] {mode}: max coord delta = {d.max():.4f} px, "
                    f"max score delta = {ds.max():.5f}")


def main():
    global ROOT, GKD, IMS, DEMO_IMAGE, SUPPORT_IMAGE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEMO_IMAGE, help="query image")
    ap.add_argument("--modes", default="all",
                    choices=["all", "text", "visual", "multimodal"])
    ap.add_argument("--bbox", nargs="*", type=float, default=[],
                    help="ROI box(es) x1 y1 x2 y2 (whole image when omitted)")
    ap.add_argument("--kps-texts", nargs="*", default=["nose", "left eye",
                                                       "right eye", "left ear",
                                                       "right ear"],
                    help="text prompts (official cat demo set by default)")
    ap.add_argument("--support-im", default=SUPPORT_IMAGE,
                    help="1-shot visual-prompt image")
    ap.add_argument("--support-kps", nargs="*", type=float, default=SUPPORT_KPS,
                    help="support keypoints x1 y1 ...")
    ap.add_argument("--skeleton", nargs="*", type=int, default=CAT_SKELETON,
                    help="1-based skeleton links drawn on both sides")
    ap.add_argument("--backend", default="", choices=[""] + BACKENDS,
                    help="gkd-cli build (default: fastest available)")
    ap.add_argument("--dtype", default="q4_K", choices=DTYPES, help="GGUF dtype")
    ap.add_argument("--checkpoint", default="",
                    help="official PyTorch checkpoint (python side; default: "
                         "cpp_ggml/models/pytorch/gkd_fullset.best)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--out-dir", default="", help="output directory "
                    "(default: output/inference_<image>_)")
    ap.add_argument("--skip-python", action="store_true", help="C++ side only")
    ap.add_argument("--skip-cpp", action="store_true", help="Python side only")
    ap.add_argument("--root", default=HERE, help="repository root (auto-derived)")
    args = ap.parse_args()

    ROOT = os.path.abspath(args.root)
    GKD = os.path.join(ROOT, "cpp_ggml")
    IMS = os.path.join(ROOT, "test_real_world", "ims1")
    if not args.checkpoint:
        args.checkpoint = os.path.join(GKD, "models", "pytorch", "gkd_fullset.best")

    image = os.path.abspath(args.image)
    if not os.path.isfile(image):
        sys.exit(f"error: image not found: {image} (official test images live in "
                 f"test_real_world/ims1 or download from github.com/Asher-1/"
                 f"cloudViewer_downloads/releases/tag/general_keypoint_detection_data)")
    stem = os.path.splitext(os.path.basename(image))[0]
    out_dir = args.out_dir or os.path.join(ROOT, "output", f"inference_{stem}")
    os.makedirs(out_dir, exist_ok=True)
    modes = ["text", "visual", "multimodal"] if args.modes == "all" else [args.modes]
    log(f"image: {image}\noutput: {out_dir}\nmodes: {modes}")

    # C++ FIRST: the official PyTorch model pins several GB of VRAM for the
    # rest of this process, which would OOM the engine subprocess; the engine
    # subprocess exits and releases its memory before the torch side loads.
    cpp = None if args.skip_cpp else run_cpp(args, image, modes, out_dir)
    py = None if args.skip_python else run_python(args, image, modes, out_dir)
    compare(args, image, modes, py, cpp, out_dir)
    log(f"[done] all outputs in {out_dir}")


if __name__ == "__main__":
    main()
