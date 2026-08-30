#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Full evaluation driver for the GGML runtime:
#   1. latency  : 15 official images x {text, visual, multimodal} prompts per
#                 (backend, dtype) config, warmup+iters, one session per config
#   2. accuracy : the three official single-object examples on 2007_007524.jpg
#                 compared against the stock PyTorch reference outputs
#
# Usage:  python scripts/run_all_tests.py [--iters 10] [--warmup 3]
# Output: benchmarks/latency_<config>.jsonl and benchmarks/accuracy.json
# ------------------------------------------------------------------------------
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))       # repo root
GKD = os.path.dirname(HERE)                         # cpp_ggml/
IMS = os.path.join(ROOT, "test_real_world", "ims1")

TEXT5 = ["nose", "left eye", "right eye", "left ear", "right ear"]
HEAD3 = ["left eye", "right eye", "nose"]
SUPPORT_IM = os.path.join(IMS, "2007_003778.jpg")
SUPPORT_KPS = [343, 166, 281, 158, 311, 197]

# per-image text prompt sets (official example prompts where defined)
PROMPTS = {
    "2007_007524.jpg": TEXT5, "2007_003778.jpg": TEXT5, "cat_dog.jpg": TEXT5,
    "000002.jpg": TEXT5, "00000016.jpg": TEXT5, "000000011511.jpg": TEXT5,
    "2008_000808.jpg": TEXT5, "3144.png": TEXT5,
    "alpaca_150.jpg": ["left eye", "right eye", "left ear", "right ear", "nose",
                       "throat", "withers", "tail", "left-front leg", "right-front leg",
                       "left-back leg", "right-back leg", "left-front knee", "right-front knee",
                       "left-back knee", "right-back knee", "left-front paw", "right-front paw",
                       "left-back paw", "right-back paw"],
    "adeliepenguin_107.jpg": ["head", "beak", "left flipper", "right flipper", "tail"],
    "fish_swim.jpg": ["head", "tail", "dorsal fin", "left fin", "right fin"],
    "pet_birds.jpg": ["head", "beak", "left claw", "right claw", "tail"],
    "pigs_stock_farming.jpg": ["head", "snout", "left ear", "right ear", "tail"],
    "car_penn2_0_1931.jpg": ["front left wheel", "front right wheel", "rear left wheel", "rear right wheel"],
    "wash_dishes_egocentric.jpg": ["left hand", "right hand"],
}

IMAGES = sorted(PROMPTS)

# Every (backend, dtype) pair with both a built gkd-cli and a converted GGUF
# enters the matrix automatically - no cherry-picking. Order fixes the chart
# and table ordering: backend cpu -> cuda -> vulkan, dtype f32 -> f16 -> q8_0
# -> q4_0 -> q4_K.
BACKEND_ORDER = ["cpu", "cuda", "vulkan"]
DTYPE_ORDER = ["f32", "f16", "q8_0", "q4_0", "q4_K"]


def discover_configs():
    configs = []
    for backend in BACKEND_ORDER:
        build = f"build-{backend}"
        cli = os.path.join(GKD, build, "bin", "gkd-cli")
        if not os.path.exists(cli):
            continue
        for dtype in DTYPE_ORDER:
            if os.path.exists(os.path.join(GKD, "models", "gguf", f"gkd_fullset-{dtype}.gguf")):
                configs.append((build, dtype))
    return configs


CONFIGS = discover_configs()


def run(cmd, env=None):
    e = dict(os.environ)
    e.setdefault("CUDA_PATH", "/usr/local/cuda-12.6")
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r.stdout


def bench_once(build, gguf, mode, images, warmup, iters):
    model = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{gguf}.gguf")
    cmd = [os.path.join(GKD, build, "bin", "gkd-cli"), "bench", "--model", model,
           "--threads", "16", "--warmup", str(warmup), "--iters", str(iters)]
    for im in images:
        cmd += ["--input", os.path.join(IMS, im)]
    texts = PROMPTS[images[0]] if mode == "text" else HEAD3  # multimodal pairs 3+3
    if mode in ("text", "multimodal"):
        for t in texts:
            cmd += ["--kps-texts", t]
    if mode in ("visual", "multimodal"):
        cmd += ["--support-image", SUPPORT_IM]
        for v in SUPPORT_KPS:
            cmd += ["--support-kps", str(v)]
    out = run(cmd)
    recs = [json.loads(l) for l in out.splitlines() if l.startswith('{"backend"')]
    for r in recs:
        r["mode"] = mode
        r["dtype"] = gguf
        r["n_prompts"] = len(texts) if mode == "text" else (len(SUPPORT_KPS) // 2 if mode == "visual" else len(HEAD3))
    return recs


def detect_once(build, gguf, mode):
    """Run the official example on 2007_007524.jpg; return parsed json dict."""
    model = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{gguf}.gguf")
    cmd = [os.path.join(GKD, build, "bin", "gkd-cli"), "detect", "--model", model,
           "--input", os.path.join(IMS, "2007_007524.jpg"), "--threads", "16"]
    if mode in ("text", "multimodal"):
        texts = TEXT5 if mode == "text" else HEAD3
        for t in texts:
            cmd += ["--kps-texts", t]
    if mode in ("visual", "multimodal"):
        cmd += ["--support-image", SUPPORT_IM]
        for v in SUPPORT_KPS:
            cmd += ["--support-kps", str(v)]
    out = run(cmd)
    for line in out.splitlines():
        if line.startswith('{"json":'):
            return json.loads(line)["json"]
    raise RuntimeError("no json output from gkd-cli detect")


def recover(norm_xy, trans):
    """-1..1 ROI coords -> original pixels using the engine's ScaleTrans
    (official mytransforms.recover_kps: (k/2+0.5)*L + offset, then /scale)."""
    scale, offx, offy = trans
    out = []
    for x, y in norm_xy:
        out.append(((x / 2 + 0.5) * 384 + offx) / scale)
        out.append(((y / 2 + 0.5) * 384 + offy) / scale)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--configs", default=None,
                    help="comma-separated config tags (e.g. cpu-f32,cuda-q4_K); default: all discovered")
    args = ap.parse_args()

    configs = discover_configs()
    if args.configs:
        want = set(args.configs.split(","))
        configs = [c for c in configs if f"{c[0].replace('build-', '')}-{c[1]}" in want]
    print("[configs] " + ", ".join(f"{b.replace('build-', '')}-{d}" for b, d in configs))

    bench_dir = os.path.join(GKD, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)

    sizes = {}
    from PIL import Image
    for im in IMAGES:
        with Image.open(os.path.join(IMS, im)) as f:
            sizes[im] = (f.width, f.height)

    # ---------------- latency matrix ----------------
    if not args.skip_latency:
        for build, gguf in configs:
            tag = f"{build.replace('build-', '')}-{gguf}"
            out_path = os.path.join(bench_dir, f"latency_{tag}.jsonl")
            with open(out_path, "w") as fout:
                for mode in ("text", "visual", "multimodal"):
                    recs = bench_once(build, gguf, mode, IMAGES, args.warmup, args.iters)
                    for r in recs:
                        fout.write(json.dumps(r) + "\n")
            print(f"[latency] {tag} -> {out_path}")

    # ---------------- accuracy vs PyTorch references ----------------
    if not args.skip_accuracy:
        ref = {   # from the stock PyTorch runs (pred_orig / pred_score taps + result.json)
            "text": {"kps": [111.76, 231.31, 153.34, 174.13, 90.96, 194.92,
                             179.33, 116.95, 90.96, 137.74],
                     "scores": [0.9243, 0.9370, 0.9542, 0.8147, 0.8845]},
            "visual": {"kps": [153.34, 174.13, 90.96, 194.92, 111.76, 231.31],
                       "scores": [0.9311, 0.9484, 0.4230]},
            "multimodal": {"kps": [153.34, 174.13, 90.96, 194.92, 111.76, 231.31],
                           "scores": [0.9347, 0.9554, 0.8796]},
        }
        acc = {}
        for build, gguf in configs:
            tag = f"{build.replace('build-', '')}-{gguf}"
            acc[tag] = {}
            for mode in ("text", "visual", "multimodal"):
                j = detect_once(build, gguf, mode)
                n = j["n_prompts"]
                kps_orig = recover([(j["kps_norm"][2 * i], j["kps_norm"][2 * i + 1]) for i in range(n)],
                                   j["trans"])
                ref_k = ref[mode]["kps"][:2 * n]
                errs = [abs(a - b) for a, b in zip(kps_orig, ref_k)]
                mean_err = sum(errs) / len(errs)
                score_diff = [abs(a - b) for a, b in zip(j["scores"], ref[mode]["scores"][:n])]
                acc[tag][mode] = {
                    "kps_orig": [round(v, 2) for v in kps_orig],
                    "scores": [round(v, 4) for v in j["scores"]],
                    "mean_coord_err_px": round(mean_err, 4),
                    "max_coord_err_px": round(max(errs), 4),
                    "mean_score_diff": round(sum(score_diff) / len(score_diff), 5),
                }
                print(f"[accuracy] {tag} {mode}: mean_err={mean_err:.3f}px "
                      f"score_diff={acc[tag][mode]['mean_score_diff']:.5f}")
        with open(os.path.join(bench_dir, "accuracy.json"), "w") as f:
            json.dump(acc, f, indent=2)
        print(f"[accuracy] -> {os.path.join(bench_dir, 'accuracy.json')}")


if __name__ == "__main__":
    main()
