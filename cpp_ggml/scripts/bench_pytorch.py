#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Official PyTorch reference latency for GKDT-L, measured through the stock
# test_real_world demo path (GKDInference.demo), mirroring the gkd-cli bench
# protocol (warmup + timed iters, end-to-end incl. preprocessing).
#
# Usage (conda env with torch+cuda):
#   python scripts/bench_pytorch.py [--image ...] [--warmup 3] [--iters 10]
# Output: benchmarks/pytorch.jsonl (one record per prompt mode)
# ------------------------------------------------------------------------------
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test_real_world"))

CAT_KPS = ["nose", "left eye", "right eye", "left ear", "right ear"]
HEAD_KPS = ["left eye", "right eye", "nose"]
SUPPORT_IM = "test_real_world/ims1/2007_003778.jpg"
SUPPORT_KPS = [343, 166, 281, 158, 311, 197]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.path.join(ROOT, "test_real_world/configs/gkd.yaml"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "cpp_ggml/models/pytorch/gkd_fullset.best"))
    ap.add_argument("--image", default=os.path.join(ROOT, "test_real_world/ims1/2007_007524.jpg"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "benchmarks", "pytorch.jsonl"))
    args = ap.parse_args()

    os.chdir(ROOT)  # official demo resolves relative paths against the repo root
    from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, demo  # noqa: E402

    infer = GKDInference(cfg_file=args.cfg, checkpoint_path=args.checkpoint)
    infer.gkd_model.eval()
    torch.set_grad_enabled(False)

    modes = {
        "text": dict(kps_texts=CAT_KPS),
        "visual": dict(support_im=SUPPORT_IM, support_kps=SUPPORT_KPS, kps_texts=[]),
        "multimodal": dict(support_im=SUPPORT_IM, support_kps=SUPPORT_KPS, kps_texts=HEAD_KPS),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fout:
        for mode, kw in modes.items():
            # warmup
            for _ in range(args.warmup):
                demo(infer, args.image, [], kw.get("support_im", ""), kw.get("support_kps", []),
                     kps_texts=kw.get("kps_texts", []))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = None
            for _ in range(args.iters):
                preds, score, _wh = demo(infer, args.image, [], kw.get("support_im", ""),
                                         kw.get("support_kps", []),
                                         kps_texts=kw.get("kps_texts", []))
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / args.iters * 1000.0
            rec = {"backend": "pytorch-cuda", "input": args.image, "mode": mode,
                   "warmup": args.warmup, "iters": args.iters, "total_ms": wall,
                   "n_prompts": int(preds.shape[1])}
            fout.write(json.dumps(rec) + "\n")
            print(json.dumps(rec))


if __name__ == "__main__":
    main()
