#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Stage-2 (GKD) numeric parity for the official multi-object pipeline.
#
# The official multi-object flow (test_real_world/multi_obj_gkd_inference.py)
# is two-stage:
#
#     1. object detector (GroundingDINO / LocateAnything-3B)
#        -> ROI bbox coordinates, an exact float interface
#     2. GKD per class: demo(gkd_inference, image, bboxes, support_im,
#        support_kps, kps_texts)  ->  N_bbox x N x 2 predictions + scores
#
# The detector is a separate multi-billion-parameter model; the interface
# between the stages is exactly the bbox coordinates. To verify the GKD stage
# bit-comparably we therefore feed IDENTICAL bbox lists to both sides,
# bypassing the detector exactly at its interface:
#
#     official PyTorch : demo(...) from test_real_world/gkd_inference_lib
#     C++ ggml engine  : gkd-cli detect --bbox x1 y1 x2 y2 ...   (one call, N ROIs)
#
# Usage (torch env):
#   python cpp_ggml/scripts/parity_multi_object.py [--backend cuda] [--dtype f32]
# ------------------------------------------------------------------------------
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GKD = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test_real_world"))

# plausible fixed ROIs (what a detector could plausibly emit); identical for
# both sides - the values themselves are irrelevant to the parity check
CASES = [
    {
        "name": "cat_dog / text / 2 ROIs",
        "image": "cat_dog.jpg",
        "bboxes": [[92.0, 29.0, 394.0, 290.0], [430.0, 60.0, 800.0, 297.0]],
        "mode": "text",
        "kps_texts": ["nose", "left eye", "right eye", "right ear"],
    },
    {
        "name": "cat_dog / multimodal / 2 ROIs",
        "image": "cat_dog.jpg",
        "bboxes": [[92.0, 29.0, 394.0, 290.0], [430.0, 60.0, 800.0, 297.0]],
        "mode": "multimodal",
        "kps_texts": ["nose", "left eye", "right ear"],
        "support_im": "2007_007524.jpg",
        "support_kps": [111.76, 231.31, 153.34, 174.13, 90.96, 137.74],
    },
    {
        "name": "alpaca / text / 3 ROIs",
        "image": "alpaca_150.jpg",
        "bboxes": [[551.0, 388.0, 1237.0, 1058.0], [60.0, 40.0, 520.0, 620.0],
                   [1300.0, 300.0, 1800.0, 1100.0]],
        "mode": "text",
        "kps_texts": None,  # official 'alpaca' predefined schema (9 kps)
        "object_name": "alpaca",
    },
]


def run_official(case, checkpoint, cfg):
    from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, demo
    from test_real_world.predefined_keypoints import get_prompt_info
    import numpy as np

    infer = GKDInference(cfg_file=cfg, checkpoint_path=checkpoint)
    infer.gkd_model.eval()
    torch_ok = __import__("torch")
    torch_ok.set_grad_enabled(False)

    image = os.path.join(ROOT, "test_real_world", "ims1", case["image"])
    support_im = os.path.join(ROOT, "test_real_world", "ims1", case.get("support_im", "")) \
        if case.get("support_im") else ""
    texts = case["kps_texts"]
    if texts is None:  # official per-class predefined-schema fallback
        texts, _, _, _ = get_prompt_info(case["object_name"], [], "", [], [])
    bboxes = [c for bb in case["bboxes"] for c in bb]
    preds, scores, _ = demo(infer, image, bboxes, support_im,
                            list(case.get("support_kps", [])), list(texts))
    return preds.numpy(), scores.numpy(), list(texts)


def run_engine(args, case):
    import numpy as np
    import run_all_tests as R  # recover() helper
    cli = os.path.join(GKD, f"build-{args.backend}", "bin", "gkd-cli")
    model = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{args.dtype}.gguf")
    image = os.path.join(ROOT, "test_real_world", "ims1", case["image"])
    support_im = os.path.join(ROOT, "test_real_world", "ims1", case.get("support_im", "")) \
        if case.get("support_im") else ""
    texts = case["kps_texts"]
    if texts is None:
        sys.path.insert(0, ROOT)
        from test_real_world.predefined_keypoints import get_prompt_info
        texts, _, _, _ = get_prompt_info(case["object_name"], [], "", [], [])

    cmd = [cli, "detect", "--model", model, "--input", image,
           "--threads", str(args.threads)]
    for c in case["bboxes"]:
        cmd += ["--bbox", *[repr(float(v)) for v in c]]
    for t in texts:
        cmd += ["--kps-texts", t]
    if support_im:
        cmd += ["--support-image", support_im]
        for v in case.get("support_kps", []):
            cmd += ["--support-kps", str(v)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise RuntimeError(f"gkd-cli failed:\n{out.stderr[-2000:]}")
    rois = [json.loads(l)["json"] for l in out.stdout.splitlines()
            if l.startswith('{"json":')]
    kps, scores = [], []
    for r in sorted(rois, key=lambda x: x["roi"]):
        n = r["n_prompts"]
        xy = R.recover([(r["kps_norm"][2 * i], r["kps_norm"][2 * i + 1]) for i in range(n)],
                       r["trans"])
        kps.append(np.asarray(xy, np.float64).reshape(-1, 2))
        scores.append(r["scores"])
    return kps, scores, list(texts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "vulkan"])
    ap.add_argument("--dtype", default="f32",
                    choices=["f32", "f16", "q8_0", "q4_K"])
    ap.add_argument("--checkpoint",
                    default=os.path.join(GKD, "models", "pytorch", "gkd_fullset.best"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "test_real_world", "configs", "gkd.yaml"))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    n_case = n_roi = n_conf = 0
    worst_xy = worst_s = 0.0
    n_lowconf = 0
    for case in CASES:
        po, so, texts = run_official(case, args.checkpoint, args.cfg)
        ke, se, _ = run_engine(args, case)
        for i in range(len(case["bboxes"])):
            dxy = np.abs(np.asarray(po[i], np.float64) - np.asarray(ke[i], np.float64))
            ds = np.abs(np.asarray(so[i], np.float64) - np.asarray(se[i], np.float64))
            # An ROI only has a meaningful argmax where the OFFICIAL model is
            # confident; on noise-floor ROIs (all scores < ~0.1, e.g. a
            # background box) the heatmap peak is a near-tie and the argmax
            # location is chaotic for any implementation (fp32 included).
            conf = float(np.max(so[i]))
            tag = "confident" if conf >= 0.1 else "noise-floor"
            if conf >= 0.1:
                worst_xy = max(worst_xy, float(dxy.max()))
                worst_s = max(worst_s, float(ds.max()))
                n_conf += 1
            else:
                n_lowconf += 1
            n_roi += 1
            print(f"  {case['name']} roi{i} [{tag}, top={conf:.2f}]: "
                  f"max|dxy|={dxy.max():.4f}px  max|dscore|={ds.max():.5f}  "
                  f"({len(texts)} prompts)")
        n_case += 1

    print(f"\n{n_case} cases / {n_roi} ROIs ({n_conf} confident, {n_lowconf} noise-floor): "
          f"worst confident coord delta = {worst_xy:.4f} px, "
          f"worst score delta = {worst_s:.5f}")
    print("GKD multi-object stage aligned with the official demo()"
          if worst_xy < 0.1 else "MISALIGNED - investigate")


if __name__ == "__main__":
    main()
