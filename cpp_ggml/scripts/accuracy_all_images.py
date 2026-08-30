#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Accuracy comparison over ALL official test data:
#   1. the 15 images of test_real_world/ims1/ (text mode — the mode with a
#      well-defined prompt set per image), stock PyTorch vs every engine config
#   2. the three official demo commands of
#      test_real_world/scripts/eval_single_obj_gkd.sh on 2007_007524.jpg:
#        - multimodal, whole image (support: 2007_003778)
#        - text, bbox ROI [33,38,241,310]
#        - visual, cross-image support (alpaca_150.jpg)
#
# Usage:
#   python scripts/accuracy_all_images.py --pytorch     # torch env; dump refs
#   python scripts/accuracy_all_images.py               # engine side + diff
#
# Output:
#   benchmarks/pytorch_ref_texts.json     (step 1)
#   benchmarks/accuracy_all_images.json   (step 2: refs + per-config per-image
#                                          errors + official examples)
# Charts: scripts/plot_benchmarks.py (accuracy_by_image.png) and
#         scripts/render_parity.py --grid (parity_official_examples.png).
# ------------------------------------------------------------------------------
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_tests as R  # noqa: E402  (reuses PROMPTS / discover_configs / run / recover)

ROOT = R.ROOT
GKD = R.GKD
IMS = R.IMS
BENCH = os.path.join(GKD, "benchmarks")
REF_JSON = os.path.join(BENCH, "pytorch_ref_texts.json")
OUT_JSON = os.path.join(BENCH, "accuracy_all_images.json")
IMAGES = sorted(R.PROMPTS)

# The three official demo commands (eval_single_obj_gkd.sh), verbatim prompts.
OFFICIAL_EXAMPLES = {
    "multimodal_whole_image": {
        "image": "2007_007524.jpg", "bbox": [],
        "kps_texts": ["left eye", "right eye", "nose"],
        "support_image": "2007_003778.jpg", "support_kps": [343, 166, 281, 158, 311, 197],
    },
    "text_bbox_roi": {
        "image": "2007_007524.jpg", "bbox": [33, 38, 241, 310],
        "kps_texts": ["nose", "left eye", "right eye", "left ear", "right ear"],
        "support_image": "", "support_kps": [],
    },
    "visual_alpaca_support": {
        "image": "2007_007524.jpg", "bbox": [],
        "kps_texts": ["left eye", "right eye", "nose"],
        "support_image": "alpaca_150.jpg", "support_kps": [615, 495, 483, 493, 521, 549],
    },
}


# ---------------------------------------------------------------- PyTorch side
def dump_pytorch_refs():
    """Stock PyTorch outputs for all 15 images (text mode) + the 3 official
    demo commands. Must run inside the torch env, from the repo root."""
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import torch  # noqa: E402
    from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, demo  # noqa: E402

    cfg = os.path.join(ROOT, "test_real_world", "configs", "gkd.yaml")
    ckpt = os.path.join(GKD, "models", "pytorch", "gkd_fullset.best")
    infer = GKDInference(cfg_file=cfg, checkpoint_path=ckpt)
    infer.gkd_model.eval()
    torch.set_grad_enabled(False)

    refs = {"images": {}, "official_examples": {}}
    for im in IMAGES:
        preds, score, _ = demo(infer, os.path.join(IMS, im), [], "", [], R.PROMPTS[im])
        refs["images"][im] = {
            "kps": [round(float(v), 3) for v in preds.reshape(-1).cpu()],
            "scores": [round(float(v), 4) for v in score.reshape(-1).cpu()],
            "n_prompts": len(R.PROMPTS[im]),
        }
        print(f"[pytorch-ref] {im}: n={len(R.PROMPTS[im])} "
              f"scores={[round(float(s), 3) for s in score.reshape(-1)[:3]]}...")

    for name, ex in OFFICIAL_EXAMPLES.items():
        support = os.path.join(IMS, ex["support_image"]) if ex["support_image"] else ""
        preds, score, _ = demo(infer, os.path.join(IMS, ex["image"]),
                               ex["bbox"], support, ex["support_kps"], ex["kps_texts"])
        refs["official_examples"][name] = {
            "kps": [round(float(v), 3) for v in preds.reshape(-1).cpu()],
            "scores": [round(float(v), 4) for v in score.reshape(-1).cpu()],
            "n_prompts": len(ex["kps_texts"]),
        }
        print(f"[pytorch-ref] official:{name}: scores="
              f"{[round(float(s), 4) for s in score.reshape(-1)]}")

    with open(REF_JSON, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"[pytorch-ref] -> {REF_JSON}")


# ----------------------------------------------------------------- engine side
def engine_detect(build, dtype, ex):
    """Run one gkd-cli detect with the given example spec; return parsed json."""
    model = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{dtype}.gguf")
    cmd = [os.path.join(GKD, build, "bin", "gkd-cli"), "detect", "--model", model,
           "--input", os.path.join(IMS, ex["image"]), "--threads", "16"]
    if ex["bbox"]:
        cmd += ["--bbox"] + [str(int(v)) for v in ex["bbox"]]
    for t in ex["kps_texts"]:
        cmd += ["--kps-texts", t]
    if ex["support_image"]:
        cmd += ["--support-image", os.path.join(IMS, ex["support_image"])]
        for v in ex["support_kps"]:
            cmd += ["--support-kps", str(v)]
    out = R.run(cmd)
    for line in out.splitlines():
        if line.startswith('{"json":'):
            return json.loads(line)["json"]
    raise RuntimeError(f"no json output ({build}-{dtype}, {ex['image']})")


def compare(kps_orig, scores, ref):
    n = len(scores)
    ref_k = ref["kps"][:2 * n]
    errs = [abs(a - b) for a, b in zip(kps_orig, ref_k)]
    sdiff = [abs(a - b) for a, b in zip(scores, ref["scores"][:n])]
    return {
        "mean_err_px": round(sum(errs) / len(errs), 4),
        "max_err_px": round(max(errs), 4),
        "mean_score_diff": round(sum(sdiff) / len(sdiff), 5),
        "kps_orig": [round(v, 2) for v in kps_orig],
        "scores": [round(v, 4) for v in scores],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch", action="store_true",
                    help="dump the stock-PyTorch references (torch env)")
    ap.add_argument("--configs", default=None, help="comma-separated config tags")
    args = ap.parse_args()

    os.makedirs(BENCH, exist_ok=True)
    if args.pytorch:
        dump_pytorch_refs()
        return

    with open(REF_JSON) as f:
        refs = json.load(f)
    configs = R.discover_configs()
    if args.configs:
        want = set(args.configs.split(","))
        configs = [c for c in configs
                   if f"{c[0].replace('build-', '')}-{c[1]}" in want]

    out = {"pytorch_ref": refs, "configs": {}, "official_examples": {}}

    # --- all images, text mode, every config ---
    for build, dtype in configs:
        tag = f"{build.replace('build-', '')}-{dtype}"
        out["configs"][tag] = {}
        for im in IMAGES:
            j = engine_detect(build, dtype,
                              {"image": im, "bbox": [], "kps_texts": R.PROMPTS[im],
                               "support_image": "", "support_kps": []})
            n = j["n_prompts"]
            kps = R.recover([(j["kps_norm"][2 * i], j["kps_norm"][2 * i + 1])
                             for i in range(n)], j["trans"])
            out["configs"][tag][im] = compare(kps, j["scores"], refs["images"][im])
        errs = [out["configs"][tag][im]["mean_err_px"] for im in IMAGES]
        print(f"[accuracy-all] {tag}: mean over images = "
              f"{sum(errs) / len(errs):.4f} px")

    # --- the three official demo commands, every config ---
    for name, ex in OFFICIAL_EXAMPLES.items():
        ref = refs["official_examples"][name]
        out["official_examples"][name] = {"ref": ref}
        for build, dtype in configs:
            tag = f"{build.replace('build-', '')}-{dtype}"
            j = engine_detect(build, dtype, ex)
            n = j["n_prompts"]
            kps = R.recover([(j["kps_norm"][2 * i], j["kps_norm"][2 * i + 1])
                             for i in range(n)], j["trans"])
            out["official_examples"][name][tag] = compare(kps, j["scores"], ref)
        print(f"[accuracy-all] official:{name} done")

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[accuracy-all] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
