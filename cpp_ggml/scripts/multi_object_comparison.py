#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-object GKD comparison: official PyTorch pipeline vs C++ ggml engine,
on the official multi-object test matrix (eval_multi_obj_gkd.sh images).

Both sides consume the SAME detector output (benchmarks/detector_bboxes.json,
produced by the official GroundingDINO adapter), so every delta below is pure
GKD-stage. The official demo() batches ROIs <= 8 per call (batch-independent
math, batching only bounds memory).

Outputs:
    benchmarks/multi_object_comparison.json  - per class/ROI numeric deltas
    benchmarks/multi_object_parity.png       - per-image overlay grid
                                               (PyTorch GKD vs C++ GKD)

Usage:  ~/anaconda3/envs/python3.12/bin/python cpp_ggml/scripts/multi_object_comparison.py \
            [--backend cuda] [--dtype q4_K] [--batch 8]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GKD = os.path.dirname(HERE)
IMS = os.path.join(ROOT, "test_real_world", "ims1")
sys.path.insert(0, ROOT)


def run_official_gkd(infer, image_path, object_name, bboxes, kps_texts, batch):
    """Official demo() over one class's bbox list, batched. Returns
    (predictions [N_roi][N_kp][2], scores [N_roi][N_kp])."""
    import numpy as np
    preds, scores = [], []
    for s in range(0, len(bboxes), batch):
        chunk = bboxes[s:s + batch]
        flat = [c for bb in chunk for c in bb]
        po, sc, _ = demo(infer, image_path, flat, "", [], list(kps_texts))
        preds.extend(po.numpy().tolist())
        scores.extend(sc.numpy().tolist())
    return preds, scores


def run_engine_gkd(args, image_path, kps_texts, bboxes):
    """C++ engine over the same bboxes (batched CLI calls). Returns same shape."""
    cli = os.path.join(GKD, f"build-{args.backend}", "bin", "gkd-cli")
    model = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{args.dtype}.gguf")
    preds, scores = [], []
    for s in range(0, len(bboxes), args.batch):
        chunk = bboxes[s:s + args.batch]
        cmd = [cli, "detect", "--model", model, "--input", image_path,
               "--threads", str(args.threads)]
        for bb in chunk:
            cmd += ["--bbox", *(repr(float(v)) for v in bb)]
        for t in kps_texts:
            cmd += ["--kps-texts", t]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"gkd-cli failed:\n{r.stderr[-1500:]}")
        rois = sorted((json.loads(l)["json"] for l in r.stdout.splitlines()
                       if l.startswith('{"json":')), key=lambda x: x["roi"])
        for roi in rois:
            n = roi["n_prompts"]
            scale, offx, offy = roi["trans"]
            xy = [[(roi["kps_norm"][2 * i] / 2 + 0.5) * 384 / scale + offx / scale,
                   (roi["kps_norm"][2 * i + 1] / 2 + 0.5) * 384 / scale + offy / scale]
                  for i in range(n)]
            preds.append(xy)
            scores.append(roi["scores"])
    return preds, scores


def render(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#46f0f0", "#f032e6", "#bcf60c", "#008080"]

    def draw_overlay(img, res, side):
        im = img.copy()
        dr = ImageDraw.Draw(im, "RGBA")
        for ci, cls in enumerate(res["classes"]):
            col = PALETTE[ci % len(PALETTE)]
            for roi in cls["rois"]:
                x1, y1, x2, y2 = roi["bbox"]
                dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
                dr.text((x1 + 2, max(0, y1 - 12)),
                        f"{cls['object_name']} {roi['detector_score']:.2f}", fill=col)
                src = roi["official"] if side == 0 else roi["engine"]
                for (x, y), s in zip(src["predictions"], src["scores"]):
                    r = 2.5 + 3.5 * min(max(s, 0.0), 1.0)
                    dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
        return im

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.4 * n), squeeze=False)
    for ri, res in enumerate(results):
        img = Image.open(res["image"]).convert("RGB")
        for ci, title in enumerate(
                ("official PyTorch (GroundingDINO + GKDT)",
                 f"C++ ggml engine (same detector boxes)")):
            ax = axes[ri][ci]
            ax.imshow(draw_overlay(img, res, ci))
            ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)
            ax.axis("off")
            ax.set_title((f"{res['image_name']}\n" if ci == 0 else "\n") + title,
                         fontsize=9)
    fig.suptitle("Multi-object GKD: official PyTorch pipeline vs C++ ggml engine\n"
                 "(identical GroundingDINO boxes; per-object keypoints, radius = score)",
                 y=0.998)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "vulkan"])
    ap.add_argument("--dtype", default="q4_K", choices=["f32", "f16", "q8_0", "q4_K"])
    ap.add_argument("--batch", type=int, default=8, help="ROIs per forward call")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--checkpoint",
                    default=os.path.join(GKD, "models", "pytorch", "gkd_fullset.best"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "test_real_world", "configs", "gkd.yaml"))
    ap.add_argument("--skip-official", action="store_true",
                    help="reuse cached official predictions in the comparison json")
    args = ap.parse_args()

    dets_path = os.path.join(GKD, "benchmarks", "detector_bboxes.json")
    with open(dets_path) as f:
        dets = json.load(f)["images"]

    out_path = os.path.join(GKD, "benchmarks", "multi_object_comparison.json")
    cached = {}
    if args.skip_official and os.path.exists(out_path):
        with open(out_path) as f:
            cached = {r["image_name"]: r for r in json.load(f)["results"]}

    global demo
    from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, demo
    from test_real_world.predefined_keypoints import get_prompt_info
    infer = GKDInference(cfg_file=args.cfg, checkpoint_path=args.checkpoint)
    infer.gkd_model.eval()

    results = []
    n_roi = n_conf = 0
    worst_xy = worst_s = 0.0
    for image_name, classes in dets.items():
        image_path = os.path.join(IMS, image_name)
        res = cached.get(image_name, {"image": image_path, "image_name": image_name,
                                      "classes": []})
        for cls in classes["classes"]:
            name = cls["object_name"]
            entries = cls["entries"]
            if not entries:
                continue
            bboxes = [e["bbox"] for e in entries]
            kps_texts, _, _, _ = get_prompt_info(name, [], "", [], [])
            if kps_texts is None:
                kps_texts = []
            if args.skip_official and res.get("classes"):
                old = next((c for c in res["classes"] if c["object_name"] == name), None)
                if old and "official" in old:
                    po, so = old["official"]["predictions"], old["official"]["scores"]
                else:
                    po, so = run_official_gkd(infer, image_path, name, bboxes,
                                              kps_texts, args.batch)
            else:
                po, so = run_official_gkd(infer, image_path, name, bboxes,
                                          kps_texts, args.batch)
            pe, se = run_engine_gkd(args, image_path, kps_texts, bboxes)
            assert len(po) == len(pe) == len(bboxes)
            rois = []
            for i, bb in enumerate(bboxes):
                import numpy as np
                dxy = np.abs(np.asarray(po[i], np.float64) - np.asarray(pe[i], np.float64))
                ds = np.abs(np.asarray(so[i], np.float64) - np.asarray(se[i], np.float64))
                conf = float(np.max(so[i]))
                tag = "confident" if conf >= 0.1 else "noise-floor"
                if conf >= 0.1:
                    worst_xy = max(worst_xy, float(dxy.max()))
                    worst_s = max(worst_s, float(ds.max()))
                    n_conf += 1
                n_roi += 1
                rois.append({"bbox": bb, "detector_score": entries[i]["score"],
                             "confident": conf >= 0.1,
                             "max_dxy": float(dxy.max()), "max_dscore": float(ds.max()),
                             "official": {"predictions": po[i], "scores": so[i]},
                             "engine": {"predictions": pe[i], "scores": se[i]}})
            res["classes"].append({"object_name": name, "kps_texts": kps_texts,
                                   "n_kps": len(kps_texts), "rois": rois})
        results.append(res)
        done = sum(len(r.get("classes", [])) for r in results)
        print(f"[{image_name}] done ({done} classes total)")

    summary = {"detector": "groundingdino_swint_ogc (official adapter)",
               "engine": f"{args.backend}-{args.dtype}",
               "note": "identical detector boxes on both sides; deltas are pure GKD stage",
               "summary": {"rois": n_roi, "confident_rois": n_conf,
                           "worst_confident_coord_px": worst_xy,
                           "worst_confident_score": worst_s},
               "results": results}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_path)
    print(f"multi-object matrix: {n_roi} ROIs ({n_conf} confident) | "
          f"worst confident coord delta = {worst_xy:.4f} px | "
          f"worst score delta = {worst_s:.5f}")
    render(results, os.path.join(GKD, "benchmarks", "multi_object_parity.png"))


if __name__ == "__main__":
    main()
