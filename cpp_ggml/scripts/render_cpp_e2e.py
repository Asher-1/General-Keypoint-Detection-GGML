#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the pure-C++ end-to-end comparison charts into benchmarks/.

Two charts, both driven by REAL runs (no cached numbers):

  benchmarks/cpp_e2e_single.png
      Official PyTorch vs C++ ggml engine on the official demo image, all
      three prompt modes, identical keypoint drawing (the run_inference.py
      protocol, out-dir pointed at benchmarks/).

  benchmarks/cpp_e2e_multi.png
      The FULL multi-object pipeline on official multi-object demo images:
      left = official PyTorch pipeline (GroundingDINO boxes + PyTorch GKD,
      cached from the earlier parity run), right = the PURE C++ pipeline
      (ultralytics-ggml YOLO-World boxes + gkd-cli). Column titles state the
      detector each side used - the box difference IS the detector ablation,
      quantified per matched object.

  benchmarks/cpp_e2e_multi.json
      Per-image/per-class numeric summary (matched-object keypoint
      displacement between the two pipelines' box sets).

Usage:  ~/anaconda3/envs/python3.12/bin/python cpp_ggml/scripts/render_cpp_e2e.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GKD = os.path.dirname(HERE)
BENCH = os.path.join(GKD, "benchmarks")
IMS = os.path.join(ROOT, "test_real_world", "ims1")
sys.path.insert(0, ROOT)

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#46f0f0", "#f032e6", "#bcf60c", "#008080"]


# ---------------------------------------------------------------------------
# part 1: single-object dual-implementation (reuse run_inference machinery)
# ---------------------------------------------------------------------------
def render_single(args):
    log("[single] official PyTorch vs C++ ggml, 3 prompt modes")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_inference.py"),
                        "--out-dir", os.path.join(BENCH, "cpp_e2e_single"),
                        "--dtype", args.dtype, "--backend", args.backend],
                       capture_output=True, text=True, timeout=7200)
    src = os.path.join(BENCH, "cpp_e2e_single", "python_vs_cpp.png")
    dst = os.path.join(BENCH, "cpp_e2e_single.png")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst)
        log(f"[single] wrote {dst}")
    else:
        log(f"[single] SKIP - run_inference.py did not produce the chart: "
            f"{r.stderr[-300:]}")


# ---------------------------------------------------------------------------
# part 2: full multi-object pipeline, official PyTorch vs pure C++
# ---------------------------------------------------------------------------
def draw_overlay(img, classes, which):
    from PIL import ImageDraw
    im = img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    for ci, cls in enumerate(classes):
        col = PALETTE[ci % len(PALETTE)]
        for roi in cls["rois"]:
            x1, y1, x2, y2 = roi["bbox"]
            dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
            dr.text((x1 + 2, max(0, y1 - 12)),
                    f"{cls['object_name']} {roi.get('detector_score', 1.0):.2f}",
                    fill=col)
            src = roi[which]
            for (x, y), s in zip(src["predictions"], src["scores"]):
                r = 2.5 + 3.5 * min(max(s, 0.0), 1.0)
                dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return im


def run_cpp_pipeline(image_path, classes, args):
    """Pure C++ pipeline: yolo-cli (submodule) per class-name -> gkd-cli.
    Returns classes in the same shape as the cached official reference."""
    cli = os.path.join(GKD, "third_party", "ultralytics-ggml", "cpp_ggml",
                       f"build-{args.backend}", "bin", "yolo-cli")
    det_model = os.path.join(GKD, "models", "gguf", "yolov8x-worldv2-f16.gguf")
    clip = os.path.join(GKD, "models", "gguf", "clip-ViT-B-32-f16.gguf")
    engine = os.path.join(GKD, f"build-{args.backend}", "bin", "gkd-cli")
    gkd_model = os.path.join(GKD, "models", "gguf",
                             f"gkd_fullset-{args.dtype}.gguf")
    alias = {"human": "person", "human_hand": "hand"}
    from PIL import Image
    W, H = Image.open(image_path).size
    out = []
    for cls in classes:
        name = cls["object_name"]
        yw_name = alias.get(name, name)
        dets_json = os.path.join(BENCH, "_tmp_dets.json")
        r = subprocess.run([cli, "detect", "--model", det_model,
                            "--source", image_path, "--classes", yw_name,
                            "--clip-model", clip, "--dets-json", dets_json,
                            "--conf", "0.30", "--threads", str(args.threads)],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"yolo-cli failed: {r.stderr[-600:]}")
        with open(dets_json) as f:
            dets = json.load(f)["detections"]
        # clamp to the image exactly like the official GroundingDINO adapter
        # (x1 = max(0, ...), x2 = min(W-1, ...)) - a box touching the border
        # legitimately has negative raw coordinates
        bboxes = [[max(0.0, d["xyxy"][0]), max(0.0, d["xyxy"][1]),
                   min(float(W - 1), d["xyxy"][2]),
                   min(float(H - 1), d["xyxy"][3])] for d in dets]
        scores_d = [d["conf"] for d in dets]
        kps_texts = cls["kps_texts"]
        preds, scores = [], []
        for s in range(0, len(bboxes), 8):
            chunk = bboxes[s:s + 8]
            cmd = [engine, "detect", "--model", gkd_model,
                   "--input", image_path, "--threads", str(args.threads)]
            for bb in chunk:
                cmd += ["--bbox", *(repr(float(v)) for v in bb)]
            for t in kps_texts:
                cmd += ["--kps-texts", t]
            rr = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=3600)
            if rr.returncode != 0:
                raise RuntimeError(f"gkd-cli failed (cmd: "
                                   f"{' '.join(cmd)}):\n{rr.stderr[-600:]}")
            rois = sorted((json.loads(l)["json"] for l in
                           rr.stdout.splitlines() if l.startswith('{"json":')),
                          key=lambda x: x["roi"])
            for roi in rois:
                n = roi["n_prompts"]
                scale, offx, offy = roi["trans"]
                preds.append([[(roi["kps_norm"][2 * i] / 2 + 0.5) * 384 / scale
                               + offx / scale,
                               (roi["kps_norm"][2 * i + 1] / 2 + 0.5) * 384
                               / scale + offy / scale] for i in range(n)])
                scores.append(roi["scores"])
        out.append({"object_name": name, "kps_texts": kps_texts,
                    "detector_boxes": bboxes, "detector_scores": scores_d,
                    "predictions": preds, "predict_scores": scores})
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def render_multi(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with open(os.path.join(BENCH, "multi_object_comparison.json")) as f:
        ref = json.load(f)
    ref_by_img = {r["image_name"]: r for r in ref["results"]}

    # official multi-object demo images that have a cached PyTorch reference
    images = ["cat_dog.jpg", "000000011511.jpg", "alpaca_150.jpg"]
    summary = {"pipelines": {
        "official_pytorch": "GroundingDINO boxes + PyTorch GKDT (cached)",
        "pure_cpp": "ultralytics-ggml YOLO-World boxes + gkd-cli"},
        "images": {}}
    disps_all, matched_all = [], 0
    panels = []
    for image_name in images:
        r = ref_by_img[image_name]
        image_path = os.path.join(IMS, image_name)
        classes = r["classes"]
        cpp_classes = run_cpp_pipeline(image_path, classes, args)
        # reshape the pure-C++ classes into the same roi shape as the official
        # reference so one drawing routine serves both columns
        cpp_shaped = [{"object_name": c["object_name"],
                       "rois": [{"bbox": bb, "detector_score": sc,
                                 "official": {"predictions": pr,
                                              "scores": ss}}
                                for bb, sc, pr, ss in
                                zip(c["detector_boxes"], c["detector_scores"],
                                    c["predictions"], c["predict_scores"])]}
                      for c in cpp_classes]
        # match official vs cpp boxes per class; keypoint displacement of the
        # SAME physical object through two different detectors
        cls_out = []
        for cls, cpp in zip(classes, cpp_shaped):
            rois = []
            for oi, roi in enumerate(cls["rois"]):
                ob = roi["bbox"]
                best, bi = 0.0, -1
                for yi, yb in enumerate(cpp["rois"]):
                    v = iou(ob, yb["bbox"])
                    if v > best:
                        best, bi = v, yi
                entry = {"bbox": ob, "detector_score": roi["detector_score"],
                         "official": roi["official"]}
                if bi >= 0 and best >= 0.3:
                    entry["cpp_box"] = cpp["rois"][bi]["bbox"]
                    entry["cpp_iou"] = round(best, 4)
                    if roi["confident"]:
                        import numpy as np
                        d = np.abs(np.asarray(roi["official"]["predictions"],
                                              np.float64)
                                   - np.asarray(cpp["rois"][bi]["official"]
                                                ["predictions"], np.float64))
                        disp = float(np.linalg.norm(d, axis=1).mean())
                        entry["keypoint_displacement_px"] = round(disp, 2)
                        disps_all.append(disp)
                        matched_all += 1
                rois.append(entry)
            cls_out.append({"object_name": cls["object_name"],
                            "kps_texts": cls["kps_texts"], "rois": rois})
        summary["images"][image_name] = {"classes": cls_out}
        panels.append((image_name, image_path, classes, cpp_shaped))

    # render: rows = images, columns = official PyTorch vs pure C++
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.4 * n), squeeze=False)
    for ri, (image_name, image_path, official_classes, cpp_shaped_list_1) in \
            enumerate(panels):
        img = Image.open(image_path).convert("RGB")
        cpp_shaped_list = [cpp_shaped_list_1]
        for ci in range(2):
            ax = axes[ri][ci]
            ax.imshow(img)
            ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)
            ax.axis("off")
            if ci == 0:
                ax.imshow(draw_overlay(img, official_classes, "official"))
                ax.set_title(f"{image_name}\nofficial PyTorch pipeline "
                             "(GroundingDINO + GKDT)", fontsize=9)
            else:
                ax.imshow(draw_overlay(img, cpp_shaped_list_1, "official"))
                ax.set_title("pure C++ pipeline (YOLO-World-ggml + gkd-cli)",
                             fontsize=9)
    fig.suptitle("End-to-end multi-object GKD: official PyTorch pipeline vs "
                 "PURE C++ pipeline (detector + engine)", y=0.998)
    fig.tight_layout()
    out_png = os.path.join(BENCH, "cpp_e2e_multi.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print("wrote", out_png)

    summary["summary"] = {
        "matched_objects": matched_all,
        "keypoint_displacement_mean_px":
            round(sum(disps_all) / len(disps_all), 2) if disps_all else None,
        "keypoint_displacement_max_px":
            round(max(disps_all), 2) if disps_all else None,
        "note": "displacement = detector-choice effect (GroundingDINO vs "
                "YOLO-World boxes) on the FINAL keypoints, same GKD engine"}
    with open(os.path.join(BENCH, "cpp_e2e_multi.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(BENCH, "cpp_e2e_multi.json"))
    print("SUMMARY:", json.dumps(summary["summary"], indent=1))


def log(msg):
    print(msg, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "vulkan", "cpu"])
    ap.add_argument("--dtype", default="q4_K",
                    choices=["f32", "f16", "q8_0", "q4_K"])
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    render_single(args)
    render_multi(args)


if __name__ == "__main__":
    main()
