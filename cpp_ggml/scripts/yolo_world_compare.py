#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation: can an open-vocabulary detector OTHER than the official
GroundingDINO fill the ROI role of the multi-object GKD pipeline?

Protocol (answers it with measurements, not assumptions):
  1. YOLO-World (yolov8x-worldv2, offline vocabulary = the same class names)
     detects on the official multi-object images.
  2. Box quality vs the official GroundingDINO boxes (benchmarks/detector_bboxes.json):
     per class-name IoU matching (GreedyIoU), recall at IoU>=0.5, extra/missing.
  3. End-to-end effect: both box sets go through the SAME C++ GKD engine; the
     per-object keypoint displacement (original-image px) quantifies how much
     the detector choice moves the final keypoints.

Runs with the SYSTEM python (ultralytics); the GKD stage is the C++ gkd-cli
(no Python involved on that side).

Outputs:
    benchmarks/yolo_world_detections.json
    benchmarks/detector_ablation.png
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GKD = os.path.dirname(HERE)
IMS = os.path.join(ROOT, "test_real_world", "ims1")

# the official demo matrix (same class names as eval_multi_obj_gkd.sh)
CASES = [
    ("alpaca_150.jpg", ["alpaca"]),
    ("cat_dog.jpg", ["cat", "dog"]),
    ("000000011511.jpg", ["human"]),
    ("wash_dishes_egocentric.jpg", ["human_hand"]),
    ("car_penn2_0_1931.jpg", ["car", "bus", "truck"]),
    ("pigs_stock_farming.jpg", ["pig"]),
    ("pet_birds.jpg", ["bird"]),
    ("fish_swim.jpg", ["fish"]),
]
# GroundingDINO's text prompt for hands is "human_hand"; YOLO-World's CLIP
# vocabulary understands plain English better - use the official alias set
# used by the LocateAnything/visual prompts of the official demo.
YW_NAME_ALIAS = {"human": "person", "human_hand": "hand", "pig": "pig",
                 "bird": "bird", "fish": "fish", "cat": "cat", "dog": "dog",
                 "car": "car", "bus": "bus", "truck": "truck",
                 "alpaca": "alpaca"}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gd_boxes, yw_boxes, iou_thr=0.5):
    """Greedy IoU matching of GroundingDINO boxes against YOLO-World boxes."""
    pairs = sorted(((iou(g, y), gi, yi)
                    for gi, g in enumerate(gd_boxes)
                    for yi, y in enumerate(yw_boxes)), reverse=True)
    used_g, used_y, out = set(), set(), []
    for v, gi, yi in pairs:
        if v <= 0 or gi in used_g or yi in used_y:
            continue
        used_g.add(gi); used_y.add(yi)
        out.append((gi, yi, v))
    return out, [gi for gi in range(len(gd_boxes)) if gi not in used_g], \
        [yi for yi in range(len(yw_boxes)) if yi not in used_y]


def run_yolo_world(model, image_path, names, conf=0.30):
    import numpy as np
    out = {}
    for name in names:
        yw_name = YW_NAME_ALIAS.get(name, name)
        model.set_classes([yw_name])
        r = model.predict(image_path, conf=conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        order = np.argsort(-scores)
        out[name] = [{"bbox": [float(v) for v in boxes[i]], "score": float(scores[i])}
                     for i in order]
    return out


def run_gkd(cli, model_gguf, image_path, kps_texts, bboxes, threads):
    cmd = [cli, "detect", "--model", model_gguf, "--input", image_path,
           "--threads", str(threads)]
    for bb in bboxes:
        cmd += ["--bbox", *(repr(float(v)) for v in bb)]
    for t in kps_texts:
        cmd += ["--kps-texts", t]
    preds, scores = [], []
    for s in range(0, len(bboxes), 8):  # batch like multi_object_comparison
        chunk = bboxes[s:s + 8]
        if s:
            cmd = cmd[:cmd.index("--kps-texts")] if "--kps-texts" in cmd else cmd[:6]
            # rebuild: base + bboxes chunk + texts (cmd layout: cli, detect,
            # --model, m, --input, img, --threads, t, then --bbox*N, then texts)
        cmd = [cli, "detect", "--model", model_gguf, "--input", image_path,
               "--threads", str(threads)]
        for bb in chunk:
            cmd += ["--bbox", *(repr(float(v)) for v in bb)]
        for t in kps_texts:
            cmd += ["--kps-texts", t]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"gkd-cli failed:\n{r.stderr[-1200:]}")
        rois = sorted((json.loads(l)["json"] for l in r.stdout.splitlines()
                       if l.startswith('{"json":')), key=lambda x: x["roi"])
        for roi in rois:
            n = roi["n_prompts"]
            scale, offx, offy = roi["trans"]
            preds.append([[(roi["kps_norm"][2 * i] / 2 + 0.5) * 384 / scale + offx / scale,
                           (roi["kps_norm"][2 * i + 1] / 2 + 0.5) * 384 / scale + offy / scale]
                          for i in range(n)])
            scores.append(roi["scores"])
    return preds, scores


def detect_stage():
    """Stage 1: YOLO-World detections only (process exits afterwards so the
    GPU memory is released before the GKD engine allocates)."""
    sys.path.insert(0, ROOT)
    from ultralytics import YOLO
    model = YOLO("yolov8x-worldv2.pt")  # auto-downloads (~180 MB)
    out = {}
    for image, names in CASES:
        path = os.path.join(IMS, image)
        out[image] = run_yolo_world(model, path, names)
        for name in names:
            print(f"  {image} {name}: {len(out[image][name])} box(es)", flush=True)
    with open(os.path.join(GKD, "benchmarks", "yolo_world_detections.json"), "w") as f:
        json.dump({"detector": "yolov8x-worldv2 offline vocabulary, conf>=0.30",
                   "detections": out}, f, indent=2)
    print("wrote yolo_world_detections.json")


def gkd_stage(args):
    sys.path.insert(0, ROOT)
    from test_real_world.predefined_keypoints import get_predefined_keypoints

    with open(os.path.join(GKD, "benchmarks", "detector_bboxes.json")) as f:
        gd = json.load(f)["images"]
    with open(os.path.join(GKD, "benchmarks", "yolo_world_detections.json")) as f:
        yw_all = json.load(f)["detections"]

    cli = None
    for b in ("cuda", "vulkan", "cpu"):
        p = os.path.join(GKD, f"build-{b}", "bin", "gkd-cli")
        if os.path.exists(p):
            cli = p
            break
    model_gguf = os.path.join(GKD, "models", "gguf", f"gkd_fullset-{args.dtype}.gguf")

    result = {"detector_a": "groundingdino_swint_ogc (official adapter, conf>=0.30)",
              "detector_b": "yolov8x-worldv2 (offline vocabulary, conf>=0.30)",
              "gkd": "C++ gkd-cli, gkd_fullset-q4_K, identical for both box sets",
              "images": {}}
    n_match = n_gd_total = n_yw_total = 0
    ious_all = []
    disps_all = []

    for image, names in CASES:
        path = os.path.join(IMS, image)
        yw = yw_all[image]
        img_res = {"classes": {}}
        for name in names:
            gd_entries = gd[image]["classes"]
            gd_boxes = [e["bbox"] for c in gd_entries
                        if c["object_name"] == name for e in c["entries"]]
            yw_boxes = [e["bbox"] for e in yw[name]]
            pairs, gd_miss, yw_extra = match(gd_boxes, yw_boxes)
            n_match += len(pairs)
            n_gd_total += len(gd_boxes)
            n_yw_total += len(yw_boxes)
            ious_all.extend(v for _, _, v in pairs)

            schema = get_predefined_keypoints(name)
            kps_texts = schema["keypoints"] if schema else []
            kp = {}
            if kps_texts:
                kp["gd"], kp["gd_scores"] = run_gkd(cli, model_gguf, path,
                                                    kps_texts, gd_boxes, 16)
                if yw_boxes:
                    kp["yw"], kp["yw_scores"] = run_gkd(cli, model_gguf, path,
                                                        kps_texts, yw_boxes, 16)
            # matched-object keypoint displacement (same physical object)
            disps = []
            for gi, yi, v in pairs:
                if kps_texts:
                    import numpy as np
                    d = np.abs(np.asarray(kp["gd"][gi], np.float64) -
                               np.asarray(kp["yw"][yi], np.float64))
                    disps.append(float(np.linalg.norm(d, axis=1).mean()))
            disps_all.extend(disps)

            img_res["classes"][name] = {
                "gd_boxes": gd_boxes,
                "yw_boxes": yw_boxes,
                "matched": [(gi, yi, round(v, 4)) for gi, yi, v in pairs],
                "gd_recall_at_iou50": len(pairs) / len(gd_boxes) if gd_boxes else None,
                "matched_iou": [round(v, 4) for _, _, v in pairs],
                "keypoint_mean_displacement_px": disps,
                "gkd_keypoints": kp,
            }
            print(f"  {image} {name}: GD={len(gd_boxes)} YW={len(yw_boxes)} "
                  f"matched@IoU50={len(pairs)} "
                  f"meanIoU={sum(v for _,_,v in pairs)/len(pairs):.3f}" if pairs else
                  f"  {image} {name}: GD={len(gd_boxes)} YW={len(yw_boxes)} matched=0",
                  flush=True)
        result["images"][image] = img_res

    result["summary"] = {
        "gd_boxes": n_gd_total, "yw_boxes": n_yw_total,
        "matched_at_iou50": n_match,
        "gd_recall_at_iou50": round(n_match / n_gd_total, 4) if n_gd_total else None,
        "matched_iou_mean": round(sum(ious_all) / len(ious_all), 4) if ious_all else None,
        "matched_objects_with_kps": len(disps_all),
        "keypoint_displacement_mean_px": round(sum(disps_all) / len(disps_all), 2)
            if disps_all else None,
        "keypoint_displacement_max_px": round(max(disps_all), 2) if disps_all else None,
    }
    out = os.path.join(GKD, "benchmarks", "yolo_world_ablation.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)
    print("SUMMARY:", json.dumps(result["summary"], indent=1))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="q4_K", choices=["f32", "f16", "q8_0", "q4_K"],
                    help="GKD engine dtype for both box sets")
    ap.add_argument("--stage", default="all", choices=["detect", "gkd", "all"],
                    help="detect: YOLO-World only (process exits to free GPU "
                         "memory); gkd: engine passes; all: run both stages")
    args = ap.parse_args()
    if args.stage in ("detect", "all"):
        detect_stage()
        if args.stage == "detect":
            return
    gkd_stage(args)


if __name__ == "__main__":
    main()
