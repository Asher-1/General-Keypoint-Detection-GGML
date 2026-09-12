#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official multi-object GKD pipeline with the C++ ggml engine as stage 2.

Mirrors test_real_world/multi_obj_gkd_inference.py exactly:

    stage 1  object detector (official code + weights, unchanged)
             GroundingDINO SwinT or LocateAnything-3B -> per-class bboxes
    stage 2  GKD per class - here the C++ ggml engine (gkd-cli) runs ALL ROIs
             of a class in one call with the class's resolved prompts
             (user prompts > official predefined per-class schema), identical
             to what demo() does for the class's bbox list
    stage 3  merge into the official result structure
             (per class: detection entries + N_bbox x N x 2 predictions +
             scores), written as JSON

Everything is derived from this script's location - no hardcoded paths. The
detector stage is optional: pass --bboxes-json to skip it (the detector emits
plain bbox coordinates, so any detector output can be injected), and with
neither --obj-type nor --bboxes-json the whole image becomes one ROI exactly
like the official script.

Examples:
    python3 run_multi_object.py --input test_real_world/ims1/cat_dog.jpg \\
        --obj-type 'cat, dog'
    python3 run_multi_object.py --input test_real_world/ims1/alpaca_150.jpg \\
        --obj-type alpaca --support-im test_real_world/ims1/alpaca_150.jpg \\
        --support-kps 615 495 483 493 521 549 --backend cuda --dtype q4_K
    python3 run_multi_object.py --input img.jpg --bboxes-json dets.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
GKD_DIR = os.path.join(ROOT, "cpp_ggml")
IMS_DIR = os.path.join(ROOT, "test_real_world", "ims1")

BACKENDS = ["cpu", "cuda", "vulkan"]
DTYPES = ["f32", "f16", "q8_0", "q4_K"]


def log(msg):
    print(msg, flush=True)


def pick_backend(backend):
    if backend:
        cli = os.path.join(GKD_DIR, f"build-{backend}", "bin", "gkd-cli")
        if not os.path.exists(cli):
            sys.exit(f"error: gkd-cli not found at {cli} (build it first, see cpp_ggml/README.md)")
        return cli
    for b in BACKENDS:  # fastest first
        cli = os.path.join(GKD_DIR, f"build-{b}", "bin", "gkd-cli")
        if os.path.exists(cli):
            return cli
    sys.exit("error: no gkd-cli build found - build one via 'cmake --preset cpu && "
             "cmake --build --preset cpu' inside cpp_ggml/ (see cpp_ggml/README.md)")


def pick_model(dtype):
    model = os.path.join(GKD_DIR, "models", "gguf", f"gkd_fullset-{dtype}.gguf")
    if not os.path.exists(model):
        sys.exit(f"error: model {model} not found - download from "
                 "https://huggingface.co/Asher-1/GKD_GGUF "
                 "(huggingface-cli download Asher-1/GKD_GGUF --local-dir cpp_ggml/models/gguf)")
    return model


def resolve_prompts(object_name, kps_texts, support_im, support_kps, skeleton):
    """Official get_prompt_info semantics: user input first, then the
    predefined per-class schema; assert the multimodal N_t == N_v rule."""
    sys.path.insert(0, ROOT)
    from test_real_world.predefined_keypoints import get_prompt_info
    return get_prompt_info(object_name, list(kps_texts), support_im,
                           list(support_kps), list(skeleton))


def run_detector(args):
    """Stage 1: the official detector code, unchanged. Returns
    detection_entries = [{'bbox': [x1,y1,x2,y2], 'score': s, 'object_name': n}]."""
    sys.path.insert(0, ROOT)
    if args.object_detector == "groundingdino":
        from test_real_world.object_detector_lib.grounding_dino_detector import GroundingDINODetector
        detector = GroundingDINODetector()
        entries = []
        for name in args.object_names:
            entries.extend(detector.detect(args.input, name))
        return entries
    from test_real_world.object_detector_lib.locateanything_detector import LocateAnythingDetector
    detector = LocateAnythingDetector()
    return detector.detect(args.input, args.object_names)


def run_detector_yolo_world_ggml(args):
    """The ultralytics-ggml submodule's yolo-cli: open-vocabulary YOLO-World
    detection fully in C++/ggml (no Python). Returns detection_entries."""
    repo = os.path.join(ROOT, "cpp_ggml", "third_party", "ultralytics-ggml")
    cli = os.path.join(repo, "cpp_ggml", "build-cuda", "bin", "yolo-cli")
    if not os.path.exists(cli):
        sys.exit("error: yolo-cli not found - build the submodule: "
                 "'git submodule update --init cpp_ggml/third_party/"
                 "ultralytics-ggml && (cd cpp_ggml/third_party/ultralytics-ggml/"
                 "cpp_ggml && cmake -B build-cuda -DYOLO_GGML_CUDA=ON && "
                 "cmake --build build-cuda -j)'")
    model = os.path.join(GKD_DIR, "models", "gguf", "yolov8x-worldv2-f16.gguf")
    clip = os.path.join(GKD_DIR, "models", "gguf", "clip-ViT-B-32-f16.gguf")
    for p in (model, clip):
        if not os.path.exists(p):
            sys.exit(f"error: detector weights missing: {p} (convert via "
                     "cpp_ggml/third_party/ultralytics-ggml/cpp_ggml/scripts/ "
                     "convert_yolo_to_gguf.py / convert_clip_to_gguf.py)")
    dets_json = os.path.join(os.path.dirname(os.path.abspath(args.input)), "_yolo_dets.json")
    names = [YW_NAME_ALIAS.get(n, n) for n in args.object_names]
    cmd = [cli, "detect", "--model", model, "--source", os.path.abspath(args.input),
           "--classes", ",".join(names), "--clip-model", clip,
           "--dets-json", dets_json, "--conf", "0.30", "--threads", str(args.threads)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        sys.exit(f"error: yolo-cli failed:\n{r.stderr[-1500:]}")
    with open(dets_json) as f:
        dets = json.load(f)
    entries = []
    for d in dets["detections"]:
        name = args.object_names[d["cls"]] if d["cls"] < len(args.object_names) \
            else args.object_names[0]
        entries.append({"bbox": [float(v) for v in d["xyxy"]],
                        "score": float(d["conf"]), "object_name": name})
    return entries


# GroundingDINO reads plain English; YOLO-World's CLIP vocabulary prefers the
# canonical COCO names for these official demo classes.
YW_NAME_ALIAS = {"human": "person", "human_hand": "hand"}


def load_bboxes_json(path):
    """External detector output: {'classes': [{'object_name': n,
    'bboxes': [[x1,y1,x2,y2],...], 'scores': [...]}, ...]}"""
    with open(path) as f:
        data = json.load(f)
    entries = []
    for cls in data.get("classes", []):
        name = cls.get("object_name", "object1")
        scores = cls.get("scores", [1.0] * len(cls.get("bboxes", [])))
        for bb, s in zip(cls.get("bboxes", []), scores):
            entries.append({"bbox": [float(v) for v in bb], "score": float(s),
                            "object_name": name})
    return entries


def run_gkd_class(cli, model, args, object_name, object_entries, kps_texts, skeleton):
    """Stage 2: one gkd-cli call for ALL ROIs of the class (the same shape as
    the official demo() call for one class)."""
    image = os.path.abspath(args.input)
    cmd = [cli, "detect", "--model", model, "--input", image,
           "--threads", str(args.threads)]
    for e in object_entries:
        cmd += ["--bbox", *(repr(float(v)) for v in e["bbox"])]
    for t in kps_texts:
        cmd += ["--kps-texts", t]
    if args.support_im:
        cmd += ["--support-image", os.path.abspath(args.support_im)]
        for v in args.support_kps:
            cmd += ["--support-kps", str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.exit(f"error: gkd-cli failed:\n{r.stderr[-2000:]}")
    rois = sorted((json.loads(l)["json"] for l in r.stdout.splitlines()
                   if l.startswith('{"json":')), key=lambda x: x["roi"])
    predictions, scores = [], []
    for roi in rois:
        n = roi["n_prompts"]
        scale, offx, offy = roi["trans"]
        xy = [[(roi["kps_norm"][2 * i] / 2 + 0.5) * 384 / scale + offx / scale,
               (roi["kps_norm"][2 * i + 1] / 2 + 0.5) * 384 / scale + offy / scale]
              for i in range(n)]
        predictions.append(xy)
        scores.append(roi["scores"])
    return predictions, scores


def main():
    global ROOT, GKD_DIR, IMS_DIR
    ap = argparse.ArgumentParser(
        description="Official multi-object GKD pipeline (official detector + "
                    "C++ ggml GKD stage). See module docstring for details.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default=os.path.join(IMS_DIR, "cat_dog.jpg"),
                    help="input image path")
    ap.add_argument("--obj-type", default="cat, dog",
                    help="comma-separated object names for the detector; "
                         "empty with --bboxes-json absent = whole image as one ROI")
    ap.add_argument("--object-detector", default="yolo-world-ggml",
                    choices=["yolo-world-ggml", "groundingdino", "locateanything"],
                    help="stage-1 detector: yolo-world-ggml = the pure C++ "
                         "ultralytics-ggml submodule (default, no Python); "
                         "groundingdino/locateanything = official Python adapters")
    ap.add_argument("--bboxes-json", default="",
                    help="skip the detector: JSON with per-class bboxes "
                         "({'classes': [{'object_name': n, 'bboxes': [[x1,y1,x2,y2],...]}]})")
    ap.add_argument("--kps-texts", nargs="*", default=[], help="keypoint texts "
                    "(omit to use the official predefined per-class schema)")
    ap.add_argument("--support-im", default="", help="1-shot support image (visual prompt)")
    ap.add_argument("--support-kps", nargs="*", type=float, default=[],
                    help="support keypoints x1 y1 x2 y2 ...")
    ap.add_argument("--skeleton", nargs="*", type=int, default=[],
                    help="1-based skeleton links for the output JSON")
    ap.add_argument("--backend", default="", choices=[""] + BACKENDS,
                    help="gkd-cli build to use (default: fastest available)")
    ap.add_argument("--dtype", default="q4_K", choices=DTYPES, help="GGUF dtype")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--out", default="", help="output JSON path "
                    "(default: <input>_gkd_multi.json next to the image)")
    ap.add_argument("--root", default=HERE, help="repository root (auto-derived)")
    args = ap.parse_args()

    # re-anchor paths when --root is overridden
    ROOT = os.path.abspath(args.root)
    GKD_DIR = os.path.join(ROOT, "cpp_ggml")
    IMS_DIR = os.path.join(ROOT, "test_real_world", "ims1")
    if not os.path.isfile(args.input):
        sys.exit(f"error: input image not found: {args.input}")

    args.object_names = [n.strip() for n in args.obj_type.split(",") if n.strip()]
    cli = pick_backend(args.backend)
    model = pick_model(args.dtype)

    # ---- stage 1: ROIs -----------------------------------------------------
    if args.bboxes_json:
        detection_entries = load_bboxes_json(args.bboxes_json)
        log(f"[detect] {len(detection_entries)} ROI(s) from {args.bboxes_json}")
    elif args.object_names and args.object_detector == "yolo-world-ggml":
        detection_entries = run_detector_yolo_world_ggml(args)
        if not detection_entries:
            sys.exit("error: YOLO-World found no objects for --obj-type "
                     f"'{args.obj_type}'")
        log(f"[detect] {len(detection_entries)} ROI(s) via yolo-world-ggml (C++)")
    elif args.object_names:
        detection_entries = run_detector(args)
        if not detection_entries:
            sys.exit("error: the detector found no objects for --obj-type "
                     f"'{args.obj_type}' (official behaviour)")
        log(f"[detect] {len(detection_entries)} ROI(s) via {args.object_detector}")
    else:
        detection_entries = []
        log("[detect] no --obj-type / --bboxes-json: whole image as one ROI")

    # ---- stage 2+3: GKD per class (C++ engine) + merge ---------------------
    from PIL import Image
    w, h = Image.open(args.input).size
    object_groups = args.object_names if args.object_names else ["object1"]
    results = {"image": os.path.abspath(args.input), "width": w, "height": h,
               "detector": args.bboxes_json and "external" or
                           (args.object_detector if args.object_names else "none"),
               "classes": []}
    for object_name in object_groups:
        entries = [e for e in detection_entries if e.get("object_name") == object_name]
        if args.object_names and not entries:
            log(f"[gkd] no detected {object_name}; skipping (official behaviour)")
            continue
        if not entries:  # official whole-image fallback entry
            entries = [{"bbox": [0.0, 0.0, float(w - 1), float(h - 1)],
                        "score": 1.0, "object_name": object_name}]
        kps_texts, skeleton, n_t, n_v = resolve_prompts(
            object_name, args.kps_texts, args.support_im, args.support_kps, args.skeleton)
        log(f"[gkd] class '{object_name}': {len(entries)} ROI(s), {n_t} text + "
            f"{n_v} visual prompts -> {cli}")
        predictions, scores = run_gkd_class(
            cli, model, args, object_name, entries, kps_texts, skeleton)
        results["classes"].append({
            "object_name": object_name,
            "entries": entries,
            "kps_texts": kps_texts,
            "skeleton": skeleton,
            "N_t": n_t,
            "N_v": n_v,
            "predictions": predictions,   # N_bbox x N x 2, original-image px
            "predict_score": scores,      # N_bbox x N
        })

    out = args.out or os.path.splitext(args.input)[0] + "_gkd_multi.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"[done] wrote {out}")
    for cls in results["classes"]:
        for i, (pred, sc) in enumerate(zip(cls["predictions"], cls["predict_score"])):
            log(f"  {cls['object_name']}#{i}: " +
                ", ".join(f"{p[0]:.1f},{p[1]:.1f} ({s:.2f})" for p, s in zip(pred, sc)))


if __name__ == "__main__":
    main()
