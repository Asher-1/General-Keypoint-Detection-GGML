#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the official GroundingDINO detector over the official multi-object test
images (the exact --obj-type sets of test_real_world/scripts/eval_multi_obj_gkd.sh)
and cache the detections for reproducible comparisons.

Output: cpp_ggml/benchmarks/detector_bboxes.json
    {"<image>": {"classes": [{"object_name": n, "entries": [{bbox, score}...]}]}}

Usage (torch env): python cpp_ggml/scripts/run_detector_official.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

# the official multi-object demo matrix (eval_multi_obj_gkd.sh)
OFFICIAL_CASES = [
    ("alpaca_150.jpg", ["alpaca"]),
    ("cat_dog.jpg", ["cat", "dog"]),
    ("000000011511.jpg", ["human"]),
    ("wash_dishes_egocentric.jpg", ["human_hand"]),
    ("car_penn2_0_1931.jpg", ["car", "bus", "truck"]),
    ("pigs_stock_farming.jpg", ["pig"]),
    ("pet_birds.jpg", ["bird"]),
    ("fish_swim.jpg", ["fish"]),
]


def main():
    from test_real_world.object_detector_lib.grounding_dino_detector import GroundingDINODetector

    out_path = os.path.join(HERE, "..", "benchmarks", "detector_bboxes.json")
    t0 = time.time()
    det = GroundingDINODetector()
    print(f"GroundingDINO init: {time.time() - t0:.1f}s")

    result = {}
    for image, names in OFFICIAL_CASES:
        path = os.path.join(ROOT, "test_real_world", "ims1", image)
        classes = []
        for name in names:
            t0 = time.time()
            entries = det.detect(path, name)
            print(f"  {image} / {name}: {len(entries)} box(es) ({time.time() - t0:.2f}s)")
            classes.append({"object_name": name,
                            "entries": [{"bbox": e["bbox"], "score": e["score"]}
                                        for e in entries]})
        result[image] = {"classes": classes}

    with open(out_path, "w") as f:
        json.dump({"detector": "groundingdino_swint_ogc (official adapter, box_threshold=0.30)",
                   "images": result}, f, indent=2)
    print("wrote", os.path.normpath(out_path))


if __name__ == "__main__":
    main()
