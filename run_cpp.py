#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-click PURE C++ end-to-end runner: detector + keypoint engine, no Python
at inference time.

Stages (each reuses its previous outputs, so a second run is fast):

  env         detect toolchains (cmake, compiler, CUDA / Vulkan) and pick the
              fastest usable backend
  submodules  git submodule update --init --recursive (ultralytics-ggml and
              its nested ggml)
  build       yolo-cli (detector) and gkd-cli (GKDT engine) per usable backend
  models      GKDT GGUF: download from HuggingFace, or convert the official
              checkpoint (one-time torch use); detector GGUFs (YOLO-World +
              CLIP text encoder) converted with the submodule's converters
              (one-time torch use)
  infer       single-object GKD in all three prompt modes (rendered images +
              JSON) AND multi-object GKD through the pure-C++ detector
  summary     list of everything produced

Examples:
    python3 run_cpp.py --help
    python3 run_cpp.py                      # everything, auto-detected
    python3 run_cpp.py --stage infer        # only the inference stage
    python3 run_cpp.py --backend vulkan --dtype f16 --image my.jpg \\
        --obj-type 'cat, dog'
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
GKD = os.path.join(ROOT, "cpp_ggml")
UW = os.path.join(GKD, "third_party", "ultralytics-ggml")
GGUF_DIR = os.path.join(GKD, "models", "gguf")
DEMO_IMAGE = os.path.join(ROOT, "test_real_world", "ims1", "2007_007524.jpg")
DEMO_MULTI = os.path.join(ROOT, "test_real_world", "ims1", "cat_dog.jpg")
DEMO_OBJTYPE = "cat, dog"
HF_REPO = "Asher-1/GKD_GGUF"
BACKENDS = ["cuda", "vulkan", "cpu"]  # fastest first
DTYPES = ["f32", "f16", "q8_0", "q4_K"]

RESULTS = []


def log(msg):
    print(msg, flush=True)


def run(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def record(stage, sub, status, detail=""):
    RESULTS.append((stage, sub, status, detail))
    log(f"  [{stage}/{sub}] {status}  {detail}")


def have_cuda_toolchain():
    return (shutil.which("nvcc") is not None
            or os.path.isdir("/usr/local/cuda"))


def have_vulkan_sdk():
    return shutil.which("glslc") is not None or \
        os.path.isdir(os.path.expanduser("~/VulkanSDK"))


def find_torch_python():
    """Any interpreter with torch (converter one-time use; CPU torch suffices
    for the GGUF converters)."""
    cands = [sys.executable]
    for base in ("/home/*/anaconda3", "/home/*/miniconda3", "/home/*/miniforge3",
                 "/opt/conda"):
        cands += sorted(glob.glob(os.path.join(base, "envs", "*", "bin", "python")))
    for py in cands:
        try:
            r = subprocess.run([py, "-c", "import torch"], capture_output=True,
                               timeout=120)
            if r.returncode == 0:
                return py
        except Exception:
            pass
    return None


def gguf_ok(dtype):
    return os.path.exists(os.path.join(GGUF_DIR, f"gkd_fullset-{dtype}.gguf"))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_env(args):
    log("[env] toolchains")
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("error: cmake is required (apt install cmake)")
    record("env", "cmake", "OK", subprocess.run([cmake, "--version"],
           capture_output=True, text=True).stdout.splitlines()[0])
    cxx = shutil.which("g++") or shutil.which("clang++")
    record("env", "cxx", "OK" if cxx else "MISSING", cxx or "install g++")
    if have_cuda_toolchain():
        record("env", "cuda", "OK")
    else:
        record("env", "cuda", "SKIP", "no nvcc/CUDA toolkit - CPU/Vulkan only")
    if have_vulkan_sdk():
        record("env", "vulkan", "OK")
    else:
        record("env", "vulkan", "SKIP", "no Vulkan SDK - CUDA/CPU only")
    if not cxx:
        sys.exit("error: no C++ compiler found")


def usable_backends(args):
    order = [args.backend] if args.backend else BACKENDS
    out = []
    for b in order:
        if b == "cuda" and not have_cuda_toolchain():
            continue
        if b == "vulkan" and not have_vulkan_sdk():
            continue
        out.append(b)
    return out or ["cpu"]


def stage_submodules(args):
    log("[submodules] ultralytics-ggml (+ nested ggml)")
    r = run(["git", "submodule", "update", "--init", "--recursive",
             os.path.relpath(UW, ROOT)])
    if r.returncode != 0:
        sys.exit(f"error: submodule init failed:\n{r.stderr[-1500:]}")
    head = run(["git", "-C", UW, "log", "--oneline", "-1"]).stdout.strip()
    record("submodules", "ultralytics-ggml", "OK", head)
    ver = run(["git", "-C", os.path.join(UW, "cpp_ggml", "third_party", "ggml"),
               "describe", "--tags"]).stdout.strip()
    record("submodules", "ggml (nested)", "OK", ver)


def stage_build(args):
    backends = usable_backends(args)
    # 1) GKDT engine
    for b in backends:
        bdir = os.path.join(GKD, f"build-{b}")
        cmake_flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGKD_GGML_BUILD_TESTS=ON"]
        if b == "cuda":
            cmake_flags += ["-DGKD_GGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=89"]
        if b == "vulkan":
            cmake_flags += ["-DGKD_GGML_VULKAN=ON"]
        env = dict(os.environ, CUDA_PATH="/usr/local/cuda-12.6") if b == "cuda" \
            else dict(os.environ)
        log(f"[build] gkd-cli ({b})")
        r = run(["cmake", "-S", GKD, "-B", bdir] + cmake_flags, timeout=900)
        r2 = run(["cmake", "--build", bdir, "-j", str(args.jobs)], timeout=7200)
        ok = os.path.exists(os.path.join(bdir, "bin", "gkd-cli"))
        record("build", f"gkd-cli-{b}", "OK" if ok else "FAIL",
               "" if ok else (r.stderr or r2.stderr)[-400:])
    # 2) detector (build once per checkout; CUDA flavor preferred, else CPU)
    det_bin = os.path.join(UW, "cpp_ggml", "build-cuda", "bin", "yolo-cli")
    if not os.path.exists(det_bin):
        for b in ("cuda", "cpu"):
            flags = ["-DYOLO_GGML_CUDA=ON"] if b == "cuda" else []
            env = dict(os.environ, CUDA_PATH="/usr/local/cuda-12.6") if b == "cuda" \
                else dict(os.environ)
            log(f"[build] yolo-cli ({b})")
            r = run(["cmake", "-S", os.path.join(UW, "cpp_ggml"),
                     "-B", os.path.join(UW, "cpp_ggml", f"build-{b}"),
                     "-DCMAKE_BUILD_TYPE=Release"] + flags, timeout=900, )
            r = run(["cmake", "--build", os.path.join(UW, "cpp_ggml",
                     f"build-{b}"), "-j", str(args.jobs)], timeout=7200)
            if os.path.exists(os.path.join(UW, "cpp_ggml", f"build-{b}",
                                           "bin", "yolo-cli")):
                det_bin = os.path.join(UW, "cpp_ggml", f"build-{b}", "bin",
                                       "yolo-cli")
                break
    record("build", "yolo-cli", "OK" if os.path.exists(det_bin) else "FAIL",
           det_bin if os.path.exists(det_bin) else "build failed")


def stage_models(args):
    log("[models] weights")
    # 1) GKDT GGUF: HF download first, else convert from the checkpoint
    py = find_torch_python()
    for dtype in ([args.dtype] if args.dtype else DTYPES):
        if gguf_ok(dtype):
            record("models", f"gkd_fullset-{dtype}", "OK", "already present")
            continue
        hf = shutil.which("huggingface-cli") or shutil.which("hf")
        if hf:
            log(f"[models] downloading gkd_fullset-{dtype}.gguf from {HF_REPO}")
            r = run([hf, "download", HF_REPO, f"gkd_fullset-{dtype}.gguf",
                     "--local-dir", GGUF_DIR], timeout=7200)
            if r.returncode == 0 and gguf_ok(dtype):
                record("models", f"gkd_fullset-{dtype}", "OK", "downloaded")
                continue
        ckpt = os.path.join(GKD, "models", "pytorch", "gkd_fullset.best")
        if py and os.path.exists(ckpt):
            log(f"[models] converting gkd_fullset-{dtype} from the checkpoint")
            r = run([py, os.path.join(GKD, "scripts", "convert_gkd_to_gguf.py"),
                     "--checkpoint", ckpt, "--dtype", dtype], timeout=7200)
            if gguf_ok(dtype):
                record("models", f"gkd_fullset-{dtype}", "OK", "converted")
                continue
        record("models", f"gkd_fullset-{dtype}", "FAIL",
               f"get it from https://huggingface.co/{HF_REPO}")
    # 2) detector GGUFs (one-time torch use via the submodule's converters)
    uw_scripts = os.path.join(UW, "cpp_ggml", "scripts")
    targets = [
        ("yolov8x-worldv2-f16.gguf",
         [py, os.path.join(uw_scripts, "convert_yolo_to_gguf.py"),
          "--model", "yolov8x-worldv2", "--dtype", "f16",
          "--output", os.path.join(GGUF_DIR, "yolov8x-worldv2-f16.gguf")],
         "yolov8x-worldv2.pt (ultralytics auto-download)"),
        ("clip-ViT-B-32-f16.gguf",
         [py, os.path.join(uw_scripts, "convert_clip_to_gguf.py"),
          "--model", "ViT-B/32", "--dtype", "f16",
          "--output", os.path.join(GGUF_DIR, "clip-ViT-B-32-f16.gguf")],
         "CLIP ViT-B/32 weights (openai clip auto-download)"),
    ]
    for name, cmd, src_hint in targets:
        if os.path.exists(os.path.join(GGUF_DIR, name)):
            record("models", name, "OK", "already present")
            continue
        if not py:
            record("models", name, "SKIP",
                   f"no torch python to run the one-time converter; get {name} "
                   "or the source weights")
            continue
        log(f"[models] converting {name}")
        r = run([c if i else c for i, c in enumerate(cmd)], timeout=7200)
        ok = os.path.exists(os.path.join(GGUF_DIR, name))
        record("models", name, "OK" if ok else "FAIL",
               "" if ok else f"converter needs: {src_hint}; {r.stderr[-300:]}")


def stage_infer(args, imdir):
    log("[infer] single-object (3 prompt modes) + multi-object")
    cli, backend = None, None
    for b in usable_backends(args):
        p = os.path.join(GKD, f"build-{b}", "bin", "gkd-cli")
        if os.path.exists(p):
            cli, backend = p, b
            break
    if not cli:
        record("infer", "engine", "FAIL", "no gkd-cli")
        return
    dtype = args.dtype or next((d for d in DTYPES if gguf_ok(d)), None)
    model = os.path.join(GGUF_DIR, f"gkd_fullset-{dtype}.gguf")
    if not os.path.exists(model):
        record("infer", "engine", "SKIP", "no GKD GGUF available")
        return
    image = args.image or DEMO_IMAGE
    if not os.path.isfile(image):
        record("infer", "demo", "SKIP", f"missing {image}")
        return
    out_dir = args.out_dir or os.path.join(ROOT, "output", "cpp_e2e")
    os.makedirs(out_dir, exist_ok=True)

    # single-object: 3 prompt modes, rendered
    for mode in ["text", "visual", "multimodal"]:
        cmd = [cli, "detect", "--model", model, "--input", image,
               "--threads", str(args.threads), "--out",
               os.path.join(out_dir, f"single_{mode}.jpg")]
        for v in args.bbox:
            cmd += ["--bbox", repr(float(v))]
        texts = list(args.kps_texts)
        if mode in ("visual", "multimodal"):
            cmd += ["--support-image", args.support_im]
            for v in args.support_kps:
                cmd += ["--support-kps", repr(float(v))]
        if mode == "multimodal":
            texts = texts[:len(args.support_kps) // 2]  # official N_t == N_v
        if mode == "text":
            for t in texts:
                cmd += ["--kps-texts", t]
        elif mode == "multimodal":
            for t in texts:
                cmd += ["--kps-texts", t]
        r = run(cmd, timeout=3600)
        j = next((json.loads(l)["json"] for l in r.stdout.splitlines()
                  if l.startswith('{"json":')), None)
        if j:
            record("infer", f"single-{mode}", "OK",
                   f"scores={[round(s, 3) for s in j['scores']]} -> "
                   f"single_{mode}.jpg")
        else:
            record("infer", f"single-{mode}", "FAIL", r.stderr[-300:])

    # multi-object: pure-C++ detector (yolo-cli) -> GKD engine
    multi_image = args.multi_image or DEMO_MULTI
    obj_type = args.obj_type
    r = run([sys.executable, os.path.join(ROOT, "run_multi_object.py"),
             "--input", multi_image, "--obj-type", obj_type,
             "--object-detector", "yolo-world-ggml",
             "--backend", backend, "--dtype", dtype,
             "--threads", str(args.threads)], timeout=7200)
    out_json = os.path.splitext(multi_image)[0] + "_gkd_multi.json"
    if r.returncode == 0 and os.path.exists(out_json):
        with open(out_json) as f:
            res = json.load(f)
        n_roi = sum(len(c["predictions"]) for c in res["classes"])
        record("infer", "multi-object", "OK",
               f"{n_roi} objects via yolo-world-ggml -> {out_json}")
    else:
        record("infer", "multi-object", "FAIL", r.stderr[-400:])


def main():
    global ROOT, GKD, UW, GGUF_DIR, DEMO_IMAGE, DEMO_MULTI, IMS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "env", "submodules", "build", "models",
                             "infer"],
                    help="run everything (default) or a single stage")
    ap.add_argument("--backend", default="", choices=[""] + BACKENDS,
                    help="force a backend (default: fastest usable)")
    ap.add_argument("--dtype", default="q4_K", choices=DTYPES,
                    help="GKD GGUF dtype to build/fetch/infer")
    ap.add_argument("--image", default="", help="single-object query image "
                    "(default: test_real_world/ims1/2007_007524.jpg)")
    ap.add_argument("--bbox", nargs="*", type=float, default=[],
                    help="ROI box x1 y1 x2 y2 for the single-object run")
    ap.add_argument("--kps-texts", nargs="*",
                    default=["nose", "left eye", "right eye", "left ear",
                             "right ear"],
                    help="text prompts (official cat demo set by default)")
    ap.add_argument("--support-im", default="",
                    help="1-shot visual-prompt image (default: the official "
                         "demo support image)")
    ap.add_argument("--support-kps", nargs="*", type=float,
                    default=[343.0, 166.0, 281.0, 158.0, 311.0, 197.0],
                    help="support keypoints for visual/multimodal prompts")
    ap.add_argument("--multi-image", default="",
                    help="multi-object image (default: cat_dog.jpg)")
    ap.add_argument("--obj-type", default=DEMO_OBJTYPE,
                    help="comma-separated object names for the detector")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8,
                    help="parallel build jobs")
    ap.add_argument("--out-dir", default="", help="inference output directory "
                    "(default: output/cpp_e2e)")
    ap.add_argument("--root", default=HERE, help="repository root (auto-derived)")
    args = ap.parse_args()

    global ROOT, GKD, UW, GGUF_DIR, DEMO_IMAGE, DEMO_MULTI, IMS
    ROOT = os.path.abspath(args.root)
    GKD = os.path.join(ROOT, "cpp_ggml")
    UW = os.path.join(GKD, "third_party", "ultralytics-ggml")
    GGUF_DIR = os.path.join(GKD, "models", "gguf")
    DEMO_IMAGE = os.path.join(ROOT, "test_real_world", "ims1", "2007_007524.jpg")
    DEMO_MULTI = os.path.join(ROOT, "test_real_world", "ims1", "cat_dog.jpg")
    IMS = os.path.join(ROOT, "test_real_world", "ims1")

    # lazy defaults (argparse defaults are resolved after --root re-anchoring)
    if not args.support_im:
        args.support_im = os.path.join(IMS, "2007_003778.jpg")
    if not args.multi_image:
        args.multi_image = os.path.join(IMS, "cat_dog.jpg")
    if not args.image:
        args.image = os.path.join(IMS, "2007_007524.jpg")

    stages = ["env", "submodules", "build", "models", "infer"] \
        if args.stage == "all" else [args.stage]
    imdir = os.path.join(ROOT, "test_real_world", "ims1")
    for s in stages:
        {"env": lambda: stage_env(args),
         "submodules": lambda: stage_submodules(args),
         "build": lambda: stage_build(args),
         "models": lambda: stage_models(args),
         "infer": lambda: stage_infer(args, imdir)}[s]()
        if s == "submodules":
            # the build may need the nested ggml re-checked after init
            pass

    log("\n" + "=" * 70)
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    for stage, sub, status, detail in RESULTS:
        log(f"  {stage}/{sub:26s} {status:6s} {detail[:90]}")
    log("-" * 70)
    log(f"FAIL count: {len(fails)}"
        + (f"  <- {'; '.join(f'{s}/{u}' for s, u, _, _ in fails)}" if fails else ""))
    log(f"engine   : {GKD}/build-*/bin/gkd-cli")
    log(f"detector : {UW}/cpp_ggml/build-*/bin/yolo-cli")
    log(f"outputs  : {args.out_dir or os.path.join(ROOT, 'output', 'cpp_e2e')}"
        " and <image>_gkd_multi.json")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
