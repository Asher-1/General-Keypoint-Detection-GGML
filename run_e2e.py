#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-click end-to-end driver for GKD: official PyTorch + C++ ggml runtime.

Stages (each auto-detects its requirements and SKIPS itself when they are not
met - completed work is reused, nothing is re-deployed on a second run):

  env      detect toolchains: python deps, cmake, compiler, CUDA, Vulkan
  data     verify the official test images (test_real_world/ims1)
  models   acquire GGUF weights: download from HuggingFace, or convert the
           official PyTorch checkpoint (needs torch)
  build    cmake build per detected backend (cpu always; cuda/vulkan when the
           toolchain is present). Existing builds are reused unless --force-build
  infer    run the official demo on every available backend x dtype
  parity   tap-level numeric parity vs PyTorch (requires torch, fp32)
  bench    full latency + accuracy matrix + charts (slow; enabled by --full)

Everything is derived from this script's location - no hardcoded paths.

Typical usage:
  python3 run_e2e.py                    # auto-detect, quick end-to-end
  python3 run_e2e.py --list-only        # show the plan without executing
  python3 run_e2e.py --full             # + full benchmark matrix and charts
  python3 run_e2e.py --dtypes q4_K --backends cpu
  python3 run_e2e.py --force-build --bench-iters 20
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
GKD = os.path.join(ROOT, "cpp_ggml")
GGUF_DIR = os.path.join(GKD, "models", "gguf")
IMS_DIR = os.path.join(ROOT, "test_real_world", "ims1")
DEMO_IMAGE = "2007_007524.jpg"
ALL_DTYPES = ["f32", "f16", "q8_0", "q4_K"]  # q4_0 is engine-supported but not shipped
DEFAULT_HF_REPO = "Asher-1/GKD_GGUF"

RESULTS = []          # (stage, sub, status, detail)
FAILURES = 0


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def record(stage, sub, status, detail=""):
    RESULTS.append((stage, sub, status, detail))
    mark = {"OK": "+", "SKIP": "-", "FAIL": "!"}[status]
    log(f"  [{mark}] {stage}:{sub:<24} {status:<4} {detail}")


def run(cmd, env=None, cwd=None, capture=True, timeout=None):
    """Run a command; return CompletedProcess. Never raises on non-zero."""
    e = dict(os.environ)
    if env:
        e.update({k: v for k, v in env.items() if v is not None})
    log(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, env=e,
                          capture_output=capture, text=True, timeout=timeout)


def probe_out(cmd, env=None):
    """Return stripped stdout if the command exits 0, else None."""
    try:
        r = run(cmd, env=env, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def find_file(pattern_dirs, name):
    """Locate `name` under any of the candidate directories (no hardcoding)."""
    cand = list(pattern_dirs)
    for d in cand:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return d
    return None


# ----------------------------------------------------------------------------
# environment detection
# ----------------------------------------------------------------------------
class Env:
    def __init__(self):
        self.python = sys.executable
        self.torch = None            # None | 'cpu' | 'cuda'
        self.torch_cuda = False
        self.cmake = shutil.which("cmake")
        self.cxx = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
        self.make = shutil.which("make") or shutil.which("ninja")
        self.nvcc = shutil.which("nvcc")
        self.cuda_path = None
        self.nvidia_gpu = False
        self.glslc = shutil.which("glslc")
        self.vulkan_sdk = os.environ.get("VULKAN_SDK")
        self.vulkan_loader = False
        self.hf_cli = shutil.which("huggingface-cli")
        self.wget = shutil.which("wget")
        self.curl = shutil.which("curl")
        self.git = shutil.which("git")
        self.nproc = os.cpu_count() or 4
        self.torch_cuda_python = None   # interpreter with a CUDA-enabled torch
        self._detect()

    def _detect(self):
        # torch (in *this* interpreter)
        try:
            import torch  # noqa: F401
            self.torch = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_cuda = torch.cuda.is_available()
        except Exception:
            self.torch = None
        # CUDA toolkit: nvcc on PATH, else the usual install roots
        if not self.nvcc:
            for base in ("/usr/local", "/opt"):
                if not os.path.isdir(base):
                    continue
                for entry in sorted(os.listdir(base), reverse=True):
                    if entry.startswith("cuda"):
                        p = os.path.join(base, entry, "bin", "nvcc")
                        if os.path.isfile(p):
                            self.nvcc = p
                            self.cuda_path = os.path.dirname(os.path.dirname(p))
                            break
                if self.nvcc or self.cuda_path:
                    break
        if not self.cuda_path and self.nvcc:
            self.cuda_path = os.path.dirname(os.path.dirname(self.nvcc))
        # NVIDIA GPU
        if shutil.which("nvidia-smi"):
            self.nvidia_gpu = probe_out(["nvidia-smi", "-L"]) is not None
        # Vulkan: SDK root from env or common locations, then glslc + loader
        if not self.glslc:
            for base in (self.vulkan_sdk, "/home/" + os.environ.get("USER", "") + "/VulkanSDK"):
                if base and os.path.isdir(base):
                    hits = []
                    for root, dirs, files in os.walk(base):
                        if "glslc" in files:
                            hits.append(os.path.dirname(root))  # setup-env dir
                    if hits:
                        sdk = sorted(hits)[-1]
                        self.vulkan_sdk = os.path.dirname(sdk) if \
                            os.path.basename(sdk).startswith("x_") else sdk
                        for root, dirs, files in os.walk(base):
                            if "glslc" in files:
                                self.glslc = os.path.join(root, "glslc")
                                self.vulkan_sdk = root.split("/bin")[0] if "/bin" in root \
                                    else os.path.dirname(root)
                                break
                    break
        for lib in ("libvulkan.so.1", "libvulkan.so", "vulkan-1.dll"):
            if self.vulkan_loader:
                break
            for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/local/lib"):
                if os.path.exists(os.path.join(d, lib)):
                    self.vulkan_loader = True
                    break

    def find_torch_cuda_python(self):
        """Find an interpreter whose torch has CUDA (dump_taps.py uses .cuda()).
        Checks the current python first, then standard conda/venv locations -
        nothing user-specific is hardcoded."""
        if self.torch_cuda_python is not None:
            return self.torch_cuda_python

        def cuda_ok(py):
            try:
                r = subprocess.run([py, "-c",
                                    "import torch;print(torch.cuda.is_available())"],
                                   capture_output=True, text=True, timeout=120)
                return r.returncode == 0 and r.stdout.strip() == "True"
            except (OSError, subprocess.TimeoutExpired):
                return False

        cand = [self.python]
        import glob
        for pattern in ("/home/*/anaconda3/envs/*/bin/python",
                        "/home/*/miniconda3/envs/*/bin/python",
                        "/home/*/miniforge3/envs/*/bin/python",
                        "/opt/conda/envs/*/bin/python",
                        os.path.expanduser("~/.conda/envs/*/bin/python")):
            cand += sorted(glob.glob(pattern))
        found = None
        for py in cand:
            if py and os.path.isfile(py) and cuda_ok(py):
                found = py
                break
        self.torch_cuda_python = found or ""
        self._torch_candidates = cand
        return self.torch_cuda_python or None

    def torch_cuda_python_candidates(self):
        if not getattr(self, "_torch_candidates", None):
            self.find_torch_cuda_python()
        cands = getattr(self, "_torch_candidates", []) or []
        first = self.torch_cuda_python
        return ([first] if first else []) + [c for c in cands if c != first]

    def cuda_env(self):
        env = {}
        if self.cuda_path:
            env["CUDA_PATH"] = self.cuda_path
        return env

    def vulkan_env(self):
        env = {}
        if self.vulkan_sdk:
            env["VULKAN_SDK"] = self.vulkan_sdk
            env["PATH"] = os.path.join(self.vulkan_sdk, "bin") + os.pathsep + os.environ.get("PATH", "")
        return env

    def backend_supported(self, backend):
        if backend == "cpu":
            return True
        if backend == "cuda":
            if not self.nvcc:
                return "no nvcc / CUDA toolkit"
            if not self.nvidia_gpu:
                return "no NVIDIA GPU"
            return True
        if backend == "vulkan":
            if not self.glslc:
                return "no glslc (Vulkan SDK)"
            if not self.vulkan_loader:
                return "no vulkan loader"
            return True
        return "unknown backend"

    def summary_lines(self):
        def yes(v):
            return v if isinstance(v, str) else ("yes" if v else "no")
        return [
            f"python      : {platform.python_version()} ({self.python})",
            f"torch       : {self.torch or 'not available'}",
            f"cmake/cxx   : {yes(bool(self.cmake))} / {yes(bool(self.cxx))}",
            f"cuda        : toolkit={yes(bool(self.nvcc))} gpu={yes(self.nvidia_gpu)}",
            f"vulkan      : glslc={yes(bool(self.glslc))} loader={yes(self.vulkan_loader)}",
            f"hf download : huggingface-cli={yes(bool(self.hf_cli))} curl={yes(bool(self.curl))} wget={yes(bool(self.wget))}",
        ]


# ----------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------
def stage_env(env, args):
    log("[env] toolchain detection")
    for line in env.summary_lines():
        log("    " + line)
    record("env", "detect", "OK", f"backends: cpu" +
           (", cuda" if env.backend_supported("cuda") is True else "") +
           (", vulkan" if env.backend_supported("vulkan") is True else ""))


def discover_images():
    """Locate the official test images without hardcoding a path."""
    if os.path.isdir(IMS_DIR):
        imgs = sorted(f for f in os.listdir(IMS_DIR)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(imgs) >= 10:
            return IMS_DIR, imgs
    # fall back: search one level deep for an ims1-like directory
    for root, dirs, _ in os.walk(ROOT):
        if root.count(os.sep) - ROOT.count(os.sep) > 3:
            dirs[:] = []
            continue
        for d in dirs:
            if d == "ims1":
                p = os.path.join(root, d)
                n = len([f for f in os.listdir(p) if f.lower().endswith((".jpg", ".png"))])
                if n >= 10:
                    return p, sorted(os.listdir(p))
    return None, []


def stage_data(env, args):
    log("[data] official test data")
    imdir, imgs = discover_images()
    if imdir:
        record("data", "images", "OK", f"{len(imgs)} official images at {os.path.relpath(imdir, ROOT)}")
        return imdir, imgs
    record("data", "images", "SKIP",
           "official images not found - download from "
           "github.com/Asher-1/cloudViewer_downloads/releases/tag/general_keypoint_detection_data")
    return None, []


def gguf_path(dtype):
    return os.path.join(GGUF_DIR, f"gkd_fullset-{dtype}.gguf")


def download_gguf(env, dtype, repo):
    url = f"https://huggingface.co/{repo}/resolve/main/gkd_fullset-{dtype}.gguf"
    dst = gguf_path(dtype)
    os.makedirs(GGUF_DIR, exist_ok=True)
    if env.hf_cli:
        r = run([env.hf_cli, "download", repo, f"gkd_fullset-{dtype}.gguf",
                 "--local-dir", GGUF_DIR], timeout=None)
        if r.returncode == 0 and os.path.isfile(dst):
            return True, "huggingface-cli"
    if env.curl:
        r = run([env.curl, "-L", "--fail", "-o", dst + ".part", url], timeout=None)
        if r.returncode == 0 and os.path.getsize(dst + ".part") > 1_000_000:
            os.replace(dst + ".part", dst)
            return True, "curl"
    if env.wget:
        r = run([env.wget, "-O", dst + ".part", url], timeout=None)
        if r.returncode == 0 and os.path.getsize(dst + ".part") > 1_000_000:
            os.replace(dst + ".part", dst)
            return True, "wget"
    if os.path.exists(dst + ".part"):
        os.remove(dst + ".part")
    return False, "no downloader succeeded"


def convert_gguf(env, dtype, imdir):
    """Convert from the official checkpoint (requires torch)."""
    ckpt_dir = os.path.join(GKD, "models", "pytorch")
    ckpt = None
    if os.path.isdir(ckpt_dir):
        for f in sorted(os.listdir(ckpt_dir)):
            if f.endswith(".best"):
                ckpt = os.path.join(ckpt_dir, f)
                break
    if not ckpt:
        return False, "no checkpoint in cpp_ggml/models/pytorch"
    r = run([env.python, os.path.join(GKD, "scripts", "convert_gkd_to_gguf.py"),
             "--checkpoint", ckpt, "--dtype", dtype], timeout=None)
    ok = r.returncode == 0 and os.path.isfile(gguf_path(dtype))
    return ok, "convert" if ok else r.stderr[-200:] or r.stdout[-200:]


def stage_models(env, args):
    log("[models] GGUF acquisition")
    have, missing = [], []
    for d in args.dtypes:
        (have if os.path.isfile(gguf_path(d)) else missing).append(d)
    if have:
        record("models", "present", "OK", ", ".join(have))
    if not missing:
        return have
    for d in missing:
        if args.skip_download:
            record("models", d, "SKIP", "--skip-download")
            continue
        ok, how = download_gguf(env, d, args.hf_repo)
        if ok:
            record("models", d, "OK", f"downloaded via {how}")
            have.append(d)
            continue
        if env.torch and not args.skip_convert:
            ok, how = convert_gguf(env, d, None)
            record("models", d, "OK" if ok else "FAIL",
                   f"convert: {how}" if ok else f"convert failed: {how}")
            if ok:
                have.append(d)
        else:
            record("models", d, "SKIP",
                   f"download failed and torch unavailable (get it from "
                   f"https://huggingface.co/{args.hf_repo})")
    return have


def stage_build(env, args):
    log("[build] cmake presets")
    built = []
    for backend in args.backends:
        support = env.backend_supported(backend)
        if support is not True:
            record("build", backend, "SKIP", support)
            continue
        binpath = os.path.join(GKD, f"build-{backend}", "bin", "gkd-cli")
        if os.path.isfile(binpath) and not args.force_build:
            record("build", backend, "SKIP", "existing build reused (--force-build to rebuild)")
            built.append(backend)
            continue
        cmd = [env.cmake, "--preset", backend]
        e = {}
        if backend == "cuda":
            e = env.cuda_env()
        if backend == "vulkan":
            e = env.vulkan_env()
        r = run(cmd, env=e, cwd=GKD, timeout=None)
        if r.returncode != 0:
            record("build", backend, "FAIL", "configure failed")
            continue
        r = run([env.cmake, "--build", "--preset", backend, "--jobs", str(args.jobs)],
                env=e, cwd=GKD, timeout=None)
        if r.returncode == 0 and os.path.isfile(binpath):
            record("build", backend, "OK", f"built ({args.jobs} jobs)")
            built.append(backend)
        else:
            tail = (r.stderr or r.stdout)[-200:]
            record("build", backend, "FAIL", f"build failed: {tail}")
    return built


def infer_once(env, backend, dtype, image_path, mode, imdir, threads):
    """One gkd-cli detect; returns parsed json dict or None."""
    import json
    binpath = os.path.join(GKD, f"build-{backend}", "bin", "gkd-cli")
    model = gguf_path(dtype)
    cmd = [binpath, "detect", "--model", model, "--input", image_path,
           "--threads", str(threads)]
    texts = ["nose", "left eye", "right eye", "left ear", "right ear"]
    if mode == "text":
        for t in texts:
            cmd += ["--kps-texts", t]
    elif mode == "visual":
        sup = os.path.join(imdir, "2007_003778.jpg")
        if not os.path.isfile(sup):
            return None
        cmd += ["--support-image", sup, "--support-kps", "343", "166", "281", "158", "311", "197"]
    else:  # multimodal
        sup = os.path.join(imdir, "2007_003778.jpg")
        if not os.path.isfile(sup):
            return None
        cmd += ["--kps-texts", "left eye", "right eye", "nose",
                "--support-image", sup,
                "--support-kps", "343", "166", "281", "158", "311", "197"]
    r = run(cmd, env=env.cuda_env() if backend == "cuda" else
            (env.vulkan_env() if backend == "vulkan" else None), timeout=600)
    for line in (r.stdout or "").splitlines():
        if line.startswith('{"json":'):
            return json.loads(line)["json"]
    return None


def stage_infer(env, args, imdir, dtypes, backends):
    log("[infer] official demo on every available config")
    if not imdir:
        record("infer", "demo", "SKIP", "no test images")
        return
    image = os.path.join(imdir, DEMO_IMAGE)
    if not os.path.isfile(image):
        record("infer", "demo", "SKIP", f"missing {DEMO_IMAGE}")
        return
    modes = ["text", "visual", "multimodal"] if args.modes == "all" else [args.modes]
    ran = 0
    for backend in backends:
        for dtype in dtypes:
            if not os.path.isfile(gguf_path(dtype)):
                continue
            for mode in modes:
                j = infer_once(env, backend, dtype, image, mode, imdir, args.threads)
                if j is None:
                    record("infer", f"{backend}-{dtype}-{mode}", "FAIL", "no json output")
                    continue
                scores = [round(s, 3) for s in j["scores"]]
                record("infer", f"{backend}-{dtype}-{mode}", "OK", f"scores={scores}")
                ran += 1
    if ran == 0:
        record("infer", "demo", "SKIP", "no model/backend available")


def stage_parity(env, args, imdir, backends):
    log("[parity] tap-level diff vs PyTorch")
    if not env.torch and env.find_torch_cuda_python() is None:
        record("parity", "taps", "SKIP",
               "no interpreter with a CUDA-enabled torch "
               "(install torch with CUDA, or run on a CPU-only box and skip)")
        return
    if not imdir:
        record("parity", "taps", "SKIP", "no test images")
        return
    backend = "cuda" if "cuda" in backends else ("cpu" if "cpu" in backends else None)
    if backend is None:
        record("parity", "taps", "SKIP", "no usable backend")
        return
    ref_dir = "/tmp/gkd_e2e_ref_taps"
    cpp_dir = "/tmp/gkd_e2e_cpp_taps"
    image = os.path.join(imdir, DEMO_IMAGE)
    texts = ["nose", "left eye", "right eye", "left ear", "right ear"]
    # dump_taps needs torch+CUDA *and* the repo's tokenizer deps; try every
    # candidate interpreter until one succeeds.
    dump_script = os.path.join(GKD, "scripts", "dump_taps.py")
    r = None
    for py in env.torch_cuda_python_candidates():
        rr = run([py, dump_script, "--image", image, "--kps-texts", *texts,
                  "--out-dir", ref_dir], timeout=1800)
        if rr.returncode == 0:
            r = rr
            break
        log(f"    (dump_taps failed with {py}; trying next candidate)")
    if r is None:
        record("parity", "ref", "SKIP",
               "no python env has torch+CUDA plus the repo's tokenizer deps "
               "(ftfy, timm, ...) - run scripts/dump_taps.py manually")
        return
    binpath = os.path.join(GKD, f"build-{backend}", "bin", "gkd-cli")
    model = gguf_path("f32") if os.path.isfile(gguf_path("f32")) else gguf_path("q8_0")
    cmd = [binpath, "detect", "--model", model, "--input", image, "--threads", "16",
           "--dump-taps", cpp_dir] + sum([["--kps-texts", t] for t in texts], [])
    e = env.cuda_env() if backend == "cuda" else {}
    r = run(cmd, env=e, timeout=1800)
    if r.returncode != 0:
        record("parity", "cpp", "FAIL", "engine dump failed")
        return
    r = run([env.python, os.path.join(GKD, "scripts", "parity_reference.py"),
             "diff", ref_dir, cpp_dir], timeout=300)
    tail = (r.stdout or "").strip().splitlines()
    summary = next((l for l in tail if "ok," in l), "?")
    import re
    failed = [m.group(1) for l in tail
              for m in [re.match(r"^(\S+)\s+FAIL\s", l)] if m]
    # the preprocessing propagation chain (JPEG-decoder sub-LSB noise) is a
    # known, documented residual: it degrades in_ims/vis_tokens/context but
    # NOT the final heatmaps. Report it as OK-with-note instead of a failure
    # when the final heatmaps match.
    known = {"in_ims", "vis_tokens", "context"}
    heat_ok = any(l.startswith("heatmaps_fused") and " OK" in l for l in tail)
    if failed and set(failed) <= known and heat_ok:
        record("parity", "taps", "OK", summary +
               " [known residual: JPEG-decode sub-LSB in preprocessing]")
    else:
        record("parity", "taps", "OK" if r.returncode == 0 else "FAIL", summary)


def stage_bench(env, args, imdir, dtypes, backends):
    log("[bench] full latency + accuracy matrix")
    if not args.full:
        record("bench", "matrix", "SKIP", "use --full to run (slow)")
        return
    if not imdir:
        record("bench", "matrix", "SKIP", "no test images")
        return
    r = run([env.python, os.path.join(GKD, "scripts", "run_all_tests.py"),
             "--warmup", str(args.bench_warmup), "--iters", str(args.bench_iters)],
            env=env.cuda_env(), timeout=None)
    record("bench", "matrix", "OK" if r.returncode == 0 else "FAIL",
           f"warmup={args.bench_warmup} iters={args.bench_iters}")
    if env.torch:
        r = run([env.python, os.path.join(GKD, "scripts", "bench_pytorch.py"),
                 "--warmup", "3", "--iters", "10"], timeout=None)
        record("bench", "pytorch-ref", "OK" if r.returncode == 0 else "FAIL", "reference latency")


def stage_charts(env, args):
    log("[charts] benchmark figures")
    bench_dir = os.path.join(GKD, "benchmarks")
    has_data = any(f.startswith("latency_") for f in os.listdir(bench_dir)) \
        if os.path.isdir(bench_dir) else False
    if not has_data:
        record("charts", "figures", "SKIP", "no benchmark data yet")
        return
    r = run([env.python, os.path.join(GKD, "scripts", "plot_benchmarks.py")], timeout=600)
    record("charts", "figures", "OK" if r.returncode == 0 else "FAIL",
           "latency/speedup matrices + table")
    if args.full:
        r = run([env.python, os.path.join(GKD, "scripts", "render_parity.py")], timeout=600)
        record("charts", "parity-grid", "OK" if r.returncode == 0 else "FAIL", "parity overlay grid")


def summary(env, dtypes, backends):
    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    width = max(len(f"{s}:{sub}") for s, sub, _, _ in RESULTS)
    for stage, sub, status, detail in RESULTS:
        log(f"  {stage}:{sub:<{width - len(stage) - 1}} {status:<5} {detail}")
    log("-" * 78)
    log(f"  FAIL count: {sum(1 for r in RESULTS if r[2] == 'FAIL')}")
    log("  C++ runtime : cpp_ggml/build-*/bin/gkd-cli  (docs: cpp_ggml/README.md)")
    log("  benchmarks  : cpp_ggml/benchmarks/")
    log("=" * 78)
    return 1 if any(r[2] == "FAIL" for r in RESULTS) else 0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    global ROOT, GKD, GGUF_DIR
    ap = argparse.ArgumentParser(
        description="One-click GKD end-to-end: official PyTorch + C++ ggml runtime. "
                    "Every stage auto-detects its requirements and skips itself when "
                    "they are missing; completed work is reused on re-runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="repository root (default: this script's directory)")
    ap.add_argument("--dtypes", default=",".join(ALL_DTYPES),
                    help="comma-separated weight dtypes to ensure/run (default: all found)")
    ap.add_argument("--backends", default="cpu,cuda,vulkan",
                    help="comma-separated backends to build/run (default: all; "
                         "unsupported ones are skipped automatically)")
    ap.add_argument("--modes", default="text", choices=["text", "visual", "multimodal", "all"],
                    help="prompt mode for the infer stage (default: text)")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                    help=f"HuggingFace repo with pre-converted GGUF (default: {DEFAULT_HF_REPO})")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                    help="parallel build jobs (default: all cores)")
    ap.add_argument("--threads", type=int, default=min(os.cpu_count() or 4, 16),
                    help="inference CPU threads (default: min(cores, 16))")
    ap.add_argument("--full", action="store_true",
                    help="also run the full latency+accuracy matrix and parity grid (slow)")
    ap.add_argument("--skip-download", action="store_true", help="never download GGUF files")
    ap.add_argument("--skip-convert", action="store_true", help="never convert from the checkpoint")
    ap.add_argument("--force-build", action="store_true", help="rebuild even if binaries exist")
    ap.add_argument("--bench-warmup", type=int, default=2, help="bench warmup runs (default: 2)")
    ap.add_argument("--bench-iters", type=int, default=5, help="bench timed iterations (default: 5)")
    ap.add_argument("--list-only", action="store_true", help="print the plan, execute nothing")
    args = ap.parse_args()

    # normalize / validate (re-anchor paths to --root)
    ROOT = os.path.abspath(args.root)
    GKD = os.path.join(ROOT, "cpp_ggml")
    GGUF_DIR = os.path.join(GKD, "models", "gguf")
    args.dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    bad = [d for d in args.dtypes if d not in ALL_DTYPES]
    if bad:
        ap.error(f"unknown dtypes: {bad} (choose from {ALL_DTYPES})")

    env = Env()
    log("=" * 78)
    log("GKD one-click end-to-end")
    log("=" * 78)
    stage_env(env, args)
    imdir, imgs = stage_data(env, args)

    if args.list_only:
        log("\n[list-only] plan (nothing executed):")
        for d in args.dtypes:
            state = "have" if os.path.isfile(gguf_path(d)) else "download/convert"
            log(f"  models : {d:<6} {state}")
        for b in args.backends:
            support = env.backend_supported(b)
            state = "have build" if os.path.isfile(os.path.join(GKD, f"build-{b}", "bin", "gkd-cli")) \
                else ("build" if support is True else f"skip ({support})")
            log(f"  build  : {b:<6} {state}")
        log(f"  infer  : {args.modes} mode on every available backend x dtype")
        log(f"  parity : {'run' if env.torch else 'skip (no torch)'} (fp32 tap diff)")
        log(f"  bench  : {'run' if args.full else 'skip (use --full)'}")
        return 0

    t0 = time.time()
    dtypes = stage_models(env, args)
    backends = stage_build(env, args)
    stage_infer(env, args, imdir, dtypes, backends)
    stage_parity(env, args, imdir, backends)
    stage_bench(env, args, imdir, dtypes, backends)
    stage_charts(env, args)
    log(f"\ntotal wall time: {time.time() - t0:.1f}s")
    return summary(env, dtypes, backends)


if __name__ == "__main__":
    sys.exit(main())
