# GKDT ggml Runtime (`cpp_ggml`)

A pure C++/ggml inference engine for **GKDT-L** (General Keypoint Detection
Transformer). The model graph is reproduced 1:1 from the official PyTorch
implementation — no Python and no PyTorch at runtime — with automatic
CPU / CUDA / Vulkan execution through `ggml_backend_sched`.

Verified parity against the stock PyTorch model (RTX 4090, same inputs):
keypoint coordinates agree to **≤ 0.0048 px** in all three prompt modes for
every backend × precision config of the full matrix (cpu/cuda/vulkan ×
f32/f16/q8_0/q4_0/q4_K). Score differences stay ≤ 0.0013 for f32/f16/q8_0;
the q4 dtypes trade up to ~0.026 (q4_0) / ~0.017 (q4_K) of score amplitude
while keeping every coordinate. Every GPU config beats the PyTorch reference
end-to-end — cuda-q4_0 runs 40.7 ms vs 231.5 ms in text mode (5.7×); see
`benchmarks/speedup_table.md`, `benchmarks/latency_matrix.png` and
`benchmarks/speedup_matrix.png`.

## Pipeline at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│ gkd_fullset.best — official PyTorch checkpoint (6.34 GB)            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  ① convert (needs torch, one-time)
                                │     python scripts/convert_gkd_to_gguf.py --dtype q4_K
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ models/gguf/gkd_fullset-<dtype>.gguf — GGUF v3, self-contained      │
│ 483 MiB (q4) … 3.39 GiB (f32) · 683 tensors · BPE vocab embedded    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  ② build (one preset per GPU backend)
                                │     cmake --preset {cpu|cuda|vulkan} && cmake --build …
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ gkd-cli / libgkdgml (C API) — pure C++, zero Python at runtime      │
│                                                                     │
│  image ──► preprocess ──► vision graph (DINOv3 ViT-L · 24 blocks)   │
│  texts ──► CLIP BPE ──►  text graph  (dinotxt · 24 blocks + anet)   │
│  1-shot ──► Gaussian pooling ──────────┐                            │
│                                        ▼                            │
│              detect graph (KG transformer ×2 + head)                │
│                    ──► heatmap decode ──► keypoints + scores        │
│                                                                     │
│  ggml_backend_sched: CUDA or Vulkan GPU + CPU fallback per graph    │
└─────────────────────────────────────────────────────────────────────┘
```

The ggml dependency is pinned: **ggml v0.21.0** (upstream commit `8599e0ea`)
as a git submodule, kept pristine — every deviation lives in
`patches/ggml/*.patch` and is applied automatically (and idempotently) at
cmake configure time.

```
cpp_ggml/
├── CMakeLists.txt / CMakePresets.json   # cpu / cuda / vulkan presets
├── patches/ggml/*.patch                 # submodule deviations, auto-applied by cmake
├── include/gkdgml.h                     # minimal public C API
├── src/                                 # engine (graphs, gguf, tokenizer, io)
│   ├── gkd_graph.cpp                    # the three graphs + session
│   ├── gguf_loader.cpp                  # GGUF v3 reader (no_alloc + stream)
│   ├── tokenizer.cpp                    # CLIP BPE (SimpleTokenizer)
│   ├── image_io.cpp                     # PIL-exact resize/pad preprocessing
│   └── ...                              # backend, capi, postprocess, common
├── scripts/
│   ├── convert_gkd_to_gguf.py           # checkpoint -> GGUF (f32/f16/q8_0/q4_0/q4_K)
│   ├── dump_taps.py                     # official PyTorch per-layer reference taps
│   ├── parity_reference.py              # tap diff tool
│   ├── bench_pytorch.py                 # stock PyTorch latency reference
│   ├── run_all_tests.py                 # full latency+accuracy matrix driver
│   ├── plot_benchmarks.py               # charts + speedup_table.md
│   └── render_parity.py                 # side-by-side parity grid
├── models/
│   ├── pytorch/                         # gkd_fullset.best (converter input)
│   └── gguf/                            # generated GGUF files (git-ignored)
├── tests/test_ops.cpp                   # golden-op tests (mul_mat/flash/rope/...)
└── benchmarks/                          # jsonl records, png charts, tables
```

## From clone to first keypoints (end-to-end walkthrough)

Prerequisites: cmake ≥ 3.14 + a C++17 compiler; per GPU backend either the
CUDA toolkit (verified: 12.6, cc 8.9) or the Vulkan SDK with `glslc` on PATH
(verified: 1.4.350.0); python3 with PyTorch **only for step 3** — the runtime
itself never needs Python.

```bash
# 1) clone with the pinned submodules (ggml v0.21.0 + stb)
git clone --recursive <repo-url>
# (existing checkout: git submodule update --init --recursive)
# ggml patches in cpp_ggml/patches/ggml/ are applied by cmake automatically

# 2) get the model weights — EITHER download the pre-converted GGUF
#    (https://huggingface.co/Asher-1/GKD_GGUF, no torch needed):
huggingface-cli download Asher-1/GKD_GGUF gkd_fullset-q4_K.gguf \
    --local-dir cpp_ggml/models/gguf
#    ... OR convert from the official PyTorch checkpoint yourself
#    (checkpoint links in the root README, section 2; torch env; ~3 min):
cp /path/to/gkd_fullset.best cpp_ggml/models/pytorch/
cd cpp_ggml
python scripts/convert_gkd_to_gguf.py --dtype q4_K
#    -> models/gguf/gkd_fullset-q4_K.gguf  (483 MiB, 683 tensors)

# 3) build (each build dir = one GPU backend + CPU fallback)
cmake --preset cpu   && cmake --build --preset cpu
CUDA_PATH=/usr/local/cuda-12.6 cmake --preset cuda && cmake --build --preset cuda
source ~/VulkanSDK/x.y.z/setup-env.sh && cmake --preset vulkan && cmake --build --preset vulkan

# 4) run — text / visual / multimodal prompts
build-cuda/bin/gkd-cli detect \
    --model models/gguf/gkd_fullset-q4_K.gguf \
    --input ../test_real_world/ims1/2007_007524.jpg \
    --kps-texts nose "left eye" "right eye" "left ear" "right ear" \
    --out result.jpg
#    -> scores ≈ [0.92, 0.93, 0.95, 0.79, 0.88]; nose at ≈ (112, 231) px

# 5) verify + benchmark against the official PyTorch model (see section 4)
python scripts/run_all_tests.py --warmup 2 --iters 5
python scripts/plot_benchmarks.py
```

## 1. Build

```bash
cd cpp_ggml
git submodule update --init third_party          # ggml v0.21.0 + stb
# (cmake automatically applies patches/ggml/*.patch to the pristine submodule:
#  forward-check ok -> apply; reverse-check ok -> already applied; else abort)

# CPU only
cmake --preset cpu && cmake --build --preset cpu

# CUDA (cc 8.9 by default; set CMAKE_CUDA_ARCHITECTURES for other GPUs)
CUDA_PATH=/usr/local/cuda-12.6 cmake --preset cuda && cmake --build --preset cuda

# Vulkan (needs the Vulkan SDK with glslc on PATH)
source ~/VulkanSDK/x.y.z/setup-env.sh
cmake --preset vulkan && cmake --build --preset vulkan
```

One build directory enables exactly one GPU backend plus the CPU fallback
(`ggml_backend_sched` splits each graph across them automatically).

## 2. Get the model weights

**Pre-converted GGUF files for all five dtypes are published at
https://huggingface.co/Asher-1/GKD_GGUF** — downloading them skips the
conversion step entirely (see `models/MODEL_CARD.md` for per-file details
and direct links):

```bash
huggingface-cli download Asher-1/GKD_GGUF --local-dir models/gguf
```

To convert from the official checkpoint yourself (custom dtype, custom
training run), `scripts/convert_gkd_to_gguf.py` reads **only**
`cpp_ggml/models/pytorch/gkd_fullset.best` and writes
`models/gguf/gkd_fullset-<dtype>.gguf`:

```bash
python scripts/convert_gkd_to_gguf.py --dtype f32    # also: f16, q8_0, q4_0, q4_K
```

Details worth knowing:
- `bias_mask` (LinearKMaskedBias) is folded into the qkv biases at export.
- RoPE periods are exported as a tensor so the engine reproduces the
  checkpoint's exact frequencies.
- ggml layout == numpy C-order with reversed shapes, therefore torch Linear
  weights are stored raw (no transpose). `anet.proj` is stored transposed
  because the official code uses `x @ proj` (a raw Parameter, not Linear).
- The cross-attention in_proj is split into q/k/v tensors: `ggml_mul_mat`
  through an offset view of a packed weight mis-addresses on some backends.
- Quantization policy: only 2-D weight matrices are quantized; vectors
  (biases, norms, tokens) always stay F32 — elementwise CUDA kernels reject
  mixed-precision broadcast sources, and patch_w's fastest dim (16) cannot
  form a block. Block-size constraints on the fastest dim: `q8_0`/`q4_0`
  need `ne0 % 32 == 0`, `q4_K` needs `ne0 % 256 == 0` (super-blocks);
  non-conforming tensors fall back to F32 automatically. The Q4_K/Q4_0
  quantizers in the converter are numpy ports of ggml's reference
  implementations and were byte-verified against them (Q4_0 byte-identical;
  Q4_K equal within the reference's own near-tie float ordering).
- All five dtypes run on all three backends; q4_K keeps coordinates exact
  and scores within ~0.03 of fp32 (see `benchmarks/accuracy.json`).

### Quantization formats: q4_0 vs q4_K

Both store 2-D weight matrices at **4.5 bits per value** (483 MiB each for
GKDT-L) — the difference is entirely in how the per-block scale is encoded,
which decides accuracy and which CPU kernel path is fastest.

**Q4_0 — one scale per 32 values (18-byte blocks)**

```
block: [ d:f16 | qs:16 bytes ]          32 values, q = 0..15 nibbles
value = (q - 8) * d                     d = signed_max / -8
```
A single scale must cover the whole block, so one large outlier in a block
degrades the resolution of the other 31 values.

**Q4_K — super-blocks of 256 with per-32 scale+min (144-byte blocks)**

```
super-block: [ d:f16 | dmin:f16 | scales:12 B | qs:128 B ]
per 32-value sub-block:  value = d*sc*(q - 8) - dmin*m

  sc, m : 6-bit quantized scale/min, 8 pairs packed into 12 bytes
  chosen by an error-minimizing search (make_qkx2_quants: 21 refinement
  steps, weights ∝ av_x + |x|), not just by fitting the extreme value
```
The separate `min` absorbs distribution asymmetry and the search minimizes
the weighted squared reconstruction error — same 4.5 bits/value, materially
less error.

**Measured on GKDT-L** (15 official images, mean; RTX 4090 + i9-14900K;
full matrix in `benchmarks/`):

| | f32 | f16 | q8_0 | q4_0 | q4_K |
|---|---|---|---|---|---|
| file size | 3.39 GiB | 1.70 GiB | 905 MiB | 483 MiB | 483 MiB |
| bits/value (2-D weights) | 32 | 16 | 8.5 | 4.5 | 4.5 |
| max score diff vs PyTorch | 0.0011 | 0.0012 | 0.0021 | 0.026 | 0.017 |
| coordinates vs PyTorch | ≤ 0.0048 px — identical for **all** dtypes |||||
| cuda · text latency | 52.8 ms | 41.4 | 41.3 | **40.7** | 41.2 |
| vulkan · text latency | 51.5 | **48.9** | 51.7 | 52.5 | 52.4 |
| cpu · text latency | 1555 | 1734 | 1671 | 1882 | **1211** |

Choosing:
- **q4_K** — the default recommendation: same size as q4_0, clearly better
  accuracy, and by far the fastest CPU config (its Goldmann-style kernels win
  even though `GGML_LLAMAFILE=ON` makes fp32 unusually strong on CPU; the
  simpler q4_0 dequant path does not).
- **q4_0** — marginally fastest on CUDA; simplest encoding.
- **q8_0** — near-lossless (≤ 0.002) at half the q4 size.
- **f16** — best Vulkan latency, near-lossless.
- **f32** — reference for parity debugging.

Correctness of the converters' Q4_0/Q4_K writers was cross-verified against
ggml's own reference quantizers (`ggml_quantize_chunk`): Q4_0 is
byte-identical; Q4_K differs only inside the reference's own near-tie float
ordering, with identical reconstruction-error statistics.

### Accuracy over all official images (not just the demo example)

`scripts/accuracy_all_images.py` compares every engine config against the
stock PyTorch model on **all 15 official images** (text mode, each image's
own prompt set) plus the **3 official demo commands** of
`test_real_world/scripts/eval_single_obj_gkd.sh` (multimodal whole-image,
text with bbox ROI, cross-image visual support). Results:
`benchmarks/accuracy_all_images.json`, `benchmarks/accuracy_by_image.png`,
`benchmarks/parity_official_examples.png`.

- **The 3 official demo commands** (the authoritative test): every config —
  including both q4 dtypes — reproduces PyTorch to ≤ 0.0004 px mean on the
  multimodal and visual examples and 0.29 px mean (worst single keypoint
  2.85 px, identical in fp32) on the bbox example; score diff ≤ 0.031.
- **All 15 images**: f32/f16/q8_0 agree point-for-point with PyTorch on the
  9 confidently-localized images (mean ≤ 0.4 px, max ≤ 1.3 px); the q4
  dtypes stay sub-pixel-to-few-px on most keypoints but can shift the argmax
  by tens of pixels on **near-flat heatmaps** (multi-object scenes where
  PyTorch itself scores < 0.1 and the peak is not meaningful). This is the
  honest boundary of 4-bit: use q4 when scores are trusted, use
  f16/q8_0/f32 when every argmax must be reproducible.

## 3. Run inference

```bash
# text prompts
build-cuda/bin/gkd-cli detect --model models/gguf/gkd_fullset-f16.gguf \
    --input ../test_real_world/ims1/2007_007524.jpg \
    --kps-texts nose "left eye" "right eye" "left ear" "right ear" \
    --skeleton 1 2 1 3 2 3 2 4 3 5 --out result.jpg

# visual prompt (1-shot support image + keypoints)
build-cuda/bin/gkd-cli detect --model models/gguf/gkd_fullset-f16.gguf \
    --input ../test_real_world/ims1/2007_007524.jpg \
    --support-image ../test_real_world/ims1/2007_003778.jpg \
    --support-kps 343 166 281 158 311 197

# multimodal: text i and visual i are fused per row (official semantics,
# requires #texts == #support keypoints)
build-cuda/bin/gkd-cli detect --model models/gguf/gkd_fullset-f16.gguf \
    --input ../test_real_world/ims1/2007_007524.jpg \
    --kps-texts "left eye" "right eye" "nose" \
    --support-image ../test_real_world/ims1/2007_003778.jpg \
    --support-kps 343 166 281 158 311 197

# steady-state latency (per image, warmup + iters)
gkd-cli bench --model models/gguf/gkd_fullset-f16.gguf --input img.jpg \
    --kps-texts nose "left eye" --warmup 3 --iters 10

# model metadata
gkd-cli info --model models/gguf/gkd_fullset-f16.gguf
```

`detect` prints one `{"json": ...}` line with the ROI transform, normalized
keypoints and scores — that is what the evaluation driver parses.

## 4. Evaluation / parity

```bash
# 1) reference taps from the official PyTorch model (conda env with torch)
python scripts/dump_taps.py --image ../test_real_world/ims1/2007_007524.jpg \
    --kps-texts nose "left eye" "right eye" "left ear" "right ear" \
    --out-dir /tmp/gkd_taps

# 2) the same tensors from the C++ engine
gkd-cli detect ... --dump-taps /tmp/gkd_cpp_taps

# 3) diff them
python scripts/parity_reference.py diff /tmp/gkd_taps /tmp/gkd_cpp_taps

# full latency + accuracy matrix (15 official images x 3 prompt modes x
# every discovered backend x dtype config: cpu/cuda/vulkan x
# f32/f16/q8_0/q4_0/q4_K — configs are auto-discovered from the built
# binaries and converted GGUF files, nothing is cherry-picked)
python scripts/run_all_tests.py --warmup 2 --iters 5
python scripts/bench_pytorch.py --warmup 3 --iters 10     # PyTorch reference
python scripts/plot_benchmarks.py                         # charts + speedup table
python scripts/render_parity.py                           # parity grid image

# accuracy on ALL 15 official images vs the stock PyTorch model (text mode,
# per-image official prompt sets) + the 3 official demo commands of
# test_real_world/scripts/eval_single_obj_gkd.sh (multimodal / bbox-ROI /
# cross-image visual support):
~/anaconda3/envs/python3.12/bin/python scripts/accuracy_all_images.py --pytorch  # torch env, once
python scripts/accuracy_all_images.py                   # every engine config
# -> benchmarks/accuracy_all_images.json + accuracy_by_image.png
python scripts/render_parity.py --grid                  # parity_official_examples.png
```

Interpretation notes for the tap diff: `in_ims`, `vis_tokens` and `context`
show a residual mean diff of ~1e-2 because the C++ JPEG decoder/resize differs
from PIL by sub-LSB pixel noise (visible nowhere in the outputs — final
coordinates still agree to 0.003 px). Everything downstream of the vision
tower matches at fp32 rounding level (heatmaps ~1e-4).

## 5. Numerical conventions

The engine follows ggml's memory model: a tensor `ne={d0,...,dn}` stores
element `(i0,...,in)` at `i0 + i1*d0 + ...`, i.e. numpy C-order with the
**reversed** shape. Two consequences that the graph code must respect:

- torch Linear weights `[out,in]` (C-order) map directly to ggml `ne={in,out}`
  — no transpose anywhere;
- elementwise broadcast ops (`ggml_mul`/`add`/`div`) tile src1 per-dimension
  with wrap-around, so per-row scaling uses an `{N,1}` src (repeat passes on
  the leading dim), never an `{N}`-on-`{N×M}` mismatch that happens to be
  divisible.

`tests/test_ops.cpp` locks these conventions down as golden tests (mul_mat
layout, packed-qkv flash attention, the split RoPE formulation, L2 norm and
bilinear upsample); `gkd-test-ops` exits non-zero on any drift.

## 6. Files & options

| option | default | description |
|---|---|---|
| `GKD_GGML_CUDA` | OFF | compile the ggml CUDA backend into the build |
| `GKD_GGML_VULKAN` | OFF | compile the ggml Vulkan backend |
| `GKD_GGML_METAL` | OFF | compile the ggml Metal backend |
| `GKD_GGML_SHARED` | OFF | build libgkdgml as a shared library |
| `GKD_GGML_BUILD_CLI` | ON | build `gkd-cli` |
| `GKD_GGML_BUILD_TESTS` | OFF | build `gkd-test-ops` |
| `CMAKE_CUDA_ARCHITECTURES` | 89 | GPU compute capability |

The C API in `include/gkdgml.h` (`gkd_session_create` / `gkd_detect` /
`gkd_session_free`) is the minimal embedding surface for applications that
cannot use the CLI.
