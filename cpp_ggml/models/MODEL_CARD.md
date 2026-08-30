# GKDT-L GGUF Model Card

## Model

- **Architecture**: GKDT-L = DINOv3 ViT-L vision tower (D=1024, 24 blocks,
  16 heads, 4 storage tokens + cls, RoPE with checkpoint periods, LayerScale
  1e-5, LinearKMaskedBias qkv) + dinotxt text tower (D=1280, 24 causal blocks,
  20 heads) + TextHead linear projection (1280→2048) + text adaptation net
  (1 CLIP-style block, QuickGELU, 2048↔1280) + KG transformer (2 blocks,
  SA+CA, d_ff=1024, mask token) + parameter-free detection head
  (bilinear 4× upsample, L2 kernel norm, 1×1 kernel "conv", per-row mask
  fusion over prompts padded to 80 rows).
- **Source checkpoint**: `models/pytorch/gkd_fullset.best` (GKDT-L for
  real-world testing, ECCV 2026 release, ~6.34 GB).
- **Input**: RGB image resized so the longer side is 384 px and center-padded
  to 384×384 with the ImageNet mean color (124, 116, 104); ImageNet
  normalization. Preprocessing matches the official
  `mytransforms` pipeline exactly.
- **Prompts**: text keypoint names (CLIP BPE tokenized), 1-shot visual
  keypoints (soft-fiber Gaussian pooling, σ=14), or both (fused per row).
- **Output**: per-prompt heatmap peak over a 96×96 grid → keypoints in the
  ROI's normalized -1..1 space + confidence score.

## Download

All five runtime files are published at
**https://huggingface.co/Asher-1/GKD_GGUF** — no conversion step is needed
unless you want a dtype we do not ship:

```bash
# everything into the engine's model directory
huggingface-cli download Asher-1/GKD_GGUF --local-dir cpp_ggml/models/gguf

# or just the recommended file
huggingface-cli download Asher-1/GKD_GGUF gkd_fullset-q4_K.gguf --local-dir cpp_ggml/models/gguf

# plain HTTPS also works (no HF client required)
wget https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/gkd_fullset-q4_K.gguf \
     -O cpp_ggml/models/gguf/gkd_fullset-q4_K.gguf
```

## Files — per-model details

| file | dtype | size | download |
|---|---|---|---|
| `gguf/gkd_fullset-f32.gguf` | F32 | 3.39 GiB | [link](https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/gkd_fullset-f32.gguf) |
| `gguf/gkd_fullset-f16.gguf` | F16 weights | 1.70 GiB | [link](https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/gkd_fullset-f16.gguf) |
| `gguf/gkd_fullset-q8_0.gguf` | Q8_0 weights | 905 MiB | [link](https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/gkd_fullset-q8_0.gguf) |
| `gguf/gkd_fullset-q4_0.gguf` | Q4_0 weights | 483 MiB | [link](https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/gkd_fullset-q4_0.gguf) |
| `gguf/gkd_fullset-q4_K.gguf` | Q4_K weights | 483 MiB | [link](https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/gkd_fullset-q4_K.gguf) |

All five run on all three backends (cpu / cuda / vulkan). Only 2-D weight
matrices are quantized; vectors (biases, norms, cls/storage and RoPE tensors)
always stay F32, and tensors whose fastest dimension cannot satisfy the block
size fall back to F32 automatically (`ne0 % 32` for q8_0/q4_0, `ne0 % 256`
for q4_K). Shared accuracy facts: keypoint **coordinates** on the official
examples match stock PyTorch to ≤ 0.0048 px for every dtype; the dtypes
differ in score amplitude and — for q4 — in argmax stability on near-flat
heatmaps (see *Verified accuracy* below).

### `gkd_fullset-f32.gguf` — fp32 reference (3.39 GiB)

- Every tensor in full fp32; the bit-exact baseline the other four are
  validated against (parity workflow in `../scripts/dump_taps.py`).
- Accuracy: score diff ≤ 0.0013 vs PyTorch; all-argmax reproducible.
- Latency (text mode): cuda 52.8 ms · vulkan 51.5 ms · cpu 1555 ms — the
  CPU number is unusually strong because `GGML_LLAMAFILE=ON` enables the
  tinyBLAS fp32 GEMM.
- Use when: debugging parity, quantifying quantization cost, or when the
  argmax of every heatmap must be bit-reproducible.

### `gkd_fullset-f16.gguf` — fp16 weights, near-lossless (1.70 GiB)

- 2-D weight matrices in F16; all vectors (biases, norms, tokens) stay F32
  (the CUDA elementwise kernels reject mixed-precision broadcasts).
- Accuracy: score diff ≤ 0.0012; coordinates identical to fp32 — no argmax
  drift anywhere in the official sweep.
- Latency (text): **cuda 41.4 ms · vulkan 48.9 ms (best Vulkan)** · cpu
  1734 ms (x86 has no native fp16 GEMM, so CPU is slightly slower than f32).
- Use when: you want near-lossless quality at half the fp32 size, or the
  best Vulkan latency.

### `gkd_fullset-q8_0.gguf` — 8-bit compact, near-lossless (905 MiB)

- Q8_0 blocks (32 values, f16 scale + int8 payload) for 2-D matrices with
  `ne0 % 32 == 0`; everything else F32.
- Accuracy: score diff ≤ 0.0021; coordinates identical to fp32 in the
  official sweep (confident-scene mean ≤ 0.5 px, max ≤ 1.3 px).
- Latency (text): cuda 41.3 ms · vulkan 51.7 ms · cpu 1671 ms.
- Use when: you want fp32-grade behavior at ~3.7× less disk/memory than
  fp32 — the safe compact choice.

### `gkd_fullset-q4_0.gguf` — simplest 4-bit (483 MiB)

- Q4_0 blocks (32 values, one f16 scale, `d = signed_max/-8`), 4.5 bits per
  weight. Simple encoding, but a single outlier in a block degrades the
  other 31 values.
- Accuracy: score diff ≤ 0.026; on the official demos keypoints match
  PyTorch to ≤ 0.0004 px mean, but on near-flat heatmaps (scenes where
  PyTorch itself scores < 0.1) the argmax can shift by tens of pixels.
- Latency (text): **cuda 40.7 ms (fastest overall, 5.7× vs PyTorch)** ·
  vulkan 52.5 ms · cpu 1882 ms (slower than fp32 — the simple dequant path
  loses to tinyBLAS fp32 on x86).
- Use when: maximum CUDA throughput matters more than last-bit score
  fidelity.

### `gkd_fullset-q4_K.gguf` — super-block 4-bit, recommended (483 MiB)

- Q4_K super-blocks (256 values, 8 sub-blocks of 32 with independent 6-bit
  scale+min chosen by an error-minimizing search), 4.5 bits per weight —
  same size as Q4_0, materially lower quantization error.
- Accuracy: score diff ≤ 0.017 (best of the two q4 variants); official demo
  keypoints ≤ 0.0004 px mean / 2.85 px worst (identical to fp32). Same
  near-flat-heatmap argmax caveat as q4_0, but milder.
- Latency (text): cuda 41.2 ms (5.6×) · vulkan 52.4 ms · **cpu 1211 ms —
  the fastest CPU config of the whole matrix** (Goldmann-style Q4_K kernels
  beat even the fp32 tinyBLAS path).
- Use when: **default recommendation** — 7× smaller than fp32, best
  accuracy-per-bit, fastest CPU inference.

## Runtime & ggml version

- Built and verified against **ggml v0.21.0** (upstream commit `8599e0ea`),
  pinned as a git submodule at `../third_party/ggml`. The submodule stays
  pristine: every deviation lives in `../patches/ggml/*.patch` and is applied
  automatically — and idempotently — at cmake configure time (forward-check →
  apply, reverse-check → skip, conflict → abort).
- Verified toolchains: CUDA 12.6 (sm_89, RTX 4090), Vulkan SDK 1.4.350.0,
  CMake ≥ 3.14 + C++17 for the CPU build.
- Runtime dependencies: **none** beyond the ggml backends — no Python, no
  PyTorch, no CLIP/dinov3 repo at inference time; the CLIP BPE vocab is
  embedded in the GGUF.

## Quantization formats

Only 2-D weight matrices are quantized; vectors (biases, norms, cls/storage
and RoPE tensors) always stay F32, and tensors whose fastest dimension cannot
satisfy the block size fall back to F32 automatically (`ne0 % 32` for
q8_0/q4_0, `ne0 % 256` for q4_K).

**Q4_0** — 32 values per 18-byte block, a single f16 scale
(`d = signed_max / -8`, `value = (q-8)·d`). Simple, but one outlier in a
block degrades the other 31 values.

**Q4_K** — 256-value super-blocks (144 bytes), each split into 8 sub-blocks
of 32 with independent 6-bit scale+min; (scale, min) are chosen by an
error-minimizing search rather than by fitting the extreme value:
`value = d·sc·(q-8) − dmin·m`. Same 4.5 bits/value as Q4_0, materially less
reconstruction error — the recommended runtime dtype.

Both writers in `../scripts/convert_gkd_to_gguf.py` are numpy ports of ggml's
reference quantizers and were cross-verified against `ggml_quantize_chunk`
(Q4_0 byte-identical; Q4_K identical modulo the reference's own near-tie
float ordering).

| | f32 | f16 | q8_0 | q4_0 | q4_K |
|---|---|---|---|---|---|
| size (GKDT-L) | 3.39 GiB | 1.70 GiB | 905 MiB | 483 MiB | 483 MiB |
| bits/value (2-D weights) | 32 | 16 | 8.5 | 4.5 | 4.5 |
| max score diff vs PyTorch | 0.0011 | 0.0012 | 0.0021 | 0.026 | 0.017 |
| best backend latency (text) | 51.5 (vulkan) | 41.4 (cuda) | 41.3 (cuda) | **40.7 (cuda)** | 41.2 (cuda) / **1211 ms cpu** |

## Conversion (only if not downloading)

The pre-built files above come from exactly this command, so converting
yourself is only needed for a custom dtype/checkpoint:

```bash
python scripts/convert_gkd_to_gguf.py --dtype f32   # also: f16, q8_0, q4_0, q4_K
# input: models/pytorch/gkd_fullset.best -> models/gguf/gkd_fullset-<dtype>.gguf
```

The converter folds `bias_mask`, exports the RoPE periods, stores torch
Linear weights raw (ggml ne = reversed shape), transposes `anet.proj` (the
official code uses `x @ proj`), splits the KG cross-attention in_proj into
q/k/v tensors, and quantizes only 2-D matrices that satisfy the block-size
constraint.

## Verified accuracy (vs stock PyTorch, 2007_007524.jpg)

Keypoint coordinates agree to **≤ 0.0048 px** (max over the three prompt
modes, original-image pixels) for **every** config of the full
backend × precision matrix — quantization changes score amplitude, never the
argmax location. Mean absolute score difference vs the stock PyTorch
reference (full detail in `../benchmarks/accuracy.json`):

| config | text | visual | multimodal |
|---|---|---|---|
| `cpu-f32` | 0.0005 | 0.0009 | 0.0007 |
| `cpu-f16` | 0.0005 | 0.0009 | 0.0007 |
| `cpu-q8_0` | 0.0009 | 0.0008 | 0.0011 |
| `cpu-q4_0` | 0.0098 | 0.0249 | 0.0078 |
| `cpu-q4_K` | 0.0056 | 0.0140 | 0.0066 |
| `cuda-f32` | 0.0005 | 0.0011 | 0.0007 |
| `cuda-f16` | 0.0005 | 0.0006 | 0.0004 |
| `cuda-q8_0` | 0.0003 | 0.0003 | 0.0013 |
| `cuda-q4_0` | 0.0098 | 0.0248 | 0.0076 |
| `cuda-q4_K` | 0.0094 | 0.0172 | 0.0068 |
| `vulkan-f32` | 0.0010 | 0.0008 | 0.0012 |
| `vulkan-f16` | 0.0012 | 0.0008 | 0.0011 |
| `vulkan-q8_0` | 0.0009 | 0.0021 | 0.0003 |
| `vulkan-q4_0` | 0.0098 | 0.0260 | 0.0069 |
| `vulkan-q4_K` | 0.0057 | 0.0115 | 0.0037 |

### All-image sweep + official demo commands

`../benchmarks/accuracy_all_images.json` extends the verification to **all 15
official test images** (text mode, per-image prompt sets) and to the **three
official demo commands** of `test_real_world/scripts/eval_single_obj_gkd.sh`
(multimodal whole-image, text with bbox ROI [33,38,241,310], cross-image
visual support from alpaca_150.jpg):

- official demo commands: **all 15 configs** reproduce PyTorch to ≤ 0.0004 px
  mean (multimodal, visual) and 0.29 px mean / 2.85 px worst keypoint (bbox;
  the same 2.85 px appears in fp32 — it is the 96×96 grid's 2 px cell size on
  a near-tie peak, not a precision artifact); score diff ≤ 0.031.
- all 15 images: f32/f16/q8_0 match PyTorch point-for-point on the 9
  confidently-localized images (mean ≤ 0.4 px, max ≤ 1.3 px). The q4 dtypes
  keep most keypoints but can shift the argmax by tens of pixels on
  near-flat heatmaps (scenes where PyTorch itself scores < 0.1) — use q4 for
  confident detection workloads, f16/q8_0 when every argmax must be
  bit-reproducible.
- visuals: `accuracy_by_image.png` (config × image error heatmap),
  `parity_official_examples.png` (keypoint overlay grid, PyTorch vs three
  q4_K backends).

End-to-end latency / speedup over the 15 official images (reference: stock
PyTorch on CUDA, RTX 4090) — generated table, see
`../benchmarks/speedup_table.md` and the matrices
`../benchmarks/latency_matrix.png` / `../benchmarks/speedup_matrix.png`:

| mode | cpu-f32 | cpu-f16 | cpu-q8_0 | cpu-q4_0 | cpu-q4_K | cuda-f32 | cuda-f16 | cuda-q8_0 | cuda-q4_0 | cuda-q4_K | vulkan-f32 | vulkan-f16 | vulkan-q8_0 | vulkan-q4_0 | vulkan-q4_K | pytorch-cuda (ref) |
| text | 1555.2 (0.1x) | 1733.6 (0.1x) | 1670.6 (0.1x) | 1882.3 (0.1x) | 1211.2 (0.2x) | 52.8 (4.4x) | 41.4 (5.6x) | 41.3 (5.6x) | 40.7 (5.7x) | 41.2 (5.6x) | 51.5 (4.5x) | 48.9 (4.7x) | 51.7 (4.5x) | 52.5 (4.4x) | 52.4 (4.4x) | 231.5 |
| visual | 1625.5 (0.1x) | 1781.3 (0.1x) | 1759.3 (0.1x) | 1911.4 (0.1x) | 1342.0 (0.1x) | 49.8 (2.7x) | 42.4 (3.1x) | 40.7 (3.2x) | 41.7 (3.2x) | 40.0 (3.3x) | 47.3 (2.8x) | 44.6 (3.0x) | 50.2 (2.6x) | 50.0 (2.6x) | 47.4 (2.8x) | 131.9 |
| multimodal | 2102.7 (0.1x) | 2342.4 (0.1x) | 2188.4 (0.1x) | 2441.4 (0.1x) | 1583.5 (0.2x) | 60.4 (5.0x) | 55.1 (5.4x) | 47.0 (6.4x) | 45.5 (6.6x) | 46.0 (6.5x) | 57.1 (5.3x) | 52.3 (5.7x) | 56.7 (5.3x) | 59.7 (5.0x) | 55.5 (5.4x) | 299.8 |

## License

Same as the GKDT release: free for academic research and education,
commercial use prohibited.
