# Speedup table

End-to-end single-image latency (ms), mean over the 15 official
test images x warmup/iters protocol in `run_all_tests.py`.
Reference: stock PyTorch GKDT-L on CUDA (RTX 4090).

| mode | cpu-f32 | cpu-f16 | cpu-q8_0 | cpu-q4_0 | cpu-q4_K | cuda-f32 | cuda-f16 | cuda-q8_0 | cuda-q4_0 | cuda-q4_K | vulkan-f32 | vulkan-f16 | vulkan-q8_0 | vulkan-q4_0 | vulkan-q4_K | pytorch-cuda (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text | 1555.2 (0.1x) | 1733.6 (0.1x) | 1670.6 (0.1x) | 1882.3 (0.1x) | 1211.2 (0.2x) | 52.8 (4.4x) | 41.4 (5.6x) | 41.3 (5.6x) | 40.7 (5.7x) | 41.2 (5.6x) | 51.5 (4.5x) | 48.9 (4.7x) | 51.7 (4.5x) | 52.5 (4.4x) | 52.4 (4.4x) | 231.5 |
| visual | 1625.5 (0.1x) | 1781.3 (0.1x) | 1759.3 (0.1x) | 1911.4 (0.1x) | 1342.0 (0.1x) | 49.8 (2.7x) | 42.4 (3.1x) | 40.7 (3.2x) | 41.7 (3.2x) | 40.0 (3.3x) | 47.3 (2.8x) | 44.6 (3.0x) | 50.2 (2.6x) | 50.0 (2.6x) | 47.4 (2.8x) | 131.9 |
| multimodal | 2102.7 (0.1x) | 2342.4 (0.1x) | 2188.4 (0.1x) | 2441.4 (0.1x) | 1583.5 (0.2x) | 60.4 (5.0x) | 55.1 (5.4x) | 47.0 (6.4x) | 45.5 (6.6x) | 46.0 (6.5x) | 57.1 (5.3x) | 52.3 (5.7x) | 56.7 (5.3x) | 59.7 (5.0x) | 55.5 (5.4x) | 299.8 |
