#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Tap-level parity diff between the official PyTorch dumps (scripts/dump_taps.py)
# and the C++ ggml runtime dumps (gkd-cli --dump-taps).
#
# Files are [int64 ndims, int64 dims..., float32 data...]. The C++ engine writes
# tensors in the same row-major order as the PyTorch reference.
#
# Usage: python3 parity_reference.py diff /tmp/gkd_taps_text /tmp/gkd_cpp_taps
# ------------------------------------------------------------------------------
import os
import struct
import sys

import numpy as np

TOL_MEAN = 2e-3
TOL_P99 = 5e-2
TOL_MAX = 5e-1


def load(path):
    with open(path, "rb") as f:
        ndims = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack("<" + "q" * ndims, f.read(8 * ndims))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return shape, data


def diff_tap(name, ref_dir, cpp_dir):
    ref_path = os.path.join(ref_dir, name + ".bin")
    cpp_path = os.path.join(cpp_dir, "cpp_" + name + ".bin")
    if not os.path.exists(ref_path):
        return None  # tap not produced by the reference run
    if not os.path.exists(cpp_path):
        return (name, "MISSING", "")
    rs, rd = load(ref_path)
    cs, cd = load(cpp_path)
    if rd.size != cd.size:
        return (name, "SIZE", f"ref shape={list(rs)} cpp shape={list(cs)}")
    d = np.abs(rd.astype(np.float64) - cd.astype(np.float64))
    scale = max(float(np.abs(rd).max()), 1e-6)
    stats = (float(d.mean()), float(np.percentile(d, 99)) if d.size else 0.0, float(d.max()))
    status = "OK" if (stats[0] < TOL_MEAN and stats[1] < TOL_P99 and stats[2] < TOL_MAX) else "FAIL"
    detail = (f"mean={stats[0]:.3e} p99={stats[1]:.3e} max={stats[2]:.3e} "
              f"(ref |max|={scale:.3f})")
    return (name, status, detail)


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "diff":
        print(__doc__)
        sys.exit(1)
    ref_dir, cpp_dir = sys.argv[2], sys.argv[3]
    names = sorted({f[:-4] for f in os.listdir(ref_dir) if f.endswith(".bin")})
    n_ok = n_fail = 0
    print(f"{'tap':<18} {'status':<8} detail")
    print("-" * 78)
    for name in names:
        r = diff_tap(name, ref_dir, cpp_dir)
        if r is None:
            continue
        name, status, detail = r
        print(f"{name:<18} {status:<8} {detail}")
        if status == "OK":
            n_ok += 1
        elif status == "FAIL":
            n_fail += 1
    print("-" * 78)
    print(f"{n_ok} ok, {n_fail} failed (tolerances: mean<{TOL_MEAN}, p99<{TOL_P99}, max<{TOL_MAX})")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
