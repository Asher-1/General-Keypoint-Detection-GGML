#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression guard for the cpp_ggml runtime: every configuration is passed
explicitly (CLI options / CMake -D cache variables / function arguments),
never through environment variables.

Scans the self-owned cpp_ggml code (the vendored third_party trees and cmake
build-* trees are excluded) and fails if any of the following appears:

  - C/C++:   getenv / setenv / putenv / _putenv_s
  - CMake:   $ENV{...}
  - Python:  os.environ / os.getenv
  - Shell:   export VAR=... (configuration through the environment)

Rationale: an env read makes behavior depend on the caller's shell session
(hidden input), which breaks reproducibility; an explicit option fails loudly
instead of silently drifting.

Usage:  python3 scripts/check_no_env.py    (from cpp_ggml/, or anywhere)
Exit:   0 = clean, 1 = violations found
"""
import argparse
import os
import re
import sys

# cpp_ggml/ root (this script lives in cpp_ggml/scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Vendored upstream trees and build trees: their env interfaces are the
# upstream projects' public configuration APIs and must not be modified here.
EXCLUDED_DIRS = {"third_party", "__pycache__"}
BUILD_PREFIX = "build-"  # any cmake build tree (build-cpu, build-cuda, ...)

PY_RULES = [re.compile(p) for p in (
    r"os\.environ",
    r"os\.getenv",
    r"(?<![.\w])getenv\s*\(",
)]
C_RULES = [re.compile(p) for p in (
    r"\bgetenv\s*\(",
    r"\bsetenv\s*\(",
    r"\bputenv\s*\(",
    r"\b_putenv_s\s*\(",
)]
CMAKE_RULES = [re.compile(r"ENV\{")]
SH_RULES = [re.compile(r"^\s*export\s+[A-Za-z_][A-Za-z0-9_]*=")]

EXT_RULES = {".py": PY_RULES}
for _ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".inc"):
    EXT_RULES[_ext] = C_RULES
SH_EXTS = (".sh", ".bash")


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in EXCLUDED_DIRS and not d.startswith(BUILD_PREFIX)]
        for fn in sorted(filenames):
            if fn == "check_no_env.py":  # do not match this file's own regexes
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            if fn == "CMakeLists.txt" or fn.endswith(".cmake"):
                yield path, rel, CMAKE_RULES
            elif os.path.splitext(fn)[1] in EXT_RULES:
                yield path, rel, EXT_RULES[os.path.splitext(fn)[1]]
            elif os.path.splitext(fn)[1] in SH_EXTS:
                yield path, rel, SH_RULES


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every scanned file, not only violations")
    args = ap.parse_args()

    violations = 0
    scanned = 0
    for path, rel, rules in iter_files():
        try:
            with open(path, encoding="utf-8", errors="strict") as f:
                lines = f.readlines()
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: skip
        scanned += 1
        if args.verbose:
            print(f"  scanned {rel}")
        for no, line in enumerate(lines, 1):
            hit = next((r for r in rules if r.search(line)), None)
            if hit is None:
                continue
            violations += 1
            print(f"{rel}:{no}: {line.rstrip()}")
            print("    cpp_ggml takes every setting as an explicit option; "
                  "env switches are forbidden.")
    print(f"\ncheck_no_env: {scanned} files scanned, {violations} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
