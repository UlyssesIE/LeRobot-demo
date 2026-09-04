#!/usr/bin/env python
"""Environment checker for the 6GB-VRAM VLA fine-tuning demo.

Targets the LeRobot 0.5.x ecosystem: draccus config (no Hydra), Python >= 3.12,
CUDA-enabled torch 2.8+ (but <2.11, the hard upper bound of LeRobot 0.5.1).

Exit code: 0 = all *required* checks pass, 1 = something missing/mismatched.
"""

from __future__ import annotations

import importlib.metadata as pkg_meta
import platform
import sys

try:
    from packaging.version import Version
except ImportError:  # packaging ships with pip, should always be present
    Version = None

# Windows PowerShell GBK safety for emoji / unicode output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Dependency spec: (dist_name, min_version, max_version, required, note)
# min/max are PEP 440 strings; None = "no constraint".
# torch/torchvision max versions mirror LeRobot 0.5.1's real constraints:
#   torch<2.11.0,>=2.7   torchvision<0.26.0,>=0.22.0
# --------------------------------------------------------------------------- #
CORE = [
    ("torch",           "2.8.0",  "2.11.0", True, "CUDA build — see CUDA block below"),
    ("torchvision",     "0.22.0", "0.26.0", True, "paired with torch 2.10.x"),
    ("lerobot",         "0.5.0",  "0.6.0",  True, "draccus-era LeRobot (0.5.x)"),
    ("draccus",         "0.7.0",  None,     True, "config framework (replaces Hydra)"),
    ("transformers",    "5.0.0",  None,     True, "v5"),
    ("datasets",        "3.0.0",  None,     True, ""),
    ("huggingface-hub", "0.30.0", None,     True, ""),
    ("peft",            "0.14.0", None,     True, "LoRA / QLoRA"),
    ("diffusers",       "0.30.0", None,     True, "Diffusion head (Octo-93M)"),
    ("accelerate",      "1.0.0",  None,     True, ""),
    ("wandb",           "0.17.0", None,     True, "logging"),
    ("matplotlib",      "3.7.0",  None,     True, "training-curve GIF"),
    ("numpy",           "1.26.0", None,     True, ""),
]

OPTIONAL = [
    ("bitsandbytes",    None, None, "OpenVLA-7B QLoRA (8-bit), optional"),
    ("onnx",            None, None, "deployment export"),
    ("onnxruntime-gpu", None, None, "deployment inference"),
]

DEV_TOOLS = [
    ("pytest",     None, None, "tests"),
    ("ruff",       None, None, "lint"),
    ("mypy",       None, None, "type check"),
    ("pre-commit", None, None, "git hooks"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def pkg_version(name: str) -> str | None:
    """Return installed version of a distribution, or None if absent."""
    try:
        return pkg_meta.version(name)
    except pkg_meta.PackageNotFoundError:
        return None


def check_version(actual: str | None, min_v: str | None, max_v: str | None) -> tuple[bool, str]:
    """Return (ok, human-readable message) for a version-range check."""
    if actual is None:
        return False, "NOT INSTALLED"
    if Version is None:  # fallback: cannot compare, treat as present
        return True, actual
    v = Version(actual)
    if min_v is not None and v < Version(min_v):
        return False, f"{actual}  (< {min_v})"
    if max_v is not None and v >= Version(max_v):
        return False, f"{actual}  (>= {max_v})"
    return True, actual


def check_python() -> tuple[bool, str]:
    info = platform.python_version()
    ok = sys.version_info >= (3, 12)
    return ok, info


def check_cuda() -> list[tuple[bool, str]]:
    """Report CUDA availability / device / VRAM from torch."""
    rows: list[tuple[bool, str]] = []
    try:
        import torch
    except ImportError:
        return [(False, "torch import failed — reinstall CUDA build")]

    rows.append((True, f"torch {torch.__version__}"))
    rows.append((True, f"torch.version.cuda = {torch.version.cuda}"))

    avail = torch.cuda.is_available()
    rows.append((avail, f"cuda.is_available() = {avail}"))

    if avail:
        name = torch.cuda.get_device_name(0)
        rows.append((True, f"device = {name}"))
        total = torch.cuda.get_device_properties(0).total_memory
        rows.append((True, f"VRAM = {total / 1024**3:.1f} GB"))
    else:
        rows.append((False, "CUDA unavailable — torch was installed as CPU-only "
                            "(reinstall via --index-url .../whl/cu126)"))
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    failures = 0

    print("=" * 72)
    print(" Environment check: 6GB-VRAM VLA fine-tuning (LeRobot 0.5.x)")
    print("=" * 72)

    # --- Python ---
    ok, v = check_python()
    mark = "✅" if ok else "❌"
    print(f"\n[Python] {mark} {v}  (required: >=3.12)")
    if not ok:
        failures += 1

    # --- Core dependencies ---
    print("\n[Core dependencies]")
    for name, min_v, max_v, required, note in CORE:
        actual = pkg_version(name)
        ok, msg = check_version(actual, min_v, max_v)
        mark = "✅" if ok else "❌"
        constraint = f"{min_v or '*'}"
        if max_v:
            constraint += f", <{max_v}"
        suffix = f"  — {note}" if note else ""
        print(f"  {mark} {name:<17} {msg:<20} (need: {constraint}){suffix}")
        if required and not ok:
            failures += 1

    # --- CUDA block ---
    print("\n[CUDA]")
    for ok, msg in check_cuda():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {msg}")
        if not ok:
            failures += 1

    # --- Optional ---
    print("\n[Optional]")
    for name, _min, _max, note in OPTIONAL:
        actual = pkg_version(name)
        if actual:
            print(f"  ✅ {name:<17} {actual}  — {note}")
        else:
            print(f"  ⚠️  {name:<17} not installed  — {note}")

    # --- Dev tools ---
    print("\n[Dev tools]")
    for name, _min, _max, note in DEV_TOOLS:
        actual = pkg_version(name)
        mark = "✅" if actual else "⚠️ "
        print(f"  {mark} {name:<17} {actual or 'not installed'}  — {note}")

    # --- Summary ---
    print("\n" + "=" * 72)
    if failures == 0:
        print(" ✅ ALL REQUIRED CHECKS PASSED")
        print("=" * 72)
        return 0
    print(f" ❌ {failures} required check(s) failed — fix before training")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
