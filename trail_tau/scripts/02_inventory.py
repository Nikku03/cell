#!/usr/bin/env python3
"""Phase 0 inventory. Loads every .mat and reports structure. NO ANALYSIS."""
import sys, os, glob, hashlib
import numpy as np
import scipy.io as sio

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")

def describe(a, name, indent="    "):
    if isinstance(a, np.ndarray):
        print(f"{indent}{name}: ndarray shape={a.shape} dtype={a.dtype}")
        if a.dtype == object:
            print(f"{indent}  object array; element shapes: "
                  f"{[getattr(x,'shape',type(x).__name__) for x in a.ravel()[:6]]}")
            for i, x in enumerate(a.ravel()[:3]):
                if isinstance(x, np.ndarray) and x.size:
                    print(f"{indent}   [{i}] shape={x.shape} dtype={x.dtype} "
                          f"first={np.asarray(x).ravel()[:5]}")
        elif a.size:
            flat = a.ravel()
            print(f"{indent}  min={np.nanmin(flat):.6g} max={np.nanmax(flat):.6g} "
                  f"n_nan={int(np.sum(np.isnan(flat))) if np.issubdtype(a.dtype,np.floating) else 0}")
            print(f"{indent}  first 8: {flat[:8]}")
    else:
        print(f"{indent}{name}: {type(a).__name__} = {a!r}"[:300])

files = sorted(f for f in glob.glob(os.path.join(RAW, "**", "*.mat"), recursive=True)
               if "__MACOSX" not in f)
print(f"# MAT FILES: {len(files)}\n")
for f in files:
    rel = os.path.relpath(f, RAW)
    try:
        d = sio.loadmat(f, squeeze_me=False, struct_as_record=False)
    except Exception as e:
        print(f"## {rel}\n    LOAD FAILED: {e}\n"); continue
    keys = [k for k in d if not k.startswith("__")]
    print(f"## {rel}  ({os.path.getsize(f)} bytes)  vars={keys}")
    for k in keys:
        describe(d[k], k)
    print()
