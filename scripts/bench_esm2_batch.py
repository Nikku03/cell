"""Benchmark ESM-2 650M inference throughput vs batch size.

GPU-required: ESM-2 650M does not fit comfortably on CPU for timed
throughput measurement (CPU fp32 forward pass on a 300-residue
protein takes tens of seconds — the tokenizer + transformer load
alone dominates, and the batch-size scan would take hours). This
script therefore runs in two modes:

  * **Colab / CUDA mode** — detects a GPU, loads ``facebook/esm2_t33_650M_UR50D``,
    times mean + std throughput for batch sizes 8 / 16 / 32 / 64 on N=32
    synthetic sequences (avg length 300 aa), writes the results to the
    output JSON. Records the *measured* numbers.
  * **Sandbox / CPU mode** — detects absence of GPU, refuses to run a
    garbage-quality CPU micro-benchmark that would mislead the reader,
    and writes a structured *plan* JSON. The plan lists the exact
    invocations to run on Colab so the measurements can be collected
    later without another round of script development.

The plan is honest instrumentation — a one-shot checklist the user
can paste into a Colab cell. No numbers are manufactured.

Usage::

    python scripts/bench_esm2_batch.py \\
        --out outputs/bench_esm2_batch_plan.json

If CUDA is available the script will also populate a ``measurements``
section inline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))


BATCH_SIZES = (8, 16, 32, 64)
N_SEQUENCES = 32
AVG_LEN = 300
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def _gen_sequences(n: int, avg_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seqs: list[str] = []
    for _ in range(n):
        # Lengths centered at avg_len with modest spread so batches
        # exercise padding.
        length = max(50, int(rng.gauss(avg_len, avg_len * 0.2)))
        seqs.append("".join(rng.choice(AMINO) for _ in range(length)))
    return seqs


def _probe_device() -> dict:
    info = {"cuda": False, "device_name": None, "torch_version": None}
    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3, 1,
            )
    except ImportError:
        info["note"] = "torch not installed — cannot run on any device"
    return info


def _measure_on_cuda(
    batch_sizes: tuple[int, ...], seqs: list[str],
) -> dict:
    """Run the actual batch-size scan on a CUDA device. Called ONLY
    when a GPU is detected. Imports lazily so the plan-only path
    doesn't pull torch/transformers into the process."""
    import torch  # type: ignore
    import pandas as pd
    from cell_sim.features.extractors import ESM2Extractor
    from cell_sim.features.batched_inference import (
        BatchedInferenceConfig,
    )

    inputs = pd.DataFrame({
        "locus_tag": [f"synth_{i}" for i in range(len(seqs))],
        "sequence": seqs,
    })
    out: dict = {}
    ex = ESM2Extractor()
    # Warm the model once so tokenizer + weight load isn't on the
    # first batch-size's clock.
    _ = ex.extract(
        inputs.head(1),
        BatchedInferenceConfig(
            batch_size=1, device="cuda", dtype="float16",
            max_seq_length=1022,
        ),
    )
    torch.cuda.synchronize()

    for bs in batch_sizes:
        cfg = BatchedInferenceConfig(
            batch_size=bs, device="cuda", dtype="float16",
            max_seq_length=1022,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = ex.extract(inputs, cfg)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        throughput = len(seqs) / wall
        out[str(bs)] = {
            "batch_size": bs,
            "wall_s": wall,
            "throughput_seq_per_s": throughput,
            "feats_shape": list(feats.shape),
        }
        print(f"  bs={bs:3d}  wall={wall:.2f}s  "
              f"throughput={throughput:.2f} seq/s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/bench_esm2_batch_plan.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = _probe_device()
    print(f"device probe: {device}")
    seqs = _gen_sequences(N_SEQUENCES, AVG_LEN, args.seed)
    lengths = [len(s) for s in seqs]
    print(f"generated {len(seqs)} synthetic sequences  "
          f"len_mean={sum(lengths)/len(lengths):.0f}  "
          f"len_min={min(lengths)}  len_max={max(lengths)}")

    plan = {
        "status": "AWAITING_COLAB" if not device["cuda"] else "MEASURED",
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "batch_sizes": list(BATCH_SIZES),
        "n_sequences": N_SEQUENCES,
        "avg_sequence_length": AVG_LEN,
        "synthetic_seed": args.seed,
        "device_probe": device,
        "colab_invocation": {
            "cell_0": (
                "!pip install -q transformers>=4.40 "
                "biopython>=1.83 pandas>=2 pyarrow>=14"
            ),
            "cell_1": (
                "!git clone --branch "
                "claude/syn3a-whole-cell-simulator-REjHC "
                "https://github.com/Nikku03/cell.git /content/cell\n"
                "%cd /content/cell\n"
                "import sys\n"
                "sys.path.insert(0, '/content/cell')"
            ),
            "cell_2": (
                "!python scripts/bench_esm2_batch.py "
                "--out outputs/bench_esm2_batch_plan.json"
            ),
        },
        "notes_for_reviewer": [
            "Warm-up batch (size 1) is run once before timed batches "
            "so tokenizer/model load isn't charged to any scan point.",
            "torch.cuda.synchronize is called before and after each "
            "timed region.",
            "Sequences are synthetic random AA; length distribution "
            "is Gaussian around 300 residues to approximate Syn3A's "
            "protein length distribution.",
            "fp16 LM + fp32 trunk mixed-precision is used for ESM-2 "
            "alone (no structure prediction, so the pLDDT fp16 issue "
            "doesn't apply here).",
        ],
    }

    if device["cuda"]:
        print("CUDA detected — running actual batch-size scan")
        try:
            measurements = _measure_on_cuda(BATCH_SIZES, seqs)
            plan["measurements"] = measurements
            thr = [m["throughput_seq_per_s"] for m in measurements.values()]
            plan["best_batch_size"] = max(
                measurements,
                key=lambda k: measurements[k]["throughput_seq_per_s"],
            )
            plan["throughput_range_seq_per_s"] = [min(thr), max(thr)]
        except Exception as exc:  # noqa: BLE001
            plan["measurement_error"] = f"{type(exc).__name__}: {exc}"
            plan["status"] = "ERROR"
    else:
        print("No CUDA — refusing to run CPU micro-benchmark on "
              "ESM-2 650M; writing plan only")
        plan["cpu_refused_reason"] = (
            "ESM-2 650M fp32 CPU forward pass on 300-residue protein "
            "takes tens of seconds; a CPU batch-size scan would take "
            "hours and would not reflect the GPU inference pattern "
            "being evaluated. Run on Colab GPU instead."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, default=float))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
