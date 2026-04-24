"""Plan: benchmark ESM-2 model-size trade-off (150M vs 650M).

Compares ``facebook/esm2_t30_150M_UR50D`` (150M params, 640-dim
embeddings) against the current production model
``facebook/esm2_t33_650M_UR50D`` (650M params, 1280-dim embeddings)
on two dimensions:

  * **Inference wall time** — raw embed throughput on 452 Syn3A CDS
  * **Downstream MCC** — retrain the Tier-1 XGBoost with 150M-derived
    features; compare against the falsified 650M number (0.443
    union MCC per ``mcc_tier1_xgboost_naive_stack``)

Both measurements require GPU. In sandbox this script writes a plan
only — the same pattern as ``bench_esm2_batch.py``. The XGBoost
retrain IS reproducible from the sandbox once the 150M parquet
exists, so the plan includes the explicit post-Colab workflow.

Why this matters: the Session-17 Tier-1 result blames the 1280:455
feature:row ratio. A 640-dim embedding would halve that ratio — if
the MCC drop from 650M to 150M is small (say -0.02), the 150M path
becomes a cheaper default AND a better match to the supervised-
learning signal available in small-organism data.

Usage::

    python scripts/bench_esm2_sizes.py \\
        --out outputs/bench_esm2_sizes_plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CANDIDATE_MODELS = [
    {
        "model_id": "facebook/esm2_t30_150M_UR50D",
        "params_M": 150,
        "embedding_dim": 640,
        "layers": 30,
        "role": "candidate",
    },
    {
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "params_M": 650,
        "embedding_dim": 1280,
        "layers": 33,
        "role": "production",
        "baseline_tier1_union_mcc": 0.443,
        "baseline_tier1_only_mcc": 0.145,
    },
]


def _probe_device() -> dict:
    info = {"cuda": False}
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
        info["note"] = "torch not installed"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/bench_esm2_sizes_plan.json",
    )
    args = ap.parse_args()

    device = _probe_device()
    plan = {
        "status": "AWAITING_COLAB",
        "models": CANDIDATE_MODELS,
        "device_probe": device,
        "inference_measurement": {
            "input": "452 Syn3A CDS sequences",
            "batch_size": 16,
            "dtype": "fp16 LM + fp32 trunk (N/A for pure embedding)",
            "metrics": [
                "wall_s_total",
                "wall_s_mean_per_sequence",
                "vram_peak_mib",
            ],
        },
        "downstream_measurement": {
            "description": (
                "Re-populate cell_sim/features/cache/esm2_150M.parquet "
                "on Colab GPU, then retrain Tier-1 XGBoost in sandbox "
                "with the 150M embeddings and compare the 5-fold CV "
                "MCC against the falsified 650M numbers."
            ),
            "sandbox_follow_up_command": (
                "python scripts/train_tier1_xgboost.py "
                "--out outputs/tier1_xgboost_150M.json "
                "# after manually copying esm2_150M.parquet + updating "
                "the script's cache path"
            ),
            "baseline_to_beat_union": 0.443,
            "baseline_to_beat_tier1_only": 0.145,
            "decision_rule": (
                "If 150M tier1_only MCC >= 0.120 AND union MCC >= 0.420, "
                "150M is the better default (half the dims, comparable "
                "signal). If either drops by more than 0.05, 650M "
                "stays."
            ),
        },
        "colab_invocation": {
            "cell_0": (
                "# Prereqs\n"
                "!pip install -q transformers>=4.40 biopython>=1.83 "
                "pyarrow>=14 pandas>=2"
            ),
            "cell_1": (
                "# Clone + stage\n"
                "!git clone --branch "
                "claude/syn3a-whole-cell-simulator-REjHC "
                "https://github.com/Nikku03/cell.git /content/cell\n"
                "%cd /content/cell\n"
                "import sys\n"
                "sys.path.insert(0, '/content/cell')"
            ),
            "cell_2": (
                "# Embed 452 CDS with 150M model\n"
                "from cell_sim.features.extractors import ESM2Extractor\n"
                "from cell_sim.features.batched_inference import (\n"
                "    BatchedInferenceConfig,\n"
                ")\n"
                "import pandas as pd\n"
                "from Bio import SeqIO\n"
                "rec = next(SeqIO.parse(\n"
                "    'cell_sim/data/Minimal_Cell_ComplexFormation/"
                "input_data/syn3A.gb', 'genbank'\n"
                "))\n"
                "rows = []\n"
                "for f in rec.features:\n"
                "    if f.type == 'CDS':\n"
                "        lt = (f.qualifiers.get('locus_tag') or [''])[0]\n"
                "        tr = (f.qualifiers.get('translation') or [''])[0]\n"
                "        if lt and tr:\n"
                "            rows.append({'locus_tag': lt, "
                "'sequence': tr.upper()})\n"
                "df = pd.DataFrame(rows)\n"
                "ex = ESM2Extractor(model_id='facebook/esm2_t30_150M_UR50D')"
                "  # NOTE: requires extractor arg support\n"
                "feats = ex.extract(df, BatchedInferenceConfig(\n"
                "    batch_size=16, device='cuda', dtype='float16',\n"
                "    max_seq_length=1022,\n"
                "))\n"
                "feats.to_parquet('cell_sim/features/cache/esm2_150M.parquet')"
            ),
        },
        "blocking_codechange": (
            "ESM2Extractor.__init__ currently hardcodes the model_id "
            "constant _MODEL_ID = 'facebook/esm2_t33_650M_UR50D'. "
            "Before running this plan, one of: (a) add a model_id "
            "kwarg to ESM2Extractor, or (b) create a parallel "
            "ESM2_150M_Extractor subclass that overrides _MODEL_ID. "
            "Choose (a) once the measurement decides 150M is worth "
            "integrating; otherwise no code change is needed."
        ),
        "notes_for_reviewer": [
            "The 650M baseline numbers come from "
            "memory_bank/facts/measured/mcc_tier1_xgboost_naive_stack.json.",
            "The 150M result is NOT yet measured. This fact file "
            "is a plan, not a conclusion.",
            "A meaningful positive result would be 150M tier1_only "
            "MCC in [0.12, 0.18] combined with 4-6x faster inference. "
            "A meaningful negative would be a collapse below 0.08.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))
    print(f"wrote {args.out}  status={plan['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
