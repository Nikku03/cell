"""Embed the Rega-2025 half-life proteins with ESM-2, and cache. A cache, not a test -- no gates.

Rega et al. 2025 Nat Commun 16:2579 (doi 10.1038/s41467-025-57537-8, PMID 40089461) measured
protein half-lives in non-transformed hTERT-RPE-1 cells with a cell-cycle phase axis. 1,839
proteins carry a half-life and ALL 1,839 carry a UniProt accession that matches the SwissProt
human proteome exactly, so the join is by accession and not by gene symbol -- no symbol-mapping
ambiguity enters the dataset at all.

CHOICES, stated because they bound what any downstream test can show:
  * ESM-2 t6_8M_UR50D. Small on purpose: loop 133 found that a 650M encoder was worth less than
    fixing how the representation was pooled, so the encoder is the cheap part and the readout is
    where the work is. If 8M shows signal, size is the next knob; if it shows none, size was never
    the problem.
  * Mean pooling over residues, the standard unfitted choice. Loop 133 also recorded that mean
    pooling hides point mutants -- that is a known limitation of this representation and it is
    written down here rather than discovered later.
  * Sequences truncated to 1022 residues (ESM-2 context 1024 with BOS/EOS). The truncated count is
    reported. For long proteins the embedding sees only the N-terminal region, which matters
    because N-terminal sequence is where N-degrons live.
  * CPU only in this environment; batched by length to keep padding down.

-> colab/data/ml/esm2_8M_halflife.npz
"""
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_replication as LR  # noqa: E402

SC = LR.SC
REGA = SC / "destroyer" / "rega_4.xlsx"
FASTA = SC / "human_proteome.fasta.gz"
OUTF = Path("colab/data/ml/esm2_8M_halflife.npz")
MAXLEN = 1022
BATCH_RES = 8000          # residues per batch, keeps CPU memory flat


def main():
    t0 = time.time()
    import pandas as pd
    import torch
    import esm

    d = pd.read_excel(REGA, sheet_name="Proteome", header=1)
    for c in ("halflife_mean", "halflife_std", "halflife_count",
              "relative_abundance_8h_mean"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[np.isfinite(d["halflife_mean"]) & (d["halflife_mean"] > 0)].copy()
    want = {str(a): i for i, a in enumerate(d["Accession"].astype(str))}
    print(f"{len(want):,} proteins with a measured half-life", flush=True)

    seqs = {}
    acc = None
    buf = []
    with gzip.open(FASTA, "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if acc and buf and acc in want:
                    seqs[acc] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                acc = m.group(1) if m else None
                buf = []
            else:
                buf.append(ln.strip())
    if acc and buf and acc in want:
        seqs[acc] = "".join(buf)
    print(f"{len(seqs):,} sequences recovered by accession", flush=True)

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()
    n_layers = model.num_layers

    order = sorted(seqs, key=lambda a: len(seqs[a]))
    n_trunc = sum(1 for a in order if len(seqs[a]) > MAXLEN)
    print(f"{n_trunc:,} sequences exceed {MAXLEN} residues and are truncated", flush=True)

    accs, embs = [], []
    batch, blen = [], 0
    done = 0

    def flush(batch):
        if not batch:
            return
        data = [(a, seqs[a][:MAXLEN]) for a in batch]
        _, _, toks = bc(data)
        with torch.no_grad():
            out = model(toks, repr_layers=[n_layers])["representations"][n_layers]
        for i, a in enumerate(batch):
            L = min(len(seqs[a]), MAXLEN)
            v = out[i, 1:L + 1].mean(0).numpy()
            accs.append(a)
            embs.append(v)

    for a in order:
        L = min(len(seqs[a]), MAXLEN)
        if blen + L > BATCH_RES and batch:
            flush(batch)
            done += len(batch)
            if done % 200 < len(batch):
                print(f"  {done:,}/{len(order):,}  [{time.time() - t0:.0f}s]", flush=True)
            batch, blen = [], 0
        batch.append(a)
        blen += L
    flush(batch)
    done += len(batch)

    X = np.array(embs, np.float32)
    OUTF.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTF, X=X, accs=np.array(accs),
        lengths=np.array([len(seqs[a]) for a in accs], np.int32),
        model="esm2_t6_8M_UR50D", pooling="mean", maxlen=MAXLEN, n_truncated=n_trunc)
    print(f"wrote {X.shape} -> {OUTF}   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
