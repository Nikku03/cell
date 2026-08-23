"""ESM-2 650M embeddings for a 500-protein screening subset. A cache, not a test -- no gates.

WHY 500 AND NOT ALL 2,178. Measured on this machine: 2.58 s/protein at 6.48 ms/residue on 4 CPU
threads, 6.4x slower than the 35M model and 15x slower than the 8M. The full set is 1.6 h of
forward passes plus a 234 s checkpoint load; 500 is 21.5 min. That buys a SCREEN and not a test,
and loop 164 states the difference in its own gates rather than leaving it to be discovered.

THE SUBSET IS DRAWN ONCE, at seed 16500, from the 2,178 proteins that already carry sequence,
geometry, electrostatic and steric blocks -- so every arm is scored on identical proteins and the
650M arm gets no coverage advantage over the 35M one it is being compared against.

-> colab/data/ml/esm650_subset.npz
"""
import gzip
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loop_replication as LR  # noqa: E402

ACCS = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/enz_accs.json")
OUT = Path("colab/data/ml/esm650_subset.npz")
N_SUB, SEED, MAXLEN, BATCH_RES = 500, 16500, 1022, 4000


def main():
    import torch
    import esm
    torch.set_num_threads(4)
    t0 = time.time()
    want = set(json.load(open(ACCS)))
    seqs, acc, buf = {}, None, []
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                if acc and buf and acc in want:
                    seqs[acc] = "".join(buf)
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                acc, buf = (m.group(1) if m else None), []
            else:
                buf.append(ln.strip())
    if acc and buf and acc in want:
        seqs[acc] = "".join(buf)

    rng = np.random.default_rng(SEED)
    pool = sorted(seqs)
    sub = sorted(rng.choice(pool, size=min(N_SUB, len(pool)), replace=False).tolist())
    print(f"{len(sub)} of {len(pool):,} proteins sampled at seed {SEED}; "
          f"{sum(len(seqs[a]) for a in sub):,} residues", flush=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()
    nl = model.num_layers
    print(f"model loaded [{time.time()-t0:.0f}s]", flush=True)

    order = sorted(sub, key=lambda a: len(seqs[a]))
    accs, embs, batch, blen, done = [], [], [], 0, 0

    def flush(b):
        if not b:
            return
        _, _, toks = bc([(a, seqs[a][:MAXLEN]) for a in b])
        with torch.no_grad():
            r = model(toks, repr_layers=[nl])["representations"][nl]
        for i, a in enumerate(b):
            L = min(len(seqs[a]), MAXLEN)
            accs.append(a)
            embs.append(r[i, 1:L + 1].mean(0).numpy())

    for a in order:
        L = min(len(seqs[a]), MAXLEN)
        if blen + L > BATCH_RES and batch:
            flush(batch)
            done += len(batch)
            if done % 50 < len(batch):
                el = time.time() - t0
                rate = done / max(el - 234, 1)
                print(f"  {done}/{len(order)} [{el:.0f}s, {rate:.2f} prot/s, "
                      f"eta {(len(order)-done)/max(rate,1e-6)/60:.1f} min]", flush=True)
            batch, blen = [], 0
        batch.append(a)
        blen += L
    flush(batch)

    X = np.array(embs, np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, accs=np.array(accs), esm650=X, seed=SEED)
    print(f"wrote {X.shape} -> {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
