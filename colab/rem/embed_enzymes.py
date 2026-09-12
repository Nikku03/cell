"""ESM-2 embeddings for the 2,178 Human-GEM enzymes that also have an AlphaFold structure. A cache.

The set is chosen by INTERSECTION on purpose: every protein here has both a sequence and a
structure, so the sequence arm and the structure arm are scored on identical proteins and neither
gets a coverage advantage. Loop 156 found ESM-2 35M beat 8M on a different target (+0.3668 vs
+0.3237), so 35M is used; the 8M is embedded too, because loop 157 measured that the two sizes
interact (+0.3339 as a pair) rather than one simply dominating.

-> colab/data/ml/esm_enzymes.npz
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
OUT = Path("colab/data/ml/esm_enzymes.npz")
MAXLEN, BATCH_RES = 1022, 6000


def main():
    import torch
    import esm
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
    print(f"{len(seqs):,} sequences", flush=True)

    out = {}
    for tagname, loader in (("esm35", esm.pretrained.esm2_t12_35M_UR50D),
                            ("esm8", esm.pretrained.esm2_t6_8M_UR50D)):
        model, alphabet = loader()
        model.eval()
        bc = alphabet.get_batch_converter()
        nl = model.num_layers
        order = sorted(seqs, key=lambda a: len(seqs[a]))
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
                if done % 400 < len(batch):
                    print(f"  {tagname} {done:,}/{len(order):,} [{time.time()-t0:.0f}s]", flush=True)
                batch, blen = [], 0
            batch.append(a)
            blen += L
        flush(batch)
        out[tagname] = (np.array(accs), np.array(embs, np.float32))
        print(f"  {tagname} done {out[tagname][1].shape} [{time.time()-t0:.0f}s]", flush=True)

    a35, x35 = out["esm35"]
    a8, x8 = out["esm8"]
    assert list(a35) == list(a8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, accs=a35, esm35=x35, esm8=x8)
    print(f"wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
