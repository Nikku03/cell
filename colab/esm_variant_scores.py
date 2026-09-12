"""ESM-2 ZERO-SHOT VARIANT SCORES -- the cache, not the experiment.

This builds one thing and tests nothing: for every protein carrying ClinVar missense variants in
loop_chain's set, the full L x 20 matrix of log P(aa | context). The variant score is then
log P(alt) - log P(ref) at the variant's residue, which is the standard zero-shot readout and costs
ONE forward pass per protein window rather than one per variant -- 52,922 variants over 276 genes
become ~900 passes.

IT SEES NO LABELS. Nothing in this file reads whether a variant is pathogenic or benign, so it
cannot be tuned toward an answer, and it is separated from loop_rescue.py for exactly that reason.

TWO MODELS, and the reason for both. ESM-2 650M is the largest that runs here (8.5 s per 1022-token
window on 4 CPU cores, no GPU). ESM-2 35M is 19x smaller and 9x faster. Both are scored so the
experiment can ask whether any result depends on model capacity -- a signal that appears at 35M and
does not grow at 650M is a different kind of claim from one that scales.

WINDOWING. ESM-2's context is 1022 residues and 162 of these 276 proteins are longer, up to titin at
35,991. Windows are tiled with 50% overlap and each position takes its log-probs from the window
where it sits most CENTRALLY, so no residue is scored from the edge of a context when a better one
exists. Scoring a position at a window boundary would silently degrade exactly the long proteins
that carry the most variants.

-> scratchpad/_esm_scores_{model}.pkl : {gene: (sequence, float32 [L, 20] log-probs)}
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

SC = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

WINDOW = 1022
STRIDE = 511
AA = "ACDEFGHIKLMNPQRSTVWY"

_B = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: _AAS[i] for i, (a, b, c) in
         enumerate((x, y, z) for x in _B for y in _B for z in _B)}

MODELS = {"35M": "esm2_t12_35M_UR50D", "650M": "esm2_t33_650M_UR50D"}


def translate(gene, cds, best):
    e = best.get(gene)
    if not e:
        return ""
    s = cds.get(e[1])
    if not s:
        return ""
    p = "".join(CODON.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))
    return p.split("*")[0] if "*" in p else p


def windows(L):
    """(start, end) tiles covering [0, L) with 50% overlap; a short protein gets one."""
    if L <= WINDOW:
        return [(0, L)]
    out = []
    s = 0
    while s < L:
        e = min(s + WINDOW, L)
        out.append((s, e))
        if e == L:
            break
        s += STRIDE
    return out


def score_protein(seq, model, alph, bc, aa_idx):
    """log-softmax over the 20 standard amino acids at every position.

    Each position keeps the window in which it is most central. A residue scored from the edge of a
    1022-token context has seen half the neighbourhood it should have, and the long proteins here
    are precisely the ones carrying the most variants.
    """
    L = len(seq)
    out = np.full((L, 20), np.nan, np.float32)
    best_centrality = np.full(L, -1.0, np.float32)
    for a, b in windows(L):
        sub = seq[a:b]
        _, _, toks = bc([("x", sub)])
        with torch.no_grad():
            logits = model(toks)["logits"][0]
        lp = torch.log_softmax(logits, dim=-1)[1:len(sub) + 1, aa_idx].numpy()
        mid = (a + b) / 2.0
        half = max((b - a) / 2.0, 1.0)
        cen = 1.0 - np.abs(np.arange(a, b) - mid) / half
        take = cen > best_centrality[a:b]
        out[a:b][take] = lp[take]
        best_centrality[a:b][take] = cen[take]
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "650M"
    name = MODELS[which]
    t0 = time.time()
    torch.set_num_threads(4)

    cds = pickle.load(open(SC / "_chain_cdsseq.pkl", "rb"))
    best = pickle.load(open(SC / "_chain_txmodel.pkl", "rb"))["best"]
    tr = pickle.load(open(SC / "_chain_translated.pkl", "rb"))
    mis = [x for x in tr if "missense" in x[6]]

    # WHICH GENES GET SCORED, and why it is not "all of them". 15,025 genes carry missense variants,
    # 9.8 M residues, which at 650M and 8.5 s per 1022-residue window is roughly 45 hours. So the set
    # is bounded by a SAMPLE-SIZE criterion: a gene needs at least min_class variants of EACH label,
    # because a within-gene AUC over three benign variants is not a measurement. That criterion reads
    # the label COUNTS and never the scores, so it cannot pull the result in either direction -- but
    # it is a choice made here rather than in the experiment, and it is recorded rather than implied.
    # Declared split: 650M over min_class=20 (276 genes), 35M over min_class=5 (1,507), so the small
    # model also answers whether the result survives a four-times-wider and thinner gene set.
    min_class = int(sys.argv[2]) if len(sys.argv) > 2 else (20 if which == "650M" else 5)
    per = {}
    for x in mis:
        d = per.setdefault(x[4], [0, 0])
        d[x[5]] += 1
    genes = sorted(g for g, (n0, n1) in per.items() if n0 >= min_class and n1 >= min_class)
    print(f"  min_class {min_class} -> {len(genes):,} genes of {len(per):,} with missense variants")
    seqs = {}
    for g in genes:
        p = translate(g, cds, best)
        if p and len(p) >= 20:
            seqs[g] = p
    # scored longest-last so a crash mid-run still leaves most of the set done
    order = sorted(seqs, key=lambda g: len(seqs[g]))
    print(f"  {len(order):,} genes, {sum(len(seqs[g]) for g in order):,} residues, model {which}")

    import esm
    model, alph = getattr(esm.pretrained, name)()
    model.eval()
    bc = alph.get_batch_converter()
    aa_idx = [alph.get_idx(a) for a in AA]

    dest = SC / f"_esm_scores_{which}.pkl"
    done = {}
    if dest.exists():
        done = pickle.load(open(dest, "rb"))
        print(f"  resuming: {len(done):,} genes already cached")

    t = time.time()
    for i, g in enumerate(order):
        if g in done:
            continue
        done[g] = (seqs[g], score_protein(seqs[g], model, alph, bc, aa_idx))
        if (i + 1) % 25 == 0:
            el = time.time() - t
            print(f"  {i+1:,}/{len(order):,}  {el/60:.1f} min  "
                  f"eta {el/max(i+1,1)*(len(order)-i-1)/60:.0f} min", flush=True)
            pickle.dump(done, open(dest, "wb"), protocol=4)
    pickle.dump(done, open(dest, "wb"), protocol=4)
    print(f"  -> {dest}  {len(done):,} genes in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
