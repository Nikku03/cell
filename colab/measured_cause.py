"""MEASURED CAUSAL cause-finder — the alibi test done with REAL interventional data.

Every network-only cause-finder in this project hit the same wall: from a static correlation graph you
cannot separate the driver from its downstream effects. The fix is INTERVENTIONAL data — measured effects of
actually knocking each gene down. The Replogle-Nadig Perturb-seq screen provides exactly that: for ~2,000
gene knockdowns across 4 cell lines, the measured transcriptome-wide effect (pseudobulk delta per gene).

THE MEASURED ALIBI TEST
-----------------------
Given a disease signature s (genes UP=+1 / DOWN=-1) and, for each knockdown target T, its measured effect
vector e_T (delta of every gene when T is knocked down):

    reversal(T) = -cosine(e_T, s)

A high positive reversal means knocking T down pushes the cell OPPOSITE to disease — it *reverses* the
phenotype. That is causal, interventional evidence that T DRIVES the disease (and is a candidate target),
not mere correlation. A downstream effector's knockdown barely moves the whole signature -> low score.
Sign convention: reversal>0 => T drives disease (disable it); reversal<0 => T protects (its loss drives
disease). This is the real 'alibi test': the suspect whose removal undoes the crime.

Works on the Replogle `cell_eval` wide matrix (rows=knockdown target, cols=per-gene delta) or a dict of
effect vectors. Cell-line aware (Jurkat = T-cell line, closest to immune disease). No generic network needed.
"""
import json, math
from collections import defaultdict


def cosine(a, b):
    """cosine over the shared support; a,b are dict gene->value."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    nb = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (na * nb)


def signature_vector(up, down):
    s = {g: 1.0 for g in up}
    s.update({g: -1.0 for g in down})
    return s


def measured_alibi(signature, effect_vectors, restrict_to_signature=True):
    """effect_vectors: {target_gene: {measured_gene: delta}}. Returns ranked list of
    dict(target, reversal, drives_disease) — the measured knockout-reversal cause-ranking."""
    rows = []
    sgenes = set(signature)
    for T, e in effect_vectors.items():
        ev = {g: v for g, v in e.items() if g in sgenes} if restrict_to_signature else e
        if not ev:
            continue
        c = cosine(ev, signature)
        rows.append(dict(target=T, reversal=round(-c, 4), cosine_with_disease=round(c, 4),
                         drives_disease=bool(c < 0), n_signature_genes_hit=len(ev)))
    rows.sort(key=lambda r: -r["reversal"])
    return rows


def load_cell_eval_parquet(path, feature_names, cell_line=None):
    """Load the Replogle cell_eval wide matrix (Colab). Requires pandas+pyarrow. Returns effect_vectors.
    `feature_names`: list mapping column index -> gene symbol (from definition/feature_names.json).
    Averages across gem_group (sequencing batch) per (cell_line, target)."""
    import pandas as pd
    df = pd.read_parquet(path)
    if cell_line:
        df = df[df["cell_line"] == cell_line]
    meta = {"cell_line", "gem_group", "gene"}
    gene_cols = [c for c in df.columns if c not in meta]
    agg = df.groupby("gene")[gene_cols].mean()          # average batches -> one effect vector per KD target
    # column names may be integer feature indices -> map to symbols
    def sym(c):
        try:
            return feature_names[int(c)]
        except (ValueError, IndexError, TypeError):
            return str(c)
    effect = {}
    for target, row in agg.iterrows():
        effect[target] = {sym(c): float(v) for c, v in row.items() if v == v and abs(v) > 0}
    return effect


def integrate_as_witness(measured_rows, top_frac=0.1):
    """Turn the measured ranking into a witness dict (target -> percentile) for the detective's consensus.
    The MEASURED witness is the strongest one: it is interventional, not correlational."""
    n = len(measured_rows)
    return {r["target"]: (n - i) / n for i, r in enumerate(measured_rows)} if n else {}


# ---- self-test on synthetic knockout data: a known driver must top the reversal ranking ----
def _selftest():
    # synthetic biology: gene DRIVER activates a module M={m1..m5}; knocking DRIVER down lowers all of M.
    # PASSENGER is downstream and only affects m5; UNRELATED affects a different module.
    module = ["m1", "m2", "m3", "m4", "m5"]
    disease_up = module                                  # disease = module ON
    disease_down = ["q1", "q2"]                           # and some genes OFF
    sig = signature_vector(disease_up, disease_down)
    effect = {
        # knocking DRIVER down: module goes DOWN (neg), q genes come UP (pos) -> strongly reverses disease
        "DRIVER":    {**{m: -1.0 for m in module}, "q1": 1.0, "q2": 1.0},
        # PASSENGER down: only m5 dips a little -> weak reversal
        "PASSENGER": {"m5": -0.4, "z1": -1.0},
        # UNRELATED down: hits an unrelated module -> ~0 reversal
        "UNRELATED": {"z1": -1.0, "z2": -1.0, "z3": 1.0},
        # PROTECTOR down: module goes UP -> makes disease WORSE (negative reversal)
        "PROTECTOR": {m: 1.0 for m in module},
    }
    ranked = measured_alibi(sig, effect)
    assert ranked[0]["target"] == "DRIVER", ranked
    assert ranked[0]["reversal"] > 0.9, ranked
    assert ranked[0]["drives_disease"] is True
    assert ranked[-1]["target"] == "PROTECTOR" and ranked[-1]["drives_disease"] is False, ranked
    print("measured_cause self-test PASSED — driver recovered, protector correctly flagged")
    for r in ranked:
        print(f"   {r['target']:10} reversal={r['reversal']:+.3f} drives_disease={r['drives_disease']}")


if __name__ == "__main__":
    _selftest()
