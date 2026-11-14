"""CELLBANK9 post hoc (descriptive; DEV genes only; written after the EXAM gates were scored): why did the Drive
datasets not help? Each source is scored ON ITS OWN as a predictor of a gene's K562 response, beside the simplest
baseline there is (the mean TRAIN response, the same for every gene), on the DEV genes that source covers.

  neighbour sources (Tahoe, Geneformer, ARCHS4) and the inputs already in M2 (analog transfer, partners,
  co-essential genes): the neighbour-weighted mean K562 response, alone.
  screens: the 13-screen translator given ONLY that screen's record (LINCS-sh, LINCS-xpr, and each older screen).
Reports median r, the median paired gain over the mean baseline, and the share of genes where it beats the baseline.
No verdicts. Run: python3 -P rem/atlas/cellbank9_posthoc.py
"""

from __future__ import annotations
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("cellbank9", os.path.join(HERE, "cellbank9.py"))
cb9 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb9)
cb8, cb4 = cb9.cb8, cb9.cb4


def main():
    import numpy as np
    G = cb9.nb9(); torch = G["torch"]
    bank6 = cb4.Bank(cb8.V6)
    score, ceil, _ = cb8.scorer(G)
    genes, dv, ex, _ = cb8.sets(G)
    te = [int(i) for i in G["rte"]]; KN, split = G["KN"], G["split"]
    K, hasK = cb8.k562_records(G, bank6)
    mean = G["Ztr"].cpu().mean(0)
    r0 = score(mean.expand(len(te), 128).clone())[0]
    out, rows = [], {}
    P_ = lambda s="": (print(s, flush=True), out.append(s))
    P_("=" * 97); P_("CELLBANK9 POST HOC -- each source alone vs predicting the mean response (DEV genes, descriptive)"); P_("=" * 97)
    P_(f"  baseline 'mean TRAIN response' on all DEV genes: median r {np.median(r0[dv]):+.3f}  (M2: see RESULTS_cellbank9_dev.txt)")
    P_(f"  {'source (alone)':34s} {'DEV genes':>9s} {'median r':>9s} {'mean-baseline r':>16s} {'median gain':>12s} {'beats mean':>11s}")

    def row(lab, pred, has):
        r = score(pred)[0]; m = dv & has
        g = r[m] - r0[m]
        rows[lab] = dict(n=int(m.sum()), r=float(np.median(r[m])), base=float(np.median(r0[m])), gain=float(np.median(g)), beats=float((g > 0).mean()))
        P_(f"  {lab:34s} {m.sum():9d} {np.median(r[m]):+9.3f} {np.median(r0[m]):+16.3f} {np.median(g):+12.3f} {(g > 0).mean():11.0%}")

    Z8, Mk8, _, _ = cb9.tensors(G, cb8.BANK8, cb9.SRC8)
    An = cb8.analog(G, Z8, Mk8, K, hasK)
    row("analog transfer (in M2)", An[te, :128], An[te, 129].numpy() > 0)
    Pt, Ct = cb4.inputs(G, bank6, False)
    row("interaction partners (in M2)", Pt[te, :128], Pt[te, 128].numpy() > 0)
    row("co-essential genes (in M2)", Ct[te, :128], Ct[te, 128].numpy() > 0)
    for n in cb9.NB:
        o, has = cb9.neighbours(G, n, K, hasK)
        row(f"{n} neighbours (new)", o[te, :128], has[te].numpy())
    Z9, Mk9, St9, Lg9 = cb9.tensors(G, cb9.BANK9, cb9.SRC9)
    S = Z9.shape[0] - 1
    train_genes = [j for j, g in enumerate(KN) if split(g) == "TRAIN"]
    tr = cb8.train_translator(G, Z9, Mk9, Lg9, train_genes, seed=99)
    for s, name in enumerate(cb9.SRC9):
        Ms = torch.zeros_like(Mk9); Ms[s] = Mk9[s]
        p, _ = cb8.translate(G, tr, Z9, Ms, Lg9, te, S)
        row(f"{name} record alone" + (" (new)" if name in cb9.NEW else ""), p, Mk9[s, te].numpy())
    open(os.path.join(HERE, "RESULTS_cellbank9_posthoc.txt"), "w").write("\n".join(out) + "\n")
    json.dump(rows, open(os.path.join(cb9.OUTD9, "posthoc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
