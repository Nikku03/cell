"""Build the three MISSED source-side blocks and cache them. Nothing is scored here.

The dataset audit found three kinds of data on disk that the model never used, and one of them was unused for a
reason worth stating plainly.

1. CROSS-LINE RESPONSE (`xline`) -- 6.7 GB of Perturb-seq in RPE1 / Jurkat / HepG2 / HCT116 / Replogle.
   adrn_ko_multiline.py already used these, but only as EXTRA TRAINING ROWS (pool / strict-pool / bank), which
   asks "do more rows of the same regression help?" -- answer was no, -0.037. It never used a gene's response in
   another line as a DESCRIPTOR OF THAT GENE. That is exactly the codep construction, and codep is the one block
   that came back ROBUST across 18 cells.

2. DENSE MEASURED FIELDS INSIDE cell_complete.json -- only 7 of ~40 fields ever become channels. `coexpr`
   (16,374 genes, ranked partners with correlations) and `ppm`/`abund` (16,015 / 16,286, protein abundance) are
   dense, measured, and defined for genes nobody has annotated.

3. INTERFACE-WEIGHTED PPI (`iface`) -- 7,594,358 AlphaFold heterodimer models, of which 587,219 are human-human,
   scored by ipSAE / pDockQ2 / clash counts. The deployed ppi block uses BINARY edges and came back DIRECTIONAL.
   This is the same relation weighted by predicted interface quality, and 3x the edge count. Only 5,730 pairs
   pass the published quality threshold, so the screen is mostly negatives -- both the confidence-weighted full
   graph and the high-confidence subgraph are built, because they are different objects.

WHAT THE READER MUST KNOW BEFORE INTERPRETING `xline`, and it applies to `codep` too:

    329 of the 400 sealed knockouts are measured in RPE1/Jurkat/HepG2.

So this block hands the model a real measurement of the held-out gene's knockout effect -- in another cell line,
not in K562, so the sealed answer is untouched and this is not leakage. But it changes the regime: from "predict
a gene never perturbed anywhere" to "transfer a gene's measured response across lines". Both are legitimate and
the second is the more realistic deployment, but they are DIFFERENT CLAIMS and must not be reported as one.
codep has the same property -- DepMap measures the held-out gene's essentiality in ~1,100 other lines -- so
today's ROBUST codep result may also be transfer rather than cold start. The scoring script stratifies sealed
genes by whether they carry cross-context measurement, which is the instrument that separates the two.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT, SP = A.OUT, A.SP
LINES = ("nlz_RPE1", "nlz_Jurkat", "nlz_HepG2", "nlz_HCT116", "nlz_Replogle")
HETERO = SP / "heterodimer_human.tsv"


def main():
    print("=" * 100, flush=True)
    print("BUILDING THE THREE MISSED BLOCKS", flush=True)
    print("=" * 100, flush=True)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    sealed = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    universe = sorted(set(kos) | sealed)
    uix = {g: i for i, g in enumerate(universe)}
    nU = len(universe)
    print(f"  universe {nU:,} genes; {len(sealed)} sealed", flush=True)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] if isinstance(g, dict) else str(g) for g in D["genes"]]
    nidx = {g: i for i, g in enumerate(names)}
    nG = len(names)

    # ---------------------------------------------------------------- 1. cross-line response
    blocks, covers = {}, {}
    mats, tgt_space = [], {}
    per_line = []
    for ln in LINES:
        d = pickle.load(open(SP / f"{ln}.pkl", "rb"))
        for prof in d.values():
            for t in prof:
                if t not in tgt_space:
                    tgt_space[t] = len(tgt_space)
        per_line.append(d)
    nT = len(tgt_space)
    print(f"  xline: {len(LINES)} lines, common target space {nT:,} genes", flush=True)
    cov = np.zeros(nU, bool)
    for ln, d in zip(LINES, per_line):
        M = np.zeros((nU, nT), np.float32)
        hit = 0
        for g, prof in d.items():
            i = uix.get(g)
            if i is None:
                continue
            hit += 1
            cov[i] = True
            for t, v in prof.items():
                M[i, tgt_space[t]] = float(v)
        mats.append(M)
        print(f"    {ln:14} covers {hit:>5,} of {nU:,}", flush=True)
    X = np.concatenate(mats, axis=1)
    print(f"  xline raw {X.shape}; {int(cov.sum()):,} of {nU:,} genes have ANY cross-line measurement "
          f"({int((cov & np.array([g in sealed for g in universe])).sum())} of {len(sealed)} sealed)", flush=True)
    blocks["xline"] = X
    covers["xline"] = cov

    # ---------------------------------------------------------------- 2. coexpr and proteomics
    ce = D.get("coexpr", {})
    C = np.zeros((nU, nG), np.float32)
    ccov = np.zeros(nU, bool)
    for k, partners in ce.items():
        try:
            gi = int(k)
        except ValueError:
            continue
        if gi >= nG:
            continue
        i = uix.get(names[gi])
        if i is None:
            continue
        ccov[i] = True
        for p in partners:
            if isinstance(p, (list, tuple)) and len(p) >= 2 and int(p[0]) < nG:
                C[i, int(p[0])] = float(p[1])
    print(f"  coexpr: {int(ccov.sum()):,} of {nU:,} genes have partners", flush=True)
    blocks["coexpr"] = C
    covers["coexpr"] = ccov

    ppm, abund = D.get("ppm", {}), D.get("abund", {})
    Pm = np.zeros((nU, 3), np.float32)
    pcov = np.zeros(nU, bool)
    for src, col in ((ppm, 0), (abund, 1)):
        for k, v in src.items():
            try:
                gi = int(k)
            except ValueError:
                continue
            if gi >= nG:
                continue
            i = uix.get(names[gi])
            if i is None:
                continue
            try:
                Pm[i, col] = float(v)
            except (TypeError, ValueError):
                continue
            pcov[i] = True
    Pm[:, 2] = np.log1p(np.maximum(Pm[:, 0], 0))
    print(f"  prot: {int(pcov.sum()):,} of {nU:,} genes have abundance", flush=True)
    blocks["prot"] = Pm
    covers["prot"] = pcov

    # ---------------------------------------------------------------- 3. interface-weighted PPI
    if HETERO.exists():
        wall, whq, seen = [], [], set()
        with open(HETERO) as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) < 5:
                    continue
                a, b = nidx.get(row[0]), nidx.get(row[1])
                if a is None or b is None or a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    ips = float(row[2])
                except ValueError:
                    ips = 0.0
                wall.append((key[0], key[1], ips))
                if row[4].strip().lower() == "true":
                    whq.append((key[0], key[1], 1.0))
        print(f"  iface: {len(wall):,} distinct human pairs mapped to the cell's symbols; "
              f"{len(whq):,} pass the quality threshold", flush=True)
        np.savez_compressed(SP / "iface_edges.npz",
                            all=np.array(wall, np.float32), hq=np.array(whq, np.float32))
    else:
        print("  iface: heterodimer_human.tsv MISSING -- block skipped", flush=True)

    np.savez_compressed(SP / "blocks_new.npz",
                        universe=np.array(universe),
                        **{f"B_{k}": v for k, v in blocks.items()},
                        **{f"C_{k}": v for k, v in covers.items()})
    print(f"\n  -> {SP / 'blocks_new.npz'}  and  {SP / 'iface_edges.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
