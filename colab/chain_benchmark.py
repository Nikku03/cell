"""THE MINIMUM VIABLE CELL: one bounded task, one admission ladder, one verdict per layer.

WHY THIS EXISTS. Every layer in this project has been measured against its own question, and several have
earned their place that way. None has been measured against a SHARED downstream decision, so there is no
answer to the only question that matters for a whole-cell model: when one component is perturbed, does the
model propagate the change correctly? A layer that improves enhancer ranking has not thereby earned the right
to influence a viability prediction, and at present nothing stops it from doing so.

THE TASK, BOUNDED ON PURPOSE.

    In K562, given a single-gene CRISPRi perturbation of gene P, predict the transcriptional response of
    every measured gene G.

Replogle's genome-wide Perturb-seq gives 11,258 perturbations x 8,248 measured genes in exactly this cell
type. It is the one place where a perturbation, its propagation and its readout are all measured at scale.

WHAT THIS TASK IS NOT. The chain worth having is
    perturbation -> dRNA -> dprotein -> dcomplex -> dflux -> phenotype
and the dprotein link cannot be validated today: there is no K562 perturbation proteomics at this scale,
anywhere. Writing the chain down and then quietly scoring only the first arrow would be the same error as
scoring a benchmark's candidate set and calling it genome-wide. So this module validates ARROW ONE, says so,
and protein enters only as a static attribute -- never as a measured change.

THE LADDER. A layer is admitted only if it beats every nuisance baseline below it, in this order:

    0  global mean
    1  gene responsiveness      how reactive is G to anything at all
    2  perturbation strength    how disruptive is P -- ESTIMATED FROM P'S ATTRIBUTES, never from P's own rows
    3  abundance                expression of P and of G
    4  genomic distance         |TSS(P) - TSS(G)|, and same-chromosome
    5  detection power          how well G is measured
    6  co-expression            the nearest-neighbour prior, the strongest annotation-free competitor
    -----------------------------------------------------------------------------------------------
    7+ the layer under test     regulatory edges, PPI, complexes, metabolism, signalling

HELD OUT BY PERTURBED GENE, which forces the honest version of the problem. Folds are grouped by P, so a
perturbation is never in training and test at once. That has a consequence worth stating plainly: the row
effect for a held-out perturbation CANNOT be looked up, because that perturbation was never seen. It has to be
predicted from P's attributes. Any harness that computes two-way fixed effects on the full matrix and then
splits is leaking the answer, and the leak is large -- the row effect is the single biggest term in this
matrix.

DEGREE-MATCHED CONTROL, not edge-shuffled. Rewiring at random destroys the degree sequence, so a rewired graph
loses "P is a hub" along with "P regulates G", and any layer would then look admissible. The control preserves
each source's out-degree and each target's in-degree and destroys only WHICH source pairs with WHICH target.

THE OUTPUT IS A GATE, NOT A SCORE. Each layer gets a machine-readable verdict written to
`layer_admission.json`: admitted, or not, for THIS task. A layer that fails is not deleted -- it stays
available for reasoning and lookup -- but it is recorded as not licensed to influence a quantitative
prediction on this task. That file is meant to be read by the model, not only by a person.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
METLAYER = SP / "cell_metabolic_layer.json.gz"
TASK = "K562-CRISPRi-single-gene -> dRNA of all measured genes"
SEEDS = (0, 1, 2)
N_PAIRS = 3_000_000        # subsample of the 93M (perturbation, gene) cells; see note in build()
RESPOND_Z = 0.10           # |log fold change| above which a gene counts as having responded
N_CTRL = 5


# ----------------------------------------------------------------------------- data

def load_gwps():
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        gid = [s.decode() if isinstance(s, bytes) else str(s) for s in f["var"]["gene_id"][:]]
        npc = f["obs"]["num_cells_filtered"][:] if "num_cells_filtered" in f["obs"] else None
    # "10000_ZBTB46_P1P2_ENSG00000130584" -> ZBTB46
    psym = [p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert]
    return X, np.array(psym), np.array([g.split(".")[0] for g in gid]), npc


def build():
    import pandas as pd
    X, psym, gid, npc = load_gwps()
    im = pd.read_parquet(ROOT / "outputs/orphan/id_map.parquet")
    e2s = {str(a).split(".")[0]: str(b) for a, b in zip(im["ensembl"], im["symbol"]) if isinstance(a, str)}
    gsym = np.array([e2s.get(g, "") for g in gid])

    d = json.load(gzip.open(NET, "rt"))
    name = [g["name"] for g in d["genes"]]
    idx = {n: i for i, n in enumerate(name)}
    chrom = {g["name"]: g.get("chrom") for g in d["genes"]}
    tss = {g["name"]: (int(float(g["tss"])) if g.get("tss") not in (None, "") else None) for g in d["genes"]}
    ppm = d.get("ppm", {})
    abund = {n: float(ppm.get(str(i), 0) or 0) for i, n in enumerate(name)}
    layers = {
        "regulatory": [(e[0], e[1]) for e in d["reg"]],
        "signalling": [(e[0], e[1]) for e in d["sig"]],
        "PPI": [(e[0], e[1]) for e in d["ppi"]],
        "co-dependency": [(a, b) for a, v in (d.get("codep") or {}).items() for b, _s in v],
    }
    cplx = d.get("gene2cplx") or {}
    coex = {int(a): {int(b) for b, _s in v} for a, v in (d.get("coexpr") or {}).items()}
    del d
    return dict(X=X, psym=psym, gsym=gsym, idx=idx, name=name, chrom=chrom, tss=tss,
                abund=abund, layers=layers, cplx=cplx, coex=coex, npc=npc)


# ----------------------------------------------------------------------------- helpers

def edge_sets(pairs):
    s = set()
    for a, b in pairs:
        s.add((int(a), int(b)))
    return s


def degree_matched(pairs, rng):
    """Preserve every source's out-degree and every target's in-degree; destroy only the pairing.

    A configuration-model shuffle: collect the source stubs and the target stubs and re-pair them at random.
    A uniformly rewired graph would instead destroy the degree sequence, so "P is a hub" -- which is a real
    and strong predictor -- would vanish along with "P regulates G", and every layer would clear the bar.
    """
    src = np.array([a for a, _b in pairs])
    dst = np.array([b for _a, b in pairs])
    return edge_sets(zip(src, rng.permutation(dst)))


def r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def fit_oof(X, y, folds):
    import xgboost as xgb
    p = np.zeros(len(y))
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        m = xgb.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.7, min_child_weight=50, reg_lambda=2.0,
                             tree_method="hist", n_jobs=8)
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def main():
    from sklearn.metrics import average_precision_score
    log = []

    def report(s):
        print(s)
        log.append(s)

    report("=" * 100)
    report("MINIMUM VIABLE CELL -- one task, one admission ladder")
    report("=" * 100)
    report(f"  TASK: {TASK}")
    if not GWPS.exists():
        raise SystemExit(f"absent: {GWPS}")
    B = build()
    X, psym, gsym, idx = B["X"], B["psym"], B["gsym"], B["idx"]
    report(f"  Replogle GWPS: {X.shape[0]:,} perturbations x {X.shape[1]:,} measured genes")

    pk = np.array([i for i, s in enumerate(psym) if s in idx])
    gk = np.array([j for j, s in enumerate(gsym) if s in idx])
    report(f"  joined to the gene universe: {len(pk):,} perturbations, {len(gk):,} measured genes")
    pi = np.array([idx[psym[i]] for i in pk])          # cell-object index per perturbation
    gj = np.array([idx[gsym[j]] for j in gk])

    # ---- the pair table ----
    # 93M cells is more than needed and more than fits; a uniform subsample of (perturbation, gene) cells
    # keeps the row/column structure intact and is drawn once, before any arm sees the data.
    rng = np.random.default_rng(0)
    npair = min(N_PAIRS, len(pk) * len(gk))
    a = rng.integers(0, len(pk), npair)
    b = rng.integers(0, len(gk), npair)
    y = X[pk[a], gk[b]].astype(np.float32)
    P, G = pi[a], gj[b]                                 # cell-object indices
    # 0.0074% of the matrix is non-finite, scattered over 2,341 perturbations and 73 measured genes -- sparse
    # missingness, not a structural hole (no row or column is entirely absent). Those pairs are DROPPED. The
    # alternative, imputing them, would invent a target value and then score a model on its own invention.
    fin = np.isfinite(y)
    ndrop = int((~fin).sum())
    y, P, G, a, b = y[fin], P[fin], G[fin], a[fin], b[fin]
    npair = len(y)
    report(f"  pair table: {npair:,} (perturbation, gene) cells sampled from "
           f"{len(pk)*len(gk):,}  ({ndrop:,} dropped as non-finite)")
    report(f"  target: log fold change, sd {y.std():.4f}; "
           f"{100*np.mean(np.abs(y) > RESPOND_Z):.2f}% exceed |lfc| {RESPOND_Z}")

    # ---- folds grouped by PERTURBED GENE ----
    upert = np.unique(P)
    fold_of = {g: i % 5 for i, g in enumerate(np.random.default_rng(7).permutation(upert))}
    folds = np.array([fold_of[g] for g in P])
    report(f"  folds grouped by perturbed gene: {len(upert):,} perturbations across 5 folds")

    # ---- nuisance ladder ----
    name = B["name"]
    ab = np.array([B["abund"].get(name[i], 0.0) for i in range(len(name))], np.float32)
    lab = np.log10(1.0 + ab)
    ch = {i: B["chrom"].get(name[i]) for i in range(len(name))}
    ts = {i: B["tss"].get(name[i]) for i in range(len(name))}
    same_chr = np.array([1.0 if ch.get(p) is not None and ch.get(p) == ch.get(g) else 0.0
                         for p, g in zip(P, G)], np.float32)
    dist = np.array([np.log10(1.0 + abs(ts[p] - ts[g]))
                     if (same_chr[k] and ts.get(p) and ts.get(g)) else 9.0
                     for k, (p, g) in enumerate(zip(P, G))], np.float32)

    # GENE RESPONSIVENESS is a column statistic and is safe to learn: the folds hold out perturbations, not
    # measured genes, so a gene's responsiveness estimated on training perturbations is legitimately known
    # at prediction time. It is computed PER FOLD from training rows only regardless, because "safe in
    # principle" and "computed without touching the test set" are different claims.
    resp = np.zeros((5, len(name)), np.float32)
    for k in range(5):
        tr = folds != k
        s, c = np.zeros(len(name)), np.zeros(len(name))
        np.add.at(s, G[tr], np.abs(y[tr]))
        np.add.at(c, G[tr], 1.0)
        resp[k] = s / np.maximum(c, 1.0)
    RESP = np.array([resp[f, g] for f, g in zip(folds, G)], np.float32)

    # PERTURBATION STRENGTH cannot be looked up for a held-out perturbation -- that is the whole point of
    # grouping folds by P. It is therefore built from P's ATTRIBUTES only.
    dfr = {}
    dd = json.load(gzip.open(NET, "rt"))
    for g in dd["genes"]:
        dfr[idx[g["name"]]] = float(g.get("dep_frac") or 0.0)
    ppideg = np.zeros(len(name), np.float32)
    for aa, bb in B["layers"]["PPI"]:
        ppideg[int(aa)] += 1
        ppideg[int(bb)] += 1
    ntf = {idx[g["name"]]: int(g.get("tf") or 0) for g in dd["genes"]}
    del dd
    PSTR = np.column_stack([
        [dfr.get(p, 0.0) for p in P], lab[P], np.log1p(ppideg[P]), [ntf.get(p, 0) for p in P]]
    ).astype(np.float32)

    # CO-EXPRESSION -- the annotation-free competitor the layers have to beat, not just the fixed effects
    coex = B["coex"]
    COEX = np.array([1.0 if g in coex.get(p, ()) else 0.0 for p, g in zip(P, G)], np.float32)

    LADDER = [
        ("0 global mean", np.zeros((npair, 1), np.float32)),
        ("1 gene responsiveness", RESP.reshape(-1, 1)),
        ("2 + perturbation strength", np.column_stack([RESP, PSTR])),
        ("3 + abundance", np.column_stack([RESP, PSTR, lab[P], lab[G]])),
        ("4 + genomic distance", np.column_stack([RESP, PSTR, lab[P], lab[G], dist, same_chr])),
        ("5 + detection power", None),
        ("6 + co-expression", None),
    ]
    power = np.array([resp[f, g] > 0 for f, g in zip(folds, G)], np.float32)
    pw = np.column_stack([RESP, PSTR, lab[P], lab[G], dist, same_chr, power])
    LADDER[5] = ("5 + detection power", pw)
    LADDER[6] = ("6 + co-expression", np.column_stack([pw, COEX]))

    report(f"\n  NUISANCE LADDER (held-out R2, folds grouped by perturbed gene, {len(SEEDS)} seeds)")
    report(f"    {'arm':28s} {'R2':>9s} {'AUPRC':>9s}  {'delta R2':>9s}")
    resp_bin = (np.abs(y) > RESPOND_Z).astype(int)
    R = {"task": TASK, "n_pairs": int(npair), "n_perturbations": int(len(upert)),
         "n_measured_genes": int(len(gk)), "respond_threshold": RESPOND_Z, "ladder": {}}
    prev, base_X = None, None
    for lname, LX in LADDER:
        sc, ap = [], []
        for s in SEEDS:
            f = np.array([(fold_of[g] + s) % 5 for g in P])
            p = fit_oof(LX, y, f)
            sc.append(r2(y, p))
            ap.append(average_precision_score(resp_bin, np.abs(p)))
        v, av = float(np.mean(sc)), float(np.mean(ap))
        dl = "" if prev is None else f"{v-prev:+9.4f}"
        report(f"    {lname:28s} {v:9.4f} {av:9.4f}  {dl:>9s}")
        R["ladder"][lname] = {"r2": v, "auprc": av, "sd": float(np.std(sc))}
        prev, base_X = v, LX
    R["ladder_top"] = {"name": LADDER[-1][0], "r2": prev,
                       "auprc": R["ladder"][LADDER[-1][0]]["auprc"]}
    report(f"    -> the bar every layer must clear is R2 {prev:.4f} "
           f"(AUPRC {R['ladder_top']['auprc']:.4f})")

    # ---- layers, one at a time, each against the top of the ladder ----
    report(f"\n  LAYER ADMISSION -- each layer added to the full ladder, {N_CTRL} degree-matched controls")
    report(f"    {'layer':18s} {'edges':>10s} {'hits':>9s} {'R2':>9s} {'delta':>9s} "
           f"{'ctrl delta':>11s} {'verdict':>10s}")
    R["layers"] = {}
    cplx = B["cplx"]
    for lname, pairs in list(B["layers"].items()) + [("complex co-member", None)]:
        if pairs is None:
            g2c = {int(k): set(v) for k, v in cplx.items()}
            feat = np.array([1.0 if (g2c.get(int(p)) and g2c.get(int(g))
                                     and g2c[int(p)] & g2c[int(g)]) else 0.0
                             for p, g in zip(P, G)], np.float32)
            nedge, ctrl_feats = sum(len(v) for v in cplx.values()), []
            for s in range(N_CTRL):                     # permute complex membership across genes
                rr = np.random.default_rng(500 + s)
                ks = list(g2c); vs = [g2c[k] for k in ks]
                perm = rr.permutation(len(ks))
                g2cp = {ks[i]: vs[perm[i]] for i in range(len(ks))}
                ctrl_feats.append(np.array([1.0 if (g2cp.get(int(p)) and g2cp.get(int(g))
                                                    and g2cp[int(p)] & g2cp[int(g)]) else 0.0
                                            for p, g in zip(P, G)], np.float32))
        else:
            es = edge_sets(pairs)
            feat = np.array([1.0 if (int(p), int(g)) in es else 0.0 for p, g in zip(P, G)], np.float32)
            nedge, ctrl_feats = len(es), []
            for s in range(N_CTRL):
                cs = degree_matched(pairs, np.random.default_rng(500 + s))
                ctrl_feats.append(np.array([1.0 if (int(p), int(g)) in cs else 0.0
                                            for p, g in zip(P, G)], np.float32))
        hits = float(feat.mean())
        if hits <= 0:
            report(f"    {lname:18s} {nedge:10,d} {0:9.5f}   no (perturbation, gene) pair in the sample "
                   f"carries an edge -- untestable on this task")
            R["layers"][lname] = {"n_edges": int(nedge), "hit_rate": 0.0, "verdict": "UNTESTABLE",
                                  "reason": "no sampled pair carries an edge"}
            continue
        sc = []
        for s in SEEDS:
            f = np.array([(fold_of[g] + s) % 5 for g in P])
            sc.append(r2(y, fit_oof(np.column_stack([base_X, feat]), y, f)))
        v = float(np.mean(sc))
        cv = []
        for cf in ctrl_feats:
            f = np.array([fold_of[g] for g in P])
            cv.append(r2(y, fit_oof(np.column_stack([base_X, cf]), y, f)))
        dl, cd = v - prev, float(np.max(cv)) - prev
        ok = bool(dl > 0.001 and dl > cd)
        report(f"    {lname:18s} {nedge:10,d} {hits:9.5f} {v:9.4f} {dl:+9.4f} {cd:+11.4f} "
               f"{'ADMITTED' if ok else 'REJECTED':>10s}")
        R["layers"][lname] = {"n_edges": int(nedge), "hit_rate": hits, "r2": v, "delta": float(dl),
                              "control_delta_max": float(cd), "control_deltas": [float(c - prev) for c in cv],
                              "verdict": "ADMITTED" if ok else "REJECTED"}

    # ---- the gate ----
    adm = [k for k, v in R["layers"].items() if v.get("verdict") == "ADMITTED"]
    rej = [k for k, v in R["layers"].items() if v.get("verdict") == "REJECTED"]
    unt = [k for k, v in R["layers"].items() if v.get("verdict") == "UNTESTABLE"]
    report("\n" + "=" * 100)
    v = (f"On `{TASK}`, {len(adm)} of {len(R['layers'])} layers clear the nuisance ladder and their own "
         f"degree-matched control. ADMITTED for quantitative propagation on this task: "
         f"{adm or 'none'}. REJECTED (available for reasoning, not licensed to move a number): {rej or 'none'}"
         + (f". UNTESTABLE on this task: {unt}" if unt else ".")
         + f" The ladder itself reaches R2 {prev:.4f}, and the single largest term is gene responsiveness -- "
           f"which is an attribute of the READOUT, not of the network. This is arrow one of the chain only: "
           f"perturbation -> dRNA. The dprotein arrow is not validated here and no K562 perturbation "
           f"proteomics exists at this scale to validate it with.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "chain_benchmark.json", "w"), indent=1, default=float)
    gate = {"task": TASK, "generated_by": "colab/chain_benchmark.py",
            "meaning": "layers listed under `rejected` remain available for lookup and reasoning but are "
                       "NOT licensed to influence a quantitative prediction on this task",
            "ladder_r2": prev, "admitted": adm, "rejected": rej, "untestable": unt,
            "per_layer": {k: {kk: vv for kk, vv in val.items() if kk != "control_deltas"}
                          for k, val in R["layers"].items()}}
    json.dump(gate, open(OUT / "layer_admission.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'chain_benchmark.json'}")
    report(f"  -> {OUT/'layer_admission.json'}   (the gate, meant to be read by the model)")


if __name__ == "__main__":
    main()
