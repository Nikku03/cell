"""THE ADMISSION GATE, RUN ON A CELL TYPE NO LAYER WAS BUILT FROM.

WHY A SECOND TASK EXISTS AT ALL. The first gate scored five layers on K562 CRISPRi -> dRNA and admitted none
of them. That result stands, but it cannot be used to score the layer built since: `reg_perturb_k562` is
DERIVED from the Replogle K562 matrix, so scoring it on the Replogle K562 task would be reporting its own
training data. It would win enormously and the number would mean nothing.

Replogle's RPE1 arm solves this exactly. Same assay, same laboratory, same processing, different cell type:
2,122 perturbations over 8,749 genes, pseudobulked from 247,914 cells against 11,485 non-targeting controls.
On RPE1 every layer is out-of-sample -- the curated ones were never built from either cell type, and the
perturbational one was built from the other. For the first time the layers compete on equal footing.

WHAT A COMPARISON ACROSS THE TWO GATES DOES AND DOES NOT MEAN. K562 and RPE1 are different tasks with
different row sets and different response structure, so a layer's number here is NOT comparable to its number
there. What IS comparable is the ordering within this run, and whether a layer clears this run's own ladder
and its own degree-matched control. The K562 figures are reprinted for context and explicitly not differenced.

THE LADDER IS REBUILT FOR THIS CELL, not carried over. Gene responsiveness and perturbation strength are
properties of RPE1 here, estimated inside the training folds. Carrying K562's ladder across would import the
very quantity the layers are supposed to beat.

THE SAME TWO GUARDS AS THE FIRST GATE:
  FOLDS GROUPED BY PERTURBED GENE, so a perturbation is never in training and test at once -- which means the
    row effect cannot be looked up for a held-out perturbation and must be predicted from its attributes. A
    harness that computes two-way fixed effects on the full matrix before splitting leaks the largest term.
  DEGREE-MATCHED CONTROLS, configuration-model rather than uniform rewiring, so "P is a hub" survives the
    permutation and only "P pairs with G" is destroyed.

AND ONE GUARD THIS GATE NEEDS THAT THE FIRST DID NOT. `reg_perturb_k562` edges were selected on K562 effect
size, so the perturbations carrying many edges are the ones that did a lot in K562 -- and a perturbation that
did a lot in K562 tends to do a lot in RPE1. That is a row-level confound, not an edge-level signal, and the
ladder's perturbation-strength rung only partly absorbs it. The layer therefore also gets a SOURCE-MATCHED
control: keep each source's edge count, redirect its targets to other genes matched on RPE1 responsiveness.
If the layer beats degree-matching but not that, what it knows is which perturbations are disruptive, not
which genes they hit.
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
PB = SP / "rpe1_pseudobulk.npz"
NET = SP / "cell_complete.json.gz"
REGL = SP / "cell_reg_perturb.json.gz"
PROTL = SP / "cell_k562_protein.json.gz"
TASK = "RPE1-CRISPRi-single-gene -> dRNA of all measured genes (no layer was built from this cell type)"
SEEDS = (0, 1, 2)
N_PAIRS = 3_000_000
RESPOND_Z = 0.10
N_CTRL = 5


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


def degree_matched(pairs, rng):
    src = np.array([a for a, _b in pairs])
    dst = np.array([b for _a, b in pairs])
    return {(int(a), int(b)) for a, b in zip(src, rng.permutation(dst))}


def main():
    from sklearn.metrics import average_precision_score
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("ADMISSION GATE ON RPE1 -- a cell type no layer was built from")
    report("=" * 100)
    report(f"  TASK: {TASK}")
    for p in (PB, REGL):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    from entity_registry import load as load_reg
    REG = load_reg()

    z = np.load(PB, allow_pickle=True)
    L, perts, rg = z["lfc"], np.array([str(x) for x in z["perts"]]), np.array([str(x) for x in z["genes"]])
    report(f"  RPE1 pseudobulk: {L.shape[0]:,} perturbations x {L.shape[1]:,} genes")

    d = json.load(gzip.open(NET, "rt"))
    name = [g["name"] for g in d["genes"]]
    idx = {n: i for i, n in enumerate(name)}
    chrom = {g["name"]: g.get("chrom") for g in d["genes"]}
    tss = {g["name"]: (int(float(g["tss"])) if g.get("tss") not in (None, "") else None)
           for g in d["genes"]}
    ppm = d.get("ppm", {})
    generic = {n: float(ppm.get(str(i), 0) or 0) for i, n in enumerate(name)}
    dfr = {g["name"]: float(g.get("dep_frac") or 0.0) for g in d["genes"]}
    ntf = {g["name"]: int(g.get("tf") or 0) for g in d["genes"]}
    layers = {
        "regulatory": [(e[0], e[1]) for e in d["reg"]],
        "signalling": [(e[0], e[1]) for e in d["sig"]],
        "PPI": [(e[0], e[1]) for e in d["ppi"]],
        "co-dependency": [(int(a), b) for a, v in (d.get("codep") or {}).items() for b, _s in v],
    }
    cplx = {int(k): set(v) for k, v in (d.get("gene2cplx") or {}).items()}
    del d

    # K562-specific protein, used as an ADDITIONAL abundance rung -- see note at the rung
    k562prot = {}
    if PROTL.exists():
        k562prot = json.load(gzip.open(PROTL, "rt")).get("abundance", {})
    report(f"  K562-measured protein available for {len(k562prot):,} genes")

    # ---- perturbational layer, mapped into cell-object index space ----
    rp = json.load(gzip.open(REGL, "rt"))
    ens2sym = {}
    for e in rg:
        eid = REG.resolve("gene", "ensembl", e)
        ent = REG.entities.get(eid) if eid else None
        if ent and ent.get("symbol"):
            ens2sym[e] = ent["symbol"]
    perturb_edges = []
    for e in rp["edges"]:
        s, t = e["source"], ens2sym.get(e["target_ensembl"])
        if s in idx and t in idx:
            perturb_edges.append((idx[s], idx[t]))
    layers["reg_perturb_k562"] = perturb_edges
    report(f"  reg_perturb_k562: {len(perturb_edges):,} edges mapped into the gene universe "
           f"(built on K562, out-of-sample here)")

    # ---- pair table ----
    pk = np.array([i for i, s in enumerate(perts) if s in idx])
    gk = np.array([j for j, g in enumerate(rg) if ens2sym.get(g) in idx])
    pi = np.array([idx[perts[i]] for i in pk])
    gj = np.array([idx[ens2sym[rg[j]]] for j in gk])
    report(f"  joined to the gene universe: {len(pk):,} perturbations, {len(gk):,} measured genes")

    rng = np.random.default_rng(0)
    npair = min(N_PAIRS, len(pk) * len(gk))
    a = rng.integers(0, len(pk), npair)
    b = rng.integers(0, len(gk), npair)
    y = L[pk[a], gk[b]].astype(np.float32)
    P, G = pi[a], gj[b]
    fin = np.isfinite(y)
    ndrop = int((~fin).sum())
    y, P, G = y[fin], P[fin], G[fin]
    npair = len(y)
    report(f"  pair table: {npair:,} cells sampled from {len(pk)*len(gk):,} ({ndrop:,} non-finite dropped)")
    report(f"  target sd {y.std():.4f}; {100*np.mean(np.abs(y) > RESPOND_Z):.2f}% exceed |lfc| {RESPOND_Z}")

    upert = np.unique(P)
    fold_of = {g: i % 5 for i, g in enumerate(np.random.default_rng(7).permutation(upert))}
    folds = np.array([fold_of[g] for g in P])
    report(f"  folds grouped by perturbed gene: {len(upert):,} perturbations across 5 folds")

    # ---- ladder, rebuilt for RPE1 ----
    lab = np.log10(1.0 + np.array([generic.get(n, 0.0) for n in name], np.float32))
    labk = np.log10(1.0 + np.array([float(k562prot.get(n, 0.0)) for n in name], np.float32))
    hask = np.array([1.0 if n in k562prot else 0.0 for n in name], np.float32)
    ch = {i: chrom.get(name[i]) for i in range(len(name))}
    ts = {i: tss.get(name[i]) for i in range(len(name))}
    same = np.array([1.0 if ch.get(p) is not None and ch.get(p) == ch.get(g) else 0.0
                     for p, g in zip(P, G)], np.float32)
    dist = np.array([np.log10(1.0 + abs(ts[p] - ts[g])) if (same[k] and ts.get(p) and ts.get(g)) else 9.0
                     for k, (p, g) in enumerate(zip(P, G))], np.float32)
    resp = np.zeros((5, len(name)), np.float32)
    for k in range(5):
        tr = folds != k
        s, c = np.zeros(len(name)), np.zeros(len(name))
        np.add.at(s, G[tr], np.abs(y[tr]))
        np.add.at(c, G[tr], 1.0)
        resp[k] = s / np.maximum(c, 1.0)
    RESP = np.array([resp[f, g] for f, g in zip(folds, G)], np.float32)
    ppideg = np.zeros(len(name), np.float32)
    for aa, bb in layers["PPI"]:
        ppideg[int(aa)] += 1
        ppideg[int(bb)] += 1
    PSTR = np.column_stack([[dfr.get(name[p], 0.0) for p in P], lab[P], np.log1p(ppideg[P]),
                            [ntf.get(name[p], 0) for p in P]]).astype(np.float32)
    power = (RESP > 0).astype(np.float32)

    LADDER = [("0 global mean", np.zeros((npair, 1), np.float32)),
              ("1 gene responsiveness", RESP.reshape(-1, 1)),
              ("2 + perturbation strength", np.column_stack([RESP, PSTR])),
              ("3 + generic abundance", np.column_stack([RESP, PSTR, lab[P], lab[G]])),
              # K562-MEASURED PROTEIN IS ITS OWN RUNG, not a replacement for the generic one. It is measured
              # in the WRONG cell type for this task, so it can only help via what is conserved; the `has
              # measurement` indicator is carried alongside because coverage is abundance-dependent and
              # missingness is itself informative.
              ("4 + K562-measured protein",
               np.column_stack([RESP, PSTR, lab[P], lab[G], labk[P], labk[G], hask[P], hask[G]])),
              ("5 + genomic distance",
               np.column_stack([RESP, PSTR, lab[P], lab[G], labk[P], labk[G], hask[P], hask[G],
                                dist, same])),
              ("6 + detection power",
               np.column_stack([RESP, PSTR, lab[P], lab[G], labk[P], labk[G], hask[P], hask[G],
                                dist, same, power]))]

    report(f"\n  NUISANCE LADDER (held-out R2, folds grouped by perturbed gene, {len(SEEDS)} seeds)")
    report(f"    {'arm':30s} {'R2':>9s} {'AUPRC':>9s}  {'delta':>9s}")
    resp_bin = (np.abs(y) > RESPOND_Z).astype(int)
    R = {"task": TASK, "n_pairs": int(npair), "n_perturbations": int(len(upert)),
         "n_measured_genes": int(len(gk)), "ladder": {}}
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
        report(f"    {lname:30s} {v:9.4f} {av:9.4f}  {dl:>9s}")
        R["ladder"][lname] = {"r2": v, "auprc": av, "sd": float(np.std(sc))}
        prev, base_X = v, LX
    R["ladder_top"] = {"name": LADDER[-1][0], "r2": prev}
    report(f"    -> the bar every layer must clear is R2 {prev:.4f}")

    # ---- layers ----
    report(f"\n  LAYER ADMISSION -- {N_CTRL} degree-matched controls each; the perturbational layer also "
           f"gets a source-matched control")
    report(f"    {'layer':20s} {'edges':>10s} {'hits':>9s} {'R2':>9s} {'delta':>9s} {'ctrl':>9s} "
           f"{'verdict':>10s}")
    R["layers"] = {}
    gresp = resp.mean(0)
    gq = np.digitize(gresp, np.quantile(gresp[gresp > 0], np.linspace(.1, .9, 9))) \
        if (gresp > 0).any() else np.zeros(len(gresp), int)
    for lname, pairs in list(layers.items()) + [("complex co-member", None)]:
        if pairs is None:
            feat = np.array([1.0 if (cplx.get(int(p)) and cplx.get(int(g))
                                     and cplx[int(p)] & cplx[int(g)]) else 0.0
                             for p, g in zip(P, G)], np.float32)
            nedge = sum(len(v) for v in cplx.values())
            ctrls = []
            for s in range(N_CTRL):
                rr = np.random.default_rng(500 + s)
                ks = list(cplx); vs = [cplx[k] for k in ks]
                pm = rr.permutation(len(ks))
                cp = {ks[i]: vs[pm[i]] for i in range(len(ks))}
                ctrls.append(np.array([1.0 if (cp.get(int(p)) and cp.get(int(g))
                                               and cp[int(p)] & cp[int(g)]) else 0.0
                                       for p, g in zip(P, G)], np.float32))
        else:
            es = {(int(x), int(y_)) for x, y_ in pairs}
            feat = np.array([1.0 if (int(p), int(g)) in es else 0.0 for p, g in zip(P, G)], np.float32)
            nedge = len(es)
            ctrls = []
            for s in range(N_CTRL):
                cs = degree_matched(pairs, np.random.default_rng(500 + s))
                ctrls.append(np.array([1.0 if (int(p), int(g)) in cs else 0.0
                                       for p, g in zip(P, G)], np.float32))
            if lname == "reg_perturb_k562":
                # SOURCE-MATCHED: keep each source's out-degree, redirect targets to genes matched on RPE1
                # responsiveness. Beats degree-matching but not this => the layer knows which perturbations
                # are disruptive, not which genes they hit.
                for s in range(N_CTRL):
                    rr = np.random.default_rng(900 + s)
                    byq = {}
                    for _a2, b2 in pairs:
                        byq.setdefault(gq[int(b2)], []).append(int(b2))
                    alt = set()
                    for a2, b2 in pairs:
                        pool = byq.get(gq[int(b2)])
                        alt.add((int(a2), int(rr.choice(pool))))
                    ctrls.append(np.array([1.0 if (int(p), int(g)) in alt else 0.0
                                           for p, g in zip(P, G)], np.float32))
        hits = float(feat.mean())
        if hits <= 0:
            report(f"    {lname:20s} {nedge:10,d} {0.0:9.5f}   no sampled pair carries an edge -- untestable")
            R["layers"][lname] = {"n_edges": int(nedge), "hit_rate": 0.0, "verdict": "UNTESTABLE"}
            continue
        sc = [r2(y, fit_oof(np.column_stack([base_X, feat]), y, np.array([(fold_of[g] + s) % 5 for g in P])))
              for s in SEEDS]
        v = float(np.mean(sc))
        cv = [r2(y, fit_oof(np.column_stack([base_X, cf]), y, np.array([fold_of[g] for g in P])))
              for cf in ctrls]
        dl, cd = v - prev, float(np.max(cv)) - prev
        ok = bool(dl > 0.001 and dl > cd)
        report(f"    {lname:20s} {nedge:10,d} {hits:9.5f} {v:9.4f} {dl:+9.4f} {cd:+9.4f} "
               f"{'ADMITTED' if ok else 'REJECTED':>10s}")
        R["layers"][lname] = {"n_edges": int(nedge), "hit_rate": hits, "r2": v, "delta": float(dl),
                              "control_delta_max": float(cd),
                              "control_deltas": [float(c - prev) for c in cv],
                              "n_controls": len(ctrls),
                              "verdict": "ADMITTED" if ok else "REJECTED"}

    adm = [k for k, v in R["layers"].items() if v.get("verdict") == "ADMITTED"]
    rej = [k for k, v in R["layers"].items() if v.get("verdict") == "REJECTED"]
    report("\n" + "=" * 100)
    pv = R["layers"].get("reg_perturb_k562", {})
    ptxt = ""
    if pv:
        ptxt = (f" The perturbational layer, the only one carrying a validated sign, scores "
                f"{pv.get('delta', float('nan')):+.4f} against its strongest control "
                f"{pv.get('control_delta_max', float('nan')):+.4f} over {pv.get('n_controls', 0)} "
                f"permutations including a source-matched one, on {pv.get('hit_rate', 0)*100:.3f}% of "
                f"sampled pairs.")
    v = (f"On `{TASK}`, {len(adm)} of {len(R['layers'])} layers clear this cell type's own nuisance ladder "
         f"(R2 {prev:.4f}) and their own degree-matched controls. ADMITTED: {adm or 'none'}. REJECTED: "
         f"{rej or 'none'}." + ptxt +
         f" Numbers here are NOT differenced against the K562 gate: different rows, different response "
         f"structure, different ladder. What transfers is the ordering and the pass/fail, not the value.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "gate_transfer.json", "w"), indent=1, default=float)
    gate = {"task": TASK, "generated_by": "colab/gate_transfer.py",
            "meaning": "layers under `rejected` remain available for lookup and reasoning but are NOT "
                       "licensed to influence a quantitative prediction on this task",
            "ladder_r2": prev, "admitted": adm, "rejected": rej,
            "per_layer": {k: {kk: vv for kk, vv in val.items() if kk != "control_deltas"}
                          for k, val in R["layers"].items()}}
    json.dump(gate, open(OUT / "layer_admission_rpe1.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'gate_transfer.json'}")
    report(f"  -> {OUT/'layer_admission_rpe1.json'}")


if __name__ == "__main__":
    main()
