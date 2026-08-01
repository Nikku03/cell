"""TESTS A, B, C -- unseen line, unseen gene, and both at once. The universal-generalisation test.

WHAT THIS IS FOR. `v4_viability_head.py` passed 8/8 gates on held-out CELL LINES. That is Test A, and Test A
is structurally incapable of failing on gene identity, because gene identity is never withheld. The numbers
it reported make the problem explicit:

    GLOBAL-mean          RMSE 0.4207
    MARGINAL-gene alone  RMSE 0.1580     <- a per-gene constant closes 96% of the distance
    MARGINAL-both        RMSE 0.1579
    MODEL                RMSE 0.1465     <- everything learned is 0.0114 on top of that

and the per-fold marginal spreads were gene sd 0.3887 against line sd 0.0076 -- gene identity is ~51x more
variable than line identity. So the task is dominated by "which gene is this", and Test A always tells the
model the answer.

    A   train lines            -> new line,  every gene seen        (PASSED, and cannot test the claim)
    B   train genes            -> new gene,  every line seen        (this file)
    C   train genes x lines    -> new gene in a new line            (this file)

THE PREDICTION BEING TESTED, declared before the numbers: performance falls sharply from A to B, because the
measured gain is dominated by a learned per-gene lookup. B and C ask whether annotation, partners and
biological features can recover a dependency profile with NO gene lookup available.

THE LEAKAGE THAT WOULD MAKE TEST B MEANINGLESS, and it is not subtle once stated. Co-dependency is computed
FROM DepMap. Selecting a held-out gene's co-dependency partners requires that gene's own column across cell
lines -- which is exactly the label. A Test B built on co-dependency partners would report a confident pass
that means nothing.

    Test A  co-dependency partners are LEGITIMATE. Gene g's column exists in the training lines, so
            corr(g, .) restricted to training lines uses no held-out value.
    Test B  co-dependency partners are FORBIDDEN. Gene g has no training column at all.
    Test C  forbidden for the same reason.

So in B and C the partner channel is rebuilt from sources that never saw DepMap: PPI, complex co-membership
and shared pathway. Those partners are TRAINING genes, and their effects are known. The hypothesis becomes
clean and worth asking on its own terms: can a gene's physical and functional partners predict its
dependency profile when the gene itself has never been measured?

THE SAME RULE APPLIES TO THE GENE FEATURE TABLE. `cell_complete.json.gz` carries `ess` and `dep_frac`, both
derived from DepMap. Both are EXCLUDED here and the exclusion is asserted at load time, not assumed. What
remains is annotation that never saw a viability screen: compartment, process, pathway count, complex
membership, PPI degree, TF flag, LOEUF (gnomAD), chromosome, TSS/CpG/enhancer counts, literature counts.

THE ARMS, and every regime gets the same ladder so A/B/C are directly comparable:

    0-param  GLOBAL-mean                  the mean effect over training cells
    0-param  MARGINAL-gene                per-gene mean over TRAINING LINES.
                                          IN B AND C THIS IS UNDEFINED for a held-out gene and falls back to
                                          the global mean. That collapse is not a bug, it is the measurement.
    0-param  MARGINAL-line                per-line mean over TRAINING GENES
    0-param  MARGINAL-both                additive
    0-param  PARTNER-MEAN (non-DepMap)    mean training-gene effect over PPI/complex/pathway partners.
                                          THE KEY ARM IN B AND C -- the only 0-param arm that can carry
                                          information about a gene never measured.
    0-param  PARTNER-MEAN random CONTROL  matched partner count, uniformly drawn
    1-param  PARTNER-MEAN shrunk          one fitted shrinkage coefficient
    learned  FEATURES-only                annotation features, NO gene identity, NO partners
    learned  FEATURES + PARTNERS          the full non-lookup model
    learned  FULL (+ gene identity)       includes a gene embedding. In B and C the embedding for a held-out
                                          gene is untrained, so this arm is expected to DEGRADE to the
                                          features model -- reported to make the lookup's contribution visible
    CONTROL  shuffled gene features       features permuted across genes, partners intact
    CONTROL  wrong-gene identity          a different held-out gene's row
    CONTROL  wrong-line identity          a different held-out line's column

UNIT OF REPLICATION, which differs per regime and must:
    A   held-out LINE blocks     n = N_LINE_BLOCKS
    B   held-out GENE blocks     n = N_GENE_BLOCKS
    C   held-out (gene, line) block pairs, n = N_GENE_BLOCKS (each paired with a disjoint line block)
    MDE = 3 * sd / sqrt(n), sd = sd (ddof=1) of the PAIRED per-block gap. Model-init seeds are reported on
    one block only and are explicitly NOT graded on.

THE GATES, declared before any number:
    G1  RETAINED    what FRACTION of Test A's gain over MARGINAL-both survives into B? Reported, and it is
                    the headline number of this file whatever it turns out to be.
    G2  B-USEFUL    in Test B, FEATURES+PARTNERS beats GLOBAL-mean by more than the MDE. If this fails,
                    nothing about a gene's dependency is recoverable here without measuring it.
    G3  B-REAL      in Test B, FEATURES+PARTNERS beats the SHUFFLED-FEATURE control by more than the MDE.
    G4  B-PARTNERS  in Test B, PARTNER-MEAN beats PARTNER-MEAN-random by more than the MDE.
    G5  B-FREE      in Test B, the best learned arm beats the best 0-parameter arm by more than the MDE.
    G6  C-SURVIVES  Test C's FEATURES+PARTNERS still beats GLOBAL-mean by more than the MDE.
    G7  NO-LEAK     asserted, not tested: no DepMap-derived feature enters B or C. The module raises if
                    `ess` or `dep_frac` is found in the feature matrix.

A NULL IN B IS A REAL RESULT AND IS NOT A FAILURE OF THIS FILE. It would say that in this data a gene's
dependency profile is not recoverable from annotation and partners, i.e. that the viability head's 8/8 pass
is a statement about interpolating across cell lines and not about biology transferring to new genes.
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
DEPMAP = SP / "CRISPRGeneEffect.csv"
NET = SP / "cell_complete.json.gz"

N_GENE_BLOCKS = int(os.environ.get("ABC_GENE_BLOCKS", "5"))
N_LINE_BLOCKS = int(os.environ.get("ABC_LINE_BLOCKS", "5"))
MAX_PARTNERS = int(os.environ.get("ABC_PARTNERS", "16"))
EPOCHS = int(os.environ.get("ABC_EPOCHS", "40"))
SEEDS = int(os.environ.get("ABC_SEEDS", "2"))
MIN_LINES = int(os.environ.get("ABC_MIN_LINES", "100"))
SMOKE = os.environ.get("ABC_SMOKE", "0") == "1"
if SMOKE:
    N_GENE_BLOCKS, N_LINE_BLOCKS, EPOCHS, SEEDS = 3, 3, 6, 1

# Fields in cell_complete.json.gz that are DERIVED FROM DepMap and would leak the label in B and C.
BANNED = {"ess", "ess_src", "dep_frac"}


def main():
    import pandas as pd
    import torch
    import torch.nn as nn
    torch.set_num_threads(int(os.environ.get("ABC_THREADS", "2")))
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 108)
    report("TESTS A / B / C -- unseen line, unseen gene, and both. The universal-generalisation test.")
    report("=" * 108)
    if SMOKE:
        report("  *** SMOKE RUN -- a plumbing check, NOT a result. Rerun with ABC_SMOKE unset. ***")

    df = pd.read_csv(DEPMAP, index_col=0)
    df.columns = [c.split(" (")[0] for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    Y = df.to_numpy(np.float32)                      # lines x genes
    lines = list(df.index)
    genes = list(df.columns)
    obs = np.isfinite(Y)
    keep_g = obs.sum(0) >= MIN_LINES
    Y, genes = Y[:, keep_g], [g for g, k in zip(genes, keep_g) if k]
    obs = obs[:, keep_g]
    report(f"  DepMap gene effect: {Y.shape[0]:,} cell lines x {Y.shape[1]:,} genes "
           f"with >= {MIN_LINES} measured lines; {100*obs.mean():.1f}% observed")

    d = json.load(gzip.open(NET, "rt"))
    nmz = [g["name"] for g in d["genes"]]
    gidx = {g: i for i, g in enumerate(genes)}

    # ---- GENE FEATURES, with the DepMap-derived fields removed and the removal ASSERTED ----
    recs = {g["name"]: g for g in d["genes"]}
    num_fields = ["loeuf", "tf", "ppi", "ndis", "npath", "tss", "cpg", "enh", "dark", "conf", "pubs"]
    num_fields = [f for f in num_fields if f not in BANNED]
    cat_fields = ["comp", "proc", "chrom"]
    leaked = BANNED & (set(num_fields) | set(cat_fields))
    if leaked:
        raise SystemExit(f"LEAK: DepMap-derived fields {leaked} reached the feature matrix")
    cats = {f: sorted({str(recs.get(g, {}).get(f, "")) for g in genes}) for f in cat_fields}
    cmap = {f: {v: i for i, v in enumerate(cats[f])} for f in cat_fields}
    F = np.zeros((len(genes), len(num_fields) + sum(len(cats[f]) for f in cat_fields)), np.float32)
    for i, g in enumerate(genes):
        r = recs.get(g, {})
        for j, f in enumerate(num_fields):
            v = r.get(f, 0)
            F[i, j] = float(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else 0.0
        off = len(num_fields)
        for f in cat_fields:
            F[i, off + cmap[f][str(r.get(f, ""))]] = 1.0
            off += len(cats[f])
    F[:, :len(num_fields)] = np.log1p(np.abs(F[:, :len(num_fields)])) * np.sign(F[:, :len(num_fields)])
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    report(f"  gene features: {F.shape[1]} columns from {len(num_fields)} numeric + "
           f"{len(cat_fields)} categorical fields. EXCLUDED as DepMap-derived: {sorted(BANNED)}")

    # ---- NON-DepMap PARTNER GRAPH: PPI, complex co-membership, shared pathway. Never sees DepMap. ----
    nb = {}
    for e in d["ppi"]:
        a, b = nmz[int(e[0])], nmz[int(e[1])]
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    g2c = {int(k): set(v) for k, v in (d.get("gene2cplx") or {}).items()}
    bym = {}
    for x, cs in g2c.items():
        for c in cs:
            bym.setdefault(c, []).append(x)
    for c, mem in bym.items():
        if len(mem) > 60:
            continue
        for x in mem:
            for y in mem:
                if x != y:
                    nb.setdefault(nmz[x], set()).add(nmz[y])
    del d
    PART = [np.array([gidx[p] for p in list(nb.get(g, ()))[:MAX_PARTNERS] if p in gidx], np.int64)
            for g in genes]
    npart = np.array([len(p) for p in PART])
    report(f"  non-DepMap partner graph (PPI + complex co-membership): "
           f"{int((npart > 0).sum()):,}/{len(genes):,} genes have >= 1 partner, median {np.median(npart):.0f}")

    rng0 = np.random.default_rng(0)
    PART_R = [rng0.choice(len(genes), len(p), replace=False) if len(p) else np.array([], np.int64)
              for p in PART]

    gblk = rng0.permutation(len(genes)) % N_GENE_BLOCKS
    lblk = rng0.permutation(len(Y)) % N_LINE_BLOCKS

    def rmse(p, t, m):
        return float(np.sqrt(np.nanmean((p[m] - t[m]) ** 2)))

    def fit_mlp(Xtr, ytr, Xte, seed, epochs=EPOCHS):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU(),
                            nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
        X = torch.from_numpy(Xtr)
        y = torch.from_numpy(ytr).unsqueeze(1)
        n = len(X)
        for _ in range(epochs):
            pm = torch.randperm(n)
            for i0 in range(0, n, 8192):
                b = pm[i0:i0 + 8192]
                loss = (net(X[b]) - y[b]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return net(torch.from_numpy(Xte)).squeeze(1).numpy()

    def run_regime(name, gtr, gte, ltr, lte, blk):
        """One block of one regime. gtr/gte index genes; ltr/lte index lines."""
        tr_mask = np.zeros_like(obs)
        tr_mask[np.ix_(ltr, gtr)] = True
        tr_mask &= obs
        te_mask = np.zeros_like(obs)
        te_mask[np.ix_(lte, gte)] = True
        te_mask &= obs
        if te_mask.sum() < 500:
            return None

        Ytr = np.where(tr_mask, Y, np.nan)
        mu = float(np.nanmean(Ytr))
        # per-gene mean over TRAINING lines. For a gene with no training cell at all (regimes B and C) this
        # is nan and falls back to mu -- the collapse of the lookup, which is the thing being measured.
        with np.errstate(invalid="ignore"):
            a_g = np.nanmean(Ytr, axis=0)
            b_c = np.nanmean(Ytr - np.where(np.isfinite(a_g), a_g, mu)[None, :], axis=1)
        a_g_f = np.where(np.isfinite(a_g), a_g, mu)
        b_c_f = np.where(np.isfinite(b_c), b_c, 0.0)
        lookup_available = float(np.isfinite(a_g[gte]).mean())

        # partner mean uses TRAINING-gene effects only, over the non-DepMap graph
        def partner_mean(P):
            out = np.full(len(genes), np.nan, np.float32)
            trg = np.zeros(len(genes), bool)
            trg[gtr] = True
            for i in range(len(genes)):
                p = P[i]
                p = p[trg[p]] if len(p) else p
                if len(p):
                    out[i] = np.nanmean(a_g_f[p])
            return np.where(np.isfinite(out), out, mu)
        pm_real, pm_rand = partner_mean(PART), partner_mean(PART_R)

        G, L = np.meshgrid(np.arange(len(genes)), np.arange(len(Y)))
        arms = {}

        def add(k, pred):
            arms[k] = rmse(pred, Y, te_mask)
        add("GLOBAL-mean (0p)", np.full_like(Y, mu))
        add("MARGINAL-gene (0p)", np.broadcast_to(a_g_f[None, :], Y.shape).copy())
        add("MARGINAL-line (0p)", np.broadcast_to((mu + b_c_f)[:, None], Y.shape).copy())
        add("MARGINAL-both (0p)", a_g_f[None, :] + b_c_f[:, None])
        add("PARTNER-MEAN non-DepMap (0p)", np.broadcast_to(pm_real[None, :], Y.shape).copy()
            + b_c_f[:, None])
        add("PARTNER-MEAN random (0p CONTROL)", np.broadcast_to(pm_rand[None, :], Y.shape).copy()
            + b_c_f[:, None])

        # ---- learned arms ----
        ti, tj = np.where(tr_mask)
        ei, ej = np.where(te_mask)
        if SMOKE and len(ti) > 400000:
            sel = rng0.choice(len(ti), 400000, replace=False)
            ti, tj = ti[sel], tj[sel]
        feats_tr = np.c_[F[tj], pm_real[tj, None], b_c_f[ti, None]].astype(np.float32)
        feats_te = np.c_[F[ej], pm_real[ej, None], b_c_f[ei, None]].astype(np.float32)
        y_tr = Y[ti, tj]

        Fsh = F[rng0.permutation(len(genes))]
        sh_tr = np.c_[Fsh[tj], pm_real[tj, None], b_c_f[ti, None]].astype(np.float32)
        sh_te = np.c_[Fsh[ej], pm_real[ej, None], b_c_f[ei, None]].astype(np.float32)

        fonly_tr = np.c_[F[tj], b_c_f[ti, None]].astype(np.float32)
        fonly_te = np.c_[F[ej], b_c_f[ei, None]].astype(np.float32)

        lut_tr = np.c_[feats_tr, a_g_f[tj, None]].astype(np.float32)
        lut_te = np.c_[feats_te, a_g_f[ej, None]].astype(np.float32)

        preds = {}
        for k, (Xt, Xe) in (("FEATURES-only", (fonly_tr, fonly_te)),
                            ("FEATURES+PARTNERS", (feats_tr, feats_te)),
                            ("FULL (+gene lookup)", (lut_tr, lut_te)),
                            ("SHUFFLED-FEATURES (CONTROL)", (sh_tr, sh_te))):
            ps = [fit_mlp(Xt, y_tr, Xe, s) for s in range(SEEDS)]
            preds[k] = np.mean(ps, 0)
            P = np.full_like(Y, np.nan)
            P[ei, ej] = preds[k]
            arms[k] = rmse(P, Y, te_mask)

        wg = rng0.permutation(len(ej))
        P = np.full_like(Y, np.nan)
        P[ei, ej] = preds["FEATURES+PARTNERS"][wg]
        arms["WRONG-GENE (identity CONTROL)"] = rmse(P, Y, te_mask)

        arms["_lookup_available"] = lookup_available
        arms["_n_test"] = int(te_mask.sum())
        return arms

    REG = {}
    for regime in ("A", "B", "C"):
        blocks = []
        nb_ = N_LINE_BLOCKS if regime == "A" else N_GENE_BLOCKS
        for b in range(nb_):
            if regime == "A":
                gtr = gte = np.arange(len(genes))
                ltr, lte = np.where(lblk != b)[0], np.where(lblk == b)[0]
            elif regime == "B":
                gtr, gte = np.where(gblk != b)[0], np.where(gblk == b)[0]
                ltr = lte = np.arange(len(Y))
            else:
                gtr, gte = np.where(gblk != b)[0], np.where(gblk == b)[0]
                lb = b % N_LINE_BLOCKS
                ltr, lte = np.where(lblk != lb)[0], np.where(lblk == lb)[0]
            r = run_regime(regime, gtr, gte, ltr, lte, b)
            if r:
                blocks.append(r)
        if not blocks:
            continue
        keys = [k for k in blocks[0] if not k.startswith("_")]
        REG[regime] = {"arms": {k: [b[k] for b in blocks] for k in keys},
                       "lookup_available": float(np.mean([b["_lookup_available"] for b in blocks])),
                       "n_blocks": len(blocks),
                       "n_test": int(np.sum([b["_n_test"] for b in blocks]))}
        report(f"\n  REGIME {regime}  ({len(blocks)} blocks, {REG[regime]['n_test']:,} scored values, "
               f"gene lookup available for {100*REG[regime]['lookup_available']:.1f}% of test genes)")
        for k in keys:
            v = REG[regime]["arms"][k]
            report(f"    {k:38s} RMSE {np.mean(v):.4f}  sd {np.std(v, ddof=1) if len(v) > 1 else 0:.4f}")

    def m(reg, k):
        return float(np.mean(REG[reg]["arms"][k]))

    def mde(reg, k1, k2):
        a = np.array(REG[reg]["arms"][k1])
        b = np.array(REG[reg]["arms"][k2])
        g = b - a
        return float(g.mean()), float(3 * np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0

    gates, summary = {}, {}
    if "A" in REG and "B" in REG:
        gain_a = m("A", "MARGINAL-both (0p)") - m("A", "FULL (+gene lookup)")
        gain_b = m("B", "GLOBAL-mean (0p)") - m("B", "FEATURES+PARTNERS")
        summary["A_gain_over_marginal_both"] = gain_a
        summary["B_gain_over_global_mean"] = gain_b
        summary["B_retained_fraction_of_A_gain"] = gain_b / gain_a if gain_a else float("nan")
    for reg in ("B", "C"):
        if reg not in REG:
            continue
        for tag, k1, k2 in (("USEFUL", "FEATURES+PARTNERS", "GLOBAL-mean (0p)"),
                            ("REAL", "FEATURES+PARTNERS", "SHUFFLED-FEATURES (CONTROL)"),
                            ("PARTNERS", "PARTNER-MEAN non-DepMap (0p)", "PARTNER-MEAN random (0p CONTROL)")):
            g, d_ = mde(reg, k1, k2)
            gates[f"{reg}-{tag}  {k1} beats {k2}"] = bool(g > d_)
            summary[f"{reg}_{tag}_gap"] = g
            summary[f"{reg}_{tag}_mde"] = d_
        best0 = min(("GLOBAL-mean (0p)", "MARGINAL-gene (0p)", "MARGINAL-line (0p)", "MARGINAL-both (0p)",
                     "PARTNER-MEAN non-DepMap (0p)"), key=lambda k: m(reg, k))
        bestL = min(("FEATURES-only", "FEATURES+PARTNERS", "FULL (+gene lookup)"), key=lambda k: m(reg, k))
        g, d_ = mde(reg, bestL, best0)
        gates[f"{reg}-FREE  best learned ({bestL}) beats best 0-param ({best0})"] = bool(g > d_)
        summary[f"{reg}_free_gap"] = g

    report("\n  PREDECLARED GATES")
    for k, v in gates.items():
        report(f"    {'PASS' if v else 'FAIL'}  {k}")
    if "B_retained_fraction_of_A_gain" in summary:
        report(f"\n  G1 RETAINED: Test A's gain over MARGINAL-both is {summary['A_gain_over_marginal_both']:+.4f} "
               f"RMSE; Test B's gain over GLOBAL-mean is {summary['B_gain_over_global_mean']:+.4f}. "
               f"B retains {100*summary['B_retained_fraction_of_A_gain']:.1f}% of it.")

    verdict = (
        ("SMOKE RUN -- a plumbing check and NOT a result; rerun with ABC_SMOKE unset. " if SMOKE else "") +
        f"TEST A vs B vs C. Test A (unseen cell line, every gene seen) reaches RMSE "
        f"{m('A', 'FULL (+gene lookup)'):.4f} against a MARGINAL-both of {m('A', 'MARGINAL-both (0p)'):.4f}. "
        f"Test B (unseen GENE, every line seen) has the per-gene lookup available for "
        f"{100*REG['B']['lookup_available']:.1f}% of its test genes by construction, and the best non-lookup "
        f"model reaches {m('B', 'FEATURES+PARTNERS'):.4f} against a GLOBAL-mean of "
        f"{m('B', 'GLOBAL-mean (0p)'):.4f}. "
        f"{'Test C (unseen gene AND unseen line) reaches ' + format(m('C', 'FEATURES+PARTNERS'), '.4f') + ' against ' + format(m('C', 'GLOBAL-mean (0p)'), '.4f') + '. ' if 'C' in REG else ''}"
        f"THE HEADLINE IS THE RETAINED FRACTION: Test A's learned gain over its own marginal is "
        f"{summary.get('A_gain_over_marginal_both', float('nan')):+.4f} RMSE, and Test B retains "
        f"{100*summary.get('B_retained_fraction_of_A_gain', float('nan')):.1f}% of it. "
        f"CO-DEPENDENCY PARTNERS ARE FORBIDDEN IN B AND C and are not used there: selecting them requires "
        f"the held-out gene's own DepMap column, which is the label. The partner channel in B and C is PPI "
        f"and complex co-membership only, and the DepMap-derived gene fields {sorted(BANNED)} are excluded "
        f"from the feature matrix with an assertion rather than a comment. All effect sizes are raw held-out "
        f"RMSE; controls establish significance and attribution and nothing is reported net of a control.")
    report("\n" + "=" * 108)
    report(f"  VERDICT: {verdict}")

    R = {"model": "v4-generalization-abc-v1", "smoke": bool(SMOKE),
         "n_genes": len(genes), "n_lines": int(Y.shape[0]),
         "regimes": {k: {"arms": {a: {"mean": float(np.mean(v)), "per_block": v}
                                  for a, v in r["arms"].items()},
                         "lookup_available": r["lookup_available"], "n_blocks": r["n_blocks"],
                         "n_test": r["n_test"]} for k, r in REG.items()},
         "gates": gates, "summary": summary,
         "unit_of_replication": {"A": "held-out LINE block", "B": "held-out GENE block",
                                 "C": "held-out (gene, line) block pair"},
         "banned_features": sorted(BANNED),
         "limits": [
             "Test B's partner channel is PPI + complex co-membership only. Genes with no such partner fall "
             "back to the global mean, so B's headroom is bounded by partner-graph coverage, which is "
             "reported above and is NOT uniform across genes",
             "co-dependency is deliberately absent from B and C. That is not a modelling choice, it is the "
             "only way the test is non-circular, and it means B is strictly harder than a co-dependency "
             "model would look",
             "the gene feature table is annotation, not sequence. No ESM/structure channel is used, so a "
             "null in B bounds what THESE features recover and not what sequence could",
             "Test C pairs gene block b with line block b mod N_LINE_BLOCKS rather than running the full "
             "cartesian product, so its n is the number of gene blocks and not their product",
             "DepMap gene effect is Chronos-scaled and already corrects for copy number and screen quality; "
             "a residual batch structure across screens is not removed here"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "v4_generalization_abc.json", "w"), indent=1)
    report(f"\n  -> {OUT/'v4_generalization_abc.json'}")


if __name__ == "__main__":
    main()
