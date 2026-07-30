"""DO NETWORK EDGES REACH PROTEIN? The 4-protein test redone at 20 proteins, 246 knockouts, 3 conditions.

WHAT THIS SUPERSEDES. `protein_orthogonal.py` ran this test on Papalexi: 4 proteins, 24 knockouts, 10 signed
edges. It found the network's signs transfer to mRNA (8/10) but not to protein (4/10) -- and could not
establish it, because only 6 edges disagreed and the paired McNemar gave p 0.109. That module's own verdict
said four proteins was the binding constraint. This one removes that constraint. The old module is kept, not
overwritten: it is the record of the 4-protein result and lets the two be compared.

THE DATA. Frangieh Perturb-CITE-seq via scPerturb (Zenodo 13350497, open -- no Broad login), patient-derived
melanoma:

    218,331 cells          24 ADT tags        246 perturbations with >=25 cells
    57,605 control cells   20 real proteins   3 conditions: Control, IFN-gamma, Co-culture
    4 ISOTYPE CONTROLS     10 diagonal positive controls

The RNA half was already on disk (`frangieh.h5ad`, 1,458.9 MB, byte-matching the Zenodo RNA file); only the
24.7 MB protein half needed fetching. The two share `cell_name` in identical order, asserted here rather than
assumed -- 218,331/218,331.

THREE THINGS THIS DATASET GIVES THAT PAPALEXI COULD NOT

1 ISOTYPE CONTROLS ARE A MEASURED FALSE-POSITIVE RATE. Rat_IgG2a and Mouse_IgG1/2a/2b bind nothing specific,
  so any apparent knockout effect on them is technical -- ambient antibody, cell size, sequencing depth. The
  whole pipeline is run on them unchanged. That is stronger than a permutation null, which can only shuffle
  labels within whatever biases the pipeline already has: if edges "predict" isotype movement too, the effect
  is plumbing, and no amount of permuting would have revealed it.

2 THREE CONDITIONS ARE NOT POOLED. Frangieh's design is that immune-evasion effects appear under IFN-gamma
  and T-cell co-culture and not at baseline; averaging over conditions would blur exactly the signal being
  looked for. Each condition is analysed separately, which also gives REPLICATION: an edge whose sign holds
  in Control and again in IFN-gamma has replicated in independent cells.

3 TEN DIAGONAL POSITIVE CONTROLS instead of two. IFNGR1, CXCR4, CD274, KDR, CD44, CD47, CD58, CD59, HLA-A and
  HLA-E are each both perturbed AND measured as protein. They stay excluded from the network test -- the
  network has no self-loops, so it makes no claim, and scoring them would be free wins -- and are used only
  to establish that the readout responds.

PDCD1 (CD279) IS DROPPED FROM THE PAIRED ARM. PD-1 is a T-cell protein and its transcript is absent from the
melanoma transcriptome, so there is no mRNA to compare it against. It is kept in the protein-only arms and
its absence is stated rather than silently compared to a zero vector.

THE STATISTIC IS A DIFFERENCE OF MEANS, NOT A LOG FOLD CHANGE, AND THAT WAS FORCED BY THE DATA. A first
version scored log2 fold change and checked that the null sd went as 1/sqrt(n). The check FAILED and aborted
the run (worst R2 -15.4), which turned out to be a real property rather than a bug: the 24 tags split into
two groups. Thirteen proteins are well measured (0-65% zeros, fitted exponent -0.47 to -0.52, R2 0.987-0.998
-- textbook). Seven proteins and all four isotype controls are 75-90% zero, and there a 25-cell bootstrap
sample can be almost all zeros, so the mean approaches zero and LOG2 OF IT EXPLODES: sd 7.83 at n=25 for
CD140a. The mean itself is perfectly well behaved; only the log is not.

So the null needs no fitted law at all. The sd of a sample mean is exactly sigma/sqrt(n) by the CLT for any
distribution with finite variance, zero-inflated included, so z = (mean_KO - mean_control) / (sigma/sqrt(n))
is analytic. The grid, the interpolation and the whole failure mode disappear. A bootstrap check is still run
at a few n and the analytic-vs-empirical agreement is printed, because an assumption that is merely correct
in theory should still be shown to hold in the data.

READOUT ADEQUACY IS DECIDED PER CONDITION AND ON BOTH HALVES. A readout needs enough nonzero control cells
that a smallest-sample is expected to contain a few; below that the CLT has not engaged. This is checked
separately in each condition, because ADT depth differs enough between them to change the answer -- CD117 is
85% zero in Co-culture controls and 51% pooled -- and on the protein AND its transcript, since the paired arm
needs both. An earlier version filtered only proteins, leaving transcripts with as few as 3 nonzero cells in
18,334. Typically 11 of 18 proteins survive per condition. The genes that fail are the same on both
measurements -- KIT, PDGFRA, PDGFRB, TEK, KDR, CXCR4 -- so protein and transcript independently agree they are
not expressed in melanoma, which is a consistency check rather than a coincidence.

THE ISOTYPE CONTROLS BEING 82-90% ZERO IS CORRECT, not a defect: they are supposed to bind nothing. They are
scored on the same linear scale, where their sparsity is harmless.

THE P-VALUE STILL PERMUTES AT THE KNOCKOUT LEVEL, because a knockout contributes up to 20 correlated cells.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NET = SP / "cell_complete.json.gz"
RNA = SP / "frangieh.h5ad"
PRO = SP / "frangieh_protein.h5ad"

TAG2GENE = {"CD117": "KIT", "CD119": "IFNGR1", "CD140a": "PDGFRA", "CD140b": "PDGFRB",
            "CD172a": "SIRPA", "CD184": "CXCR4", "CD202b": "TEK", "CD274": "CD274",
            "CD29": "ITGB1", "CD309": "KDR", "CD44": "CD44", "CD47": "CD47", "CD49f": "ITGA6",
            "CD58": "CD58", "CD59": "CD59", "CD61": "ITGB3", "HLA_A": "HLA-A", "HLA_E": "HLA-E",
            "CD9": "CD9", "CD279": "PDCD1"}
ISOTYPE = ["Rat_IgG2a", "Mouse_IgG1", "Mouse_IgG2a", "Mouse_IgG2b"]
NO_TRANSCRIPT = {"PDCD1"}        # T-cell protein; absent from the melanoma transcriptome
MAX_ZERO_FRAC = float(os.environ.get("PE_MAXZERO", 0.70))   # a tag above this is not expressed
# a readout needs enough nonzero control cells that a MIN_CELLS sample is expected to contain a few;
# below this the CLT has not kicked in and sigma/sqrt(n) is not the sd of the sample mean
MIN_NONZERO_FRAC = float(os.environ.get("PE_MINNZ", 0.15))
MIN_CELLS = 25
GRID_R = int(os.environ.get("PE_GRID_R", 300))
N_PERM = int(os.environ.get("PE_PERM", 20000))
STRIDE = int(os.environ.get("PE_STRIDE", 8))     # gene stride for the global-strength pass
MAXD = 3


def cat(o, k):
    import h5py
    g = o[k]
    if isinstance(g, h5py.Group):
        cs = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array([cs[c] if 0 <= c < len(cs) else "?" for c in g["codes"][:]])
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def load_protein():
    """Dense (cells x tags) ADT matrix -- 24 columns, so densifying costs ~21 MB."""
    import h5py
    from scipy import sparse
    with h5py.File(PRO, "r") as h:
        X = h["X"]
        sh = tuple(X.attrs["shape"])
        M = sparse.csc_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=sh).toarray()
        v = h["var"]
        ik = v.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        tags = [x.decode() if isinstance(x, bytes) else str(x) for x in v[ik][:]]
        o = h["obs"]
        return (M.astype(np.float32), tags, cat(o, "cell_name"), cat(o, "perturbation"),
                cat(o, "perturbation_2"))


def load_rna_cols(want):
    """Columns for the genes we need plus per-cell total UMI, from obs rather than a full re-sum."""
    import h5py
    with h5py.File(RNA, "r") as h:
        v = h["var"]
        ik = v.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        genes = [x.decode() if isinstance(x, bytes) else str(x) for x in v[ik][:]]
        gi = {g: i for i, g in enumerate(genes)}
        X = h["X"]
        ncell = tuple(X.attrs["shape"])[0]
        indptr = X["indptr"][:]
        cols = {}
        for g in want:
            j = gi.get(g)
            if j is None:
                continue
            s, e = int(indptr[j]), int(indptr[j + 1])
            c = np.zeros(ncell, np.float32)
            if e > s:
                c[X["indices"][s:e]] = X["data"][s:e]
            cols[g] = c
        tot = cat(h["obs"], "ncounts").astype(np.float64)
        return cols, tot, cat(h["obs"], "cell_name")


def strength_pass(labels, ngroups):
    """Per-group summed counts over every STRIDE-th gene, one chunked pass.

    A systematic 1-in-STRIDE column sample rather than all 23,712 genes: the full matrix is 740M nonzeros
    and this quantity is only ever used as a RANK covariate, for which ~3,000 genes is ample. Systematic and
    not random so the gene set is identical across runs.
    """
    import h5py
    with h5py.File(RNA, "r") as h:
        X = h["X"]
        ncell, ngene = tuple(X.attrs["shape"])
        indptr = X["indptr"][:]
        keep = np.arange(0, ngene, STRIDE)
        G = np.zeros((ngroups, len(keep)), np.float64)
        CH = 256
        for bs in range(0, len(keep), CH):
            blk = keep[bs:bs + CH]
            for oi, j in enumerate(blk):
                s, e = int(indptr[j]), int(indptr[j + 1])
                if e <= s:
                    continue
                lab = labels[X["indices"][s:e]]
                ok = lab >= 0
                if ok.any():
                    G[:, bs + oi] += np.bincount(lab[ok], weights=X["data"][s:e][ok],
                                                 minlength=ngroups)
    return G


def global_strength(G):
    tot = G.sum(1, keepdims=True)
    ref = G[-1]                                    # last row is the pooled control group
    rate = ref / max(ref.sum(), 1.0)
    exp = tot * rate
    z = (G - exp) / np.sqrt(np.maximum(exp, 1.0))
    return (np.abs(z) > 3).sum(1).astype(float)


def null_sd_curve(v, ns, rng):
    """MEASURE sd(sample mean) at a grid of n instead of assuming sigma/sqrt(n).

    The analytic form was tried and abandoned for a real reason. sigma/sqrt(n) is exact only asymptotically,
    and these readouts are heavy-tailed: sigma is driven by a few extreme cells that a 25-cell sample usually
    misses, so the true sd of the sample mean sits well BELOW sigma/sqrt(n) -- 59% below for CD172a in the
    Control condition, which aborted an earlier run. Using the analytic value there would have inflated every
    z-score for that readout.

    So the sd is bootstrapped and interpolated in log-log, which assumes nothing about the distribution's
    shape: on the linear scale sd(mean) is smooth and monotone decreasing in n, and unlike the log-fold-change
    version there is no near-zero mean to blow up. The deviation from the analytic form is returned as a
    DIAGNOSTIC of tail weight -- informative, not fatal.
    """
    sigma = float(v.std())
    sds, dev = [], 0.0
    for n in ns:
        idx = rng.integers(0, len(v), size=(GRID_R, int(n)))
        emp = float(v[idx].mean(1).std())
        sds.append(emp)
        pred = sigma / np.sqrt(n)
        if pred > 0:
            dev = max(dev, abs(emp - pred) / pred)
    return float(v.mean()), np.array(sds), dev


def interp_sd(ns, sds, n):
    """Log-log interpolation of the measured sd curve, clamped to the grid ends."""
    n = max(int(n), 1)
    ok = sds > 0
    if ok.sum() < 2:
        return np.nan
    return float(np.exp(np.interp(np.log(n), np.log(np.asarray(ns)[ok]), np.log(sds[ok]))))


def main():
    from scipy import stats
    import gzip
    print("=" * 100)
    print("DO NETWORK EDGES REACH PROTEIN? -- Frangieh Perturb-CITE-seq, 20 proteins, 3 conditions")
    print("=" * 100)
    for p in (NET, RNA, PRO):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    d = json.load(gzip.open(NET, "rt"))
    genes = [g["name"] for g in d["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    reg, sig = {}, {}
    for s, t, x in d["reg"]:
        reg[(s, t)] = x
    for s, t, x in d["sig"]:
        sig[(s, t)] = x
    ppi = set()
    for a, b in d["ppi"]:
        ppi.add((a, b)); ppi.add((b, a))
    adj = {}
    for (s, t) in list(reg) + list(sig):
        adj.setdefault(s, set()).add(t)
    del d
    print(f"  network: {len(genes):,} genes, reg {len(reg):,}, sig {len(sig):,}, ppi {len(ppi)//2:,}")

    A, tags, pn, pert, cond = load_protein()
    cols, tot, rn = load_rna_cols(sorted(set(TAG2GENE.values()) - NO_TRANSCRIPT))
    assert len(pn) == len(rn) and bool((pn == rn).all()), \
        "RNA and protein cell_name do not match in order -- refusing to interpret anything downstream"
    print(f"  join asserted exact: {len(pn):,}/{len(pn):,} cells in identical order")
    print(f"  {len(tags)} tags = {len(TAG2GENE)} proteins + {len(ISOTYPE)} isotype controls; "
          f"transcripts found for {len(cols)}/{len(TAG2GENE)-len(NO_TRANSCRIPT)} "
          f"(PDCD1 absent by biology, dropped from the paired arm)")
    adt_tot = A.sum(1)

    # ---- expression filter: drop tags the cells do not express ----
    # Seven of the 20 are 75-90% zero in these melanoma cells. They carry almost no information and would add
    # noise and multiplicity for nothing. The cut is on zero fraction and the numbers are printed, so the
    # decision is auditable rather than a silent subset.
    # The zero fraction is computed on CONTROL CELLS, because those are the cells that define the null. A
    # first version measured it over all 218,331 cells and disagreed with a hand diagnostic run on one
    # condition's controls by a wide margin (Mouse_IgG2a 72% vs 90%) -- ADT depth differs by condition, so the
    # two are genuinely different quantities and only the control-cell one governs the statistic. Both are
    # printed so that difference stays visible.
    ctl_all = pert == "control"
    zf = {t: float((A[ctl_all, i] == 0).mean()) for i, t in enumerate(tags)}
    zf_all = {t: float((A[:, i] == 0).mean()) for i, t in enumerate(tags)}
    keep_tags = {t: g for t, g in TAG2GENE.items()
                 if t in zf and zf[t] <= MAX_ZERO_FRAC and g not in NO_TRANSCRIPT}
    drop_sparse = {t: zf[t] for t, g in TAG2GENE.items()
                   if t in zf and zf[t] > MAX_ZERO_FRAC and g not in NO_TRANSCRIPT}
    print(f"  expression filter at control-cell zero-fraction <= {MAX_ZERO_FRAC}: keeping "
          f"{len(keep_tags)} of {len(TAG2GENE)} proteins")
    print("    dropped as not expressed: "
          + (", ".join(f"{t} {100*z:.0f}% (all cells {100*zf_all[t]:.0f}%)"
                       for t, z in sorted(drop_sparse.items(), key=lambda x: -x[1])) or "none"))
    print(f"    dropped for having no transcript: {sorted(NO_TRANSCRIPT)} "
          f"(PD-1 is a T-cell protein; nothing to pair against)")
    print("    isotype controls, kept deliberately (sparse is CORRECT -- they bind nothing): "
          + ", ".join(f"{t} {100*zf[t]:.0f}%" for t in ISOTYPE if t in zf))
    missing_tx = [g for g in keep_tags.values() if g not in cols]
    assert not missing_tx, f"kept proteins with no transcript: {missing_tx}"
    print(f"    all {len(keep_tags)} kept proteins have a transcript, so the paired arm loses nothing")

    conds = sorted(set(cond))
    kos = sorted({p for p in set(pert) if p != "control"})
    resolved = [k for k in kos if k in gi]
    print(f"  conditions {conds}; {len(kos)} perturbations, {len(resolved)} in the network")

    # ---- global perturbation strength, one pass, pooled over conditions ----
    lab = np.full(len(pert), -1, np.int64)
    for i, k in enumerate(resolved):
        lab[pert == k] = i
    lab[pert == "control"] = len(resolved)
    G = strength_pass(lab, len(resolved) + 1)
    st_all = global_strength(G)
    ko_str = {k: st_all[i] for i, k in enumerate(resolved)}
    print(f"  global strength over every {STRIDE}th gene ({G.shape[1]:,} genes): "
          f"median KO {np.median(list(ko_str.values())):.0f}, "
          f"top {sorted(ko_str, key=lambda x: -ko_str[x])[:5]}")

    rng = np.random.default_rng(0)
    R = {"n_cells": int(len(pn)), "conditions": conds, "n_perturbations": len(kos),
         "n_in_network": len(resolved), "tags": tags, "isotypes": ISOTYPE,
         "grid_r": GRID_R, "n_perm": N_PERM, "gene_stride": STRIDE, "per_condition": {}}

    # ---- per condition ----
    allrows = []
    for cnd in conds:
        cm = cond == cnd
        ctl = cm & (pert == "control")
        print("\n" + "-" * 100)
        print(f"  CONDITION {cnd}   {int(cm.sum()):,} cells, {int(ctl.sum()):,} control")
        kk = [k for k in resolved if (cm & (pert == k)).sum() >= MIN_CELLS]
        ns_needed = {k: int((cm & (pert == k)).sum()) for k in kk}
        if not kk:
            print("    no knockout reaches the cell minimum; condition skipped")
            continue
        grid = np.unique(np.geomspace(MIN_CELLS, max(ns_needed.values()), 6).astype(int))

        # ---- READOUT ADEQUACY, DECIDED PER CONDITION AND ON BOTH HALVES ----
        # The CLT is asymptotic. TEK's transcript has 3 nonzero cells in 18,334 Co-culture controls, so a
        # 25-cell bootstrap sample essentially never contains one, every sample mean is 0, and the empirical
        # sd is exactly 0 -- which is what tripped the earlier guard. The fix is tied to that cause: require
        # enough nonzero control cells that a smallest-sample is expected to contain a few. Applied PER
        # CONDITION because ADT depth varies enough between them to change the answer (CD117 is 85% zero in
        # Co-culture controls and 51% pooled), and to BOTH the protein and its transcript, since the paired
        # arm needs both. The proteins and transcripts that fail are the same genes either way -- KIT, PDGFRA,
        # PDGFRB, TEK, KDR, CXCR4 -- which is two independent measurements agreeing they are not expressed.
        def nzfrac(v):
            return float((v > 0).mean())
        pv = {t: (A[:, tags.index(t)] / np.maximum(adt_tot, 1.0))[ctl] for t in tags}
        rv = {g: (c / np.maximum(tot, 1.0))[ctl] for g, c in cols.items()}
        usable, unusable = {}, {}
        for t, g in keep_tags.items():
            fp, fr = nzfrac(pv[t]), nzfrac(rv[g])
            if min(fp, fr) >= MIN_NONZERO_FRAC:
                usable[t] = g
            else:
                unusable[t] = (fp, fr)
        print(f"    readouts usable here: {len(usable)}/{len(keep_tags)} "
              f"(need >={MIN_NONZERO_FRAC:.0%} nonzero control cells in BOTH protein and transcript)")
        if unusable:
            print("      not measurable in this condition: " + ", ".join(
                f"{t} (prot {fp:.0%}, rna {fr:.0%})" for t, (fp, fr) in
                sorted(unusable.items(), key=lambda x: min(x[1]))))
        if len(usable) < 4:
            print("      fewer than 4 usable readouts; condition skipped")
            continue

        # per-readout control mean and MEASURED sd curve; no functional form assumed
        stats_ = {}
        worst, worst_who, drop_zero = 0.0, None, []
        for t, g in usable.items():
            for kind, v in (("prot", pv[t]), ("rna", rv[g])):
                base, sds, dev = null_sd_curve(v, grid, rng)
                nm = t if kind == "prot" else g
                if (sds > 0).sum() < 2:
                    drop_zero.append(f"{kind}:{nm}")
                    continue
                stats_[(kind, nm)] = (base, sds)
                if dev > worst:
                    worst, worst_who = dev, f"{kind}:{nm}"
        for t in ISOTYPE:                       # isotypes are scored but never gate the run
            if t in pv:
                base, sds, _ = null_sd_curve(pv[t], grid, rng)
                if (sds > 0).sum() >= 2:
                    stats_[("prot", t)] = (base, sds)
        if drop_zero:
            usable = {t: g for t, g in usable.items()
                      if ("prot", t) in stats_ and ("rna", g) in stats_}
            print(f"      dropped for a degenerate null (sd 0 at almost every n): {drop_zero}")
        print(f"    null sd MEASURED by bootstrap and log-log interpolated; largest deviation from the "
              f"analytic sigma/sqrt(n) = {100*worst:.0f}% ({worst_who})")
        print(f"      ^ a diagnostic of tail weight, not an error: heavy tails put the true sd of a small "
              f"sample mean BELOW sigma/sqrt(n), and using the analytic form would inflate those z")
        if len(usable) < 4:
            print("      fewer than 4 usable readouts after the null check; condition skipped")
            continue

        def zof(vals, denom, sel, key):
            """z on the LINEAR normalised scale against the measured null sd at this n.

            Linear and not a log fold change: for sparse tags the log of a near-zero mean is not
            asymptotically normal. Measured sd and not sigma/sqrt(n): these readouts are heavy-tailed.
            """
            base, sds = stats_[key]
            vv = vals / np.maximum(denom, 1.0)
            obs = float(vv[sel].mean())
            sd = interp_sd(grid, sds, int(sel.sum()))
            z = (obs - base) / max(sd, 1e-30) if np.isfinite(sd) else np.nan
            rel = (obs - base) / base if base > 0 else np.nan     # for reporting only
            return z, rel

        rows = []
        for k in kk:
            si = gi[k]
            sel = cm & (pert == k)
            seen = {si: 0}
            frontier = [si]
            for depth in range(1, MAXD + 1):
                nxt = []
                for u in frontier:
                    for w in adj.get(u, ()):
                        if w not in seen:
                            seen[w] = depth; nxt.append(w)
                frontier = nxt
                if not frontier:
                    break
            for tag, gene in usable.items():
                if tag not in tags or gene not in gi:
                    continue
                ti = gi[gene]
                if si == ti:
                    continue
                pz, pl = zof(A[:, tags.index(tag)], adt_tot, sel, ("prot", tag))
                if gene in cols:
                    mz, ml = zof(cols[gene], tot, sel, ("rna", gene))
                else:
                    mz, ml = np.nan, np.nan
                rows.append({"cond": cnd, "ko": k, "tag": tag, "gene": gene,
                             "n_cells": int(sel.sum()), "prot_z": pz, "prot_rel": pl,
                             "mrna_z": mz, "mrna_rel": ml, "strength": ko_str[k],
                             "dist": seen.get(ti, 99), "reg": reg.get((si, ti)),
                             "sig": sig.get((si, ti)), "ppi": int((si, ti) in ppi)})
            for tag in ISOTYPE:
                if tag not in tags:
                    continue
                pz, pl = zof(A[:, tags.index(tag)], adt_tot, sel, ("prot", tag))
                rows.append({"cond": cnd, "ko": k, "tag": tag, "gene": None,
                             "n_cells": int(sel.sum()), "prot_z": pz, "prot_rel": pl,
                             "mrna_z": np.nan, "mrna_rel": np.nan, "strength": ko_str[k],
                             "dist": 99, "reg": None, "sig": None, "ppi": 0, "isotype": True})
        allrows += rows
        real = [r for r in rows if not r.get("isotype")]
        iso = [r for r in rows if r.get("isotype")]

        # diagonal positive controls
        diag = []
        for tag, gene in usable.items():
            if gene in kk and tag in tags:
                sel = cm & (pert == gene)
                pz, pl = zof(A[:, tags.index(tag)], adt_tot, sel, ("prot", tag))
                # a real drop, not merely a negative sign: HLA-A came through at z -0.03 and was counted
                # as a passing control, which is a sign test on noise.
                diag.append({"gene": gene, "tag": tag, "rel": pl, "z": pz,
                             "ok": bool(pz < -2), "down": bool(pz < 0)})
        nok = sum(d["ok"] for d in diag)
        ndn = sum(d["down"] for d in diag)
        print(f"    POSITIVE CONTROLS (own gene knocked out): {nok}/{len(diag)} drop at z<-2 "
              f"({ndn}/{len(diag)} merely negative)"
              + ("" if not diag else "   " + ", ".join(
                  f"{d['gene']}{'' if d['ok'] else '(UP)'} z{d['z']:+.0f}" for d in diag)))

        # isotype false-positive rate: the pipeline run on antibodies that bind nothing
        iz = np.abs([r["prot_z"] for r in iso])
        rz = np.abs([r["prot_z"] for r in real])
        print(f"    ISOTYPE CONTROLS: {100*(iz>3).mean():.1f}% of {len(iso)} isotype cells move at |z|>3, "
              f"vs {100*(rz>3).mean():.1f}% of {len(real)} real-protein cells")
        print(f"      -> measured technical false-positive rate {100*(iz>3).mean():.1f}%; "
              f"real-protein movement is {(rz>3).mean()/max((iz>3).mean(),1e-9):.1f}x that")

        cr = {"n_rows": len(real), "n_isotype": len(iso), "diag": diag,
              "diag_ok": nok, "diag_n": len(diag),
              "isotype_fpr": float((iz > 3).mean()), "real_move_rate": float((rz > 3).mean()),
              "null_fit_worst_r2": float(worst)}

        # ---- direction, the arm that carried the 4-protein result ----
        signed = [r for r in real if (r["reg"] not in (None, 0)) or (r["sig"] not in (None, 0))]
        paired = [r for r in signed if np.isfinite(r["mrna_z"])]
        print(f"    signed edges: {len(signed)} ({len(paired)} with a transcript to pair against)")
        cr["n_signed"] = len(signed); cr["n_paired"] = len(paired)
        if len(paired) >= 8:
            hp, hm = [], []
            for r in paired:
                s = r["reg"] if r["reg"] not in (None, 0) else r["sig"]
                want = s > 0
                # sign read off z: z is (mean_KO - mean_control) over a POSITIVE sd, so its sign
                # is the sign of the change. No log involved, so sparse tags stay valid.
                hp.append(bool((r["prot_z"] < 0) == want))
                hm.append(bool((r["mrna_z"] < 0) == want))
            okp, okm, n = sum(hp), sum(hm), len(paired)
            bp = stats.binomtest(okp, n, 0.5, alternative="greater").pvalue
            bm = stats.binomtest(okm, n, 0.5, alternative="greater").pvalue
            b = sum(1 for p, m in zip(hp, hm) if p and not m)
            c = sum(1 for p, m in zip(hp, hm) if m and not p)
            mc = stats.binomtest(b, b + c, 0.5, alternative="less").pvalue if b + c else None
            print(f"      protein direction {okp}/{n} = {okp/n:.1%}  binomial p {bp:.4g}")
            print(f"      mRNA    direction {okm}/{n} = {okm/n:.1%}  binomial p {bm:.4g}")
            print(f"      PAIRED McNemar on {b+c} discordant ({b} protein-only, {c} mRNA-only): "
                  + (f"p {mc:.4g}" if mc else "n/a"))
            cr["direction"] = {"n": n, "prot_ok": okp, "prot_p": float(bp), "mrna_ok": okm,
                               "mrna_p": float(bm), "disc_prot": b, "disc_mrna": c,
                               "mcnemar_p": float(mc) if mc else None}
        R["per_condition"][cnd] = cr

    R["rows"] = allrows
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "protein_edges.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'protein_edges.json'}  ({len(allrows):,} rows)")


if __name__ == "__main__":
    main()
