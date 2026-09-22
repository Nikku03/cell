"""reg_perturb_k562 -- a regulatory layer built from INTERVENTIONS, and validated by transferring to RPE1.

WHAT IS WRONG WITH THE LAYER THIS REPLACES. The cell object's `reg` is 612,133 triples `[source, target,
sign]` with no field for evidence and `sign = 0` on 558,005 of them (91.2%). Run against a bounded task --
K562 CRISPRi perturbation to the transcriptional response of every measured gene -- it added -0.0002 held-out
R2 over a nuisance ladder, below its own degree-matched control. Asked directly and at full power it does
carry something, but only 2.1% elevation in |log fold change|, which is magnitude and not direction. The
ladder itself stalls at R2 0.0151 for exactly the same reason: nothing in it supplies a SIGN.

A curated edge says someone reported that A relates to B. Replogle's genome-scale Perturb-seq says what
actually happened to B when A was knocked down, in this cell type, with an effect size and a direction:

    curated                          A -> B
    what an intervention gives       E[dB | do(A), K562]  and  P(dB != 0 | do(A), K562)

THE CIRCULARITY THAT HAD TO BE AVOIDED. The admission benchmark's task IS Replogle K562. A layer derived from
that matrix and then scored on it would be reporting its own training data, and it would score spectacularly.
So the layer is built on K562 and validated on a DIFFERENT CELL LINE -- Replogle's RPE1 arm, 247,914 cells
over 2,394 perturbations with 11,485 non-targeting controls. Same assay, same lab, same processing, different
biology. That is the transfer question and the tiering question at once: an edge that survives the move is
context-general, an edge that dies is K562-specific, and both facts are worth storing.

THE NORMALISATION IS THE COLUMN EFFECT, DELIBERATELY. Some genes respond to almost any perturbation -- they
are stress and proliferation reporters -- and a raw fold change is dominated by them. Every effect here is a
per-gene robust z, (lfc - median over perturbations) / (1.4826 * MAD), so an edge means "B moved more than B
usually moves", not "B moved". This is the same quantity the benchmark's ladder already spends its
gene-responsiveness rung on, so an edge defined this way is not re-selling a rung the ladder already has.

TWO QUESTIONS, AND THE SECOND IS THE ONE THAT MATTERS.
    MAGNITUDE TRANSFER   do K562 edges show elevated |z| in RPE1, against pairs matched on RPE1 perturbation
                         strength x RPE1 gene responsiveness? Matching on both is what stops the answer being
                         "essential genes are disruptive everywhere".
    SIGN TRANSFER        among those, does the DIRECTION agree above chance? Sign is the thing the curated
                         layer does not have and the ladder cannot reach, so sign agreement is the result
                         that would actually change what the model can do.

NOT OVERWRITING `reg`. This is written as a separate tier. A curated edge and a perturbational edge are
different epistemic objects and merging them into one binary graph is what produced a 612k-edge layer that
cannot move a number.
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
GWPS = SP / "gwps.h5ad"          # K562 pseudobulk, 11,258 perturbations x 8,248 genes
RPE1 = SP / "rpe1.h5ad"          # RPE1 cell level, 247,914 cells x 8,749 genes
LAYER = SP / "cell_reg_perturb.json.gz"
PB_CACHE = SP / "rpe1_pseudobulk.npz"
Z_EDGE = 5.0                     # |robust z| at which a (perturbation, gene) pair becomes an edge
MIN_CELLS = 25                   # a perturbation below this has no usable pseudobulk
CHUNK = 20_000


def robust_z(M):
    """Per-COLUMN robust z: (x - median) / (1.4826 * MAD), columns = measured genes.

    Median and MAD rather than mean and sd because the column is exactly where a few enormous effects live,
    and those are the signal -- letting them set the scale would shrink them toward the noise.
    """
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)          # a gene with no variation cannot be z-scored
    return (M - med) / mad


def load_k562():
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        gid = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    return X, psym, np.array(gid)


def _cat(f, grp, key):
    o = f[grp][key]
    import h5py
    if isinstance(o, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        return np.array(cats), o["codes"][:]
    v = o[:]
    return None, v


def pseudobulk_rpe1(report):
    """Sum raw counts per perturbation, CPM-normalise, log1p, subtract the non-targeting profile.

    Read in chunks: the full matrix is 247,914 x 8,749 float32 = 8.7 GB and the container has ~7 GB of disk
    headroom, let alone RAM. Summing into a (perturbation x gene) accumulator needs one chunk resident.
    """
    import h5py
    if PB_CACHE.exists():
        z = np.load(PB_CACHE, allow_pickle=True)
        report(f"    RPE1 pseudobulk from cache: {z['lfc'].shape[0]:,} perturbations")
        return z["lfc"], np.array([str(x) for x in z["perts"]]), np.array([str(x) for x in z["genes"]]), \
            z["ncells"]
    with h5py.File(RPE1, "r") as f:
        cats, codes = _cat(f, "obs", "gene")
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var"]["ensembl_id"][:]])
        n, G = f["X"].shape
        S = np.zeros((len(cats), G), dtype=np.float64)
        C = np.zeros(len(cats), dtype=np.int64)
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            blk = f["X"][i0:i1]
            cd = codes[i0:i1]
            np.add.at(S, cd, blk)
            np.add.at(C, cd, 1)
            if (i0 // CHUNK) % 4 == 0:
                report(f"      pseudobulk {i1:,}/{n:,} cells")
    tot = S.sum(1, keepdims=True)
    cpm = np.divide(S, np.maximum(tot, 1), where=True) * 1e6
    lg = np.log2(1.0 + cpm)
    ci = int(np.where(cats == "non-targeting")[0][0])
    lfc = lg - lg[ci]
    keep = (C >= MIN_CELLS) & (np.arange(len(cats)) != ci)
    report(f"    RPE1 pseudobulk: {keep.sum():,}/{len(cats):,} perturbations with >= {MIN_CELLS} cells "
           f"(control n = {C[ci]:,})")
    np.savez_compressed(PB_CACHE, lfc=lfc[keep].astype(np.float32), perts=cats[keep],
                        genes=gid, ncells=C[keep])
    return lfc[keep].astype(np.float32), cats[keep], gid, C[keep]


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("reg_perturb_k562 -- causal regulatory edges from intervention, validated by transfer to RPE1")
    report("=" * 100)
    for p in (GWPS, RPE1):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    from entity_registry import load as load_reg
    REG = load_reg()

    # ---- K562 edges ----
    X, psym, kg = load_k562()
    report(f"  K562: {X.shape[0]:,} perturbations x {X.shape[1]:,} measured genes")
    Zk = robust_z(X)
    report(f"  per-gene robust z; |z| >= {Z_EDGE} defines an edge")

    # ---- RPE1 ----
    report(f"\n  RPE1 (held-out cell line)")
    Lr, rperts, rg, rn = pseudobulk_rpe1(report)
    Zr = robust_z(Lr)
    report(f"    RPE1: {Lr.shape[0]:,} perturbations x {Lr.shape[1]:,} genes, "
           f"median {np.median(rn):.0f} cells/perturbation")

    # ---- shared coordinates, joined through the registry ----
    _g, st_k = REG.join("gene", "ensembl", list(kg), 0.80, "K562 measured genes")
    _g, st_r = REG.join("gene", "ensembl", list(rg), 0.80, "RPE1 measured genes")
    report(f"\n  registry joins: K562 genes {st_k['rate']:.1%}, RPE1 genes {st_r['rate']:.1%}")
    _p, st_p = REG.join("gene", "symbol", list(psym), 0.80, "K562 perturbations")
    _p, st_q = REG.join("gene", "symbol", list(rperts), 0.80, "RPE1 perturbations")
    report(f"  registry joins: K562 perturbations {st_p['rate']:.1%}, "
           f"RPE1 perturbations {st_q['rate']:.1%}")

    kgi = {g: i for i, g in enumerate(kg)}
    shared_g = [g for g in rg if g in kgi]
    kcol = np.array([kgi[g] for g in shared_g])
    rcol = np.array([i for i, g in enumerate(rg) if g in kgi])
    kpi = {s: i for i, s in enumerate(psym)}
    shared_p = [s for s in rperts if s in kpi]
    krow = np.array([kpi[s] for s in shared_p])
    rrow = np.array([i for i, s in enumerate(rperts) if s in kpi])
    report(f"  shared with RPE1: {len(shared_p):,} perturbations x {len(shared_g):,} genes "
           f"= {len(shared_p)*len(shared_g):,} testable pairs")

    ZK = Zk[np.ix_(krow, kcol)]
    ZR = Zr[np.ix_(rrow, rcol)]
    ok = np.isfinite(ZK) & np.isfinite(ZR)
    report(f"  finite in both: {ok.sum():,} ({100*ok.mean():.1f}%)")

    # ---- the transfer test ----
    edge = ok & (np.abs(ZK) >= Z_EDGE)
    report(f"\n  K562 edges among testable pairs: {int(edge.sum()):,} "
           f"({100*edge.sum()/max(ok.sum(),1):.3f}%)")

    # matched null: same RPE1 perturbation-strength decile x RPE1 gene-responsiveness decile.
    # Matching on BOTH is what stops the answer collapsing to "essential genes are disruptive everywhere".
    rs = np.nanmean(np.abs(ZR), axis=1)
    gr = np.nanmean(np.abs(ZR), axis=0)
    rq = np.digitize(rs, np.nanquantile(rs, np.linspace(.1, .9, 9)))
    gq = np.digitize(gr, np.nanquantile(gr, np.linspace(.1, .9, 9)))
    ei, ej = np.where(edge)
    rng = np.random.default_rng(0)
    key = {}
    for a, b in zip(ei, ej):
        key.setdefault((rq[a], gq[b]), []).append((a, b))
    nulls = []
    for (qa, qb), grp in key.items():
        ra, rb = np.where(rq == qa)[0], np.where(gq == qb)[0]
        if len(ra) and len(rb):
            s = ZR[rng.choice(ra, len(grp)), rng.choice(rb, len(grp))]
            nulls.append(s[np.isfinite(s)])
    nl = np.concatenate(nulls)
    ev = ZR[ei, ej]
    p_mag = float(stats.mannwhitneyu(np.abs(ev), np.abs(nl), alternative="two-sided")[1])
    report(f"\n  MAGNITUDE TRANSFER (does a K562 edge move its target in RPE1?)")
    report(f"    {'group':34s} {'n':>9s} {'mean |z| RPE1':>14s} {'frac |z|>2':>11s}")
    report(f"    {'K562 edge':34s} {len(ev):9,d} {np.abs(ev).mean():14.4f} "
           f"{np.mean(np.abs(ev) > 2):11.4f}")
    report(f"    {'matched on strength x responsive':34s} {len(nl):9,d} {np.abs(nl).mean():14.4f} "
           f"{np.mean(np.abs(nl) > 2):11.4f}")
    report(f"    ratio {np.abs(ev).mean()/np.abs(nl).mean():.3f}x   MWU p {p_mag:.3g}")

    # ---- sign, the result that would change what the model can do ----
    #
    # AND ITS CONTROL, WITHOUT WHICH THE NUMBER IS UNREADABLE. Both cell lines share a dominant stress and
    # proliferation axis: knock down anything essential and a stereotyped set of genes moves the same way in
    # K562 and in RPE1. That alone would produce high sign agreement with no gene-specific regulation
    # anywhere. So agreement is measured against two permutations that each destroy one thing:
    #
    #   SWAPPED SOURCE   keep the target gene B and the K562 sign, but read RPE1 at a DIFFERENT perturbation
    #                    A' matched on RPE1 strength. If a global axis is doing the work, A' does as well as
    #                    A, because B's preferred direction is a property of B.
    #   SWAPPED TARGET   keep the perturbation A, read RPE1 at a different gene B' matched on
    #                    responsiveness. This one is expected to sit near chance and is the sanity check on
    #                    the other.
    #
    # The quantity that means anything is the edge value MINUS the swapped-source value. Selection on RPE1
    # |z| >= 2 conditions on the outcome and inflates agreement, so its control is conditioned identically.
    agree = np.sign(ZK[ei, ej]) == np.sign(ev)
    strong = np.abs(ev) >= 2
    p_sign = float(stats.binomtest(int(agree.sum()), len(agree), 0.5).pvalue)
    p_sign_s = float(stats.binomtest(int(agree[strong].sum()), int(strong.sum()), 0.5).pvalue) \
        if strong.sum() > 20 else float("nan")

    def swap_agree(mode, seed):
        r2 = np.random.default_rng(seed)
        if mode == "source":
            alt = np.empty(len(ei), dtype=int)
            for qa in np.unique(rq):
                m = rq[ei] == qa
                pool = np.where(rq == qa)[0]
                alt[m] = r2.choice(pool, int(m.sum()))
            zr = ZR[alt, ej]
        else:
            alt = np.empty(len(ej), dtype=int)
            for qb in np.unique(gq):
                m = gq[ej] == qb
                pool = np.where(gq == qb)[0]
                alt[m] = r2.choice(pool, int(m.sum()))
            zr = ZR[ei, alt]
        f = np.isfinite(zr)
        ag = (np.sign(ZK[ei, ej]) == np.sign(zr))[f]
        st = np.abs(zr[f]) >= 2
        return float(ag.mean()), (float(ag[st].mean()) if st.sum() > 20 else float("nan"))

    sw_src = [swap_agree("source", 100 + s) for s in range(5)]
    sw_tgt = [swap_agree("target", 200 + s) for s in range(5)]
    a_src, a_src_s = float(np.mean([a for a, _b in sw_src])), float(np.mean([b for _a, b in sw_src]))
    a_tgt, a_tgt_s = float(np.mean([a for a, _b in sw_tgt])), float(np.mean([b for _a, b in sw_tgt]))
    report(f"\n  SIGN TRANSFER (the thing the curated layer does not have) -- with permutation controls")
    report(f"    {'arm':34s} {'n':>9s} {'agrees':>9s} {'|z|>=2 subset':>15s}")
    report(f"    {'K562 edge -> RPE1 same pair':34s} {len(agree):9,d} {agree.mean():9.4f} "
           f"{agree[strong].mean():15.4f}")
    report(f"    {'swapped source (matched strength)':34s} {len(agree):9,d} {a_src:9.4f} {a_src_s:15.4f}")
    report(f"    {'swapped target (matched respons.)':34s} {len(agree):9,d} {a_tgt:9.4f} {a_tgt_s:15.4f}")
    report(f"    edge-specific increment over the swapped-source control: "
           f"{agree.mean()-a_src:+.4f}  (conditioned subset {agree[strong].mean()-a_src_s:+.4f})")
    report(f"    binomial p vs 0.5: all {p_sign:.3g}, conditioned {p_sign_s:.3g} -- but 0.5 is the WRONG "
           f"null here;\n      the swapped-source arm is the null, because a shared stress axis makes "
           f"agreement exceed 0.5 with no edge at all")

    R = {"z_edge": Z_EDGE, "min_cells": MIN_CELLS,
         "k562": {"n_perturbations": int(X.shape[0]), "n_genes": int(X.shape[1])},
         "rpe1": {"n_perturbations": int(Lr.shape[0]), "n_genes": int(Lr.shape[1]),
                  "median_cells": float(np.median(rn))},
         "shared": {"perturbations": len(shared_p), "genes": len(shared_g),
                    "testable_pairs": int(ok.sum()), "k562_edges": int(edge.sum())},
         "magnitude_transfer": {"mean_abs_z_edge": float(np.abs(ev).mean()),
                                "mean_abs_z_matched": float(np.abs(nl).mean()),
                                "ratio": float(np.abs(ev).mean() / np.abs(nl).mean()), "p": p_mag},
         "sign_transfer": {"n": int(len(agree)), "agree": float(agree.mean()), "p_vs_half": p_sign,
                           "n_strong": int(strong.sum()), "agree_strong": float(agree[strong].mean()),
                           "p_strong_vs_half": p_sign_s,
                           "swapped_source": a_src, "swapped_source_strong": a_src_s,
                           "swapped_target": a_tgt, "swapped_target_strong": a_tgt_s,
                           "increment_over_swapped_source": float(agree.mean() - a_src),
                           "increment_strong": float(agree[strong].mean() - a_src_s),
                           "null_note": "0.5 is the wrong null: a shared stress/proliferation axis makes "
                                        "sign agree above chance with no edge at all. The swapped-source "
                                        "arm is the null."},
         "registry_joins": {"k562_genes": st_k, "rpe1_genes": st_r,
                            "k562_perturbations": st_p, "rpe1_perturbations": st_q}}

    # ---- write the layer ----
    ai, aj = np.where(np.isfinite(Zk) & (np.abs(Zk) >= Z_EDGE))
    report(f"\n  writing {len(ai):,} K562 perturbational edges "
           f"({100*len(ai)/max(np.isfinite(Zk).sum(),1):.3f}% of finite pairs)")
    rset = {(a, b) for a, b in zip(ei, ej)}
    shp = {s: i for i, s in enumerate(shared_p)}
    shg = {g: i for i, g in enumerate(shared_g)}
    edges = []
    for a, b in zip(ai, aj):
        src, tgt = psym[a], kg[b]
        e = {"source": str(src), "target_ensembl": str(tgt),
             "z": round(float(Zk[a, b]), 3), "lfc": round(float(X[a, b]), 4),
             "sign": int(np.sign(Zk[a, b]))}
        si, gi2 = shp.get(str(src)), shg.get(str(tgt))
        if si is not None and gi2 is not None and (si, gi2) in rset:
            e["rpe1_z"] = round(float(ZR[si, gi2]), 3)
            e["rpe1_sign_agrees"] = bool(np.sign(Zk[a, b]) == np.sign(ZR[si, gi2]))
        edges.append(e)
    tested = sum(1 for e in edges if "rpe1_z" in e)
    layer = {
        "model": "reg-perturb-k562-v1",
        "evidence_class": "perturbational",
        "assay": "CRISPRi Perturb-seq (Replogle genome-scale)",
        "cell_type": "K562",
        "definition": f"per-gene robust z of log fold change, |z| >= {Z_EDGE}",
        "n_edges": len(edges),
        "does_not_replace": "`reg` (612,133 curated triples) is a different evidence class and is left "
                            "intact; these are stored as a separate tier",
        "validation": {k: R[k] for k in ("magnitude_transfer", "sign_transfer", "shared")},
        "limits": [
            "CRISPRi knocks DOWN rather than out, so an edge absent here may reflect incomplete knockdown "
            "rather than absent regulation",
            "direct and indirect are not separated -- an edge is a causal consequence of perturbing the "
            "source, not evidence the source binds the target; ENCODE K562 TF ChIP is the intended "
            "discriminator and is not applied here",
            "only 8,248 genes are measured, so absence of an edge to an unmeasured gene is not evidence",
            f"transfer is tested on RPE1 only; {tested:,} of {len(edges):,} edges carry an RPE1 result and "
            f"the rest are untested rather than validated",
            "z is computed against the distribution across perturbations, so it is a within-experiment "
            "quantity and is not comparable across datasets without renormalisation"],
        "edges": edges}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    R["n_edges_written"] = len(edges)
    R["n_edges_with_rpe1_test"] = tested

    report("\n" + "=" * 100)
    mt, sg = R["magnitude_transfer"], R["sign_transfer"]
    # THE SIGN RESULT IS READ AGAINST THE SWAPPED-SOURCE ARM, NOT AGAINST 0.5. Agreement above chance is
    # what a shared stress axis produces on its own; agreement above a perturbation-swapped control is what
    # an edge produces.
    inc, inc_s = sg["increment_over_swapped_source"], sg["increment_strong"]
    magok = mt["p"] < 0.05 and mt["ratio"] > 1.1
    if magok and inc > 0.03:
        v = (f"PERTURBATIONAL EDGES TRANSFER, AND THEY CARRY A SIGN. K562 edges move their targets "
             f"{mt['ratio']:.2f}x more in RPE1 than pairs matched on RPE1 perturbation strength x gene "
             f"responsiveness (p {mt['p']:.3g}). Direction agrees {100*sg['agree']:.1f}% against "
             f"{100*sg['swapped_source']:.1f}% when the source is swapped for another perturbation of "
             f"matched strength -- an edge-specific increment of {100*inc:+.1f} points, rising to "
             f"{100*inc_s:+.1f} on the RPE1 |z| >= 2 subset ({100*sg['agree_strong']:.1f}% vs "
             f"{100*sg['swapped_source_strong']:.1f}%). The raw 0.5 comparison is not the null: a shared "
             f"stress and proliferation axis already pushes agreement to "
             f"{100*sg['swapped_source']:.1f}% with no edge at all. Sign is what the curated layer lacks on "
             f"91.2% of its edges and what the benchmark ladder cannot reach.")
    elif magok and inc > 0:
        v = (f"MAGNITUDE TRANSFERS; THE SIGN IS MOSTLY A SHARED RESPONSE AXIS. K562 edges move their targets "
             f"{mt['ratio']:.2f}x more in RPE1 than matched pairs (p {mt['p']:.3g}), and direction agrees "
             f"{100*sg['agree']:.1f}% -- but swapping the source for another perturbation of matched "
             f"strength still gives {100*sg['swapped_source']:.1f}%, leaving an edge-specific increment of "
             f"only {100*inc:+.1f} points. Most of the apparent sign agreement is a property of the target "
             f"gene's preferred direction under any stress, not of the edge.")
    elif magok:
        v = (f"MAGNITUDE TRANSFERS, SIGN DOES NOT. K562 edges move their targets {mt['ratio']:.2f}x more in "
             f"RPE1 than matched pairs (p {mt['p']:.3g}), but direction agrees {100*sg['agree']:.1f}% "
             f"against {100*sg['swapped_source']:.1f}% for a swapped source -- an increment of "
             f"{100*inc:+.1f} points. These edges say a gene will move, not which way, which is the same "
             f"limitation the curated layer has.")
    else:
        v = (f"NO TRANSFER TO RPE1. K562 edges show a {mt['ratio']:.2f}x magnitude ratio (p {mt['p']:.3g}) "
             f"and a sign increment of {100*inc:+.1f} points over a swapped-source control. These edges are "
             f"K562-specific as measured, which is a real finding about context but means the layer cannot "
             f"be used as a general regulatory prior.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "causal_reg.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB, {len(edges):,} edges)")
    report(f"  -> {OUT/'causal_reg.json'}")


if __name__ == "__main__":
    main()
