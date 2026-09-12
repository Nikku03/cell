"""IS QUANTITATIVE Hi-C WORTH ANYTHING ON TOP OF CTCF? The fair test, finally runnable.

WHAT THIS CORRECTS. `contact_decompose.py` concluded that CTCF counting is the whole of "contact" and that
measured Hi-C adds -0.0026/-0.0076 on top of it. That was measured with `loop_connects`, a BINARY called-loop
indicator -- and a loop caller already folds distance, CTCF and coverage into its own thresholding before
emitting a call. A binary loop call is close to designed to be redundant with CTCF features, so that test was
never a fair one, and the recorded conclusion was corrected to the narrower claim it supports.

This is the test the conclusion actually needs, and it could not be run before: no contact matrix had ever
been on disk, only loop calls and a TAD bedpe. K562 ENCFF026LJG (in situ Hi-C, hg38, 311 MB) is now
converted to a 5 kb cooler, so contact frequency can be read for every K562 pair directly.

WHY A RESIDUAL AND NOT RAW CONTACT. Raw contact is dominated by distance: two elements 10 kb apart contact
constantly, two 500 kb apart rarely, and the model already has log-distance. Feeding raw contact would mostly
re-supply distance and any gain would be uninterpretable. The quantity that carries new information is

    C_resid = log(observed) - E[log(observed) | distance, CTCF, coverage]

i.e. does this pair touch MORE than pairs like it. Expected is fitted from the data itself, per distance bin,
so no external expected-model is assumed. Three further contact features are built alongside, all of which
are invisible to a binary loop call:

    obs_over_exp       observed / distance-matched expected
    contact_rank_gene  rank of this candidate's contact among the same gene's candidates
    contact_rank_elem  rank of this promoter among the ones this element contacts

K562 ONLY, AND THAT IS STATED. The matrix is K562, so only the 12,274 K562 pairs (717 positives) can be
scored. That is 87% of the benchmark's positives, so it is not a small corner -- but it means this cannot be
a LOCO test, and the result is a within-K562 statement.

THE COMPARISON THAT MATTERS is M4 -> M4 + residual contact, on RANKING as well as pooled AUPRC, because the
whole point of `ranking_gate.py` was that pooled AUPRC answers a question nobody asks. A shuffled-contact arm
is included: four new columns will move a metric slightly whether or not they carry signal.
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
RES = int(os.environ.get("RC_RES", 25000))
COOL = SP / "hic" / f"k562_{RES//1000}kb.cool"
SEEDS = (0, 1, 2, 3, 4)
CELL = "K562"


def contact_for(rows, idx):
    """Observed contact at (element bin, TSS bin) for each row, read from the cooler.

    Read per chromosome as a coordinate-sorted pixel query rather than one fetch per pair: 12k random
    fetches against an HDF5-backed matrix is minutes of seek time, one contiguous read per chromosome is
    seconds.
    """
    import cooler
    c = cooler.Cooler(str(COOL))
    chroms = set(c.chromnames)
    # This .hic names chromosomes 1,2,3... while the benchmark uses chr1,chr2,... A silent mismatch reads
    # zero contacts for every pair, which is exactly what happened on the first run and is why the read
    # fraction is asserted rather than assumed.
    def cname(ch):
        if ch in chroms:
            return ch
        alt = ch[3:] if ch.startswith("chr") else "chr" + ch
        return alt if alt in chroms else None
    obs = np.full(len(idx), np.nan)
    by = {}
    for k, i in enumerate(idx):
        r = rows[i]
        ch = cname(r["chrom"])
        if ch is None:
            continue
        b1 = (int(r["chromStart"]) + int(r["chromEnd"])) // 2 // RES
        b2 = int(float(r["startTSS"])) // RES
        by.setdefault(ch, []).append((k, min(b1, b2), max(b1, b2)))
    for ch, items in by.items():
        lo = min(min(a, b) for _, a, b in items)
        hi = max(max(a, b) for _, a, b in items)
        try:
            mat = c.matrix(balance=False, sparse=True).fetch(
                f"{ch}:{lo*RES}-{min((hi+1)*RES, c.chromsizes[ch])}").tocsr()
        except Exception:
            continue
        for k, a, b in items:
            ia, ib = a - lo, b - lo
            if 0 <= ia < mat.shape[0] and 0 <= ib < mat.shape[1]:
                obs[k] = mat[ia, ib]
    return obs


def main():
    from scipy import stats
    from sklearn.metrics import average_precision_score
    from ranking_gate import (load, build, folds_chrom, fit_oof, groups_of, rank_metrics, summarise)
    print("=" * 100)
    print("RESIDUAL CONTINUOUS Hi-C -- the fair test of quantitative contact on top of CTCF")
    print("=" * 100)
    if not COOL.exists():
        raise SystemExit(f"cooler absent at {COOL}")
    rows = load()
    d = build(rows)
    ct = d["ct"]
    idx = np.where(ct == CELL)[0]
    print(f"  {len(idx):,} {CELL} pairs of {len(rows):,} total, "
          f"{int(d['y'][idx].sum())} positives ({100*d['y'][idx].sum()/d['y'].sum():.0f}% of all positives)")

    obs = contact_for(rows, idx)
    got = np.isfinite(obs)
    print(f"  contact read for {int(got.sum()):,}/{len(idx):,} pairs ({100*got.mean():.1f}%)")
    if got.mean() < 0.8:
        raise SystemExit("contact missing for >20% of pairs -- the bin lookup is wrong somewhere and no "
                         "verdict from this run should be believed")
    obs = np.where(got, obs, 0.0)
    # ZERO-INFLATION IS THE LIMIT ON THIS FEATURE, so it is measured and printed rather than discovered
    # later. At 5 kb, 65.1% of K562 pairs had zero contacts and the zero rate ran from 0% under 10 kb to
    # 89% beyond 1 Mb -- so the "residual" was largely presence/absence of any read, which is depth times
    # distance, and cannot be detrended out of a mostly-zero variable.
    zf = float((obs == 0).mean())
    print(f"  zero-contact fraction at {RES//1000} kb: {zf:.3f}")
    if zf > 0.4:
        print(f"    ** {100*zf:.0f}% of pairs have no contact at this resolution; the residual is then "
              f"mostly an indicator of whether any read was seen, and a coarser bin is the right fix")
    lo = np.log1p(obs)
    dist = d["dist"][idx]
    ld = np.log10(dist)

    # expected from the data itself, per distance decile, so no external expected-model is assumed
    # 100 quantile bins, and a within-bin linear detrend on top. 20 bins left the residual correlated with
    # distance at Spearman +0.181, which means it was still partly re-supplying the one feature the model
    # already has -- and any gain would then be uninterpretable.
    NB = 100
    bins = np.quantile(ld, np.linspace(0, 1, NB + 1))
    bi = np.clip(np.digitize(ld, bins[1:-1]), 0, NB - 1)
    exp = np.zeros_like(lo)
    for b in range(NB):
        m = bi == b
        if m.sum() >= 8 and ld[m].std() > 0:
            k, c0 = np.polyfit(ld[m], lo[m], 1)
            exp[m] = k * ld[m] + c0
        elif m.any():
            exp[m] = lo[m].mean()
    resid = lo - exp
    ooe = np.where(exp > 0, lo / np.maximum(exp, 1e-9), 0.0)
    print(f"  log1p(contact): median {np.median(lo):.3f}; residual sd {resid.std():.3f}")
    rho_res = float(stats.spearmanr(resid, ld)[0])
    print(f"  Spearman(raw contact, distance) {stats.spearmanr(lo, ld)[0]:+.3f}   "
          f"Spearman(residual, distance) {rho_res:+.3f}")
    if abs(rho_res) > 0.08:
        print(f"    ** residual still tracks distance at {rho_res:+.3f}; part of any gain below may be "
              f"distance re-entering the model, and the number should be read with that in mind")

    key = [d["key"][i] for i in idx]
    grank = np.zeros(len(idx))
    g = {}
    for j, k in enumerate(key):
        g.setdefault(k, []).append(j)
    for k, v in g.items():
        v = np.array(v)
        grank[v] = np.argsort(np.argsort(-resid[v])) + 1
    C = np.column_stack([resid, ooe, lo, grank])
    cnames = ["contact_residual", "obs_over_exp", "log_contact", "contact_rank_in_gene"]

    y = d["y"][idx]
    ch = d["ch"][idx]
    BASE = np.hstack([d["D"][idx], d["A"][idx], d["C"][idx], d["P"][idx]])
    groups = groups_of(key, y)
    print(f"  {len(groups)} evaluable {CELL} gene groups\n")

    rng = np.random.default_rng(0)
    ARMS = {"M4 baseline (K562)": BASE,
            "+residual contact": np.hstack([BASE, C[:, :1]]),
            "+all 4 contact feats": np.hstack([BASE, C]),
            "CTRL +shuffled contact": np.hstack([BASE, C[rng.permutation(len(idx))]])}
    R = {"cell": CELL, "n_pairs": int(len(idx)), "n_pos": int(y.sum()),
         "contact_read_frac": float(got.mean()), "contact_features": cnames,
         "n_groups": len(groups), "arms": {}}
    print(f"    {'arm':26s} {'pooled':>7s} {'R@1':>6s} {'R@3':>6s} {'MRR':>6s} {'dMRR p':>8s}")
    base_per = None
    for aname, M in ARMS.items():
        ap, pers = [], []
        for s in SEEDS:
            f = folds_chrom(ch, s)
            p = fit_oof(M, y, f)
            ap.append(average_precision_score(y, p))
            pers.append(rank_metrics(p, y, groups))
        per = {k: {m: float(np.mean([pp[k][m] for pp in pers]))
                   for m in ("r1", "r3", "mrr", "rank", "nrank", "ap")} for k in groups}
        sm = summarise(per)
        if base_per is None:
            base_per, pv = per, np.nan
        else:
            dm = np.array([per[k]["mrr"] - base_per[k]["mrr"] for k in groups])
            nz = dm[dm != 0]
            pv = (stats.binomtest(int((nz > 0).sum()), len(nz), 0.5,
                                  alternative="greater").pvalue if len(nz) else np.nan)
        print(f"    {aname:26s} {np.mean(ap):7.4f} {sm['r1']:6.3f} {sm['r3']:6.3f} {sm['mrr']:6.3f} "
              f"{pv if np.isfinite(pv) else float('nan'):8.4f}")
        R["arms"][aname] = {"pooled_auprc": float(np.mean(ap)), **sm, "dmrr_sign_p": pv}

    b = R["arms"]["M4 baseline (K562)"]
    r1 = R["arms"]["+residual contact"]
    r4 = R["arms"]["+all 4 contact feats"]
    sh = R["arms"]["CTRL +shuffled contact"]
    print("\n  RAW DELTAS vs the K562 baseline (effect size; NOT delta-minus-shuffled):")
    for nm, a in (("+residual", r1), ("+all 4", r4), ("+shuffled", sh)):
        print(f"    {nm:12s} pooled {a['pooled_auprc']-b['pooled_auprc']:+.4f}   "
              f"R@1 {a['r1']-b['r1']:+.4f}   MRR {a['mrr']-b['mrr']:+.4f}")
    # read the RESIDUAL arm: it is the construction this module exists to test, and the 4-feature arm mixes
    # it with raw log-contact, which is mostly distance. Ranking is weighted over pooled AUPRC because
    # ranking_gate.py established pooled AUPRC answers a question nobody asks.
    dp = r1["pooled_auprc"] - b["pooled_auprc"]
    dm = r1["mrr"] - b["mrr"]
    dr = r1["r1"] - b["r1"]
    shm = sh["mrr"] - b["mrr"]
    pv = r1["dmrr_sign_p"]
    R["zero_fraction"] = zf
    R["resolution_bp"] = RES
    R["residual_arm"] = {"d_pooled": dp, "d_r1": dr, "d_mrr": dm, "sign_p": pv,
                         "shuffled_d_mrr": shm, "resid_vs_distance_spearman": rho_res}
    margin_mrr, margin_r1 = dm - shm, dr - (sh["r1"] - b["r1"])
    R["residual_arm"]["margin_over_shuffled_mrr"] = margin_mrr
    R["residual_arm"]["margin_over_shuffled_r1"] = margin_r1
    print(f"    margin over shuffled: R@1 {margin_r1:+.4f}   MRR {margin_mrr:+.4f}")
    if dm > 0.005 and pv < 0.05 and margin_mrr > 0.005 and zf < 0.4:
        v = (f"QUANTITATIVE CONTACT HELPS THE RANKING, AND ONLY THE RANKING. Residual contact adds "
             f"{dp:+.4f} pooled AUPRC -- nothing -- but {dr:+.4f} R@1 and {dm:+.4f} MRR with a paired "
             f"per-group sign test at p {pv:.4g}, against a shuffled-contact control of {shm:+.4f} MRR. "
             f"So the earlier 'Hi-C adds nothing' was an artefact of the BINARY called-loop feature, which "
             f"a loop caller had already collapsed against distance and CTCF. It was also invisible to "
             f"pooled AUPRC, which is the case for the ranking metric in one number. "
             f"Caveat: the residual still tracks distance at Spearman {rho_res:+.3f}, {100*zf:.0f}% of "
             f"pairs have zero contact, and this is K562 only (87% of positives, not a cross-cell-type "
             f"claim).")
    elif dm > 0.005 and pv < 0.05:
        v = (f"SUGGESTIVE, NOT ESTABLISHED. Residual contact adds {dr:+.4f} R@1 and {dm:+.4f} MRR "
             f"(sign test p {pv:.3g}) -- but the SHUFFLED contact control adds {sh['r1']-b['r1']:+.4f} R@1 "
             f"and {shm:+.4f} MRR, leaving a margin of only {margin_r1:+.4f}/{margin_mrr:+.4f}. With "
             f"{100*zf:.0f}% of pairs at zero contact and the residual still tracking distance at "
             f"{rho_res:+.3f}, that margin is not enough to call it. What it DOES establish is that the "
             f"binary called-loop test was the wrong instrument, and that any real effect is invisible to "
             f"pooled AUPRC ({dp:+.4f}) while showing up only in ranking.")
    elif dp > 0.01 and dm > 0.005:
        v = (f"QUANTITATIVE CONTACT ADDS SOMETHING -- pooled {dp:+.4f}, MRR {dm:+.4f} on top of a model that "
             f"already has CTCF. The binary called-loop test that concluded otherwise was measuring a "
             f"feature a loop caller had already collapsed.")
    elif dp > 0.01:
        v = (f"POOLED ONLY -- contact adds {dp:+.4f} pooled but {dm:+.4f} MRR, so it does not improve the "
             f"within-gene decision. Same pattern as the differential block.")
    else:
        v = (f"NO -- quantitative contact adds {dp:+.4f} pooled and {dm:+.4f} MRR on top of CTCF, on "
             f"{len(idx):,} K562 pairs with contact read for {100*got.mean():.0f}% of them. This is the "
             f"FAIR version of the test the binary loop feature failed, and it agrees: buy CTCF ChIP, not "
             f"Hi-C. K562 only, so not a cross-cell-type claim.")
    R["verdict"] = v
    print(f"\n  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "residual_contact.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'residual_contact.json'}")


if __name__ == "__main__":
    main()
