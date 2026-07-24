"""denoise_test -- would DENOISING the target (what within-line replicates buy) raise the ceiling? A training-free test with the data we have.

error_localization showed the misses are the rare PRIVATE movers, and that those are REAL (6.2x cross-line reproducible). This asks the
follow-on that decides whether replicates are worth chasing: if we score only against the CONFIRMED-REAL (cross-line-reproducible) part of
each K562 response -- the part replicates would keep and sharpen -- does the achievable recall jump? And are the model's MISSES concentrated
in the non-reproducing (noise / cell-specific) part it can never be expected to get right?

For each held-out K562 KO also measured in >=1 other line:
  full   = all K562 specific movers (tide-removed).
  repro  = the subset that ALSO moves (|z|>=1) in >=1 other line's same KO  (denoised / confirmed-real -- what replicates would keep).
  noise  = full \\ repro  (non-reproducing: measurement noise OR genuinely cell-specific -- indistinguishable without WITHIN-line replicates).
Report model & oracle recall@50 against full vs repro, and where the model's hits and misses fall. A big repro-vs-full recall jump = the
ceiling is set by target NOISE, so replicates would raise it. Also a consensus-target oracle (multi-line mean) as a second denoising angle.
Deterministic; no training."""
import json
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy

OUT = Path("outputs/orphan")
OTHER = ["RPE1", "HepG2", "Jurkat"]
K = 50


def recall_vs(vec, truth, nontide_idx, k=K):
    if not truth:
        return float("nan")
    top = nontide_idx[np.argsort(-vec[nontide_idx])][:k]
    return len(set(top.tolist()) & truth) / len(truth)


def main():
    Hk = Harness("K562")
    have, X, Y = build_xy(Hk)
    hidx = {k: i for i, k in enumerate(have)}
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    HL = {L: Harness(L) for L in OTHER}

    rng = np.random.RandomState(0)
    scor = [k for k in Hk.scorable if k in hidx]
    perm = rng.permutation(len(scor)); nt = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:nt]); test_set = set(test)
    tr = np.array([hidx[k] for k in have if k not in test_set])

    def dep_pred(ko):
        i = hidx[ko]; s = Xn[tr] @ Xn[i]; return Y[tr[np.argsort(-s)[:10]]].mean(0)

    rec = {"model_full": [], "model_repro": [], "oracle_full": [], "oracle_repro": []}
    frac_repro, hits_repro, hits_noise, miss_repro, miss_noise = [], [], [], [], []
    cons_oracle, cons_single = [], []
    n_paired = 0
    for ko in test:
        rk = Hk.ki[ko]
        paired = [L for L in OTHER if ko in HL[L].ki]
        if not paired:
            continue
        n_paired += 1
        full = [int(g) for g in np.where(Hk.mover[rk])[0] if Hk.nontide[g]]
        repro = set()
        for g in full:
            nm = Hk.genes[g]
            for L in paired:
                jl = HL[L].gi.get(nm)
                if jl is not None and HL[L].A[HL[L].ki[ko], jl] >= 1.0:
                    repro.add(g); break
        noise = set(full) - repro
        full_s = set(full)
        if not full_s:
            continue
        frac_repro.append(len(repro) / len(full_s))

        mv = dep_pred(ko); ov = Hk._references(ko, K)["ORACLE*"]
        rec["model_full"].append(recall_vs(mv, full_s, Hk.nontide_idx))
        rec["model_repro"].append(recall_vs(mv, repro, Hk.nontide_idx) if repro else np.nan)
        rec["oracle_full"].append(recall_vs(ov, full_s, Hk.nontide_idx))
        rec["oracle_repro"].append(recall_vs(ov, repro, Hk.nontide_idx) if repro else np.nan)
        # where do the model's top-50 hits / misses fall?
        top = set(Hk.nontide_idx[np.argsort(-mv[Hk.nontide_idx])][:K].tolist())
        hit = full_s & top; miss = full_s - top
        if hit:
            hits_repro.append(len(hit & repro) / len(hit)); hits_noise.append(len(hit & noise) / len(hit))
        if miss:
            miss_repro.append(len(miss & repro) / len(miss)); miss_noise.append(len(miss & noise) / len(miss))

        # consensus-target denoising angle: mean |z| across K562 + paired lines (shared genes), recall its own movers via oracle
        cons = Hk.A[rk].copy(); cnt = np.ones(Hk.nG)
        for L in paired:
            al = HL[L].A[HL[L].ki[ko]]
            for nm_i, nm in enumerate(Hk.genes):
                jl = HL[L].gi.get(nm)
                if jl is not None:
                    cons[nm_i] += al[jl]; cnt[nm_i] += 1
        cons /= cnt
        cons_mv = set(int(g) for g in np.where((cons >= 1.0) & Hk.nontide)[0])
        if cons_mv:
            cons_oracle.append(recall_vs(ov, cons_mv, Hk.nontide_idx))
            cons_single.append(recall_vs(ov, full_s, Hk.nontide_idx))

    m = lambda a: float(np.nanmean(a)) if len(a) else float("nan")
    R = {k: m(v) for k, v in rec.items()}
    fr = m(frac_repro)
    print(f"paired held-out K562 KOs (measured in >=1 other line): {n_paired}")
    print(f"fraction of each response that is cross-line CONFIRMED-REAL (reproducible): {fr:.0%}  (rest is noise or cell-specific)\n")
    print(f"{'target':28s} {'MODEL r@50':>10s} {'ORACLE r@50':>11s}")
    print(f"{'full (all specific movers)':28s} {R['model_full']:10.2f} {R['oracle_full']:11.2f}")
    print(f"{'repro (denoised, real only)':28s} {R['model_repro']:10.2f} {R['oracle_repro']:11.2f}")
    print(f"\nmodel top-50 HITS: {m(hits_repro):.0%} are confirmed-real, {m(hits_noise):.0%} non-reproducing")
    print(f"model MISSES:      {m(miss_repro):.0%} are confirmed-real, {m(miss_noise):.0%} non-reproducing")
    print(f"consensus-target oracle {m(cons_oracle):.2f} vs single-line oracle {m(cons_single):.2f} (same KOs)")

    model_gain = R["model_repro"] - R["model_full"]; oracle_gain = R["oracle_repro"] - R["oracle_full"]
    miss_is_noise = m(miss_noise) > 0.5
    helps = model_gain > 0.05 or miss_is_noise
    verdict = (
        f"DENOISE TEST (denoise_test.py): would denoising the target (what within-line replicates buy) raise the ceiling? Scored against only "
        f"the cross-line CONFIRMED-REAL part of each K562 response. {n_paired} paired held-out KOs; {fr:.0%} of a typical response is "
        f"confirmed-real, the rest non-reproducing (noise or cell-specific). "
        f"Restricting to the real subset: model recall {R['model_full']:.2f}->{R['model_repro']:.2f} ({model_gain:+.2f}), oracle "
        f"{R['oracle_full']:.2f}->{R['oracle_repro']:.2f} ({oracle_gain:+.2f}). The model's MISSES are {m(miss_noise):.0%} non-reproducing "
        f"and its HITS are {m(hits_repro):.0%} confirmed-real. "
        + ("So the ceiling is set largely by target NOISE: the model already finds the real movers and mostly misses the non-reproducing ones, "
           "so cleaner targets (WITHIN-line replicates) would raise the achievable recall and let the metric stop penalising unconfirmable "
           "calls. Replicates are the worthwhile data lever." if helps else
           "Restricting to the reproducible subset does NOT lift recall much and the misses are not mostly noise -- so the missed signal is "
           "real-but-cell-specific rather than noise, and replicates would sharpen but not rescue it; the residual is genuinely idiosyncratic.")
        + f" (Raw counts are not in-repo, so a proper multi-guide re-aggregation would require fetching the raw Perturb-seq.) Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_paired": n_paired, "frac_confirmed_real": round(fr, 3),
               "recall": {k: round(v, 4) for k, v in R.items()},
               "model_hits_repro": round(m(hits_repro), 3), "model_miss_noise": round(m(miss_noise), 3),
               "consensus_oracle": round(m(cons_oracle), 4), "single_oracle": round(m(cons_single), 4),
               "verdict": verdict, "note": verdict}, open(OUT / "denoise_test.json", "w"), indent=1)
    print("\n  -> outputs/orphan/denoise_test.json")


if __name__ == "__main__":
    main()
