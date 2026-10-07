"""causal_patch — extend causal reach with the intervention data we ALREADY have on disk.

causal_reach showed structure (STRING) cannot compute a removal's effect. But we hold a SECOND, genome-wide
intervention dataset: DepMap CRISPR (18,443 genes knocked out across 1,150 contexts). Perturb-seq measured the
transcriptional effect of removing 2,058 pieces; DepMap measured the FITNESS effect of removing all 18,443. The honest
question this tests: does DepMap CO-ESSENTIALITY (which genes share a knockout-fitness profile across 1,150 lines)
predict the MEASURED Perturb-seq response (which genes actually move when you knock the piece down)? If yes, DepMap is a
genome-wide proxy for "remove X -> here's the effect", extending the debugger from 2,058 to ~18,443 pieces with data
already on disk. Compared head-to-head against the STRING-structure predictor (the causal_reach null) and chance.
-> outputs/orphan/causal_patch.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def load():
    import biosim
    b = biosim.BioSim()                                        # syms (measured cols), M (|response|), pidx, propagate
    z = np.load(f"{OUT}/depmap_vecs.npz", allow_pickle=True)
    dsyms = [str(s) for s in z["syms"]]
    Z = np.nan_to_num(z["Z"].astype("float32"), nan=0.0)
    nz = np.linalg.norm(Z, axis=1, keepdims=True); nz[nz == 0] = 1.0
    Zn = Z / nz                                                # unit rows -> cosine co-essentiality via dot
    didx = {s: i for i, s in enumerate(dsyms)}
    return b, Zn, didx, dsyms


def run(sample=400, seed=0):
    from scipy.stats import spearmanr
    b, Zn, didx, dsyms = load()
    rng = np.random.default_rng(seed)
    # measured columns that also exist in DepMap (so co-essentiality is defined for the response targets)
    col_d = np.array([didx.get(s, -1) for s in b.syms])        # DepMap index for each measured column (-1 if absent)
    keep = col_d >= 0
    meas_cols = np.where(keep)[0]                              # positions in b.syms
    dcols = col_d[keep]                                        # matching DepMap rows
    # knockouts present in BOTH Perturb-seq and DepMap
    kos = [g for g in b.pidx if g in didx]
    if len(kos) > sample:
        kos = list(np.array(kos, dtype=object)[rng.choice(len(kos), sample, replace=False)])
    dep, stru, rand = [], [], []
    for g in kos:
        meas = b.M[b.pidx[g]][meas_cols]                       # measured |response| on shared genes
        if np.count_nonzero(meas) < 20:
            continue
        # DepMap prediction: |co-essentiality of the KO gene with each shared response gene|
        coess = np.abs(Zn[dcols] @ Zn[didx[g]])
        c = spearmanr(coess, meas).statistic
        if np.isfinite(c):
            dep.append(c)
        # structure prediction: STRING propagation from g, restricted to shared genes (the causal_reach method)
        r = b.propagate(g)
        if r is not None:
            c2 = spearmanr(r[meas_cols], meas).statistic
            if np.isfinite(c2):
                stru.append(c2)
        # random baseline: co-essentiality of a RANDOM gene vs this KO's response
        rgi = didx[dsyms[rng.integers(len(dsyms))]]
        c3 = spearmanr(np.abs(Zn[dcols] @ Zn[rgi]), meas).statistic
        if np.isfinite(c3):
            rand.append(c3)

    def s(x):
        x = np.array(x)
        return {"mean": round(float(np.nanmean(x)), 3), "median": round(float(np.nanmedian(x)), 3), "n": len(x)}

    out = {"depmap_coessentiality": s(dep), "string_structure": s(stru), "random": s(rand),
           "n_measured_cols_shared_with_depmap": int(keep.sum()), "n_knockouts_tested": len(dep),
           "depmap_coverage_genes": len(dsyms), "perturbseq_coverage_genes": len(b.pidx)}
    return out


def main():
    print("=" * 92)
    print("CAUSAL PATCH — can on-disk DepMap intervention data predict the measured removal effect?")
    print("=" * 92)
    o = run()
    dep, stru, rnd = o["depmap_coessentiality"], o["string_structure"], o["random"]
    print(f"\n  For knockouts measured in BOTH Perturb-seq and DepMap (n={o['n_knockouts_tested']}), predict the MEASURED")
    print(f"  Perturb-seq response over {o['n_measured_cols_shared_with_depmap']} shared genes. Mean Spearman(predicted, measured):\n")
    print(f"    DepMap co-essentiality (on disk, genome-wide) : {dep['mean']:+.3f}   (median {dep['median']:+.3f})")
    print(f"    STRING structure  (the causal_reach null)     : {stru['mean']:+.3f}   (median {stru['median']:+.3f})")
    print(f"    random gene                                    : {rnd['mean']:+.3f}   (median {rnd['median']:+.3f})")
    lift_struct = dep["mean"] - stru["mean"]; lift_rand = dep["mean"] - rnd["mean"]
    works = dep["mean"] > 0.10 and lift_rand > 0.08 and dep["mean"] > stru["mean"] + 0.05
    if works:
        verdict = (
            f"YES — DepMap EXTENDS causal reach with data already on disk. Co-essentiality across 1,150 knockout "
            f"contexts predicts the MEASURED transcriptional removal-response at Spearman {dep['mean']:+.3f} — well "
            f"above structure ({stru['mean']:+.3f}, the null) and chance ({rnd['mean']:+.3f}). Because DepMap covers "
            f"{o['depmap_coverage_genes']:,} genes vs Perturb-seq's {o['perturbseq_coverage_genes']:,}, this turns "
            f"'remove X -> here's who is affected' from a 2,058-piece lookup into a ~{o['depmap_coverage_genes']//1000}k-"
            f"piece PREDICTION — the honest way to extend the causal engine without new wet-lab data. Wire it as a "
            f"genome-wide removal-effect channel.")
    else:
        verdict = (
            f"PARTIAL/NULL — DepMap co-essentiality predicts the measured response at {dep['mean']:+.3f} "
            f"(structure {stru['mean']:+.3f}, chance {rnd['mean']:+.3f}, lift over chance {lift_rand:+.3f}). "
            f"{'It beats structure and chance but is weak' if dep['mean'] > rnd['mean'] else 'It does not beat chance'} "
            f"— co-essentiality (shared FITNESS dependency) and transcriptional RESPONSE are related but not the same "
            f"axis. Honest read: it is a genome-wide FUNCTIONAL-coupling proxy, not a faithful response predictor; use "
            f"it to name the affected MODULE, not to compute the response vector.")
    print("\n" + "=" * 92 + f"\nVERDICT: {verdict}\n" + "=" * 92)
    o["verdict"] = verdict; o["extends_reach"] = bool(works)
    json.dump(o, open(f"{OUT}/causal_patch.json", "w"), indent=1)
    return o


if __name__ == "__main__":
    main()
