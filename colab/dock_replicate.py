"""CONFIRMATORY TEST OF ONE PRE-SPECIFIED ARM: does elec_frac_fav survive at proper sampling?

WHAT THIS IS, AND WHY IT IS A DIFFERENT KIND OF TEST FROM THE ONE BEFORE IT.

`dock_distogram` was EXPLORATORY. It computed 19 arms, took the maximum, and found `e_elec_frac_fav` separating
near-native from close-wrong at AUC 0.811 while its Spearman read -0.047. A permutation test put that at
p = 0.020 against a best-of-19 null whose median was 0.675 -- meaning most of that table (bsa_proxy 0.750,
shape 0.667, desolv_median_nz 0.673) sat at or below what chance selection produces and were not findings.

This is CONFIRMATORY. Exactly ONE arm is pre-specified before the run:

    e_elec_frac_fav   the FRACTION of interface spheres carrying favourable (negative) electrostatic energy

Nothing else may be promoted to the headline. Other arms are computed for context and are explicitly labelled
exploratory; if one of them beats the pre-specified arm that is a hypothesis for another day, not a result. A
single pre-registered hypothesis needs no multiple-comparison correction, which is the whole point of running
this separately rather than re-reading the first table.

WHAT CHANGES, and it is the reason replication is worth the compute:

    sampling      2,400 rotations, not 600. The coverage sweep showed 600 leaves the correct pose ABSENT from
                  half the test set; at 2,400 all 12 complexes contain one and basins hold 4-16 near-native
                  members instead of 2-4.
    complexes     12 usable instead of 6 -- the positive class doubles in count and doubles in breadth.
    power         the first test had 2-4 positives per complex, which is why its null median was 0.675.

WHY THIS IS CHEAP DESPITE 4x THE POSES.  AUC_close compares near-native against close-wrong ONLY. RMSD is
cheap for all ~9,600 poses; the 46 ms/pose energy cost is spent only on poses that actually enter a comparison
-- every near-native, a capped sample of close-wrong, and a capped random sample of far. Selecting poses BY
RMSD is legitimate here: AUC is a supervised metric that always uses labels, and the selection touches which
poses are compared, never the score any pose receives.

PREDECLARED, before any number:
    AUC_close >= 0.70 and permutation p < 0.05      -> CONFIRMED. The arm is a near-native detector at proper
                                                       sampling, on twice the complexes, as a single
                                                       pre-registered hypothesis. It belongs in the pipeline.
    AUC_close in 0.55-0.70, p >= 0.05               -> WEAKENED. Real direction, not strong enough to act on.
    AUC_close ~ 0.5                                 -> the 0.811 was selection noise from taking the best of
                                                       19 arms on 6 underpowered complexes, and the honest
                                                       record is that the exploratory finding did not replicate.

The permutation here shuffles RMSD labels within each complex for the SINGLE pre-specified arm -- no max over
arms, because there is no max to take.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_energy as DE
import dock_fft as DF
import flex_physics as fp
import nexus_pairs as NP

OUT = A.OUT
N_ROT = 2400                 # the coverage-validated budget, not the 600 every earlier run used
PEAKS = 4
NEAR, FAR = 5.0, 10.0
CAP_CLOSE, CAP_FAR = 400, 400
PRESPEC = "elec_frac_fav"    # THE hypothesis. Fixed here, before any number is produced.
CONTEXT = ("vdw_frac_fav", "desolv_frac_fav", "induc_frac_fav", "elec_sum", "vdw_median_nz")
N_PERM = 2000


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score

    report("=" * 100)
    report("CONFIRMATORY: does elec_frac_fav survive at proper sampling, as ONE pre-specified arm?")
    report("=" * 100)
    report(f"  PRE-SPECIFIED, fixed before the run: {PRESPEC}. Everything else is context and cannot be")
    report("  promoted to the headline. One pre-registered hypothesis needs no multiple-comparison correction.")
    report(f"  Sampling {N_ROT} rotations (coverage-validated) vs the 600 that left the answer absent from half")
    report("  the test set. PREDECLARED: AUC_close >= 0.70 with p < 0.05 => CONFIRMED. 0.55-0.70 => weakened.")
    report("  ~0.5 => the 0.811 was best-of-19 selection noise and did not replicate.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    rots = DF.rotations(N_ROT)
    rng = np.random.default_rng(0)

    rows, store = [], {}
    for pdb in test_pdbs:
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        t = fp.table(pdb)
        ca, cb = p["ca"], p["cb"]
        rn_map = NP.resnames(pdb)
        bmask = t["ch"] == cb
        scm = (t["ch"] == ca) & ~np.isin(t["nm"], list(fp.BB) + ["CB"])
        res_pool = [(int(x), np.where(scm & (t["rn"] == x))[0])
                    for x in sorted(set(t["rn"][scm].tolist()))]
        res_pool = [(a, b) for a, b in res_pool if len(b)]
        if len(res_pool) < 5:
            continue
        centre, L, lcen = p["centre"], p["L"], p["lcen"]
        Lc = L - lcen
        FA = p["F_skin"] - CW * p["F_core"]
        L0 = t["co"][bmask] - lcen

        # ---- stage 1: RMSD for every pose (cheap, no energy) ----
        t1 = time.time()
        cand = []
        for ri, Rm in enumerate(rots):
            g = DF.rasterise(Lc @ Rm.T + centre, centre, p["offs_L"])
            if g is None:
                continue
            sc = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), p["shape"])
            for pk in DF.nms_peaks(sc, PEAKS, DF.NMS_CELLS):
                if sc[pk] <= 0:
                    continue
                sh = DF.shift_of(pk)
                rm = float(np.sqrt((((Lc @ Rm.T + centre + sh) - L) ** 2).sum(1).mean()))
                cand.append((ri, sh, rm))
        if not cand:
            continue
        R_all = np.array([c[2] for c in cand])
        near = np.where(R_all <= NEAR)[0]
        close = np.where((R_all > NEAR) & (R_all <= FAR))[0]
        far = np.where(R_all > FAR)[0]
        if len(near) == 0 or len(close) == 0:
            rows.append({"pdb": pdb, "n_poses": int(len(R_all)), "n_near": int(len(near)),
                         "n_close": int(len(close)), "skipped": "no positive or no close class"})
            report(f"    {pdb}: {len(R_all)} poses, near {len(near)}, close {len(close)} -- SKIPPED")
            continue
        sel = np.concatenate([near,
                              rng.choice(close, min(CAP_CLOSE, len(close)), replace=False),
                              rng.choice(far, min(CAP_FAR, len(far)), replace=False) if len(far) else
                              np.array([], int)])

        # ---- stage 2: energies ONLY on poses that enter a comparison ----
        vals, rsel = {k: [] for k in (PRESPEC,) + CONTEXT}, []
        for si in sel:
            ri, sh, rm = cand[si]
            co = t["co"].copy()
            co[bmask] = L0 @ rots[ri].T + centre + sh
            e = DE.sphere_energies(t, co, ca, cb, res_pool, rn_map, cKDTree(co[bmask]), soft=True)
            if e is None:
                continue
            summ = {term: DE.summarise(v) for term, v in e.items()}
            for k in vals:
                term, stat = k.rsplit("_", 1) if k.endswith(("sum",)) else (k.split("_")[0], k.split("_", 1)[1])
                vals[k].append(summ[term][stat])
            rsel.append(rm)
        rsel = np.array(rsel)
        if (rsel <= NEAR).sum() == 0 or ((rsel > NEAR) & (rsel <= FAR)).sum() == 0:
            continue
        store[pdb] = ({k: np.array(v) for k, v in vals.items()}, rsel)
        rows.append({"pdb": pdb, "n_poses": int(len(R_all)), "n_near": int(len(near)),
                     "n_close": int(len(close)), "n_scored": int(len(rsel)), "secs": time.time() - t1})
        report(f"    {pdb}: {len(R_all)} poses -> {len(rsel)} scored in {time.time()-t1:.0f}s | "
               f"near {int((rsel<=NEAR).sum())}  close {int(((rsel>NEAR)&(rsel<=FAR)).sum())}")

    if len(store) < 5:
        report(f"\n  only {len(store)} usable complexes -- not enough to confirm")
        return 1

    def auc_close(v, r, lab=None):
        lab = r if lab is None else lab
        near = lab <= NEAR
        close = (lab > NEAR) & (lab <= FAR)
        m = near | close
        if near.sum() == 0 or close.sum() == 0 or np.std(v[m]) < 1e-12:
            return None
        return roc_auc_score(near[m], v[m])

    def auc_far(v, r):
        near = r <= NEAR
        far = r > FAR
        m = near | far
        if near.sum() == 0 or far.sum() == 0 or np.std(v[m]) < 1e-12:
            return None
        return roc_auc_score(near[m], v[m])

    report(f"\n  {len(store)} complexes with both classes | positives per complex: "
           f"{[int((r<=NEAR).sum()) for _, r in store.values()]}")

    ac = [auc_close(v[PRESPEC], r) for v, r in store.values()]
    ac = [x for x in ac if x is not None]
    af = [auc_far(v[PRESPEC], r) for v, r in store.values()]
    af = [x for x in af if x is not None]
    obs = float(np.median(ac))
    report(f"\n  PRE-SPECIFIED ARM: {PRESPEC}")
    report(f"    AUC_close median {obs:.3f}   per complex {[round(x,3) for x in sorted(ac)]}")
    report(f"    AUC_far   median {np.median(af):.3f}")
    report(f"    correct-signed (>0.5) in {sum(1 for x in ac if x > 0.5)}/{len(ac)} complexes")

    null = []
    for _ in range(N_PERM):
        vs = []
        for v, r in store.values():
            x = auc_close(v[PRESPEC], r, lab=rng.permutation(r))
            if x is not None:
                vs.append(x)
        if vs:
            null.append(float(np.median(vs)))
    null = np.array(null)
    pval = float((null >= obs).mean())
    report(f"    PERMUTATION ({N_PERM} shuffles, single arm, no max): null median {np.median(null):.3f}, "
           f"95th {np.percentile(null,95):.3f}")
    report(f"    p = {pval:.4f}")

    report(f"\n  CONTEXT ARMS (exploratory, cannot be promoted -- no correction applied to these)")
    ctx = {}
    for k in CONTEXT:
        a = [auc_close(v[k], r) for v, r in store.values()]
        a = [x for x in a if x is not None]
        if a:
            ctx[k] = float(np.median(a))
            report(f"    {k:<20}{ctx[k]:>8.3f}")

    report("\n  READING")
    if obs >= 0.70 and pval < 0.05:
        report(f"  CONFIRMED. {PRESPEC} reaches AUC_close {obs:.3f} (p={pval:.4f}) on {len(ac)} complexes at")
        report(f"  {N_ROT} rotations, as a SINGLE pre-registered hypothesis needing no correction. The")
        report("  exploratory 0.811 was not selection noise. Electrostatic favourability across the patch")
        report("  distinguishes a correct pose from a nearly-right one, which shape complementarity cannot.")
    elif obs >= 0.55:
        report(f"  WEAKENED. AUC_close {obs:.3f} (p={pval:.4f}) -- the direction survives but the effect is")
        report(f"  smaller than the exploratory 0.811 and does not clear the predeclared bar. Real, not usable.")
    else:
        report(f"  DID NOT REPLICATE. AUC_close {obs:.3f} (p={pval:.4f}) against an exploratory 0.811. That")
        report("  0.811 was best-of-19 selection noise on 6 underpowered complexes, and the honest record is")
        report("  that the distogram finding did not survive proper sampling.")

    json.dump({"test": "dock_replicate", "prespecified": PRESPEC, "n_rot": N_ROT, "near_A": NEAR, "far_A": FAR,
               "auc_close_median": obs, "auc_close_per_complex": ac, "auc_far_median": float(np.median(af)),
               "p_value": pval, "null_median": float(np.median(null)),
               "null_p95": float(np.percentile(null, 95)), "n_complexes": len(ac),
               "context_arms_exploratory": ctx, "complexes": rows, "log": log},
              open(OUT / "dock_replicate.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_replicate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
