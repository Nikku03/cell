"""ENERGY LEVELS IN EACH SPHERE -- NEXUS's real physics on the FFT pose sets, which it has never seen.

THE HOLE THIS FILLS.  Everything tried so far for pose ranking was geometry or learned embeddings:

    dock_fft        grid shape complementarity           Spearman -0.059, 0/12 top-10
    dock_rescore    learned contact embeddings           +0.003, worse than random weights
    dock_poserank   retrained on 1.4M decoy contacts     +0.003
    dock_global     charge/phobic complementarity        -0.005 / +0.011, no gain

**The actual physics was never applied to these poses.** `flex_physics.sidechain_vdw` computes cross-interface
Lennard-Jones, electrostatics, induction and desolvation inside an 8 A sphere around each interface sidechain --
which is the sphere, and this measures the energy levels in it.

AND IT ANSWERS A QUESTION LEFT OPEN IN THE LEDGER.  `nexus_pairs` found vdW ranks the native interface first in
7 of 10 complexes against a 6% chance rate, and that entry recorded explicitly that whether the judge "survives
HARDER decoys than random rotations" was UNTESTED. FFT poses are those harder decoys: shape-optimised,
clash-filtered, 2,400 per complex with known RMSD. If vdW cannot rank them, the 70% was against easy negatives.

WHY A DISTRIBUTION AND NOT A SUM.  Lennard-Jones is r^-12. A single bad atom pair reached 5.6e11 in a probe,
which would swamp any total and turn "energy" into "did one atom clash". So each pose's spheres are summarised
several ways, and the robust ones are the point:

    sum          the classical total. Dominated by the worst contact, and stated as such.
    mean         same problem, scaled.
    median       the TYPICAL sphere. Immune to the r^-12 outlier -- this is the arm that tests whether the
                 interface is broadly favourable rather than whether one atom happens to clash.
    frac_fav     fraction of spheres with favourable (negative) energy. Also outlier-immune.
    worst        the single worst sphere. A pure clash detector, included to show what the sum is really doing.

Both soft-core and raw LJ are computed. Soft-core caps per-pair repulsion, so if raw and soft disagree the
difference IS the outlier effect, measured rather than assumed.

TERMS: vdw, elec, induc, desolv. Electrostatics only became live today -- sidechain_vdw looked `wt=` up in a
1-letter table while nexus_pairs passed 3-letter PDB resnames, so every charge was zero and the "z -0.04" in
that entry was an artifact of C-terminal OXT atoms leaking through. Fixed and verified on 34 charged residues.

SIGN CONVENTION FIXED BEFORE THE RUN: every arm is defined so HIGHER = better pose (energies negated, since
more negative energy is more favourable), therefore a working arm has NEGATIVE Spearman(score, RMSD).

PREDECLARED, before any number:
    an energy arm beats shape on Spearman AND does not lose on top-100 or enrichment
                        -> the physics ranks poses where geometry and learning could not, and NEXUS has a real
                           job in the docking pipeline.
    energy ~ shape      -> the physics adds nothing over grid overlap on these poses.
    energy fails while nexus_pairs' 70% stands
                        -> the judge does NOT survive hard decoys. The 70% was against easy negatives, and that
                           is a correction to a headline NEXUS result rather than a new null.

POWER: 2-4 near-native of 2,400 per complex, so top-k cannot resolve a modest effect over 6 testable complexes.
Spearman over all ~2,400 poses carries the weight. Stated before the numbers.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import dock_poserank as PRK
import flex_physics as fp
import nexus_pairs as NP

OUT = A.OUT
MAX_RES = 30                 # interface sidechains scored per pose
IFACE_CUT = 6.0              # MUST match _lj_sum's cutoff. At 8.0 (the first version) every sphere in the
                             # 6-8 A band scored EXACTLY zero, and the median went to 0.0 for the native pose
                             # and a 27 A decoy alike -- the "robust" arm was measuring nothing.
TOPK = (1, 10, 100)
NEAR_NATIVE = DF.NEAR_NATIVE
TERMS = ("vdw", "elec", "induc", "desolv")
STATS = ("sum", "mean", "median", "median_nz", "frac_fav", "worst")
# median_nz is over spheres that actually carry energy. Needed because elec/induc/desolv are zero for every
# UNCHARGED residue, so a plain median over all spheres is structurally 0 whatever the pose does.


def sphere_energies(t, co_now, ca, cb, res_pool, rn_map, tree_lig, soft):
    """energy levels inside each interface sphere. Returns per-term lists over spheres, or None."""
    t2 = dict(t)
    t2["co"] = co_now
    # which receptor residues have a sidechain within reach of the moved ligand
    hits = []
    for pos, idx in res_pool:
        if len(tree_lig.query_ball_point(co_now[idx], IFACE_CUT)):
            hits.append(pos)
        if len(hits) >= MAX_RES:
            break
    if len(hits) < 3:
        return None
    out = {k: [] for k in TERMS}
    for pos in hits:
        try:
            sc = fp.sidechain_vdw(t2, ca, pos, soft=soft, wt=rn_map.get((ca, int(pos))))
        except Exception:
            sc = None
        if not sc:
            continue
        for k in TERMS:
            out[k].append(float(sc.get(k, 0.0)))
    if len(out["vdw"]) < 3:
        return None
    return out


def summarise(vals):
    """energy LEVELS, not one number. Negated so higher = better, since favourable energy is negative."""
    a = np.asarray(vals, float)
    if not len(a):
        return None
    nz = a[np.abs(a) > 1e-12]
    return {"sum": -float(a.sum()), "mean": -float(a.mean()), "median": -float(np.median(a)),
            "median_nz": -float(np.median(nz)) if len(nz) else 0.0,
            "frac_fav": float((a < 0).mean()), "worst": -float(a.max())}


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    def f3(x):
        """a dead arm has spearman None. Printing it must not kill the run that found it -- the first version
        formatted None with :+.3f and crashed on the very first complex."""
        return "  DEAD " if x is None else f"{x:+.3f}"

    report("=" * 100)
    report("ENERGY LEVELS IN EACH SPHERE -- NEXUS's real physics on FFT poses, which it has never seen")
    report("=" * 100)
    report("  Geometry and learned embeddings both measured ~0 on these pose sets. The physics engine was")
    report("  never applied to them. It also answers a question the NEXUS ledger left open: vdW ranked native")
    report("  first 7/10 vs 6% chance against RANDOM-ROTATION decoys, and whether that survives HARD decoys")
    report("  was recorded as untested. FFT poses are the hard decoys.")
    report("  PREDECLARED: an energy arm beating shape on Spearman without losing top-100 or enrichment => the")
    report("  physics ranks poses. energy ~ shape => adds nothing. energy fails => the 70% was easy negatives,")
    report("  which corrects a headline NEXUS result. Sign fixed before the run: working arm has NEGATIVE rho.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    report(f"\n  test complexes (pinned from dock_fft.json): {test_pdbs}")

    rows = []
    for pdb in test_pdbs:
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        p["_cw"] = CW
        t = fp.table(pdb)
        if t is None:
            continue
        ca, cb = p["ca"], p["cb"]
        rn_map = NP.resnames(pdb)
        amask = t["ch"] == ca
        bmask = t["ch"] == cb
        # receptor sidechain atom indices per residue, computed ONCE (the structure never moves)
        sc_mask = amask & ~np.isin(t["nm"], list(fp.BB) + ["CB"])
        res_pool = []
        for pos in sorted(set(t["rn"][sc_mask].tolist())):
            idx = np.where(sc_mask & (t["rn"] == pos))[0]
            if len(idx):
                res_pool.append((int(pos), idx))
        if len(res_pool) < 5:
            continue

        poses, rots = PRK.fft_poses(p)
        if not poses:
            continue
        L0 = t["co"][bmask] - p["lcen"]
        t1 = time.time()
        rec, rmsd, shape = [], [], []
        for v, ri, sh, rm in poses:
            co = t["co"].copy()
            co[bmask] = L0 @ rots[ri].T + p["centre"] + sh
            e = sphere_energies(t, co, ca, cb, res_pool, rn_map,
                                cKDTree(co[bmask]), soft=True)
            eh = sphere_energies(t, co, ca, cb, res_pool, rn_map,
                                 cKDTree(co[bmask]), soft=False)
            if e is None or eh is None:
                continue
            d = {}
            for term in TERMS:
                s = summarise(e[term])
                sh_ = summarise(eh[term])
                if s is None or sh_ is None:
                    d = None
                    break
                for st in STATS:
                    d[f"soft_{term}_{st}"] = s[st]
                    d[f"raw_{term}_{st}"] = sh_[st]
            if d is None:
                continue
            rec.append(d); rmsd.append(rm); shape.append(v)
        if len(rec) < 100:
            report(f"    {pdb}: only {len(rec)} poses scored -- skipped")
            continue
        keys = list(rec[0])
        Dv = {k: np.array([r[k] for r in rec], float) for k in keys}
        rm = np.array(rmsd)
        Dv["shape"] = np.array(shape)
        frac = float((rm <= NEAR_NATIVE).mean())
        res = {}
        for k, v in Dv.items():
            live = float(np.std(v)) > 1e-12 and np.isfinite(v).all()
            o = np.argsort(-np.where(np.isfinite(v), v, -np.inf))
            res[k] = {"hits": {str(x): bool((rm[o][:x] <= NEAR_NATIVE).any()) for x in TOPK},
                      "spearman": float(spearmanr(v, rm).correlation) if live else None,
                      "top100_mean_rmsd": float(rm[o][:100].mean()), "live": bool(live)}
        rows.append({"pdb": pdb, "n_poses": int(len(rm)), "near_native_frac": frac,
                     "best_rmsd": float(rm.min()), "mean_rmsd": float(rm.mean()), "arms": res})
        report(f"    {pdb}: {len(rm)} poses in {time.time()-t1:.0f}s, frac {frac:.4f} | "
               f"shape {f3(res['shape']['spearman'])}  "
               f"soft_vdw_median_nz {f3(res['soft_vdw_median_nz']['spearman'])}  "
               f"raw_vdw_sum {f3(res['raw_vdw_sum']['spearman'])}  "
               f"soft_elec_sum {f3(res['soft_elec_sum']['spearman'])}")

    if not rows:
        report("\n  nothing scored")
        return 1

    keys = [k for k in rows[0]["arms"] if k != "shape"]
    tt = [r for r in rows if r["near_native_frac"] > 0]
    setmean = float(np.mean([r["mean_rmsd"] for r in rows]))
    report(f"\n  {len(rows)} complexes | {len(tt)} have a near-native pose: {[r['pdb'] for r in tt]}")

    dead = [k for k in keys if all(not r["arms"][k]["live"] for r in rows)]
    if dead:
        report(f"    LIVENESS: {dead} dead in every complex -- rows below mean nothing.")

    def agg(k):
        sp = [r["arms"][k]["spearman"] for r in rows if r["arms"][k]["spearman"] is not None]
        return {"top": {str(x): sum(r["arms"][k]["hits"][str(x)] for r in tt) for x in TOPK},
                "spearman_median": float(np.median(sp)) if sp else None,
                "top100_mean_rmsd": float(np.mean([r["arms"][k]["top100_mean_rmsd"] for r in rows]))}

    tab = {k: agg(k) for k in ["shape"] + keys}
    sh_ = tab["shape"]
    order = sorted(keys, key=lambda k: (tab[k]["spearman_median"] if tab[k]["spearman_median"] is not None else 9))
    report(f"\n  {'arm':<22}{'top-1':>7}{'top-10':>8}{'top-100':>9}{'Spearman':>11}{'top100 RMSD':>13}")
    report(f"  {'shape (baseline)':<22}{sh_['top']['1']:>7}{sh_['top']['10']:>8}{sh_['top']['100']:>9}"
           f"{f3(sh_['spearman_median']):>11}{sh_['top100_mean_rmsd']:>13.1f}")
    for k in order[:16]:
        report(f"  {k:<22}{tab[k]['top']['1']:>7}{tab[k]['top']['10']:>8}{tab[k]['top']['100']:>9}"
               f"{f3(tab[k]['spearman_median']):>11}{tab[k]['top100_mean_rmsd']:>13.1f}")
    ch = {str(x): float(np.mean([1 - (1 - r["near_native_frac"]) ** x for r in tt])) * len(tt) for x in TOPK}
    report(f"  {'chance':<22}{ch['1']:>7.2f}{ch['10']:>8.2f}{ch['100']:>9.2f}{0.0:>+11.3f}{setmean:>13.1f}")
    report(f"  (top-k denominator {len(tt)}; Spearman over all {len(rows)}; set mean RMSD {setmean:.1f} A)")

    best = order[0]
    bs = tab[best]["spearman_median"]
    if bs is None:
        report("\n  every arm is dead -- nothing to read")
        return 1
    dominates = (bs < sh_["spearman_median"] - 0.02
                 and tab[best]["top"]["100"] >= sh_["top"]["100"]
                 and tab[best]["top100_mean_rmsd"] <= sh_["top100_mean_rmsd"])
    report("\n  RAW vs SOFT-CORE -- how much is one clashing atom worth?")
    for term in TERMS:
        r_ = tab.get(f"raw_{term}_sum", {}).get("spearman_median")
        s_ = tab.get(f"soft_{term}_sum", {}).get("spearman_median")
        m_ = tab.get(f"soft_{term}_median", {}).get("spearman_median")
        report(f"    {term:<8} raw sum {f3(r_)}   soft sum {f3(s_)}   soft median-nz sphere "
               f"{f3(tab.get(f'soft_{term}_median_nz', {}).get('spearman_median'))}")

    report("\n  READING")
    if dominates:
        report(f"  THE PHYSICS RANKS POSES. '{best}' reaches Spearman {bs:+.3f} against shape's "
               f"{sh_['spearman_median']:+.3f},")
        report(f"  top-100 {tab[best]['top']['100']}/{len(tt)} vs {sh_['top']['100']}/{len(tt)}, enrichment "
               f"{tab[best]['top100_mean_rmsd']:.1f} vs {sh_['top100_mean_rmsd']:.1f} A. Geometry and learned")
        report("  embeddings both measured ~0 on these same poses; the energy in the spheres does not.")
    elif bs < sh_["spearman_median"] - 0.02:
        report(f"  PARTIAL. '{best}' improves Spearman to {bs:+.3f} from shape's {sh_['spearman_median']:+.3f}")
        report("  but loses on top-100 or enrichment, so it is not a better ranker overall -- different")
        report("  metrics pick different winners, which is not a result.")
    else:
        report(f"  THE PHYSICS DOES NOT RANK THESE POSES. Best arm '{best}' at Spearman {bs:+.3f} against")
        report(f"  shape's {sh_['spearman_median']:+.3f}. nexus_pairs measured vdW ranking the native first")
        report("  7/10 against 6% chance -- but those decoys were random rotations, which clash grossly. The")
        report("  FFT decoys are shape-optimised and clash-free, and against those the same physics is flat.")
        report("  **That 70% was against easy negatives**, and this corrects it rather than adding a new null.")

    json.dump({"test": "dock_energy", "max_res": MAX_RES, "iface_cut": IFACE_CUT, "terms": list(TERMS),
               "stats": list(STATS), "arms": tab, "chance": ch, "complexes": rows,
               "n_testable": len(tt), "dead_arms": dead, "best_arm": best, "dominates_shape": bool(dominates),
               "log": log}, open(OUT / "dock_energy.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_energy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
