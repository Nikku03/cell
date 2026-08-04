"""NEXUS ON 10 PAIRS: can the physics tell a real interface from a wrong one?

WHAT WAS ASKED, AND THE ONE SUBSTITUTION I HAD TO MAKE.  "Run NEXUS on 10 PPIs or non-PPIs." NEXUS scores an
interface that already exists -- given two chains in contact it returns cross-interface van der Waals, buried
contacts, electrostatics, induction and desolvation. A non-interacting PAIR has no complex structure, so there is
no interface to score; you would have to dock them first, and docking is the stage that does not exist here.

So the honest equivalent, on the same 10 pairs: **the real interface versus a WRONG one.** Take each complex,
keep chain A fixed, and rigid-body rotate chain B into a different orientation at the same separation. That is a
decoy interface -- same two proteins, same proximity, wrong geometry. If the physics cannot separate the native
from decoys, then the accept/reject loop in the proposed pipeline has no working judge, and it would converge
confidently on wrong structures while its score kept improving.

This is the go/no-go for the whole docking idea, and it is 10 complexes rather than a benchmark suite.

SCORING.  For every residue of chain A with a sidechain within reach of chain B, `flex_physics.sidechain_vdw`
returns the cross-interface terms. The interface score is their aggregate. Reported separately as well as
combined, because they can disagree and the disagreement is informative.

DECOYS.  Chain B is rotated by a uniformly random rotation about its own centroid, then translated so the
centroid separation matches the native. This keeps the two proteins in contact range and changes only WHICH
surfaces face each other -- the hard negative, not a trivially distant one.

PREDECLARED, before any number:
    native ranks 1st among native+decoys, well above chance   -> the physics is a usable judge; a search loop
                                                                 built on it can converge on something real.
    native indistinguishable from decoys                      -> there is no judge, and no amount of sampling
                                                                 fixes that. The docking pipeline stops here.
    native ranks WORSE than decoys                            -> the score is anti-correlated with truth, which
                                                                 would be more informative than a null and needs
                                                                 explaining before anything is built on it.

Chance for "native ranks 1st" is 1/(1+N_DECOY).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import flex_physics as fp

OUT = A.OUT
N_PAIRS, N_DECOY, MAX_RES = 10, 15, 60
SEED = 0


def rot_matrix(rng):
    """uniform random rotation via QR of a gaussian matrix"""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def interface_residues(t, ch_a, ch_b, cut=8.0):
    a = t["ch"] == ch_a
    b = t["ch"] == ch_b
    if not a.any() or not b.any():
        return []
    d = np.linalg.norm(t["co"][a][:, None, :] - t["co"][b][None, :, :], axis=2).min(1)
    near = set(t["rn"][a][d < cut].tolist())
    return sorted(near)


def resnames(pdb):
    """residue NAMES keyed by (chain, resnum). flex_physics.table() drops these, and without them
    sidechain_vdw cannot assign residue-aware charges -- electrostatics came back EXACTLY 0.000 in all ten
    complexes on the first run. That was a dead arm, not a null: the term was never tested."""
    out = {}
    path = f"{fp.PDBDIR}/{pdb}.pdb"
    if not os.path.exists(path):
        return out
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        try:
            out[(line[21], int(line[22:26]))] = line[17:20].strip()
        except ValueError:
            continue
    return out


def score_interface(t, ch_a, ch_b, residues, rnames=None):
    """aggregate NEXUS cross-interface terms over chain A's interface residues"""
    tot = {"vdw": 0.0, "contacts": 0, "elec": 0.0, "induc": 0.0, "desolv": 0.0}
    n = 0
    for pos in residues:
        try:
            wt = (rnames or {}).get((ch_a, int(pos)))
            sc = fp.sidechain_vdw(t, ch_a, pos, wt=wt)
        except Exception:
            sc = None
        if not sc:
            continue
        for k in tot:
            tot[k] += sc.get(k, 0.0)
        n += 1
    tot["n_scored"] = n
    return tot


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("NEXUS ON 10 PAIRS -- native interface vs rigid-body decoys")
    report("=" * 100)
    report(f"  PREDECLARED: native ranks 1st well above chance ({1/(1+N_DECOY):.3f}) => the physics is a usable")
    report("  judge. Indistinguishable => no judge, and the docking pipeline stops here.")

    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb"))
    rng = np.random.default_rng(SEED)
    rows = []
    for pdb in have:
        if len(rows) >= N_PAIRS:
            break
        try:
            t = fp.table(pdb)
        except Exception:
            continue
        if t is None:
            continue
        chains = sorted(set(t["ch"].tolist()))
        if len(chains) < 2:
            continue
        ca, cb = chains[0], chains[1]
        res = interface_residues(t, ca, cb)
        if len(res) < 5:
            continue
        res = res[:MAX_RES]

        rn_map = resnames(pdb)
        nat = score_interface(t, ca, cb, res, rn_map)
        if nat["n_scored"] < 3:
            continue

        # DECOYS: rotate chain B, keep centroid separation
        bmask = t["ch"] == cb
        amask = t["ch"] == ca
        bco = t["co"][bmask]
        bcen = bco.mean(0)
        acen = t["co"][amask].mean(0)
        sep = bcen - acen
        dec = []
        for _ in range(N_DECOY):
            R = rot_matrix(rng)
            t2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in t.items()}
            newb = (bco - bcen) @ R.T + acen + (R @ sep)
            t2["co"] = t["co"].copy()
            t2["co"][bmask] = newb
            r2 = interface_residues(t2, ca, cb)
            if len(r2) < 3:
                dec.append(None)
                continue
            dec.append(score_interface(t2, ca, cb, r2[:MAX_RES], rn_map))
        good = [d for d in dec if d and d["n_scored"] >= 3]
        if len(good) < 5:
            continue

        rows.append({"pdb": pdb, "chains": [ca, cb], "n_iface": len(res),
                     "native": nat, "decoys": good})
        report(f"    {pdb} {ca}/{cb}: {len(res)} interface residues, {len(good)}/{N_DECOY} usable decoys | "
               f"native vdw {nat['vdw']:.1f} contacts {nat['contacts']} elec {nat['elec']:.1f}")

    if len(rows) < 5:
        report(f"\n  only {len(rows)} usable complexes -- not enough to conclude")
        return 1

    report(f"\n  {len(rows)} complexes scored\n")
    terms = ["vdw", "contacts", "elec", "induc", "desolv"]
    # sign convention: more-negative vdw/elec = better; more contacts = better
    better = {"vdw": -1, "elec": -1, "induc": -1, "desolv": -1, "contacts": +1}
    res_tab = {}
    dead = [t_ for t_ in terms
            if all(abs(r["native"][t_]) < 1e-12 for r in rows)]
    if dead:
        report(f"    LIVENESS: {dead} are exactly zero in every complex -- dead arms, not nulls. "
               f"Their rows below mean nothing.")
    for term in terms:
        top1 = 0
        zs = []
        for r in rows:
            nv = r["native"][term] * better[term]
            dv = np.array([d[term] for d in r["decoys"]], float) * better[term]
            top1 += int(nv > dv.max())
            sd = dv.std()
            zs.append((nv - dv.mean()) / sd if sd > 1e-9 else 0.0)
        res_tab[term] = {"native_top1": top1 / len(rows), "mean_z": float(np.mean(zs))}
        report(f"    {term:<10} native ranks 1st in {top1}/{len(rows)} "
               f"({top1/len(rows):.2f})   mean z vs decoys {np.mean(zs):+.2f}")
    report(f"    {'chance':<10} {1/(1+N_DECOY):.2f}")

    best = max(terms, key=lambda x: res_tab[x]["native_top1"])
    bt = res_tab[best]["native_top1"]
    report("\n  READING")
    if bt >= 0.7:
        report(f"  '{best}' puts the native interface first in {bt:.0%} of complexes against a "
               f"{1/(1+N_DECOY):.0%} chance rate.")
        report("  The physics IS a usable judge at this scale, so a search loop built on it has something real")
        report("  to converge toward. Next question is whether it survives softer decoys than random rotations.")
    elif bt > 1.5 / (1 + N_DECOY):
        report(f"  Best term '{best}' ranks native first {bt:.0%} of the time vs {1/(1+N_DECOY):.0%} chance --")
        report("  above chance but far from reliable. A search loop would drift; it needs a stronger judge")
        report("  before the accept/reject stage means anything.")
    else:
        report(f"  No term separates native from decoys ({best} at {bt:.0%} vs {1/(1+N_DECOY):.0%} chance).")
        report("  There is no judge. Sampling more poses cannot fix a score that cannot rank them, and the")
        report("  docking pipeline stops here until the scoring is replaced.")

    json.dump({"test": "nexus_pairs", "n_pairs": len(rows), "n_decoy": N_DECOY,
               "chance_top1": 1 / (1 + N_DECOY), "per_term": res_tab,
               "complexes": [{"pdb": r["pdb"], "chains": r["chains"], "n_iface": r["n_iface"],
                              "native": r["native"], "n_decoys": len(r["decoys"])} for r in rows],
               "log": log}, open(OUT / "nexus_pairs.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_pairs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
