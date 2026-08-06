"""STEP 2: CAN A BINDING CALCULATION PICK THE CATALYST OUT OF A SHORTLIST?  True enzyme vs 9 size-matched
decoys, on AlphaFold structures, well-powered.

WHY THIS IS A RE-TEST AND NOT A NEW IDEA.  `nexus_doesitbind` already asked the underlying question -- do
these two proteins interact at all, without needing the complex -- and answered it badly:

    docking distribution features   AUC 0.456        size_only control  AUC 0.438
    label permutation               median 0.496     permutation p = 0.66

Below chance, and indistinguishable from how big the two chains are. Sizing the catalyst design on 0.755 or
0.814 was my error: those are scored on a complex that ALREADY EXISTS. So the honest expectation here is
failure, and the reason to run it anyway is that the earlier test had n=80 crystal pairs, which is
underpowered, and this is a different population -- enzyme against its actual substrate, with size-matched
decoys drawn from the enzyme vocabulary rather than foreign chains.

    600 pairs instead of 80, and the decoys are OTHER ENZYMES, which is the discrimination the task needs.

WHAT IS BEING RANKED.  For a reaction whose catalyst is known and whose substrate is a protein, the true
catalyst plus 9 decoys are each docked against the substrate. If the physics knows anything, the true one
scores higher. This is the tie-break step from `nexus_screen_feasibility` -- the search version needs AUC
0.998 and is not attemptable.

THE DECOYS DECIDE WHETHER THE RESULT MEANS ANYTHING.  Bigger proteins make more contacts and score higher, so
decoys are matched to the true catalyst's length within +-25% and drawn from the catalyst vocabulary, never
random proteins. A SIZE-ONLY arm is reported alongside: if size alone separates true from decoy, the matching
failed and no docking number from this run may be believed.

ALPHAFOLD, AND THE TWO THINGS IT FORCES.  Structures come from AlphaFold DB v6 (v4 and v5 are 404; the path
resolves through /api/prediction/{acc}). Predicted models carry long disordered tails that are not real
surface, so residues below pLDDT 70 are dropped -- measured on four proteins, that takes CSNK2A1 from a 149 A
diameter to 73 A. And the FFT correlation is circular, so a pair whose two diameters exceed the box could
score a wrapped pose as a contact; pairs violating that guard are excluded, not clipped.

PREDECLARED, before any number:
    AUC ~0.5, size-only also ~0.5  -> the binding calculation cannot pick a catalyst out of even a
                                      10-candidate shortlist. Combined with the 0.998 a real screen needs,
                                      the structural route to catalyst-finding is closed with this machinery,
                                      and step 3 should not be run.
    AUC materially > 0.5 and > size-only
                                   -> the tie-break composition is live, and step 3 measures whether it beats
                                      the 0.463 the language arm delivers alone.
    size-only > 0.5                -> the decoy matching failed; nothing else in the run is interpretable.
"""
import json
import os
import sys
import time
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_ties_nn as T
import dock_fft as DF

OUT = A.OUT
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
AF = SP / "af"
UNI = SP / "uniprot_human.tsv"
BENCH = OUT / "catalyst_pilot_bench.json"
FEATS = OUT / "catalyst_pilot_features.json"

# A bigger box than dock_fft's 96: AlphaFold monomers are larger than the crystal chains that file was tuned
# on, and the circular-correlation guard rejects any pair whose diameters do not fit.
GRID, SPACING = 128, 1.4
N_ROT = 300              # partner discrimination reads the score DISTRIBUTION, so it does not need the
                         # 2,400 rotations pose-finding needed -- that was measured in dock_coverage
PLDDT_MIN = 70.0
N_RX, N_DECOY = 60, 9
SIZE_TOL = 0.25
MIN_AT, MAX_AT = 400, 12000
SEED = 3


def _acc_map():
    import csv
    g2u, glen = {}, {}
    with open(UNI) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            g = r["Gene Names (primary)"]
            if g and g not in g2u:
                g2u[g] = r["Entry"]
                glen[g] = int(r["Length"])
    return g2u, glen


def fetch_af(acc):
    p = AF / f"{acc}.pdb"
    if p.exists() and p.stat().st_size > 2000:
        return p
    AF.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(
                f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb", p)
            return p if p.stat().st_size > 2000 else None
        except Exception:
            time.sleep(2 ** attempt)
    return None


VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}


def load_struct(acc):
    """coordinates + vdW radii, with low-confidence residues dropped. A predicted tail is not surface."""
    p = fetch_af(acc)
    if p is None:
        return None
    co, rad = [], []
    for l in open(p):
        if not l.startswith("ATOM"):
            continue
        if float(l[60:66]) < PLDDT_MIN:
            continue
        el = (l[76:78].strip() or l[12:16].strip()[:1]).upper()
        co.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        rad.append(VDW.get(el, 1.7))
    if len(co) < MIN_AT or len(co) > MAX_AT:
        return None
    co = np.array(co, float)
    return {"co": co, "rad": np.array(rad, float),
            "diam": float(np.linalg.norm(co - co.mean(0), axis=1).max() * 2), "n": len(co)}


def build_bench():
    B = T.build()
    steps, cats = B["steps"], B["cats"]
    g2u, glen = _acc_map()
    PROT = {"PROTEIN", "COMPLEX", "SET"}
    rng = np.random.default_rng(SEED)

    pool = []
    for i in cats:
        s = steps[i]
        cs = [c for c in cats[i] if c in g2u]
        if not cs:
            continue
        subs = [g for w in ("in", "out") for p in (s.get(w) or []) if p.get("type") in PROT
                for g in (p.get("genes") or []) if g in g2u and g not in cats[i]]
        if not subs:
            continue
        cat, sub = cs[0], subs[0]
        if glen[cat] > 700 or glen[sub] > 700:
            continue
        pool.append({"rx": int(i), "catalyst": cat, "substrate": sub,
                     "cat_len": glen[cat], "sub_len": glen[sub]})
    print(f"  candidate reactions (known catalyst + protein substrate, both <=700 aa): {len(pool):,}")

    vocab = sorted({c for i in cats for c in cats[i] if c in g2u and glen[c] <= 700})
    vlen = np.array([glen[g] for g in vocab])
    pick = rng.choice(len(pool), min(N_RX * 3, len(pool)), replace=False)
    bench = []
    for k in pick:
        e = dict(pool[int(k)])
        L = e["cat_len"]
        ok = [g for g, n in zip(vocab, vlen)
              if abs(n - L) <= SIZE_TOL * L and g != e["catalyst"] and g not in cats[e["rx"]]]
        if len(ok) < N_DECOY:
            continue
        d = rng.choice(len(ok), N_DECOY, replace=False)
        e["decoys"] = [ok[int(x)] for x in d]
        bench.append(e)
        if len(bench) >= N_RX:
            break
    json.dump({"bench": bench, "g2u": {g: g2u[g] for e in bench
                                       for g in [e["catalyst"], e["substrate"]] + e["decoys"]},
               "glen": {g: glen[g] for e in bench
                        for g in [e["catalyst"], e["substrate"]] + e["decoys"]}},
              open(BENCH, "w"), indent=1)
    print(f"  wrote {len(bench)} reactions x {1 + N_DECOY} candidates -> {BENCH}")
    return 0


def dock_pair(sub, cand, rots, offs_cache):
    """dock one candidate against the substrate; return distribution features, never a pose.

    The circular-correlation box guard is enforced here: if the two diameters do not fit, a wrapped placement
    could register as a contact and the number would be fiction."""
    if sub["diam"] + cand["diam"] > GRID * SPACING:
        return None
    centre = sub["co"].mean(0)
    offs_R = offs_cache.setdefault(round(float(np.mean(sub["rad"])), 2), DF.offsets_for(sub["rad"]))
    solid_R = DF.rasterise(sub["co"], centre, DF.offsets_for(sub["rad"]))
    if solid_R is None:
        return None
    skin_R = DF._morph(solid_R, DF.SKIN, True) & ~solid_R
    core_R = DF._morph(solid_R, DF.CORE_ERODE, False)
    F_skin = np.fft.rfftn(skin_R.astype(np.float32))
    F_core = np.fft.rfftn(core_R.astype(np.float32))
    Lc = cand["co"] - cand["co"].mean(0)
    offs_L = DF.offsets_for(cand["rad"])
    best, clash_at_best, tops = [], [], []
    for R_ in rots:
        solid_L = DF.rasterise(Lc @ R_.T + centre, centre, offs_L)
        if solid_L is None:
            continue
        FL = np.conj(np.fft.rfftn(solid_L.astype(np.float32)))
        inter = np.fft.irfftn(F_skin * FL, solid_R.shape)
        clash = np.fft.irfftn(F_core * FL, solid_R.shape)
        sc = inter - 1.0 * clash
        j = int(np.argmax(sc))
        best.append(float(sc.reshape(-1)[j]))
        clash_at_best.append(float(clash.reshape(-1)[j]))
        tops.append(float(np.partition(sc.reshape(-1), -50)[-50:].mean()))
    if len(best) < 20:
        return None
    b = np.array(best)
    z = (b - b.mean()) / (b.std() + 1e-9)
    return {"best": float(b.max()), "top10": float(np.sort(b)[-10:].mean()),
            "mean": float(b.mean()), "std": float(b.std()),
            "skew": float(((z ** 3).mean())), "n_z2": float((z > 2).mean()),
            "top50cell": float(np.mean(tops)), "clash": float(np.mean(clash_at_best)),
            "n_atoms": int(cand["n"]), "diam": float(cand["diam"])}


def run_dock():
    DF.GRID, DF.SPACING = GRID, SPACING          # dock_fft's helpers read these at call time
    d = json.load(open(BENCH))
    bench, g2u = d["bench"], d["g2u"]
    rots = DF.rotations(N_ROT)
    rots = [np.asarray(r, float) for r in rots]
    out, t0, offs_cache = {}, time.time(), {}
    done = json.load(open(FEATS)) if FEATS.exists() else {}
    for bi, e in enumerate(bench):
        key = str(e["rx"])
        if key in done:
            out[key] = done[key]
            continue
        sub = load_struct(g2u.get(e["substrate"], ""))
        if sub is None:
            continue
        rec = {}
        for g in [e["catalyst"]] + e["decoys"]:
            c = load_struct(g2u.get(g, ""))
            if c is None:
                continue
            f = dock_pair(sub, c, rots, offs_cache)
            if f is not None:
                rec[g] = f
        if len(rec) >= 5 and e["catalyst"] in rec:
            out[key] = {"catalyst": e["catalyst"], "feats": rec}
        done.update(out)
        json.dump(done, open(FEATS, "w"))
        print(f"  [{bi+1}/{len(bench)}] rx {e['rx']} {e['catalyst']} vs {len(rec)-1} decoys "
              f"| {time.time()-t0:.0f}s elapsed", flush=True)
    print(f"  docked {len(out)} reactions -> {FEATS}")
    return 0


def score():
    from sklearn.metrics import roc_auc_score
    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    d = json.load(open(FEATS))
    report("=" * 100)
    report("STEP 2 -- can a binding calculation pick the catalyst out of a 10-candidate shortlist?")
    report("=" * 100)
    report("  PRIOR, measured: nexus_doesitbind put does-A-bind-B at AUC 0.456 with a size-only control of")
    report("  0.438 and permutation p=0.66. This is a better-powered re-test on enzyme-substrate pairs with")
    report("  size-matched decoys, not a new idea. Failure is the expected outcome.")
    report(f"\n  {len(d)} reactions scored")
    if not d:
        report("  NOTHING DOCKED. Do not read this as a negative -- it is an empty run.")
        return 1

    KEYS = ("best", "top10", "mean", "top50cell", "n_z2", "skew", "std", "clash")
    ys, per = [], {k: [] for k in KEYS}
    per["size_only"] = []
    ranks = []
    for rx, rec in d.items():
        cat, F = rec["catalyst"], rec["feats"]
        gs = sorted(F)
        if cat not in gs or len(gs) < 5:
            continue
        y = [1 if g == cat else 0 for g in gs]
        ys += y
        for k in KEYS:
            v = np.array([F[g][k] for g in gs], float)
            v = (v - v.mean()) / (v.std() + 1e-9)          # z within the shortlist, as doesitbind did
            per[k] += list(v)
        s = np.array([F[g]["n_atoms"] for g in gs], float)
        per["size_only"] += list((s - s.mean()) / (s.std() + 1e-9))
        b = np.array([F[g]["best"] for g in gs], float)
        ranks.append(int((b > b[gs.index(cat)]).sum()) + 1)

    y = np.array(ys)
    report(f"  {int(y.sum())} true catalysts against {int((1-y).sum())} decoys")
    report(f"\n  {'feature':<14}{'AUC':>8}   (0.5 = the shortlist is being reordered at random)")
    aucs = {}
    for k in list(KEYS) + ["size_only"]:
        v = np.array(per[k])
        a = float(roc_auc_score(y, v)) if v.std() > 1e-12 else float("nan")
        aucs[k] = a
        mark = "   <- THE ARTEFACT CONTROL" if k == "size_only" else ""
        report(f"  {k:<14}{a:>8.3f}{mark}")
    best_k = max((k for k in KEYS), key=lambda k: abs(aucs[k] - 0.5))
    rk = np.array(ranks)
    report(f"\n  rank of the true catalyst among its shortlist: mean {rk.mean():.2f}, "
           f"median {np.median(rk):.0f}, top-1 {np.mean(rk == 1):.3f} (chance ~{1/np.mean([1]*1)/10:.3f})")

    report("\n  READING")
    ba, sa = aucs[best_k], aucs["size_only"]
    if abs(sa - 0.5) > 0.08:
        report(f"  DECOY MATCHING FAILED. size_only reaches {sa:.3f}, so the shortlists differ systematically")
        report("  in size and no docking number here is interpretable. Fix the matching before reading on.")
    elif abs(ba - 0.5) < 0.06:
        report(f"  THE BINDING CALCULATION CANNOT PICK THE CATALYST. Best feature '{best_k}' at {ba:.3f} with")
        report(f"  the size control at {sa:.3f}. On a shortlist of ten, that is reordering at random.")
        report("  This reproduces nexus_doesitbind's 0.456 at more than seven times the sample size and on")
        report("  the population the task actually presents. Combined with the 0.998 a full screen would")
        report("  need, the structural route to catalyst-finding is CLOSED with this machinery, and step 3")
        report("  has nothing to compose -- reranking by a chance-level score cannot beat the language arm.")
    else:
        report(f"  SIGNAL. '{best_k}' reaches {ba:.3f} against a size control of {sa:.3f}. The tie-break")
        report("  composition is live and step 3 is worth running.")

    json.dump({"test": "nexus_catalyst_pilot", "n_reactions": len(d), "auc": aucs,
               "best_feature": best_k, "rank_mean": float(rk.mean()),
               "top1": float(np.mean(rk == 1)), "log": log},
              open(OUT / "nexus_catalyst_pilot.json", "w"), indent=2)
    report(f"\n  -> {OUT/'nexus_catalyst_pilot.json'}")
    return 0


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "score"
    raise SystemExit({"bench": build_bench, "dock": run_dock, "score": score}[m]())
