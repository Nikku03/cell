"""SELECTIVE RE-RANKING: use the structural signal ONLY where the model says it is unsure.

WHY THIS IS THE RIGHT NEXT MOVE, AND IT COMES ENTIRELY FROM MEASUREMENTS.

    four re-rankers lost globally    ESM 0.122 vs 0.675, rank-prior fusion 0.643, pocket-hinge 0.483 vs
                                     0.763, docking 0.600 vs 0.825. Each was applied to EVERY reaction.
    but docking reads the substrate  docking the true substrate beat docking a RANDOM metabolite into the
                                     same pockets, 0.600 vs 0.475, and +0.208 on obscure. The control
                                     separated. The signal is real; it is simply weaker than the prior it
                                     was overwriting.
    and the model knows when it is wrong   declared confidence splits rank-1 accuracy 0.891 against 0.456
                                     over 147 and 136 reactions. A gap of +0.435.

Those three facts have one implication. A global re-ranker spends most of its effort overwriting answers
that were already right -- on the confident half the incumbent is right 89% of the time, so almost any
change is a loss. On the unsure half it is right 46% of the time, and there is room to move.

    apply the structural score only where the model has told you it is unsure

That is the one form of composition not yet tried, and it is the form the evidence has been pointing at
since the first re-ranker failed.

ARMS
    tournament          the current best ordering, and the bar. Nothing is deployed unless it beats this.
    selective_dock      tournament, with docking applied ONLY to reactions below the confidence threshold
    global_dock         docking applied everywhere. The control that shows whether SELECTIVITY is doing the
                        work, or whether docking would have helped anyway on this subset.
    selective_random    random reordering applied to the SAME low-confidence subset. The control that
                        separates "the docking score is informative there" from "that subset is so weak
                        that any change helps". Without it a gain could be an artifact of picking the
                        subset rather than of the physics.

THE THRESHOLD IS SWEPT, NOT CHOSEN.  Picking the cut after seeing the answers would manufacture the result.
Accuracy is reported across the whole range of confidence cuts so the shape is visible, and the headline is
quoted at the pre-registered median split used when the confidence gap was first measured.

PREDECLARED, before any number:
    selective_dock beats tournament AND beats selective_random
        -> composition works when it is targeted; the four global failures were about WHERE the signal was
           applied rather than about the signal. This is the first usable structural contribution.
    selective_dock beats tournament but not selective_random
        -> the low-confidence subset is simply weak enough that perturbation helps, which is not a
           structural result and must not be reported as one.
    selective_dock does not beat tournament
        -> even targeted, the structural signal is too weak to add to the model's own ordering, and the
           shortlist ships as the tournament order with a confidence flag attached.

-> outputs/orphan/cell_orphan_selective.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from cell_orphan_chem_coverage import load_chebi, UBIQUITOUS  # noqa: E402
from nexus_pocket_modes import load_ca_and_heavy, pockets, AF  # noqa: E402
from nexus_dock_screen import ligand_3d, grids, dock, NDECOY  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TOPK = 10
SEED = 233
MAXPOS = 400


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("SELECTIVE RE-RANKING -- structure only where the model says it is unsure")
    report("=" * 100)
    report("  Four re-rankers lost when applied globally, but docking's control separated (0.600 real")
    report("  substrate vs 0.475 random) so the signal is real, and declared confidence splits accuracy")
    report("  0.891 vs 0.456. A global re-ranker spends its effort overwriting answers already right;")
    report("  on the unsure half the incumbent is right 46% of the time and there is room to move.")

    meta = json.load(open(OUT / "holdout_meta.json"))
    keep = json.load(open(OUT / "rankv2_meta.json"))
    ans = json.load(open(OUT / "rankv2_answers.json"))
    struct = json.load(open(SP / "cand_struct.json"))
    name2id, id2smi = load_chebi()
    pe2chebi = {}
    with open(SP / "rx_ChEBI2Reactome_PE_All_Levels.txt", errors="replace") as fh:
        for line in fh:
            c = line.split("\t")
            if len(c) > 1 and c[1].startswith("R-ALL-"):
                pe2chebi.setdefault(c[1], c[0].strip())

    def smiles_of(q):
        cid = pe2chebi.get(q.get("id"))
        if cid and cid in id2smi:
            return id2smi[cid]
        cid = name2id.get((q.get("name") or "").strip().lower())
        return id2smi.get(cid) if cid else None

    B = T.build()
    steps = B["steps"]
    rng = np.random.default_rng(SEED)

    jobs = []
    for pid, m in keep.items():
        a = ans.get(pid)
        if not a or not a.get("tournament"):
            continue
        o = meta[m["orig"]]
        s = steps[o["step"]]
        ins = [q for q in (s.get("in") or [])
               if q.get("type") == "METABOLITE" and not q.get("currency")
               and (q.get("name") or "").strip().lower() not in UBIQUITOUS]
        smi = smiles_of(max(ins, key=lambda q: len(q.get("name") or ""))) if ins else None
        cands = [g for g in a["tournament"][:TOPK] if g]
        jobs.append({"pid": pid, "order": cands, "truth": set(o["truth"]),
                     "obscure": o["obscure"], "conf": float(a.get("confidence") or 0.5),
                     "smiles": smi,
                     "dockable": [g for g in cands if g in struct]})
    conf = np.array([j["conf"] for j in jobs])
    thr = float(np.median(conf))
    n_low = int((conf < thr).sum())
    n_dock = sum(1 for j in jobs if j["conf"] < thr and j["smiles"] and len(j["dockable"]) >= 5)
    report(f"\n  {len(jobs)} reactions with a tournament order and a confidence")
    report(f"  median confidence {thr:.2f} -> {n_low} below it; of those {n_dock} have a resolvable")
    report(f"  substrate and >=5 cached structures, so {n_dock} are actually re-rankable")
    if n_dock < 15:
        report("  *** too few dockable low-confidence reactions to measure anything; stopping")
        return 1

    # ---- dock, but only what will be used -----------------------------------------------------------
    pocket_cache, ligcache, dsc = {}, {}, {}

    def lig(smi):
        if smi not in ligcache:
            ligcache[smi] = ligand_3d(smi)
        return ligcache[smi]

    def pg(g):
        if g in pocket_cache:
            return pocket_cache[g]
        try:
            ca, heavy = load_ca_and_heavy(AF / f"{struct[g]}.pdb")
            pk = pockets(heavy, ca)
            if not pk or len(heavy) < 100:
                pocket_cache[g] = None
            else:
                lin = ca[pk[0]["lining"]] if len(pk[0]["lining"]) else ca
                c = lin.mean(0)
                G = grids(heavy, ["C"] * len(heavy), np.zeros(len(heavy)), c, np.full(3, 9.0))
                pocket_cache[g] = (G, c + (rng.random((MAXPOS, 3)) - 0.5) * 10.0)
        except Exception:
            pocket_cache[g] = None
        return pocket_cache[g]

    pool = sorted({j["smiles"] for j in jobs if j["smiles"]})
    decoys = [d for d in (lig(pool[i]) for i in
                          rng.choice(len(pool), min(NDECOY, len(pool)), replace=False)) if d]
    report(f"  {len(decoys)} decoy ligands for pocket normalisation")

    todo = [j for j in jobs if j["conf"] < thr and j["smiles"] and len(j["dockable"]) >= 5]
    for k, j in enumerate(todo):
        L = lig(j["smiles"])
        if L is None:
            continue
        sc = {}
        for g in j["dockable"]:
            p = pg(g)
            if p is None:
                continue
            G, pts = p
            real = dock(L, G, pts, np.random.default_rng(SEED + 1))
            dec = [dock(d, G, pts, np.random.default_rng(SEED + 2)) for d in decoys[:6]]
            mu, sd = float(np.mean(dec)), float(np.std(dec) + 1e-9)
            sc[g] = (real - mu) / sd
        if len(sc) >= 5:
            dsc[j["pid"]] = sc
        if (k + 1) % 10 == 0:
            report(f"    docked {k+1}/{len(todo)} at {time.time()-t0:.0f}s")
    report(f"  docked {len(dsc)} low-confidence reactions at {time.time()-t0:.0f}s")

    def rr(order, sc):
        base = {g: -i for i, g in enumerate(order)}
        return sorted(order, key=lambda g: -(0.35 * base[g] - sc.get(g, 0.0)))

    def arm(name, use):
        h1, h3 = [], []
        for j in jobs:
            o = use(j)
            h1.append(bool(j["truth"] & set(o[:1])))
            h3.append(bool(j["truth"] & set(o[:3])))
        return np.array(h1), np.array(h3)

    ob = np.array([j["obscure"] for j in jobs])
    low = conf < thr
    ARMS = {
        "tournament": lambda j: j["order"],
        "selective_dock": lambda j: (rr(j["order"], dsc[j["pid"]])
                                     if (j["conf"] < thr and j["pid"] in dsc) else j["order"]),
        "global_dock": lambda j: rr(j["order"], dsc.get(j["pid"], {})),
        "selective_random": lambda j: (rr(j["order"], {g: rng.normal() for g in j["order"]})
                                       if (j["conf"] < thr and j["pid"] in dsc) else j["order"]),
    }
    report(f"\n  ALL {len(jobs)} reactions ({int(ob.sum())} obscure)")
    report(f"    {'arm':<20}{'@1':>8}{'@3':>8}{'@1 obs':>9}{'@1 LOW-conf':>13}")
    res = {}
    for a, fn in ARMS.items():
        h1, h3 = arm(a, fn)
        res[a] = {"@1": float(h1.mean()), "@3": float(h3.mean()),
                  "@1_obscure": float(h1[ob].mean()) if ob.sum() else None,
                  "@1_lowconf": float(h1[low].mean()) if low.sum() else None}
        report(f"    {a:<20}{h1.mean():>8.3f}{h3.mean():>8.3f}"
               f"{(h1[ob].mean() if ob.sum() else float('nan')):>9.3f}"
               f"{(h1[low].mean() if low.sum() else float('nan')):>13.3f}")

    # threshold sweep, so the median split is not a number chosen after the fact
    report(f"\n  THRESHOLD SWEEP -- the cut is swept rather than chosen, since picking it after seeing")
    report("  the answers would manufacture the result. Headline is the pre-registered median.")
    report(f"    {'cut':>6}{'n re-ranked':>13}{'selective @1':>14}{'tournament @1':>15}")
    sweep = []
    for q in (0.25, 0.4, 0.5, 0.6, 0.75):
        t_ = float(np.quantile(conf, q))
        h = []
        for j in jobs:
            o = (rr(j["order"], dsc[j["pid"]]) if (j["conf"] < t_ and j["pid"] in dsc) else j["order"])
            h.append(bool(j["truth"] & set(o[:1])))
        n_ = sum(1 for j in jobs if j["conf"] < t_ and j["pid"] in dsc)
        sweep.append({"q": q, "cut": t_, "n": n_, "acc": float(np.mean(h))})
        report(f"    {t_:>6.2f}{n_:>13}{np.mean(h):>14.3f}{res['tournament']['@1']:>15.3f}")

    report("\n  READING")
    s, t_, g, r_ = (res["selective_dock"]["@1"], res["tournament"]["@1"],
                    res["global_dock"]["@1"], res["selective_random"]["@1"])
    report(f"  selective {s:.3f} | tournament {t_:.3f} | global dock {g:.3f} | selective random {r_:.3f}")
    if s > t_ + 0.01 and s > r_ + 0.01:
        report("  TARGETED COMPOSITION WORKS. The four global failures were about WHERE the signal was")
        report("  applied, not about the signal. Docking helps exactly where the model admits doubt, and")
        report("  leaving the confident half alone is what makes it a gain instead of a loss.")
    elif s > r_ + 0.01:
        report("  It beats random perturbation of the same subset but not the tournament order, so the")
        report("  docking score is informative there and still not informative enough to overwrite.")
    elif s > t_ + 0.01:
        report("  It beats the tournament but NOT random perturbation of the same subset -- so the gain is")
        report("  that the low-confidence subset is weak enough for any change to help, which is not a")
        report("  structural result and must not be reported as one.")
    else:
        report("  EVEN TARGETED, THE STRUCTURAL SIGNAL DOES NOT ADD. That closes structure-based")
        report("  re-ranking for this task: global failed four times, targeted fails once with the")
        report("  strongest available signal on the subset most able to benefit. The deliverable is the")
        report("  tournament order with a confidence flag, and the confidence flag is the real product --")
        report("  0.891 on the half it vouches for is worth more than any reordering measured here.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_orphan_selective", "n": len(jobs), "threshold": thr,
               "n_docked": len(dsc), "arms": res, "sweep": sweep, "log": log},
              open(OUT / "cell_orphan_selective.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_selective.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
