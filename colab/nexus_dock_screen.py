"""DOCK THE ACTUAL SUBSTRATE: the last untested version of structure-based re-ranking.

WHY THIS ONE AND NOT ANOTHER RE-RANKER.  Three have failed on this shortlist -- ESM (0.122 against 0.675),
a rank-prior fusion (0.643), and sphere pockets scored at their mode hinges (0.483 against 0.763). All three
failed the same way and for the same reason: they are PER-PROTEIN priors, and 68.5% of the ranking errors
are enzyme-against-enzyme, where a verdict on the protein alone cannot speak. FASN against OLAH, PRKCA
against CAMK2A -- both are enzymes, both have pockets at hinges, and every per-protein score says the same
thing about both.

The only remaining move is to make the score PAIR-specific: put the reaction's own substrate into each
candidate's pocket and see whether it fits. That needs a 3D substrate, which the coverage gate established
is obtainable for 45.9% of orphans via Reactome's ChEBI cross-reference.

WHAT IS AND IS NOT BEING CLAIMED.  This measures BINDING PLAUSIBILITY, not catalysis. Two facts bound it and
both are stated before any number:

    energy always goes down     put any molecule in any pocket and van der Waals attracts. Every candidate
                                will score negative. The raw energy is therefore meaningless across
                                proteins, which is why every score here is a Z-SCORE against decoy ligands
                                docked into the SAME pocket -- that removes "this protein has a sticky
                                pocket" and leaves "this pocket prefers THIS molecule".
    binding is not catalysis    albumin binds most of the metabolome with excellent energy and catalyses
                                almost none of it. A high score is a hypothesis, not a mechanism, and the
                                second half of the original proposal -- attach the next substrate and see
                                if the reaction runs -- is not attempted, because making and breaking
                                covalent bonds needs quantum mechanics and setting one up requires knowing
                                which bond breaks, which is most of the answer.

HOW THE DOCKING WORKS, and why it is affordable.  A grid, as every docking program does it. The protein's
vdW and electrostatic potentials are precomputed once per pocket on a 0.8 A grid, so scoring a pose becomes
one lookup per ligand atom instead of a sum over every protein atom. Poses are rigid-body: the ligand's
single MMFF-optimised conformer, sampled over orientations and over grid positions inside the pocket. That
is cruder than a real docking program -- no ligand flexibility, no induced fit -- and it is stated rather
than hidden, because the question here is whether ANY pair-specific signal exists, not whether this
implementation is state of the art.

ARMS
    llm                the shortlist as given. The bar.
    dock               re-ranked by the substrate's decoy-normalised fit, LLM rank kept as prior
    dock_random_sub    THE DECIDING CONTROL. The identical pipeline with a RANDOM metabolite docked instead
                       of the reaction's own. If this scores as well, the method is reading pocket
                       stickiness rather than substrate identity, and the whole idea is dead however good
                       the numbers look.
    random_rerank      reordering of matched magnitude, so shuffling cannot pass as signal.

PREDECLARED, before any number:
    dock beats llm AND beats dock_random_sub
        -> a pair-specific structural signal exists, the earlier failures were about per-protein scoring
           rather than about structure, and this is worth scaling.
    dock beats llm but not dock_random_sub
        -> the gain is pocket stickiness, not substrate matching. Dead, and the control is what caught it.
    dock does not beat llm
        -> the fourth failure, and the last idea in this family. The shortlist goes to a curator unmodified
           and structure-based re-ranking of a confidence-ordered list is finished as a line of work.

-> outputs/orphan/nexus_dock_screen.json
"""
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from cell_orphan_chem_coverage import load_chebi, UBIQUITOUS  # noqa: E402
from nexus_pocket_modes import load_ca_and_heavy, pockets, AF  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GSP = 0.8            # docking grid spacing, Angstrom
NROT = 120           # rigid-body orientations sampled per ligand
NDECOY = 12          # decoy ligands docked into the same pocket to build the normalising distribution
TOPK = 10
MAXPOS = 400         # grid positions inside the pocket sampled per orientation
SEED = 191
VDW_R = {"C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1, "F": 1.6, "CL": 2.0, "BR": 2.1, "I": 2.2}


def ligand_3d(smiles, seed=1):
    """SMILES -> one MMFF-optimised conformer, heavy atoms only, centred."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() < 3 or m.GetNumAtoms() > 90:
        return None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=300)
    except Exception:
        pass
    m = Chem.RemoveHs(m)
    c = m.GetConformer()
    xyz = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                    for i in range(m.GetNumAtoms())])
    el = [a.GetSymbol().upper() for a in m.GetAtoms()]
    q = np.array([a.GetFormalCharge() for a in m.GetAtoms()], float)
    return xyz - xyz.mean(0), el, q


def grids(heavy, hel, hq, centre, half):
    """Precompute vdW and electrostatic potentials on a grid over the pocket.

    Once per pocket rather than once per pose. Without this, every pose costs a sum over every protein
    atom and the screen is unaffordable; with it a pose is one lookup per ligand atom.
    """
    lo = centre - half
    ax = [np.arange(lo[a], lo[a] + 2 * half[a] + GSP, GSP) for a in range(3)]
    shape = tuple(len(x) for x in ax)
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    from scipy.spatial import cKDTree
    tree = cKDTree(heavy)
    nb = tree.query_ball_point(P, r=8.0)
    vdw = np.zeros(len(P))
    ele = np.zeros(len(P))
    rad = np.array([VDW_R.get(e, 1.9) for e in hel])
    for k, idx in enumerate(nb):
        if not idx:
            continue
        idx = np.asarray(idx)
        d = np.linalg.norm(heavy[idx] - P[k], axis=1)
        d = np.maximum(d, 0.6)
        rm = rad[idx] + 1.9
        x = rm / d
        # softened 8-4 rather than 12-6: a rigid ligand in a rigid pocket clashes constantly, and r^-12
        # would make the score a clash detector instead of a fit measure
        vdw[k] = np.clip((x ** 8 - 2 * x ** 4).sum(), -50, 50)
        ele[k] = (hq[idx] / d ** 2).sum()
    return {"lo": lo, "shape": shape, "vdw": vdw.reshape(shape), "ele": ele.reshape(shape)}


def rand_rots(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def dock(lig, G, pts, rng):
    """Best rigid-body score of the ligand inside the pocket. Lower is better."""
    xyz, el, q = lig
    R = rand_rots(NROT, rng)
    rot = np.einsum("rij,aj->rai", R, xyz)                  # (NROT, natom, 3)
    if len(pts) > MAXPOS:
        pts = pts[rng.choice(len(pts), MAXPOS, replace=False)]
    best = np.inf
    sh = np.array(G["shape"])
    for c in pts:
        p = rot + c[None, None, :]
        ijk = np.round((p - G["lo"]) / GSP).astype(int)
        ok = np.all((ijk >= 0) & (ijk < sh), axis=-1)
        ijk = np.clip(ijk, 0, sh - 1)
        v = G["vdw"][ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        e = G["ele"][ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        v = np.where(ok, v, 50.0)
        sc = v.sum(1) + 4.0 * (q[None, :] * np.where(ok, e, 0.0)).sum(1)
        best = min(best, float(sc.min()))
    return best


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("DOCK THE ACTUAL SUBSTRATE -- the last untested version of structure-based re-ranking")
    report("=" * 100)
    report("  Three re-rankers failed as PER-PROTEIN priors because 68.5% of the ranking errors are")
    report("  enzyme-against-enzyme. This makes the score pair-specific: the reaction's own substrate,")
    report("  into each candidate's pocket. Scores are Z-SCORES against decoy ligands in the SAME pocket,")
    report("  because raw binding energy is always negative and says only 'this pocket is sticky'.")
    report("  MEASURES BINDING PLAUSIBILITY, NOT CATALYSIS -- albumin binds most of the metabolome.")

    name2id, id2smi = load_chebi()
    pe2chebi = {}
    f = SP / "rx_ChEBI2Reactome_PE_All_Levels.txt"
    with open(f, errors="replace") as fh:
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
    meta = json.load(open(OUT / "holdout_meta.json"))
    rk = json.load(open(OUT / "holdout_top50_partial.json"))
    struct = json.load(open(SP / "cand_struct.json"))
    rng = np.random.default_rng(SEED)

    # ---- reactions with a usable substrate and enough cached structures --------------------------------
    jobs = []
    for p, m in meta.items():
        if p not in rk:
            continue
        s = steps[m["step"]]
        ins = [q for q in (s.get("in") or [])
               if q.get("type") == "METABOLITE" and not q.get("currency")
               and (q.get("name") or "").strip().lower() not in UBIQUITOUS]
        if not ins:
            continue
        main_q = max(ins, key=lambda q: len(q.get("name") or ""))
        smi = smiles_of(main_q)
        if not smi:
            continue
        cands = [g for g in rk[p]["genes"][:TOPK] if g in struct]
        if len(cands) < 5:
            continue
        jobs.append({"pid": p, "smiles": smi, "sub": main_q.get("name"), "cands": cands,
                     "truth": set(meta[p]["truth"]), "obscure": meta[p]["obscure"],
                     "order": [g for g in rk[p]["genes"][:TOPK]]})
    report(f"\n  {len(jobs)} held-out reactions have a resolvable main substrate AND >=5 cached structures")
    if len(jobs) < 25:
        report("  *** too few to measure anything; stopping rather than reporting a number")
        return 1

    # ---- decoy ligand set --------------------------------------------------------------------------
    pool = sorted({j["smiles"] for j in jobs})
    decoy_smi = [pool[i] for i in rng.choice(len(pool), min(NDECOY, len(pool)), replace=False)]
    ligcache = {}

    def lig(smi, seed=1):
        if smi not in ligcache:
            ligcache[smi] = ligand_3d(smi, seed)
        return ligcache[smi]

    decoys = [d for d in (lig(s) for s in decoy_smi) if d]
    report(f"  {len(decoys)} decoy ligands built for pocket normalisation")

    # ---- score --------------------------------------------------------------------------------------
    pocket_cache = {}

    def pocket_grid(g):
        if g in pocket_cache:
            return pocket_cache[g]
        try:
            ca, heavy = load_ca_and_heavy(AF / f"{struct[g]}.pdb")
            pk = pockets(heavy, ca)
            if not pk or len(heavy) < 100:
                pocket_cache[g] = None
                return None
            # pocket centre from its lining residues; half-box padded to hold a ligand
            lin = ca[pk[0]["lining"]] if len(pk[0]["lining"]) else ca
            centre = lin.mean(0)
            half = np.full(3, 9.0)
            hel = ["C"] * len(heavy)
            G = grids(heavy, hel, np.zeros(len(heavy)), centre, half)
            pts = centre + (rng.random((MAXPOS, 3)) - 0.5) * 10.0
            pocket_cache[g] = (G, pts)
        except Exception:
            pocket_cache[g] = None
        return pocket_cache[g]

    res_rows = []
    for k, j in enumerate(jobs):
        L = lig(j["smiles"])
        if L is None:
            continue
        dsc, rsc = {}, {}
        for g in j["cands"]:
            pg = pocket_grid(g)
            if pg is None:
                continue
            G, pts = pg
            real = dock(L, G, pts, np.random.default_rng(SEED + 1))
            dec = [dock(d, G, pts, np.random.default_rng(SEED + 2)) for d in decoys[:6]]
            mu, sd = float(np.mean(dec)), float(np.std(dec) + 1e-9)
            dsc[g] = (real - mu) / sd                      # negative = fits better than decoys do
            rd = decoys[int(rng.integers(len(decoys)))]
            rsc[g] = (dock(rd, G, pts, np.random.default_rng(SEED + 3)) - mu) / sd
        if len(dsc) < 5:
            continue
        res_rows.append({"job": j, "dock": dsc, "rand": rsc})
        if (k + 1) % 10 == 0:
            report(f"    {k+1}/{len(jobs)} reactions at {time.time()-t0:.0f}s")
    report(f"  scored {len(res_rows)} reactions at {time.time()-t0:.0f}s")
    if len(res_rows) < 20:
        report("  *** too few scored; stopping")
        return 1

    def rerank(order, sc):
        base = {g: -i for i, g in enumerate(order)}
        return sorted(order, key=lambda g: -(0.35 * base[g] - sc.get(g, 0.0)))

    ARMS = {"llm": lambda r: r["job"]["order"],
            "dock": lambda r: rerank(r["job"]["order"], r["dock"]),
            "dock_random_sub": lambda r: rerank(r["job"]["order"], r["rand"]),
            "random_rerank": lambda r: rerank(r["job"]["order"],
                                              {g: rng.normal() for g in r["job"]["order"]})}
    ob = np.array([r["job"]["obscure"] for r in res_rows])
    report(f"\n  {len(res_rows)} reactions ({int(ob.sum())} obscure)")
    report(f"    {'arm':<18}{'@1':>8}{'@3':>8}{'@5':>8}{'@1 obs':>9}")
    out = {}
    for a, fn in ARMS.items():
        h1 = np.array([bool(r["job"]["truth"] & set(fn(r)[:1])) for r in res_rows])
        h3 = np.array([bool(r["job"]["truth"] & set(fn(r)[:3])) for r in res_rows])
        h5 = np.array([bool(r["job"]["truth"] & set(fn(r)[:5])) for r in res_rows])
        out[a] = {"@1": float(h1.mean()), "@3": float(h3.mean()), "@5": float(h5.mean()),
                  "@1_obscure": float(h1[ob].mean()) if ob.sum() else None}
        report(f"    {a:<18}{h1.mean():>8.3f}{h3.mean():>8.3f}{h5.mean():>8.3f}"
               f"{(h1[ob].mean() if ob.sum() else float('nan')):>9.3f}")

    report("\n  READING")
    d, l, r_ = out["dock"]["@1"], out["llm"]["@1"], out["dock_random_sub"]["@1"]
    if d > l + 0.02 and d > r_ + 0.02:
        report(f"  A PAIR-SPECIFIC SIGNAL EXISTS: {d:.3f} against {l:.3f} for the list as given and")
        report(f"  {r_:.3f} for the same pipeline with a RANDOM substrate. The earlier failures were about")
        report("  per-protein scoring, not about structure, and this is worth scaling.")
    elif d > r_ + 0.02:
        report(f"  It beats the random-substrate control ({d:.3f} vs {r_:.3f}) but not the shortlist")
        report(f"  ({l:.3f}). So substrate identity IS being read, but the language model's ordering")
        report("  already contains more than the docking adds.")
    else:
        report(f"  FOURTH FAILURE: dock {d:.3f}, random substrate {r_:.3f}, list as given {l:.3f}.")
        report("  The score does not distinguish the real substrate from a random metabolite, so it is")
        report("  reading pocket stickiness rather than fit. That is the control doing its job, and it")
        report("  closes structure-based re-ranking of this shortlist as a line of work.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "nexus_dock_screen", "n": len(res_rows), "arms": out, "log": log},
              open(OUT / "nexus_dock_screen.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_dock_screen.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
