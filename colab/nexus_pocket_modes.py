"""THE CHROMATIN MACHINERY ON CANDIDATE ENZYMES: pockets from spheres, hinges from slow modes.

THE IDEA, AND IT REUSES EVERYTHING ALREADY BUILT AND TESTED.  The catalyst shortlist has the true gene
somewhere in its top-50 for 80.8% of obscure reactions but puts it first only 42.5% of the time, so ranking
is the bottleneck and there is 0.38 of headroom. The chromatin work built an elastic network solver, a mode
decomposition and a sphere tiling, all of which run on any structure. Applied to a candidate enzyme they
answer two questions docking cannot, and neither needs a POSE:

    where are the cavities        sphere filling IS the pocket-detection algorithm -- a cavity is a
                                  connected cluster of buried probe spheres, which is the same
                                  neighbour-connectivity that made the two-level solver work
    which cavity is catalytic     enzyme active sites sit at mechanical HINGES, points where the slowest
                                  collective modes have least motion. That is a real structural regularity
                                  and it is exactly what the eigensolver already computes

WHAT THIS CAN AND CANNOT DO, STATED BEFORE THE RUN BECAUSE IT BOUNDS THE RESULT.  An elastic network knows
shape and mechanics. It does not know chemistry: it cannot tell which bond breaks, and it has no
representation of the substrate at all unless a 3D structure of that substrate is supplied, which the model
does not have -- reactions carry metabolite NAMES.

So what is computable here is a PER-PROTEIN prior: is this thing enzyme-shaped, with a large pocket at a
hinge. It is not a pair score. That matters because the measured error profile says 68.5% of ranking
mistakes are wrong-FAMILY, but the families are usually both enzymes -- FASN against OLAH, PRKCA against
CAMK2A, PRCP against F12. A per-protein enzyme-likeness prior is silent between two enzymes.

The honest expectation is therefore modest: it should down-rank the non-enzymes that the top-50 prompt
deliberately placed in ranks 21-50, and say nothing useful between two catalytic proteins.

THE CONTROL THAT DECIDES WHETHER ANY OF IT IS WORTH KEEPING.  If a free annotation lookup -- "is this gene
annotated with an EC number" -- reorders the list as well as the mechanics does, then the structures, the
modes and the sphere filling have bought nothing, and the honest conclusion is to use the annotation. That
comparison is the point of the run, not the mechanics score in isolation.

ARMS
    llm             the shortlist as given. The baseline every arm must beat.
    hinge_pocket    re-ranked by pocket volume at a hinge, LLM rank kept as prior
    pocket_only     pocket volume alone, ignoring the modes. Isolates what the MODES add over
                    "has a big cavity", which is the part that needs the eigensolver.
    ec_annotation   re-ranked by whether the gene carries an EC number. Free, no structure, no physics.
    random_rerank   the same magnitude of reordering applied at random, so "any perturbation of the
                    ordering helps" cannot masquerade as signal.

PREDECLARED, before any number:
    hinge_pocket beats llm AND beats ec_annotation
        -> the mechanics earns its place; the chromatin machinery transfers to the catalyst problem.
    hinge_pocket beats llm but not ec_annotation
        -> the gain is "is it an enzyme", which a lookup supplies for free. Use the lookup.
    hinge_pocket does not beat llm
        -> re-ranking a confidence-ordered list with a structural prior fails a third time here, and the
           shortlist should be handed to a curator as-is.

-> outputs/orphan/nexus_pocket_modes.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
AF = SP / "af"
GRID = 1.6          # probe grid spacing, Angstrom
PROBE = 3.2         # a grid point is FREE if this far from every heavy atom (water-sized probe)
PSP_MIN = 5         # buried if enclosed along at least this many of 7 scan axes (LIGSITE-style)
ANM_CUT = 13.0      # CA-level elastic network cutoff, the standard value for this resolution
NMODE = 10          # slowest internal modes used for the hinge score
TOPK = 10           # re-rank this many candidates per reaction
SEED = 163
AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float)


def load_ca_and_heavy(path):
    ca, heavy = [], []
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        e = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if e == "H":
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        heavy.append(xyz)
        if line[12:16].strip() == "CA":
            ca.append(xyz)
    return np.array(ca, float), np.array(heavy, float)


def anm_msf(ca, cut=ANM_CUT, nmode=NMODE):
    """Per-residue mean-square fluctuation in the slowest internal modes.

    Same anisotropic network and same eigensolver as the chromatin work, at residue resolution. A LOW value
    means the residue barely moves in the collective modes -- a hinge -- which is where catalytic sites sit.
    """
    n = len(ca)
    if n < 30:
        return None
    d = ca[:, None, :] - ca[None, :, :]
    r = np.linalg.norm(d, axis=2)
    m = (r < cut) & (r > 1e-6)
    H = np.zeros((3 * n, 3 * n))
    ii, jj = np.where(m)
    for a in range(3):
        for b in range(3):
            v = -(d[ii, jj, a] * d[ii, jj, b]) / (r[ii, jj] ** 2)
            np.add.at(H, (3 * ii + a, 3 * jj + b), v)
    for i in range(n):
        blk = -H[3 * i:3 * i + 3].reshape(n, 3, 3).sum(0)
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = blk
    try:
        w, V = eigsh(sp.csr_matrix(H), k=min(nmode + 6, 3 * n - 2), sigma=-1e-3, which="LM")
    except Exception:
        return None
    o = np.argsort(w)
    w, V = w[o], V[:, o]
    keep = w > 1e-8
    w, V = w[keep][:nmode], V[:, keep][:, :nmode]
    if not len(w):
        return None
    msf = ((V ** 2).reshape(n, 3, -1).sum(1) / w).sum(1)
    return msf


def pockets(heavy, ca):
    """Cavities by sphere filling: free grid points that are ENCLOSED, then connected into clusters.

    This is the pocket-detection algorithm rather than an analogy for it -- a cavity is a connected cluster
    of buried probe spheres. Returns each pocket's volume and the CA indices lining it.
    """
    if len(heavy) < 50:
        return []
    from scipy.spatial import cKDTree
    lo, hi = heavy.min(0) - 4.0, heavy.max(0) + 4.0
    gs = [np.arange(lo[a], hi[a], GRID) for a in range(3)]
    if max(len(g) for g in gs) > 150:            # keep the grid tractable on very large chains
        return []
    shape = tuple(len(g) for g in gs)
    G = np.stack(np.meshgrid(*gs, indexing="ij"), -1)
    P = G.reshape(-1, 3)
    # NEAREST-ATOM DISTANCE BY KD-TREE. The first version compared every grid point against every atom in
    # chunks -- 1e6 points x 1e4 atoms is 1e10 distance evaluations and it alone took tens of seconds.
    free = (cKDTree(heavy).query(P, k=1)[0] > PROBE)
    occ = (~free).reshape(shape)
    freeg = free.reshape(shape)

    def shifted(A, off):
        """A translated by an integer voxel offset, zero-filled at the boundary (np.roll would wrap)."""
        out = np.zeros_like(A)
        src = [slice(max(0, -o), A.shape[i] - max(0, o)) for i, o in enumerate(off)]
        dst = [slice(max(0, o), A.shape[i] - max(0, -o)) for i, o in enumerate(off)]
        out[tuple(dst)] = A[tuple(src)]
        return out

    # BURIEDNESS by shift-accumulation rather than index construction. Along each axis, is there protein on
    # BOTH sides within the scan range -- the LIGSITE criterion. Building a (shape,3) float index grid per
    # axis per step, as the first version did, was 300x the work of just shifting the occupancy array.
    psp = np.zeros(shape, np.int8)
    for ax in AXES:
        step = np.sign(ax).astype(int)
        both = np.ones(shape, bool)
        for sign in (1, -1):
            seen = np.zeros(shape, bool)
            for t in range(1, 20):
                seen |= shifted(occ, tuple(step * sign * t))
            both &= seen
        psp += both
    buried = freeg & (psp >= PSP_MIN)
    if buried.sum() < 5:
        return []
    # connected clusters of buried points
    from scipy import ndimage
    lab, nlab = ndimage.label(buried)
    out = []
    for c in range(1, nlab + 1):
        pts = np.argwhere(lab == c)
        if len(pts) < 6:
            continue
        xyz = lo + pts * GRID
        vol = len(pts) * GRID ** 3
        dca = np.linalg.norm(ca[:, None, :] - xyz[None, :, :], axis=2).min(1)
        lining = np.where(dca < 8.0)[0]
        out.append({"vol": float(vol), "lining": lining})
    out.sort(key=lambda p: -p["vol"])
    return out[:5]


def score_protein(path):
    ca, heavy = load_ca_and_heavy(path)
    if len(ca) < 30:
        return None
    pk = pockets(heavy, ca)
    if not pk:
        return {"vol": 0.0, "hinge": 0.0, "n_res": len(ca)}
    msf = anm_msf(ca)
    top = pk[0]
    if msf is None or not len(top["lining"]):
        return {"vol": top["vol"], "hinge": 0.0, "n_res": len(ca)}
    # hinge score: how far BELOW the protein's median fluctuation the pocket lining sits.
    # Positive means the pocket is at a stiff point of the slow modes, i.e. a hinge.
    med = float(np.median(msf))
    lin = float(np.median(msf[top["lining"]]))
    hinge = float((med - lin) / max(med, 1e-12))
    return {"vol": top["vol"], "hinge": hinge, "n_res": len(ca)}


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("POCKETS FROM SPHERES, HINGES FROM SLOW MODES -- the chromatin machinery on candidate enzymes")
    report("=" * 100)
    report("  The shortlist holds the true gene in its top-50 for 80.8% of obscure reactions but ranks it")
    report("  first only 42.5% of the time, so ranking is the bottleneck. Sphere filling finds cavities")
    report("  without a pose, and the slowest modes say which cavity is at a hinge, where active sites sit.")
    report("  STATED BEFORE THE NUMBERS: with no substrate 3D structure this is a PER-PROTEIN prior, not a")
    report("  pair score. It should down-rank non-enzymes and stay silent between two enzymes -- and 68.5%")
    report("  of the ranking errors are enzyme-vs-enzyme. The EC-annotation arm is the control that decides")
    report("  whether the physics beat a free lookup.")

    meta = json.load(open(OUT / "holdout_meta.json"))
    rk = json.load(open(OUT / "holdout_top50_partial.json"))
    struct = json.load(open(SP / "cand_struct.json"))
    report(f"\n  {len(rk)} reactions | {len(struct)} candidate genes with a local AlphaFold structure")

    need = sorted({g for p, v in rk.items() for g in v["genes"][:TOPK] if g in struct})
    report(f"  scoring the top-{TOPK} of each list -> {len(need)} distinct proteins to characterise")

    sc, bad = {}, 0
    for k, g in enumerate(need):
        f = AF / f"{struct[g]}.pdb"
        if not f.exists():
            bad += 1
            continue
        try:
            s = score_protein(f)
        except Exception:
            s = None
        if s is None:
            bad += 1
            continue
        sc[g] = s
        if (k + 1) % 200 == 0:
            report(f"    {k+1}/{len(need)} at {time.time()-t0:.0f}s")
    report(f"  characterised {len(sc)} proteins ({bad} failed or too small) at {time.time()-t0:.0f}s")
    if len(sc) < 50:
        report("  *** too few proteins characterised to score anything")
        return 1

    vols = np.array([s["vol"] for s in sc.values()])
    hin = np.array([s["hinge"] for s in sc.values()])
    report(f"  pocket volume  median {np.median(vols):.0f} A^3 | hinge score median {np.median(hin):+.3f}")

    # EC annotation, free and structure-free
    ecmap = {}
    ecf = SP / "uniprot_ec.tsv"
    if ecf.exists():
        import csv
        with open(ecf) as fh:
            r = csv.reader(fh, delimiter="\t")
            hdr = next(r)
            gi = [i for i, h in enumerate(hdr) if "gene" in h.lower()]
            ei = [i for i, h in enumerate(hdr) if "ec" in h.lower()]
            for row in r:
                if not row or not ei or len(row) <= max(ei + gi):
                    continue
                if row[ei[0]].strip() and gi:
                    for g in row[gi[0]].split():
                        ecmap[g.upper()] = 1
    report(f"  EC annotation available for {len(ecmap)} genes")

    rng = np.random.default_rng(SEED)
    zv = {g: (sc[g]["vol"] - vols.mean()) / (vols.std() + 1e-9) for g in sc}
    zh = {g: (sc[g]["hinge"] - hin.mean()) / (hin.std() + 1e-9) for g in sc}

    def rerank(genes, key):
        base = {g: -i for i, g in enumerate(genes)}          # LLM rank as prior, best = highest
        return sorted(genes, key=lambda g: -(base[g] * 0.35 + key(g)))

    ARMS = {
        "llm": lambda genes: genes,
        "hinge_pocket": lambda genes: rerank(genes, lambda g: zv.get(g, 0.0) + zh.get(g, 0.0)),
        "pocket_only": lambda genes: rerank(genes, lambda g: zv.get(g, 0.0)),
        "ec_annotation": lambda genes: rerank(genes, lambda g: 1.5 * ecmap.get(g, 0)),
        "random_rerank": lambda genes: rerank(genes, lambda g: rng.normal()),
    }

    ob = []
    hits = {a: {1: [], 3: [], 5: []} for a in ARMS}
    for p, v in rk.items():
        if p not in meta:
            continue
        genes = [g for g in v["genes"][:TOPK]]
        if sum(g in sc for g in genes) < 5:
            continue
        tr = set(meta[p]["truth"])
        ob.append(meta[p]["obscure"])
        for a, fn in ARMS.items():
            o = fn(list(genes))
            for k in (1, 3, 5):
                hits[a][k].append(bool(tr & set(o[:k])))
    ob = np.array(ob)
    n = len(ob)
    report(f"\n  {n} reactions scored ({int(ob.sum())} obscure); candidates must have >=5 of top-{TOPK} "
           f"characterised")
    report(f"    {'arm':<16}{'@1':>8}{'@3':>8}{'@5':>8}{'@1 obs':>9}{'@5 obs':>9}")
    res = {}
    for a in ARMS:
        h = {k: np.array(hits[a][k]) for k in (1, 3, 5)}
        res[a] = {f"@{k}": float(h[k].mean()) for k in (1, 3, 5)}
        res[a]["@1_obscure"] = float(h[1][ob].mean()) if ob.sum() else None
        res[a]["@5_obscure"] = float(h[5][ob].mean()) if ob.sum() else None
        report(f"    {a:<16}{h[1].mean():>8.3f}{h[3].mean():>8.3f}{h[5].mean():>8.3f}"
               f"{(h[1][ob].mean() if ob.sum() else float('nan')):>9.3f}"
               f"{(h[5][ob].mean() if ob.sum() else float('nan')):>9.3f}")

    report("\n  READING")
    b, l, e = res["hinge_pocket"]["@1"], res["llm"]["@1"], res["ec_annotation"]["@1"]
    r_ = res["random_rerank"]["@1"]
    if b > l + 0.02 and b > e + 0.02:
        report(f"  THE MECHANICS EARNS ITS PLACE: {b:.3f} against {l:.3f} for the shortlist as given and")
        report(f"  {e:.3f} for a free EC lookup. Sphere-filled pockets scored at their slow-mode hinges")
        report("  carry information neither the language model nor the annotation has.")
    elif b > l + 0.02:
        report(f"  IT BEATS THE SHORTLIST ({b:.3f} vs {l:.3f}) BUT NOT THE FREE LOOKUP ({e:.3f}). The gain")
        report("  is 'is this an enzyme', which an EC annotation supplies without a structure, an")
        report("  eigensolve or a grid. Use the lookup and keep the physics out of the pipeline.")
    else:
        report(f"  RE-RANKING FAILS AGAIN: {b:.3f} against {l:.3f} for the list as given. That is the third")
        report("  time a structural prior has failed to improve a confidence-ordered shortlist here, after")
        report("  ESM-as-reranker (0.122 vs 0.675) and the rank-prior fusion (0.643). The consistent")
        report("  finding is that the language model's own ordering already carries what these signals")
        report("  know, and the shortlist should go to a curator unmodified.")
    report(f"  random_rerank sits at {r_:.3f}; any arm not clearly above it has only shuffled the order.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "nexus_pocket_modes", "n": n, "n_proteins": len(sc), "arms": res, "log": log},
              open(OUT / "nexus_pocket_modes.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_pocket_modes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
