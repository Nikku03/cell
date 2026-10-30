"""CAN THE INTERFACE MODEL SUPPLY THE ORIENTATION THE DOCKING SEARCH IS MISSING?

WHERE THIS COMES FROM.  Two results that fail in complementary ways:

    dock_fft      exhaustive FFT search REACHES near-native (6/12 blind, 11/12 given the right rotation) but
                  RANKS it 0/12 in the top ten. Shape complementarity cannot pick the orientation.
    dmasif        interface discrimination at 0.9019 on held-out complexes. Knows WHICH surface patches bind.

The join is obvious: the piece that works needs orientation, and the piece at 0.90 constrains orientation. This
tests whether that join is real, by re-ranking the SAME 2,400 poses per complex with the learned model instead
of with shape, and asking whether near-native now reaches the top ten.

WHY THE EMBEDDINGS ARE CHEAP HERE. A rigid pose does not change either protein's own surface geometry, so the
embeddings are POSE-INVARIANT: compute them once per chain, and each pose only changes WHICH point pairs are in
contact. That turns a 29,000-pose rescoring job from hours into seconds.

=======================================================================================================
A LEAK IN THE 0.9019 ITSELF, WHICH THIS FILE HAS TO FIX BEFORE IT CAN USE THE MODEL AT ALL
=======================================================================================================
`dmasif.point_cloud` builds the surface from the WHOLE COMPLEX's atoms, and `precompute_graph` builds the KNN
graph over ALL chains' points together. So at an interface, a receptor point's 16 nearest neighbours include
LIGAND points a few Angstrom away, and the geodesic convolution aggregates across the interface. The training
task is then "do these two points contact?" -- partly answered by the input, because a point's embedding already
encodes the partner sitting on top of it. The complex's surface is also missing the buried interface entirely.

None of that is available when docking: you have two proteins APART. So this file rebuilds each chain's surface
from THAT CHAIN'S ATOMS ONLY, with its own APBS field, and its own KNN graph. That is the unbound setting the
docking stage actually faces.

Both models are then trained on the same complexes and scored on the same held-out set, which turns the caveat
into a number:

    bound-graph AUC     the setting 0.9019 was measured in: whole-complex surface, cross-chain KNN
    unbound-graph AUC   per-chain surface, per-chain APBS, per-chain KNN -- what docking can actually use
    the gap            IS the leak

PREDECLARED, before any number:
    unbound AUC ~ bound AUC        -> the cross-chain graph was not doing the work and 0.9019 stands as stated.
    unbound AUC << bound AUC       -> 0.9019 was inflated by seeing the partner, and the docking-relevant
                                      number is the unbound one. That is a correction to a result I reported.

AND FOR THE RE-RANKING, arms and controls:
    shape              the existing baseline, 0/12 in top-10
    learned_mean       mean predicted contact probability over the pose's contacting point pairs
    learned_sum        the SUM. Predeclared to behave like raw contact counting, which nexus_pairs measured as
                       ACTIVELY WRONG (z -1.81, decoys have more contacts). If sum beats mean, that finding
                       does not replicate and something is off.
    shape+learned      z-combined
    UNTRAINED          identical architecture, random init, same everything else. THE control: if random
                       weights rank as well as trained ones, the learning is not what is doing the work.
    random             exact chance rate 1-(1-f)^k from the known near-native fraction f of each pose set.

PREDECLARED for the re-ranking:
    learned puts near-native in top-10 for a majority AND beats both shape and UNTRAINED -> the join is real.
    learned ~ untrained                                                                  -> geometry of the
                                                                                            embedding, not
                                                                                            learning.
    learned ~ shape ~ chance                                                             -> the 0.90 predictor
                                                                                            does not transfer
                                                                                            to pose ranking,
                                                                                            and the join fails.

HOMOLOGY CAVEAT, stated up front: the 12 test complexes include several near-identical variants (1CSE/1CSO/
1CT0/1CT2/1CT4 are all subtilisin+eglin). Training excludes the test PDB ids exactly but not their homologues,
so the learned arms are if anything FLATTERED. The UNTRAINED control is the guard that matters here: leakage
cannot help a network whose weights were never fitted.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dmasif as D
import dock_fft as DF
import flex_physics as fp

OUT = A.OUT
N_TRAIN, N_EVAL = 50, 14         # complexes for training the two models / for the AUC comparison
EPOCHS = 25
CONTACT = 4.5                    # Angstrom: surface points this close count as touching
MIN_PAIRS = 10                   # a pose needs this many contacting pairs to be scored at all
TOPK = (1, 10, 100)
NEAR_NATIVE = DF.NEAR_NATIVE


def chain_pdb(pdb, ch):
    """write a chain-only PDB into PDBDIR so point_cloud/APBS see ONE protein, the unbound setting. The complex
    surface has the interface buried and its KNN graph reaches across chains; neither is available at dock time."""
    tag = f"{pdb}__{ch}"
    dst = f"{D.PDBDIR}/{tag}.pdb"
    if not os.path.exists(dst):
        src = f"{D.PDBDIR}/{pdb}.pdb"
        if not os.path.exists(src):
            return None
        with open(src) as f, open(dst, "w") as g:
            n = 0
            for line in f:
                if line.startswith("ATOM") and line[21] == ch:
                    g.write(line)
                    n += 1
            g.write("END\n")
        if n < 200:
            os.remove(dst)
            return None
    return tag


def build_unbound(pdb, ca, cb):
    """per-chain surface + per-chain APBS + per-chain KNN for both partners"""
    out = {}
    for ch in (ca, cb):
        tag = chain_pdb(pdb, ch)
        if tag is None:
            return None
        try:
            pc = D.point_cloud(tag)
        except Exception:
            return None
        if pc is None or len(pc["pts"]) < 100:
            return None
        out[ch] = {"pc": pc, "g": D.precompute_graph(pc)}
    return out


def cross_pairs(P, Q, contact=CONTACT, far=12.0, cap=1500, seed=0):
    """contacting and far-apart point pairs between two separate clouds"""
    from scipy.spatial import cKDTree
    tq = cKDTree(Q)
    d, nn = tq.query(P)
    rng = np.random.default_rng(seed)
    pos = [(int(i), int(nn[i])) for i in np.where(d < contact)[0]]
    fk = np.where(d > far)[0]
    neg = []
    if len(fk):
        for k in rng.choice(fk, min(len(fk), 4 * max(len(pos), 1)), replace=False):
            neg.append((int(k), int(nn[k])))
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:cap], neg[:cap]


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from scipy.spatial import cKDTree
    torch.set_num_threads(4)

    report("=" * 100)
    report("CAN THE INTERFACE MODEL SUPPLY THE ORIENTATION THE DOCKING SEARCH IS MISSING?")
    report("=" * 100)
    report("  PREDECLARED (leak): unbound AUC ~ bound AUC => 0.9019 stands. unbound << bound => 0.9019 was")
    report("  inflated by the graph reaching across the interface, and the docking number is the unbound one.")
    report("  PREDECLARED (join): learned beats BOTH shape and UNTRAINED in top-10 => the join is real.")
    report("  learned ~ untrained => embedding geometry, not learning. learned ~ chance => the join fails.")

    class GeoConv(nn.Module):
        def __init__(s, din, dout):
            super().__init__()
            s.edge = nn.Sequential(nn.Linear(4 + din, dout), nn.ReLU())
            s.logsig = nn.Parameter(torch.tensor(1.0))
            s.self = nn.Sequential(nn.Linear(din + dout, dout), nn.ReLU())

        def forward(s, feat, idx, lc, dist):
            e = s.edge(torch.cat([lc, dist.unsqueeze(-1), feat[idx]], -1))
            w = torch.exp(-(dist ** 2) / (torch.exp(s.logsig) ** 2 + 1e-6)).unsqueeze(-1)
            return s.self(torch.cat([feat, (w * e).sum(1) / (w.sum(1) + 1e-6)], -1))

    class Net(nn.Module):
        def __init__(s, fd):
            super().__init__()
            s.c1, s.c2, s.out = GeoConv(fd, 32), GeoConv(32, 32), nn.Linear(32, 32)

        def forward(s, feat, idx, lc, dist):
            h = s.c2(s.c1(feat, idx, lc, dist), idx, lc, dist)
            e = s.out(h)
            return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def emb(model, d):
        pc, g = d["pc"], d["g"]
        return model(torch.tensor(pc["feat"]), torch.tensor(g["idx"]),
                     torch.tensor(g["lc"]), torch.tensor(g["dist"]))

    # ---------------- select complexes: real interface, both chains usable unbound ----------------
    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb") and "__" not in f)
    dockj = json.load(open(OUT / "dock_fft.json"))
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    report(f"\n  {len(have):,} structures | docking TEST set to re-rank: {test_pdbs}")

    report("\n  building UNBOUND per-chain surfaces (own atoms, own APBS, own KNN)...")
    pool, tb = {}, time.time()
    order = test_pdbs + [p for p in have if p not in test_pdbs]
    for pdb in order:
        if len(pool) >= N_TRAIN + N_EVAL + len(test_pdbs):
            break
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        ub = build_unbound(pdb, p["ca"], p["cb"])
        if ub is None:
            continue
        try:
            bpc = D.point_cloud(pdb)
            if bpc is None or len(set(bpc["ch"])) < 2:
                continue
            bg = D.precompute_graph(bpc)
            bpos, bneg = D.interface_point_pairs(bpc)
            if len(bpos) < 12:
                continue
        except Exception:
            continue
        A_, B_ = ub[p["ca"]], ub[p["cb"]]
        upos, uneg = cross_pairs(A_["pc"]["pts"], B_["pc"]["pts"])
        if len(upos) < 12 or len(uneg) < 12:
            continue
        pool[pdb] = {"prep": p, "ub": ub, "upos": upos, "uneg": uneg,
                     "bound": {"pc": bpc, "g": bg}, "bpos": bpos, "bneg": bneg}
        if len(pool) % 10 == 0:
            report(f"    {len(pool)} built ({time.time()-tb:.0f}s)")
    report(f"    {len(pool)} complexes usable unbound, in {time.time()-tb:.0f}s")

    tst = [p for p in test_pdbs if p in pool]
    rest = [p for p in sorted(pool) if p not in test_pdbs]
    ev = rest[:N_EVAL]
    tr = rest[N_EVAL:N_EVAL + N_TRAIN]
    if len(tr) < 12 or len(ev) < 5 or len(tst) < 5:
        report(f"\n  not enough: train {len(tr)} eval {len(ev)} test {len(tst)}")
        return 1
    report(f"    train {len(tr)} | AUC-eval {len(ev)} | docking test {len(tst)}: {tst}")

    # ---------------- train the two models: bound graph vs unbound graph ----------------
    FD = pool[tr[0]]["ub"][pool[tr[0]]["prep"]["ca"]]["pc"]["feat"].shape[1]

    def train(mode, seed=0):
        torch.manual_seed(seed)
        model = Net(FD)
        scale = nn.Parameter(torch.tensor(4.0))
        opt = torch.optim.Adam(list(model.parameters()) + [scale], lr=3e-3)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(EPOCHS):
            np.random.shuffle(tr)
            model.train()
            for pdb in tr:
                c = pool[pdb]
                opt.zero_grad()
                if mode == "bound":
                    e = emb(model, c["bound"])
                    P = np.array(c["bpos"]); Q = np.array(c["bneg"])
                    if not len(P) or not len(Q):
                        continue
                    sp = (e[P[:, 0]] * e[P[:, 1]]).sum(1) * scale
                    sn = (e[Q[:, 0]] * e[Q[:, 1]]).sum(1) * scale
                else:
                    pr = c["prep"]
                    ea = emb(model, c["ub"][pr["ca"]]); eb = emb(model, c["ub"][pr["cb"]])
                    P = np.array(c["upos"]); Q = np.array(c["uneg"])
                    if not len(P) or not len(Q):
                        continue
                    sp = (ea[P[:, 0]] * eb[P[:, 1]]).sum(1) * scale
                    sn = (ea[Q[:, 0]] * eb[Q[:, 1]]).sum(1) * scale
                loss = lf(torch.cat([sp, sn]),
                          torch.cat([torch.ones(len(sp)), torch.zeros(len(sn))]))
                loss.backward()
                opt.step()
        return model, scale

    def auc_of(model, scale, mode, pdbs):
        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for pdb in pdbs:
                c = pool[pdb]
                if mode == "bound":
                    e = emb(model, c["bound"])
                    P = np.array(c["bpos"]); Q = np.array(c["bneg"])
                    if not len(P) or not len(Q):
                        continue
                    sp = (e[P[:, 0]] * e[P[:, 1]]).sum(1) * scale
                    sn = (e[Q[:, 0]] * e[Q[:, 1]]).sum(1) * scale
                else:
                    pr = c["prep"]
                    ea = emb(model, c["ub"][pr["ca"]]); eb = emb(model, c["ub"][pr["cb"]])
                    P = np.array(c["upos"]); Q = np.array(c["uneg"])
                    if not len(P) or not len(Q):
                        continue
                    sp = (ea[P[:, 0]] * eb[P[:, 1]]).sum(1) * scale
                    sn = (ea[Q[:, 0]] * eb[Q[:, 1]]).sum(1) * scale
                ss += sp.tolist() + sn.tolist()
                ys += [1] * len(sp) + [0] * len(sn)
        return float(roc_auc_score(ys, ss)) if len(set(ys)) > 1 else float("nan")

    report(f"\n  TRAINING both models on the same {len(tr)} complexes, {EPOCHS} epochs each")
    t1 = time.time()
    mb, sb = train("bound")
    ab = auc_of(mb, sb, "bound", ev)
    report(f"    bound   graph (whole complex, cross-chain KNN): AUC {ab:.4f}   {time.time()-t1:.0f}s")
    t1 = time.time()
    mu, su = train("unbound")
    au = auc_of(mu, su, "unbound", ev)
    report(f"    unbound graph (per chain, own APBS + KNN)     : AUC {au:.4f}   {time.time()-t1:.0f}s")
    report(f"    THE LEAK = {ab-au:+.4f}   (evaluated on the same {len(ev)} held-out complexes)")

    torch.manual_seed(1234)
    mrand, srand = Net(FD), torch.tensor(4.0)
    ar = auc_of(mrand, srand, "unbound", ev)
    report(f"    UNTRAINED control (random init, unbound)      : AUC {ar:.4f}")

    # ---------------- re-rank the docking poses ----------------
    report(f"\n  RE-RANKING the FFT pose sets of {len(tst)} complexes with the UNBOUND model")
    rots = DF.rotations(DF.N_ROT)
    CLASH_W = dockj["clash_w_chosen"]
    rows = []
    for pdb in tst:
        c = pool[pdb]
        p = c["prep"]
        centre, L, lcen = p["centre"], p["L"], p["lcen"]
        Lc = L - lcen
        FA = p["F_skin"] - CLASH_W * p["F_core"]

        # regenerate the identical blind pose set
        best = []
        for ri, Rm in enumerate(rots):
            g = DF.rasterise(Lc @ Rm.T + centre, centre, p["offs_L"])
            if g is None:
                continue
            sc = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), p["shape"])
            for pk in DF.nms_peaks(sc, DF.PEAKS_PER_ROT, DF.NMS_CELLS):
                if sc[pk] > 0:
                    best.append((float(sc[pk]), ri, DF.shift_of(pk)))
        if not best:
            continue

        Pr = c["ub"][p["ca"]]["pc"]["pts"]                  # receptor surface points, fixed
        Pl = c["ub"][p["cb"]]["pc"]["pts"] - lcen           # ligand surface, centred like the atoms
        with torch.no_grad():
            ea = emb(mu, c["ub"][p["ca"]]).numpy()
            eb = emb(mu, c["ub"][p["cb"]]).numpy()
            ra = emb(mrand, c["ub"][p["ca"]]).numpy()
            rb = emb(mrand, c["ub"][p["cb"]]).numpy()
        tr_recep = cKDTree(Pr)
        sc_shape, sc_mean, sc_sum, sc_unt, rmsd = [], [], [], [], []
        for v, ri, shift in best:
            Rm = rots[ri]
            lig = Pl @ Rm.T + centre + shift
            nb = tr_recep.query_ball_point(lig, CONTACT)
            ii, jj = [], []
            for j, lst in enumerate(nb):
                for i in lst:
                    ii.append(i); jj.append(j)
            if len(ii) < MIN_PAIRS:
                sc_shape.append(v); sc_mean.append(-1e9); sc_sum.append(-1e9); sc_unt.append(-1e9)
            else:
                ii = np.array(ii); jj = np.array(jj)
                d = (ea[ii] * eb[jj]).sum(1) * float(su)
                pr_ = 1.0 / (1.0 + np.exp(-d))
                du = (ra[ii] * rb[jj]).sum(1) * float(srand)
                sc_shape.append(v)
                sc_mean.append(float(pr_.mean()))
                sc_sum.append(float(pr_.sum()))
                sc_unt.append(float((1.0 / (1.0 + np.exp(-du))).mean()))
            rmsd.append(float(np.sqrt((((Lc @ Rm.T + centre + shift) - L) ** 2).sum(1).mean())))

        rm = np.array(rmsd)
        frac = float((rm <= NEAR_NATIVE).mean())
        z = lambda a: (np.array(a) - np.mean(a)) / (np.std(a) + 1e-9)
        arms = {"shape": np.array(sc_shape), "learned_mean": np.array(sc_mean),
                "learned_sum": np.array(sc_sum), "untrained": np.array(sc_unt),
                "shape+learned": z(sc_shape) + z(sc_mean)}
        # LIVENESS: a ranking arm that is constant cannot rank anything
        deadarms = [k for k, v in arms.items() if float(np.std(v)) < 1e-12]
        hits = {}
        for k, v in arms.items():
            o = np.argsort(-v)
            hits[k] = {str(t): bool((rm[o][:t] <= NEAR_NATIVE).any()) for t in TOPK}
        rows.append({"pdb": pdb, "n_poses": len(rm), "near_native_frac": frac,
                     "best_rmsd": float(rm.min()), "hits": hits, "dead_arms": deadarms,
                     "chance": {str(t): 1 - (1 - frac) ** t for t in TOPK}})
        report(f"    {pdb}: {len(rm)} poses, near-native frac {frac:.3f} | top10 "
               + "  ".join(f"{k} {str(hits[k]['10']):<5}" for k in
                           ("shape", "learned_mean", "learned_sum", "untrained"))
               + (f"  DEAD:{deadarms}" if deadarms else ""))

    if not rows:
        report("\n  nothing re-ranked")
        return 1

    report(f"\n  {len(rows)} complexes re-ranked")
    dead = sorted({d for r in rows for d in r["dead_arms"]})
    if dead:
        report(f"    LIVENESS: {dead} were CONSTANT in at least one complex -- those rows mean nothing.")
    report(f"\n    {'arm':<16}" + "".join(f"{'top-'+str(t):>10}" for t in TOPK))
    tab = {}
    for k in ("shape", "learned_mean", "learned_sum", "shape+learned", "untrained"):
        tab[k] = {str(t): sum(r["hits"][k][str(t)] for r in rows) for t in TOPK}
        report(f"    {k:<16}" + "".join(f"{str(tab[k][str(t)])+'/'+str(len(rows)):>10}" for t in TOPK))
    ch = {str(t): float(np.mean([r["chance"][str(t)] for r in rows])) * len(rows) for t in TOPK}
    report(f"    {'chance':<16}" + "".join(f"{ch[str(t)]:>10.2f}" for t in TOPK))

    report("\n  READING")
    lm, sh, un = tab["learned_mean"]["10"], tab["shape"]["10"], tab["untrained"]["10"]
    if abs(ab - au) > 0.03:
        report(f"  LEAK CONFIRMED: bound-graph AUC {ab:.4f} vs unbound {au:.4f}, a gap of {ab-au:.4f} on the")
        report("  same held-out complexes. The reported 0.9019 was measured with the KNN graph reaching across")
        report("  the interface, so a point's embedding already saw its partner. The docking-relevant number")
        report(f"  is the unbound one, {au:.4f}. This corrects a result recorded earlier today.")
    else:
        report(f"  No meaningful leak: bound {ab:.4f} vs unbound {au:.4f}. 0.9019 stands as reported.")
    if lm > len(rows) / 2 and lm > sh and lm > un:
        report(f"  And the JOIN WORKS: learned re-ranking puts near-native in the top ten for {lm}/{len(rows)}")
        report(f"  against shape {sh} and an untrained network {un}. The 0.90 predictor supplies the")
        report("  orientation signal the FFT search was missing.")
    elif lm > 0 and lm <= un:
        report(f"  The join does NOT work: learned {lm}/{len(rows)} vs UNTRAINED {un}/{len(rows)}. Whatever")
        report("  ranking there is comes from the geometry of the embedding, not from what was learned.")
    else:
        report(f"  The join FAILS: learned {lm}/{len(rows)}, shape {sh}, untrained {un}, chance "
               f"{ch['10']:.2f}. A model that discriminates interface points does not thereby rank poses --")
        report("  scoring contacts of a GIVEN pose and choosing among poses are different problems.")

    json.dump({"test": "dock_rescore", "n_train": len(tr), "n_eval": len(ev), "epochs": EPOCHS,
               "auc_bound_graph": ab, "auc_unbound_graph": au, "auc_untrained": ar, "leak": ab - au,
               "clash_w": CLASH_W, "contact_A": CONTACT, "complexes": rows, "arms": tab,
               "chance": ch, "n_reranked": len(rows), "test_pdbs": tst, "log": log},
              open(OUT / "dock_rescore.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_rescore.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
