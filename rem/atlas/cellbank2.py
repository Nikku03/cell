"""The cell memory bank, rebuilt on the reasoning engine -- and, for the first time, scored on data.

WHAT CHANGES (lessons of chainsnap / transvs / biomemory):
  MEMORY, TRANSFORMER-STYLE. Every signed fact (source -> target, sign, layer) stays individually in
    view and is retrieved by content (its source); nothing is compressed into a state. That is the
    lookup that beat every compressed memory in transvs.py and biomemory.py.
  STEP-REPEATING REASONING WITH A SNAP. One regulatory hop per step; after each hop every reached
    gene is snapped to a discrete direction (+1 / -1) and WRITTEN DOWN -- it becomes a source for the
    next hop and is never re-decided (chainsnap.py's written-down pointer).
  VOTES, NOT VETOES. dyncell.py assigned only on UNANIMOUS incoming signs and left disagreement
    unassigned (1,169 genes; its own D4.2 called the rule a choice). Here each fact casts a weighted
    vote, the snap takes the sign of the sum, and only exact ties are contradictions.
  CONFIDENCE DECAYS PER HOP, lambda = 0.863 (biomemory.py's synaptic trace). For ranking only; it
    never changes a sign. Carried over as a convention, not as biology of gene regulation.
  The protein / DNA-binding-domain gates of dyncell.may_act are kept unchanged, and every state
    change is journalled with the facts that voted (replayable).
LEARNED ATTENTION. A content-based relevance per fact, sigma(theta . phi(fact)), from seven fact
  features (bias, signalling layer, positive sign, log source out-degree, log target in-degree, log
  target protein, hop), learnt on TRAINING knockdowns only: R = P(target moves | fact fired),
  G = P(fact's direction is right | target moved). Vote weight = R (2G - 1): a class of facts that is
  usually wrong can flip, from data.
SELF-REVISION FROM DATA. dyncell.py downgraded edges when the network contradicted ITSELF (D4.4:
  "no measurement enters this loop"). Here a fact is downgraded (weight 0) when TRAINING knockdowns
  contradict it more often than they confirm it, at least twice.

THE DATA. K562 genome-wide-screen essential-gene CRISPRi Perturb-seq (Replogle et al., Cell 2022,
doi 10.1016/j.cell.2022.05.013), pseudobulk z-scores, 1,971 knockdowns x 8,563 genes (HuggingFace
mirror nicolas-lynn/replogle-perturb, k562ess; the mirror declares no licence -- aggregates only,
raw file in the gitignored cache). A knockdown seeds its gene at -1. A MOVER is |z| >= 3; the
knocked-down gene itself is never scored. Eligible knockdowns: the gene has >= 1 fact as a source
and its 3-hop reach contains >= 10 measured genes. Split 50/50 by a fixed hash into TRAIN / TEST.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
E0 HARNESS, BLOCKING. CRISPRi must show in the data: median z of each knocked-down gene's OWN
   expression <= -1 (over knockdowns whose gene is measured).
E1 PROVENANCE, BLOCKING. The new bank's journal replays to its state exactly (dyncell's D1).
Per knockdown, per hop h = 1, 2, 3 (genes first reached at hop h):
   precision = movers / measured genes given a direction; base = movers / all measured genes;
   enrichment = precision / base; direction accuracy = predicted sign == sign(z), over movers.
   Units are knockdowns; comparisons are paired two-sided sign tests at p < 0.05; direction
   accuracy needs >= 5 scored movers in that knockdown for both arms.
V1 DOES THE OLD BANK (dyncell rule, 3 hops) PREDICT WHICH GENES MOVE? enrichment > 1 vs < 1 across
   knockdowns, per hop -> PREDICTS / ANTI-PREDICTS / NO SIGNAL.
V2 DOES IT PREDICT DIRECTION? old bank vs the SIGN-SHUFFLED bank (topology identical), paired.
V3 NEW ENGINE vs OLD BANK on TEST knockdowns: direction accuracy (paired), and movers given a
   direction (coverage).
V4 LEARNED ATTENTION vs SYMBOLIC NEW ENGINE on TEST knockdowns: direction accuracy, and AUROC of the
   confidence ranking (movers vs non-movers among genes given a direction), paired.
V5 DATA SELF-REVISION: TEST direction accuracy at hops 2-3 with train-derived downgrades vs without.
V6 WHAT THIS IS NOT. One cell line, one perturbation type, steady state. Direction of a CRISPRi
   response is not a mechanism; a correct sign through a wrong path counts as correct.
"""

from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_cellbank2.txt")
ART = R("outputs", "cellbank2.json")
BANK2 = R("memory_bank", "cell_v2")
PSEQ = os.path.join(HERE, "_cache", "perturbseq", "k562ess_perturbation_zscores.parquet")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("dyncell", os.path.join(HERE, "dyncell.py"))
dc = importlib.util.module_from_spec(_s); _s.loader.exec_module(dc)

LAM, HOPS, ZMOV = 0.863, 3, 3.0


def facts_table(D, names, shuffle=False, seed=20261005):
    rows = [(names[s], names[t], int(np.sign(w)), 0) for s, t, w in D["reg"] if w] + \
           [(names[s], names[t], int(np.sign(w)), 1) for s, t, w in D["sig"] if w]
    if shuffle:
        rng = np.random.default_rng(seed)
        sg = np.array([r[2] for r in rows]); rng.shuffle(sg)
        rows = [(a, b, int(c), l) for (a, b, _, l), c in zip(rows, sg)]
    by_src = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r[0]].append(i)
    outdeg = collections.Counter(r[0] for r in rows); indeg = collections.Counter(r[1] for r in rows)
    return rows, by_src, outdeg, indeg


def features(rows, outdeg, indeg, protein, hop_of=None):
    F = np.zeros((len(rows), 7))
    for i, (s, t, sg, l) in enumerate(rows):
        F[i] = (1.0, l, 1.0 if sg > 0 else 0.0, math.log1p(outdeg[s]), math.log1p(indeg[t]),
                math.log1p(protein.get(t, 0.0)), 0.0)
    return F


def old_bank(C, rows, by_src, seeds):
    """dyncell's rule over the same facts, 3 waves: unanimous assigns, disagreement = conflict."""
    E = collections.defaultdict(list)
    for s, t, sg, l in rows:
        E[s].append((t, sg))
    dc.propagate(C, E, seeds, max_waves=HOPS)
    hop = {}
    for e in C.journal:
        if e["op"] == "set" and e["rule"].startswith("propagate_wave_"):
            hop[e["gene"]] = int(e["rule"].rsplit("_", 1)[1]) + 1
    return {g: (C.dir[g], hop[g], 1.0) for g in hop}


def new_engine(C, rows, by_src, seeds, weight=None, revised=None, journal=True):
    """Transformer-style fact memory + one hop per step + snap. weight: per-fact vote weight
    (None = 1). revised: set of fact indices downgraded to 0. Returns gene -> (dir, hop, conf)."""
    for g, d in seeds.items():
        C.set(g, d, "perturbation", [])
    C.voters = {}
    frontier = set(seeds)
    res = {}
    for hop in range(1, HOPS + 1):
        votes = collections.defaultdict(float); why = collections.defaultdict(list)
        for src in frontier:
            ok, _ = dc.may_act(C, src)
            if not ok:
                continue
            sd = C.dir.get(src)
            for i in by_src.get(src, ()):
                t = rows[i][1]
                if t in C.dir or t in C.conflict:
                    continue
                w = 1.0 if weight is None else weight[hop][i]
                if revised is not None and i in revised:
                    w = 0.0
                if w == 0.0:
                    continue
                votes[t] += sd * rows[i][2] * w
                why[t].append(i)
        nxt = set()
        for t, v in votes.items():
            if v == 0:
                C.flag_conflict(t, [(rows[i][0], rows[i][2]) for i in why[t]], f"tie_hop_{hop}")
                continue
            d = 1 if v > 0 else -1
            C.set(t, d, f"vote_hop_{hop}", [rows[i][0] for i in why[t][:6]])
            C.voters[t] = list(why[t])
            res[t] = (d, hop, abs(v) * LAM ** hop)
            nxt.add(t)
        if not nxt:
            break
        frontier = nxt
    return res


def score(pred, z, genes_idx, ko):
    """Per hop: n given a direction & measured, movers among them, base rate, direction hits."""
    zk = z
    meas = np.isfinite(zk)
    mov_all = (np.abs(zk) >= ZMOV) & meas
    if ko in genes_idx:
        mov_all[genes_idx[ko]] = False
    n_meas = int(meas.sum()) - (1 if ko in genes_idx else 0)
    base = mov_all.sum() / max(n_meas, 1)
    out = {}
    for h in range(1, HOPS + 1):
        n = m = hit = 0
        for g, (d, hop, c) in pred.items():
            if hop != h or g == ko or g not in genes_idx:
                continue
            j = genes_idx[g]
            n += 1
            if abs(zk[j]) >= ZMOV:
                m += 1
                hit += int(np.sign(zk[j]) == d)
        out[h] = dict(n=n, movers=m, hits=hit, base=float(base))
    return out


def auroc(pred, z, genes_idx, ko):
    s, y = [], []
    for g, (d, hop, c) in pred.items():
        if g == ko or g not in genes_idx:
            continue
        s.append(c); y.append(abs(z[genes_idx[g]]) >= ZMOV)
    s = np.array(s); y = np.array(y)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    for v in np.unique(s):
        idx = s == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))


def logistic(X, y, l2=1.0, iters=200):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / len(y) + l2 * w / len(y)
        Hm = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(X.shape[1]) / len(y)
        w -= np.linalg.solve(Hm, g)
    return w


def sign_test(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d); w = sum(x > 0 for x in d)
    if n == 0:
        return 0, 0, 1.0
    k = min(w, n - w)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return w, n - w, p


def main():
    import pyarrow.parquet as pq
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("THE CELL MEMORY BANK ON THE REASONING ENGINE, SCORED ON K562 PERTURB-SEQ"); P_(RULE)
    D = json.load(gzip.open(dc.ENCY)); names = [r["name"] for r in D["genes"]]
    protein = {names[int(k)]: v for k, v in D["ppm"].items() if k.isdigit()}
    dbd_dead = {}
    ij = R("outputs", "isoform_edge_bounds.json")
    if os.path.exists(ij):
        for g, v in json.load(open(ij)).items():
            dbd_dead[g] = len(v.get("dbd_disrupted", [])) / max(v.get("n_isoforms") or 1, 1)
    rows, by_src, outdeg, indeg = facts_table(D, names)
    rows_sh, by_src_sh, _, _ = facts_table(D, names, shuffle=True)
    P_(f"  facts in memory {len(rows):,} (regulatory {sum(r[3] == 0 for r in rows):,}, signalling {sum(r[3] == 1 for r in rows):,}) "
       f"from {len(by_src):,} sources")
    tab = pq.read_table(PSEQ).to_pandas()
    kos = list(tab.index); gcols = [c for c in tab.columns if c != "__index_level_0__"]
    Z = tab[gcols].values.astype(float); gidx = {g: j for j, g in enumerate(gcols)}; kidx = {k: i for i, k in enumerate(kos)}
    own = [Z[kidx[k], gidx[k]] for k in kos if k in gidx]
    P_(f"  Perturb-seq: {len(kos):,} knockdowns x {len(gcols):,} genes; knocked-down gene measured in {len(own):,}")
    e0 = float(np.median(own)) <= -1
    P_(f"  E0 CRISPRi visible: median own-gene z {np.median(own):+.2f} -> {'PASS' if e0 else 'FAIL'}")
    if not e0:
        open(OUT, "w").write("\n".join(out) + "\n"); return

    mk = lambda: dc.Cell(names, protein, {}, dbd_dead, {})
    elig = []
    lim = int(os.environ.get("CELLBANK2_SMOKE", "0"))
    for k in (kos[:lim * 20] if lim else kos):
        if k not in by_src:
            continue
        C = mk(); pr = new_engine(C, rows, by_src, {k: -1})
        if sum(1 for g in pr if g in gidx and g != k) >= 10:
            elig.append(k)
    split = {k: ("TRAIN" if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 2 == 0 else "TEST") for k in elig}
    train = [k for k in elig if split[k] == "TRAIN"]; test = [k for k in elig if split[k] == "TEST"]
    P_(f"  eligible knockdowns {len(elig)} -> TRAIN {len(train)}, TEST {len(test)}")

    # ---- E1 provenance on the new engine -------------------------------------------------------
    Cx = mk(); prx = new_engine(Cx, rows, by_src, {g: -1 for g in ("TP53", "NR3C1", "RELA", "MYC", "STAT3")})
    e1 = Cx.replay() == Cx.dir and all(e.get("rule") for e in Cx.journal if e["op"] == "set")
    P_(f"  E1 new bank journal replays exactly (5-regulator knockdown, {len(Cx.dir):,} genes set, "
       f"{len(Cx.conflict):,} ties): {'PASS' if e1 else 'FAIL'}")
    if not e1:
        open(OUT, "w").write("\n".join(out) + "\n"); return

    def run_all(kind, ks, **kw):
        res = {}
        for k in ks:
            C = mk()
            if kind == "old":
                pr = old_bank(C, rows, by_src, {k: -1})
            elif kind == "old_shuffled":
                pr = old_bank(C, rows_sh, by_src_sh, {k: -1})
            else:
                pr = new_engine(C, rows, by_src, {k: -1}, **kw)
            res[k] = dict(score=score(pr, Z[kidx[k]], gidx, k), auc=auroc(pr, Z[kidx[k]], gidx, k), pred=pr)
        return res

    def acc(r, hops=(1, 2, 3)):
        m = sum(r["score"][h]["movers"] for h in hops); h_ = sum(r["score"][h]["hits"] for h in hops)
        return (h_ / m if m >= 5 else None), m

    old = run_all("old", elig); oldsh = run_all("old_shuffled", elig)
    P_("\n" + RULE); P_("V1/V2  THE OLD BANK (dyncell rule) ON ALL ELIGIBLE KNOCKDOWNS"); P_(RULE)
    res = {}
    for h in range(1, HOPS + 1):
        enr = [r["score"][h]["movers"] / r["score"][h]["n"] / r["score"][h]["base"] - 1
               for r in old.values() if r["score"][h]["n"] >= 5 and r["score"][h]["base"] > 0]
        w, l, p = sign_test(enr)
        pm = sum(r["score"][h]["movers"] for r in old.values()); pn = sum(r["score"][h]["n"] for r in old.values())
        pb = np.mean([r["score"][h]["base"] for r in old.values()])
        v = "PREDICTS" if p < 0.05 and w > l else "ANTI-PREDICTS" if p < 0.05 else "NO SIGNAL"
        res[f"V1_hop{h}"] = dict(enriched=w, depleted=l, p=p, verdict=v, pooled_precision=pm / max(pn, 1), mean_base=float(pb))
        P_(f"  hop {h}: genes given a direction {pn:,}, movers {pm:,} (precision {pm / max(pn, 1):.3f} vs mean base rate {pb:.3f}); "
           f"enriched in {w}, depleted in {l} knockdowns, p {p:.3g} -> WHICH GENES MOVE: {v}")
    da = [(acc(old[k])[0], acc(oldsh[k])[0]) for k in elig]
    da = [(a, b) for a, b in da if a is not None and b is not None]
    w, l, p = sign_test([a - b for a, b in da])
    v = "PREDICTS DIRECTION" if p < 0.05 and w > l else "WORSE THAN SHUFFLED" if p < 0.05 else "NO DIRECTION SIGNAL"
    res["V2"] = dict(n=len(da), real_mean=float(np.mean([a for a, _ in da])) if da else None,
                     shuffled_mean=float(np.mean([b for _, b in da])) if da else None, wins=w, losses=l, p=p, verdict=v)
    P_(f"  V2 direction accuracy, real signs {res['V2']['real_mean']:.3f} vs shuffled signs {res['V2']['shuffled_mean']:.3f} "
       f"over {len(da)} knockdowns; real better in {w}, worse in {l}, p {p:.3g} -> {v}")

    # ---- learned attention (TRAIN only) -------------------------------------------------------
    Fx = features(rows, outdeg, indeg, protein)
    Xs, ym, Xg, yg = [], [], [], []
    for k in train:
        C = mk(); pr = new_engine(C, rows, by_src, {k: -1})
        zk = Z[kidx[k]]
        for e in C.journal:
            if e["op"] != "set" or not e["rule"].startswith("vote_hop_"):
                continue
            t = e["gene"]
            if t == k or t not in gidx:
                continue
            hop = int(e["rule"].rsplit("_", 1)[1])
            zt = zk[gidx[t]]
            for i in C.voters[t]:
                src = rows[i][0]
                pred_dir = C.dir.get(src, 0) * rows[i][2]
                f = Fx[i].copy(); f[6] = hop
                Xs.append(f); ym.append(float(abs(zt) >= ZMOV))
                if abs(zt) >= ZMOV:
                    Xg.append(f); yg.append(float(np.sign(zt) == pred_dir))
    Xs, ym, Xg, yg = map(np.array, (Xs, ym, Xg, yg))
    mu, sd = Xs[:, 1:].mean(0), Xs[:, 1:].std(0) + 1e-9
    nz = lambda X: np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], 1)
    wR = logistic(nz(Xs), ym); wG = logistic(nz(Xg), yg)
    P_("\n" + RULE); P_("LEARNED ATTENTION, fitted on TRAIN knockdowns only"); P_(RULE)
    P_(f"  training facts {len(ym):,} (movers {int(ym.sum()):,}); direction-labelled {len(yg):,} (right {yg.mean():.3f})")
    fn = ["bias", "signalling layer", "positive sign", "log src out-degree", "log tgt in-degree", "log tgt protein", "hop"]
    P_("  feature              relevance R     reliability G")
    for nmf, a, b in zip(fn, wR, wG):
        P_(f"  {nmf:<20} {a:>+10.3f}     {b:>+10.3f}")

    def weights(hop):
        f = Fx.copy(); f[:, 6] = hop
        Rf = 1 / (1 + np.exp(-(nz(f) @ wR))); Gf = 1 / (1 + np.exp(-(nz(f) @ wG)))
        return Rf * (2 * Gf - 1)
    Wl = {h: weights(float(h)) for h in range(1, HOPS + 1)}   # hop-specific vote weights
    flipped = float((Wl[1] < 0).mean())
    P_(f"  facts whose learnt hop-1 vote weight is NEGATIVE (the data says they usually point the wrong way): {flipped:.3f}")

    # ---- data self-revision (TRAIN only) ------------------------------------------------------
    tally = collections.Counter()
    for k in train:
        C = mk(); new_engine(C, rows, by_src, {k: -1})
        zk = Z[kidx[k]]
        for e in C.journal:
            if e["op"] != "set" or not e["rule"].startswith("vote_hop_"):
                continue
            t = e["gene"]
            if t == k or t not in gidx or abs(zk[gidx[t]]) < ZMOV:
                continue
            for i in C.voters[t]:
                ok_ = np.sign(zk[gidx[t]]) == C.dir.get(rows[i][0], 0) * rows[i][2]
                tally[i] += 1 if ok_ else -1
    revised = {i for i, c in tally.items() if c <= -2}
    P_(f"  data self-revision: facts judged on TRAIN {len(tally):,}; downgraded (contradicted >= 2 more than confirmed) {len(revised):,}")

    # ---- TEST comparisons ---------------------------------------------------------------------
    sym = run_all("new", test); lrn = run_all("new", test, weight=Wl); rev = run_all("new", test, revised=revised)
    P_("\n" + RULE); P_(f"V3-V5  ON THE {len(test)} TEST KNOCKDOWNS"); P_(RULE)

    def summary(nm, rr):
        a = [acc(r)[0] for r in rr.values()]; a = [x for x in a if x is not None]
        cov = sum(acc(r)[1] for r in rr.values())
        au = [r["auc"] for r in rr.values() if r["auc"] is not None]
        P_(f"  {nm:<34} direction accuracy {np.mean(a) if a else float('nan'):.3f} (over {len(a)} knockdowns)   movers given a direction {cov:,}   "
           f"AUROC {np.mean(au) if au else float('nan'):.3f}")
        return dict(acc=float(np.mean(a)) if a else None, n=len(a), movers=cov, auroc=float(np.mean(au)) if au else None)

    oldt = {k: old[k] for k in test}
    for nm, rr, key in (("OLD bank (unanimity, no data)", oldt, "old"), ("NEW engine, symbolic votes", sym, "new"),
                        ("NEW engine, learnt attention", lrn, "learned"), ("NEW engine, data-revised facts", rev, "revised")):
        res[f"test_{key}"] = summary(nm, rr)

    def paired(a, b, f):
        pairs = [(f(a[k]), f(b[k])) for k in test]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        w, l, p = sign_test([x - y for x, y in pairs])
        return w, l, p, len(pairs), (float(np.mean([x - y for x, y in pairs])) if pairs else float("nan"))

    for key, a, b, f, nm in (("V3_direction", sym, oldt, lambda r: acc(r)[0], "V3 new vs old, direction accuracy"),
                             ("V3_coverage", sym, oldt, lambda r: acc(r)[1], "V3 new vs old, movers given a direction"),
                             ("V4_direction", lrn, sym, lambda r: acc(r)[0], "V4 learnt vs symbolic, direction accuracy"),
                             ("V4_auroc", lrn, sym, lambda r: r["auc"], "V4 learnt vs symbolic, AUROC"),
                             ("V5_direction_hops23", rev, sym, lambda r: acc(r, (2, 3))[0], "V5 revised vs unrevised, accuracy hops 2-3")):
        w, l, p, n, md = paired(a, b, f)
        v = "BETTER" if p < 0.05 and w > l else "WORSE" if p < 0.05 else "NO DIFFERENCE"
        res[key] = dict(better=w, worse=l, p=p, n=n, mean_diff=md, verdict=v)
        P_(f"  {nm:<46} better {w}, worse {l} of {n}, mean {md:+.3f}, p {p:.3g} -> {v}")

    # ---- write the new bank ---------------------------------------------------------------------
    os.makedirs(BANK2, exist_ok=True)
    json.dump({"schema": "facts kept individually (transformer-style memory); one hop per step; snapped directions; "
                         "weighted votes; learnt attention R*(2G-1); data-revised facts; direction only",
               "learned_attention": {"features": fn, "relevance": wR.tolist(), "reliability": wG.tolist(),
                                     "normalisation_mu": mu.tolist(), "normalisation_sd": sd.tolist()},
               "lambda_per_hop": LAM, "mover_threshold_z": ZMOV,
               "revised_facts": sorted([f"{rows[i][0]}->{rows[i][1]}" for i in revised])[:5000],
               "n_revised": len(revised), "data": "Replogle 2022 K562 essential CRISPRi, TRAIN knockdowns only"},
              open(os.path.join(BANK2, "bank.json"), "w"), indent=1)
    with open(os.path.join(BANK2, "journal_5regulator_knockdown.jsonl"), "w") as fh:
        for e in Cx.journal:
            fh.write(json.dumps(e) + "\n")
    json.dump({"fingerprint": Cx.fingerprint(), "perturbation": {g: -1 for g in ("TP53", "NR3C1", "RELA", "MYC", "STAT3")},
               "n_assigned": len(Cx.dir), "n_ties": len(Cx.conflict), "state": Cx.dir},
              open(os.path.join(BANK2, "state_5regulator_knockdown.json"), "w"))
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"results": res, "eligible": len(elig), "train": train, "test": test,
               "per_ko_test": {k: {"old": acc(oldt[k]), "new": acc(sym[k]), "learned": acc(lrn[k]), "revised": acc(rev[k])} for k in test}},
              open(ART, "w"), indent=1, default=float)
    P_(f"\n  bank written to memory_bank/cell_v2/ ; artifact outputs/cellbank2.json ; runtime {time.time() - t0:.0f}s")
    P_("  V6: one cell line, CRISPRi, steady state; a right sign through a wrong path counts as right.")
    open(OUT, "w").write("\n".join(out) + "\n")


def by_src_lookup(by_src, rows, srcs, t):
    """Fact indices from the listed sources to target t (the facts that voted)."""
    out = []
    for s in set(srcs):
        out += [i for i in by_src.get(s, ()) if rows[i][1] == t]
    return out


if __name__ == "__main__":
    main()
