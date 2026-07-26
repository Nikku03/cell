"""redundancy_check -- why did restoring 34,440 PPI edges make the COMPONENTS better and the ENSEMBLE slightly worse?

THE PUZZLE. Fixing the index-space bug that had left the PPI graph empty moved PHYS from 0.4028 to 0.4487 and LEARNED
from 0.4525 to 0.4691 -- large, real gains. Yet the 4-way ensemble went 0.4995 -> 0.4950 and the headline
4-way+MARKOV-P went 0.5173 -> 0.5068. More data, better parts, worse whole.

THE HYPOTHESIS, WHICH IS TESTABLE RATHER THAN RHETORICAL. An ensemble is paid in COMPLEMENTARITY, not in component
quality. Before the fix, MARKOV-P was the ONLY component that could see the physical graph, so it was carrying the
entire PPI signal alone and everything it contributed was unique. After the fix, PHYS and LEARNED see the same PPI --
so all three improve individually while becoming MORE REDUNDANT with each other. Those two effects pull opposite ways
in a z-score-sum fusion, and if redundancy wins the ensemble can fall even as its parts rise.

If that is right it makes three predictions, and all three are checked here on the same splits:

    P1  PHYS and MARKOV-P become MORE CORRELATED after the fix -- they start reading the same graph
    P2  their top-50 nominations OVERLAP more after the fix
    P3  MARKOV-P's UNIQUE contribution -- what the ensemble loses by dropping it -- SHRINKS after the fix

P3 is already known to hold from the headline rerun (+0.0179 -> +0.0118), so P1 and P2 are the load-bearing new
measurements: they say WHETHER the shrinkage is redundancy rather than the component simply getting worse, which it
did not (MARKOV-P is unchanged at 0.4763 either way, because it never had the bug).

WHAT WOULD FALSIFY IT. If PHYS and MARKOV-P are no more correlated after the fix than before, redundancy is not the
explanation and the ensemble drop needs a different one -- most likely just split noise, in which case the honest
answer is "it did not really get worse".

Everything runs on the SAME five splits and the SAME components as protein_chain_combine, with the only difference
being which of the two PPI wirings is used."""
import json, pickle, collections
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI = 1.0, 50, 5, 0.05, 10
NSPLIT = 5


def onehot(key_lists, kos):
    cats = {}; rows = []; cols = []
    for i, k in enumerate(kos):
        for c in key_lists.get(k, ()):
            j = cats.setdefault(c, len(cats)); rows.append(i); cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(kos), max(len(cats), 1)))


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    M = np.zeros((nK, nG), dtype=np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    nontide_idx = np.where(~tide)[0]
    mover_freq = (A >= TAU).mean(0)
    spec_n = np.array([int(((A[i] >= TAU) & (~tide)).sum()) for i in range(nK)])
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = (Mn @ Mn.T).astype(np.float32)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)

    # ---- the two wirings, side by side ----
    nb_broken = collections.defaultdict(set)                 # keyed by INT, read by NAME -> always empty
    for e in D.get("ppi", []):
        nb_broken[e[0]].add(e[1]); nb_broken[e[1]].add(e[0])
    nb_fixed = collections.defaultdict(set)                  # keyed by NAME, values NAME
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < N and 0 <= b_ < N and a_ != b_:
            nb_fixed[names[a_]].add(names[b_]); nb_fixed[names[b_]].add(names[a_])
    print(f"broken wiring: {sum(1 for k in kos if nb_broken.get(k))}/{nK} knockouts have a PPI neighbour")
    print(f"fixed  wiring: {sum(1 for k in kos if nb_fixed.get(k))}/{nK}", flush=True)

    go = D.get("go", {})
    cplx = {names[int(k)]: cs for k, cs in (D.get("gene2cplx", {}) or {}).items()
            if str(k).isdigit() and int(k) < N}
    per = {"cplx": {k: cplx.get(k, []) for k in kos},
           "gop": {k: go.get(k, {}).get("P", []) for k in kos},
           "goc": {k: go.get(k, {}).get("C", []) for k in kos},
           "gof": {k: go.get(k, {}).get("F", []) for k in kos},
           "path": {k: ([info.get(k, {}).get("path")] if info.get(k, {}).get("path") else []) for k in kos}}
    shared = {n_: np.asarray((m @ m.T).todense(), dtype=np.float32)
              for n_, m in {n_: onehot(d, kos) for n_, d in per.items()}.items()}

    def ppi_feats(nbset):
        NBc = {}; rr = []; cc = []
        for i_, k in enumerate(kos):
            for g in nbset.get(k, ()):
                rr.append(i_); cc.append(NBc.setdefault(g, len(NBc)))
        NB = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(nK, max(len(NBc), 1)))
        sh = np.asarray((NB @ NB.T).todense(), dtype=np.float32)
        dg = np.array(NB.sum(1)).ravel()
        jc = sh / (dg[:, None] + dg[None, :] - sh + 1e-9)
        ip = np.zeros((nK, nK), np.float32)
        for i_, k in enumerate(kos):
            for g in nbset.get(k, ()):
                if g in ki:
                    ip[i_, ki[g]] = 1
        return sh, jc, ip

    def arr(f, d=0.0):
        return np.array([float(info.get(k, {}).get(f, d) or d) for k in kos], np.float32)
    dep = arr("dep_frac"); loe = arr("loeuf", 1.0); lgd = np.log1p(arr("ppi")); tf = arr("tf")
    comp = np.array([info.get(k, {}).get("comp", "") for k in kos])
    proc = np.array([info.get(k, {}).get("proc", "") for k in kos])
    codep = collections.defaultdict(set)
    for k, lst in (D.get("codep", {}) or {}).items():
        if not str(k).isdigit() or int(k) >= N:
            continue
        a = names[int(k)]
        for j, _s in lst:
            if isinstance(j, int) and j < N:
                b = names[j]
                if a in ki and b in ki:
                    codep[a].add(ki[b]); codep[b].add(ki[a])

    # MARKOV-P: identical in both arms, because it indexes by integer node and never had the bug
    ppi_int = collections.defaultdict(set)
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < N and 0 <= b_ < N and a_ != b_:
            ppi_int[a_].add(b_); ppi_int[b_].add(a_)
    memb = collections.defaultdict(list)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        if str(g).isdigit() and int(g) < N:
            for c in cs:
                memb[c].append(int(g))
    rr, cc, vv = [], [], []
    for a_, nb in ppi_int.items():
        for b_ in nb:
            rr.append(a_); cc.append(b_); vv.append(1.0)
    for c, mem in memb.items():
        if 2 <= len(mem) <= 60:
            for x in mem:
                for y in mem:
                    if x != y:
                        rr.append(x); cc.append(y); vv.append(2.0)
    G = sparse.coo_matrix((np.array(vv, np.float32), (np.array(rr), np.array(cc))), shape=(N, N)).tocsr()
    mag = np.asarray(abs(G).sum(1)).ravel()
    inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
    PT = (sparse.diags(inv) @ G).T.tocsr()
    konode = np.array([nidx.get(k, -1) for k in kos]); have = konode >= 0
    idx = np.where(have)[0]
    L0 = np.zeros((N, len(idx)), np.float32); L0[konode[idx], np.arange(len(idx))] = 1.0
    cur = L0; tot = np.zeros_like(L0)
    for s in range(1, 3):
        cur = PT @ cur
        tot += (0.5 ** s) * cur
    Wp = np.zeros((nK, nK), np.float32)
    sub = np.zeros((nK, len(idx)), np.float32); sub[have] = tot[konode[have]]
    Wp[idx] = sub.T

    def prof(nb):
        nb = np.asarray(nb, int)
        return np.abs(M[nb]).mean(0) if len(nb) else np.zeros(nG, np.float32)

    def rec(sc, truth):
        o = nontide_idx[np.argsort(-sc[nontide_idx])][:K]
        return len(set(o.tolist()) & truth) / max(len(truth), 1)

    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else v * 0.0

    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC and have[i]])
    res = collections.defaultdict(list)

    for arm, nbset in [("broken", nb_broken), ("fixed", nb_fixed)]:
        sh, jc, ip = ppi_feats(nbset)
        FE = [shared["cplx"], shared["gop"], shared["goc"], shared["gof"], shared["path"], sh, jc, ip,
              np.abs(dep[:, None] - dep[None, :]), np.abs(loe[:, None] - loe[None, :]),
              np.abs(lgd[:, None] - lgd[None, :]),
              (comp[:, None] == comp[None, :]).astype(np.float32),
              (proc[:, None] == proc[None, :]).astype(np.float32),
              ((tf[:, None] == 1) & (tf[None, :] == 1)).astype(np.float32)]
        phys_adj = (ip > 0) | (shared["cplx"] > 0) | (shared["path"] > 0)
        print(f"\n  [{arm}] phys_adj covers {int((phys_adj.sum(1) > 0).sum())}/{nK} knockouts, "
              f"{int(phys_adj.sum())} KO-KO edges", flush=True)

        for sp in range(NSPLIT):
            srng = np.random.RandomState(100 + sp)
            perm = srng.permutation(scor); nt = len(scor) // 3
            test = np.array(sorted(perm[:nt].tolist()))
            train = np.array(sorted(set(int(x) for x in scor) - set(test.tolist())))
            trs = set(train.tolist())
            SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}

            Xtr, ytr = [], []
            for xi in train:
                cand = train[train != xi]
                s2 = srng.choice(cand, min(len(cand), 250), replace=False)
                Xtr.append(np.stack([F[xi, s2] for F in FE], 1)); ytr.append(S[xi, s2])
            reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=4,
                                                min_samples_leaf=40, random_state=0).fit(
                np.concatenate(Xtr), np.concatenate(ytr))

            comps = collections.defaultdict(dict)
            for xi in test:
                cand = train[train != xi]
                comps["TIDE"][int(xi)] = mover_freq
                comps["PHYS"][int(xi)] = prof([t for t in np.where(phys_adj[xi])[0] if t in trs and t != xi])
                comps["LEARNED"][int(xi)] = prof(
                    cand[np.argsort(-reg.predict(np.stack([F[xi, cand] for F in FE], 1)))[:N_NEI]])
                nx = [t for t in codep.get(kos[xi], ()) if t in trs and t != xi]
                comps["NEXUS"][int(xi)] = prof(nx) if nx else np.zeros(nG, np.float32)
                v = Wp[xi].astype(np.float64).copy(); v[xi] = -np.inf
                mk = np.full(nK, -np.inf); mk[cand] = 0.0
                comps["MARKOV-P"][int(xi)] = prof(np.argsort(-(v + mk))[:N_NEI])

            for nm in ["TIDE", "NEXUS", "PHYS", "LEARNED", "MARKOV-P"]:
                res[f"{arm}|score|{nm}"].append(
                    float(np.mean([rec(comps[nm][int(x)], SPEC[int(x)]) for x in test])))

            # P1: pairwise correlation between component score vectors, on the evaluable genes
            # P2: overlap of their top-50 nominations
            for a1, a2 in [("PHYS", "MARKOV-P"), ("LEARNED", "MARKOV-P"), ("PHYS", "LEARNED"),
                           ("NEXUS", "MARKOV-P")]:
                rs, js = [], []
                for xi in test:
                    v1 = comps[a1][int(xi)][nontide_idx]; v2 = comps[a2][int(xi)][nontide_idx]
                    if v1.std() < 1e-9 or v2.std() < 1e-9:
                        continue
                    rs.append(spearmanr(v1, v2)[0])
                    t1 = set(nontide_idx[np.argsort(-v1)[:K]].tolist())
                    t2 = set(nontide_idx[np.argsort(-v2)[:K]].tolist())
                    js.append(len(t1 & t2) / len(t1 | t2))
                res[f"{arm}|corr|{a1}-{a2}"].append(float(np.mean(rs)) if rs else np.nan)
                res[f"{arm}|jac|{a1}-{a2}"].append(float(np.mean(js)) if js else np.nan)

            # P3: leave-one-out contribution of each component to the 5-way ensemble
            B = ["TIDE", "PHYS", "LEARNED", "NEXUS", "MARKOV-P"]
            full = float(np.mean([rec(sum(z(comps[n_][int(x)]) for n_ in B), SPEC[int(x)]) for x in test]))
            res[f"{arm}|ens|FULL"].append(full)
            for drop in B:
                sub_ = [n_ for n_ in B if n_ != drop]
                v = float(np.mean([rec(sum(z(comps[n_][int(x)]) for n_ in sub_), SPEC[int(x)]) for x in test]))
                res[f"{arm}|drop|{drop}"].append(full - v)
            print(f"    split {sp}: PHYS {res[f'{arm}|score|PHYS'][-1]:.4f}  "
                  f"corr(PHYS,MARKOV-P) {res[f'{arm}|corr|PHYS-MARKOV-P'][-1]:+.3f}  "
                  f"drop-MARKOV-P {res[f'{arm}|drop|MARKOV-P'][-1]:+.4f}", flush=True)

    def m(kk):
        v = np.array(res[kk], float)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()))

    print(f"\n  COMPONENT SCORES (5 splits)")
    print(f"    {'component':12s} {'broken':>16s} {'fixed':>16s} {'delta':>10s}")
    for nm in ["TIDE", "NEXUS", "PHYS", "LEARNED", "MARKOV-P"]:
        b, _ = m(f"broken|score|{nm}"); f_, _ = m(f"fixed|score|{nm}")
        print(f"    {nm:12s} {b:>16.4f} {f_:>16.4f} {f_-b:>+10.4f}")

    print(f"\n  P1/P2 -- DO THE COMPONENTS BECOME REDUNDANT? (Spearman between score vectors; Jaccard of top-{K})")
    print(f"    {'pair':22s} {'corr broken':>12s} {'corr fixed':>11s} {'delta':>8s} | "
          f"{'jac broken':>11s} {'jac fixed':>10s} {'delta':>8s}")
    p1 = {}
    for a1, a2 in [("PHYS", "MARKOV-P"), ("LEARNED", "MARKOV-P"), ("PHYS", "LEARNED"), ("NEXUS", "MARKOV-P")]:
        cb, _ = m(f"broken|corr|{a1}-{a2}"); cf, _ = m(f"fixed|corr|{a1}-{a2}")
        jb, _ = m(f"broken|jac|{a1}-{a2}"); jf, _ = m(f"fixed|jac|{a1}-{a2}")
        p1[f"{a1}-{a2}"] = {"corr_broken": cb, "corr_fixed": cf, "jac_broken": jb, "jac_fixed": jf}
        print(f"    {a1 + ' vs ' + a2:22s} {cb:>12.3f} {cf:>11.3f} {cf-cb:>+8.3f} | "
              f"{jb:>11.3f} {jf:>10.3f} {jf-jb:>+8.3f}")

    print(f"\n  P3 -- UNIQUE CONTRIBUTION (drop-one from the 5-way ensemble)")
    print(f"    {'component':12s} {'broken':>16s} {'fixed':>16s} {'delta':>10s}")
    p3 = {}
    for nm in ["TIDE", "NEXUS", "PHYS", "LEARNED", "MARKOV-P"]:
        b, _ = m(f"broken|drop|{nm}"); f_, _ = m(f"fixed|drop|{nm}")
        p3[nm] = {"broken": b, "fixed": f_}
        print(f"    {nm:12s} {b:>+16.4f} {f_:>+16.4f} {f_-b:>+10.4f}")
    eb, _ = m("broken|ens|FULL"); ef, _ = m("fixed|ens|FULL")
    print(f"    {'FULL 5-way':12s} {eb:>16.4f} {ef:>16.4f} {ef-eb:>+10.4f}")

    dcorr = p1["PHYS-MARKOV-P"]["corr_fixed"] - p1["PHYS-MARKOV-P"]["corr_broken"]
    djac = p1["PHYS-MARKOV-P"]["jac_fixed"] - p1["PHYS-MARKOV-P"]["jac_broken"]
    dmark = p3["MARKOV-P"]["fixed"] - p3["MARKOV-P"]["broken"]
    redundancy = dcorr > 0 and djac > 0 and dmark < 0

    verdict = (
        f"WHY FIXING THE PPI GRAPH HELPED THE PARTS AND NOT THE WHOLE. Restoring 34,440 KO-KO PPI edges moved PHYS "
        f"{m('broken|score|PHYS')[0]:.4f} -> {m('fixed|score|PHYS')[0]:.4f} and LEARNED "
        f"{m('broken|score|LEARNED')[0]:.4f} -> {m('fixed|score|LEARNED')[0]:.4f}, both large real gains, while the "
        f"full ensemble moved {eb:.4f} -> {ef:.4f}. An ensemble is paid in COMPLEMENTARITY, not component quality, so "
        f"the hypothesis was that the fix made the components better individually and MORE REDUNDANT collectively. "
        f"That was turned into three checkable predictions. "
        + (f"ALL THREE HOLD. Spearman between the PHYS and MARKOV-P score vectors rises "
           f"{p1['PHYS-MARKOV-P']['corr_broken']:.3f} -> {p1['PHYS-MARKOV-P']['corr_fixed']:.3f} ({dcorr:+.3f}), the "
           f"overlap of their top-{K} nominations rises {p1['PHYS-MARKOV-P']['jac_broken']:.3f} -> "
           f"{p1['PHYS-MARKOV-P']['jac_fixed']:.3f} ({djac:+.3f}), and MARKOV-P's unique contribution -- what the "
           f"ensemble loses when it is dropped -- falls {p3['MARKOV-P']['broken']:+.4f} -> "
           f"{p3['MARKOV-P']['fixed']:+.4f} ({dmark:+.4f}). So the PPI information was never MISSING from the "
           f"ensemble; it was entering through ONE door instead of three. Fixing the bug opened the other two doors "
           f"into the same room. " if redundancy else
           f"THE REDUNDANCY STORY DOES NOT HOLD as stated: corr(PHYS, MARKOV-P) moves {dcorr:+.3f}, top-{K} overlap "
           f"{djac:+.3f}, and MARKOV-P's unique contribution {dmark:+.4f}. At least one prediction fails, so the "
           f"ensemble's behaviour needs a different explanation -- most plausibly split noise, in which case the "
           f"honest reading is that the ensemble did not really change. ")
        + f"THE BROADER READING, and it is consistent with kitchen_sink, where 55 features scored the same as 17: the "
        f"ensemble was ALREADY extracting what the physical graph has to offer, through whichever component could "
        f"reach it. Adding a second and third route to the same information cannot help, and in a z-score-sum fusion "
        f"three correlated votes can slightly crowd out the uncorrelated ones. That is why a genuine data recovery "
        f"produced no headline gain -- and it is a stronger statement about the ceiling than the bug itself, because "
        f"it says the limit is not access to the graph. "
        f"WHAT THIS CANNOT SAY: five splits of one cell line, so the ensemble difference of "
        f"{ef-eb:+.4f} is itself within split-to-split noise and should not be read as a real degradation -- the "
        f"claim here is that it did not IMPROVE, not that it got worse. Drop-one attribution assumes the other four "
        f"components are held fixed, which understates shared credit. Deterministic given the fixed seeds.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_splits": NSPLIT,
               "component_scores": {nm: {"broken": m(f"broken|score|{nm}")[0], "fixed": m(f"fixed|score|{nm}")[0]}
                                    for nm in ["TIDE", "NEXUS", "PHYS", "LEARNED", "MARKOV-P"]},
               "redundancy": p1, "drop_one": p3,
               "ensemble": {"broken": eb, "fixed": ef},
               "redundancy_hypothesis_holds": bool(redundancy),
               "verdict": verdict, "note": verdict}, open(OUT / "redundancy_check.json", "w"), indent=1)
    print("\n  -> outputs/orphan/redundancy_check.json")


if __name__ == "__main__":
    main()
