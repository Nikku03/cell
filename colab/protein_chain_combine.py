"""protein_chain_combine -- put a Markov chain and a Monte Carlo on the PROTEIN chain, and fuse them into the ensemble
that reached 0.470.

WHICH 0.49 / 0.348? THEY ARE DIFFERENT SETUPS AND MUST NOT BE MIXED. The 0.348 figure is program_coldstart in the
UNCENSORED setup (|z|>=4.2, MINSRC=20), whose in-setup tide-null is ~0.11-0.13. The ~0.47-0.49 figure is the harness
setup (TAU=1.0, MIN_SPEC=5), whose tide-null is 0.26 and oracle 0.62. A number from one is meaningless against the
other. EVERYTHING HERE RUNS IN THE HARNESS SETUP, alongside the components that produced 0.470, so every comparison is
like-for-like. The module self-calibrates first: if it cannot reproduce tide ~0.26 and oracle ~0.62 it is not
trustworthy and says so.

WHY THE CHAIN IS BUILT AS A NEIGHBOUR SELECTOR, NOT A DIFFUSION. mechanistic_propagate already ran the obvious version
-- diffuse over the PPI graph and rank genes by diffusion score -- and got 0.055 against a 0.26 tide-null, 5x worse
than predicting nothing, because converging a diffusion concentrates on hubs and the PPI graph is physical binding
while the readout is transcriptional. markov_outofgame then lost to a gene-identity prior by nine sigma doing the same
thing on the regulatory graph. So ranking genes by walk score is a twice-measured dead end and is NOT repeated.
What DOES work in this harness is neighbour transfer: PHYS scores 0.373 by averaging the measured profiles of graph
neighbours. So the chain is used to CHOOSE WHICH TRAIN KNOCKOUTS TO AVERAGE -- a similarity, not a prediction.

WHAT IS GENUINELY NEW: COMPLEX CO-MEMBERSHIP AS A FIRST-CLASS EDGE. multihop_diagnosis measured it at +0.1369
(p=7.70e-04, 213 sources) -- comparable to the direct regulatory edge's +0.1079 and far above the physical-PPI hop-2
effect -- and UPDATES.md states plainly that this recommendation was "not yet built or tested". That is the protein
chain: a knockout removes a subunit, the complex it sits in fails, and everything routed through that complex feels it.
The chain here walks PPI edges and complex co-membership edges with separately weighted, swept mixing.

THE TWO NEW COMPONENTS:
  MARKOV-P  truncated 2-step signed walk on the protein graph from the knocked-out gene; train knockouts scoring
            highest become the behavioural neighbours whose measured profiles are averaged. Truncated because
            multi_hop_chain measured discrimination collapsing 6.3x -> 1.02x by hop 3, and because path multiplicity
            at hop 2 is the one multi-hop effect that survived.
  MC-P      the same walk with the protein graph RESAMPLED each replicate (edge dropout, complex-vs-PPI weight,
            depth), accumulating a soft vote over neighbours instead of a hard top-N. This is where sampling could
            genuinely pay: neighbour SELECTION is a discrete choice that a point estimate makes brittle, unlike the
            monotone probability in mc_outofgame that sampling could not improve.

THE CONTROL THAT DECIDES IT. wall_combine already caught a fusion that looked like a gain and was only generic
z-fusion re-weighting. So each new component is also fused in a SHUFFLED form -- same score distribution, knockout
labels permuted. If shuffled-MARKOV lifts the ensemble as much as real MARKOV, the gain is fusion arithmetic, not the
protein chain, and it is reported as such."""
import json, pickle, collections
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI = 1.0, 50, 5, 0.05, 10
MC_REP = 40
CHAIN_CFG = [(1, 1.0, 0.0), (2, 0.5, 0.0), (2, 0.5, 1.0), (2, 0.5, 2.0), (3, 0.5, 1.0)]  # (steps, damp, cplx_weight)


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
    nontide_idx = np.where(~tide)[0]; mover_freq = (A >= TAU).mean(0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = (Mn @ Mn.T).astype(np.float32)
    print(f"harness setup: {nK} knockouts x {nG} genes; {int(tide.sum())} tide genes", flush=True)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)

    # ---------------- the PROTEIN CHAIN graph: PPI + complex co-membership ----------------
    pa, pb = [], []
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pa.append(a); pb.append(b); pa.append(b); pb.append(a)     # PPI is undirected
    ca, cb = [], []
    memb = collections.defaultdict(list)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        if g.isdigit() and int(g) < N:
            for c in cs:
                memb[c].append(int(g))
    for c, mem in memb.items():
        if 2 <= len(mem) <= 60:            # skip mega-"complexes"; they are annotation buckets, not machines
            for x in mem:
                for y in mem:
                    if x != y:
                        ca.append(x); cb.append(y)
    print(f"protein chain: {len(pa)//2} PPI edges, {len(ca)} complex co-membership edges "
          f"from {len(memb)} complexes", flush=True)

    def build_protein_P(cw, keep_ppi=None, keep_cplx=None):
        """Row-normalised protein-chain operator. cw = weight on complex co-membership relative to PPI."""
        ra = np.array(pa if keep_ppi is None else np.array(pa)[keep_ppi])
        rb = np.array(pb if keep_ppi is None else np.array(pb)[keep_ppi])
        v = np.ones(len(ra), np.float32)
        if cw > 0 and ca:
            xa = np.array(ca if keep_cplx is None else np.array(ca)[keep_cplx])
            xb = np.array(cb if keep_cplx is None else np.array(cb)[keep_cplx])
            ra = np.concatenate([ra, xa]); rb = np.concatenate([rb, xb])
            v = np.concatenate([v, np.full(len(xa), cw, np.float32)])
        G = sparse.coo_matrix((v, (ra, rb)), shape=(N, N)).tocsr()
        mag = np.asarray(abs(G).sum(1)).ravel()
        inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        return (sparse.diags(inv) @ G).T.tocsr()                        # transposed for L <- P^T L

    konode = np.array([nidx.get(k, -1) for k in kos])
    have_node = konode >= 0
    print(f"{int(have_node.sum())}/{nK} knockouts are nodes in the protein chain", flush=True)

    def chain_ko_scores(PT, steps, damp, seed_rows):
        """Walk from each knockout's own protein node; return score AT EVERY KNOCKOUT's node (nK x nK)."""
        idx = [i for i in seed_rows if have_node[i]]
        L0 = np.zeros((N, len(idx)), np.float32)
        L0[konode[idx], np.arange(len(idx))] = 1.0
        cur = L0; tot = np.zeros_like(L0)
        for s in range(1, steps + 1):
            cur = PT @ cur
            tot = tot + (damp ** s) * cur
        out = np.zeros((nK, len(idx)), np.float32)
        out[have_node] = tot[konode[have_node]]
        return idx, out.T                                               # rows = seeds, cols = all knockouts

    # ---------------- split ----------------
    rng = np.random.RandomState(0)
    scor = [i for i in range(nK) if len([j for j in np.where(A[i] >= TAU)[0] if ~tide[j]]) >= MIN_SPEC]
    perm = rng.permutation(scor); ntest = len(scor) // 3
    test = set(perm[:ntest].tolist()); train = np.array([i for i in range(nK) if i not in test])
    trainset = set(train.tolist()); train_scor = [i for i in train if i in set(scor)]
    print(f"{len(scor)} scorable; train {len(train)} / test {len(test)}", flush=True)

    # ---------------- existing components (identical construction to wall_combine) ----------------
    codep_partners = collections.defaultdict(set)
    for k, lst in D.get("codep", {}).items():
        if not k.isdigit() or int(k) >= len(names):
            continue
        a = names[int(k)]
        for j, _s in lst:
            if isinstance(j, int) and j < len(names):
                b = names[j]
                if a in ki and b in ki:
                    codep_partners[a].add(ki[b]); codep_partners[b].add(ki[a])
    go = D.get("go", {})
    cplx = {names[int(k)]: cs for k, cs in (D.get("gene2cplx", {}) or {}).items()
            if k.isdigit() and int(k) < len(names)}
    per = {"cplx": {k: cplx.get(k, []) for k in kos},
           "gop": {k: go.get(k, {}).get("P", []) for k in kos},
           "goc": {k: go.get(k, {}).get("C", []) for k in kos},
           "gof": {k: go.get(k, {}).get("F", []) for k in kos},
           "path": {k: ([info.get(k, {}).get("path")] if info.get(k, {}).get("path") else []) for k in kos}}
    Msp = {n_: onehot(d, kos) for n_, d in per.items()}
    shared = {n_: np.asarray((m @ m.T).todense(), dtype=np.float32) for n_, m in Msp.items()}
    nbset = collections.defaultdict(set)
    for e in D.get("ppi", []):
        nbset[e[0]].add(e[1]); nbset[e[1]].add(e[0])
    NBcols = {}; rr = []; cc = []
    for i, k in enumerate(kos):
        for g in nbset.get(k, ()):
            rr.append(i); cc.append(NBcols.setdefault(g, len(NBcols)))
    NB = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(nK, max(len(NBcols), 1)))
    shared_nbr = np.asarray((NB @ NB.T).todense(), dtype=np.float32)
    deg = np.array(NB.sum(1)).ravel()
    jac_nbr = shared_nbr / (deg[:, None] + deg[None, :] - shared_nbr + 1e-9)
    is_ppi = np.zeros((nK, nK), dtype=np.float32)
    for i, k in enumerate(kos):
        for g in nbset.get(k, ()):
            if g in ki:
                is_ppi[i, ki[g]] = 1

    def arr(f, default=0.0):
        return np.array([float(info.get(k, {}).get(f, default) or default) for k in kos], dtype=np.float32)
    dep = arr("dep_frac"); loeuf = arr("loeuf", 1.0); logdeg = np.log1p(arr("ppi")); tf = arr("tf")
    comp = np.array([info.get(k, {}).get("comp", "") for k in kos])
    proc = np.array([info.get(k, {}).get("proc", "") for k in kos])
    FEATS = [shared["cplx"], shared["gop"], shared["goc"], shared["gof"], shared["path"], shared_nbr, jac_nbr, is_ppi,
             np.abs(dep[:, None] - dep[None, :]), np.abs(loeuf[:, None] - loeuf[None, :]),
             np.abs(logdeg[:, None] - logdeg[None, :]),
             (comp[:, None] == comp[None, :]).astype(np.float32),
             (proc[:, None] == proc[None, :]).astype(np.float32),
             ((tf[:, None] == 1) & (tf[None, :] == 1)).astype(np.float32)]

    def feat_rows(xi, yidx):
        return np.stack([Fm[xi, yidx] for Fm in FEATS], axis=1)

    Xtr, ytr = [], []
    for xi in train_scor:
        cand = train[train != xi]
        sub = rng.choice(cand, min(len(cand), 250), replace=False)
        Xtr.append(feat_rows(xi, sub)); ytr.append(S[xi, sub])
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=4, min_samples_leaf=40,
                                        random_state=0).fit(np.concatenate(Xtr), np.concatenate(ytr))
    phys_adj = (is_ppi > 0) | (shared["cplx"] > 0) | (shared["path"] > 0)

    # ---------------- MARKOV-P: pre-register the chain config on TRAIN only ----------------
    test_list = sorted(test); train_arr = np.array(sorted(trainset))

    def neighbours_from_scores(vec, exclude, pool, n=N_NEI):
        v = vec.copy(); v[exclude] = -np.inf
        mask = np.full(nK, -np.inf, np.float32); mask[pool] = 0.0
        return np.argsort(-(v + mask))[:n]

    def profile_from(nb):
        return np.abs(M[nb]).mean(0) if len(nb) else np.zeros(nG, np.float32)

    def rec(scores, truth):
        order = nontide_idx[np.argsort(-scores[nontide_idx])][:K]
        return len(set(order.tolist()) & truth) / max(len(truth), 1)

    SPEC = {xi: set(int(j) for j in np.where(A[xi] >= TAU)[0] if ~tide[j]) for xi in test_list}

    # config chosen on TRAIN knockouts scored against TRAIN neighbours -- never on test
    inner = [i for i in train_scor][:200]
    best_cfg, best_v = CHAIN_CFG[0], -1.0
    for (st, dp, cw) in CHAIN_CFG:
        PT = build_protein_P(cw)
        idx, sc = chain_ko_scores(PT, st, dp, inner)
        pool = np.array([i for i in train_arr])
        v = []
        for r, xi in enumerate(idx):
            nb = neighbours_from_scores(sc[r], xi, pool[pool != xi])
            tr_spec = set(int(j) for j in np.where(A[xi] >= TAU)[0] if ~tide[j])
            if len(tr_spec) >= MIN_SPEC:
                v.append(rec(profile_from(nb), tr_spec))
        m = float(np.mean(v)) if v else 0.0
        print(f"    chain cfg steps={st} damp={dp} cplx_w={cw}: TRAIN recall@50 {m:.4f}", flush=True)
        if m > best_v:
            best_v, best_cfg = m, (st, dp, cw)
    print(f"  PRE-REGISTERED chain config (chosen on TRAIN): steps={best_cfg[0]} damp={best_cfg[1]} "
          f"cplx_weight={best_cfg[2]}", flush=True)

    PT_best = build_protein_P(best_cfg[2])
    idx_t, SC_markov = chain_ko_scores(PT_best, best_cfg[0], best_cfg[1], test_list)
    markov_row = {xi: SC_markov[r] for r, xi in enumerate(idx_t)}

    # ---------------- MC-P: resample the protein graph ----------------
    print(f"  MC-P: {MC_REP} replicates resampling the protein graph (edge dropout, complex weight, depth)", flush=True)
    mc_vote = {xi: np.zeros(nK, np.float32) for xi in test_list}
    nppi, ncpl = len(pa), len(ca)
    for r in range(MC_REP):
        kp = rng.rand(nppi) < 0.8
        kc = rng.rand(ncpl) < 0.8 if ncpl else None
        cw = float(rng.uniform(0.0, 2.5))
        PTr = build_protein_P(cw, keep_ppi=kp, keep_cplx=kc)
        st = int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])); dp = float(rng.uniform(0.3, 0.7))
        ii, sc = chain_ko_scores(PTr, st, dp, test_list)
        for q, xi in enumerate(ii):
            nb = neighbours_from_scores(sc[q], xi, train_arr[train_arr != xi])
            mc_vote[xi][nb] += 1.0
        if (r + 1) % 10 == 0:
            print(f"    replicate {r+1}/{MC_REP}", flush=True)

    # ---------------- score every component on the SAME test knockouts ----------------
    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else v * 0.0

    comps = collections.defaultdict(dict)
    for xi in test_list:
        comps["TIDE"][xi] = mover_freq.copy()
        pnb = [j for j in np.where(phys_adj[xi])[0] if j in trainset and j != xi]
        comps["PHYS"][xi] = profile_from(pnb)
        cand = train_arr[train_arr != xi]
        pred = reg.predict(feat_rows(xi, cand))
        comps["LEARNED"][xi] = profile_from(cand[np.argsort(-pred)[:N_NEI]])
        nx = [j for j in codep_partners.get(kos[xi], ()) if j in trainset and j != xi]
        comps["NEXUS"][xi] = profile_from(nx) if nx else np.zeros(nG, np.float32)
        comps["MARKOV-P"][xi] = profile_from(neighbours_from_scores(markov_row[xi], xi, cand)) \
            if xi in markov_row else np.zeros(nG, np.float32)
        comps["MC-P"][xi] = profile_from(np.argsort(-mc_vote[xi])[:N_NEI]) if mc_vote[xi].sum() > 0 \
            else np.zeros(nG, np.float32)
    # SHUFFLED controls: same score distribution, knockout labels permuted -> destroys KO-specificity only
    sh = rng.permutation(test_list)
    for nm in ("MARKOV-P", "MC-P"):
        comps[nm + "-shuf"] = {xi: comps[nm][sh[i]] for i, xi in enumerate(test_list)}

    ORACLE = {}
    for xi in test_list:
        sim = S[xi].copy(); sim[xi] = -np.inf
        mask = np.full(nK, -np.inf, np.float32); mask[train_arr] = 0.0
        ORACLE[xi] = profile_from(np.argsort(-(sim + mask))[:N_NEI])

    singles = {}
    for nm, d in list(comps.items()) + [("ORACLE*", ORACLE)]:
        singles[nm] = float(np.mean([rec(d[xi], SPEC[xi]) for xi in test_list]))
    singles["RANDOM"] = float(np.mean([rec(rng.rand(nG).astype(np.float32), SPEC[xi]) for xi in test_list]))

    print(f"\n  SINGLES (harness setup, tide-removed specific-mover recall@{K}, {len(test_list)} held-out KOs)")
    for nm in ["RANDOM", "TIDE", "NEXUS", "PHYS", "LEARNED", "MARKOV-P", "MARKOV-P-shuf", "MC-P", "MC-P-shuf",
               "ORACLE*"]:
        print(f"    {nm:16s} {singles[nm]:.4f}")

    # SELF-CALIBRATION: the bench must reproduce its own known references or nothing below is trustworthy
    calib_ok = (0.20 <= singles["TIDE"] <= 0.32) and (0.55 <= singles["ORACLE*"] <= 0.70)
    print(f"\n  SELF-CALIBRATION: tide {singles['TIDE']:.3f} (expect ~0.26), oracle {singles['ORACLE*']:.3f} "
          f"(expect ~0.62) -> {'OK' if calib_ok else 'OUT OF RANGE, results below are NOT trustworthy'}")

    def fuse(nms):
        return float(np.mean([rec(sum(z(comps[n_][xi]) for n_ in nms), SPEC[xi]) for xi in test_list]))

    BASE = ["TIDE", "PHYS", "LEARNED", "NEXUS"]
    ens = {"PHYS+LEARNED": fuse(["PHYS", "LEARNED"]),
           "TIDE+PHYS+LEARNED": fuse(["TIDE", "PHYS", "LEARNED"]),
           "4-way (the 0.470 ensemble)": fuse(BASE),
           "4-way + MARKOV-P": fuse(BASE + ["MARKOV-P"]),
           "4-way + MARKOV-P-shuf": fuse(BASE + ["MARKOV-P-shuf"]),
           "4-way + MC-P": fuse(BASE + ["MC-P"]),
           "4-way + MC-P-shuf": fuse(BASE + ["MC-P-shuf"]),
           "5-way + both": fuse(BASE + ["MARKOV-P", "MC-P"])}
    print(f"\n  ENSEMBLES")
    for nm, v in ens.items():
        print(f"    {nm:28s} {v:.4f}")

    base4 = ens["4-way (the 0.470 ensemble)"]
    gain_markov = ens["4-way + MARKOV-P"] - base4
    gain_markov_shuf = ens["4-way + MARKOV-P-shuf"] - base4
    gain_mc = ens["4-way + MC-P"] - base4
    gain_mc_shuf = ens["4-way + MC-P-shuf"] - base4
    real_markov = gain_markov > 0 and gain_markov > gain_markov_shuf
    real_mc = gain_mc > 0 and gain_mc > gain_mc_shuf
    best_ens = max(ens.items(), key=lambda kv: kv[1])
    print(f"\n  vs the shuffled control (the test wall_combine says decides it):")
    print(f"    MARKOV-P adds {gain_markov:+.4f}; its SHUFFLE adds {gain_markov_shuf:+.4f}  -> "
          f"{'real' if real_markov else 'NOT distinguishable from fusion arithmetic'}")
    print(f"    MC-P     adds {gain_mc:+.4f}; its SHUFFLE adds {gain_mc_shuf:+.4f}  -> "
          f"{'real' if real_mc else 'NOT distinguishable from fusion arithmetic'}")

    verdict = (
        f"MARKOV CHAIN AND MONTE CARLO ON THE PROTEIN CHAIN, FUSED INTO THE 0.470 ENSEMBLE. Setup discipline first: the "
        f"0.348 figure belongs to the UNCENSORED setup (tide-null ~0.11-0.13) and the ~0.47-0.49 figure to the HARNESS "
        f"setup (tide-null 0.26, oracle 0.62); mixing them is meaningless, so everything here runs in the harness setup "
        f"and self-calibrates against it -- measured tide {singles['TIDE']:.3f}, oracle {singles['ORACLE*']:.3f}, "
        f"{'in range' if calib_ok else 'OUT OF RANGE so nothing here should be trusted'}. "
        f"THE CHAIN IS A NEIGHBOUR SELECTOR, NOT A DIFFUSION, because ranking genes by walk score is a twice-measured "
        f"dead end here: mechanistic_propagate scored 0.055 against the 0.26 null and markov_outofgame lost to a "
        f"gene-identity prior by nine sigma. What works in this harness is neighbour transfer, so the walk chooses "
        f"which TRAIN knockouts to average. THE NEW INGREDIENT is complex co-membership as a FIRST-CLASS EDGE -- "
        f"multihop_diagnosis measured it at +0.1369 (p=7.7e-04), on par with the direct regulatory edge, and flagged it "
        f"as never built. "
        f"SINGLES: TIDE {singles['TIDE']:.4f}, NEXUS {singles['NEXUS']:.4f}, PHYS {singles['PHYS']:.4f}, LEARNED "
        f"{singles['LEARNED']:.4f}, MARKOV-P {singles['MARKOV-P']:.4f}, MC-P {singles['MC-P']:.4f}, ORACLE* "
        f"{singles['ORACLE*']:.4f}. ENSEMBLES: the 4-way baseline reproduces at {base4:.4f}; adding MARKOV-P gives "
        f"{ens['4-way + MARKOV-P']:.4f} ({gain_markov:+.4f}), adding MC-P gives {ens['4-way + MC-P']:.4f} "
        f"({gain_mc:+.4f}), both together {ens['5-way + both']:.4f}. "
        + (f"THE SHUFFLED CONTROL KILLS IT: permuting the knockout labels while keeping the score distribution "
           f"identical adds {gain_markov_shuf:+.4f} for MARKOV-P and {gain_mc_shuf:+.4f} for MC-P, so the apparent "
           f"ensemble movement is generic z-fusion re-weighting rather than protein-chain information -- exactly the "
           f"trap wall_combine caught once already. " if not (real_markov or real_mc) else
           f"AND IT SURVIVES THE SHUFFLED CONTROL: label-permuted versions add only {gain_markov_shuf:+.4f} "
           f"(MARKOV-P) and {gain_mc_shuf:+.4f} (MC-P), so the gain is knockout-specific protein-chain information, "
           f"not fusion arithmetic. ")
        + f"Best configuration overall: {best_ens[0]} at {best_ens[1]:.4f}, against the oracle ceiling of "
        f"{singles['ORACLE*']:.4f}. The chain configuration (steps={best_cfg[0]}, damping={best_cfg[1]}, complex "
        f"weight={best_cfg[2]}) was PRE-REGISTERED on training knockouts, never on the test set. "
        f"WHAT THIS CANNOT SAY: this is one train/test split of knockouts in one cell line, so differences of a few "
        f"thousandths are not resolvable; only the shuffled-control comparison is load-bearing. Complexes larger than "
        f"60 members are dropped as annotation buckets rather than machines, which is a judgement call. Deterministic "
        f"given seed 0.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"setup": "harness (TAU=1.0, MIN_SPEC=5, TIDE_FRAC=0.05)", "n_test": len(test_list),
               "self_calibration_ok": bool(calib_ok), "singles": singles, "ensembles": ens,
               "chain_config_preregistered": {"steps": best_cfg[0], "damping": best_cfg[1], "cplx_weight": best_cfg[2],
                                              "train_recall": best_v},
               "gains_vs_4way": {"markov": gain_markov, "markov_shuffled": gain_markov_shuf,
                                 "mc": gain_mc, "mc_shuffled": gain_mc_shuf},
               "markov_real": bool(real_markov), "mc_real": bool(real_mc),
               "n_ppi_edges": len(pa) // 2, "n_complex_edges": len(ca),
               "verdict": verdict, "note": verdict}, open(OUT / "protein_chain_combine.json", "w"), indent=1)
    print("\n  -> outputs/orphan/protein_chain_combine.json")


if __name__ == "__main__":
    main()
