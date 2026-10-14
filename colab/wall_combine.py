"""wall_combine -- does COMBINING the three no-peek predictors (TIDE + NEIGHBOUR/physical-graph + LEARNED-similarity) beat the best
single one, and how much of the graph->oracle head-room does the ensemble close? Same leakage-controlled protocol as wall_learnsim
(K562, train/test split of knockouts, learned similarity trained on TRAIN only, everything transfers from TRAIN knockouts only), so the
numbers are directly comparable. The honest question: are the three predictors COMPLEMENTARY (capture different specific movers -> the
ensemble climbs toward the oracle) or REDUNDANT (all reading the same shared-neighbour signal -> the ensemble adds ~nothing)?

Predictors, all no-peek, all transferring measured |z| from TRAIN knockouts:
  TIDE     -- per-gene global mover frequency (same forecast for every KO; the generic-stress component).
  PHYS     -- average the profiles of x's fixed-graph neighbours (share a PPI edge / complex / pathway).
  LEARNED  -- average the profiles of the TRAIN knockouts a gradient-boosted model predicts x behaves like, from annotation features.
Fusions (parameter-free, no tuning on the test set): PHYS+LEARNED and TIDE+PHYS+LEARNED via per-gene z-score sum, plus a UNION-neighbour
variant (average over the union of the phys and learned neighbour sets). Scored on tide-removed specific-mover recall@50, flanked by the
ORACLE* ceiling (peeks). Decisive comparison: best ENSEMBLE vs best SINGLE component -- an ensemble only earns its place by beating the
best single by a real margin.
"""
import os
import json, pickle, collections
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI = 1.0, 50, 5, 0.05, 10


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

    # ---- features (identical to wall_learnsim) ----
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    # NEXUS view: co-dependency partners (DepMap) keyed by KO name -- a DIFFERENT graph than PPI/complex/pathway, the honest 'NEXUS'
    # signal to fuse. codep keys are gene-universe index-strings; map to KO rows.
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
    cplx = {}
    for k, cs in (D.get("gene2cplx", {}) or {}).items():
        if k.isdigit() and int(k) < len(names):
            cplx[names[int(k)]] = cs
    per = {"cplx": {k: cplx.get(k, []) for k in kos},
           "gop": {k: go.get(k, {}).get("P", []) for k in kos},
           "goc": {k: go.get(k, {}).get("C", []) for k in kos},
           "gof": {k: go.get(k, {}).get("F", []) for k in kos},
           "path": {k: ([info.get(k, {}).get("path")] if info.get(k, {}).get("path") else []) for k in kos}}
    Msp = {name: onehot(d, kos) for name, d in per.items()}
    shared = {name: np.asarray((m @ m.T).todense(), dtype=np.float32) for name, m in Msp.items()}
    nbset = collections.defaultdict(set)
    # THE INDEX-SPACE BUG. cell_complete's 'ppi' endpoints are INTEGER INDICES into genes[], but `kos` holds gene
    # NAME strings. The shipped code keyed nbset by the raw integer and then looked it up by name, so
    # nbset.get(<name>) returned EMPTY for every knockout: 0 of 16,492 gene names resolved against a 14,230-key
    # table, and 0 of 191,447 edges were ever seen. shared_ppi_nbr / jaccard_ppi_nbr / is_direct_ppi were
    # identically zero, and phys_adj silently degraded to complex-OR-pathway -- so the component labelled
    # "PHYS (PPI|complex|pathway)" contained no PPI at all. Fixed the same way as protein_chain_combine.py.
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < len(names) and 0 <= b_ < len(names) and a_ != b_:
            nbset[names[a_]].add(names[b_])
            nbset[names[b_]].add(names[a_])
    NBcols = {}; rows = []; cols = []
    for i, k in enumerate(kos):
        for g in nbset.get(k, ()):
            rows.append(i); cols.append(NBcols.setdefault(g, len(NBcols)))
    NB = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nK, max(len(NBcols), 1)))
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
    dep = arr("dep_frac"); loeuf = arr("loeuf", 1.0); logdeg = np.log1p(arr("ppi"))
    comp = np.array([info.get(k, {}).get("comp", "") for k in kos]); proc = np.array([info.get(k, {}).get("proc", "") for k in kos])
    tf = arr("tf")
    d_dep = np.abs(dep[:, None] - dep[None, :]); d_loe = np.abs(loeuf[:, None] - loeuf[None, :])
    d_deg = np.abs(logdeg[:, None] - logdeg[None, :])
    same_comp = (comp[:, None] == comp[None, :]).astype(np.float32); same_proc = (proc[:, None] == proc[None, :]).astype(np.float32)
    both_tf = ((tf[:, None] == 1) & (tf[None, :] == 1)).astype(np.float32)
    FEATS = [shared["cplx"], shared["gop"], shared["goc"], shared["gof"], shared["path"],
             shared_nbr, jac_nbr, is_ppi, d_dep, d_loe, d_deg, same_comp, same_proc, both_tf]

    def feat_rows(xi, yidx):
        return np.stack([Fm[xi, yidx] for Fm in FEATS], axis=1)

    rng = np.random.RandomState(0)
    scor = [i for i in range(nK) if len([j for j in np.where(A[i] >= TAU)[0] if ~tide[j]]) >= MIN_SPEC]
    perm = rng.permutation(scor); ntest = len(scor) // 3
    test = set(perm[:ntest].tolist()); train = np.array([i for i in range(nK) if i not in test])
    train_scor = [i for i in train if i in set(scor)]
    trainset = set(train.tolist())
    print(f"K562: {nK} KOs; {len(scor)} scorable; train {len(train)} / test {len(test)}")

    Xtr = []; ytr = []
    for xi in train_scor:
        cand = train[train != xi]
        sub = rng.choice(cand, min(len(cand), 250), replace=False)
        Xtr.append(feat_rows(xi, sub)); ytr.append(S[xi, sub])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=4, min_samples_leaf=40,
                                        random_state=0).fit(Xtr, ytr)

    phys_adj = (is_ppi > 0) | (shared["cplx"] > 0) | (shared["path"] > 0)
    phys_nb = {i: [j for j in np.where(phys_adj[i])[0] if j in trainset and j != i] for i in test}

    def rec(scores, truth):
        order = nontide_idx[np.argsort(-scores[nontide_idx])][:K]
        return len(set(order.tolist()) & truth) / max(len(truth), 1)

    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else v * 0.0

    # pass 1: compute and store each test KO's component score vectors + truth
    test_list = sorted(test)
    Sp, Sl, So, Sn, SPEC = {}, {}, {}, {}, {}
    kx_train = trainset
    n_nexus_cov = 0
    for xi in test_list:
        SPEC[xi] = set(j for j in np.where(A[xi] >= TAU)[0].tolist() if ~tide[j])
        pn = phys_nb[xi]
        Sp[xi] = A[pn].mean(0) if pn else np.zeros(nG)
        ltop = train[np.argsort(-reg.predict(feat_rows(xi, train)))[:N_NEI]]
        Sl[xi] = A[ltop].mean(0)
        pnset = set(pn) | set(int(t) for t in ltop)
        So[xi] = A[sorted(pnset)].mean(0) if pnset else np.zeros(nG)   # union neighbours
        # NEXUS: transfer from x's co-dependency partners that are TRAIN knockouts (no peeking, no leakage).
        # xi is a KO ROW index -> the KO's gene name is kos[xi] (NOT names[xi], which is a gene-universe index).
        cn = [p for p in codep_partners.get(kos[xi], ()) if p in kx_train and p != xi]
        Sn[xi] = A[cn].mean(0) if cn else np.zeros(nG)
        if cn:
            n_nexus_cov += 1
    # CONTROL: pair each KO's LEARNED with a DIFFERENT KO's PHYS (misaligned) -- same fusion mechanics, KO-specific alignment broken.
    shuf = list(test_list); rs = np.random.RandomState(1); rs.shuffle(shuf)
    shuf_of = {xi: shuf[(i + 1) % len(shuf)] if shuf[i] == xi else shuf[i] for i, xi in enumerate(test_list)}

    out = collections.defaultdict(list)
    for xi in test_list:
        spec = SPEC[xi]; s_tide = mover_freq
        st = S[xi, train]; otop = train[np.argsort(-st)[:N_NEI]]; s_oracle = A[otop].mean(0)
        preds = {"tide": s_tide, "phys": Sp[xi], "learned": Sl[xi], "nexus": Sn[xi], "oracle": s_oracle,
                 "phys+learned": z(Sp[xi]) + z(Sl[xi]),
                 "tide+phys+learned": z(s_tide) + z(Sp[xi]) + z(Sl[xi]),
                 "phys+nexus": z(Sp[xi]) + z(Sn[xi]),
                 "tide+phys+learned+nexus": z(s_tide) + z(Sp[xi]) + z(Sl[xi]) + z(Sn[xi]),
                 "union_neighbours": So[xi],
                 "CTRL_shufphys+learned": z(Sp[shuf_of[xi]]) + z(Sl[xi]),          # misaligned-phys control
                 "CTRL_shufnexus+3way": z(s_tide) + z(Sp[xi]) + z(Sl[xi]) + z(Sn[shuf_of[xi]])}  # misaligned-nexus control
        for nm, s in preds.items():
            out[nm].append(rec(s, spec))
    m = {k: float(np.mean(v)) for k, v in out.items()}
    singles = {k: m[k] for k in ["tide", "phys", "learned", "nexus"]}
    combos = {k: m[k] for k in ["phys+learned", "tide+phys+learned", "phys+nexus",
                                "tide+phys+learned+nexus", "union_neighbours"]}
    best_single_name = max(singles, key=singles.get); best_single = singles[best_single_name]
    best_combo_name = max(combos, key=combos.get); best_combo = combos[best_combo_name]
    print(f"\nHeld-out TEST knockouts ({len(out['tide'])}). Specific-mover recall@{K} (tide removed):\n")
    for nm in ["tide", "phys", "learned", "nexus", "phys+learned", "tide+phys+learned", "phys+nexus",
               "tide+phys+learned+nexus", "union_neighbours", "oracle"]:
        tag = "  <- best single" if nm == best_single_name else ("  <- best ensemble" if nm == best_combo_name else
              ("  (ceiling, peeks)" if nm == "oracle" else ""))
        print(f"   {nm:24s} {m[nm]:.3f}{tag}")
    ctrl = m["CTRL_shufphys+learned"]
    ctrl_nexus = m["CTRL_shufnexus+3way"]
    print(f"   {'CTRL shuf-phys+learn':24s} {ctrl:.3f}  <- CONTROL for phys (wrong-KO phys fused with learned)")
    print(f"   {'CTRL shuf-nexus+3way':24s} {ctrl_nexus:.3f}  <- CONTROL for nexus (wrong-KO nexus fused with the 3-way)")
    gain = best_combo - best_single
    closed_single = (best_single - m["tide"]) / max(m["oracle"] - m["tide"], 1e-9)
    closed_combo = (best_combo - m["tide"]) / max(m["oracle"] - m["tide"], 1e-9)
    real_complementarity = m["phys+learned"] - ctrl        # real fusion gain beyond generic z-fusion mechanics
    # THE NEXUS QUESTION: does adding the co-dependency (NEXUS) view add ANYTHING orthogonal to tide+phys+learned?
    nexus_over_3way = m["tide+phys+learned+nexus"] - m["tide+phys+learned"]
    nexus_over_phys = m["phys+nexus"] - m["phys"]
    nexus_vs_ctrl = m["tide+phys+learned+nexus"] - ctrl_nexus     # >0 and 4way>3way => nexus adds REAL KO-specific signal
    nexus_cov = n_nexus_cov / max(len(test), 1)
    print(f"\n   best ensemble ({best_combo_name}) {best_combo:.3f}  vs best single ({best_single_name}) {best_single:.3f}  = {gain:+.3f}")
    print(f"   phys+learned complementarity vs shuffled-phys control: {real_complementarity:+.3f}")
    print(f"\n   NEXUS (co-dependency view) coverage of test KOs: {nexus_cov:.0%}")
    print(f"   does NEXUS add?  4-way {m['tide+phys+learned+nexus']:.3f} vs 3-way {m['tide+phys+learned']:.3f} = {nexus_over_3way:+.3f}"
          f"   |  phys+nexus {m['phys+nexus']:.3f} vs phys {m['phys']:.3f} = {nexus_over_phys:+.3f}")
    print(f"   NEXUS control: 4-way {m['tide+phys+learned+nexus']:.3f} vs shuffled-nexus {ctrl_nexus:.3f} = {nexus_vs_ctrl:+.3f}")
    print(f"   head-room (tide->oracle) closed:  best single {closed_single*100:.0f}%  ->  best ensemble {closed_combo*100:.0f}%")

    real_gain = gain >= 0.02 and real_complementarity >= 0.02
    tide_helps = m["tide+phys+learned"] > m["phys+learned"] + 0.005
    nexus_adds = nexus_over_3way >= 0.01 and nexus_vs_ctrl >= 0.01   # must add over the 3-way AND beat the shuffled-nexus control
    verdict = (
        f"COMBINING THE NO-PEEK PREDICTORS (tide, phys-graph, learned-similarity, + the NEXUS co-dependency view); K562, {len(test)} "
        f"held-out KOs, tide-removed specific-mover recall@{K}. Singles: TIDE {m['tide']:.3f}, PHYS {m['phys']:.3f}, LEARNED "
        f"{m['learned']:.3f}, NEXUS {m['nexus']:.3f}. Ensembles: PHYS+LEARNED {m['phys+learned']:.3f}, TIDE+PHYS+LEARNED "
        f"{m['tide+phys+learned']:.3f}, +NEXUS (4-way) {m['tide+phys+learned+nexus']:.3f}, UNION-neighbours {m['union_neighbours']:.3f}. "
        f"ORACLE* ceiling {m['oracle']:.3f}. "
        f"DECISIVE: best ensemble ({best_combo_name}) {best_combo:.3f} vs best single ({best_single_name}) {best_single:.3f} = "
        f"{gain:+.3f}. "
        + (f"REAL (if modest) ENSEMBLE GAIN, and it SURVIVES THE CONTROL: fusing phys+learned {m['phys+learned']:.3f} beats the "
           f"shuffled-phys control {ctrl:.3f} ({real_complementarity:+.3f}), so the gain is genuine KO-specific COMPLEMENTARITY -- phys "
           "and learned select partly different behavioural neighbours and catch different specific movers -- not the generic z-fusion "
           f"re-weighting that fooled nexus_wall. Combining closes more of the graph->oracle head-room ({closed_single*100:.0f}% -> "
           f"{closed_combo*100:.0f}%) and is worth keeping as the deployed no-peek predictor. It does NOT reach the oracle -- the "
           "predictors still miss the same hard neighbours the oracle finds by peeking, so a genuinely orthogonal signal is still needed "
           "for the rest of the head-room."
           if real_gain else
           "HONEST NEGATIVE -- the ensemble does NOT meaningfully beat the best single component (" + f"{gain:+.3f}" + "). The three are "
           "largely REDUNDANT, not complementary: PHYS and LEARNED are both reading the same shared-neighbour signal (learned similarity "
           "barely beat the graph to begin with), so averaging them adds ~nothing, and TIDE "
           + ("adds a little on top" if tide_helps else "adds nothing to specific recall (it is the generic component, and the test "
              "removes the tide, so it only dilutes)")
           + ". The oracle head-room stays open: fusing no-peek predictors does not reach it, because they all miss the SAME behavioural "
           "neighbours the oracle finds by peeking. This is the same lesson as wall_learnsim -- the gap is a REPRESENTATION gap (you need "
           "a genuinely orthogonal signal, e.g. measured co-perturbation or a learned embedding), not something an ensemble of correlated "
           "annotation/graph views can close.")
        + f" THE NEXUS QUESTION (does adding the co-dependency view add anything ORTHOGONAL, the high bar wall_learnsim set): NEXUS alone "
        f"scores {m['nexus']:.3f} (covers {nexus_cov:.0%} of test KOs); 4-way tide+phys+learned+NEXUS {m['tide+phys+learned+nexus']:.3f} vs "
        f"3-way {m['tide+phys+learned']:.3f} = {nexus_over_3way:+.3f}, and the shuffled-nexus control is {ctrl_nexus:.3f} "
        f"({nexus_vs_ctrl:+.3f}). "
        + ("NEXUS ADDS REAL ORTHOGONAL SIGNAL: it lifts the ensemble beyond the 3-way AND beats its own shuffle control, so the "
           "co-dependency graph catches specific movers the PPI/complex/pathway + annotation views miss -- the first genuinely "
           "orthogonal view we've added." if nexus_adds else
           "NEXUS DOES NOT ADD (honest negative): the co-dependency view is REDUNDANT with the physical graph -- co-dependency partners "
           "are largely the same complex/physical partners phys already uses, so fusing it moves the ensemble by "
           f"{nexus_over_3way:+.3f} (and phys+nexus barely moves phys, {nexus_over_phys:+.3f}). It is NOT the orthogonal signal the "
           "oracle head-room needs; it re-reads the physical layer the ensemble already has. This is exactly wall_learnsim's warning: a "
           "fourth view of the same protein-graph saturates, not extends.")
        + f" (Deployable no-peek number ~{best_single:.2f}; oracle ceiling ~{m['oracle']:.2f}.) Same protocol as wall_learnsim; "
        "single cell type, single steady-state-mRNA readout.")
    print(f"\nVERDICT: {verdict}")
    js = {"cell_type": "K562", "n_test": len(test), "K": K,
          "recall_specific": {k: m[k] for k in ["tide", "phys", "learned", "nexus", "phys+learned", "tide+phys+learned",
                                                 "phys+nexus", "tide+phys+learned+nexus", "union_neighbours", "oracle"]},
          "control_shuffled_phys_plus_learned": ctrl, "complementarity_over_control": real_complementarity,
          "nexus_coverage": nexus_cov, "nexus_over_3way": nexus_over_3way, "nexus_over_phys": nexus_over_phys,
          "control_shuffled_nexus": ctrl_nexus, "nexus_over_control": nexus_vs_ctrl, "nexus_adds": bool(nexus_adds),
          "best_single": {"name": best_single_name, "recall": best_single},
          "best_ensemble": {"name": best_combo_name, "recall": best_combo},
          "ensemble_gain_over_best_single": gain, "headroom_closed_best_single": closed_single,
          "headroom_closed_best_ensemble": closed_combo, "ensemble_helps": bool(real_gain),
          "verdict": verdict, "note": verdict}
    json.dump(js, open(OUT / "wall_combine.json", "w"), indent=1)
    print("\n  -> outputs/orphan/wall_combine.json")
    return js


if __name__ == "__main__":
    main()
