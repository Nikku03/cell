"""ceiling_cartography -- the training-free "is any door past 0.62 open?" suite that decides whether v7 should be a model at all.

The whole transformer arc (v2..v6, self-graph, causal) washed against a ~0.62 ORACLE. That oracle PEEKS at the held-out KO's true
profile and RETRIEVES neighbours (sim = Mn@Mn[r], mean-transfer of measured |z|) -- so 0.62 is a RETRIEVAL ceiling. Before building any
heavier v7, measure -- with NO training -- which of the three structurally-distinct doors is even open. Each door has a cheap upper-bound
oracle AND the control that falsifies a false-positive:

  DOOR 1  REACHABILITY (better neighbour selection): can a KO's INPUT features even identify the good neighbours the true-profile oracle
          finds? input-similarity retrieval vs train-restricted true-profile retrieval. If inputs already reach the oracle -> the gap to
          the current model is ENCODER headroom (algo). If inputs fall far short -> the response is not a function of the inputs (data).
          Can only APPROACH 0.62, never beat it.
  DOOR 2  GENERATION (emit a mover no single neighbour carries): NMF program basis on train specific responses; NNLS each held-out KO's
          TRUE profile onto it; reconstruct; recall. Beats 0.62 iff the specific response is low-rank-compositional. CONTROL: loading-
          shuffle must collapse. GATE: are the true loadings even predictable from inputs (ridge)? -- a decoder is pointless otherwise.
  DOOR 3  CROSS-LINE (use the held-out KO's OWN measurement in another cell line -- info the K562 oracle structurally never sees): transfer
          the same KO's other-line |z| to K562; recall vs the same-subset K562 oracle. Beats it iff the K562-specific residue transfers.
          CONTROL: wrong-KO other-line profile must collapse to the marginal (tide) floor.

Held-out K562, 3 splits, strict train/test separation. Deterministic. Emits three decision numbers + the modal verdict. No model is
trained; nothing is built until a door is shown open."""
import os, json, pickle, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy
from transformer_ko import build_context

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
SMOKE = bool(os.environ.get("CC_SMOKE"))
V5_REF = 0.50            # current best model (v5) held-out specific recall@50, for interpreting the encoder-headroom gap
OTHER_LINES = ["RPE1", "HepG2", "Jurkat", "HCT116", "Melanoma"]
N_NEI = 10


def splits(H, have, seed, frac=0.30):
    """3-way: test (held-out scorable K562), the rest is the train gallery. Mirrors transformer_ko5 split discipline."""
    hidx = {k: i for i, k in enumerate(have)}
    scor = [k for k in H.scorable if k in hidx]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(scor)); n_test = int(frac * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    train = [k for k in have if k not in test_set]
    return train, test, hidx


def unit(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def ctx_pool(H, have, Fg, tok_idx, tok_mask):
    """per-KO context representation = mean of its token node-features (the input the transformer actually sees)."""
    P = np.zeros((len(have), Fg.shape[1]), np.float32)
    for r in range(len(have)):
        m = tok_mask[r] > 0.5
        P[r] = Fg[tok_idx[r][m]].mean(0)
    return P


def recall_of(H, vec, ko, K=50):
    r = H.ki[ko]; mv = np.where(H.mover[r])[0]
    spec = set(int(i) for i in mv if H.nontide[i])
    if len(spec) < 1:
        return None
    return H._recall(vec, spec, H.nontide_idx, K)


# ------------------------------------------------------------------ DOOR 1: reachability
def door1_reachability(H, have, X, Y, ctxP, train, test, hidx):
    """input-similarity retrieval vs train-restricted TRUE-profile retrieval (the reachable ceiling), per held-out KO."""
    tr_rows = np.array([hidx[k] for k in train])
    Yn = unit(Y)                                   # true response direction (== Mn on the have-subset magnitude)
    Xn = unit((X - X[tr_rows].mean(0)) / (X[tr_rows].std(0) + 1e-6))
    Cn = unit((ctxP - ctxP[tr_rows].mean(0)) / (ctxP[tr_rows].std(0) + 1e-6))

    def retrieve(simrow):
        top = tr_rows[np.argsort(-simrow)[:N_NEI]]
        return Y[top].mean(0)

    res = collections.defaultdict(list)
    reach_gap, missed_pairs = [], []
    for ko in test:
        i = hidx[ko]
        oracle_tr = retrieve(Yn[tr_rows] @ Yn[i])          # peek at true profile -> best train neighbours
        dep = retrieve(Xn[tr_rows] @ Xn[i])                # DepMap-similarity retrieval
        ctx = retrieve(Cn[tr_rows] @ Cn[i])                # context-pooled retrieval
        ro = recall_of(H, oracle_tr, ko); rd = recall_of(H, dep, ko); rc = recall_of(H, ctx, ko)
        if ro is None:
            continue
        res["oracle_tr"].append(ro); res["dep"].append(rd); res["ctx"].append(rc)
        res["best_input"].append(max(rd, rc))
        # separability of the neighbour the oracle used but inputs missed
        j_or = tr_rows[np.argmax(Yn[tr_rows] @ Yn[i])]
        j_in = tr_rows[np.argmax(Cn[tr_rows] @ Cn[i])]
        if j_or != j_in:
            missed_pairs.append(float(Cn[j_or] @ Cn[i] - Cn[j_in] @ Cn[i]))   # <0 => oracle's neighbour looks LESS similar in inputs
    m = lambda a: float(np.mean(a)) if a else float("nan")
    orc, bi = m(res["oracle_tr"]), m(res["best_input"])
    reachable_frac = bi / orc if orc > 1e-9 else float("nan")
    return {"oracle_tr": orc, "dep_retrieval": m(res["dep"]), "ctx_retrieval": m(res["ctx"]),
            "best_input_retrieval": bi, "reachable_frac": reachable_frac,
            "encoder_headroom_vs_v5": bi - V5_REF,           # how far the current model sits below the input-reachable ceiling
            "sep_missed_median": float(np.median(missed_pairs)) if missed_pairs else float("nan"),
            "n": len(res["oracle_tr"])}


# ------------------------------------------------------------------ DOOR 2: generation
def door2_generation(H, have, X, Y, ctxP, train, test, hidx, seed, ranks):
    """NMF program basis on TRAIN specific responses; project each held-out KO's TRUE profile; reconstruct; recall. + shuffle + gate."""
    from sklearn.decomposition import NMF
    from sklearn.linear_model import Ridge
    nti = H.nontide_idx
    tr_rows = np.array([hidx[k] for k in train]); te_rows = np.array([hidx[k] for k in test])
    Atr = Y[tr_rows][:, nti]; Ate = Y[te_rows][:, nti]          # |z| on non-tide genes only (what the metric scores)
    Xtr, Xte = X[tr_rows], X[te_rows]
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
    out = {}
    rng = np.random.RandomState(seed)
    for r in ranks:
        nmf = NMF(n_components=r, init="nndsvd", max_iter=250, random_state=seed)
        W = nmf.fit_transform(Atr); comp = nmf.components_             # comp: (r, |nontide|)
        Wte = nmf.transform(Ate)                                        # NNLS-projection of TRUE test profiles
        recs, recs_sh, recs_gate, recs_meanW = [], [], [], []
        # gate: can inputs predict the loadings? ridge Xtr->W, predict Wte_hat
        ridge = Ridge(alpha=10.0).fit((Xtr - xmu) / xsd, W)
        Wte_hat = np.clip(ridge.predict((Xte - xmu) / xsd), 0, None)
        Wbar = W.mean(0, keepdims=True)
        sh = rng.permutation(len(test))
        for a, ko in enumerate(test):
            full = np.zeros(H.nG, np.float32); full[nti] = Wte[a] @ comp
            recs.append(recall_of(H, full, ko))
            fsh = np.zeros(H.nG, np.float32); fsh[nti] = Wte[sh[a]] @ comp      # loading-shuffle control
            recs_sh.append(recall_of(H, fsh, ko))
            fg = np.zeros(H.nG, np.float32); fg[nti] = Wte_hat[a] @ comp        # predicted-loading (decoder proxy)
            recs_gate.append(recall_of(H, fg, ko))
            fm = np.zeros(H.nG, np.float32); fm[nti] = Wbar[0] @ comp           # mean-loading baseline
            recs_meanW.append(recall_of(H, fm, ko))
        mm = lambda a: float(np.mean([x for x in a if x is not None]))
        out[str(r)] = {"gen_oracle": mm(recs), "loading_shuffle": mm(recs_sh),
                       "gate_predicted_loading": mm(recs_gate), "mean_loading_baseline": mm(recs_meanW)}
    return out


# ------------------------------------------------------------------ DOOR 3: cross-line
def door3_crossline(H, test):
    """transfer the SAME KO's other-line |z| to K562; recall vs the same-subset K562 oracle. + wrong-KO + marginal controls."""
    per_line = {}
    pooled = collections.defaultdict(list)
    for L in OTHER_LINES:
        try:
            KL = pickle.load(open(f"{SP}/nlz_{L}.pkl", "rb"))
        except FileNotFoundError:
            continue
        kos_L = sorted(KL)
        paired = [ko for ko in test if ko in KL]
        if len(paired) < 8:
            per_line[L] = {"n": len(paired), "note": "too few paired KOs"}
            continue

        def vec_from_L(ko_src):
            v = np.zeros(H.nG, np.float32)
            for g, z in KL[ko_src].items():
                j = H.gi.get(g)
                if j is not None:
                    v[j] = abs(z)
            return v

        rng = np.random.RandomState(hash(L) & 0xffff)
        wrong = [kos_L[i] for i in rng.randint(0, len(kos_L), len(paired))]
        xl, orc, wr, mar = [], [], [], []
        for a, ko in enumerate(paired):
            r = H.ki[ko]
            xl.append(recall_of(H, vec_from_L(ko), ko))                  # same-KO cross-line transfer
            wr.append(recall_of(H, vec_from_L(wrong[a]), ko))            # wrong-KO control
            mar.append(recall_of(H, H.mover_freq.copy(), ko))           # marginal/tide floor
            sim = H.Mn @ H.Mn[r]; sim[r] = -1
            top = np.argsort(-sim)[:N_NEI]
            orc.append(recall_of(H, H.A[top].mean(0), ko))              # same-subset K562 oracle (peek)
        mm = lambda a: float(np.mean([x for x in a if x is not None]))
        per_line[L] = {"n": len(paired), "crossline_transfer": mm(xl), "k562_oracle_subset": mm(orc),
                       "wrongKO_control": mm(wr), "marginal_floor": mm(mar)}
        pooled["xl"] += xl; pooled["orc"] += orc; pooled["wr"] += wr; pooled["mar"] += mar
    mm = lambda a: float(np.mean([x for x in a if x is not None])) if a else float("nan")
    pool = {"crossline_transfer": mm(pooled["xl"]), "k562_oracle_subset": mm(pooled["orc"]),
            "wrongKO_control": mm(pooled["wr"]), "marginal_floor": mm(pooled["mar"]),
            "n": len([x for x in pooled["xl"] if x is not None])}
    return {"per_line": per_line, "pooled": pool}


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    ctxP = ctx_pool(H, have, Fg, tok_idx, tok_mask)
    ranks = [32, 64] if SMOKE else [32, 64, 128, 256]
    seeds = [0] if SMOKE else [0, 1, 2]

    agg1 = collections.defaultdict(list); agg2 = collections.defaultdict(lambda: collections.defaultdict(list)); agg3 = collections.defaultdict(list)
    for s in seeds:
        print(f"\n{'='*60}\n[SPLIT SEED {s}]{'  (SMOKE)' if SMOKE else ''}")
        train, test, hidx = splits(H, have, s)
        print(f"    train gallery {len(train)}  |  held-out test {len(test)}")
        d1 = door1_reachability(H, have, X, Y, ctxP, train, test, hidx)
        print(f"  DOOR1 reachability: oracle_tr {d1['oracle_tr']:.3f} | best-input retrieval {d1['best_input_retrieval']:.3f} "
              f"(dep {d1['dep_retrieval']:.3f}/ctx {d1['ctx_retrieval']:.3f}) | reachable {d1['reachable_frac']:.0%} | "
              f"encoder-headroom vs v5 {d1['encoder_headroom_vs_v5']:+.3f}")
        for k, v in d1.items():
            agg1[k].append(v)
        d2 = door2_generation(H, have, X, Y, ctxP, train, test, hidx, s, ranks)
        for r, v in d2.items():
            print(f"  DOOR2 generation r={r:>3}: gen-oracle {v['gen_oracle']:.3f} | shuffle {v['loading_shuffle']:.3f} | "
                  f"gate(pred-load) {v['gate_predicted_loading']:.3f} | mean-load {v['mean_loading_baseline']:.3f}")
            for k, x in v.items():
                agg2[r][k].append(x)
        d3 = door3_crossline(H, test)
        p = d3["pooled"]
        print(f"  DOOR3 cross-line (pooled n={p['n']}): transfer {p['crossline_transfer']:.3f} | K562-oracle(subset) "
              f"{p['k562_oracle_subset']:.3f} | wrongKO {p['wrongKO_control']:.3f} | marginal {p['marginal_floor']:.3f}")
        for k, v in p.items():
            agg3[k].append(v)

    A1 = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in agg1.items()}
    A2 = {r: {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in d.items()} for r, d in agg2.items()}
    A3 = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in agg3.items()}
    best_r = max(A2, key=lambda r: A2[r]["gen_oracle"][0]) if A2 else None
    gen = A2[best_r]["gen_oracle"][0] if best_r else float("nan")
    gen_sh = A2[best_r]["loading_shuffle"][0] if best_r else float("nan")
    gate = A2[best_r]["gate_predicted_loading"][0] if best_r else float("nan")
    gate_base = A2[best_r]["mean_loading_baseline"][0] if best_r else float("nan")
    oracle_ref = A1["oracle_tr"][0]

    # ---- decision logic ----
    # DOOR 2 has TWO conditions: (a) the generation CEILING beats retrieval and is KO-specific (shuffle collapses); (b) it is
    # REALIZABLE -- inputs predict the loadings well enough that a decoder would beat the CURRENT model (v5), not just the mean baseline.
    door2_ceiling = (gen > 0.64) and ((gen - gen_sh) > 0.10)
    door2_realizable = gate > (V5_REF + 0.01)
    door2_open = door2_ceiling and door2_realizable
    door3_open = (A3["crossline_transfer"][0] > A3["k562_oracle_subset"][0]) and (A3["crossline_transfer"][0] > A3["wrongKO_control"][0] + 0.05)
    door1_algo = A1["reachable_frac"][0] > 0.90 and A1["encoder_headroom_vs_v5"][0] > 0.03

    print(f"\n{'='*60}\nCEILING CARTOGRAPHY -- {len(seeds)} split(s), held-out K562")
    print(f"  DOOR 1 (reachability): best-input retrieval {A1['best_input_retrieval'][0]:.3f} vs train-oracle {oracle_ref:.3f} "
          f"= {A1['reachable_frac'][0]:.0%} reachable; encoder-headroom over v5(~{V5_REF}) {A1['encoder_headroom_vs_v5'][0]:+.3f} "
          f"-> {'ALGO headroom (modest encoder bump justified)' if door1_algo else 'no clean algo headroom'}")
    d2msg = ("OPEN (higher ceiling AND realizable)" if door2_open else
             f"ceiling exceeds retrieval ({gen:.3f}>0.62) but NOT realizable -- loadings predict at {gate:.3f} < v5 ~{V5_REF}" if door2_ceiling else
             "SHUT (ceiling does not beat retrieval)")
    print(f"  DOOR 2 (generation, best r={best_r}): gen-oracle {gen:.3f} vs ORACLE ~0.62; shuffle {gen_sh:.3f}; "
          f"gate {gate:.3f} vs mean-load {gate_base:.3f} / v5 ~{V5_REF} -> {d2msg}")
    print(f"  DOOR 3 (cross-line): same-KO transfer {A3['crossline_transfer'][0]:.3f} vs K562-oracle(subset) "
          f"{A3['k562_oracle_subset'][0]:.3f}; wrongKO {A3['wrongKO_control'][0]:.3f} -> {'OPEN' if door3_open else 'SHUT'}")

    doors = []
    if door2_open: doors.append("GENERATION (build a gated program-loading decoder)")
    if door3_open: doors.append("CROSS-LINE (build a cross-line-conditioned decoder)")
    if door1_algo: doors.append("ENCODER (a modest controlled capacity bump)")
    door2_verdict = ("OPEN" if door2_open else
                     ("OPEN-IN-PRINCIPLE-SHUT-IN-PRACTICE (generation ceiling %.3f > 0.62 and KO-specific, but the true loadings predict "
                      "from inputs at only %.3f -- below v5 ~%.2f -- so a decoder would score WORSE than current retrieval; not worth "
                      "building unless the loadings become far more predictable)" % (gen, gate, V5_REF) if door2_ceiling else
                      "SHUT (generation cannot exceed retrieval)"))
    verdict = (
        f"CEILING CARTOGRAPHY (ceiling_cartography.py): training-free test of whether ANY door past the ~0.62 retrieval oracle is open, "
        f"to decide if v7 should be a model at all. {len(seeds)} splits, held-out K562. "
        f"DOOR 1 REACHABILITY: a KO's input features reach {A1['reachable_frac'][0]:.0%} of the train-restricted true-profile oracle "
        f"({A1['best_input_retrieval'][0]:.3f} vs {oracle_ref:.3f}); current v5 (~{V5_REF}) sits {A1['encoder_headroom_vs_v5'][0]:+.3f} "
        f"from that input-reachable ceiling. DOOR 2 GENERATION: an NMF program basis reconstructing the TRUE held-out profile recalls "
        f"{gen:.3f} (best r={best_r}) vs oracle ~0.62; loading-shuffle {gen_sh:.3f}; and inputs predict the loadings at {gate:.3f} vs "
        f"{gate_base:.3f} mean-baseline -> {door2_verdict}. "
        f"DOOR 3 CROSS-LINE: transferring the same KO's other-line |z| to K562 recalls {A3['crossline_transfer'][0]:.3f} vs the same-subset "
        f"K562 oracle {A3['k562_oracle_subset'][0]:.3f} (wrong-KO {A3['wrongKO_control'][0]:.3f}, marginal {A3['marginal_floor'][0]:.3f}) "
        f"-> {'OPEN' if door3_open else 'SHUT (the K562-specific residue does not transfer across lines)'}. "
        + (f"OPEN DOOR(S): {', '.join(doors)}." if doors else
           "ALL THREE DOORS SHUT: 0.62 is confirmed a DATA ceiling (the K562-specific residue is idiosyncratic, unreachable from inputs, "
           "not low-rank-compositional, and non-transferring across lines). v7 should NOT be a heavier retrieval/generation model -- the "
           "honest v7 is a calibrated selective-prediction wrapper on v5 that reports which predictions to trust.")
        + " Deterministic; no training.")
    print(f"\nVERDICT: {verdict}")
    if not SMOKE:
        blob = {"door1_reachability": {k: [round(A1[k][0], 4), round(A1[k][1], 4)] for k in A1},
                "door2_generation": {r: {k: [round(A2[r][k][0], 4), round(A2[r][k][1], 4)] for k in A2[r]} for r in A2},
                "door3_crossline": {k: [round(A3[k][0], 4), round(A3[k][1], 4)] for k in A3},
                "doors_open": doors, "verdict": verdict, "note": verdict}
        json.dump(blob, open(OUT / "ceiling_cartography.json", "w"), indent=1)
        print("\n  -> outputs/orphan/ceiling_cartography.json")
    else:
        print("\n[SMOKE OK]")


if __name__ == "__main__":
    main()
