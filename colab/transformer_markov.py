"""transformer_markov -- give the transformer a Markov-walk neighbourhood instead of the curated partner list.

WHY THIS IS NOT A NINTH TRANSFORMER RERUN. Transformers are well explored here and the architecture is exhausted:
transformer_ko reaches 0.490 against a 0.249 tide-null and a 0.607 oracle; transformer_graphnet found that
message-passing over the curated WIRING beats a bag-of-partners by +0.004, and by +0.005 against SHUFFLED wiring of
the same density -- indistinguishable; transformer_selfgraph found a self-built kNN graph LOSES to the curated one
(0.340 vs 0.462); transformer_ko6 found regulatory token features actively HURT (-0.021). ceiling_cartography put the
remaining encoder head-room at about +0.012. So changing the ARCHITECTURE is a dead end and is not attempted.

WHAT IS NEW IS THE INPUT, AND THERE IS NOW A SPECIFIC REASON TO THINK IT WAS WRONG. transformer_graphnet concluded
"the wiring adds nothing" using the CURATED graph. This session then measured that the curated map is 94% disjoint
from the causal edges actually measured in this cell line (infer_network), and that swapping a curated layer for a
measured/walk-based one changes results enormously -- markov_bridge's curated last hop scored +0.15 where the measured
one scored +0.36. graph_null separately showed the PPI walk's advantage is REAL wiring, not hubness: scrambling
who-binds-whom while holding every degree fixed collapses it from 0.4725 to 0.1516, z=+46. So "wiring adds nothing"
may have been a statement about map QUALITY rather than about graphs, and that is testable by holding the transformer
completely fixed and changing only WHICH GENES BECOME TOKENS.

THE EXPERIMENT. Identical encoder, identical features, identical training, identical splits. Only the partner source
for each knockout's token set varies:

    curated        DepMap co-dependency -> complex co-members -> PPI neighbours   (the published 0.490 configuration)
    markov         top partners by the 2-step damped walk on the PPI graph -- the neighbourhood that scored 0.4763
                   alone in protein_chain_combine, above every hand-built component
    union          both, curated first
    markov_rewired CONTROL: the same walk on a DEGREE-PRESERVING rewired graph. If this matches the real walk, the
                   tokens are being chosen by degree and not by wiring.

Every arm indexes into ONE shared feature matrix built over the UNION token universe, so the comparison isolates token
CHOICE and cannot be confounded by different feature scaling.

NO LEAKAGE BY CONSTRUCTION: the walk uses only the static PPI graph. Unlike the measured causal graph that won in
bridge_measured, it touches no response data at all, so it needs no train/test partitioning of edges and cannot leak
the labels it is scored against."""
import json, collections, sys
from pathlib import Path
import numpy as np
from scipy import sparse
from eval_harness import Harness
from neural_ko import build_xy
import transformer_ko as TK

OUT = Path("outputs/orphan")
M_PART = TK.M_PART
SEEDS = [0, 1, 2]
WALK_STEPS, WALK_DAMP = 2, 0.5          # the configuration pre-registered by protein_chain_combine
KEY = "TRANSFORMER (self+neighbourhood)"


def build_multi_context(H, have):
    """One shared feature matrix over the union token universe; one token-index array per partner source."""
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    nidx = {n: i for i, n in enumerate(names)}
    essmap = {g["name"]: float(g.get("ess") or 0) for g in D["genes"]}

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    gi = H.gi; nG = H.nG
    rows, cols = [], []
    for g, nbs in H.nb.items():
        a = gi.get(g)
        if a is None:
            continue
        for b in nbs:
            bb = gi.get(b)
            if bb is not None:
                rows.append(a); cols.append(bb)
    Anet = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nG, nG))
    Anet = Anet.maximum(Anet.T)
    deg = np.asarray(Anet.sum(1)).ravel() + 1.0
    Dinv = csr_matrix((1.0 / np.sqrt(deg), (range(nG), range(nG))), shape=(nG, nG))
    U, S, _ = svds((Dinv @ Anet @ Dinv).astype(np.float32), k=64)
    Gemb = (U * S).astype(np.float32)
    name2net = {H.genes[i]: Gemb[i] for i in range(nG)}

    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if 0 <= a < N and 0 <= b < N and a != b:
            ppi[a].add(b); ppi[b].add(a)
    ppideg = {i: len(v) for i, v in ppi.items()}
    gcplx = {int(k): v for k, v in (D.get("gene2cplx", {}) or {}).items() if str(k).isdigit()}
    cmem = collections.defaultdict(set)
    for g, cs in gcplx.items():
        for c in cs:
            cmem[c].add(g)
    codep = {int(k): [int(j) for j, _ in v] for k, v in D.get("codep", {}).items()}

    def curated_partners(gname):
        gi_ = nidx.get(gname)
        if gi_ is None:
            return []
        seen, out = {gname}, []
        for j in codep.get(gi_, [])[:20]:
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 1))
        cp = set()
        for ci in gcplx.get(gi_, ()):
            cp.update(cmem[ci])
        for j in sorted(cp, key=lambda x: -ppideg.get(x, 0)):
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 2))
        for j in sorted(ppi.get(gi_, ()), key=lambda x: -ppideg.get(x, 0)):
            nm = names[j]
            if nm not in seen and nm in srow:
                seen.add(nm); out.append((nm, 3))
        return out[:M_PART]

    # ---------- the Markov walk over the PPI graph ----------
    pr, pc = [], []
    for a, nb in ppi.items():
        for b in nb:
            pr.append(a); pc.append(b)
    pr = np.array(pr); pc = np.array(pc)

    def walk_partner_fn(rewire_seed=None):
        r_, c_ = pr, pc
        if rewire_seed is not None:                       # degree-preserving rewire: stub matching, then clean up
            rs = np.random.RandomState(rewire_seed)
            und = np.stack([np.minimum(pr, pc), np.maximum(pr, pc)], 1)
            und = np.unique(und, axis=0)
            stubs = np.concatenate([und[:, 0], und[:, 1]]); rs.shuffle(stubs)
            h = stubs.reshape(-1, 2); keep = h[:, 0] != h[:, 1]
            e = np.unique(np.stack([np.minimum(h[keep, 0], h[keep, 1]),
                                    np.maximum(h[keep, 0], h[keep, 1])], 1), axis=0)
            r_ = np.concatenate([e[:, 0], e[:, 1]]); c_ = np.concatenate([e[:, 1], e[:, 0]])
        G = sparse.coo_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(N, N)).tocsr()
        mag = np.asarray(abs(G).sum(1)).ravel()
        inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        PT = (sparse.diags(inv) @ G).T.tocsr()
        srcs = [k for k in have if k in nidx]
        seed = np.zeros((N, len(srcs)), np.float32)
        seed[[nidx[k] for k in srcs], np.arange(len(srcs))] = 1.0
        cur = seed; tot = np.zeros_like(seed)
        for s in range(1, WALK_STEPS + 1):
            cur = PT @ cur
            tot += (WALK_DAMP ** s) * cur
        # only genes with a DepMap profile can be tokens at all
        cand = np.array([i for i in range(N) if names[i] in srow])
        sub = tot[cand]
        out = {}
        for q, k in enumerate(srcs):
            v = sub[:, q]
            nzi = np.where(v > 0)[0]
            if len(nzi) == 0:
                out[k] = []
                continue
            topi = nzi[np.argsort(-v[nzi])][:M_PART + 1]
            out[k] = [(names[cand[t]], 4) for t in topi if names[cand[t]] != k][:M_PART]
        return out

    print("  computing Markov-walk neighbourhoods (real graph)...", flush=True)
    walk_real = walk_partner_fn(None)
    print("  computing Markov-walk neighbourhoods (degree-preserving rewired control)...", flush=True)
    walk_rewired = walk_partner_fn(rewire_seed=7)

    ARMS = {}
    ARMS["curated"] = {k: curated_partners(k) for k in have}
    ARMS["markov"] = {k: walk_real.get(k, []) for k in have}
    ARMS["markov_rewired"] = {k: walk_rewired.get(k, []) for k in have}
    uni = {}
    for k in have:
        seen = {k}; out = []
        for nm, role in ARMS["curated"][k] + ARMS["markov"][k]:
            if nm not in seen:
                seen.add(nm); out.append((nm, role))
        uni[k] = out[:M_PART]
    ARMS["union"] = uni

    # ---------- ONE shared feature matrix over the union of every arm's tokens ----------
    tok_names = set(k for k in have if k in srow)
    for d in ARMS.values():
        for k in have:
            tok_names.update(nm for nm, _ in d[k])
    Gtok = sorted(tok_names); t2i = {n: i for i, n in enumerate(Gtok)}
    Fdep = np.stack([Z[srow[n]] for n in Gtok]).astype(np.float32)
    Fnet = np.stack([name2net.get(n, np.zeros(64, np.float32)) for n in Gtok]).astype(np.float32)
    Fess = np.array([[essmap.get(n, 0.0)] for n in Gtok], np.float32)
    Fdeg = np.array([[np.log1p(ppideg.get(nidx.get(n, -1), 0))] for n in Gtok], np.float32)
    zc = lambda Aa: (Aa - Aa.mean(0)) / (Aa.std(0) + 1e-6)
    Fg = np.concatenate([zc(Fdep), zc(Fnet), zc(Fess), zc(Fdeg)], axis=1).astype(np.float32)

    packed = {}
    for arm, d in ARMS.items():
        nk = len(have)
        ti = np.zeros((nk, 1 + M_PART), np.int64)
        tr = np.zeros((nk, 1 + M_PART), np.int64)
        tm = np.zeros((nk, 1 + M_PART), np.float32)
        for r, k in enumerate(have):
            ti[r, 0] = t2i.get(k, 0); tr[r, 0] = 0; tm[r, 0] = 1.0
            for c, (nm, role) in enumerate(d[k], start=1):
                ti[r, c] = t2i[nm]; tr[r, c] = min(role, 3); tm[r, c] = 1.0
        npart = tm.sum(1) - 1
        packed[arm] = (ti, tr, tm)
        print(f"    {arm:16s} partners/KO median {int(np.median(npart)):3d} mean {npart.mean():5.1f}  "
              f"{int((npart == 0).sum())} isolated", flush=True)
    return Fg, packed, len(Gtok)


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, packed, ntok = build_multi_context(H, have)
    print(f"shared token universe: {ntok} genes; {Fg.shape[1]} features\n", flush=True)

    scores = collections.defaultdict(list)
    for arm in ["curated", "markov", "union", "markov_rewired"]:
        ti, tr, tm = packed[arm]
        for s in SEEDS:
            res, _, _ = TK.run(s, H, have, Y, hidx, Fg, ti, tr, tm)
            scores[arm].append(res[KEY])
            for ref in ["TIDE-null", "ORACLE* (ceiling)", "MLP-self (feats, no neighbourhood)"]:
                if ref in res:
                    scores[f"__{ref}"].append(res[ref])
            print(f"  [{arm:15s} seed {s}] {KEY} = {res[KEY]:.4f}", flush=True)

    agg = {a: (float(np.mean(v)), float(np.std(v))) for a, v in scores.items()}
    tide = float(np.mean(scores["__TIDE-null"])); orc = float(np.mean(scores["__ORACLE* (ceiling)"]))
    mlp = float(np.mean(scores["__MLP-self (feats, no neighbourhood)"]))
    print(f"\n  TOKEN-SOURCE COMPARISON (identical encoder, features, training, splits; {len(SEEDS)} seeds)")
    print(f"    {'partner source':18s} {'recall@50':>12s} {'sd':>7s} {'head-room closed':>18s}")
    for a in ["curated", "markov", "union", "markov_rewired"]:
        m, sd = agg[a]
        print(f"    {a:18s} {m:>12.4f} {sd:>7.4f} {(m-tide)/max(orc-tide,1e-9):>17.0%}")
    print(f"    {'-- references --':18s}")
    print(f"    {'TIDE-null':18s} {tide:>12.4f}")
    print(f"    {'MLP-self':18s} {mlp:>12.4f}")
    print(f"    {'ORACLE*':18s} {orc:>12.4f}")

    def paired(a, b):
        d = np.array(scores[a]) - np.array(scores[b])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
        v = "indistinguishable" if abs(d.mean()) <= 2 * se else ("BETTER" if d.mean() > 0 else "WORSE")
        return float(d.mean()), float(se), v
    mc = paired("markov", "curated"); uc = paired("union", "curated"); mr = paired("markov", "markov_rewired")
    print(f"\n  PAIRED ACROSS SEEDS (a claim needs |diff| > 2 SE)")
    for lab, cc in [("markov - curated", mc), ("union - curated", uc), ("markov - rewired control", mr)]:
        print(f"    {lab:26s} {cc[0]:+.4f} +- {cc[1]:.4f}   {cc[2]}")

    best = max(["curated", "markov", "union"], key=lambda a: agg[a][0])
    walk_helps = mc[2] == "BETTER" or uc[2] == "BETTER"
    wiring_real = mr[2] == "BETTER"

    verdict = (
        f"TRANSFORMER x MARKOV WALK: change the TOKENS, not the architecture. The architecture here is already "
        f"exhausted -- transformer_graphnet got +0.004 from message-passing over curated wiring and +0.005 against "
        f"SHUFFLED wiring of the same density, transformer_selfgraph found a self-built kNN graph LOSES to curated "
        f"(0.340 vs 0.462), transformer_ko6 found regulatory features HURT by 0.021, and ceiling_cartography put the "
        f"remaining encoder head-room near +0.012 -- so no architectural change was attempted. What IS new is a "
        f"specific reason to doubt the INPUT: transformer_graphnet's 'wiring adds nothing' was measured on the CURATED "
        f"graph, and this session established that the curated map is 94% disjoint from measured causality, while the "
        f"PPI walk's advantage is real wiring rather than hubness (collapses 0.4725 -> 0.1516 under a "
        f"degree-preserving rewire, z=+46). So only the partner source varies, over one shared feature matrix built on "
        f"the union token universe. "
        f"RESULT ({len(SEEDS)} seeds, tide-null {tide:.3f}, oracle {orc:.3f}): curated {agg['curated'][0]:.4f}, "
        f"markov-walk {agg['markov'][0]:.4f}, union {agg['union'][0]:.4f}, rewired-walk control "
        f"{agg['markov_rewired'][0]:.4f}. "
        + (f"THE WALK NEIGHBOURHOOD HELPS: markov - curated = {mc[0]:+.4f} +- {mc[1]:.4f} ({mc[2]}), union - curated = "
           f"{uc[0]:+.4f} +- {uc[1]:.4f} ({uc[2]}). So transformer_graphnet's negative WAS about map quality, not "
           f"about graphs -- feeding the encoder a walk-derived neighbourhood beats the curated partner list it was "
           f"given. " if walk_helps else
           f"THE WALK NEIGHBOURHOOD DOES NOT HELP: markov - curated = {mc[0]:+.4f} +- {mc[1]:.4f} ({mc[2]}) and union "
           f"- curated = {uc[0]:+.4f} +- {uc[1]:.4f} ({uc[2]}). transformer_graphnet's finding therefore stands and "
           f"generalises -- for THIS encoder a knockout is well described as a bag of partners, and it does not matter "
           f"much which sensible graph supplies them. The walk's large advantage as a standalone retrieval scorer "
           f"(0.4763, above every hand-built component) does NOT transfer into the transformer, because the encoder "
           f"was already extracting that information from the curated bag. ")
        + (f"The rewired control confirms the tokens are chosen by wiring rather than degree "
           f"({mr[0]:+.4f} +- {mr[1]:.4f}). " if wiring_real else
           f"AND THE REWIRED CONTROL IS NOT BEATEN ({mr[0]:+.4f} +- {mr[1]:.4f}, {mr[2]}): a degree-preserving "
           f"rewired walk supplies tokens as good as the real one, so within the transformer the neighbourhood is "
           f"acting as a degree-matched sample of plausible genes rather than as specific biology -- which is exactly "
           f"what transformer_ko's own random-partner control was built to detect. ")
        + f"Best configuration: {best} at {agg[best][0]:.4f}, closing "
        f"{(agg[best][0]-tide)/max(orc-tide,1e-9):.0%} of the tide-to-oracle head-room. "
        f"NO LEAKAGE BY CONSTRUCTION: the walk uses only the static PPI graph and touches no response data, unlike the "
        f"measured causal graph that won in bridge_measured, so it needs no edge-level train/test partitioning. "
        f"WHAT THIS CANNOT SAY: three seeds is a small paired sample and differences below ~0.01 are not resolvable "
        f"here. This tests the WALK neighbourhood, not the MEASURED-CAUSAL one that won in bridge_measured -- that "
        f"would require rebuilding the token context per split to avoid leaking response data into the tokens, which "
        f"is the obvious follow-up and is not done here. Deterministic given the fixed seeds.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"seeds": SEEDS, "walk_steps": WALK_STEPS, "walk_damping": WALK_DAMP,
               "scores": {a: v for a, v in scores.items()},
               "aggregate": {a: {"mean": agg[a][0], "sd": agg[a][1]} for a in
                             ["curated", "markov", "union", "markov_rewired"]},
               "references": {"tide": tide, "oracle": orc, "mlp_self": mlp},
               "paired": {"markov_minus_curated": {"mean": mc[0], "se": mc[1], "verdict": mc[2]},
                          "union_minus_curated": {"mean": uc[0], "se": uc[1], "verdict": uc[2]},
                          "markov_minus_rewired": {"mean": mr[0], "se": mr[1], "verdict": mr[2]}},
               "walk_helps": bool(walk_helps), "wiring_beats_rewired": bool(wiring_real), "best_arm": best,
               "verdict": verdict, "note": verdict}, open(OUT / "transformer_markov.json", "w"), indent=1)
    print("\n  -> outputs/orphan/transformer_markov.json")


if __name__ == "__main__":
    main()
