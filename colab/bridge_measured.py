"""bridge_measured -- rebuild the bridge's final hop from MEASURED causal edges instead of the curated regulatory layer.

WHY. markov_bridge failed, and the diagnosis pointed at one component: its last hop ran on the curated regulatory
graph, 612,133 edges of which 91% carry no sign and none were cell-type filtered. infer_network already measured how
bad that is -- building the causal graph directly from the knockout data yields 21,228 directed edges of which
94.0% appear in NO curated layer at all (98.0% for knockout-specific edges). The databases and the measured causal
graph are, in that module's words, almost disjoint objects. So the bridge was conditioning through a map that is
mostly wrong, and the obvious repair is to condition through the map that was measured.

    middle hops   PHYSICAL PPI -- confirmed real wiring twice now, most recently at z=+29.7 against a
                  degree-preserving rewired null, so this layer is not the problem
    final hop     MEASURED CAUSAL edges: knocking out A moved B, therefore A -> B, in this exact cell line

THE CIRCULARITY THAT WOULD MAKE THIS MEANINGLESS, AND HOW IT IS BROKEN. A measured causal edge A -> B exists precisely
BECAUSE knocking out A moved B. The bridge would then nominate A as an intermediate for source G because A causally
drives G's movers -- and the validation asks whether response(A) resembles response(G). Selecting A for overlapping
with G's response and then testing whether A overlaps G's response is the same measurement twice. It would produce a
large, entirely fake effect.

THE FIX IS A DISJOINT GENE SPLIT. The readout genes are cut in half:
    HALF A   used to define the measured causal edges AND the endpoint the bridge conditions on
    HALF B   used ONLY to score response similarity in the validation
Nothing that enters the nomination is allowed to enter the test. Knockouts are ALSO split, so the causal graph is
built from TRAIN knockouts only and scored on held-out ones -- both splits are required, because gene-splitting alone
would still let a test knockout's own profile define its own edges.

WHAT IS EXPECTED TO BE HARD. The measured graph is accurate but barely walkable: only 337 of 1,971 knockouts move
enough genes to carry any edge, and only ~12% of its edges land on a gene that was itself knocked out, which is what a
second step would require. So coverage -- for how many test knockouts the measured last hop yields any candidate at
all -- is reported as a first-class number rather than hidden in an average.

THE COMPARISON IS THREE-WAY, all scored identically on half B: measured last hop, curated last hop (the arm that
failed), and forward-only PPI (the arm that beat the bridge at +0.1354 and is the one to beat)."""
import json, pickle, collections
from pathlib import Path
import numpy as np
from scipy import sparse

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
TOPM = 5
NSPLIT = 5
NREWIRE = 10
T_PPI = 1                    # PPI hops before the final causal hop (T=2 total, the depth markov_bridge selected)


def main():
    from scipy.stats import wilcoxon

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
    nontide = ~tide
    spec_n = np.array([int(((A[i] >= TAU) & nontide).sum()) for i in range(nK)])

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)
    P_pairs = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            P_pairs.add((min(a, b), max(a, b)))
    P_e = np.array(sorted(P_pairs), dtype=np.int32)
    R_src, R_dst = [], []
    for e in D.get("reg") or []:
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            R_src.append(a); R_dst.append(b)
    R_src = np.array(R_src); R_dst = np.array(R_dst)

    def rownorm(r, c, v):
        G = sparse.coo_matrix((v, (r, c)), shape=(N, N)).tocsr()
        mag = np.asarray(abs(G).sum(1)).ravel()
        inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        return sparse.diags(inv) @ G

    def make_P(pairs):
        r = np.concatenate([pairs[:, 0], pairs[:, 1]]); c = np.concatenate([pairs[:, 1], pairs[:, 0]])
        return rownorm(r, c, np.ones(len(r), np.float32))
    Pm = make_P(P_e)
    Rm_curated = rownorm(R_src, R_dst, np.ones(len(R_src), np.float32))

    konode = np.array([nidx.get(k, -1) for k in kos]); have = konode >= 0
    gnode = np.array([nidx.get(g, -1) for g in genes]); gok = gnode >= 0
    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC and have[i]])
    pdeg = np.zeros(N)
    for a, b in P_e:
        pdeg[a] += 1; pdeg[b] += 1
    print(f"{nK} knockouts x {nG} genes; {len(scor)} scorable; {len(P_e)} PPI pairs; "
          f"{len(R_src)} curated regulatory edges", flush=True)

    rng = np.random.RandomState(0)
    results = collections.defaultdict(list)
    cover = collections.defaultdict(list)
    example = None

    for sp in range(NSPLIT):
        srng = np.random.RandomState(500 + sp)
        # --- gene split: half A defines edges + endpoint, half B scores the validation. Nothing crosses. ---
        gperm = srng.permutation(np.where(nontide)[0])
        halfA = np.zeros(nG, bool); halfA[gperm[:len(gperm) // 2]] = True
        halfB = np.zeros(nG, bool); halfB[gperm[len(gperm) // 2:]] = True
        # --- knockout split ---
        kperm = srng.permutation(scor); ntest = len(scor) // 3
        test = np.array(sorted(kperm[:ntest].tolist()))
        train = np.array(sorted(set(int(x) for x in scor) - set(test.tolist())))

        # response similarity is computed on HALF B ONLY
        Mb = M[:, halfB] / (np.linalg.norm(M[:, halfB], axis=1, keepdims=True) + 1e-9)
        SIMB = (Mb @ Mb.T).astype(np.float32)

        # --- MEASURED causal graph, from TRAIN knockouts, on HALF A genes only ---
        cr, cc = [], []
        for xi in train:
            tg = np.where((A[xi] >= TAU) & halfA & gok)[0]
            if len(tg) == 0:
                continue
            cr.extend([konode[xi]] * len(tg)); cc.extend(gnode[tg].tolist())
        n_meas_edges = len(cr)
        Cm = rownorm(np.array(cr), np.array(cc), np.ones(len(cr), np.float32)) if cr else None
        n_meas_src = len(set(cr))

        # --- endpoint: the test knockout's movers, restricted to HALF A ---
        er, ec, ev = [], [], []
        for c, xi in enumerate(test):
            mv = np.where((A[xi] >= TAU) & halfA & gok)[0]
            er.extend(gnode[mv].tolist()); ec.extend([c] * len(mv)); ev.extend([1.0] * len(mv))
        End = np.asarray(sparse.coo_matrix((ev, (er, ec)), shape=(N, len(test))).todense(), dtype=np.float32)

        seed = np.zeros((N, len(test)), np.float32)
        seed[konode[test], np.arange(len(test))] = 1.0
        fwd = seed
        for _ in range(T_PPI):
            fwd = Pm.T.tocsr() @ fwd

        arms = {"measured_last_hop": (Cm @ End) if Cm is not None else np.zeros_like(fwd),
                "curated_last_hop": Rm_curated @ End}
        SC = {"forward_only": fwd}
        for nm, bwd in arms.items():
            SC[nm] = fwd * bwd

        # --- candidates are TRAIN knockouts (their responses are the held-out check) ---
        cand_ko = train
        cnode = konode[cand_ko]
        rank_of = np.empty(len(cand_ko), int)
        rank_of[np.argsort(pdeg[cnode])] = np.arange(len(cand_ko))
        order_deg = np.argsort(pdeg[cnode])

        for nm, S_ in SC.items():
            hit, ctl, ncov = [], [], 0
            for c, xi in enumerate(test):
                v = S_[cnode, c].astype(np.float64)
                top = [q for q in np.argsort(-v)[:TOPM] if v[q] > 0]
                if not top:
                    continue
                ncov += 1
                hs = [SIMB[xi, cand_ko[q]] for q in top]
                cs = []
                for q in top:
                    r = rank_of[q]; lo, hi = max(0, r - 40), min(len(cand_ko), r + 40)
                    pool = [z_ for z_ in order_deg[lo:hi] if z_ != q]
                    if pool:
                        cs.extend([SIMB[xi, cand_ko[d]] for d in
                                   srng.choice(pool, size=min(4, len(pool)), replace=False)])
                if not cs:
                    continue
                hit.append(float(np.mean(hs))); ctl.append(float(np.mean(cs)))
            hit = np.array(hit); ctl = np.array(ctl)
            results[nm].append(float((hit - ctl).mean()) if len(hit) else np.nan)
            results[nm + "_sim"].append(float(hit.mean()) if len(hit) else np.nan)
            results[nm + "_ctl"].append(float(ctl.mean()) if len(ctl) else np.nan)
            cover[nm].append(ncov / max(len(test), 1))
        print(f"  split {sp}: measured graph {n_meas_edges} edges from {n_meas_src} sources | "
              f"coverage measured {cover['measured_last_hop'][-1]:.0%} curated "
              f"{cover['curated_last_hop'][-1]:.0%} | delta measured "
              f"{results['measured_last_hop'][-1]:+.4f} curated {results['curated_last_hop'][-1]:+.4f} "
              f"forward {results['forward_only'][-1]:+.4f}", flush=True)

        if sp == 0:
            for c, xi in enumerate(test):
                if kos[xi] == "GATA1":
                    v = SC["measured_last_hop"][cnode, c]
                    top = [q for q in np.argsort(-v)[:TOPM] if v[q] > 0]
                    example = {"source": "GATA1",
                               "intermediates": [{"gene": kos[cand_ko[q]], "score": float(v[q]),
                                                  "halfB_similarity": float(SIMB[xi, cand_ko[q]])} for q in top]}
                    break

    def agg(nm):
        a = np.array(results[nm], float)
        return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum()))

    print(f"\n  THREE-WAY, {NSPLIT} splits, gene-disjoint (edges+endpoint on half A, scoring on half B)")
    print(f"    {'arm':22s} {'delta':>18s} {'similarity':>11s} {'control':>9s} {'coverage':>9s}")
    summ = {}
    for nm in ["measured_last_hop", "curated_last_hop", "forward_only"]:
        m, se = agg(nm)
        summ[nm] = {"delta": m, "se": se, "sim": float(np.nanmean(results[nm + "_sim"])),
                    "ctl": float(np.nanmean(results[nm + "_ctl"])), "coverage": float(np.mean(cover[nm]))}
        print(f"    {nm:22s} {m:+.4f} +- {se:.4f} {summ[nm]['sim']:>11.4f} {summ[nm]['ctl']:>9.4f} "
              f"{summ[nm]['coverage']:>8.0%}")

    def cmp(a, b):
        d = np.array(results[a], float) - np.array(results[b], float)
        d = d[np.isfinite(d)]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
        v = "indistinguishable" if abs(d.mean()) <= 2 * se else ("BETTER" if d.mean() > 0 else "WORSE")
        return float(d.mean()), float(se), v
    mc = cmp("measured_last_hop", "curated_last_hop")
    mf = cmp("measured_last_hop", "forward_only")
    print(f"\n  PAIRED ACROSS SPLITS")
    print(f"    measured vs curated last hop  {mc[0]:+.4f} +- {mc[1]:.4f}   {mc[2]}")
    print(f"    measured vs forward-only PPI  {mf[0]:+.4f} +- {mf[1]:.4f}   {mf[2]}")
    if example:
        print(f"\n  EXAMPLE -- GATA1, measured last hop, top intermediates (similarity scored on held-out gene half):")
        for d_ in example["intermediates"]:
            print(f"    {d_['gene']:10s} score {d_['score']:.4f}   half-B similarity to GATA1 "
                  f"{d_['halfB_similarity']:+.3f}")

    m_ = summ["measured_last_hop"]; c_ = summ["curated_last_hop"]; f_ = summ["forward_only"]
    fixes_bridge = mc[2] == "BETTER"
    beats_fwd = mf[2] == "BETTER"

    verdict = (
        f"BRIDGE WITH A MEASURED LAST HOP. markov_bridge failed and the diagnosis blamed one component: its final hop "
        f"ran on the curated regulatory layer, 91% unsigned and -- per infer_network -- 94% disjoint from the causal "
        f"edges actually measured in this cell line. So the last hop was rebuilt from MEASURED causality (knocking out "
        f"A moved B, therefore A -> B), keeping PPI for the middle because that layer is confirmed real (z=+29.7 "
        f"against a degree-preserving null). "
        f"THE CIRCULARITY THIS WOULD OTHERWISE HAVE was severe and is broken by a DISJOINT GENE SPLIT: a measured edge "
        f"exists because knocking out A moved B, so nominating A for overlapping with the source's response and then "
        f"testing that same overlap would be one measurement counted twice. Edges and the conditioning endpoint are "
        f"built on gene half A; response similarity is scored ONLY on gene half B; knockouts are split as well, so the "
        f"graph comes from train sources and is tested on held-out ones. "
        f"RESULT over {NSPLIT} splits: measured last hop {m_['delta']:+.4f} +- {m_['se']:.4f} (coverage "
        f"{m_['coverage']:.0%} of test knockouts), curated last hop {c_['delta']:+.4f} +- {c_['se']:.4f} (coverage "
        f"{c_['coverage']:.0%}), forward-only PPI {f_['delta']:+.4f} +- {f_['se']:.4f} (coverage {f_['coverage']:.0%}). "
        + (f"SWAPPING IN MEASURED CAUSALITY REPAIRS THE LAST HOP: it beats the curated version by {mc[0]:+.4f} +- "
           f"{mc[1]:.4f}, so the bridge's failure really was the map and not the method. " if fixes_bridge else
           (f"THE SWAP DOES NOT REPAIR IT: measured versus curated is {mc[0]:+.4f} +- {mc[1]:.4f} ({mc[2]}), so the "
            f"bridge's problem was NOT simply that the regulatory layer was wrong. " if mc[2] != "BETTER" else ""))
        + (f"AND IT NOW BEATS PLAIN FORWARD-ONLY PPI by {mf[0]:+.4f} +- {mf[1]:.4f}, which is the first time in this "
           f"project that conditioning on the measured endpoint has earned its place. " if beats_fwd else
           f"BUT IT STILL DOES NOT BEAT PLAIN FORWARD-ONLY PPI ({mf[0]:+.4f} +- {mf[1]:.4f}, {mf[2]}). One PPI hop "
           f"from the knocked-out gene remains the best route-finder available here, and every attempt to sharpen it "
           f"by conditioning on where the response actually landed has now failed with BOTH a curated and a measured "
           f"regulatory map. That is the stronger negative: the problem is not map quality. ")
        + f"COVERAGE IS THE STRUCTURAL LIMIT and is reported rather than averaged away: the measured causal graph has "
        f"few usable sources, so it can nominate anything at all for only {m_['coverage']:.0%} of test knockouts "
        f"against the curated layer's {c_['coverage']:.0%}. An accurate map you cannot walk on buys less than an "
        f"inaccurate one you can. "
        f"WHAT THIS CANNOT SAY: half the readout genes are spent on breaking the circularity, so every similarity here "
        f"is measured on ~3,600 genes rather than 7,223 and the numbers are not comparable to markov_bridge's. "
        f"Response similarity shows a shared route, not an ordering -- it cannot separate an intermediate from a "
        f"co-regulator. And the measured causal graph is itself thresholded response data, so 'measured' means "
        f"'reproducible in this screen', not 'mechanistically verified'. Deterministic given seed 0.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_splits": NSPLIT, "T_ppi_hops": T_PPI, "summary": summ,
               "paired": {"measured_vs_curated": {"mean": mc[0], "se": mc[1], "verdict": mc[2]},
                          "measured_vs_forward": {"mean": mf[0], "se": mf[1], "verdict": mf[2]}},
               "per_split": {k: v for k, v in results.items()}, "coverage": {k: v for k, v in cover.items()},
               "example_GATA1": example, "fixes_bridge": bool(fixes_bridge), "beats_forward": bool(beats_fwd),
               "verdict": verdict, "note": verdict}, open(OUT / "bridge_measured.json", "w"), indent=1)
    print("\n  -> outputs/orphan/bridge_measured.json")


if __name__ == "__main__":
    main()
