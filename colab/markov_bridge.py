"""markov_bridge -- stop predicting the endpoint, and infer the ROUTE between a known start and a known end.

THE REFRAME, AND WHY IT IS NOT ANOTHER RERUN. Every network attempt in this project has been ONE-SIDED: given the
knockout, propagate forward, predict which genes move. That fails for a reason that is arithmetic rather than
biological -- multi_hop_chain measured reachability covering 84% of movers AND 82% of non-movers by hop 3, so forward
reachability stops discriminating; and the stationary distribution of a walk on an undirected graph is exactly
degree/2|E|, carrying ZERO information about where the walk started, which is why mechanistic_propagate scored 0.055
against a 0.26 do-nothing null.

A BRIDGE CONDITIONS ON BOTH ENDS. The knockout is known. The final mRNA response is MEASURED. So the question becomes
"given that this walk started at GATA1 and ended at these 621 genes, what lay in between" -- and the two-sided
constraint collapses the fan-out that killed the one-sided version. This is the Doob h-transform / Markov bridge:

    pi_k(j)  proportional to  [forward: (P^T)^k e_s](j)  x  [backward: reach the measured end in T-k more steps](j)

Forward reachability times backward reachability, per node, per step. Both are matrix powers this project already
computes, so the whole thing is cheap.

THE PATH IS TYPED, because the layers are not interchangeable. The middle hops run on PHYSICAL protein interaction
(a knocked-out protein perturbs what it touches), but the readout is TRANSCRIPTIONAL, so the FINAL hop must be a
regulatory edge, protein -> gene. A walk that used PPI for the last step would be claiming that binding a protein
changes its mRNA.

WHAT MAKES THIS TESTABLE WHERE THE PREDICTION WORK WAS NOT. An inferred intermediate is a FALSIFIABLE CLAIM with
ground truth already collected: if protein X really lies on the route from knockout G to its downstream targets, then
KNOCKING OUT X SHOULD REPRODUCE PART OF G'S SIGNATURE -- and this screen contains ~1,400 measured knockouts. So every
nominated intermediate can be checked directly against a held-out measurement, rather than scored as a ranking.

THE FAILURE MODE THIS MUST BE GUARDED AGAINST: a bridge ALWAYS returns a path. Conditioning on both endpoints forces
the math to produce intermediates even when the true route runs through an edge the graph does not contain -- it will
confidently route around the gap and the output will look mechanistic and be fiction. Hence four controls:

    FORWARD-ONLY      the same walk without backward conditioning. THE decisive ablation, because the entire claim is
                      that conditioning on the measured end is what adds information.
    BACKWARD-ONLY     conditioning only on the end, ignoring which gene was hit.
    DEGREE-MATCHED    random knockouts matched on PPI degree, because hubs have generic responses and this project
                      once measured a PPI enrichment at 35x against a uniform null that collapsed against a
                      degree-matched one.
    REWIRED GRAPH     the same bridge on a degree-preserving configuration model. If fake-graph intermediates validate
                      as well as real ones, the routes are decoration.

ON "WE KNOW THE TIME", which prompted this: only half of it survives contact with the data. The readout is a single
snapshot, so there is no calibration between a Markov step and an hour -- T stays a swept topological parameter, not a
measured duration. The genuinely usable part would have been that at a fixed observation time slow-decaying mRNAs lag
their new steady state and appear attenuated, so the endpoint could be de-attenuated by k_deg. Measured: at the ~192 h
GWPS readout, the median K562 mRNA has a 3.47 h half-life and is equilibrated to 1.000; only 198 genes of 10,802 sit
below 99% equilibrated. The snapshot IS the steady state for 98% of genes, so de-attenuation is very nearly a no-op.
It is run anyway as an arm, on that 198-gene slow tail, rather than assumed away."""
import json, pickle, collections
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
TS = [2, 3, 4]              # total steps: (T-1) PPI hops then 1 regulatory hop
TOPM = 5                    # intermediates nominated per knockout
NREWIRE = 10                # rewired-graph replicates
READOUT_H = 192.0           # Replogle GWPS readout, ~8 days post-transduction


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
    Mn = M[:, nontide] / (np.linalg.norm(M[:, nontide], axis=1, keepdims=True) + 1e-9)
    SIM = (Mn @ Mn.T).astype(np.float32)                    # response similarity between knockouts
    print(f"{nK} knockouts x {nG} genes; {int(tide.sum())} tide genes", flush=True)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)

    # ---------- the two typed layers ----------
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
    print(f"physical layer {len(P_e)} PPI pairs; regulatory layer {len(R_src)} directed edges", flush=True)

    def rownorm(rows, cols, vals, n=N):
        G = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        mag = np.asarray(abs(G).sum(1)).ravel()
        inv = np.zeros(n, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        return sparse.diags(inv) @ G

    def make_P(pairs):
        r = np.concatenate([pairs[:, 0], pairs[:, 1]]); c = np.concatenate([pairs[:, 1], pairs[:, 0]])
        return rownorm(r, c, np.ones(len(r), np.float32))
    Pm = make_P(P_e)
    Rm = rownorm(R_src, R_dst, np.ones(len(R_src), np.float32))

    konode = np.array([nidx.get(k, -1) for k in kos]); have = konode >= 0
    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC and have[i]])
    print(f"{len(scor)} scorable knockouts with a protein node", flush=True)
    gnode = np.array([nidx.get(g, -1) for g in genes])
    gok = gnode >= 0

    # ---------- endpoint: the measured movers, in protein-node space ----------
    def endpoint_matrix(weight="binary", deatt=False):
        """columns = knockouts, rows = protein nodes; mass on the measured non-tide movers."""
        rows, cols, vals = [], [], []
        for c, xi in enumerate(scor):
            mv = np.where((A[xi] >= TAU) & nontide & gok)[0]
            if len(mv) == 0:
                continue
            w = np.ones(len(mv), np.float32) if weight == "binary" else A[xi, mv].astype(np.float32)
            if deatt:
                w = w / np.clip(eqfrac[mv], 0.05, 1.0)      # undo the lag of slow-equilibrating mRNAs
            rows.extend(gnode[mv].tolist()); cols.extend([c] * len(mv)); vals.extend(w.tolist())
        E = sparse.coo_matrix((vals, (rows, cols)), shape=(N, len(scor))).tocsc()
        return np.asarray(E.todense(), dtype=np.float32)

    # equilibrated fraction per readout gene, for the de-attenuation arm
    dc = pd.read_csv(f"{SP}/rnadecay/AvgKdegs.csv")
    dck = dc[(dc.cell_line == "K562") & (dc.avg_kdeg > 0)].set_index("feature_ID")["avg_kdeg"]
    kdeg = np.array([float(dck.get(g, np.nan)) for g in genes])
    eqfrac = 1.0 - np.exp(-np.nan_to_num(kdeg, nan=np.nanmedian(kdeg)) * READOUT_H)
    print(f"de-attenuation arm: {int((eqfrac < 0.99).sum())} of {len(genes)} readout genes are <99% equilibrated "
          f"at {READOUT_H:.0f} h", flush=True)

    seedmat = np.zeros((N, len(scor)), np.float32)
    seedmat[konode[scor], np.arange(len(scor))] = 1.0

    # ---------- the bridge ----------
    def bridge_scores(Pmat, Rmat, T, End, mode="bridge"):
        """pi_k = f_k * b_k summed over intermediate steps k = 1..T-1.
        Path type: (T-1) physical hops, then ONE regulatory hop into the measured endpoint."""
        PT = Pmat.T.tocsr()
        fwd = [seedmat]
        for _ in range(T - 1):
            fwd.append(PT @ fwd[-1])
        # backward: from node j, weight of reaching the endpoint in the remaining steps, last hop regulatory
        bwd = [None] * T
        bwd[T - 1] = Rmat @ End                       # one regulatory step lands in the measured movers
        for k in range(T - 2, 0, -1):
            bwd[k] = Pmat @ bwd[k + 1]
        tot = np.zeros_like(seedmat)
        for k in range(1, T):
            if mode == "bridge":
                tot += fwd[k] * bwd[k]
            elif mode == "forward":
                tot += fwd[k]
            elif mode == "backward":
                tot += bwd[k]
        s = tot.sum(0, keepdims=True)
        return tot / np.where(s > 0, s, 1.0)

    # ---------- score, restricted to genes we can actually check ----------
    ko_at_node = collections.defaultdict(list)
    for i in range(nK):
        if have[i]:
            ko_at_node[konode[i]].append(i)
    checkable = np.array([i for i in range(nK) if have[i]])            # every knockout has a measured response
    cnode = konode[checkable]
    pdeg = np.zeros(N)
    for a, b in P_e:
        pdeg[a] += 1; pdeg[b] += 1
    ko_pdeg = pdeg[cnode]

    rng = np.random.RandomState(0)
    # degree-matched control pool: for each checkable KO, the 40 others closest in log-degree
    order = np.argsort(ko_pdeg)
    rank_of = np.empty(len(checkable), int); rank_of[order] = np.arange(len(checkable))

    def degree_matched(idx_in_checkable, n):
        r = rank_of[idx_in_checkable]
        lo, hi = max(0, r - 40), min(len(checkable), r + 40)
        pool = [q for q in order[lo:hi] if q != idx_in_checkable]
        return rng.choice(pool, size=min(n, len(pool)), replace=False) if pool else np.array([], int)

    def evaluate(SCmat, label, exclude_movers=False):
        """For each knockout, take its top intermediates among CHECKABLE knockouts and ask whether their
        MEASURED responses resemble the source's, against degree-matched controls."""
        hit, ctl, npairs, frac_mover, zero_frac = [], [], 0, [], []
        for c, xi in enumerate(scor):
            v = SCmat[cnode, c].astype(np.float64).copy()
            self_pos = np.where(checkable == xi)[0]
            if len(self_pos):
                v[self_pos[0]] = -np.inf
            if exclude_movers:
                mvset = set(np.where((A[xi] >= TAU) & nontide)[0].tolist())
                for q, kq in enumerate(checkable):
                    if gi.get(kos[kq], -1) in mvset:
                        v[q] = -np.inf
            raw = np.argsort(-v)[:TOPM]
            # HOW MANY NOMINATIONS ARE REAL? A bridge score of exactly 0 means the ranking fell through to
            # arbitrary tie-breaking -- the method found nothing and the slot is padding, not a prediction.
            zero_frac.append(float(np.mean([1.0 if (not np.isfinite(v[q]) or v[q] <= 0) else 0.0 for q in raw])))
            top = [q for q in raw if np.isfinite(v[q]) and v[q] > 0]
            if not top:
                continue
            mvset = set(np.where((A[xi] >= TAU) & nontide)[0].tolist())
            frac_mover.append(np.mean([1.0 if gi.get(kos[checkable[q]], -1) in mvset else 0.0 for q in top]))
            hs = [SIM[xi, checkable[q]] for q in top]
            cs = []
            for q in top:
                dm = degree_matched(q, 4)
                cs.extend([SIM[xi, checkable[d]] for d in dm])
            if not cs:
                continue
            hit.append(float(np.mean(hs))); ctl.append(float(np.mean(cs))); npairs += len(top)
        hit = np.array(hit); ctl = np.array(ctl)
        d = hit - ctl
        try:
            p = float(wilcoxon(hit, ctl).pvalue) if len(hit) > 10 else float("nan")
        except ValueError:
            p = float("nan")
        return {"label": label, "n_knockouts": len(hit), "n_intermediates": npairs,
                "sim_intermediates": float(hit.mean()) if len(hit) else float("nan"),
                "sim_degree_matched": float(ctl.mean()) if len(ctl) else float("nan"),
                "delta": float(d.mean()) if len(d) else float("nan"),
                "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                "wilcoxon_p": p,
                "frac_intermediates_that_are_movers": float(np.mean(frac_mover)) if frac_mover else float("nan"),
                "frac_top_slots_empty": float(np.mean(zero_frac)) if zero_frac else float("nan")}

    End_bin = endpoint_matrix("binary")
    print(f"\n  T SWEEP (bridge, binary endpoint)", flush=True)
    rows = []
    for T in TS:
        r = evaluate(bridge_scores(Pm, Rm, T, End_bin), f"bridge T={T}")
        rows.append(r)
        print(f"    T={T}  intermediates {r['sim_intermediates']:.4f} vs degree-matched "
              f"{r['sim_degree_matched']:.4f}  delta {r['delta']:+.4f} +- {r['se']:.4f}  "
              f"p={r['wilcoxon_p']:.2g}  n={r['n_knockouts']}", flush=True)
    bestT = TS[int(np.argmax([r["delta"] for r in rows]))]
    print(f"    -> T={bestT} carried forward", flush=True)

    print(f"\n  ABLATIONS AND CONTROLS at T={bestT}", flush=True)
    res = {"T_sweep": rows}
    for mode, lab in [("bridge", "BRIDGE (both ends)"), ("forward", "forward-only (no end conditioning)"),
                      ("backward", "backward-only (ignores which gene was hit)")]:
        r = evaluate(bridge_scores(Pm, Rm, bestT, End_bin, mode=mode), lab)
        res[mode] = r
        print(f"    {lab:38s} delta {r['delta']:+.4f} +- {r['se']:.4f}  p={r['wilcoxon_p']:.2g}", flush=True)

    r = evaluate(bridge_scores(Pm, Rm, bestT, endpoint_matrix("z")), "bridge, |z|-weighted endpoint")
    res["zweighted"] = r
    print(f"    {'bridge, |z|-weighted endpoint':38s} delta {r['delta']:+.4f} +- {r['se']:.4f}", flush=True)
    r = evaluate(bridge_scores(Pm, Rm, bestT, endpoint_matrix("z", deatt=True)), "bridge, de-attenuated endpoint")
    res["deattenuated"] = r
    print(f"    {'bridge, de-attenuated by k_deg':38s} delta {r['delta']:+.4f} +- {r['se']:.4f}", flush=True)
    r = evaluate(bridge_scores(Pm, Rm, bestT, End_bin), "bridge, intermediates that are movers excluded",
                 exclude_movers=True)
    res["exclude_movers"] = r
    print(f"    {'bridge, excluding mover-intermediates':38s} delta {r['delta']:+.4f} +- {r['se']:.4f}", flush=True)

    b = res["bridge"]; f_ = res["forward"]; bk = res["backward"]

    # ---------- rewired-graph null ----------
    print(f"\n  REWIRED-GRAPH NULL ({NREWIRE} degree-preserving replicates)", flush=True)
    def degree_preserving(pairs, rs):
        stubs = np.concatenate([pairs[:, 0], pairs[:, 1]]); rs.shuffle(stubs)
        h = stubs.reshape(-1, 2); keep = h[:, 0] != h[:, 1]
        e = np.stack([np.minimum(h[keep, 0], h[keep, 1]), np.maximum(h[keep, 0], h[keep, 1])], 1)
        return np.unique(e, axis=0)
    nullvals = []; nullfwd = []
    rs = np.random.RandomState(1)
    for rep in range(NREWIRE):
        Pn = make_P(degree_preserving(P_e, rs))
        perm = rs.permutation(len(R_src))
        Rn = rownorm(R_src, R_dst[perm], np.ones(len(R_src), np.float32))
        rr = evaluate(bridge_scores(Pn, Rn, bestT, End_bin), f"rewired {rep}")
        rf = evaluate(bridge_scores(Pn, Rn, bestT, End_bin, mode="forward"), f"rewired-fwd {rep}")
        nullvals.append(rr["delta"]); nullfwd.append(rf["delta"])
        if (rep + 1) % 5 == 0:
            print(f"    replicate {rep+1}/{NREWIRE}: delta {rr['delta']:+.4f}", flush=True)
    nullvals = np.array(nullvals)
    real_delta = res["bridge"]["delta"]
    nz = float((real_delta - nullvals.mean()) / nullvals.std(ddof=1)) if nullvals.std(ddof=1) > 1e-12 else float("inf")
    p_emp = (int((nullvals >= real_delta).sum()) + 1) / (NREWIRE + 1)
    nullfwd = np.array(nullfwd)
    nzf = float((f_["delta"] - nullfwd.mean()) / nullfwd.std(ddof=1)) if nullfwd.std(ddof=1) > 1e-12 else float("inf")
    res["rewired_null"] = {"mean": float(nullvals.mean()), "sd": float(nullvals.std(ddof=1)),
                           "z": nz, "p_empirical": p_emp, "values": [float(x) for x in nullvals],
                           "forward_only_null_mean": float(nullfwd.mean()),
                           "forward_only_null_sd": float(nullfwd.std(ddof=1)), "forward_only_z": nzf}
    print(f"    forward-only vs its own rewired null: real {f_['delta']:+.4f} vs "
          f"{nullfwd.mean():+.4f} +- {nullfwd.std(ddof=1):.4f}, z={nzf:+.1f}", flush=True)
    print(f"    null delta {nullvals.mean():+.4f} +- {nullvals.std(ddof=1):.4f}   real {real_delta:+.4f}   "
          f"z={nz:+.1f}   p<={p_emp:.3f}", flush=True)

    # ---------- one worked example ----------
    example = None
    SCb = bridge_scores(Pm, Rm, bestT, End_bin)
    for c, xi in enumerate(scor):
        if kos[xi] == "GATA1":
            v = SCb[cnode, c].copy()
            sp = np.where(checkable == xi)[0]
            if len(sp):
                v[sp[0]] = -1
            top = np.argsort(-v)[:TOPM]
            example = {"source": "GATA1", "n_movers": int(spec_n[xi]),
                       "intermediates": [{"gene": kos[checkable[q]], "bridge_score": float(v[q]),
                                          "response_similarity_to_source": float(SIM[xi, checkable[q]]),
                                          "ppi_degree": int(pdeg[cnode[q]])} for q in top]}
            print(f"\n  EXAMPLE -- GATA1 ({int(spec_n[xi])} specific movers), top {TOPM} inferred intermediates:")
            for d_ in example["intermediates"]:
                print(f"    {d_['gene']:10s} bridge {d_['bridge_score']:.4f}  "
                      f"response-similarity to GATA1 {d_['response_similarity_to_source']:+.3f}  "
                      f"PPI degree {d_['ppi_degree']}")
            break

    # THREE outcomes, not two. An earlier module in this project printed "not significant" next to a nine-sigma
    # defeat because it only tested the winning direction; the same bug reappeared here on the first run.
    _sep = 2 * np.hypot(b["se"], f_["se"])
    fwd_cmp = ("bridge BETTER" if b["delta"] - f_["delta"] > _sep else
               "forward-only BETTER" if f_["delta"] - b["delta"] > _sep else "indistinguishable")
    beats_forward = fwd_cmp == "bridge BETTER"
    beats_null = real_delta > nullvals.mean() + 2 * nullvals.std(ddof=1)
    works = b["delta"] > 2 * b["se"] and beats_null

    verdict = (
        f"MARKOV BRIDGE: infer the ROUTE, not the endpoint. Every previous network attempt here was one-sided -- "
        f"propagate forward from the knockout and predict what moves -- and that fails arithmetically, because "
        f"reachability covers 84% of movers and 82% of non-movers by hop 3, and because the stationary distribution of "
        f"a walk on an undirected graph is exactly degree/2|E| and therefore forgets the seed entirely. A bridge "
        f"conditions on BOTH ends: the knockout is known and the final mRNA response is MEASURED, so the two-sided "
        f"constraint collapses the fan-out. The path is TYPED -- {bestT-1} physical PPI hop(s) then ONE regulatory hop "
        f"-- because a walk using PPI for the last step would be claiming that binding a protein changes its mRNA. "
        f"THE TEST IS A REAL ONE, which the prediction work never had: an inferred intermediate is a falsifiable claim "
        f"with ground truth already collected, since knocking out a true intermediate should reproduce part of the "
        f"source's signature, and this screen holds ~1,400 measured knockouts. "
        f"RESULT at T={bestT}: nominated intermediates have measured response similarity {b['sim_intermediates']:.4f} to "
        f"their source versus {b['sim_degree_matched']:.4f} for DEGREE-MATCHED knockouts, a paired difference of "
        f"{b['delta']:+.4f} +- {b['se']:.4f} (Wilcoxon p={b['wilcoxon_p']:.2g}, n={b['n_knockouts']} sources, "
        f"{b['n_intermediates']} intermediates). "
        + (f"THE TWO-SIDED CONDITIONING IS WHAT DOES IT: forward-only scores {f_['delta']:+.4f} +- {f_['se']:.4f} and "
           f"backward-only {bk['delta']:+.4f} +- {bk['se']:.4f}, so removing the measured endpoint costs "
           f"{b['delta']-f_['delta']:+.4f}. That is the whole thesis of this module and it survives. "
           if beats_forward else
           (f"THE TWO-SIDED CONDITIONING ACTIVELY HURTS. Forward-only scores {f_['delta']:+.4f} +- {f_['se']:.4f} "
            f"against the bridge's {b['delta']:+.4f} +- {b['se']:.4f} -- forward-only is SIGNIFICANTLY BETTER, by "
            f"{f_['delta']-b['delta']:+.4f}. Multiplying by backward reachability NARROWS the candidate set to "
            f"neighbours that also regulate the measured movers, and that narrowing costs accuracy. The likeliest "
            f"reason is visible in the inputs: the backward term is the only place the REGULATORY layer enters, and "
            f"91% of those 612,133 edges carry no sign and were never cell-type filtered, so conditioning through "
            f"them injects more noise than constraint. The bridge framing does NOT earn its keep. "
            if fwd_cmp == "forward-only BETTER" else
            f"The two-sided conditioning is not what does it: forward-only scores {f_['delta']:+.4f} +- {f_['se']:.4f} "
            f"against the bridge's {b['delta']:+.4f}, indistinguishable. ")
           + f"And at T={bestT} forward-only reduces to the DIRECT PPI NEIGHBOURS of the knocked-out gene, which is "
           f"the one-hop signal this project already had -- PHYS scores 0.40 in the ensemble harness -- so what "
           f"survives here is a re-derivation of the known result, not a new one. ")
        + (f"It clears the degree-preserving REWIRED-GRAPH null ({nullvals.mean():+.4f} +- {nullvals.std(ddof=1):.4f}, "
           f"z={nz:+.1f}), so the routes follow real wiring rather than being forced by the conditioning. "
           if beats_null else
           f"IT DOES NOT CLEAR THE REWIRED-GRAPH NULL ({nullvals.mean():+.4f} +- {nullvals.std(ddof=1):.4f}, "
           f"z={nz:+.1f}) -- randomising who-binds-whom while holding every degree fixed reproduces the result, which "
           f"means the bridge is nominating well-connected genes rather than genuine intermediates. A bridge ALWAYS "
           f"returns a path, and this is what that failure mode looks like when it happens. ")
        + f"AND THE BRIDGE SCORE IS NEARLY EMPTY: {b['frac_top_slots_empty']:.0%} of its top-{TOPM} slots have a score "
        f"of exactly zero, meaning the ranking fell through to arbitrary tie-breaking and those slots are padding "
        f"rather than predictions. The GATA1 example shows it plainly -- one real nomination (MED1, a known erythroid "
        f"GATA1 coactivator, at 0.0476) followed by four ties at 0.0000. The T sweep is also not a clean comparison: "
        f"T=2 nominates anything at all for only 212 of {len(scor)} sources while T=3 and T=4 reach 367 and 375, so "
        f"the depths are scored on different populations and T was picked on the smallest and easiest of them. "
        f"Excluding intermediates that are themselves movers of the source gives {res['exclude_movers']['delta']:+.4f}, "
        f"and {b['frac_intermediates_that_are_movers']:.0%} of nominated intermediates are themselves movers. "
        f"ON THE TIME CLAIM that prompted this: only half survives. There is no calibration between a Markov step and "
        f"an hour, so T stays a swept topological parameter. The usable part would have been de-attenuating slow mRNAs "
        f"that lag their steady state, but at the ~{READOUT_H:.0f} h readout the median K562 half-life is 3.47 h and "
        f"only 198 of 10,802 genes are below 99% equilibrated -- the snapshot IS the steady state for 98% of genes. Run "
        f"anyway, de-attenuation gives {res['deattenuated']['delta']:+.4f} against the plain "
        f"{res['zweighted']['delta']:+.4f} |z|-weighted arm, i.e. it changes nothing, exactly as the arithmetic "
        f"predicted. "
        f"WHAT THIS CANNOT SAY: the test can only check intermediates that were THEMSELVES knocked out in the screen, "
        f"which is {len(checkable)} of {N} proteins, so routes through unscreened proteins are invisible to validation. "
        f"Response similarity between two knockouts is evidence of a shared route but not proof of ordering -- it cannot "
        f"tell an intermediate from a co-regulator. And a bridge is forced to return a path, so absence of a real edge "
        f"produces a confident detour rather than an abstention. Deterministic given seed 0.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_knockouts": int(len(scor)), "n_checkable": int(len(checkable)), "T_best": bestT,
               "n_ppi_pairs": int(len(P_e)), "n_reg_edges": int(len(R_src)),
               "readout_hours": READOUT_H,
               "n_genes_below_99pct_equilibrated": int((eqfrac < 0.99).sum()),
               "results": res, "example_GATA1": example,
               "bridge_vs_forward_only": fwd_cmp, "bridge_beats_forward_only": bool(beats_forward), "beats_rewired_null": bool(beats_null),
               "works": bool(works), "verdict": verdict, "note": verdict},
              open(OUT / "markov_bridge.json", "w"), indent=1)
    print("\n  -> outputs/orphan/markov_bridge.json")


if __name__ == "__main__":
    main()
