"""Answers the gates predeclared in db5_unselected.py. No bar is defined in this file.

Every threshold used here is quoted from that module's docstring, which was committed before
the run. This file is arithmetic on the shards, not a place to decide what counts as a pass.
"""
from __future__ import annotations
import glob, json, sys
sys.path.insert(0, ".")
import numpy as np

OK = ("high", "medium", "acceptable")
TOPK = 20                      # "reaches the top 20", from the gate
CHANCE_SD = 2.0                # "exceeds chance by >= 2 sd", from the gate
Q1_RHO, Q1_P = -0.10, 1e-3     # the ORIGINAL Q1 bars, unchanged, so this is a replication


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def spearman_p(rho, n):
    if not np.isfinite(rho) or n < 10:
        return float("nan")
    from math import erfc, sqrt
    return float(erfc(abs(rho) * sqrt(n - 1) / sqrt(2.0)))


def partial(a, b, c):
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    ra, rb, rc = ra - ra.mean(), rb - rb.mean(), rc - rc.mean()
    def cr(x, y):
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return (x * y).sum() / d if d > 0 else float("nan")
    rab, rac, rbc = cr(ra, rb), cr(ra, rc), cr(rb, rc)
    return float((rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2)))


def chance_hit(N, k, top=TOPK):
    """P(a random top-`top` of N contains >=1 of the k acceptable) = 1 - C(N-k,top)/C(N,top).

    Computed as a running product so it is exact for the N and k that occur here and never
    forms a large binomial coefficient.
    """
    if k <= 0 or N <= 0:
        return 0.0
    t = min(top, N)
    q = 1.0
    for j in range(t):
        num = N - k - j
        if num <= 0:
            return 1.0
        q *= num / (N - j)
    return float(1.0 - q)


def load(pattern="benchmarks/unsel_w*.json"):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out += json.load(open(f))
        except Exception:
            pass
    return out


def rank_norm(x, descending=False):
    r = _rank(np.asarray(x, float))
    if descending:
        r = len(r) - 1 - r
    return r / max(1.0, len(r) - 1)


def main():
    data = load(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/unsel_w*.json")
    data = [c for c in data if len(c["poses"]) >= 20]
    if not data:
        print("  no shards")
        return 1
    npose = sum(len(c["poses"]) for c in data)
    print(f"  {len(data)} complexes, {npose} sampled poses "
          f"(median {int(np.median([len(c['poses']) for c in data]))}/complex)\n")

    # ---------------- screen validation (ledger defect M) ----------------
    dI = max(c["validate"]["max_dI"] for c in data)
    dL = max(c["validate"]["max_dL"] for c in data)
    unsafe = [c["id"] for c in data if not c["validate"]["safe"]]
    print("  SCREEN VALIDATION  (the ceiling screen is a claimed bound, so it is checked)")
    print(f"      max |dI_rmsd| cheap-vs-exact  {dI:.4f} A   (margin 2.0)")
    print(f"      max |dL_rmsd| cheap-vs-exact  {dL:.4f} A   (margin 5.0)")
    print(f"      complexes flagged unsafe      {len(unsafe)}"
          f"{'  ' + ','.join(unsafe[:6]) if unsafe else ''}")
    if unsafe:
        print("      -> the ceiling below is NOT trustworthy on those complexes.")

    # ---------------- STEP 0: the unselected ceiling ----------------
    print("\n  STEP 0  THE UNSELECTED CEILING -- did the SEARCH ever generate an "
          "acceptable pose?")
    nacc = np.array([c["n_acceptable_full"] for c in data])
    ncand = np.array([c["ceiling"]["n_candidates"] for c in data])
    nscr = np.array([c["ceiling"]["n_screened"] for c in data])
    ceil = int((nacc > 0).sum())
    print(f"      candidates per complex        {int(np.median(ncand))} (median), "
          f"{int(ncand.sum())} total")
    print(f"      passed the RMSD screen        {int(nscr.sum())} total "
          f"({int(np.median(nscr))} median per complex)")
    print(f"      CEILING                       {ceil}/{len(data)} complexes have >= 1 "
          f"CAPRI-acceptable pose ANYWHERE in the search output")
    if ceil:
        w = nacc[nacc > 0]
        print(f"      acceptable poses per such complex   min {w.min()}, median "
              f"{int(np.median(w))}, max {w.max()}")
    bi = np.array([min(p["I_rmsd"] for p in c["poses"]) for c in data])
    print(f"      best I_rmsd in the sample     min {bi.min():.2f}, median "
          f"{np.median(bi):.2f}, max {bi.max():.2f} A")

    # ---------------- STEP 2: Q1 off the collider ----------------
    # Run FIRST among the analyses because it does not depend on the ceiling: a correlation
    # across the I_rmsd range needs variation in I_rmsd, not acceptable poses.
    TS = np.array([p["TS"] for c in data for p in c["poses"]])
    IR = np.array([p["I_rmsd"] for c in data for p in c["poses"]])
    G = np.array([abs(p["grid"]) for c in data for p in c["poses"]])
    rho = spearman(TS, IR); pv = spearman_p(rho, len(TS))
    per = np.array([x for x in (spearman([p["TS"] for p in c["poses"]],
                                         [p["I_rmsd"] for p in c["poses"]]) for c in data)
                    if np.isfinite(x)])
    par = partial(TS, IR, G)
    print("\n  STEP 2  Q1 RETEST -- does basin breadth still track nativeness when the poses "
          "were NOT chosen by a scorer?")
    print(f"      pooled Spearman(T*S_conf, I_rmsd)   {rho:+.4f}   "
          f"(n = {len(TS)}, p = {pv:.2e})   bar <= {Q1_RHO}")
    print(f"      per-complex median                  {np.median(per):+.4f}   "
          f"(negative on {int((per < 0).sum())}/{len(per)})")
    print(f"      partial | interface contacts        {par:+.4f}   bar <= {Q1_RHO}")
    q1 = bool(rho <= Q1_RHO and pv < Q1_P and np.median(per) < 0 and par <= Q1_RHO)
    q1msg = ("REPLICATES off the collider -- the correlation is a property of interfaces, "
             "not of the scorer that picked the poses") if q1 else (
             "FAILS -- the shortlist -0.45 was a SELECTION ARTIFACT and is WITHDRAWN")
    print(f"      Q1 {q1msg}")
    print(f"      domain of the claim: I_rmsd spans {IR.min():.2f} to {IR.max():.2f} A")

    # the collider mechanism, measured directly rather than argued
    hi, lo = [], []
    for c in data:
        g = np.array([abs(p["grid"]) for p in c["poses"]])
        t = np.array([p["TS"] for p in c["poses"]])
        i_ = np.array([p["I_rmsd"] for p in c["poses"]])
        cut = np.quantile(g, 0.80)
        m = g >= cut
        if m.sum() >= 5:
            hi.append(spearman(t[m], i_[m]))
        if (~m).sum() >= 5:
            lo.append(spearman(t[~m], i_[~m]))
    hi = np.array([x for x in hi if np.isfinite(x)])
    lo = np.array([x for x in lo if np.isfinite(x)])
    print(f"      COLLIDER CHECK -- the same correlation inside vs outside the top grid "
          f"scores:")
    print(f"        top 20% by grid score   median rho {np.median(hi):+.4f}  (n={len(hi)})")
    print(f"        bottom 80%              median rho {np.median(lo):+.4f}  (n={len(lo)})")
    print(f"        whole sample            median rho {np.median(per):+.4f}")
    print(f"        -> a correlation that exists ONLY in the top slice would be the "
          f"selection effect the prior study was exposed to.")

    # ---------------- STEP 4: degenerate poses ----------------
    ndeg = sum(1 for c in data for p in c["poses"] if p.get("degenerate"))
    print(f"\n  STEP 4  degenerate poses (fewer than 2 repackable residues, T*S_conf = 0): "
          f"{ndeg}/{npose} = {100.0 * ndeg / npose:.1f}%")

    # ---------------- STEP 3: the retrieval test ----------------
    print("\n  STEP 3  RETRIEVAL -- how often does an acceptable pose reach the top "
          f"{TOPK}?")
    usable = [c for c in data if any(p["quality"] in OK for p in c["poses"])]
    if not usable:
        print(f"      VOID. No sampled pose in ANY complex is CAPRI-acceptable, so the "
              f"ceiling on this test is 0/{len(data)} and no ranking -- energy, entropy, "
              f"blend, or an oracle reading the answer key -- could score above zero.")
        print(f"      This gate is declared VOID and NOT null (ledger defect N): it has no "
              f"power to distinguish the rankings, so it reports nothing about them.")
        print(f"      The finding is STEP 0's: the SEARCH is the binding constraint.")
    else:
        exp, var = 0.0, 0.0
        for c in usable:
            N = len(c["poses"]); k = sum(p["quality"] in OK for p in c["poses"])
            pr = chance_hit(N, k)
            exp += pr; var += pr * (1 - pr)
        sd = float(np.sqrt(var))
        print(f"      complexes with >= 1 acceptable pose IN THE SAMPLE: "
              f"{len(usable)}/{len(data)}  (this is the ceiling for this test)")
        print(f"      chance expectation under a random ranking: {exp:.2f} +- {sd:.2f} "
              f"hits; bar = {exp + CHANCE_SD * sd:.2f}")
        print(f"      {'ranking':>12s} {'hits':>6s} {'vs chance':>12s}  verdict")
        rankings = {
            "energy": lambda ps: rank_norm([p["ve"] for p in ps]),
            "entropy": lambda ps: rank_norm([p["TS"] for p in ps], descending=True),
            "50/50": lambda ps: 0.5 * rank_norm([p["ve"] for p in ps])
                              + 0.5 * rank_norm([p["TS"] for p in ps], descending=True),
        }
        for name, fn in rankings.items():
            hits = 0
            for c in usable:
                ps = c["poses"]
                order = np.argsort(fn(ps))[:TOPK]
                hits += any(ps[int(j)]["quality"] in OK for j in order)
            z = (hits - exp) / sd if sd > 0 else float("nan")
            ok = hits >= exp + CHANCE_SD * sd
            print(f"      {name:>12s} {hits:6d} {z:+11.2f}sd  "
                  f"{'SIGNAL' if ok else 'not above chance'}")
        # DIAGNOSTIC ONLY -- not a gate. 50/50 above is the predeclared point.
        print(f"      lambda sweep of rank(E) + lambda*rank(-T*S) -- DIAGNOSTIC, NOT A GATE:")
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9):
            hits = 0
            for c in usable:
                ps = c["poses"]
                s = rank_norm([p["ve"] for p in ps]) + lam * rank_norm(
                    [p["TS"] for p in ps], descending=True)
                order = np.argsort(s)[:TOPK]
                hits += any(ps[int(j)]["quality"] in OK for j in order)
            tag = "entropy only" if lam > 1e8 else ("energy only" if lam == 0 else "")
            print(f"        lambda={lam:<8.2f} hits={hits:3d}  {tag}")
        # STEP 4's promise: if degenerates are >10% of the sample, show what they were doing
        # to the entropy ranking rather than leaving it to be assumed.
        if ndeg > 0.10 * npose:
            print(f"      degenerates are {100.0 * ndeg / npose:.1f}% of the sample, so the "
                  f"entropy ranking is re-run with them EXCLUDED:")
            for name, fn in rankings.items():
                hits, exp2, var2 = 0, 0.0, 0.0
                for c in usable:
                    ps = [p for p in c["poses"] if not p.get("degenerate")]
                    if len(ps) < TOPK or not any(p["quality"] in OK for p in ps):
                        continue
                    pr = chance_hit(len(ps), sum(p["quality"] in OK for p in ps))
                    exp2 += pr; var2 += pr * (1 - pr)
                    order = np.argsort(fn(ps))[:TOPK]
                    hits += any(ps[int(j)]["quality"] in OK for j in order)
                sd2 = float(np.sqrt(var2))
                z = (hits - exp2) / sd2 if sd2 > 0 else float("nan")
                print(f"        {name:>10s} {hits:4d} hits vs chance {exp2:.2f} "
                      f"+- {sd2:.2f}  ({z:+.2f} sd)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
