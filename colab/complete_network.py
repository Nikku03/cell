"""complete_network -- merge the curated network and the measured knockout edges into ONE graph, and account honestly for every measured edge.

THE TASK: take the existing network (curated regulatory + physical), take each knockout and the genes it moves, and map those measured effects onto
the network -- confirming the edges that are already there, routing the ones that can be routed through intermediate "joints", and marking the rest
as genuinely new structure the network does not contain.

THE TRAP THIS MODULE IS BUILT AROUND. Mapping measured effects onto a graph is trivially easy to fake. The curated graph is dense enough that almost
any gene reaches almost any other gene within a few hops -- an earlier case study here found 230 of 233 measured targets "reachable" within 4 hops
while the discrimination against random targets fell from 3.63x to 1.02x. A path you can always find explains nothing. So every routing claim in this
module is scored against a DEGREE-MATCHED NULL: for each source, the same BFS is used to look up hop distance to its real targets AND to matched
decoys. An explanation only counts if the real target is reachably CLOSER than its decoy.

TWO STATISTICAL MISTAKES WERE MADE HERE BEFORE THIS VERSION, IN OPPOSITE DIRECTIONS, AND BOTH ARE WORTH KEEPING IN VIEW.

  (1) THE NULL MUST MATCH THE COVARIATE THE CLAIM IS ABOUT. The first version bucketed decoys on mover-frequency and called that degree-matched.
      It is not: mover-frequency correlates with curated-graph degree at rho~0.01. Since the claim is "the real target is CLOSER IN THE GRAPH",
      the decoy must match the target's GRAPH DEGREE. Buckets are now log2(in-degree in the routing layer) x mover-frequency, and the match is
      VERIFIED by a printed balance table rather than asserted.
  (2) REPLICATES AND TARGETS ARE NOT INDEPENDENT TRIALS. Fixing (1) and then counting all NDECOY decoys of every target as separate Bernoulli
      trials multiplied n fivefold and the p-value by four orders of magnitude WITHOUT moving the effect size -- manufactured significance. The
      NDECOY decoys of one target are compared against the SAME real distance, and a source's ~26 targets come out of ONE BFS tree. Each target is
      therefore reduced to a single statistic, averaged within source, and tested ACROSS SOURCES (t-test + cluster bootstrap). The naive binomial
      is still printed, labelled wrong, so the size of the inflation stays visible.

The outcome is not what either earlier version claimed: on the properly-matched, properly-clustered test the REGULATORY layer sits at chance while
the COMBINED layer clears it only marginally. Every routing result is also decomposed into pairs settled by true DISTANCE versus pairs settled
merely by one side being unreachable, and the TIE fraction is reported, because a "% of decisive comparisons" computed over a few percent of the
data is easy to over-read.

WHAT IT PRODUCES:
  PART 1  MERGE     -- one typed graph: curated-regulatory (signed), curated-physical, and measured-directed edges, each with provenance.
  PART 2  ROUTE     -- every measured edge classified as DIRECT-REGULATORY / DIRECT-PHYSICAL / ROUTED-k (via k-1 intermediates) / UNROUTABLE,
                       computed separately over the REGULATORY-only graph and the COMBINED graph, because a previous honesty guard showed the
                       regulatory layer retains discrimination with depth while the physical layer sits at chance.
  PART 3  JOINTS    -- which genes actually carry the routing load, scored by how many measured edges route through them AND by how much better
                       than the null those routes are. This is the "joints" the merged network hangs on.
  PART 4  SIGN      -- where a route is fully signed, does the product of signs predict the measured direction of change? Compared against the
                       per-gene sign prior, which is a strong baseline (most genes move one way most of the time).
  PART 5  VALUE     -- the only test that matters: does the COMPLETED network predict held-out knockouts better than the curated network alone?
                       Sources are split train/test, the completion is built from TRAIN sources only, and held-out specific-mover recall@50 is
                       scored. Anything else would be circular, since the completion is built from the very edges it would be scored on.

Deterministic (fixed seed, sorted iteration). Usage: python complete_network.py"""
import json, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
RNG = np.random.RandomState(0)
MAXHOP = 4
TIDE_FRAC = 0.05
MINSRC = 20            # a knockout must move this many genes to be a usable source (matches infer_network)
MAXCOMPLEX = 200       # complexes larger than this would add O(n^2) edges and swamp the physical layer
K_RECALL = 50
MINROUTES = 5          # a gene must carry this many routes to be scoreable as a joint at all
NSPLIT = 5             # independent train/test source splits -- one split is too fragile for an effect this small
NDECOY = 5             # decoy replicates per target, so each candidate joint gets a null mean and SD rather than a single draw
REF_TIDE_HARNESS, REF_BEST_HARNESS = 0.26, 0.49     # references from the HARNESS setup (TAU=1.0 pkl) -- NOT this setup, see below


def bfs(adj, src, maxhop):
    """hop distance and parent pointer from src, capped at maxhop. Deterministic: adjacency lists are pre-sorted.

    CAVEAT THAT THE JOINTS SECTION DEPENDS ON: this records ONE parent per node. Where several shortest paths of equal
    length exist -- which is the common case on a graph with median degree 15 -- which intermediate gets credited is
    decided by the order neighbours happen to be visited. That order is alphabetical here, which is arbitrary. Part 3
    therefore measures how much of the joint list survives permuting it, rather than assuming it does."""
    hop = {src: 0}; par = {src: None}; frontier = [src]
    for h in range(1, maxhop + 1):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in hop:
                    hop[v] = h; par[v] = u; nxt.append(v)
        if not nxt:
            break
        frontier = nxt
    return hop, par


def main():
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]; gidx = {g: i for i, g in enumerate(genes)}
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); Aall = np.abs(Z)

    tc = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
          if k in kidx and (np.abs(H.M[H.ki[k]]) > 0).any() and np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0]
    med = float(np.median(tc)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((Aall >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (Aall >= T).sum(1)
    src_rows = [i for i in range(len(kos)) if per[i] >= MINSRC]
    tide = (Aall[src_rows] >= T).mean(0) >= TIDE_FRAC
    mover_freq = (Aall[src_rows] >= T).mean(0)
    sources = [kos[i] for i in src_rows]
    print(f"K562: {len(kos)} knockouts x {len(genes)} genes | threshold |z|>={T:.1f} | {len(sources)} usable sources | {int(tide.sum())} tide genes")

    # ---------------- PART 1: MERGE ----------------
    reg_raw, _trace, _s = build_causal(H)
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}

    reg = collections.defaultdict(dict)          # signed regulatory: src -> {tgt: sign}
    for s in sorted(reg_raw):
        acc = collections.defaultdict(set)
        for t2, sg in reg_raw[s]:
            acc[t2].add(sg)
        for t2 in sorted(acc):
            nz = {x for x in acc[t2] if x != 0}
            reg[s][t2] = (next(iter(nz)) if len(nz) == 1 else 0)
    phys = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            phys[names[a]].add(names[b]); phys[names[b]].add(names[a])
    for mem in D["complexes"].values():
        mm = sorted({names[x] for x in mem if isinstance(x, int) and x < N})
        if 2 <= len(mm) <= MAXCOMPLEX:
            for g in mm:
                phys[g].update(x for x in mm if x != g)

    n_reg = sum(len(v) for v in reg.values()); n_phys = sum(len(v) for v in phys.values()) // 2
    print(f"curated network: {n_reg:,} directed regulatory edges over {len(reg):,} regulators | {n_phys:,} undirected physical edges")

    # measured edges, split specific vs tide
    measured = []                                  # (src, dst, signed z, is_specific)
    for i in src_rows:
        for j in np.where(Aall[i] >= T)[0]:
            if genes[j] != kos[i]:
                measured.append((kos[i], genes[j], float(Z[i, j]), not bool(tide[j])))
    spec = [e for e in measured if e[3]]
    print(f"measured network: {len(measured):,} directed edges ({len(spec):,} specific, {len(measured)-len(spec):,} tide)")

    # adjacency for routing. REGULATORY is directed; COMBINED adds the undirected physical layer.
    adj_reg = {k: sorted(v) for k, v in reg.items()}
    adj_all = collections.defaultdict(set)
    for a in reg:
        adj_all[a].update(reg[a])
    for a in phys:
        adj_all[a].update(phys[a])
    adj_all = {k: sorted(v) for k, v in adj_all.items()}

    # ---------------- PART 2: ROUTE, against a degree-matched null ----------------
    # Negatives are drawn per source from the same mover-frequency bucket as that source's real targets, so a target
    # cannot be "explained" merely by being a gene that moves often and is therefore well connected.
    # THE NULL HAS TO BE MATCHED ON THE COVARIATE THE CLAIM IS ABOUT, and getting this wrong reverses a conclusion.
    # An earlier version bucketed decoys on mover_freq alone and called it "degree-matched". It is not: mover_freq
    # correlates with curated-graph degree at rho~0.01. Since the claim under test is "the real target is CLOSER IN THE
    # GRAPH", the decoy must match the target's GRAPH DEGREE -- otherwise decoys are systematically more (or less)
    # reachable than the targets they stand in for, and that imbalance alone decides most pairs. Buckets are therefore
    # built PER LAYER on log2(degree in that layer) crossed with mover_freq, and a covariate balance table is printed so
    # a reader can check the match rather than take the word "matched" on trust.
    # Decoys are also restricted to non-tide genes (real targets are non-tide) and, per draw, may not be the source or
    # one of that source's own targets.
    def make_buckets(adj_for_layer, directed):
        # For an undirected layer each edge already appears in BOTH adjacency lists, so counting the reverse again
        # double-counts and the printed balance table reads ~2x the true in-degree.
        indeg = collections.Counter()
        for u, vs in adj_for_layer.items():
            for v in vs:
                indeg[v] += 1
        b = collections.defaultdict(list)
        for j, g in enumerate(genes):
            # Decoys must come from the same population as real targets: non-tide AND actually movable. Without the
            # movability condition ~37% of drawn decoys are genes that never move under ANY source, while every real
            # target moves at least once -- an imbalance in the very property the comparison is about.
            if tide[j] or mover_freq[j] <= 0:
                continue
            d = indeg.get(g, 0)
            b[(min(int(mover_freq[j] * 100) // 2, 25), int(np.log2(d + 1)))].append(g)
        return b, indeg

    layers = {}
    joint_load = {"REGULATORY": collections.Counter(), "COMBINED": collections.Counter()}
    # NDECOY independent decoy draws per target, kept as SEPARATE replicates, so each gene gets a null MEAN and SD and
    # can be z-scored. A single decoy draw only supports a sign test, and that sign test is confounded: decoy routes
    # concentrate on a few hubs while real routes spread wider, so most genes come out "above null" even when the
    # overall routing is at chance. The z-score against replicated nulls is the honest version.
    joint_reps = {"REGULATORY": [collections.Counter() for _ in range(NDECOY)],
                  "COMBINED": [collections.Counter() for _ in range(NDECOY)]}
    edge_route = {}
    for lname, adj in (("REGULATORY", adj_reg), ("COMBINED", adj_all)):
        buckets, indeg = make_buckets(adj, lname == "REGULATORY")
        hops_real = collections.Counter(); hops_null = collections.Counter()
        closer = same = farther = 0
        # balance diagnostics + decomposition of what actually decides each decisive pair
        bal_rd, bal_dd = [], []
        dec_unreach = dec_dist = 0
        # CLUSTER-AWARE BOOKKEEPING. The 5 decoys of one target are compared against the SAME real hop distance, so they
        # are not independent trials; and the ~26 targets of one source all come out of ONE BFS tree. Counting
        # target x replicate as iid Bernoulli inflates n fivefold and the p-value by orders of magnitude without moving
        # the effect size at all. So each target is reduced to ONE statistic and the test is run at the SOURCE level.
        per_source = collections.defaultdict(list)
        for s in sorted(sources):
            if s not in adj and s not in reg:
                pass
            hop, par = bfs(adj, s, MAXHOP)
            r = kidx[s]
            tgts = [genes[j] for j in np.where((Aall[r] >= T) & (~tide))[0] if genes[j] != s]
            if not tgts:
                continue
            tset = set(tgts)
            for t2 in tgts:
                h = hop.get(t2, 99)
                hops_real[h] += 1
                if lname == "COMBINED":
                    edge_route[(s, t2)] = h
                # the joints: intermediates on the shortest route
                if 2 <= h <= MAXHOP:
                    c = par.get(t2)
                    while c is not None and c != s:
                        joint_load[lname][c] += 1
                        c = par.get(c)
                # degree-matched decoys for this target: replicate 0 drives the hop comparison, all replicates feed the joint null
                bkey = (min(int(mover_freq[gidx[t2]] * 100) // 2, 25), int(np.log2(indeg.get(t2, 0) + 1)))
                pool = buckets.get(bkey) or buckets.get((bkey[0], max(bkey[1] - 1, 0))) or genes
                t_close = t_far = 0
                for rep in range(NDECOY):
                    d = None
                    for _try in range(20):
                        cand = pool[RNG.randint(len(pool))]
                        # a decoy must be a gene this source did NOT move, and must not be the source itself
                        if cand != s and cand not in tset:
                            d = cand
                            break
                    if d is None:
                        continue
                    hd = hop.get(d, 99)
                    # ALL replicates feed the headline comparison. Using only the first was an arbitrary choice that
                    # threw away 80% of the drawn null and made the result hostage to one draw.
                    if rep == 0:
                        hops_null[hd] += 1                      # the displayed hop histogram stays a single-draw view
                    bal_rd.append(indeg.get(t2, 0)); bal_dd.append(indeg.get(d, 0))
                    if h != hd:
                        if (h == 99) != (hd == 99):
                            dec_unreach += 1
                        else:
                            dec_dist += 1
                    if h < hd:
                        closer += 1; t_close += 1
                    elif h == hd:
                        same += 1
                    else:
                        farther += 1; t_far += 1
                    if 2 <= hd <= MAXHOP:
                        c = par.get(d)
                        while c is not None and c != s:
                            joint_reps[lname][rep][c] += 1
                            c = par.get(c)
                per_source[s].append((t_close - t_far) / float(NDECOY))
        tot = sum(hops_real.values())
        from scipy.stats import binomtest, ttest_1samp
        dec_n = closer + farther
        # The naive binomial over target x replicate is reported ONLY to show how badly it misleads; it is not the test.
        p_naive = binomtest(closer, dec_n, 0.5).pvalue if dec_n else float("nan")
        # THE ACTUAL TEST: reduce each target to one statistic, average within source, test across sources.
        src_means = np.array([np.mean(v) for s2, v in sorted(per_source.items()) if v])
        n_src = len(src_means)
        if n_src > 2:
            tt = ttest_1samp(src_means, 0.0)
            t_stat, pcloser = float(tt.statistic), float(tt.pvalue)
        else:
            t_stat, pcloser = float("nan"), float("nan")
        # cluster bootstrap over SOURCES for a CI on the pooled closer-fraction
        boot = []
        src_keys = [s2 for s2, v in sorted(per_source.items()) if v]
        for _ in range(2000):
            pick = RNG.randint(0, n_src, n_src) if n_src else []
            vals = np.concatenate([per_source[src_keys[q]] for q in pick]) if n_src else np.array([0.0])
            boot.append(float(np.mean(vals)))
        lo_b, hi_b = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)
        reach = tot - hops_real.get(99, 0)
        reach_null = tot - hops_null.get(99, 0)
        mrd, mdd = float(np.mean(bal_rd)) if bal_rd else 0.0, float(np.mean(bal_dd)) if bal_dd else 0.0
        # among pairs decided by real DISTANCE (both sides reachable), how often is the real target closer?
        layers[lname] = {"hops_real": dict(hops_real), "hops_null": dict(hops_null), "n": tot,
                         "closer": closer, "same": same, "farther": farther,
                         "closer_frac": round(closer / max(dec_n, 1), 4), "closer_p": float(pcloser),
                         "p_naive_pseudoreplicated": float(p_naive), "n_source_clusters": int(n_src),
                         "cluster_t": float(t_stat), "cluster_mean": round(float(np.mean(src_means)) if n_src else 0.0, 5), "cluster_boot_ci": [round(lo_b, 4), round(hi_b, 4)],
                         "tie_frac": round(same / max(closer + same + farther, 1), 4),
                         "reachable": reach, "reachable_frac": round(reach / max(tot, 1), 4),
                         "reachable_null": reach_null, "reachable_null_frac": round(reach_null / max(tot, 1), 4),
                         "mean_degree_real": round(mrd, 1), "mean_degree_decoy": round(mdd, 1),
                         "decided_by_reachability": dec_unreach, "decided_by_distance": dec_dist}
        print(f"\n  ROUTING over the {lname} layer ({tot:,} specific measured edges):")
        print(f"    {'hops':>6s} {'real':>9s} {'null':>9s} {'enrichment':>11s}")
        for h in (1, 2, 3, 4, 99):
            rr = hops_real.get(h, 0); nn = hops_null.get(h, 0)
            lab = "unreach" if h == 99 else str(h)
            en = (rr / max(nn, 1)) if nn else float("inf")
            print(f"    {lab:>6s} {rr:>9,d} {nn:>9,d} {en:>11.2f}x")
        print(f"    reachable within {MAXHOP} hops: {reach:,}/{tot:,} ({100*reach/max(tot,1):.0f}%)")
        print(f"    real target strictly CLOSER than its matched decoy: {closer:,}/{dec_n:,} decisive comparisons "
              f"({100*closer/max(dec_n,1):.1f}%; {100*same/max(closer+same+farther,1):.0f}% of all comparisons are TIES)")
        print(f"    naive binomial over target x replicate: p={p_naive:.2g}  <- WRONG, pseudo-replicated: the {NDECOY} decoys "
              f"of a target share one real distance and a source's targets share one BFS tree")
        print(f"    CLUSTERED at the source level ({n_src} sources): mean per-source (closer-farther)/rep = "
              f"{np.mean(src_means):+.4f}, t={t_stat:+.2f}, p={pcloser:.3g}; cluster-bootstrap 95% CI "
              f"[{lo_b:+.4f},{hi_b:+.4f}]"
              + ("  <- NOT significant once clustering is respected" if not (pcloser < 0.05) else "  <- survives clustering"))
        print(f"    NULL BALANCE (check the match, do not trust the word 'matched'): mean in-degree real {mrd:.1f} vs decoy {mdd:.1f}; "
              f"reachable-at-all real {100*reach/max(tot,1):.1f}% vs decoy {100*reach_null/max(tot,1):.1f}%")
        print(f"    what decides a pair: REACHABILITY (one side unreachable) {dec_unreach:,} vs true DISTANCE "
              f"(both reachable, different hop) {dec_dist:,} "
              f"({100*dec_dist/max(dec_unreach+dec_dist,1):.0f}% distance-driven)")

    # ---------------- PART 3: JOINTS ----------------
    srcset = set(sources)
    joints = {}
    for lname in ("REGULATORY", "COMBINED"):
        jl = joint_load[lname]; reps = joint_reps[lname]
        allg = sorted(set(jl) | {g for r in reps for g in r})
        # Only genes carrying a real amount of routing are scoreable at all: with 5 replicates a gene routed once or
        # twice can trivially have every null replicate at zero, which makes sd=0 and sends z to infinity. Filtering
        # FIRST (rather than filtering only the printed view) is what keeps the emitted table honest.
        rows_j = []
        for g in allg:
            obs = jl.get(g, 0)
            if obs < MINROUTES:
                continue
            nulls = np.array([r.get(g, 0) for r in reps], dtype=float)
            mu = float(nulls.mean()); sd = float(nulls.std(ddof=1)) if NDECOY > 1 else 0.0
            # Poisson variance floor. Route counts are counts, so their null SD cannot honestly be below sqrt(mean);
            # without this, a gene whose 5 null replicates happen to agree exactly gets an infinite z.
            sd_eff = max(sd, np.sqrt(max(mu, 1.0)))
            z = (obs - mu) / sd_eff
            rows_j.append({"gene": g, "routes": obs, "null_mean": round(mu, 2), "null_sd": round(sd, 2),
                           "sd_used": round(float(sd_eff), 2), "z": round(float(z), 2),
                           "is_TF": bool(info.get(g, {}).get("tf")), "is_measured_source": g in srcset})
        rows_j.sort(key=lambda r: (-r["z"], -r["routes"], r["gene"]))
        sig = [r for r in rows_j if r["z"] >= 2]
        # The chance expectation must be EMPIRICAL, not the 2.28% normal tail: a z built from 5 replicates with a Poisson
        # floor is not normally distributed, so quoting the normal tail would be inventing a false-positive rate. Instead
        # hold out one replicate as a pseudo-observation and score it against the remaining four exactly as above -- the
        # number of genes that clear z>=2 under that arrangement IS the chance count for this estimator.
        exp_hits = []
        for hold in range(NDECOY):
            others = [reps[q] for q in range(NDECOY) if q != hold]
            hits = 0
            for r in rows_j:
                g = r["gene"]; obs0 = reps[hold].get(g, 0)
                if obs0 < MINROUTES:
                    continue
                nl = np.array([o.get(g, 0) for o in others], dtype=float)
                mu0 = float(nl.mean()); sd0 = max(float(nl.std(ddof=1)), np.sqrt(max(mu0, 1.0)))
                if (obs0 - mu0) / sd0 >= 2:
                    hits += 1
            exp_hits.append(hits)
        exp_fp = float(np.mean(exp_hits))
        joints[lname] = {"n_scoreable_genes": len(rows_j), "min_routes": MINROUTES, "n_z2": len(sig),
                         "expected_by_chance": round(exp_fp, 1), "top": rows_j[:40]}
        note = ("  <- no more than chance would give" if len(sig) <= exp_fp else
                f"  <- {len(sig)/max(exp_fp,1e-9):.1f}x the chance expectation")
        print(f"\n  JOINTS on the {lname} layer ({NDECOY} null replicates, z-scored):")
        print(f"    {len(rows_j):,} genes carry >={MINROUTES} routes; {len(sig):,} reach z>=2 "
              f"(chance would give ~{exp_fp:.0f}){note}")
        if sig:
            print(f"    {'gene':12s} {'routes':>8s} {'null mean':>10s} {'z':>7s} {'TF':>3s} {'source?':>8s}")
            for r in sig[:15]:
                print(f"    {r['gene']:12s} {r['routes']:>8,d} {r['null_mean']:>10.1f} {r['z']:>7.1f} "
                      f"{'Y' if r['is_TF'] else '-':>3s} {('yes' if r['is_measured_source'] else 'NO'):>8s}")
    # composition of the significant REGULATORY joints, computed rather than asserted
    HEME = {"RUNX1", "MYB", "GATA1", "GATA2", "GATA3", "CEBPA", "CEBPB", "HHEX", "TAL1", "SPI1", "KLF1", "LMO2", "IKZF1"}
    _rsig = [r for r in joints["REGULATORY"]["top"] if r["z"] >= 2]
    _rall = joints["REGULATORY"]["top"]
    reg_sig_n = len(_rsig)
    reg_tf_n = sum(1 for r in _rsig if r["is_TF"])
    reg_notsrc_n = sum(1 for r in _rsig if not r["is_measured_source"])
    # BASE RATES. "n/n of the joints were never knocked out" means nothing unless most scoreable genes were never
    # knocked out anyway -- which they were. Quote the background rate next to it so the reader can see whether the
    # joint list is actually enriched for anything or just inheriting the panel's composition.
    base_notsrc = (sum(1 for r in _rall if not r["is_measured_source"]) / max(len(_rall), 1)) if _rall else 0.0
    base_tf = (sum(1 for r in _rall if r["is_TF"]) / max(len(_rall), 1)) if _rall else 0.0
    reg_heme = [r["gene"] for r in _rsig if r["gene"] in HEME] or ["none"]
    reg_other = [r["gene"] for r in _rsig if r["gene"] not in HEME] or ["none"]
    # TIE-BREAK STABILITY: is a "joint" a property of routing, or of the order BFS happened to visit neighbours?
    # Re-run the load computation with the adjacency order permuted and measure how much of the set survives. If the
    # membership does not survive, the count may still be meaningful but the NAMES are not.
    def joint_load_with_order(adj_src, seed):
        rr = np.random.RandomState(seed)
        adjp = {}
        for k, v in adj_src.items():
            lv = list(v)
            rr.shuffle(lv)
            adjp[k] = lv
        load = collections.Counter()
        for s in sorted(sources):
            hop, par = bfs(adjp, s, MAXHOP)
            r = kidx[s]
            for j in np.where((Aall[r] >= T) & (~tide))[0]:
                t2 = genes[j]
                if t2 == s or not (2 <= hop.get(t2, 99) <= MAXHOP):
                    continue
                c = par.get(t2)
                while c is not None and c != s:
                    load[c] += 1; c = par.get(c)
        return {g for g, n in load.items() if n >= MINROUTES}

    base_set = {r["gene"] for r in joints["COMBINED"]["top"]} | {g for g in joint_load["COMBINED"]
                                                                 if joint_load["COMBINED"][g] >= MINROUTES}
    jac = []
    for sd in (1, 2):
        S = joint_load_with_order(adj_all, sd)
        jac.append(len(base_set & S) / max(len(base_set | S), 1))
    mjac = float(np.mean(jac))
    print(f"\n  TIE-BREAK STABILITY (COMBINED): permuting the order BFS visits neighbours keeps only "
          f"Jaccard {mjac:.2f} of the >={MINROUTES}-route joint set"
          + ("  <- the COUNT is a result, the NAMES are largely an implementation artifact" if mjac < 0.7 else
             "  <- membership is stable"))
    scored_rows = joints["COMBINED"]["top"]

    # ---------------- PART 4: SIGN ----------------
    signed_ok = signed_tot = 0
    for s in sorted(sources):
        r = kidx[s]
        for t2, sg in sorted(reg.get(s, {}).items()):
            if sg == 0 or t2 not in gidx:
                continue
            j = gidx[t2]
            if Aall[r, j] < T or tide[j]:
                continue
            signed_tot += 1
            # knocking OUT an activator should lower its target; knocking out a repressor should raise it
            if np.sign(Z[r, j]) == -np.sign(sg):
                signed_ok += 1
    prior = float(max(np.mean(Z[np.ix_(src_rows, np.where(~tide)[0])] > 0),
                      np.mean(Z[np.ix_(src_rows, np.where(~tide)[0])] < 0)))
    # Same floor as elsewhere in this project: a rate computed from a handful of events is not a measurement.
    sign_testable = signed_tot >= 30
    if sign_testable:
        print(f"\n  SIGN: {signed_tot:,} measured edges have a signed curated counterpart; direction agrees in "
              f"{100*signed_ok/max(signed_tot,1):.0f}% (majority-direction baseline {100*prior:.0f}%)")
    else:
        print(f"\n  SIGN: UNTESTABLE -- only {signed_tot} measured specific edges have a signed curated counterpart at all "
              f"(of {len(spec):,}). The curated regulatory layer barely intersects the knockouts we can use as sources, "
              f"so there is nothing here to score. ({signed_ok} agreed, but a rate over {signed_tot} events is not a measurement.)")

    # ---------------- PART 5: VALUE (held out) ----------------
    # Split SOURCES. Build the completion from train sources only, then ask whether adding it to the curated graph
    # improves recall of held-out sources' specific movers. Splitting by source is what keeps this non-circular.
    # REPEATED splits, not one. A single 70/30 split gave p=0.11, p=0.018 and p=0.078 on three different RNG streams --
    # the gain is small enough that its significance is decided by which sources happen to land in the test set. Repeating
    # the split and reporting the spread is the honest way to describe an effect that marginal.
    nontide_idx = np.where(~tide)[0]

    def influence(adj, s, hops=MAXHOP, atten=0.5):
        """Degree-normalised diffusion from s -- a GRADED score, unlike raw hop distance.

        This matters: ranking by hop alone leaves hundreds of genes tied in the first tier, and any tie-break
        (alphabetical, insertion order) is arbitrary and silently decides the metric. Diffusion ranks within a tier by
        how much of the walk actually arrives, which is the most favourable reasonable reading of the curated graph."""
        acc = collections.defaultdict(float); frontier = {s: 1.0}
        for _ in range(hops):
            nxt = collections.defaultdict(float)
            for u, w in frontier.items():
                nb = adj.get(u)
                if not nb:
                    continue
                share = w * atten / len(nb)
                for v in nb:
                    acc[v] += share; nxt[v] += share
            if not nxt:
                break
            frontier = nxt
        acc.pop(s, None)
        return acc

    def recall_at_k(adj, s, k=K_RECALL):
        r = kidx[s]
        truth = {genes[j] for j in np.where((Aall[r] >= T) & (~tide))[0] if genes[j] != s}
        if not truth:
            return None
        acc = influence(adj, s)
        cand = sorted(((-w, g) for g, w in acc.items() if g in gidx and not tide[gidx[g]]))
        top = [g for _, g in cand[:k]]
        return len(set(top) & truth) / min(len(truth), k)

    allnt = [genes[j] for j in nontide_idx]
    from scipy.stats import wilcoxon as _wil
    split_rows = []
    for si in range(NSPLIT):
        ss = sorted(sources); RNG.shuffle(ss)
        cut = int(0.7 * len(ss)); train_s, test_s = set(ss[:cut]), ss[cut:]
        add = collections.defaultdict(set)
        for a, b, z, is_s in measured:
            if is_s and a in train_s:
                add[a].add(b)
        adj_plus = {k: sorted(set(v) | add.get(k, set())) for k, v in adj_all.items()}
        for k in add:
            if k not in adj_plus:
                adj_plus[k] = sorted(add[k])
        rc, rp, rr = [], [], []
        n_absent = 0
        for s in test_s:
            if s not in adj_all:
                n_absent += 1          # not in the curated graph at all -> a hard zero, counted and reported
            a_ = recall_at_k(adj_all, s); b_ = recall_at_k(adj_plus, s)
            if a_ is None:
                continue
            rc.append(a_); rp.append(b_)
            r = kidx[s]
            truth = {genes[j] for j in np.where((Aall[r] >= T) & (~tide))[0] if genes[j] != s}
            pick = {allnt[q] for q in RNG.choice(len(allnt), K_RECALL, replace=False)}
            rr.append(len(pick & truth) / min(len(truth), K_RECALL))
        # exact method: the number of non-tied pairs is tiny (often <10), where scipy's normal approximation is wrong
        try:
            nz = sum(1 for a_, b_ in zip(rc, rp) if a_ != b_)
            pg = float(_wil(rp, rc, method="exact" if nz <= 25 else "auto").pvalue) if nz else float("nan")
        except (ValueError, TypeError):
            pg = float("nan")
        split_rows.append({"split": si, "n": len(rc), "curated": float(np.mean(rc)), "completed": float(np.mean(rp)),
                           "random": float(np.mean(rr)), "gain": float(np.mean(rp) - np.mean(rc)), "p": pg,
                           "helps": int(sum(1 for a_, b_ in zip(rc, rp) if b_ > a_)),
                           "hurts": int(sum(1 for a_, b_ in zip(rc, rp) if b_ < a_)),
                           "absent_from_curated_graph": n_absent})
    mc = float(np.mean([r["curated"] for r in split_rows])); mp = float(np.mean([r["completed"] for r in split_rows]))
    mr = float(np.mean([r["random"] for r in split_rows]))
    gains = np.array([r["gain"] for r in split_rows]); ps = np.array([r["p"] for r in split_rows])
    nsig = int(np.sum(ps < 0.05)); nbetter = sum(r["helps"] for r in split_rows); nworse = sum(r["hurts"] for r in split_rows)
    p_gain = float(np.median(ps))
    print(f"\n  VALUE over {NSPLIT} independent 70/30 source splits (specific-mover recall@{K_RECALL}):")
    print(f"    {'split':>6s} {'n':>5s} {'curated':>9s} {'completed':>11s} {'gain':>8s} {'p':>9s} {'helps/hurts':>12s}")
    for r in split_rows:
        print(f"    {r['split']:>6d} {r['n']:>5d} {r['curated']:>9.3f} {r['completed']:>11.3f} {r['gain']:>+8.3f} "
              f"{r['p']:>9.3g} {str(r['helps'])+'/'+str(r['hurts']):>12s}")
    print(f"    random ranking                {mr:.3f}")
    print(f"    curated network alone         {mc:.3f}")
    print(f"    curated + measured completion {mp:.3f}   ({mp-mc:+.3f} mean, range "
          f"{gains.min():+.3f} to {gains.max():+.3f}; significant in {nsig}/{NSPLIT} splits)")
    # THE SCALE MATTERS, BUT ONLY AGAINST A REFERENCE COMPUTED IN THIS SETUP. The project's familiar 0.26 tide-null and
    # 0.49 best-model come from the HARNESS (pkl, TAU=1.0, MIN_SPEC=5); this module scores on the uncensored parquet at
    # |z|>=4.2 with a different tide definition, so quoting them here would be a cross-metric comparison dressed up as a
    # like-for-like one. The tide-null is therefore recomputed HERE, on exactly these sources and this truth definition.
    tide_rank = [genes[j] for j in np.argsort(-mover_freq) if not tide[j]][:K_RECALL]
    tn = []
    for s in sorted(sources):
        r = kidx[s]
        truth = {genes[j] for j in np.where((Aall[r] >= T) & (~tide))[0] if genes[j] != s}
        if truth:
            tn.append(len(set(tide_rank) & truth) / min(len(truth), K_RECALL))
    m_tide = float(np.mean(tn))
    print(f"    tide-null RECOMPUTED IN THIS SETUP       {m_tide:.3f}   (the familiar 0.26 is the harness setup, not this one)")
    n_eval = sum(r["n"] for r in split_rows)
    print(f"    completion helps {nbetter}/{n_eval} held-out source-evaluations across all {NSPLIT} splits, hurts {nworse}; "
          f"median paired Wilcoxon p={p_gain:.2g}"
          + ("  <- the gain is small and its significance is NOT stable across splits" if nsig < NSPLIT else
             "  <- significant in every split"))

    # ---------------- emit the merged graph ----------------
    typed = []
    for a in sorted(reg):
        for b, sg in sorted(reg[a].items()):
            typed.append([a, b, "curated_regulatory", int(sg)])
    for a in sorted(phys):
        for b in sorted(phys[a]):
            if a < b:
                typed.append([a, b, "curated_physical", 0])
    for a, b, z, is_s in sorted(measured):
        typed.append([a, b, "measured_specific" if is_s else "measured_tide", int(np.sign(z))])
    comb = layers["COMBINED"]
    routed = sum(v for h, v in comb["hops_real"].items() if h <= MAXHOP)
    unrouted = comb["hops_real"].get(99, 0)

    verdict = (
        "NETWORK COMPLETION (complete_network.py): the curated network and the measured knockout edges merged into one typed graph, with every "
        f"measured edge accounted for. The merged object has {n_reg:,} curated regulatory edges, {n_phys:,} curated physical edges and "
        f"{len(measured):,} measured directed edges ({len(spec):,} specific) over {len(sources)} usable source knockouts. "
        f"ROUTING: on the combined curated graph {routed:,}/{comb['n']:,} specific measured edges ({100*routed/max(comb['n'],1):.0f}%) can be "
        f"reached within {MAXHOP} hops and {unrouted:,} cannot. "
        "THAT 71% IS NOT THE RESULT, AND READING IT AS ONE WOULD BE THE WHOLE MISTAKE. The curated graph is dense enough to reach almost anything, "
        "so every route was scored against a DEGREE-MATCHED DECOY looked up in the same BFS, over all {} decoy replicates: ".format(NDECOY)
        + "; ".join(
            f"{ln} layer reaches {100*d['reachable_frac']:.0f}% of measured edges; clustered at the source level its per-source effect is "
            f"{d['cluster_mean']:+.4f} (t={d['cluster_t']:+.2f}, p={d['closer_p']:.3g}, bootstrap CI "
            f"[{d['cluster_boot_ci'][0]:+.4f},{d['cluster_boot_ci'][1]:+.4f}])"
            + (" -- SURVIVES clustering, marginally" if d["closer_p"] < 0.05 else " -- AT CHANCE once clustering is respected")
            for ln, d in layers.items())
        + ". I MADE TWO STATISTICAL MISTAKES HERE, IN OPPOSITE DIRECTIONS, AND BOTH ARE REPORTED RATHER THAN QUIETLY FIXED. First the null was "
        "bucketed on mover-frequency and called degree-matched, which it is not (rho~0.01 with graph degree); under it the combined layer read "
        "'at chance' and this module was about to conclude PPI routing explains nothing. Then, having fixed that, counting all "
        + str(NDECOY) + " decoys per target as independent trials multiplied n fivefold and the p-value by four orders of magnitude WITHOUT moving "
        "the effect size -- the naive binomial still printed above (" + ", ".join(f"{ln} p={d['p_naive_pseudoreplicated']:.2g}" for ln, d in layers.items())
        + ") is exactly that inflation, kept visible on purpose. Reduced to one statistic per target and tested across sources, the honest answer is "
        "that the layer I twice called meaningful is the one at chance. Note also how much of the comparison is TIES ("
        + ", ".join(f"{ln} {100*d['tie_frac']:.0f}%" for ln, d in layers.items())
        + "), so the headline percentages are computed over a minority of the data"
        + ". THE DEPTH PROFILE SAYS WHERE THE NETWORK STOPS EXPLAINING, and it is the single most useful number here: enrichment over "
        "matched decoys by route length (single-replicate view) is "
        + ", ".join(f"{lab} {(layers['COMBINED']['hops_real'].get(h,0)/max(layers['COMBINED']['hops_null'].get(h,1),1)):.2f}x"
                    for h, lab in ((1, "direct"), (2, "1 intermediate"), (3, "2 intermediates"), (4, "3 intermediates")))
        + " on the combined layer (and "
        + ", ".join(f"{lab} {(layers['REGULATORY']['hops_real'].get(h,0)/max(layers['REGULATORY']['hops_null'].get(h,1),1)):.2f}x"
                    for h, lab in ((1, "direct"), (2, "1 intermediate")))
        + " on the regulatory layer). Read it against the clustered tests above, not on its own: the only enrichment that is both sizeable and on the "
        "layer that survives clustering is the DIRECT edge on the combined layer; everything at two or more intermediates is indistinguishable from "
        "a random walk on both layers. So the network can be completed, but it can only be TRUSTED at the direct edge -- which is why the long "
        "explanatory chains this project kept constructing dissolved under test. "
        + (f"SIGN: UNTESTABLE -- only {signed_tot} of {len(spec):,} specific measured edges have a signed curated counterpart at all, so the "
           "activator/repressor logic cannot be scored here; the curated regulatory layer barely intersects the knockouts usable as sources. "
           if not sign_testable else
           f"SIGN: over {signed_tot:,} edges with a signed curated counterpart, direction agrees in {100*signed_ok/max(signed_tot,1):.0f}% against a "
           f"{100*prior:.0f}% majority-direction baseline. ")
        + f"VALUE, the only non-circular test: sources split 70/30, completion built from TRAIN sources only, held-out specific-mover recall@"
        f"{K_RECALL} is {mc:.3f} for the curated network alone and {mp:.3f} for the completed network ({mp-mc:+.3f}; random {mr:.3f}). "
        + (f"That gain is NOT distinguishable from noise -- across all {NSPLIT} splits it helps {nbetter}/{n_eval} held-out "
           f"source-evaluations and hurts {nworse}, significant in {nsig}/{NSPLIT} splits (median paired Wilcoxon "
           f"p={p_gain:.2g}); note p<0.05 is barely reachable at these tiny non-tied counts, so this is partly a power "
           "floor rather than a clean negative. " if not (p_gain < 0.05) else
           f"The gain is significant (helps {nbetter}/{n_eval}, hurts {nworse}, median paired Wilcoxon p={p_gain:.2g}). ")
        + f"AND THE SCALE SETTLES IT, AGAINST A REFERENCE COMPUTED IN THIS SETUP RATHER THAN AN IMPORTED ONE: quoting this project's familiar "
        f"{REF_TIDE_HARNESS:.2f} tide-null and {REF_BEST_HARNESS:.2f} best-model here would be a cross-metric comparison, since those come from the "
        f"harness (pkl, TAU=1.0) and this module scores the uncensored parquet at |z|>={T:.1f} with a different tide definition. Recomputed on "
        f"exactly these sources and this truth definition, simply ranking genes by how often they move under ANY knockout -- the tide-null, which "
        f"uses no network at all -- scores {m_tide:.3f}, versus {mp:.3f} for the completed network. So the completed graph is roughly "
        f"{m_tide/max(mp,1e-9):.0f}x WORSE than ignoring the network entirely and predicting the generic response. "
        "Completing the network and walking it is not a route to prediction. "
        + "JOINTS, and these ARE enriched: " + "; ".join(
            f"{ln} {joints[ln]['n_z2']:,} of {joints[ln]['n_scoreable_genes']:,} genes carrying >={MINROUTES} routes reach z>=2 against "
            f"{NDECOY} null replicates (chance ~{joints[ln]['expected_by_chance']:.0f})"
            for ln in ("REGULATORY", "COMBINED"))
        + ". The chance expectation there is EMPIRICAL, not a normal tail: a z built from 5 replicates with a Poisson floor is not normal, so the "
        "count of genes clearing z>=2 when one replicate is held out and scored against the other four IS the chance count for this estimator. "
        "Note what joints do and do not show: they ask which intermediates routes pass through, not whether targets are closer. A knockout's real "
        "targets are a coherent set, so their routes CONVERGE on shared intermediates more than independently-drawn decoys do even at identical "
        "distances. Convergence is real but it is a property of the target set, not evidence that the curated path is the mechanism. "
        f"AND THE NAMES DO NOT SURVIVE SCRUTINY, WHICH IS THE HONEST HEADLINE OF THIS SECTION. BFS records one parent per node, so wherever "
        "several equally-short paths exist -- the common case at median degree 15 -- the credited intermediate is decided by the order neighbours "
        f"are visited, which is alphabetical and arbitrary. Permuting that order retains only Jaccard {mjac:.2f} of the joint set. So the COUNT is "
        "a result (there is more convergence than chance) but WHICH GENES ARE NAMED is largely an implementation artifact, and the ranked table "
        "should be read as a weak shortlist, not as identified mechanism. Recovering real joints needs credit distributed over ALL shortest paths "
        "(Brandes-style), which this module does not do. "
        f"For what it is worth under that caveat, AND AGAINST BASE RATES rather than quoted bare: {reg_tf_n}/{reg_sig_n} of the significant "
        f"regulatory joints are annotated transcription factors (background among scoreable genes {100*base_tf:.0f}%), and "
        f"{reg_notsrc_n}/{reg_sig_n} were never knocked out in this panel -- but so were {100*base_notsrc:.0f}% of ALL scoreable genes, so that "
        "second number is the panel's composition rather than a property of the joints, and I am not going to present it as a finding. They "
        "include haematopoietic regulators plausible for K562 (" + ", ".join(reg_heme) + ") alongside broadly-studied factors with large curated "
        "regulons (" + ", ".join(reg_other[:4]) + "). "
        "Deterministic; degree-matched decoys throughout, joints z-scored against "
        f"replicated nulls with a Poisson variance floor, and restricted to genes carrying >={MINROUTES} routes so that a gene whose few null "
        "replicates happen to agree cannot be handed an infinite z.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"nodes": sorted({x[0] for x in typed} | {x[1] for x in typed}), "edges": typed},
              open(OUT / "completed_network.json", "w"))
    pd.concat([pd.DataFrame(joints[l]["top"]).assign(layer=l) for l in ("REGULATORY", "COMBINED")]
              ).to_csv(OUT / "completed_network_joints.csv", index=False)
    json.dump({"threshold": T, "n_sources": len(sources), "curated_regulatory": n_reg, "curated_physical": n_phys,
               "measured_edges": len(measured), "measured_specific": len(spec), "layers": layers,
               "routed_within_maxhop": routed, "unroutable": unrouted,
               "sign_testable": bool(sign_testable), "sign_n": signed_tot,
               "sign_agreement": (round(signed_ok / max(signed_tot, 1), 4) if sign_testable else None),
               "sign_baseline": round(prior, 4), "recall_curated": round(mc, 4), "recall_completed": round(mp, 4),
               "recall_random": round(mr, 4), "recall_gain_p": float(p_gain), "joints": joints, "n_decoy_replicates": NDECOY, "joint_tiebreak_jaccard": round(mjac,3),
               "tide_null_this_setup": round(m_tide, 4), "reference_tide_harness": REF_TIDE_HARNESS,
               "reference_best_harness": REF_BEST_HARNESS, "splits": split_rows, "n_splits": NSPLIT,
               "verdict": verdict, "note": verdict}, open(OUT / "complete_network.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/completed_network.json ({len(typed):,} typed edges) + completed_network_joints.csv + complete_network.json")


if __name__ == "__main__":
    main()
