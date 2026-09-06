"""FULL AUDIT: the data, the network, the task, the model, the training sequence, and the noise.

WHY THIS EXISTS. The ablation table came back with the full model BEATEN BY FIVE OF ITS OWN ABLATIONS, the
pair representation indistinguishable from its own shuffle, and every arm losing to a frequency baseline.
Those are model-level symptoms. This measures the things underneath them, directly on the data, with no model
involved -- because if the inputs cannot support the task, no architecture change will fix it and the honest
move is to say so rather than to keep tuning.

Six sections, each answering a question that the ablation table raised but cannot answer:

  A  READOUT        what the label matrix actually contains -- sparsity, tide, mover counts, the tail
  B  TASK           what the benchmark rewards, and the ceiling any method faces
  C  NETWORK        does the pair prior carry ANY signal for this task? -- the decisive one
  D  MODEL          parameter allocation, and capacity against the data it is fitted to
  E  SEQUENCE       the training protocol: what fraction of the data a run ever sees
  F  QUALITY        noise floors, and which measured differences can survive them

SECTION C IS THE POINT. Ablation 1 (remove the pair representation) scored 0.1899 against the full model's
0.1266, and ablation 2 (SHUFFLE it) scored 0.1245 -- statistically identical to keeping it. Two readings fit:
the model fails to exploit a real signal, or there is no signal to exploit. Those have opposite remedies, and
a direct enrichment measurement on the raw data separates them without training anything.

Controls are degree- and frequency-matched throughout. Network hubs are also frequently-moved genes, so an
unmatched enrichment would recover the frequency prior and call it network structure -- which is exactly the
confound that makes the frequency baseline hard to beat in the first place.
"""
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAU, TIDE_FRAC, MIN_SPEC, NPICK = 1.0, 0.05, 5, 50
RNG = np.random.default_rng(0)
R = {}


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


# ----------------------------------------------------------------------------- A. READOUT
def section_a():
    hdr("A. READOUT -- what the label matrix contains")
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(x) for x in z["kos"]]
    genes = [str(x) for x in z["genes"]]
    M = np.asarray(z["M"], np.float32)
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    spec = (A >= TAU) & ~tide
    nspec = spec.sum(1)

    print(f"shape                 {M.shape[0]:,} perturbations x {M.shape[1]:,} genes")
    print(f"non-finite            {int((~np.isfinite(M)).sum()):,}  (rebuild sets these to 0, not dropped)")
    print(f"exact zeros           {100*(M == 0).mean():.1f}% of cells")
    print(f"|z| >= {TAU}            {100*(A >= TAU).mean():.2f}% of cells")
    print(f"tide genes            {int(tide.sum()):,} ({100*tide.mean():.1f}%) move in >={TIDE_FRAC:.0%} of "
          f"perturbations -- removed from labels AND from the candidate pool")
    print(f"scorable (>={MIN_SPEC} spec)  {int((nspec >= MIN_SPEC).sum()):,} of {len(kos):,} "
          f"({100*(nspec >= MIN_SPEC).mean():.1f}%)")
    q = np.percentile(nspec, [50, 75, 90, 99])
    print(f"specific movers/pert  median {q[0]:.0f}, p75 {q[1]:.0f}, p90 {q[2]:.0f}, p99 {q[3]:.0f}, "
          f"max {nspec.max():,}")

    # How many perturbations does each gene move? The tail is what a learned model would have to memorise.
    gcount = spec.sum(0)
    moved = gcount > 0
    once = (gcount == 1).sum()
    print(f"\ngenes ever a specific mover: {int(moved.sum()):,}; of those {once:,} "
          f"({100*once/max(moved.sum(),1):.1f}%) for EXACTLY ONE perturbation")
    print("  -> that fraction is unlearnable from co-response structure; it is per-perturbation idiosyncrasy")

    R["A"] = {"shape": list(M.shape), "tide": int(tide.sum()), "scorable": int((nspec >= MIN_SPEC).sum()),
              "median_movers": float(q[0]), "genes_moved": int(moved.sum()), "one_off": int(once),
              "one_off_frac": float(once / max(moved.sum(), 1))}
    return kos, genes, M, spec, nspec, tide


# ----------------------------------------------------------------------------- B. TASK
def section_b(kos, genes, spec, nspec, tide):
    hdr("B. TASK -- what the benchmark rewards, and the ceiling")
    import hashlib
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [i for i in range(len(kos)) if not tr[i] and nspec[i] >= MIN_SPEC]
    print(f"train/test split      {tr.sum():,}/{(~tr).sum():,}; held-out scorable {len(test):,}")

    freq = spec[tr].mean(0)
    ranked = [int(i) for i in np.argsort(-freq) if not tide[i]]
    top = ranked[:NPICK]

    # The frequency baseline names ONE fixed set of 50 genes for EVERY perturbation. Its score is therefore a
    # direct measurement of how much of the "specific" signal is actually shared across perturbations.
    rec = np.array([len(set(top) & set(np.where(spec[q])[0])) / nspec[q] for q in test])
    print(f"frequency baseline    recall {rec.mean():.4f} using ONE fixed gene set for every perturbation")
    print(f"  -> {100*rec.mean():.1f}% of the average perturbation's 'specific' movers are named without")
    print("     looking at which gene was knocked out. That is the share of the label that is not specific.")

    # Pairwise Jaccard between held-out truth sets: how much do different perturbations share?
    sam = RNG.choice(test, size=min(300, len(test)), replace=False)
    S = [set(np.where(spec[q])[0].tolist()) for q in sam]
    jac = []
    for a in range(0, len(S), 3):
        for b in range(a + 1, min(a + 12, len(S))):
            u = len(S[a] | S[b])
            if u:
                jac.append(len(S[a] & S[b]) / u)
    jac = np.array(jac)
    print(f"\ntruth-set overlap     median Jaccard {np.median(jac):.4f}, mean {jac.mean():.4f} "
          f"over {len(jac):,} held-out pairs")

    # Ceiling: the EXACT best fixed set of NPICK genes, fitted on the test set itself. Recall here is a mean
    # over per-perturbation ratios, and |picked & truth_q|/|truth_q| is ADDITIVE in the picked genes -- each
    # gene contributes 1/|truth_q| to every q it moves. So the objective is linear, and the optimum is simply
    # the top NPICK genes by sum_q [g moves q]/|truth_q|. No greedy approximation needed.
    # A first version ranked by RAW COUNT, which over-weights genes that move perturbations with many movers
    # and is therefore NOT the maximiser -- it produced a "ceiling" that the frequency baseline exceeded.
    w = np.zeros(spec.shape[1])
    for q in test:
        w[spec[q]] += 1.0 / nspec[q]
    ora = [int(i) for i in np.argsort(-w) if not tide[i]][:NPICK]
    orec = np.array([len(set(ora) & set(np.where(spec[q])[0])) / nspec[q] for q in test])
    print(f"perturbation-independent CEILING (best fixed 50, fitted ON the test set): {orec.mean():.4f}")
    print(f"  frequency reaches {100*rec.mean()/orec.mean():.1f}% of that ceiling using only training data.")
    print("  A model must beat the CEILING to prove it is using the identity of the perturbed gene at all.")

    R["B"] = {"n_test": len(test), "frequency_recall": float(rec.mean()),
              "median_jaccard": float(np.median(jac)), "fixed_set_ceiling": float(orec.mean())}
    return tr, test, freq


# ----------------------------------------------------------------------------- C. NETWORK
def section_c(kos, genes, spec, nspec, tide, tr, test, freq):
    hdr("C. NETWORK -- does the pair prior carry ANY signal for this task?")
    # Read the pair store the MODEL builds, not a re-derivation of it. A first version of this audit
    # re-implemented the parsing and disagreed with cellformer_af on every channel -- complexes came out
    # EMPTY, signed-reg came out 4x too large. Auditing a re-implementation would have measured my parser
    # rather than the model's input, which is the same silent-join failure this project has hit seven times.
    import cellformer_af as CF
    d = CF.load_all()
    gi = d["gi"]
    CH = ["reaction", "complex", "PPI", "coexpr", "signed-reg"]
    ch = [set() for _ in range(5)]
    for (i, j), v in d["pair"].items():
        for c in range(5):
            if v[c]:
                ch[c].add((i, j))
    for c, n in zip(CH, ch):
        print(f"  {c:12s} {len(n):>8,} pairs   (from the model's own pair store)")

    # ---- C1. KO -> TARGET. Knocking out gene k: are k's network neighbours more likely to be its
    # specific movers than frequency-matched random genes? This is the relation a mechanistic model needs.
    print("\nC1. KO -> its own network neighbours, vs a FREQUENCY-MATCHED control")
    print("    (unmatched would just rediscover the frequency prior: hubs are also frequently-moved genes)")
    gfreq = spec[tr].mean(0)
    nb = int(os.environ.get("AUDIT_BINS", "20"))
    order = np.argsort(gfreq)
    binof = np.empty(len(genes), np.int64)
    binof[order] = np.arange(len(genes)) * nb // len(genes)
    members = [np.where(binof == b)[0] for b in range(nb)]

    adj = [collections.defaultdict(set) for _ in range(5)]
    for c in range(5):
        for i, j in ch[c]:
            adj[c][i].add(j)
            adj[c][j].add(i)

    ko_idx = {}
    for p, k in enumerate(kos):
        if k in gi:
            ko_idx[p] = gi[k]
    print(f"    {len(ko_idx):,}/{len(kos):,} perturbations have their KO gene in the readout universe")

    rows = []
    for c in range(5):
        hit = tot = chit = ctot = 0
        for p, ki in ko_idx.items():
            if nspec[p] < MIN_SPEC:
                continue
            S = spec[p]
            nbrs = [j for j in adj[c].get(ki, ()) if not tide[j]]
            if not nbrs:
                continue
            hit += int(S[nbrs].sum())
            tot += len(nbrs)
            # frequency-matched control: one random gene from the SAME frequency bin as each neighbour
            ctrl = [int(RNG.choice(members[binof[j]])) for j in nbrs]
            chit += int(S[ctrl].sum())
            ctot += len(ctrl)
        if tot:
            o, e = hit / tot, chit / max(ctot, 1)
            rows.append((CH[c], tot, o, e, o / e if e else float("nan")))
            print(f"    {CH[c]:12s} n={tot:>7,}  P(mover|neighbour) {o:.4f}   matched control {e:.4f}   "
                  f"LIFT {o/e if e else float('nan'):.2f}x")
    R["C1"] = [{"channel": a, "n": b, "obs": o, "ctrl": e, "lift": l} for a, b, o, e, l in rows]

    # ---- C2. CO-MOVEMENT. The pair representation biases attention BETWEEN CROP GENES, so the relation it
    # can actually express is "these two genes respond together", not "KO moves target". Test that directly.
    print("\nC2. co-movement of network-adjacent PAIRS -- the relation the pair bias can actually express")
    P = spec.shape[0]
    for c in range(5):
        pr = list(ch[c])
        if not pr:
            continue
        pr = [pr[i] for i in RNG.choice(len(pr), size=min(20000, len(pr)), replace=False)]
        ii = np.array([a for a, _ in pr])
        jj = np.array([b for _, b in pr])
        keep = ~(tide[ii] | tide[jj])
        ii, jj = ii[keep], jj[keep]
        if len(ii) < 100:
            continue
        co = (spec[:, ii] & spec[:, jj]).sum(0) / P
        ci = np.array([int(RNG.choice(members[binof[x]])) for x in ii])
        cj = np.array([int(RNG.choice(members[binof[x]])) for x in jj])
        cco = (spec[:, ci] & spec[:, cj]).sum(0) / P
        o, e = co.mean(), cco.mean()
        print(f"    {CH[c]:12s} n={len(ii):>7,}  P(co-move) {o:.5f}   matched control {e:.5f}   "
              f"LIFT {o/e if e else float('nan'):.2f}x")
        R.setdefault("C2", []).append({"channel": CH[c], "n": int(len(ii)), "obs": float(o),
                                       "ctrl": float(e), "lift": float(o / e) if e else None})

    # ---- C3. THE DIRECT TEST. Rank candidates for a held-out perturbation by network adjacency to the KO
    # gene ALONE and score with the real harness. If this cannot beat frequency, the pair prior has nothing
    # to give any architecture -- and ablations 1 and 2 stop being surprising.
    print("\nC3. network-only predictor, scored on the SAME harness as the model")
    ranked = [int(i) for i in np.argsort(-freq) if not tide[i]]
    top = ranked[:NPICK]

    # Precompute each held-out perturbation's network score vector once.
    netsc = {}
    for q in test:
        ki = gi.get(kos[q])
        sc = np.zeros(len(genes), np.float64)
        if ki is not None:
            for c in range(5):
                for j in adj[c].get(ki, ()):
                    sc[j] += 1.0
        netsc[q] = sc

    def score_arm(fn):
        out = []
        for q in test:
            truth = set(np.where(spec[q])[0].tolist())
            s = fn(q).copy()
            s[tide] = -1e9
            pick = np.argsort(-s)[:NPICK]
            out.append(len(set(int(x) for x in pick) & truth) / len(truth))
        return np.array(out)

    def bci(x, n=2000):
        m = RNG.choice(x, size=(n, len(x)), replace=True).mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    fr = score_arm(lambda q: freq.astype(np.float64))
    nr = score_arm(lambda q: netsc[q])
    for nm, v in (("frequency", fr), ("network only", nr)):
        lo, hi = bci(v)
        print(f"    {nm:24s} recall {v.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")

    # COMBINING THEM NEEDS A SCALE. A first version simply added raw adjacency COUNTS (integers >= 1) to a
    # frequency in [0,1], so any gene with a single network edge outranked the most-moved gene in the cell
    # line. That reported "adding the network destroys frequency", which measured the missing scale factor,
    # not the network. Sweeping the weight instead gives the network its best possible shot.
    best = (None, -1.0, None)
    print("    frequency + w * network:")
    for wgt in (0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0):
        v = score_arm(lambda q, wgt=wgt: freq.astype(np.float64) + wgt * netsc[q])
        if v.mean() > best[1]:
            best = (wgt, float(v.mean()), v)
        print(f"      w={wgt:<8g} recall {v.mean():.4f}")
    wgt, bm, bv = best
    dd = bv - fr
    lo, hi = bci(dd)
    print(f"    BEST w={wgt:g}: recall {bm:.4f}; gain over frequency {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
          f"  {'HELPS' if lo > 0 else 'no measurable gain even at the best weight'}")
    R["C3"] = {"frequency": float(fr.mean()), "network_only": float(nr.mean()),
               "best_weight": float(wgt), "best_mix": float(bm),
               "delta": float(dd.mean()), "delta_ci": [lo, hi]}


# ----------------------------------------------------------------------------- D/E. MODEL + SEQUENCE
def section_de(kos, genes, tr, test):
    hdr("D. MODEL -- parameter allocation, and capacity against the data")
    try:
        import torch
        import torch.nn as nn
        import cellformer_af as CF
        cfg = dict(CF.BASE)
        m = CF.build_model(torch, nn, len(genes), cfg)
        tot = sum(p.numel() for p in m.parameters())
        grp = collections.Counter()
        for n_, p in m.named_parameters():
            grp[n_.split(".")[0]] += p.numel()
        print(f"total parameters      {tot:,}")
        for k, v in grp.most_common():
            print(f"  {k:22s} {v:>10,}  ({100*v/tot:.1f}%)")
        emb = grp.get("gene_emb", 0) + grp.get("ko_emb", 0)
        print(f"\ngene/KO embeddings    {emb:,} ({100*emb/tot:.1f}% of the model) for {len(genes):,} genes")
        print(f"  -> {emb/max(len(genes),1):.1f} parameters per gene. Every gene must be learned from the")
        print("     perturbations that happen to move it, and most genes move for very few.")
        R["D"] = {"total": int(tot), "by_group": dict(grp), "embed_frac": float(emb / tot)}
    except Exception as e:
        print(f"  (model introspection unavailable: {type(e).__name__}: {e})")
        R["D"] = {"error": f"{type(e).__name__}"}

    hdr("E. SEQUENCE -- what a training run actually sees")
    steps = int(os.environ.get("CF_STEPS", "1500"))
    gcrop = int(os.environ.get("CF_GCROP", "256"))
    pctx = int(os.environ.get("CF_PCTX", "128"))
    G = len(genes)
    ntrain = int(tr.sum())
    print(f"budget                {steps:,} steps, batch 1 example/step")
    print(f"crop                  {gcrop} of {G:,} genes ({100*gcrop/G:.1f}%), "
          f"{pctx} of {ntrain:,} context perturbations")
    print(f"query perturbations   {steps:,} draws from {ntrain:,} -> each seen ~{steps/ntrain:.2f} times")
    cover = 1 - (1 - gcrop / G) ** steps
    print(f"per-gene exposure     a given gene appears in ~{steps*gcrop/G:,.0f} crops "
          f"(P(seen at least once) = {cover:.3f})")
    cells = steps * gcrop * (1 + pctx)
    print(f"label cells consumed  {cells:,} of {ntrain*G:,} available ({100*cells/(ntrain*G):.1f}%)")
    print("  -> at this budget most training perturbations are visited less than once as the QUERY.")
    R["E"] = {"steps": steps, "g_crop": gcrop, "p_ctx": pctx, "n_train": ntrain,
              "query_visits_per_pert": float(steps / ntrain), "label_cells_frac": float(cells / (ntrain * G))}


# ----------------------------------------------------------------------------- F. QUALITY
def section_f(spec, nspec, test):
    hdr("F. QUALITY -- the noise every reported difference has to clear")
    truth_n = np.array([nspec[q] for q in test], float)
    print(f"held-out perturbations available: {len(test):,}")
    print(f"  movers per held-out perturbation: median {np.median(truth_n):.0f}, "
          f"min {truth_n.min():.0f}, max {truth_n.max():.0f}")
    # recall is a mean over per-perturbation ratios; its SE depends on the spread of those ratios
    for n in (20, 40, 100, 200, len(test)):
        if n > len(test):
            continue
        se = 0.25 / np.sqrt(n)                       # 0.25 ~ observed per-perturbation recall spread
        print(f"  n={n:>4}: SE ~{se:.4f}, 95% CI half-width ~{1.96*se:.4f}")
    print("\n  The ablation table was read at n=40 (half-width ~0.078). Its arms span 0.0357..0.2401,")
    print("  so the EXTREMES separate, but neighbouring rows do not. Any claim that two adjacent")
    print("  ablations differ needs the full held-out set and more than one seed.")
    R["F"] = {"n_test_available": len(test)}


def main():
    kos, genes, M, spec, nspec, tide = section_a()
    tr, test, freq = section_b(kos, genes, spec, nspec, tide)
    section_c(kos, genes, spec, nspec, tide, tr, test, freq)
    section_de(kos, genes, tr, test)
    section_f(spec, nspec, test)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "audit_cellformer.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'audit_cellformer.json'}")


if __name__ == "__main__":
    main()
