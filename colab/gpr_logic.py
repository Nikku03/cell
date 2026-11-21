"""GPR LOGIC: does the AND/OR gate change the answer? Isozymes buffer; subunits do not.

WHAT EVERY PREVIOUS TEST IN THIS PROJECT GOT WRONG ABOUT REACTIONS. The `reaction` channel of the network
joins any two genes that appear in the same reaction, as one undirected edge. That is biologically incoherent,
and incoherent in a way that cancels:

    a OR b   isozymes. Knock out a, and b still runs the reaction. NOTHING should propagate.
    a AND b  subunits of one complex. Knock out a, and the reaction stops. EVERYTHING should propagate.

An undirected "shares a reaction" edge treats those identically, so a real signal (AND) is averaged against a
real ANTI-signal (OR). Human-GEM is dominated by the buffered case -- 3,830 reactions contain an OR against
700 containing an AND -- so the average is pulled toward the case where nothing should happen. If the gate
matters, splitting the channel should separate two arms that the pooled edge hid.

This is build 1's GPR arithmetic (`ec_capacity.py`) applied to the RESPONSE data rather than to flux:

    a gene is ESSENTIAL to a reaction if evaluating the GPR with that gene alone set False turns the whole
    rule False. It is REDUNDANT if the rule still evaluates True.

Then three things get measured, all against frequency-matched controls:

  1 are AND-partners (both essential) enriched for being movers of each other's knockout?
  2 are OR-partners (redundant) depleted, as the biology says they should be?
  3 does GPR-aware propagation -- only through reactions the knockout actually BREAKS -- beat the naive
    "shares any reaction" edge, and does either approach the frequency baseline?

COVERAGE IS STATED FIRST, not discovered later. Human-GEM covers metabolism only, so most knockouts in a
genome-wide screen are not in it at all, and that caps what this can explain regardless of how well the logic
works.
"""
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_SPEC, NPICK = 5, 50
RNG = np.random.default_rng(0)
R = {}


def hdr(s):
    print("\n" + "=" * 104)
    print(s)
    print("=" * 104)


def bci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    return tuple(float(v) for v in np.percentile(
        r.choice(x, size=(n, len(x)), replace=True).mean(1), [2.5, 97.5]))


# ------------------------------------------------------------------ SBML GPR parsing
TAG = re.compile(r'<(/?)fbc:(and|or|geneProductRef)\b([^>]*)>')
GP = re.compile(r'fbc:geneProduct="([^"]+)"')


def parse_gpr(block):
    """Nested fbc:and / fbc:or / fbc:geneProductRef -> ('and'|'or', [children]) or ('leaf', gene_id).

    Built with an explicit stack over the tag stream rather than an XML parser, because the namespace
    prefixes make ElementTree awkward on fragments and the grammar here is three tags wide.
    """
    stack, root = [], None
    for m in TAG.finditer(block):
        close, tag, attrs = m.group(1), m.group(2), m.group(3)
        if tag == "geneProductRef":
            g = GP.search(attrs)
            if g and stack:
                stack[-1][1].append(("leaf", g.group(1)))
            elif g:
                root = ("leaf", g.group(1))
            continue
        if not close:
            node = (tag, [])
            if stack:
                stack[-1][1].append(node)
            stack.append(node)
        else:
            node = stack.pop()
            if not stack:
                root = node
    return root


def alive(node, dead):
    """Evaluate the rule with every gene in `dead` knocked out."""
    if node is None:
        return True
    t = node[0]
    if t == "leaf":
        return node[1] not in dead
    if t == "and":
        return all(alive(c, dead) for c in node[1])
    return any(alive(c, dead) for c in node[1])


def leaves(node, out=None):
    out = [] if out is None else out
    if node is None:
        return out
    if node[0] == "leaf":
        out.append(node[1])
    else:
        for c in node[1]:
            leaves(c, out)
    return out


def load_reactions():
    """reaction id -> (gpr tree, [gene symbols]).  Also returns ensembl -> symbol."""
    path = SP / "HumanGEM.xml"
    assert path.exists(), f"{path} missing"
    sym = {}
    for line in open(path):
        if "<fbc:geneProduct " in line:
            i = re.search(r'fbc:id="([^"]+)"', line)
            l = re.search(r'fbc:label="([^"]+)"', line)
            if i and l:
                sym[i.group(1)] = l.group(1)
    rxns = {}
    rid, buf, ingpr = None, [], False
    for line in open(path):
        m = re.search(r'<reaction [^>]*\bid="([^"]+)"', line)
        if m:
            rid = m.group(1)
        if "<fbc:geneProductAssociation" in line:
            ingpr, buf = True, []
        if ingpr:
            buf.append(line)
        if ingpr and "</fbc:geneProductAssociation>" in line:
            ingpr = False
            tree = parse_gpr(" ".join(buf))
            if rid and tree:
                rxns[rid] = tree
        if "</reaction>" in line:
            rid = None
    return rxns, sym


def main():
    import cellformer_af as CF
    d = CF.load_all()
    genes, kos, gi = d["genes"], d["kos"], d["gi"]
    spec, tide, nspec = d["spec"], d["tide"], d["nspec"]
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [q for q in range(len(kos)) if not tr[q] and nspec[q] >= MIN_SPEC and kos[q] in gi]
    freq = spec[tr].mean(0)
    cnt = spec[tr].sum(0)

    hdr("GPR LOGIC -- isozymes (OR) vs subunits (AND)")
    rxns, ens2sym = load_reactions()
    print(f"Human-GEM: {len(rxns):,} reactions with a gene rule, {len(ens2sym):,} gene products")

    # per reaction: which genes are ESSENTIAL (removing them breaks it) vs REDUNDANT
    ess_of = collections.defaultdict(set)      # gene symbol -> reactions it is essential to
    red_of = collections.defaultdict(set)      # gene symbol -> reactions where it is redundant
    rxn_genes = {}
    n_and = n_or = 0
    for rid, tree in rxns.items():
        lv = sorted(set(leaves(tree)))
        if not lv:
            continue
        syms = [ens2sym.get(g, g) for g in lv]
        rxn_genes[rid] = syms
        ess_here = 0
        for g, s in zip(lv, syms):
            if not alive(tree, {g}):
                ess_of[s].add(rid)
                ess_here += 1
            else:
                red_of[s].add(rid)
        n_and += (ess_here == len(lv) and len(lv) > 1)
        n_or += (ess_here == 0 and len(lv) > 1)
    print(f"  multi-gene reactions where EVERY gene is essential (pure AND): {n_and:,}")
    print(f"  multi-gene reactions where NO gene is essential (pure OR):     {n_or:,}")
    print(f"  genes essential to >=1 reaction: {len(ess_of):,}; redundant in >=1: {len(red_of):,}")

    ko_in_gem = [q for q in test if kos[q] in ess_of or kos[q] in red_of]
    ko_breaks = [q for q in test if kos[q] in ess_of]
    print(f"\nCOVERAGE (stated first):")
    print(f"  held-out scorable                       {len(test):,}")
    print(f"  KO gene appears in Human-GEM            {len(ko_in_gem):,} ({100*len(ko_in_gem)/len(test):.1f}%)")
    print(f"  KO gene BREAKS >=1 reaction             {len(ko_breaks):,} ({100*len(ko_breaks)/len(test):.1f}%)"
          f"   <- the only ones GPR-aware propagation can ever explain")

    # ---------------------------------------------------------------- 1/2. AND vs OR partner enrichment
    hdr("1-2. Are AND-partners enriched, and OR-partners depleted?")
    # POOL OVER EVERY PERTURBATION, not just the held-out 419. This is a descriptive enrichment, not a fitted
    # predictor, so there is no train/test leakage to worry about -- and restricted to held-out only it had
    # 247-746 pairs at a ~0.004 base rate, where the matched control expects 1-3 hits and returned ZERO. That
    # produced LIFT = nan for every arm: not a negative result, an unmeasurable one. Power first.
    allp = [q for q in range(len(kos)) if nspec[q] >= MIN_SPEC and kos[q] in gi]
    print(f"  pooled over {len(allp):,} scorable perturbations (train+test) for statistical power")
    by_cnt = collections.defaultdict(list)
    moved = np.where(spec.any(0) & ~tide)[0]
    for g in moved:
        by_cnt[int(cnt[g])].append(int(g))
    by_cnt = {k: np.array(v) for k, v in by_cnt.items()}

    # ONE control draw per pair left the control estimate on 2-5 expected hits, so it swung 13x between arms
    # (0.0078 / 0.0008 / 0.0006) and every LIFT was a ratio of Poisson noise. Averaging K independent matched
    # draws per pair cuts the control's variance by K without touching the observed side.
    KCTRL = int(os.environ.get("GPR_KCTRL", "40"))

    def enrich(pairs_of):
        hit = tot = 0
        chit = ctot = 0
        for q in allp:
            k = kos[q]
            partners = pairs_of(k)
            partners = [p for p in partners if p in gi and not tide[gi[p]] and p != k]
            if not partners:
                continue
            for p in partners:
                j = gi[p]
                tot += 1
                hit += int(spec[q, j])
                cand = by_cnt.get(int(cnt[j]))
                if cand is not None and len(cand) > 1:
                    picks = RNG.choice(cand, size=KCTRL)
                    ctot += KCTRL
                    chit += int(spec[q, picks].sum())
        if not tot or not ctot:
            return None
        o, e = hit / tot, chit / ctot
        return {"n": tot, "hits": hit, "obs": o, "ctrl": e, "lift": (o / e) if e else float("nan")}

    def and_partners(k):
        """genes sharing a reaction that k is ESSENTIAL to, and that are themselves essential there."""
        out = set()
        for rid in ess_of.get(k, ()):
            for s in rxn_genes.get(rid, ()):
                if s != k and rid in ess_of.get(s, ()):
                    out.add(s)
        return out

    def or_partners(k):
        """genes sharing a reaction where k is REDUNDANT -- the isozyme case."""
        out = set()
        for rid in red_of.get(k, ()):
            for s in rxn_genes.get(rid, ()):
                if s != k:
                    out.add(s)
        return out

    def any_partners(k):
        out = set()
        for rid in list(ess_of.get(k, ())) + list(red_of.get(k, ())):
            for s in rxn_genes.get(rid, ()):
                if s != k:
                    out.add(s)
        return out

    rows = [("AND-partner (both essential)", and_partners),
            ("OR-partner (knockout is redundant)", or_partners),
            ("ANY reaction partner (the naive edge)", any_partners)]
    R["enrich"] = {}
    for tag, fn in rows:
        r = enrich(fn)
        if r:
            print(f"  {tag:40s} n={r['n']:>6,}  P(mover) {r['obs']:.4f}  matched control {r['ctrl']:.4f}  "
                  f"LIFT {r['lift']:.2f}x   (obs hits {r['hits']}, control averaged over {KCTRL} draws/pair)")
            R["enrich"][tag] = r
        else:
            print(f"  {tag:40s} too few pairs to measure")

    # ---------------------------------------------------------------- 3. GPR-aware propagation
    hdr("3. GPR-aware propagation vs the naive reaction edge")
    sub = ko_breaks if len(ko_breaks) >= 20 else ko_in_gem
    print(f"  scored on the {len(sub):,} held-out perturbations whose KO gene is in Human-GEM"
          f"{' and breaks a reaction' if sub is ko_breaks else ''}")

    # NO FREQUENCY TIEBREAK. A first version added 1e-6 * freq to break ties among the ~10-30 genes a
    # reaction rule actually names. Because that leaves ~8,200 genes tied at zero, the top-50 was then filled
    # ALMOST ENTIRELY BY FREQUENCY, and every arm scored 0.32-0.38 next to a 0.3843 frequency baseline --
    # including "isozyme partners only", which was supposed to be near zero and came out HIGHEST at 0.3778.
    # That was the tell: the arms were measuring the frequency prior with a rounding error on top, and the
    # "+0.0088, the gate matters" that fell out of it was a reshuffle of frequency on n=26.
    # Reported instead: what the rule alone reaches, and how pure that set is.
    def score_arm(fn):
        cov, prec, npart = [], [], []
        for q in sub:
            T = set(int(x) for x in np.where(spec[q])[0] if not tide[x])
            if not T:
                continue
            P = {gi[p] for p in fn(kos[q]) if p in gi and not tide[gi[p]]}
            npart.append(len(P))
            cov.append(len(P & T) / len(T))
            prec.append(len(P & T) / max(len(P), 1))
        return np.array(cov), np.array(prec), np.array(npart)

    arms = [("GPR-aware: only BROKEN reactions", and_partners),
            ("naive: shares ANY reaction", any_partners),
            ("isozyme partners (buffered -- expect ~0)", or_partners)]
    R["arms"] = {}
    print(f"    {'arm':42s} {'partners':>9s} {'coverage':>10s} {'precision':>10s}")
    for tag, fn in arms:
        cov, prec, npart = score_arm(fn)
        if len(cov) == 0:
            continue
        lo, hi = bci(cov)
        print(f"    {tag:42s} {npart.mean():>9.1f} {cov.mean():>10.4f} {prec.mean():>10.4f}"
              f"   cov CI [{lo:.4f},{hi:.4f}]")
        R["arms"][tag] = {"coverage": float(cov.mean()), "cov_ci": [lo, hi],
                          "precision": float(prec.mean()), "mean_partners": float(npart.mean()),
                          "n": int(len(cov))}
    fv = []
    for q in sub:
        T = set(int(x) for x in np.where(spec[q])[0] if not tide[x])
        if not T:
            continue
        f = freq.copy()
        f[tide] = -1
        fv.append(len(set(int(i) for i in np.argsort(-f)[:NPICK]) & T) / len(T))
    fv = np.array(fv)
    lo, hi = bci(fv)
    print(f"    {'frequency baseline (same subset)':40s} recall {fv.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
    R["frequency_same_subset"] = float(fv.mean())

    # is the GPR-aware arm better than the naive one, paired?
    a, ap, _ = score_arm(and_partners)
    b, bp, _ = score_arm(any_partners)
    m = min(len(a), len(b))
    if m:
        dd = ap[:m] - bp[:m]
        lo, hi = bci(dd)
        print(f"\n  PRECISION, GPR-aware MINUS naive: {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{'-> restricting to broken reactions is purer' if lo > 0 else '-> the gate does not change it'}")
        R["gpr_minus_naive_precision"] = {"delta": float(dd.mean()), "ci": [lo, hi], "n": int(m)}
    print(f"\n  For scale: the frequency baseline reaches {fv.mean():.4f} recall on this same subset,")
    print(f"  and n={len(sub)} is far too small for any of these to be a confident claim.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "gpr_logic.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'gpr_logic.json'}")


if __name__ == "__main__":
    main()
