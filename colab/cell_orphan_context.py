"""BIOLOGICAL CONTEXT FOR ORPHAN CATALYSTS: does WHERE and WHICH WAY beat chemistry alone?

THE PROPOSAL, and an honest accounting of which parts are already done.  For a missing enzyme we know the
substrate coming in and the product going out; the suggestion is to narrow candidates by clearing out the
impossible first, then anchoring on the upstream structure, the compartment, and the downstream product.

    "clear out proteins not found in a cell"    ALREADY ENFORCED, adds nothing. The candidate vocabulary is the
                                                5,282 genes annotated as catalysts in Reactome + HumanGEM.
                                                Every one is human by construction; there is nothing foreign to
                                                remove.
    "clear out molecules the cell doesn't use"  ALSO ALREADY ENFORCED. The metabolite universe is the same two
                                                human-curated sources. No foreign compounds are in the network.
    "anchor on the structure before it"         PARTLY NEW -- see `direction` below.
    "spatial data for further specificity"      GENUINELY NEW. 28,498 of 28,528 reactions carry a compartment on
                                                at least one participant, and every one of the 5,282 catalysts
                                                can be given a compartment. Never used.
    "look at the end product too"               PARTLY NEW -- see `direction` below.

So two real levers, and this measures both against the chemistry-only baseline.

    COMPARTMENT   the baseline ignores location entirely. A nuclear enzyme cannot catalyse a mitochondrial
                  reaction, yet nothing stops it being proposed. Candidates are PARTITIONED into
                  compartment-consistent and not, each partition keeping its chemistry order. This is a pure
                  re-ordering with NO free parameter -- nothing to tune, so nothing to tune on the test set.
    DIRECTION     the baseline pools `in` and `out` participants into one bag and takes Jaccard over the union,
                  so a neighbour matching this reaction's PRODUCTS scores the same as one matching its
                  SUBSTRATES. Directional scoring matches in-set to in-set and out-set to out-set, which is what
                  "connect the thing before to the thing after" actually means.

THE CIRCULARITY THIS HAS TO AVOID, and it is easy to get wrong.  Gene -> compartment is learned FROM catalysed
reactions. If the held-out reaction contributes to its own catalyst's compartment profile, the filter is being
told the answer. The map is therefore built once with per-reaction contributions and the held-out reaction's
contribution is SUBTRACTED before it is used, every time.

CONTROLS, since a re-ordering always changes recall somehow:
    compart_perm   the identical partition with gene -> compartment SHUFFLED across genes. If the real map does
                   not beat this, the gain is an artifact of re-ordering, not of location.
    freq           rank by how often a gene is a catalyst, ignoring chemistry entirely.
    rand           uniform draw from the vocabulary. The floor.

PREDECLARED, before any number:
    compartment beats chem at every k AND beats compart_perm    -> location is informative and belongs in the
                                                                   candidate generator.
    compartment ~ compart_perm                                  -> the gain is re-ordering, not biology.
    compartment ~ chem                                          -> location adds nothing over chemistry, which
                                                                   would mean chemistry already implies it.
    direction beats chem at every k                              -> substrate/product asymmetry carries signal
                                                                   the pooled bag was discarding.

Recall is measured exactly as `cell_orphan_candidates` did -- hide a known catalyst, ask whether it comes back
in the top k -- so the baseline column is directly comparable to the recorded 0.276 / 0.443 / 0.552 / 0.596.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT = A.OUT
NONCAT = ("binding", "dissociation")
ABSTRACT = ("omitted", "uncertain")
KS = (1, 5, 20, 100)
N_EVAL = 1500

# HumanGEM single letters and Reactome long names name the SAME places in one field. Coarse buckets, because
# "mitochondrial matrix" vs "mitochondrial inner membrane" is a finer distinction than the candidate generator
# can use, and over-fine buckets would make nothing match.
_LETTER = {"c": "cytosol", "e": "extracellular", "m": "mitochondrion", "n": "nucleus", "r": "er",
           "x": "peroxisome", "g": "golgi", "l": "lysosome", "i": "mitochondrion", "s": "extracellular"}
_PREFIX = [("cytosol", "cytosol"), ("cytoplasm", "cytosol"), ("nucleo", "nucleus"), ("nucleus", "nucleus"),
           ("plasma membrane", "plasma membrane"), ("extracellular", "extracellular"),
           ("endoplasmic", "er"), ("mitochondri", "mitochondrion"), ("golgi", "golgi"),
           ("lysosom", "lysosome"), ("peroxisom", "peroxisome"), ("endosom", "endosome"),
           ("ciliary", "cilium"), ("cilium", "cilium"), ("nuclear envelope", "nucleus")]


def norm_comp(x):
    x = (x or "").strip().lower()
    if not x:
        return None
    if len(x) <= 2:
        return _LETTER.get(x, x)
    for pre, tgt in _PREFIX:
        if x.startswith(pre):
            return tgt
    return x


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("BIOLOGICAL CONTEXT FOR ORPHAN CATALYSTS -- does WHERE and WHICH WAY beat chemistry alone?")
    report("=" * 100)
    report("  Two of the proposed filters are ALREADY ENFORCED and add nothing: the candidate vocabulary is")
    report("  human catalysts from Reactome + HumanGEM, and the metabolite universe is the same two human")
    report("  sources. There are no non-human proteins or foreign molecules in the network to remove.")
    report("  The two real levers are COMPARTMENT (never used) and DIRECTION (in/out currently pooled).")
    report("  PREDECLARED: compartment beats chem at every k AND beats a SHUFFLED compartment map => location")
    report("  is informative. compartment ~ shuffled => the gain is re-ordering, not biology.")

    net = json.load(open(OUT / "reaction_network.json"))
    steps = net["steps"]
    currency = set(str(x) for x in (net.get("currency") or []))

    def side(s, which):
        return {str(p.get("id") or p.get("name") or "")
                for p in (s.get(which) or [])} - currency - {""}

    P = [side(s, "in") | side(s, "out") for s in steps]
    PIN = [side(s, "in") for s in steps]
    POUT = [side(s, "out") for s in steps]
    CMP = [{norm_comp(p.get("compartment"))
            for w in ("in", "out") for p in (s.get(w) or [])} - {None} for s in steps]

    cats = {i: list(s["catalysts"]) for i, s in enumerate(steps) if s.get("catalysts")}
    cat_of_part = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of_part[p].add(i)
    allcats = sorted({c for cs in cats.values() for c in cs})
    freq_rank = [c for c, _ in Counter(c for cs in cats.values() for c in cs).most_common()]
    report(f"\n  {len(steps):,} steps | {len(cats):,} catalysed | {len(allcats):,} distinct catalysts (vocabulary)")

    # gene -> compartment counter, with per-reaction contributions so the held-out one can be SUBTRACTED
    g2c = defaultdict(Counter)
    for i, cs in cats.items():
        for g in cs:
            g2c[g].update(CMP[i])
    ncomp = len({c for v in g2c.values() for c in v})
    single = sum(1 for g in g2c if len(g2c[g]) == 1)
    report(f"  compartments: {ncomp} normalised buckets | {len(g2c):,} catalysts placed, "
           f"{single:,} in exactly one ({single/max(len(g2c),1):.0%})")

    rng = np.random.default_rng(A.SEED)
    perm_genes = list(g2c)
    perm_target = list(perm_genes)
    rng.shuffle(perm_target)
    g2c_perm = {a: g2c[b] for a, b in zip(perm_genes, perm_target)}

    def gene_comps(g, held, mp):
        """compartments of gene g, with the held-out reaction's contribution removed -- otherwise the filter is
        told the answer by the very reaction it is being asked about."""
        c = mp.get(g)
        if not c:
            return set()
        if held is not None and g in cats.get(held, ()):
            c = Counter(c)
            c.subtract(CMP[held])
            return {k for k, v in c.items() if v > 0}
        return set(c)

    def rank(i, directional, comp_map):
        """chemistry score, optionally directional, optionally partitioned by compartment consistency"""
        sc = defaultdict(float)
        pi, pin, pout = P[i], PIN[i], POUT[i]
        if not pi:
            return []
        seen = set()
        for p in pi:
            for j in cat_of_part.get(p, ()):
                if j in seen or j == i:
                    continue
                seen.add(j)
                if directional:
                    a = len(pin & PIN[j]) / len(pin | PIN[j]) if (pin | PIN[j]) else 0.0
                    b = len(pout & POUT[j]) / len(pout | POUT[j]) if (pout | POUT[j]) else 0.0
                    v = 0.5 * (a + b)
                else:
                    u = len(pi | P[j])
                    v = len(pi & P[j]) / u if u else 0.0
                for c in cats[j]:
                    if v > sc[c]:
                        sc[c] = v
        order = [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]
        if comp_map is None:
            return order
        want = CMP[i]
        if not want:
            return order
        hit = [c for c in order if gene_comps(c, i, comp_map) & want]
        miss = [c for c in order if c not in set(hit)]
        return hit + miss

    real = [i for i, s in enumerate(steps)
            if not s.get("catalysts") and s.get("category") not in NONCAT + ABSTRACT]
    evalable = [i for i in cats if P[i] and CMP[i]]
    pick = rng.choice(len(evalable), min(N_EVAL, len(evalable)), replace=False)
    ev = [evalable[int(x)] for x in pick]
    report(f"  real orphan pool {len(real):,} | evaluating on {len(ev):,} reactions with a KNOWN catalyst, hidden")

    ARMS = ("chem", "chem_compart", "chem_dir", "chem_dir_compart", "compart_perm", "freq", "rand")
    hits = {a: {k: 0 for k in KS} for a in ARMS}
    nmatch = 0
    for i in ev:
        truth = set(cats[i])
        base = rank(i, False, None)
        comp = rank(i, False, g2c)
        dirn = rank(i, True, None)
        dcmp = rank(i, True, g2c)
        prm = rank(i, False, g2c_perm)
        rnd = list(rng.choice(allcats, min(max(KS), len(allcats)), replace=False))
        nmatch += bool(base and set(base[:20]) != set(comp[:20]))
        for k in KS:
            hits["chem"][k] += bool(truth & set(base[:k]))
            hits["chem_compart"][k] += bool(truth & set(comp[:k]))
            hits["chem_dir"][k] += bool(truth & set(dirn[:k]))
            hits["chem_dir_compart"][k] += bool(truth & set(dcmp[:k]))
            hits["compart_perm"][k] += bool(truth & set(prm[:k]))
            hits["freq"][k] += bool(truth & set(freq_rank[:k]))
            hits["rand"][k] += bool(truth & set(rnd[:k]))
    n = len(ev)
    report(f"  compartment partition changed the top-20 for {nmatch:,}/{n:,} reactions "
           f"({nmatch/n:.0%}) -- if this were ~0 the filter could not matter either way")

    report(f"\n  RECALL@k  (n={n:,})")
    report(f"    {'arm':<20}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    rec = {}
    for a in ARMS:
        rec[a] = {str(k): hits[a][k] / n for k in KS}
        report(f"    {a:<20}" + "".join(f"{rec[a][str(k)]:>9.3f}" for k in KS))

    report("\n  READING")
    c_beats_chem = all(rec["chem_compart"][str(k)] > rec["chem"][str(k)] for k in KS)
    c_beats_perm = all(rec["chem_compart"][str(k)] >= rec["compart_perm"][str(k)] for k in KS) and \
        any(rec["chem_compart"][str(k)] > rec["compart_perm"][str(k)] for k in KS)
    d_beats = all(rec["chem_dir"][str(k)] > rec["chem"][str(k)] for k in KS)
    if c_beats_chem and c_beats_perm:
        report(f"  LOCATION IS INFORMATIVE. compartment lifts recall@20 from {rec['chem']['20']:.3f} to "
               f"{rec['chem_compart']['20']:.3f}, and beats a SHUFFLED compartment map "
               f"({rec['compart_perm']['20']:.3f}),")
        report("  so the gain is where the enzyme actually is, not the re-ordering itself.")
    elif c_beats_chem:
        report(f"  compartment lifts recall@20 to {rec['chem_compart']['20']:.3f} from {rec['chem']['20']:.3f}, "
               f"BUT a shuffled map reaches {rec['compart_perm']['20']:.3f}.")
        report("  The gain is the partition, not the biology: any re-ordering that pushes some candidates up")
        report("  helps here. Location adds nothing real.")
    else:
        report(f"  LOCATION ADDS NOTHING. compartment {rec['chem_compart']['20']:.3f} vs chem "
               f"{rec['chem']['20']:.3f} at k=20.")
        report("  Chemistry already implies location -- reactions sharing metabolites tend to share a")
        report("  compartment, so the filter is redundant with the neighbours it is filtering.")
    if d_beats:
        report(f"  DIRECTION HELPS: matching substrates to substrates and products to products lifts recall@20")
        report(f"  to {rec['chem_dir']['20']:.3f} from the pooled {rec['chem']['20']:.3f}.")
    else:
        report(f"  DIRECTION DOES NOT HELP: {rec['chem_dir']['20']:.3f} vs pooled {rec['chem']['20']:.3f}. "
               f"Matching the bag of participants is as good as matching sides.")

    json.dump({"test": "cell_orphan_context", "n_eval": n, "ks": list(KS), "recall": rec,
               "n_vocab": len(allcats), "n_compartments": ncomp, "n_catalysts_placed": len(g2c),
               "frac_top20_changed": nmatch / n, "already_enforced": ["non-human proteins", "foreign molecules"],
               "log": log}, open(OUT / "cell_orphan_context.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'cell_orphan_context.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
