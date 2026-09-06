"""WHAT ACTUALLY COVERS THE (KNOCKOUT -> MOVER) PAIRS? Annotation completeness is not answer completeness.

THE OBJECTION THIS ANSWERS, and it is a fair one. The cell model holds ~99% of genes, essentially all
proteins, PPI, complexes, and roughly half the pathways. So why does the network reach only 4.79% of a
knockout's true movers?

BECAUSE COVERING EVERY GENE IS NOT COVERING EVERY PAIR. With 8,246 genes there are ~34 million possible gene
pairs. The whole network holds ~250k edges -- 0.7% of them. Having a row for every gene says the NODES are
complete; it says nothing about whether the specific pair (knocked-out gene, gene whose mRNA moved) is one of
the 0.7% that got an edge. Those are different completeness claims and this file separates them.

THE DEEPER POSSIBILITY, which is the real question: MODALITY MISMATCH. A PPI network records who physically
touches whom. Perturb-seq records whose mRNA changes when you break something. Those are not the same
relation. A kinase and its substrate touch; knocking out the kinase need not change the substrate's mRNA at
all. Conversely two genes whose mRNA always moves together may never touch.

If that is the issue, then a relation measured in the SAME modality should cover far more -- and one exists
that is not circular: the same gene knocked out in a DIFFERENT cell line. Same kind of measurement, different
experiment. So this compares, on identical (knockout, mover) pairs:

    curated structure   PPI, complex, reaction, co-expression, signed regulation, GO/pathway co-membership
    same modality       did this gene also move when the same gene was knocked out in RPE1 / HepG2 / Jurkat /
                        HCT116 / Melanoma?

If curated structure covers ~5% and cross-cell-line co-response covers much more, the problem is not that the
databases are incomplete. It is that they describe a different relation than the one being predicted, and no
amount of adding edges of that kind will close it.
"""
import collections
import gzip
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
RNG = np.random.default_rng(0)
R = {}


def hdr(s):
    print("\n" + "=" * 104)
    print(s)
    print("=" * 104)


def load_line(name):
    d = pickle.load(open(SP / f"nlz_{name}.pkl", "rb"))
    return d


def main():
    import cellformer_af as CF
    d = CF.load_all()
    genes, kos, gi = d["genes"], d["kos"], d["gi"]
    spec, tide, nspec = d["spec"], d["tide"], d["nspec"]
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [q for q in range(len(kos)) if not tr[q] and nspec[q] >= MIN_SPEC and kos[q] in gi]
    freq = spec[tr].mean(0)
    n = len(genes)

    hdr("1. NODES ARE COMPLETE. PAIRS ARE NOT. These are different claims.")
    adj = collections.defaultdict(set)
    chan = collections.defaultdict(lambda: collections.defaultdict(set))
    CH = ["reaction", "complex", "PPI", "coexpr", "signed-reg"]
    nedge = 0
    for (i, j), v in d["pair"].items():
        nedge += 1
        for c in range(5):
            if v[c]:
                chan[CH[c]][i].add(j)
                chan[CH[c]][j].add(i)
        adj[i].add(j)
        adj[j].add(i)
    poss = n * (n - 1) // 2
    print(f"  genes in the readout                {n:,}")
    print(f"  genes with >=1 network edge         {len(adj):,} ({100*len(adj)/n:.1f}%)   <- NODE coverage")
    print(f"  possible gene pairs                 {poss:,}")
    print(f"  pairs with an edge                  {nedge:,} ({100*nedge/poss:.3f}%)      <- PAIR coverage")
    print("\n  Having a row for every gene is node coverage. Prediction needs the specific pair")
    print("  (knocked-out gene, gene whose mRNA moved) to be one of the 0.7% that carries an edge.")
    R["node_vs_pair"] = {"genes": n, "genes_with_edge": len(adj), "possible_pairs": poss, "edges": nedge}

    # ---------------------------------------------------------------- GO / pathway co-membership
    C = json.load(gzip.open(Path(__file__).resolve().parent / "data" / "cell_complete.json.gz", "rt"))
    names = [g["name"] for g in C["genes"]]
    go = C.get("go") or {}
    term_members = collections.defaultdict(set)
    for g, dd in go.items():
        if isinstance(dd, dict) and g in gi:
            for br in ("P", "F", "C"):
                for t in (dd.get(br) or []):
                    term_members[(br, t)].add(gi[g])
    for idx in (C.get("pathways") or {}).values():
        if isinstance(idx, list):
            s = {gi[names[i]] for i in idx if 0 <= i < len(names) and names[i] in gi}
            if 2 <= len(s) <= 2000:
                term_members[("PW", len(term_members))] = s
    gterm = collections.defaultdict(set)
    for t, mem in term_members.items():
        if 2 <= len(mem) <= 2000:
            for g in mem:
                gterm[g].add(t)
    print(f"\n  GO/pathway groups usable (2..2000 members): {sum(1 for m in term_members.values() if 2<=len(m)<=2000):,}")

    def go_partners(k):
        out = set()
        for t in gterm.get(k, ()):
            out |= term_members[t]
        out.discard(k)
        return out

    # ---------------------------------------------------------------- same-modality: other cell lines
    hdr("2. SAME MODALITY -- did this gene also move in a DIFFERENT cell line?")
    lines = {}
    for nm in ("RPE1", "HepG2", "Jurkat", "HCT116", "Melanoma"):
        p = SP / f"nlz_{nm}.pkl"
        if p.exists():
            lines[nm] = load_line(nm)
    print(f"  loaded {len(lines)} other cell lines: {', '.join(lines)}")
    # movers of gene k in line L, mapped into the K562 gene index
    other_movers = collections.defaultdict(set)
    ko_seen = collections.Counter()
    for nm, dd in lines.items():
        for k, prof in dd.items():
            if k not in gi:
                continue
            ko_seen[k] += 1
            for g, z in prof.items():
                if abs(z) >= TAU and g in gi and not tide[gi[g]]:
                    other_movers[k].add(gi[g])
    have_other = [q for q in test if kos[q] in other_movers]
    print(f"  held-out perturbations whose KO gene was also screened elsewhere: "
          f"{len(have_other):,}/{len(test):,} ({100*len(have_other)/len(test):.1f}%)")
    print(f"  mean movers recorded for those genes in other lines: "
          f"{np.mean([len(other_movers[kos[q]]) for q in have_other]):.0f}")

    # ---------------------------------------------------------------- coverage head-to-head
    hdr("3. COVERAGE OF THE SAME (knockout -> mover) PAIRS, on the SAME perturbations")
    sources = [("PPI", lambda k: chan["PPI"].get(k, set())),
               ("co-expression", lambda k: chan["coexpr"].get(k, set())),
               ("complex", lambda k: chan["complex"].get(k, set())),
               ("signed regulation", lambda k: chan["signed-reg"].get(k, set())),
               ("reaction", lambda k: chan["reaction"].get(k, set())),
               ("ALL curated edges", lambda k: adj.get(k, set())),
               ("GO/pathway co-membership", go_partners),
               ("+ GO/pathway, union with edges", lambda k: adj.get(k, set()) | go_partners(k))]

    sub = have_other                      # score every source on the SAME perturbations, for fairness
    print(f"  scored on the {len(sub):,} held-out perturbations that ALL sources can address\n")
    print(f"  {'source':34s} {'set size':>9s} {'coverage':>10s} {'precision':>10s}")
    rows = {}
    for tag, fn in sources:
        cov, prec, sz = [], [], []
        for q in sub:
            k = gi[kos[q]]
            T = {int(x) for x in np.where(spec[q])[0] if not tide[x]}
            if not T:
                continue
            P = {x for x in fn(k) if not tide[x]}
            sz.append(len(P))
            cov.append(len(P & T) / len(T))
            prec.append(len(P & T) / max(len(P), 1))
        rows[tag] = (float(np.mean(cov)), float(np.mean(prec)), float(np.mean(sz)))
        print(f"  {tag:34s} {np.mean(sz):>9.0f} {np.mean(cov):>10.4f} {np.mean(prec):>10.4f}")

    cov, prec, sz = [], [], []
    for q in sub:
        T = {int(x) for x in np.where(spec[q])[0] if not tide[x]}
        if not T:
            continue
        P = other_movers[kos[q]]
        sz.append(len(P))
        cov.append(len(P & T) / len(T))
        prec.append(len(P & T) / max(len(P), 1))
    rows["SAME GENE, OTHER CELL LINE"] = (float(np.mean(cov)), float(np.mean(prec)), float(np.mean(sz)))
    print(f"  {'-'*34} {'-'*9} {'-'*10} {'-'*10}")
    print(f"  {'SAME GENE, OTHER CELL LINE':34s} {np.mean(sz):>9.0f} {np.mean(cov):>10.4f} "
          f"{np.mean(prec):>10.4f}   <- same modality, different experiment")

    # size-matched control: the cross-line set is large, so give the curated union the same budget
    best_cur = max(rows[t][0] for t in rows if t != "SAME GENE, OTHER CELL LINE")
    print(f"\n  best curated coverage {best_cur:.4f}   vs same-modality {rows['SAME GENE, OTHER CELL LINE'][0]:.4f}"
          f"   ({rows['SAME GENE, OTHER CELL LINE'][0]/max(best_cur,1e-9):.1f}x)")
    R["coverage"] = rows

    # is the cross-line advantage just its larger set? match on set size by random truncation.
    hdr("4. IS THE SAME-MODALITY ADVANTAGE JUST A BIGGER SET?")
    tgt = int(np.mean([len(adj.get(gi[kos[q]], set())) for q in sub]))
    cov_m = []
    for q in sub:
        T = {int(x) for x in np.where(spec[q])[0] if not tide[x]}
        if not T:
            continue
        P = list(other_movers[kos[q]])
        if len(P) > tgt:
            P = [int(x) for x in RNG.choice(P, size=tgt, replace=False)]
        cov_m.append(len(set(P) & T) / len(T))
    print(f"  curated union set size {tgt}; cross-line truncated to the SAME size at random:")
    print(f"    cross-line coverage, size-matched: {np.mean(cov_m):.4f}   vs curated {rows['ALL curated edges'][0]:.4f}"
          f"   ({np.mean(cov_m)/max(rows['ALL curated edges'][0],1e-9):.1f}x)")
    R["size_matched"] = {"target_size": tgt, "crossline_cov": float(np.mean(cov_m)),
                         "curated_cov": rows["ALL curated edges"][0]}


    # ---------------------------------------------------------------- 5. does it actually PREDICT?
    hdr("5. THE PREDICTOR -- cross-cell-line co-response vs the frequency baseline")
    # The cross-line set has precision 0.1564 at a set size of 73. The specification derived in
    # network_requirement.py was: to match frequency at recall@50 you need precision >= 0.1051 inside the
    # reported 50. This is the first relation measured in this project that CLEARS that bar, so it gets
    # scored as a predictor rather than described.
    NPICK = 50
    fr, xr, mx = [], [], []
    for q in sub:
        T = {int(x) for x in np.where(spec[q])[0] if not tide[x]}
        if not T:
            continue
        f = freq.copy().astype(float); f[tide] = -1e9
        fr.append(len({int(i) for i in np.argsort(-f)[:NPICK]} & T) / len(T))
        sx = np.zeros(n); 
        for j in other_movers[kos[q]]:
            sx[j] = 1.0
        sx = sx + 1e-6 * freq          # frequency only breaks ties WITHIN the cross-line set
        sx[tide] = -1e9
        xr.append(len({int(i) for i in np.argsort(-sx)[:NPICK]} & T) / len(T))
        sm = sx + 0.5 * (freq / max(freq.max(), 1e-9))
        sm[tide] = -1e9
        mx.append(len({int(i) for i in np.argsort(-sm)[:NPICK]} & T) / len(T))
    fr, xr, mx = np.array(fr), np.array(xr), np.array(mx)

    def bci(x, nb=2000, seed=0):
        r = np.random.default_rng(seed)
        return tuple(float(v) for v in np.percentile(
            r.choice(x, size=(nb, len(x)), replace=True).mean(1), [2.5, 97.5]))

    for tag, v in (("frequency baseline", fr), ("cross-cell-line only", xr),
                   ("cross-line + frequency", mx)):
        lo, hi = bci(v)
        print(f"    {tag:28s} recall@50 {v.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
    for tag, v in (("cross-line only", xr), ("cross-line + frequency", mx)):
        dd = v - fr
        lo, hi = bci(dd)
        print(f"    {tag:28s} MINUS frequency: {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"  {'*** BEATS FREQUENCY ***' if lo > 0 else ('WORSE (CI excludes 0)' if hi < 0 else 'not distinguishable')}")
    R["predictor"] = {"frequency": float(fr.mean()), "crossline": float(xr.mean()),
                      "crossline_plus_freq": float(mx.mean()), "n": int(len(fr))}

    # THE CONFOUND: genes that move everywhere move in every line too, so a cross-line hit could be the
    # frequency prior wearing a different hat. Restrict to movers the frequency baseline MISSES.
    hdr("6. IS IT JUST THE FREQUENCY PRIOR? -- scored only on movers frequency does NOT name")
    ftop = {int(i) for i in np.argsort(-np.where(tide, -1e9, freq))[:NPICK]}
    hard_x, hard_n = [], 0
    for q in sub:
        T = {int(x) for x in np.where(spec[q])[0] if not tide[x]} - ftop
        if len(T) < 2:
            continue
        hard_n += 1
        P = other_movers[kos[q]]
        hard_x.append(len(P & T) / len(T))
    hard_x = np.array(hard_x)
    lo, hi = bci(hard_x)
    print(f"    on the {hard_n:,} perturbations with >=2 movers OUTSIDE frequency's top-50:")
    print(f"    cross-line coverage of those HARD movers: {hard_x.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
    print("    (frequency's coverage of them is 0 by construction -- they are what it misses)")
    R["hard_movers"] = {"n": hard_n, "crossline_coverage": float(hard_x.mean()), "ci": [lo, hi]}


    # ---------------------------------------------------------------- 7. combine, tuned on TRAIN only
    hdr("7. COMBINED, with the weight tuned on TRAINING perturbations only")
    # Section 6 showed the two sources are complementary: cross-line reaches 15.7% of the movers frequency
    # cannot name. A naive sum did not exploit that -- every cross-line gene carries the same score, so the
    # weight decides whether cross-line simply overwrites frequency's ranking. Sweeping the weight on the
    # HELD-OUT set would be fitting to the test set; it is tuned on training perturbations and applied once.
    fmax = max(float(freq.max()), 1e-9)
    fnorm = (freq / fmax).astype(float)
    train_q = [q for q in range(len(kos))
               if tr[q] and nspec[q] >= MIN_SPEC and kos[q] in gi and kos[q] in other_movers]
    print(f"    tuning on {len(train_q):,} TRAINING perturbations, evaluating on {len(sub):,} held-out")

    def score_set(qs, w):
        out = []
        for q in qs:
            T = {int(x) for x in np.where(spec[q])[0] if not tide[x]}
            if not T:
                continue
            sx = np.zeros(n)
            for j in other_movers[kos[q]]:
                sx[j] = 1.0
            sc = fnorm + w * sx
            sc[tide] = -1e9
            out.append(len({int(i) for i in np.argsort(-sc)[:NPICK]} & T) / len(T))
        return np.array(out)

    grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0]
    tr_scores = [(w, float(score_set(train_q, w).mean())) for w in grid]
    for w, v in tr_scores:
        print(f"      w={w:<5g} train recall {v:.4f}")
    best_w = max(tr_scores, key=lambda t: t[1])[0]
    print(f"    chosen on TRAIN: w={best_w:g}")

    comb = score_set(sub, best_w)
    lo, hi = bci(comb)
    dd = comb - fr[:len(comb)] if len(comb) == len(fr) else None
    print(f"\n    frequency baseline      recall@50 {fr.mean():.4f}")
    print(f"    combined (w={best_w:g})        recall@50 {comb.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
    if dd is not None:
        dlo, dhi = bci(dd)
        verdict = "*** BEATS FREQUENCY ***" if dlo > 0 else ("worse" if dhi < 0 else "not distinguishable")
        print(f"    combined MINUS frequency: {dd.mean():+.4f} [{dlo:+.4f},{dhi:+.4f}]  {verdict}")
        R["combined"] = {"w": float(best_w), "recall": float(comb.mean()),
                         "delta": float(dd.mean()), "ci": [dlo, dhi]}
    # ROBUSTNESS, reported as a check and NOT as the headline: the headline is the single train-chosen w.
    # If the gain only existed at one weight it would be a lucky pick; a broad plateau means it is not.
    print("\n    robustness (held-out at every weight -- the reported result is the train-chosen w only):")
    for w in grid:
        v = score_set(sub, w)
        d2 = v - fr[:len(v)]
        l2, h2 = bci(d2)
        mark = "  <- chosen on train" if w == best_w else ""
        print(f"      w={w:<5g} held-out {v.mean():.4f}  vs freq {d2.mean():+.4f} [{l2:+.4f},{h2:+.4f}]{mark}")
    R["robustness"] = [{"w": float(w), "recall": float(score_set(sub, w).mean())} for w in grid]

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "coverage_gap.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'coverage_gap.json'}")


if __name__ == "__main__":
    main()
