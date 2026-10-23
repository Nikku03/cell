"""BUILD 4 of 5 -- AN INTERVENTIONAL REGULON, AND WHETHER IT BEATS THE BINDING LAYER IT REPLACES.

THE PROBLEM IT ADDRESSES. DESIGN.md sec 5 calls the regulation layer the real blocker: 91% of it is ChIP BINDING,
which was measured against actual regulation in this same cell line and overlapped AT CHANCE (fold 0.96, p = 0.68).
The direct-regulon layer contributed 1 hit in 39 in the sensor model. Binding is not regulation, and no amount of
modelling fixes data that is wrong for the purpose.

THE PROPOSED REPLACEMENT. The perturbation screen IS an interventional regulatory map. Knock out a gene, and
whatever moves is its functional regulon -- causal by construction, unlike ChIP. 5,120 knockouts is 5,120 regulons
for free.

THE CATCH, WHICH IS WHY BUILD 3 HAD TO COME FIRST. Roughly 40% of what moves after any knockout is the shared
stress core, which is not that gene's regulon -- it is the cell being hurt. So the map must be built on the
SPECIFIC RESIDUAL, and build 3 exists to produce that residual and to establish whether it carries mechanism at
all. This file consumes build 3's residual matrix and reports build 3's verdict alongside its own, because a
regulon built on a structureless residual is a regulon of noise however well it is engineered.

THE TEST, AND WHY IT IS NOT CIRCULAR. A gene's own regulon comes from its own knockout, so "predict a knockout
from its regulon" would be reading the answer. The non-circular use is TRANSFER: predict a HELD-OUT perturbation's
response from the interventional regulons of its network NEIGHBOURS -- complex partners, PPI partners,
co-expression partners -- none of which is the gene itself, and all of whose regulons come from the DONOR half of
the perturbations only. The donor/test split is the same one build 3 used to fit the core, so the core removal and
the donor pool never see a test perturbation.

    predicted response of g  =  aggregate of { residual profile of d : d a neighbour of g, d in DONOR half }

FIVE ARMS, ALL SCORED IDENTICALLY, ON THE SAME HELD-OUT PERTURBATIONS

    residual + neighbours    the thing being built
    raw + neighbours         same transfer, shared core NOT removed -- isolates what the decomposition buys
    ChIP targets             the layer sec 5 wants to replace; the bar it must clear
    frequency                the globally most-moved genes, needing no biology at all -- the real bar
    random                   drawn from the measured readout, to price the answer space

PRE-REGISTERED CRITERION (DESIGN.md sec 5): the interventional regulon must beat the ChIP layer -- a low bar,
since ChIP is at chance -- AND beat the shared core alone, which is not a low bar, because the frequency baseline
has beaten seven mechanistic methods in this project already.
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TAU, TIDE_FRAC, MIN_SPEC, NPICK = 1.0, 0.05, 5, 50


def load():
    z = np.load(SP / "residual_K562_gwps.npz", allow_pickle=True)
    kos, genes = [str(x) for x in z["kos"]], [str(x) for x in z["genes"]]
    R, train = np.asarray(z["R"], np.float32), np.asarray(z["train"], bool)
    y = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    M = np.asarray(y["M"], np.float32)
    assert [str(x) for x in y["kos"]] == kos and [str(x) for x in y["genes"]] == genes, \
        "residual and readout are not aligned -- refusing to continue"
    assert np.isfinite(R).all() and np.isfinite(M).all(), "NON-FINITE matrices"
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    spec = [set(np.where((A[i] >= TAU) & ~tide)[0].tolist()) for i in range(len(kos))]
    print(f"loaded: {len(kos):,} perturbations x {len(genes):,} genes; "
          f"donor half {int(train.sum()):,}, test half {int((~train).sum()):,}")
    return kos, genes, M, R, train, tide, spec


def neighbours(kos, names_all):
    """gene -> set of other PERTURBED genes it is functionally tied to. Every source's join size is printed;
    the PPI and co-expression layers are INDEX-keyed in cell_complete, which is where the bug that silently
    dropped 191,447 edges lived, so indices are resolved explicitly rather than assumed to be names."""
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    ki = {k: i for i, k in enumerate(kos)}
    src = {}

    cx = collections.defaultdict(set)
    for cname, mem in (D.get("complexes") or {}).items():
        ms = [names[i] for i in mem if isinstance(i, int) and 0 <= i < N]
        ms = [g for g in ms if g in ki]
        if 2 <= len(ms) <= 30:
            for a in ms:
                cx[a] |= set(ms) - {a}
    src["complex"] = cx
    print(f"  complex partners: {len(cx):,} perturbed genes have at least one")

    pp = collections.defaultdict(set)
    nedge = 0
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= a < N and 0 <= b < N) or a == b:
            continue
        nedge += 1
        na, nb = names[a], names[b]
        if na in ki and nb in ki:
            pp[na].add(nb)
            pp[nb].add(na)
    src["ppi"] = pp
    print(f"  PPI: {nedge:,} edges resolved, {len(pp):,} perturbed genes have a perturbed partner")
    assert nedge > 1000, "PPI JOIN EMPTY -- index space disagrees, refusing to continue"

    ce = collections.defaultdict(set)
    for k, v in (D.get("coexpr") or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if not (0 <= i < N) or names[i] not in ki:
            continue
        # values are [neighbour_index, correlation] PAIRS, not bare indices -- reading them as ints silently
        # produced an empty co-expression layer on the first run
        for j in (v if isinstance(v, (list, tuple)) else []):
            t = j[0] if isinstance(j, (list, tuple)) and j else j
            if isinstance(t, int) and 0 <= t < N and names[t] in ki and t != i:
                ce[names[i]].add(names[t])
    src["coexpr"] = ce
    print(f"  co-expression: {len(ce):,} perturbed genes have a perturbed partner")
    assert len(ce) > 100, "CO-EXPRESSION JOIN EMPTY -- value shape disagrees, refusing to continue"

    chip = collections.defaultdict(set)
    for e in D.get("reg", []):
        try:
            s, t = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s < N and 0 <= t < N and s != t:
            chip[names[s]].add(names[t])
    print(f"  ChIP/regulation out-edges: {len(chip):,} source genes")
    union = collections.defaultdict(set)
    for d in (cx, pp, ce):
        for a, b in d.items():
            union[a] |= b
    src["any"] = union
    print(f"  union of the three functional sources: {len(union):,} perturbed genes have a donor")
    return src, chip


def transfer(X, donors_idx, k=NPICK):
    """Aggregate donors' profiles into a ranked prediction. Mean |z| across donors: a gene that moves for several
    of a knockout's partners is a better guess than one that moves hugely for a single partner."""
    if not donors_idx:
        return []
    s = np.abs(X[donors_idx]).mean(0)
    return np.argsort(-s)[:k].tolist()


def score(pred, truth):
    if not truth:
        return None
    hit = len(set(pred) & truth)
    return hit / len(truth), hit / max(len(pred), 1)


def main():
    kos, genes, M, R, train, tide, spec = load()
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes)}
    src, chip = neighbours(kos, genes)

    try:
        d3 = json.load(open(OUT / "decompose.json"))
        print(f"\nbuild 3 verdict on whether the residual carries mechanism at all: {d3.get('verdict')} "
              f"(complex-partner coherence z = {d3.get('coherence', {}).get('RESIDUAL (core removed)', {}).get('perm_z')})")
    except FileNotFoundError:
        print("\nbuild 3 output missing -- run decompose.py first")
        return

    test = [i for i in range(len(kos)) if not train[i] and len(spec[i]) >= MIN_SPEC]
    print(f"scoring on {len(test):,} held-out perturbations with >= {MIN_SPEC} specific movers")
    if len(test) < 100:
        print("  TOO FEW held-out scorable perturbations -- refusing to report")
        return

    freq = (np.abs(M[train]) >= TAU).mean(0)
    freq_pick = [i for i in np.argsort(-freq)[:NPICK * 3] if not tide[i]][:NPICK]
    rng = np.random.default_rng(0)
    nontide = np.where(~tide)[0]

    ARMS = ["residual+neighbours", "raw+neighbours", "ChIP targets", "frequency", "random"]
    acc = {a: {"rec": [], "prec": [], "n_empty": 0} for a in ARMS}
    per_source = {s: [] for s in ("complex", "ppi", "coexpr", "any")}

    for i in test:
        g = kos[i]
        truth = spec[i]
        don = [ki[d] for d in src["any"].get(g, ()) if d in ki and train[ki[d]]]
        for arm, pick in (
            ("residual+neighbours", transfer(R, don)),
            ("raw+neighbours", transfer(M, don)),
            ("ChIP targets", [gi[t] for t in sorted(chip.get(g, ()), key=lambda t: -freq[gi[t]] if t in gi else 0)
                              if t in gi and not tide[gi[t]]][:NPICK]),
            ("frequency", freq_pick),
            ("random", rng.choice(nontide, size=NPICK, replace=False).tolist()),
        ):
            if not pick:
                acc[arm]["n_empty"] += 1
                acc[arm]["rec"].append(0.0)
                acc[arm]["prec"].append(0.0)
                continue
            r, p = score(pick, truth)
            acc[arm]["rec"].append(r)
            acc[arm]["prec"].append(p)
        for s in ("complex", "ppi", "coexpr", "any"):
            d = [ki[x] for x in src[s].get(g, ()) if x in ki and train[ki[x]]]
            pk = transfer(R, d)
            per_source[s].append(score(pk, truth)[0] if pk else 0.0)

    print("\n" + "=" * 84)
    print(f"{'arm':26s} {'recall':>8s} {'precision':>10s} {'empty':>7s}   vs frequency (paired bootstrap)")
    base = np.array(acc["frequency"]["rec"])
    out = {}
    for a in ARMS:
        r = np.array(acc[a]["rec"])
        d = r - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        tail = "" if a == "frequency" else f"   {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]"
        print(f"{a:26s} {r.mean():>8.4f} {np.mean(acc[a]['prec']):>10.4f} "
              f"{acc[a]['n_empty']:>7,}{tail}")
        out[a] = {"recall": float(r.mean()), "precision": float(np.mean(acc[a]["prec"])),
                  "n_empty": acc[a]["n_empty"],
                  "delta_vs_frequency": float(d.mean()), "ci": [float(lo), float(hi)]}
    print("=" * 84)
    print("\nresidual transfer by neighbour source (recall):")
    for s, v in per_source.items():
        print(f"  {s:<10s} {np.mean(v):.4f}   (n with a donor: {sum(1 for x in v if x > 0):,}/{len(v):,})")

    res, ch, fr = out["residual+neighbours"], out["ChIP targets"], out["frequency"]
    beat_chip = res["recall"] > ch["recall"]
    beat_freq = res["ci"][0] > 0
    print(f"\nPRE-REGISTERED CRITERION (DESIGN.md sec 5)")
    print(f"  beats the ChIP layer it replaces : {beat_chip}  ({res['recall']:.4f} vs {ch['recall']:.4f})")
    print(f"  beats the shared-core/frequency  : {beat_freq}  ({res['recall']:.4f} vs {fr['recall']:.4f}, "
          f"CI [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}])")
    verdict = "PASS" if (beat_chip and beat_freq) else "FAIL"
    print(f"  VERDICT: {verdict}")

    json.dump({"n_test": len(test), "npick": NPICK, "arms": out,
               "by_source": {k: float(np.mean(v)) for k, v in per_source.items()},
               "beats_chip": bool(beat_chip), "beats_frequency": bool(beat_freq), "verdict": verdict},
              open(OUT / "regulon.json", "w"), indent=1)
    print(f"\n  -> {OUT/'regulon.json'}")


if __name__ == "__main__":
    main()
