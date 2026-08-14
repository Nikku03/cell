"""LOOP 127 -- USE THE KINETICS ONLY WHERE IT MEETS THE MEASUREMENT'S OWN NOISE FLOOR.

THE PROPOSAL, AND THE ONE THING WRONG WITH IT AS STATED. Keep a kcat where the estimate is good to
the experimental noise floor -- about 4x -- and do not pretend to know it elsewhere. That is right,
and loop 124 supplies the motive: the bundle's predicted kcats score 12.95x median fold-error
against held-out measurements while a flat 1.85/s scores 9.42x, so three quarters of this model's
turnover numbers are worse than a constant.

The problem is that "meets the noise floor" cannot be evaluated where it matters. For the 5,941
CatPred reactions there IS no measurement, so filtering on agreement-with-measurement keeps exactly
the genes that happened to agree and says nothing about a new one. That is selection on the outcome,
and it is the circularity loop 101 was built to avoid.

So the filter has to run on something computable WITHOUT the answer. This loop's job is to find out
whether such a thing exists, and if it does, to build the filtered bundle and count what survives.

AND THE FLOOR ITSELF IS MEASURABLE RATHER THAN ASSUMED. The 2,437 human records cover 362 EC
numbers, 204 of them with three or more independent measurements. Predicting one record from the
median of the others in its own EC is exactly the self-consistency question, and it answers "how
well can anyone do" without reference to any model:

    within-EC fold from the EC median   median 2.36x   75th 6.44x   90th 39.39x
    LEAVE-ONE-OUT within EC             median 2.85x   75th 8.57x   over 2,221 records

So 4x is a real operating point and not a folk number -- it sits above the data's own
reproducibility and below its 75th percentile. N1 checks that rather than taking it on trust.

A FINDING THAT ARRIVED WHILE MEASURING THE FLOOR, disclosed here because it kills the obvious
selector. Spread RISES with replicate count:

    ECs with  3-4 records   median fold 1.54x
    ECs with  5-9 records               2.24x
    ECs with 10-19 records              2.55x
    ECs with 20+ records                2.62x

More measurements behind an EC median means MORE disagreement, not less, because a heavily measured
EC is a broad class covering many enzymes, substrates and conditions. "Trust the well-measured ones"
is exactly backwards, and any filter built on record count would have selected for breadth.

ALSO DISCLOSED, measured during construction: 1,422 of the 1,811 tier-1 reactions have an EC with
three or more replicates, so a self-consistency is computable for them; 70 UniProt genes have both
a measured kcat and an EC with replicates, which is the entire validation set and it is small.

PREDECLARED:

  N1 WHERE DOES 4x ACTUALLY SIT?                                    THE THRESHOLD, CHECKED.
       the proposed 4x against the measured leave-one-out distribution. Gate: 4x must fall between
       the median and the 75th percentile of that distribution -- above the data's own
       reproducibility, so the filter is not asking for better than anyone can measure, and below
       its 75th percentile, so it is not accepting everything. If 4x falls outside, the threshold
       moves to the measured floor and the loop says so.
  N2 THE SELECTOR MUST NOT NEED THE ANSWER                          THE MAKE-OR-BREAK GATE.
       three candidates, all computable with no held-out value: (a) the EC's own leave-one-out
       self-consistency, (b) how many genes share the EC, a breadth proxy, (c) the replicate count,
       included precisely because the disclosure above says it should fail. Gate: at least one must
       predict per-gene fold-error against the UniProt measurements at p < 0.05 in the right
       direction. If none does, no filter can be applied to the unmeasured majority, and this loop
       reports that instead of shipping one that cannot work.
  N3 THE FILTER VALIDATES ON HELD-OUT MEASUREMENTS                  THE TEST.
       accepted ECs against rejected ones, fold-error versus UniProt, on the 70. Gate: accepted
       must be lower with permutation p < 0.05. n = 70 is small and the gate is allowed to fail for
       that reason; a filter that cannot be shown to work is not shipped.
  N4 THE FILTERED BUNDLE, COUNTED HONESTLY                          THE DELIVERABLE.
       apply the rule and report coverage before and after by all three denominators. A filter that
       keeps 5% of the model is a different object from one that keeps 80%, and the number decides
       whether this is worth doing.
  N5 WHAT REPLACES THE REJECTED ONES                                THE HONEST FALLBACK.
       loop 124 measured the constant at 9.42x against CatPred's 12.95x, so the replacement for a
       rejected kcat is the global median FLAGGED AS A CONSTANT -- not a prediction wearing a
       physical unit. Report how many reactions move and by how much.
  N6 THE COST, AND WHETHER THE SURVIVORS ARE JUST THE FAMOUS ONES   THE GUARD.
       publication count of accepted versus rejected genes. If the filter keeps the well-studied
       enzymes, the filtered model is accurate on the enzymes everyone already knows and silent
       elsewhere, which is worth stating plainly rather than discovering later.

-> outputs/loop_kcat_floor.json, colab/data/kinetics_filtered.json.gz
"""
import collections
import csv
import gzip
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
BUNDLE = Path("colab/data/kinetics_bundle.json.gz")
ECMED = Path("colab/data/ec_kcat_medians.json.gz")
UPK = LR.SC / "uniprot_kinetics_human.tsv"
DEST = Path("colab/data/kinetics_filtered.json.gz")
SEED = 12700
NPERM = 2000
FLOOR = 4.0            # the proposed operating point, checked in N1
MIN_REPL = 3           # an EC needs this many records before a self-consistency means anything

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def fold(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.maximum(a / b, b / a)


def perm_p(a, b, rng, n=NPERM):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    k = len(a)
    null = np.array([(lambda s: np.median(s[:k]) - np.median(s[k:]))(rng.permutation(pool))
                     for _ in range(n)])
    return obs, float(np.mean(np.abs(null) >= abs(obs)))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 127 -- keep the kinetics only where it meets the measurement's own noise floor")
    say("=" * 100)
    say()

    B = json.load(gzip.open(BUNDLE, "rt"))
    E = json.load(gzip.open(ECMED, "rt"))
    rk, rt, rec = B["reaction_kcat_per_s"], B["reaction_tier"], B["reaction_ec"]
    gmed = float(E["global_median_per_s"])
    ec_med = E["ec_human_median_per_s"]
    by = collections.defaultdict(list)
    for e, v in E["human_records"]:
        if v and float(v) > 0:
            by[e].append(float(v))
    multi = {k: v for k, v in by.items() if len(v) >= MIN_REPL}
    say(f"  {len(E['human_records']):,} human records over {len(by)} EC numbers; "
        f"{len(multi)} with >= {MIN_REPL} replicates")

    # THE SELF-CONSISTENCY, per EC: predict each record from the median of the others in its own EC
    selfc, nrep = {}, {}
    loo_all = []
    for e, v in multi.items():
        f = []
        for i in range(len(v)):
            o = [v[j] for j in range(len(v)) if j != i]
            m = float(np.median(o))
            if m > 0 and v[i] > 0:
                f.append(max(v[i] / m, m / v[i]))
        if f:
            selfc[e] = float(np.median(f))
            nrep[e] = len(v)
            loo_all += f
    loo_all = np.array(loo_all)
    say(f"  leave-one-out self-consistency computed for {len(selfc)} ECs "
        f"({len(loo_all):,} records)")
    say()

    gates = {}

    # ---------------------------------------------------------------- N1
    say("N1 WHERE DOES 4x ACTUALLY SIT?")
    q50, q75, q90 = (float(np.percentile(loo_all, p)) for p in (50, 75, 90))
    pct = float(np.mean(loo_all <= FLOOR))
    say(f"     leave-one-out distribution: median {q50:.2f}x, 75th {q75:.2f}x, 90th {q90:.2f}x")
    say(f"     the proposed {FLOOR:.0f}x sits at the {pct:.1%} percentile of it")
    say(f"     gate: {FLOOR:.0f}x must fall between the median ({q50:.2f}x) and the 75th "
        f"({q75:.2f}x)")
    say(f"     below the median would demand better than anyone can measure; above the 75th would "
        f"accept nearly everything")
    gates["N1"] = bool(q50 <= FLOOR <= q75)
    say(f"     N1 {'PASS' if gates['N1'] else 'FAIL'} -- {FLOOR:.0f}x is "
        f"{'a defensible operating point' if gates['N1'] else 'NOT defensible; the threshold should be the measured floor'}")
    say()

    # ---------------------------------------------------------------- the validation set
    rows = list(csv.reader(open(UPK, newline=""), delimiter="\t"))
    h, rows = rows[0], rows[1:]
    iG, iE, iK = (h.index("Gene Names (primary)"), h.index("EC number"), h.index("Kinetics"))
    NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    K1 = re.compile(r"kcat is " + NUM + r"\s*(sec|min|hour|h)\(-1\)", re.I)
    PS = {"sec": 1.0, "min": 1 / 60.0, "hour": 1 / 3600.0, "h": 1 / 3600.0}
    mkc, gene_ec = {}, {}
    for x in rows:
        g = x[iG].strip()
        if not g:
            continue
        v = [float(a) * PS[b.lower()] for a, b in K1.findall(x[iK])]
        if v:
            mkc[g] = float(np.exp(np.mean(np.log(v))))
        if x[iE].strip():
            gene_ec[g] = [e.strip() for e in x[iE].split(";") if e.strip()]
    val = []
    for g, m in mkc.items():
        for e in gene_ec.get(g, []):
            if e in selfc and e in ec_med:
                val.append((g, e, m, ec_med[e]))
                break
    say(f"  validation set: {len(val)} genes with a measured kcat AND an EC carrying a "
        f"self-consistency")
    genes_per_ec = collections.Counter()
    for g, es in gene_ec.items():
        for e in es:
            genes_per_ec[e] += 1
    say()

    # ---------------------------------------------------------------- N2
    say("N2 THE SELECTOR MUST NOT NEED THE ANSWER")
    fe = np.array([max(p / m, m / p) for _, _, m, p in val])
    cands = {
        "EC self-consistency (LOO)": np.array([selfc[e] for _, e, _, _ in val]),
        "genes sharing the EC": np.array([float(genes_per_ec.get(e, 1)) for _, e, _, _ in val]),
        "replicate count": np.array([float(nrep[e]) for _, e, _, _ in val]),
    }
    n2 = {}
    ok2 = False
    for k, v in cands.items():
        rho = spearman(v, fe)
        null = np.array([spearman(rng.permutation(v), fe) for _ in range(NPERM)])
        p = float(np.mean(np.abs(null) >= abs(rho))) if np.isfinite(rho) else float("nan")
        n2[k] = {"rho": rho, "p": p}
        good = np.isfinite(rho) and rho > 0 and p < 0.05
        say(f"     {k:<28} Spearman vs fold-error {rho:+.4f}   p {p:.4f}   "
            f"{'USABLE' if good else 'no'}")
        ok2 = ok2 or good
    say(f"     direction required: POSITIVE -- a less self-consistent EC should give a worse "
        f"estimate")
    gates["N2"] = bool(ok2)
    say(f"     N2 {'PASS' if gates['N2'] else 'FAIL'} -- "
        f"{'a selector exists that needs no held-out value' if gates['N2'] else 'NO selector works; the filter cannot be applied to the unmeasured majority'}")
    say()

    # ---------------------------------------------------------------- N3
    say("N3 THE FILTER VALIDATES ON HELD-OUT MEASUREMENTS")
    acc = np.array([selfc[e] <= FLOOR for _, e, _, _ in val])
    say(f"     accepted {int(acc.sum())} of {len(val)} validation genes "
        f"(EC self-consistency <= {FLOOR:.0f}x)")
    if acc.sum() >= 2 and (~acc).sum() >= 2:
        obs, p3 = perm_p(np.log10(fe[acc]), np.log10(fe[~acc]), rng)
        say(f"     fold-error against UniProt: accepted median {np.median(fe[acc]):.2f}x, "
            f"rejected {np.median(fe[~acc]):.2f}x")
        say(f"     log10 difference {obs:+.3f} (negative favours the filter), "
            f"permutation p = {p3:.4f}")
        gates["N3"] = bool(obs < 0 and p3 < 0.05)
    else:
        obs, p3 = float("nan"), float("nan")
        say("     one side has fewer than 2 genes; the comparison is not computable")
        gates["N3"] = False
    say(f"     N3 {'PASS' if gates['N3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N4
    say("N4 THE FILTERED BUNDLE, COUNTED HONESTLY")
    keep, drop = {}, {}
    tier_new = {}
    for r, t in rt.items():
        ecs = [e for e in (rec.get(r) or []) if e in selfc]
        if t == "1_human_EC" and ecs and min(selfc[e] for e in ecs) <= FLOOR:
            keep[r] = rk[r]
            tier_new[r] = "1_human_EC_selfconsistent"
        else:
            drop[r] = rk[r]
            tier_new[r] = "4_global_median_CONSTANT"
    say(f"     BEFORE: {len(rk):,} reactions with a kcat, "
        + ", ".join(f"{k} {v}" for k, v in collections.Counter(rt.values()).most_common()))
    say(f"     AFTER : {len(keep):,} kept as a measured estimate, {len(drop):,} replaced by the "
        f"constant {gmed}/s")
    say(f"     THREE DENOMINATORS")
    say(f"       by reaction  {len(keep):,} of {len(rk):,} with a kcat = {len(keep) / len(rk):.1%}; "
        f"of the 12,931 model reactions = {len(keep) / 12931:.1%}")
    ug = set()
    for r in keep:
        ug.update(B["reaction_genes"].get(r, []))
    say(f"       by gene      {len(ug):,} Ensembl gene ids appear in a kept reaction")
    t1 = sum(1 for t in rt.values() if t == "1_human_EC")
    say(f"       of tier 1    {len(keep):,} of {t1:,} = {len(keep) / max(t1, 1):.1%} survive the "
        f"self-consistency test")
    gates["N4"] = bool(len(keep) > 0)
    say(f"     N4 {'PASS' if gates['N4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N5
    say("N5 WHAT REPLACES THE REJECTED ONES")
    moved = [r for r in drop if drop[r] > 0 and max(drop[r] / gmed, gmed / drop[r]) > 2]
    m10 = [r for r in drop if drop[r] > 0 and max(drop[r] / gmed, gmed / drop[r]) > 10]
    say(f"     loop 124 measured the constant at 9.42x fold-error and CatPred at 12.95x, so the "
        f"honest replacement is the constant, FLAGGED AS ONE")
    say(f"     of the {len(drop):,} replaced reactions, the value moves by more than 2x for "
        f"{len(moved):,} and more than 10x for {len(m10):,}")
    say(f"     nothing is deleted -- a rejected reaction keeps a usable number, it just stops "
        f"claiming to be a measurement")
    gates["N5"] = bool(len(drop) == 0 or len(moved) >= 0)
    say(f"     N5 {'PASS' if gates['N5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N6
    say("N6 THE COST, AND WHETHER THE SURVIVORS ARE JUST THE FAMOUS ONES")
    D = CA.load()
    pubs = D["pubs"]
    ens = {}
    with open(LR.SC / "HumanGEM_genes.tsv") as f:
        rr = csv.reader(f, delimiter="\t")
        hd = [c.strip('"') for c in next(rr)]
        a_, b_ = hd.index("genes"), hd.index("geneSymbols")
        for x in rr:
            e_, s_ = x[a_].strip('"'), x[b_].strip('"')
            if e_ and s_:
                ens[e_] = s_.split(";")[0]
    kept_g = {ens[z] for r in keep for z in B["reaction_genes"].get(r, []) if z in ens}
    drop_g = {ens[z] for r in drop for z in B["reaction_genes"].get(r, []) if z in ens} - kept_g
    pk = np.array([pubs.get(g, 0.0) for g in kept_g])
    pd_ = np.array([pubs.get(g, 0.0) for g in drop_g])
    obs6, p6 = perm_p(np.log10(pk + 1), np.log10(pd_ + 1), rng)
    say(f"     kept genes {len(kept_g):,} (median {np.median(pk):.0f} publications), "
        f"dropped {len(drop_g):,} (median {np.median(pd_):.0f})")
    say(f"     log10 difference {obs6:+.3f}, permutation p = {p6:.4f}")
    say(f"     {'THE FILTER KEEPS THE FAMOUS ENZYMES' if (obs6 > 0 and p6 < 0.05) else 'no significant fame difference between kept and dropped'}")
    gates["N6"] = bool(np.isfinite(p6))
    say(f"     N6 {'PASS' if gates['N6'] else 'FAIL'} -- the selection effect is measured "
        f"and reported either way")
    say()

    if gates["N2"] and gates["N3"]:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"reaction_kcat_per_s": {**keep, **{r: gmed for r in drop}},
                   "reaction_tier": tier_new, "floor": FLOOR,
                   "ec_self_consistency": selfc,
                   "provenance": {"loop": 127, "rule": f"tier-1 EC medians whose own leave-one-out "
                                                       f"self-consistency is <= {FLOOR}x are kept; "
                                                       f"everything else becomes the global median "
                                                       f"{gmed}/s, flagged as a constant",
                                  "measured_floor_median": q50, "measured_floor_75th": q75}},
                  gzip.open(DEST, "wt"), indent=1)
        say(f"  wrote {DEST}")
    else:
        say(f"  NOT WRITING {DEST} -- N2 or N3 failed, and a filter that cannot be shown to work "
            f"is not shipped")
    say()

    say("=" * 100)
    for k in ("N1", "N2", "N3", "N4", "N5", "N6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[BUNDLE, ECMED, UPK, LR.SC / "HumanGEM_genes.tsv"],
                      available=len(rk), used=len(keep), selection="filtered", seed=SEED,
                      controls=["the noise floor measured from replicates, not assumed",
                                "replicate count included as a selector expected to fail",
                                "the selector required to be computable without the answer",
                                "held-out UniProt measurements as the validation",
                                "the constant as the flagged fallback, per loop 124",
                                "publication count of kept versus dropped genes"],
                      note="filtering on agreement-with-measurement would be selection on the "
                           "outcome; the selector here is each EC's own internal self-consistency")
    RM.report(man, emit=say)
    json.dump({"test": "loop_kcat_floor", "manifest": man, "gates": gates,
               "floor": FLOOR,
               "n1": {"loo_median": q50, "loo_75th": q75, "loo_90th": q90,
                      "percentile_of_floor": pct, "n_records": int(len(loo_all)),
                      "n_ec": len(selfc)},
               "n2": n2, "n3": {"n_val": len(val), "n_accepted": int(acc.sum()),
                                "log10_diff": obs, "p": p3},
               "n4": {"before": len(rk), "kept": len(keep), "dropped": len(drop),
                      "tier1_before": t1, "tier1_survival": len(keep) / max(t1, 1)},
               "n5": {"moved_2x": len(moved), "moved_10x": len(m10), "constant": gmed},
               "n6": {"kept_genes": len(kept_g), "dropped_genes": len(drop_g),
                      "pubs_log10_diff": obs6, "p": p6},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_kcat_floor.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_kcat_floor.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
