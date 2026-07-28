"""Reproduce the EC-median kcat file that the sandbox rollback destroyed, and make it DURABLE.

WHAT WAS LOST AND WHY IT MATTERS. dlkcat.tsv (despite the name, JSON) held DLKcat's BRENDA+SABIO-RK
compilation: 17,010 kcat records with EC number, organism and value. It supplied two things nothing else
could -- the LABELS that kcat_headtohead.py scored every kcat source against, and tiers 1 and 3 of the
capacity hierarchy (human-EC median, 2.62x median fold error; any-organism EC median). Without it the
kinetics bundle falls back to CatPred at 8.38x for 7,592 of 8,184 reactions.

THE REPRODUCTION IS VERIFIED, NOT ASSUMED. Re-fetching a file from a URL proves nothing about whether it is
the file the earlier measurements used -- upstream repositories change. The check that actually binds:
filtering to Homo sapiens records with a valid positive Value and an EC number yields 2,437, which is
exactly the count kcat_headtohead.py recorded in its own docstring. A derived count matching to the record
is evidence the source is identical; a matching download URL is not.

WHAT THIS WRITES, and why it is not the raw file. The raw table is 11.7 MB, most of it protein Sequence and
Smiles strings this project never reads. Committing that to survive the next rollback would be storing 11 MB
to use 60 KB of it. This writes the DERIVED statistics plus the stripped records needed to REDERIVE them:

    ec_human_median    EC -> median kcat over Homo sapiens records      tier 1, 2.62x measured
    ec_all_median      EC -> median over every organism                 tier 3, coverage fallback
    ec_all_max         EC -> max over every organism                    ECMpy's rule; kept to DOCUMENT the
                                                                        exclusion, not to use -- 40.7x, bias
                                                                        +1.61, an upper bound not an estimator
    global_median      one number for every enzyme                      tier 4, the null, 14.23x
    human_records      (EC, organism, value) triples, no sequences      so leave-one-out can be re-run

LEAVE-ONE-OUT IS THE POINT OF KEEPING human_records. EC medians are computed FROM these labels, so scoring
them without removing the record being predicted lets the predictor see its own answer. kcat_headtohead.py
was written to avoid exactly that; keeping the records keeps that check reproducible instead of leaving a
2.62x claim in the repo with nothing able to re-derive it.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
RAW = SP / "dlkcat_raw.json"
DEST = Path("colab/data/ec_kcat_medians.json.gz")

# kcat_headtohead.py's docstring recorded this count for the same filter. If a re-fetch stops matching it,
# upstream changed and every accuracy figure derived from these labels needs re-measuring before reuse.
EXPECT_HUMAN = 2437
EXPECT_TOTAL = 17010


def valid(rec):
    """A record is usable only with an EC, a parseable value, and a positive finite magnitude."""
    ec = (rec.get("ECNumber") or "").strip()
    if not ec:
        return None
    try:
        v = float(rec.get("Value"))
    except (TypeError, ValueError):
        return None
    if not (v > 0) or not np.isfinite(v):
        return None
    return ec, (rec.get("Organism") or "").strip(), v


def leave_one_out(by_ec):
    """Median fold error of the EC-median rule, predicting each record from the OTHER records for its EC.

    This is the ONLY honest way to score a statistic against the labels it was computed from. It is also why
    the number here (2.80x over 2,337 records) is not the 2.62x in kcat_headtohead.py: that figure was scored
    on the 915 labels EVERY contender covered, so the two differ by label set, not by method. Reporting one as
    if it were the other is the same error this project already made once, comparing CatPred's n=950 against a
    null's n=2,437.
    """
    fe, n_ec = [], 0
    for vs in by_ec.values():
        if len(vs) < 2:
            continue
        n_ec += 1
        a = np.asarray(vs, float)
        for i in range(len(a)):
            pred = np.median(np.delete(a, i))
            if pred > 0 and a[i] > 0:
                fe.append(abs(np.log10(pred / a[i])))
    fe = np.asarray(fe)
    return fe, n_ec


def main():
    assert RAW.exists(), (
        f"{RAW} missing. Re-fetch with:\n"
        "  curl -sSL -o dlkcat_raw.json https://raw.githubusercontent.com/SysBioChalmers/DLKcat/"
        "master/DeeplearningApproach/Data/database/Kcat_combination_0918.json")
    recs = json.load(open(RAW))
    print(f"raw records: {len(recs):,}  (expected {EXPECT_TOTAL:,})")

    units = collections.Counter(r.get("Unit") for r in recs)
    assert set(units) == {"s^(-1)"}, f"MIXED UNITS {dict(units)} -- medians across units are meaningless"

    hum, allo = collections.defaultdict(list), collections.defaultdict(list)
    human_records = []
    for r in recs:
        p = valid(r)
        if p is None:
            continue
        ec, org, v = p
        allo[ec].append(v)
        if org == "Homo sapiens":
            hum[ec].append(v)
            human_records.append([ec, v])

    print(f"usable: {sum(len(v) for v in allo.values()):,} records over {len(allo):,} ECs")
    print(f"human : {len(human_records):,} records over {len(hum):,} ECs  (expected {EXPECT_HUMAN:,})")
    assert len(human_records) == EXPECT_HUMAN, (
        f"HUMAN COUNT {len(human_records)} != {EXPECT_HUMAN} -- this is NOT the table the 2.62x/8.38x/40.7x "
        "figures were measured on. Upstream changed; re-run kcat_headtohead.py before trusting any tier label.")

    ec_hum = {e: float(np.median(v)) for e, v in hum.items()}
    ec_all = {e: float(np.median(v)) for e, v in allo.items()}
    ec_max = {e: float(np.max(v)) for e, v in allo.items()}
    gmed = float(np.median([x for v in hum.values() for x in v]))

    # How much of the median's own support is a single record: a "median" over n=1 is that record. Stated
    # because tier 1's 2.62x is an average over ECs whose support varies by two orders of magnitude.
    n1 = sum(1 for v in hum.values() if len(v) == 1)
    print(f"\nEC medians: human {len(ec_hum):,}, any-organism {len(ec_all):,}, global median {gmed:.4g} /s")
    print(f"  human ECs supported by a SINGLE record: {n1:,}/{len(ec_hum):,} ({100*n1/len(ec_hum):.1f}%)")
    print(f"  human-EC median kcat spans {min(ec_hum.values()):.3g} .. {max(ec_hum.values()):.3g} /s")
    # ECMpy's max rule vs the median rule, on the ECs both cover. This is PART of why max-rule sources run
    # high, not the whole of it: 4.62x is 0.66 log units, while EC-max's measured bias against the labels was
    # +1.61. The remainder comes from the max being taken over ALL organisms including extremophile and
    # engineered enzymes, then applied to a human reaction. Quoting this ratio as if it explained the full
    # bias would be the same error as quoting ECMpy's own 6-8x figure against this project's labels.
    both = [e for e in ec_max if e in ec_hum]
    ratio = np.median([ec_max[e] / ec_hum[e] for e in both])
    print(f"  EC-max / human-EC-median, median ratio over {len(both):,} shared ECs: {ratio:.2f}x "
          f"({np.log10(ratio):.2f} log units; EC-max's measured bias vs labels was +1.61 -- see note in source)")

    # Score the rule we are about to ship, on the labels we are about to ship, rather than inheriting a
    # figure measured elsewhere on a different subset.
    fe, n_ec = leave_one_out(hum)
    loo_med = float(10 ** np.median(fe))
    gfe = np.abs(np.log10(gmed / np.array([v for _, v in human_records])))
    null_med = float(10 ** np.median(gfe))
    print(f"\nleave-one-out, human-EC median rule: n={len(fe):,} records over {n_ec:,} ECs with >=2 records")
    print(f"  median fold error : {loo_med:.2f}x   (kcat_headtohead.py measured 2.62x on the 915-label "
          f"COMMON subset -- different labels, not a discrepancy)")
    print(f"  within 4x         : {100*(fe < np.log10(4)).mean():.1f}% of records")
    print(f"  null, global median: {null_med:.2f}x   (14.23x on the common subset, same caveat)")
    print(f"  advantage over null: {null_med/loo_med:.2f}x")
    assert loo_med < null_med, "EC-median rule must beat the global-median null or it is not a tier"

    bundle = {
        "ec_human_median_per_s": ec_hum,
        "ec_all_median_per_s": ec_all,
        "ec_all_max_per_s": ec_max,
        "global_median_per_s": gmed,
        "human_records": human_records,
        "ec_human_n": {e: len(v) for e, v in hum.items()},
        "provenance": {
            "source": "DLKcat DeeplearningApproach/Data/database/Kcat_combination_0918.json "
                      "(BRENDA + SABIO-RK compilation)",
            "url": "https://raw.githubusercontent.com/SysBioChalmers/DLKcat/master/DeeplearningApproach/"
                   "Data/database/Kcat_combination_0918.json",
            "raw_records": len(recs),
            "human_records": len(human_records),
            "identity_check": f"human/valid/EC count == {EXPECT_HUMAN}, the count kcat_headtohead.py "
                              "recorded -- confirms this is the table the accuracy figures were measured on",
            "units": "s^-1, verified single-unit",
            # TWO label sets, both reported, because a fold-error figure is meaningless without the subset it
            # was scored on. The full-set numbers are the ones re-derivable from this file.
            "measured_accuracy_full_human_set": {
                "labels": f"{len(fe):,} leave-one-out records over {n_ec:,} ECs with >=2 human records",
                "human_EC_median_fold_error": round(loo_med, 3),
                "global_median_null_fold_error": round(null_med, 3),
                "frac_within_4x": round(float((fe < np.log10(4)).mean()), 4),
            },
            "measured_accuracy_common_subset": {
                "labels": "915 labels covered by EVERY contender (kcat_headtohead.py)",
                "human_EC_median": "2.62x", "CatPred": "8.38x", "global_median": "14.23x",
                "EC_max": "40.69x, bias +1.61 -- upper bound by construction, NOT used",
                "ecHumanGEM": "66.33x, bias +1.71 -- same reason",
            },
            "single_record_ECs": n1,
        },
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEST, "wt") as f:
        json.dump(bundle, f)
    print(f"\n  -> {DEST}  ({DEST.stat().st_size/1e6:.2f} MB)")

    chk = json.load(gzip.open(DEST, "rt"))
    assert len(chk["ec_human_median_per_s"]) == len(ec_hum), "roundtrip lost human ECs"
    assert len(chk["human_records"]) == EXPECT_HUMAN, "roundtrip lost records"
    print(f"  roundtrip OK: {len(chk['ec_human_median_per_s']):,} human ECs, "
          f"{len(chk['ec_all_median_per_s']):,} any-organism ECs, {len(chk['human_records']):,} records")


if __name__ == "__main__":
    main()
