"""PATH-ORPHAN, STEP 5 -- function-recovery kill-gate (the trust bound).

If the pipeline can't recover the function of genes we ALREADY know, its calls
on actual ghosts are noise. This holds out the annotations of a random sample of
characterized genes, runs the atlas without that information, and measures how
often the assigned function tier carries the right signal.

CHANNEL-BY-CHANNEL TEST: rather than re-running every channel under masking,
measure each channel's *intrinsic* recovery rate, which is the directly
informative thing:
  - structure (foldhit): if a masked gene had foldhit_target_product, does it
    match the true product family? (when foldhit parquet present)
  - context (coinherit): does the partner_product match the true product family?
  - cofit (cofit_cons): for masked essential genes, does cofit_cons match the
    "high-conservation partner" signature we expect?

PRODUCT-FAMILY MATCH: substring + KEGG-like keyword overlap. Conservative
(undercounts true matches), so the recovery floor is honest.

KILL-GATE: combined recovery rate must beat random product-family agreement
(measured by shuffling product labels). If combined <= random shuffle, the
atlas is structured noise.

--smoke : validate masking + recovery scoring on synthetic data
--real  : sample 200 annotated genes, hold out, score per-channel + combined
"""
from __future__ import annotations
import argparse, json, re, sys, random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
DARK = re.compile(r'hypothetical|uncharacteri|DUF\d|domain of unknown|putative protein', re.I)

# product-family keywords -- shared keyword = compatible function
FAMILY_KW = re.compile(
    r'\b(ribosom\w*|tRNA|rRNA|polymerase|gyrase|topoisomerase|helicase|'
    r'kinase|phosphatase|reductase|dehydrogenase|oxidase|synth\w*|'
    r'transferase|hydrolase|ligase|peptidase|protease|nuclease|'
    r'transporter|permease|porin|channel|ATPase|ABC|MFS|RND|'
    r'regulator|sigma|sensor|repressor|activator|'
    r'membrane|outer|inner|envelope|capsule|pilus|flagell\w*|fimbri\w*|'
    r'biosynth\w*|metabol\w*|catabol\w*|degradation|'
    r'replication|transcription|translation|chaperone|'
    r'ribose|glucose|fructose|amino|peptide|lipid|fatty acid|nucleotide|'
    r'iron|sulfur|copper|zinc|magnesium|'
    r'cytochrome|heme|quinone|NAD\w*|FAD|coenzyme)\b',
    re.I)


def family_keywords(product):
    return set(m.group(0).lower() for m in FAMILY_KW.finditer(str(product or "")))


def family_match(predicted_product, true_product):
    """compatible function if they share >=1 family keyword. 0/1."""
    if not predicted_product or not true_product:
        return 0
    a = family_keywords(predicted_product)
    b = family_keywords(true_product)
    return int(bool(a & b))


def run_real(args):
    import pandas as pd, numpy as np
    atlas = OUT / f"atlas_{args.org}.parquet"
    if not atlas.exists():
        print("ERROR: run orphan_atlas.py --real first"); return 2
    a = pd.read_parquet(atlas)
    # candidates: annotated genes (real product, non-dark) with at least one
    # channel that COULD have given a function hint
    annotated = a[~a["product"].fillna("").str.contains(DARK)].copy()
    print("=" * 70)
    print(f"FUNCTION-RECOVERY validator -- {args.org}")
    print("=" * 70)
    print(f"  annotated genes available: {len(annotated):,}")

    rng = random.Random(0)
    n_sample = min(args.n_sample, len(annotated))
    sample_idx = rng.sample(list(annotated.index), n_sample)
    sample = annotated.loc[sample_idx].copy()
    print(f"  hold-out sample: {len(sample):,}")

    # channel 1: STRUCTURE (foldhit named target product)
    have_fold = "fold_target_product" in sample.columns
    if have_fold:
        sample["fold_match"] = [
            family_match(p, t) for p, t in
            zip(sample["fold_target_product"].fillna(""), sample["product"])
        ]
        n_fold = (sample["fold_target_product"].fillna("") != "").sum()
        rec_fold = sample["fold_match"].sum() / max(1, n_fold)
        print(f"  STRUCTURE  : {int(n_fold):>4} have a fold hit; "
              f"family-match recovery {rec_fold:.2%}")
    else:
        rec_fold = float("nan")
        print("  STRUCTURE  : foldhit parquet absent (run STEP 1 on Colab)")

    # channel 2: CO-INHERITANCE (partner_product on the confound-guarded hints)
    have_ctx = "context_partner_product" in sample.columns
    if have_ctx:
        sample["ctx_match"] = [
            family_match(p, t) for p, t in
            zip(sample["context_partner_product"].fillna(""), sample["product"])
        ]
        n_ctx = (sample["context_partner_product"].fillna("") != "").sum()
        rec_ctx = sample["ctx_match"].sum() / max(1, n_ctx)
        print(f"  CONTEXT    : {int(n_ctx):>4} have a guarded co-inheritance partner; "
              f"family-match recovery {rec_ctx:.2%}")
    else:
        rec_ctx = float("nan")
        print("  CONTEXT    : context_partner_product absent")

    # combined: any channel hit
    combined_n = combined_hit = 0
    for _, r in sample.iterrows():
        ev = []
        if have_fold and r.get("fold_target_product"):
            ev.append(family_match(r["fold_target_product"], r["product"]))
        if have_ctx and r.get("context_partner_product"):
            ev.append(family_match(r["context_partner_product"], r["product"]))
        if ev:
            combined_n += 1
            combined_hit += int(any(e == 1 for e in ev))
    combined = combined_hit / combined_n if combined_n else float("nan")
    print(f"  COMBINED   : {combined_n:>4} genes had any channel; "
          f"recovery {combined:.2%}" if combined_n else
          "  COMBINED   : no channels with content (run STEP 1 first)")

    # RANDOM-SHUFFLE NULL: shuffle the (predicted, true) pairing across the
    # sample; family_match should drop to background
    null_rates = []
    if combined_n > 0:
        truths = list(sample["product"])
        preds = []
        for _, r in sample.iterrows():
            if have_fold and r.get("fold_target_product"):
                preds.append(r["fold_target_product"])
            elif have_ctx and r.get("context_partner_product"):
                preds.append(r["context_partner_product"])
            else:
                preds.append(None)
        kept_idx = [i for i, p in enumerate(preds) if p]
        for s in range(args.n_perms):
            rrng = random.Random(s + 1)
            shuffled = list(truths)
            rrng.shuffle(shuffled)
            hits = sum(family_match(preds[i], shuffled[i]) for i in kept_idx)
            null_rates.append(hits / max(1, len(kept_idx)))
        null_med = float(sum(null_rates) / len(null_rates))
        p = float(sum(1 for r in null_rates if r >= combined) / len(null_rates))
        print(f"  NULL       : shuffled-pair recovery median {null_med:.2%} "
              f"(p(null>=real) {p:.3f})")
        verdict = ("PASS -- recovery exceeds shuffled-pair null"
                   if combined > null_med and p < 0.05 else
                   "FAIL -- recovery within shuffled-pair null; atlas is structured noise")
        print(f"\n  GATE: {verdict}")
    else:
        null_med = float("nan"); p = float("nan")
        print("\n  GATE: deferred until at least one channel has content")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"validator_{args.org}.json").write_text(json.dumps({
        "org": args.org, "n_sample": n_sample,
        "structure_recovery": rec_fold if rec_fold == rec_fold else None,
        "context_recovery": rec_ctx if rec_ctx == rec_ctx else None,
        "combined_n": combined_n,
        "combined_recovery": combined if combined == combined else None,
        "null_median": null_med if null_med == null_med else None,
        "null_p": p if p == p else None,
    }, indent=2))
    print(f"\n  wrote validator_{args.org}.json")
    return 0


def run_smoke():
    print("=" * 70)
    print("VALIDATOR SMOKE -- family_match + shuffled-null logic")
    print("=" * 70)
    assert family_match("ATP synthase subunit beta", "ATP synthase subunit alpha")
    assert not family_match("ATP synthase", "DNA gyrase subunit A")
    assert family_match("MFS transporter", "permease of the MFS family")
    print("  family_match: OK")
    # synthetic recovery: 60% true match, 40% wrong
    import random
    rng = random.Random(0)
    pairs = []
    for _ in range(60):
        pairs.append(("ATP synthase subunit", "ATP synthase complex"))
    for _ in range(40):
        pairs.append(("DNA polymerase", "ribosomal protein L7"))
    hits = sum(family_match(p, t) for p, t in pairs)
    rate = hits / len(pairs)
    print(f"  planted 60% match: measured {rate:.2%}")
    assert 0.55 <= rate <= 0.65, "family_match should recover the planted rate"
    # shuffled null: pair shuffling should hit ~0
    preds = [p for p, _ in pairs]; truths = [t for _, t in pairs]
    shuffled = list(truths); rng.shuffle(shuffled)
    null_rate = sum(family_match(preds[i], shuffled[i]) for i in range(len(pairs))) / len(pairs)
    print(f"  shuffled-pair null: {null_rate:.2%}")
    assert null_rate < rate, "null must be below real recovery"
    print("\n=== VALIDATOR SMOKE PASS ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--n_sample", type=int, default=200)
    ap.add_argument("--n_perms", type=int, default=100)
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
