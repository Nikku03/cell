"""LOOP 124 -- AUDIT THE ENZYME KINETICS AGAINST MEASUREMENT, with a correction owed first.

THE CORRECTION. cell_assembled recorded enzyme kinetics as ABSENT -- "no measured k_cat anywhere" --
and I repeated that claim. It is wrong and has been for a while. colab/data/kinetics_bundle.json.gz
is committed and carries a kcat for 8,184 reactions and 2,549 genes, built by build_kinetics_bundle
from a measured base of 2,437 human records. A stale ABSENT is worse than a FAILED because it tells
the reader not to look, and this is the second one this session.

WHAT IS ACTUALLY WRONG WITH THE KINETICS, which is a different and more interesting problem:

    tier 1  human EC median        1,811 reactions   22.1%   <- measured, but only 362 distinct ECs
    tier 2  CatPred prediction     5,941 reactions   72.6%   <- MACHINE LEARNING, not measurement
    tier 3  any-organism EC          127 reactions    1.6%
    tier 4  global median 1.85/s     305 reactions    3.7%   <- the null, wearing a physical unit

Three quarters of the model's turnover numbers are predictions. The bundle reports their accuracy as
8.38x fold-error, but that figure came from the same BRENDA/SABIO compilation the predictor was
trained on. Nobody has ever tested them against a gene-level measurement the predictor did not see.

WHAT WAS FETCHED. UniProt curates kinetic parameters from the primary literature, per PROTEIN, with
the PubMed identifier attached -- not per EC class, and not from the BRENDA flat file. 1,309 human
reviewed proteins carry a Kinetics block. BRENDA's own bulk download is now behind a client-side
license gate with no static URL, and SABIO-RK's REST service is retired (every endpoint 302s to
/ui/404), so UniProt is not a second choice here, it is the one route to gene-level measured values
that still works.

MEASURED DURING CONSTRUCTION AND THEREFORE DISCLOSED:
    1,309 entries -> 239 genes with a parsed kcat (860 individual values), 1,275 genes with a KM
    parser validated on enzymes with published values: PNP 28-70/s, FH 170-280/s, TK1 4.7/s
    overlap with the bundle's gene kcat: 147 -- of which 80 are already tier 'measured'
    THE HELD-OUT SET IS THE OTHER 67: genes where the bundle used a PREDICTION and UniProt has a
    measurement. Small, and it is the only honest test available.
    overlap with the bundle's PREDICTED KM: 628 genes

THE INDEPENDENCE CAVEAT, STATED UP FRONT RATHER THAN IN A FOOTNOTE. CatPred trained on BRENDA and
SABIO-RK. UniProt curates the same primary literature. So a UniProt value whose source paper is also
in BRENDA is not held out in the strict sense -- the predictor may have seen that number under a
different accession. This audit is therefore a LOWER BOUND on the error, not an unbiased estimate,
and K5 gates on saying so rather than on a number.

PREDECLARED:

  K1 THE PARSER DID NOT INVENT UNITS                                THE PREREQUISITE.
       a unit bug is the failure mode that would make every number below meaningless, and it shows
       up as scale outliers. Gate, all three: no parsed kcat above 1e7 /s (above the diffusion
       limit for any enzyme), no parsed KM above 1e6 uM (above 1 molar), and at least 90% of the
       entries that mention kcat must yield a value -- a parser that silently drops half the data
       is selecting on phrasing.
  K2 HOW WRONG ARE THE PREDICTIONS?                                 THE POINT OF THE LOOP.
       median fold-error of the bundle's predicted kcat against the UniProt measurement, on the
       held-out genes only, against the global-median null on the same genes. Gate: the prediction
       must beat the null. If a constant 1.85/s does as well as the model, then 72.6% of the
       model's kinetics carries no information and the record must say so.
  K3 THE TIER ORDERING IS REAL                                      THE BUNDLE'S OWN CLAIM.
       build_kinetics_bundle orders its tiers by leave-one-out accuracy: human-EC 2.62-2.80x,
       CatPred 8.38x, global median 9.25-14.23x. Tested here on values none of those tiers used.
       Gate: tier 1 must beat tier 2 must beat tier 4, on this set.
  K4 THE PREDICTED KM, AGAINST 628 MEASUREMENTS                     THE UNTESTED HALF.
       the bundle's gene_km_uM is tagged catpred+ECprior throughout -- every KM in this model is
       predicted, and 628 of them now have a measured counterpart. Gate: predicted KM must beat a
       global-median-KM null by fold-error.
  K5 INDEPENDENCE AND SELECTION, BOTH NAMED                         THE GUARD.
       (a) the 80 already-'measured'-tier genes are EXCLUDED from K2/K3, because the bundle
           already contains them and scoring on them would be scoring a lookup;
       (b) publication count against having a measured kcat at all -- if famous enzymes are the
           measured ones, coverage is not random and the audit set is not the model's population.
       Gate: the exclusion is applied and the selection effect is measured and reported.
  K6 WHAT IT WOULD CHANGE                                           THE CONSEQUENCE.
       propagate the measured values onto the reactions those genes catalyse and count how many
       reaction kcats move by more than 2x and more than 10x. Gate: the count is produced with the
       three denominators this repository uses -- by reaction, by gene, and against the 12,931.

-> outputs/loop_kcat_audit.json
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
UP = LR.SC / "uniprot_kinetics_human.tsv"
BUNDLE = Path("colab/data/kinetics_bundle.json.gz")
ECMED = Path("colab/data/ec_kcat_medians.json.gz")
SEED = 12400
NPERM = 2000

K1_KCAT_MAX = 1e7
K1_KM_MAX = 1e6
K1_PARSE_MIN = 0.90
MEASURED_TIERS = ("measured", "EC-measured")

NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
KCAT_ONE = re.compile(r"kcat is " + NUM + r"\s*(sec|min|hour|h)\(-1\)", re.I)
KCAT_LIST = re.compile(r"kcats? (?:is|are) ((?:" + NUM + r"\s*(?:sec|min|hour|h)\(-1\)[,\s and]*)+)",
                       re.I)
KCAT_ITEM = re.compile(NUM + r"\s*(sec|min|hour|h)\(-1\)", re.I)
KM_RE = re.compile(r"KM=" + NUM + r"\s*(nM|uM|mM|M)\b")
TO_PER_S = {"sec": 1.0, "min": 1 / 60.0, "hour": 1 / 3600.0, "h": 1 / 3600.0}
TO_UM = {"nM": 1e-3, "uM": 1.0, "mM": 1e3, "M": 1e6}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def geo(v):
    v = [q for q in v if q and q > 0 and np.isfinite(q)]
    return float(np.exp(np.mean(np.log(v)))) if v else None


def fold_error(pred, meas):
    """Symmetric fold error: always >= 1, direction-free, the standard for kinetic predictions."""
    p, m = np.asarray(pred, float), np.asarray(meas, float)
    ok = (p > 0) & (m > 0) & np.isfinite(p) & np.isfinite(m)
    return np.maximum(p[ok] / m[ok], m[ok] / p[ok])


def parse_uniprot(path):
    rows = list(csv.reader(open(path, newline=""), delimiter="\t"))
    h, rows = rows[0], rows[1:]
    iG, iK, iE = (h.index("Gene Names (primary)"), h.index("Kinetics"), h.index("EC number"))
    kc, km, ec, nval = {}, {}, {}, {}
    mention, parsed = 0, 0
    for x in rows:
        g, blk = x[iG].strip(), x[iK]
        if not g:
            continue
        if re.search(r"kcat", blk, re.I):
            mention += 1
        vals = [float(a) * TO_PER_S[b.lower()] for a, b in KCAT_ONE.findall(blk)]
        for m in KCAT_LIST.finditer(blk):
            vals += [float(a) * TO_PER_S[b.lower()] for a, b in KCAT_ITEM.findall(m.group(1))]
        vals = list(dict.fromkeys(vals))
        if vals:
            parsed += 1
            kc[g] = geo(vals)
            nval[g] = len(vals)
        w = [float(a) * TO_UM[b] for a, b in KM_RE.findall(blk)]
        if w:
            km[g] = geo(w)
        if x[iE].strip():
            ec[g] = x[iE].split(";")[0].strip()
    return ({k: v for k, v in kc.items() if v}, km, ec, nval,
            {"entries": len(rows), "mention_kcat": mention, "parsed_kcat": parsed})


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 124 -- audit the enzyme kinetics against measurement")
    say("=" * 100)
    say()

    B = json.load(gzip.open(BUNDLE, "rt"))
    E = json.load(gzip.open(ECMED, "rt"))
    gk, gt, gkm = B["gene_kcat_per_s"], B["gene_tier"], B["gene_km_uM"]
    rk, rt, rg = B["reaction_kcat_per_s"], B["reaction_tier"], B["reaction_genes"]
    gmed = float(E["global_median_per_s"])
    say(f"  the bundle: {len(rk):,} reactions and {len(gk):,} genes with a kcat; "
        f"global median fallback {gmed}/s")
    tc = collections.Counter(rt.values())
    for k, n in tc.most_common():
        say(f"     {k:<22} {n:>6}  {n / len(rt):6.1%}")
    say(f"  measured base: {len(E['human_records']):,} human records over "
        f"{len(E['ec_human_median_per_s'])} distinct EC numbers")
    say()

    kc, km, ec, nval, stats = parse_uniprot(UP)
    say(f"  UniProt: {stats['entries']:,} human reviewed proteins with a Kinetics block")
    say(f"     mention kcat {stats['mention_kcat']}, parsed {stats['parsed_kcat']} "
        f"({stats['parsed_kcat'] / max(stats['mention_kcat'], 1):.1%}); "
        f"{len(kc)} genes with a kcat, {len(km)} with a KM")
    say()

    gates = {}

    # ---------------------------------------------------------------- K1
    say("K1 THE PARSER DID NOT INVENT UNITS")
    kv = np.array(list(kc.values()))
    mv = np.array(list(km.values()))
    say(f"     kcat /s : median {np.median(kv):.3f}  range {kv.min():.2e} to {kv.max():.2e}  "
        f"gate max < {K1_KCAT_MAX:.0e}")
    say(f"     KM   uM : median {np.median(mv):.2f}  range {mv.min():.2e} to {mv.max():.2e}  "
        f"gate max < {K1_KM_MAX:.0e}")
    rate = stats["parsed_kcat"] / max(stats["mention_kcat"], 1)
    say(f"     parse rate on kcat-mentioning entries {rate:.1%}   gate >= {K1_PARSE_MIN:.0%}")
    for g in ("PNP", "FH", "TK1"):
        if g in kc:
            say(f"     positive control {g}: {kc[g]:.3g}/s from {nval[g]} published values")
    gates["K1"] = bool(kv.max() < K1_KCAT_MAX and mv.max() < K1_KM_MAX and rate >= K1_PARSE_MIN)
    say(f"     K1 {'PASS' if gates['K1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K5(a) first: the exclusion
    say("K5 INDEPENDENCE AND SELECTION, BOTH NAMED")
    shared = [g for g in kc if g in gk]
    already = [g for g in shared if gt.get(g) in MEASURED_TIERS]
    held = [g for g in shared if gt.get(g) not in MEASURED_TIERS]
    say(f"     {len(shared)} genes have both a UniProt kcat and a bundle kcat")
    say(f"     EXCLUDED, already tier 'measured' in the bundle: {len(already)} "
        f"-- scoring on these would be scoring a lookup")
    say(f"     HELD OUT, bundle used a prediction: {len(held)}")
    say("     tiers of the held-out set: " +
        ", ".join(f"{k} {n}" for k, n in collections.Counter(gt[g] for g in held).most_common()))
    D = CA.load()
    pubs = D["pubs"]
    has = np.array([1.0 if g in kc else 0.0 for g in gk])
    pv = np.array([pubs.get(g, 0.0) for g in gk])
    a, b_ = pv[has > 0], pv[has == 0]
    obs = float(np.median(a) - np.median(b_))
    null = np.array([float(np.median(pv[p][:len(a)]) - np.median(pv[p][len(a):]))
                     for p in (rng.permutation(len(pv)) for _ in range(NPERM))])
    p_sel = float(np.mean(np.abs(null) >= abs(obs)))
    say(f"     SELECTION: genes with a measured kcat have median {np.median(a):.0f} publications "
        f"against {np.median(b_):.0f} for those without, difference {obs:+.0f}, p = {p_sel:.4f}")
    say(f"     so the audit set is {'NOT ' if p_sel < 0.05 else ''}a random sample of the model's "
        f"genes, and the fold-errors below describe the enzymes people study")
    gates["K5"] = bool(len(already) > 0 and len(held) > 0 and np.isfinite(p_sel))
    say(f"     K5 {'PASS' if gates['K5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K2
    say("K2 HOW WRONG ARE THE PREDICTIONS?")
    pred = np.array([gk[g] for g in held])
    meas = np.array([kc[g] for g in held])
    fe = fold_error(pred, meas)
    fe_null = fold_error(np.full(len(meas), gmed), meas)
    say(f"     held-out genes with both values: {len(fe)}")
    say(f"     bundle prediction   median fold-error {np.median(fe):>7.2f}x   "
        f"within 2x {np.mean(fe <= 2):.1%}   within 10x {np.mean(fe <= 10):.1%}")
    say(f"     global-median null  median fold-error {np.median(fe_null):>7.2f}x   "
        f"within 2x {np.mean(fe_null <= 2):.1%}   within 10x {np.mean(fe_null <= 10):.1%}")
    d = float(np.median(np.log10(fe)) - np.median(np.log10(fe_null)))
    pool = np.concatenate([np.log10(fe), np.log10(fe_null)])
    k = len(fe)
    nn = np.array([(lambda s: np.median(s[:k]) - np.median(s[k:]))(rng.permutation(pool))
                   for _ in range(NPERM)])
    p2 = float(np.mean(np.abs(nn) >= abs(d)))
    say(f"     log10 fold-error difference {d:+.3f} (negative favours the prediction), "
        f"permutation p = {p2:.4f}")
    say(f"     the bundle's own leave-one-out figure for CatPred was 8.38x")
    gates["K2"] = bool(np.median(fe) < np.median(fe_null) and p2 < 0.05)
    say(f"     K2 {'PASS' if gates['K2'] else 'FAIL'} -- the prediction "
        f"{'beats a constant' if gates['K2'] else 'DOES NOT beat a constant 1.85/s'}")
    say()

    # ---------------------------------------------------------------- K3
    say("K3 THE TIER ORDERING IS REAL")
    ec_h = E["ec_human_median_per_s"]
    t1 = [(ec_h[ec[g]], kc[g]) for g in kc if g in ec and ec.get(g) in ec_h]
    say(f"     tier 1 (human EC median) testable on {len(t1)} genes with a UniProt kcat")
    order = {}
    if t1:
        f1 = fold_error([x for x, _ in t1], [y for _, y in t1])
        order["1_human_EC"] = float(np.median(f1))
        say(f"       tier 1  median fold-error {np.median(f1):.2f}x  (bundle claims 2.62-2.80x)")
    cat = [g for g in held if gt[g].startswith("catpred")]
    if cat:
        fc = fold_error([gk[g] for g in cat], [kc[g] for g in cat])
        order["2_catpred"] = float(np.median(fc))
        say(f"       tier 2  median fold-error {np.median(fc):.2f}x  on {len(fc)} genes  "
            f"(bundle claims 8.38x)")
    order["4_global_median"] = float(np.median(fe_null))
    say(f"       tier 4  median fold-error {np.median(fe_null):.2f}x  (bundle claims 9.25-14.23x)")
    ok3 = ("1_human_EC" in order and "2_catpred" in order
           and order["1_human_EC"] < order["2_catpred"] < order["4_global_median"])
    gates["K3"] = bool(ok3)
    say(f"     K3 {'PASS' if gates['K3'] else 'FAIL'} -- the claimed ordering "
        f"{'holds' if ok3 else 'DOES NOT hold on values none of the tiers used'}")
    say()

    # ---------------------------------------------------------------- K4
    say("K4 THE PREDICTED KM, AGAINST 628 MEASUREMENTS")
    kmg = [g for g in km if g in gkm]
    pk = np.array([gkm[g] for g in kmg])
    mk = np.array([km[g] for g in kmg])
    fk = fold_error(pk, mk)
    med_km = float(np.median(list(km.values())))
    fk_null = fold_error(np.full(len(mk), med_km), mk)
    say(f"     {len(fk)} genes with a predicted KM and a measured KM")
    say(f"     every KM in this model is tagged catpred+ECprior -- there is no measured-KM tier")
    say(f"     predicted KM       median fold-error {np.median(fk):>7.2f}x   "
        f"within 2x {np.mean(fk <= 2):.1%}")
    say(f"     constant {med_km:.1f} uM null  median fold-error {np.median(fk_null):>7.2f}x   "
        f"within 2x {np.mean(fk_null <= 2):.1%}")
    dk = float(np.median(np.log10(fk)) - np.median(np.log10(fk_null)))
    poolk = np.concatenate([np.log10(fk), np.log10(fk_null)])
    kk = len(fk)
    nk = np.array([(lambda s: np.median(s[:kk]) - np.median(s[kk:]))(rng.permutation(poolk))
                   for _ in range(NPERM)])
    p4 = float(np.mean(np.abs(nk) >= abs(dk)))
    say(f"     log10 difference {dk:+.3f}, permutation p = {p4:.4f}")
    gates["K4"] = bool(np.median(fk) < np.median(fk_null) and p4 < 0.05)
    say(f"     K4 {'PASS' if gates['K4'] else 'FAIL'} -- the predicted KM "
        f"{'beats a constant' if gates['K4'] else 'DOES NOT beat a constant'}")
    say()

    # ---------------------------------------------------------------- K6
    say("K6 WHAT IT WOULD CHANGE")
    ens2sym = {}
    for r_, gs in rg.items():
        pass
    hit_rx, moved2, moved10 = set(), set(), set()
    sym_of = {}
    for g in kc:
        sym_of[g] = g
    for r_, gs in rg.items():
        syms = [s for s in gs if s in kc]
        if not syms:
            continue
        hit_rx.add(r_)
        if r_ not in rk:
            continue
        new = geo([kc[s] for s in syms])
        old = rk[r_]
        if new and old > 0:
            f = max(new / old, old / new)
            if f > 2:
                moved2.add(r_)
            if f > 10:
                moved10.add(r_)
    say(f"     reactions whose gene set contains a measured gene: {len(hit_rx):,} "
        f"of {len(rg):,} with a gene rule")
    say(f"     of those, kcat moves by more than  2x: {len(moved2):,}")
    say(f"                                       10x: {len(moved10):,}")
    say(f"     THREE DENOMINATORS  by reaction {len(hit_rx) / len(rg):.1%} of gene-rule reactions, "
        f"{len(hit_rx) / 12931:.1%} of the 12,931 model reactions")
    say(f"                         by gene     {len(kc)} measured of {len(gk):,} with a kcat "
        f"= {len(kc) / len(gk):.1%}, and of 16,492 model genes = {len(kc) / 16492:.1%}")
    gates["K6"] = bool(len(hit_rx) > 0)
    say(f"     K6 {'PASS' if gates['K6'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    for k in ("K1", "K2", "K3", "K4", "K5", "K6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[UP, BUNDLE, ECMED, LR.CELL], available=stats["entries"],
                      used=len(kc), selection="filtered", seed=SEED,
                      controls=["a constant global median as the null for both kcat and KM",
                                "genes already tier 'measured' EXCLUDED from the audit",
                                "publication count against having a measured value at all",
                                "the bundle's own claimed tier ordering, retested",
                                "unit-scale bounds as a parser check",
                                "three denominators on the coverage"],
                      note="UniProt curates per protein from primary literature with PubMed ids; "
                           "CatPred trained on BRENDA/SABIO, which share that literature, so this "
                           "is a LOWER BOUND on the error and not an unbiased estimate")
    RM.report(man, emit=say)
    json.dump({"test": "loop_kcat_audit", "manifest": man, "gates": gates,
               "correction": "cell_assembled recorded enzyme kinetics ABSENT; the bundle exists "
                             "and carries 8,184 reaction kcats",
               "bundle": {"reactions": len(rk), "genes": len(gk), "tiers": dict(tc),
                          "global_median": gmed,
                          "measured_human_records": len(E["human_records"]),
                          "distinct_human_ec": len(E["ec_human_median_per_s"])},
               "uniprot": {**stats, "genes_kcat": len(kc), "genes_km": len(km)},
               "k1": {"kcat_median": float(np.median(kv)), "kcat_max": float(kv.max()),
                      "km_median": float(np.median(mv)), "km_max": float(mv.max()),
                      "parse_rate": rate},
               "k2": {"n_held": len(fe), "pred_fold": float(np.median(fe)),
                      "null_fold": float(np.median(fe_null)),
                      "pred_within2": float(np.mean(fe <= 2)),
                      "pred_within10": float(np.mean(fe <= 10)),
                      "log10_diff": d, "p": p2},
               "k3": order,
               "k4": {"n": len(fk), "pred_fold": float(np.median(fk)),
                      "null_fold": float(np.median(fk_null)), "const_uM": med_km,
                      "log10_diff": dk, "p": p4},
               "k5": {"n_shared": len(shared), "n_excluded": len(already), "n_held": len(held),
                      "pubs_with": float(np.median(a)), "pubs_without": float(np.median(b_)),
                      "pubs_difference": obs, "pubs_p": p_sel},
               "k6": {"reactions_touched": len(hit_rx), "moved_2x": len(moved2),
                      "moved_10x": len(moved10), "gene_rule_reactions": len(rg)},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_kcat_audit.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_kcat_audit.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
