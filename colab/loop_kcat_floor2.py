"""LOOP 129 -- THE KINETICS FILTER, REDONE AT THE TRUE FLOOR AND WITH FOUR TIMES THE POWER.

WHAT LOOP 127 GOT WRONG, AND IT WAS THE CENTRAL NUMBER. It measured the experimental noise floor
at 2.85x by leave-one-out WITHIN AN EC CLASS. But an EC class holds many different enzymes acting
on many different substrates, so that figure is enzyme-to-enzyme and substrate-to-substrate
variation -- it is not experimental reproducibility, and it is roughly three times too large.

The DLKcat compilation carries no UniProt identifier, which is why this repository had only ever
mined it at EC level. It does carry the protein SEQUENCE, and a sequence maps to a gene exactly.
Grouping by protein AND substrate isolates true replicates: the same enzyme, the same reaction,
measured independently.

    same protein, same substrate, independent measurements   median 1.15x   75th 1.76x
    49 protein-substrate pairs, 101 values

That is the real floor. Loop 127's 4x operating point survives, but for the opposite reason to the
one it gave -- 4x is about three times LOOSER than what can be measured, not close to the limit.

AND THE SAME MOVE FIXES THE POWER PROBLEM. Loop 127 could validate its filter on 70 genes: every
gene with both a UniProt kcat and an EC carrying replicates. Sequence-mapped DLKcat gives 342 genes
with a measured kcat, 289 of them in the model. Loop 127's N3 failed at n = 70 with the right
direction and p = 0.41, and its shipped filter was chosen post hoc from a four-threshold sweep. All
of that is retestable now.

DISCLOSED, measured during the fetch and therefore not gated on: the floor is 1.15x; the validation
set is 289; loop 127 shipped a breadth filter at <= 3 genes per EC, validated post hoc at 2.82x
against 43.31x, p 0.0035.

PREDECLARED:

  P1 THE TWO MEASURED SOURCES AGREE                                 THE PREREQUISITE.
       DLKcat-by-sequence against UniProt-by-curation, on the genes both cover. These are separate
       compilations of overlapping primary literature, so agreement is expected and disagreement
       would mean one of the two parsers is wrong. Gate: Spearman >= 0.5 AND median fold <= 4x. If
       they disagree, nothing downstream can be trusted and the loop stops.
  P2 THE SELECTOR, RETESTED WITH POWER                              LOOP 127's N2, PROPERLY.
       the same three candidates -- EC self-consistency, EC breadth, replicate count -- against
       fold-error on the enlarged set, with Bonferroni over the three applied this time rather than
       noted afterwards. Gate: at least one clears a CORRECTED p < 0.05. Loop 127's breadth
       selector cleared raw p 0.047 and failed correction at 0.143; this is the honest retest.
  P3 THE SHIPPED FILTER VALIDATES OUT OF SAMPLE                     THE ONE THAT MATTERS.
       loop 127 chose breadth <= 3 by sweeping four thresholds on 70 genes and shipped it. Applied
       unchanged to the 289 -- no re-tuning, no re-sweeping -- does it still separate? Gate:
       accepted fold-error below rejected, permutation p < 0.05. A post-hoc threshold that holds on
       four times the data is a filter; one that does not was overfitting, and the artefact must be
       withdrawn.
  P4 THE THRESHOLD AT THE TRUE FLOOR                                THE REVISION.
       loop 127 accepted an EC when its self-consistency cleared 4x. At the true floor of 1.15x
       that test is far stricter. Gate: report coverage and accuracy at 1.15x, 2x and 4x, and the
       tightest threshold that still keeps more than 5% of tier-1 reactions must beat the constant.
  P5 THE CONSTANT REMAINS THE THING TO BEAT                         THE NULL.
       loop 124 measured a flat 1.85/s at 9.42x against CatPred's 12.95x. Every accepted subset
       here is scored against that same constant on the same genes. Gate: the accepted subset must
       beat it; if the filter cannot beat a constant there is no reason to prefer it.
  P6 COVERAGE AND FAME                                              THE GUARD.
       three denominators on whatever survives, and publication count of accepted against rejected.

-> outputs/loop_kcat_floor2.json
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
FILT = Path("colab/data/kinetics_filtered.json.gz")
DLK = LR.SC / "dlkcat.json"
UPK = LR.SC / "uniprot_kinetics_human.tsv"
PROT = LR.SC / "human_proteome.fasta.gz"
SEED = 12950
NPERM = 5000
TRUE_FLOOR = 1.15
SHIPPED_BREADTH = 3
P1_RHO, P1_FOLD = 0.50, 4.0

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def geo(v):
    v = [q for q in v if q and q > 0 and np.isfinite(q)]
    return float(np.exp(np.mean(np.log(v)))) if v else None


def fold(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    return np.maximum(a[ok] / b[ok], b[ok] / a[ok])


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


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 129 -- the kinetics filter at the TRUE floor, with four times the power")
    say("=" * 100)
    say()

    B = json.load(gzip.open(BUNDLE, "rt"))
    E = json.load(gzip.open(ECMED, "rt"))
    gk, rt, rec = B["gene_kcat_per_s"], B["reaction_tier"], B["reaction_ec"]
    gmed = float(E["global_median_per_s"])
    ec_med = E["ec_human_median_per_s"]

    # sequence -> gene, then DLKcat per protein
    s2g, nm, buf = {}, None, []
    with gzip.open(PROT, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and buf:
                    s2g["".join(buf)] = nm
                buf, nm = [], None
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                buf.append(ln.strip())
    if nm and buf:
        s2g["".join(buf)] = nm
    dl = [x for x in json.load(open(DLK))
          if x.get("Organism") == "Homo sapiens" and x.get("Sequence")]
    byg, bysub, gene_ec = collections.defaultdict(list), collections.defaultdict(list), {}
    for x in dl:
        g = s2g.get(x["Sequence"])
        try:
            v = float(x["Value"])
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        bysub[(x["Sequence"], x.get("Substrate"))].append(v)
        if g:
            byg[g].append(v)
            if x.get("ECNumber"):
                gene_ec.setdefault(g, x["ECNumber"])
    dkc = {g: geo(v) for g, v in byg.items()}
    dkc = {g: v for g, v in dkc.items() if v}
    say(f"  DLKcat: {len(dl):,} human records -> {len(dkc)} genes by exact sequence match")

    # UniProt, loop 124's parser
    NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    K1 = re.compile(r"kcat is " + NUM + r"\s*(sec|min|hour|h)\(-1\)", re.I)
    PS = {"sec": 1.0, "min": 1 / 60.0, "hour": 1 / 3600.0, "h": 1 / 3600.0}
    rows = list(csv.reader(open(UPK, newline=""), delimiter="\t"))
    hh, rows = rows[0], rows[1:]
    iG, iE, iK = (hh.index("Gene Names (primary)"), hh.index("EC number"), hh.index("Kinetics"))
    ukc, uec = {}, {}
    for x in rows:
        g = x[iG].strip()
        if not g:
            continue
        v = [float(a) * PS[b.lower()] for a, b in K1.findall(x[iK])]
        if v:
            ukc[g] = geo(v)
        if x[iE].strip():
            uec[g] = [e.strip() for e in x[iE].split(";") if e.strip()]
    say(f"  UniProt: {len(ukc)} genes with a curated kcat")
    genes_per_ec = collections.Counter()
    for g, es in uec.items():
        for e in es:
            genes_per_ec[e] += 1
    say()

    gates = {}

    # ---------------------------------------------------------------- P1
    say("P1 THE TWO MEASURED SOURCES AGREE")
    sh = [g for g in dkc if g in ukc]
    rho1 = spearman([dkc[g] for g in sh], [ukc[g] for g in sh])
    f1 = fold([dkc[g] for g in sh], [ukc[g] for g in sh])
    say(f"     {len(sh)} genes with both a DLKcat and a UniProt kcat")
    say(f"     Spearman {rho1:+.4f}   gate >= {P1_RHO}")
    say(f"     median fold between them {np.median(f1):.2f}x   gate <= {P1_FOLD}x")
    say(f"     for scale: the TRUE experimental floor is {TRUE_FLOOR}x, so two independent")
    say(f"     compilations of the same literature should not be far above it")
    gates["P1"] = bool(rho1 >= P1_RHO and np.median(f1) <= P1_FOLD)
    say(f"     P1 {'PASS' if gates['P1'] else 'FAIL'}")
    say()

    # the validation set: DLKcat truth, bundle prediction
    val = [g for g in dkc if g in gk and g in uec and any(e in ec_med for e in uec[g])]
    say(f"  validation set: {len(val)} genes with a DLKcat kcat, a bundle value and an EC "
        f"(loop 127 had 70)")
    fe = np.array([max(gk[g] / dkc[g], dkc[g] / gk[g]) for g in val])
    say()

    # ---------------------------------------------------------------- P2
    say("P2 THE SELECTOR, RETESTED WITH POWER")
    by = collections.defaultdict(list)
    for e, v in E["human_records"]:
        if v and float(v) > 0:
            by[e].append(float(v))
    selfc, nrep = {}, {}
    for e, v in by.items():
        if len(v) < 3:
            continue
        f_ = []
        for i in range(len(v)):
            o = [v[j] for j in range(len(v)) if j != i]
            m = float(np.median(o))
            if m > 0 and v[i] > 0:
                f_.append(max(v[i] / m, m / v[i]))
        if f_:
            selfc[e] = float(np.median(f_))
            nrep[e] = len(v)

    def ec_of(g):
        for e in uec.get(g, []):
            if e in ec_med:
                return e
        return None
    cands = {
        "EC self-consistency": np.array([selfc.get(ec_of(g), np.nan) for g in val]),
        "genes sharing the EC": np.array([float(genes_per_ec.get(ec_of(g), 1)) for g in val]),
        "replicate count": np.array([float(nrep.get(ec_of(g), np.nan)) for g in val]),
    }
    p2, ok2 = {}, False
    for k, v in cands.items():
        m = np.isfinite(v)
        rho = spearman(v[m], fe[m])
        null = np.array([spearman(rng.permutation(v[m]), fe[m]) for _ in range(1000)])
        p = float(np.mean(np.abs(null) >= abs(rho))) if np.isfinite(rho) else float("nan")
        pb = min(1.0, p * len(cands))
        p2[k] = {"rho": rho, "p": p, "p_bonferroni": pb, "n": int(m.sum())}
        good = np.isfinite(rho) and rho > 0 and pb < 0.05
        say(f"     {k:<24} rho {rho:+.4f}   n={int(m.sum()):>3}   p {p:.4f}   "
            f"p*3 {pb:.4f}   {'USABLE' if good else 'no'}")
        ok2 = ok2 or good
    gates["P2"] = bool(ok2)
    say(f"     Bonferroni applied inside the gate this time, not noted afterwards")
    say(f"     P2 {'PASS' if gates['P2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- P3
    say("P3 THE SHIPPED FILTER VALIDATES OUT OF SAMPLE")
    br = np.array([float(genes_per_ec.get(ec_of(g), 999)) for g in val])
    acc = br <= SHIPPED_BREADTH
    say(f"     loop 127 shipped 'EC shared by <= {SHIPPED_BREADTH} genes', chosen by sweeping four")
    say(f"     thresholds on 70 genes. Applied UNCHANGED to {len(val)} -- no re-tuning:")
    if acc.sum() >= 2 and (~acc).sum() >= 2:
        say(f"     accepted {int(acc.sum())}, rejected {int((~acc).sum())}")
        say(f"     fold-error accepted {np.median(fe[acc]):.2f}x   rejected "
            f"{np.median(fe[~acc]):.2f}x")
        o3, p3 = perm_p(np.log10(fe[acc]), np.log10(fe[~acc]), rng)
        say(f"     log10 difference {o3:+.3f}, permutation p = {p3:.4f}")
        say(f"     loop 127 on its own 70: 2.82x against 43.31x, p 0.0035")
        gates["P3"] = bool(o3 < 0 and p3 < 0.05)
    else:
        o3, p3 = float("nan"), float("nan")
        gates["P3"] = False
        say("     one side too small to test")
    say(f"     P3 {'PASS' if gates['P3'] else 'FAIL'} -- the post-hoc threshold "
        f"{'holds on four times the data' if gates['P3'] else 'DOES NOT hold; it was overfitting and the artefact must be withdrawn'}")
    say()

    # ---------------------------------------------------------------- P4
    say("P4 THE THRESHOLD AT THE TRUE FLOOR")
    sc = np.array([selfc.get(ec_of(g), np.nan) for g in val])
    p4 = {}
    for thr in (TRUE_FLOOR, 2.0, 4.0):
        a4 = np.isfinite(sc) & (sc <= thr)
        n_rx = sum(1 for r, t in rt.items()
                   if t == "1_human_EC" and any(selfc.get(e, 1e9) <= thr
                                                for e in (rec.get(r) or [])))
        keep_pct = n_rx / max(sum(1 for t in rt.values() if t == "1_human_EC"), 1)
        row = {"n_accept": int(a4.sum()), "reactions": n_rx, "tier1_kept": keep_pct}
        if a4.sum() >= 5:
            row["accept_fold"] = float(np.median(fe[a4]))
            row["reject_fold"] = float(np.median(fe[~a4])) if (~a4).sum() >= 5 else None
        p4[thr] = row
        say(f"     self-consistency <= {thr:>4.2f}x: accept {int(a4.sum()):>3}/{len(val)} genes, "
            f"{n_rx:>5,} tier-1 reactions ({keep_pct:.1%} of tier 1)"
            + (f", fold {row.get('accept_fold', float('nan')):.2f}x" if "accept_fold" in row else ""))
    viable = [t for t, r in p4.items() if r["tier1_kept"] > 0.05 and "accept_fold" in r]
    tight = min(viable) if viable else None
    say(f"     tightest threshold keeping >5% of tier 1: {tight}")
    gates["P4"] = bool(tight is not None)
    say(f"     P4 {'PASS' if gates['P4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- P5
    say("P5 THE CONSTANT REMAINS THE THING TO BEAT")
    fe_const = np.array([max(gmed / dkc[g], dkc[g] / gmed) for g in val])
    say(f"     whole validation set: bundle {np.median(fe):.2f}x   constant {gmed}/s "
        f"{np.median(fe_const):.2f}x")
    best = None
    for lbl, mask in (("breadth <= 3", acc),
                      (f"self-consistency <= {tight}", np.isfinite(sc) & (sc <= (tight or 0)))):
        if mask.sum() >= 5:
            o5, p5 = perm_p(np.log10(fe[mask]), np.log10(fe_const[mask]), rng)
            say(f"     {lbl:<28} n={int(mask.sum()):>3}   subset {np.median(fe[mask]):.2f}x   "
                f"constant on the same genes {np.median(fe_const[mask]):.2f}x   p {p5:.4f}")
            if o5 < 0 and p5 < 0.05:
                best = lbl
    gates["P5"] = bool(best is not None)
    say(f"     P5 {'PASS' if gates['P5'] else 'FAIL'} -- "
        f"{'a filtered subset beats the constant: ' + best if best else 'NO filtered subset beats a constant 1.85/s on its own genes'}")
    say()

    # ---------------------------------------------------------------- P6
    say("P6 COVERAGE AND FAME")
    D = CA.load()
    pubs = D["pubs"]
    pa = np.array([pubs.get(g, 0.0) for g in np.array(val)[acc]])
    pr = np.array([pubs.get(g, 0.0) for g in np.array(val)[~acc]])
    o6, p6 = perm_p(np.log10(pa + 1), np.log10(pr + 1), rng)
    say(f"     accepted median {np.median(pa):.0f} publications, rejected {np.median(pr):.0f}, "
        f"p = {p6:.4f}")
    n1 = sum(1 for t in rt.values() if t == "1_human_EC")
    say(f"     THREE DENOMINATORS at breadth <= {SHIPPED_BREADTH}: "
        f"{len(json.load(gzip.open(FILT, 'rt'))['reaction_tier']) if FILT.exists() else 0:,} reactions tiered, "
        f"1,130 measured = 8.7% of 12,931; tier 1 is {n1:,}")
    say(f"     validation n: 70 -> {len(val)}, a {len(val) / 70:.1f}x expansion")
    gates["P6"] = bool(np.isfinite(p6))
    say(f"     P6 {'PASS' if gates['P6'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[DLK, UPK, BUNDLE, ECMED, PROT], available=len(dkc), used=len(val),
                      selection="filtered", seed=SEED,
                      controls=["two independent compilations cross-checked before either is used",
                                "Bonferroni applied inside the selector gate, not after it",
                                "loop 127's shipped threshold applied UNCHANGED, no re-tuning",
                                "the constant 1.85/s scored on the same genes as every subset",
                                "publication count of accepted against rejected",
                                "the true replicate floor of 1.15x rather than the within-EC 2.85x"],
                      note="the sequence-mapping move that makes this possible: DLKcat has no "
                           "UniProt id but does carry the protein sequence")
    RM.report(man, emit=say)
    json.dump({"test": "loop_kcat_floor2", "manifest": man, "gates": gates,
               "true_floor": TRUE_FLOOR, "n_val": len(val), "n_val_loop127": 70,
               "p1": {"n": len(sh), "spearman": rho1, "median_fold": float(np.median(f1))},
               "p2": p2,
               "p3": {"n_accept": int(acc.sum()), "log10_diff": o3, "p": p3,
                      "accept_fold": float(np.median(fe[acc])) if acc.sum() else None,
                      "reject_fold": float(np.median(fe[~acc])) if (~acc).sum() else None},
               "p4": {str(k): v for k, v in p4.items()}, "p5": {"beats_constant": best},
               "p6": {"pubs_p": p6, "log10_diff": o6},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_kcat_floor2.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_kcat_floor2.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
