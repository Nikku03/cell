"""THE FIRST VARIANT-ADDRESSABLE SLOT -- and an honest statement of what it does not answer.

WHERE THIS COMES FROM.  capability_audit scored the six question types the goal names. `any point
mutation` was worst on every axis: 13 of 16,492 genes carry residue-level data, 0.08%, no module
addresses it end to end, and nothing scores it against a control. `biomarkers` looks relevant and is
not -- it is keyed on GENE, so you cannot hand it a variant. There is no slot in this model a mutation
can be addressed to.

WHAT IS BUILT.  ClinVar's GRCh38 VCF, joined to the metabolic model, gives one: every missense variant
with a clinical assertion, mapped to a gene that has stoichiometry behind it. That is the ENCODE axis.
The PIPELINE axis needs the variant to reach a cell outcome, and that is where the modelling is.

THE MECHANISM, and it is genuinely different from a knockout.  A missense variant is not a deletion. It
usually leaves a protein that still works, less well. So the quantity that ought to matter is not
"is this gene essential" but HOW SHARPLY growth falls as that enzyme's capacity is reduced -- the
DOSE-RESPONSE. Two genes can both be lethal at zero capacity and behave completely differently at 50%:
one has a flat shoulder and tolerates halving, the other is already failing. Only the second should
accumulate pathogenic missense variants. A knockout screen cannot see that difference; a model with
enzyme capacity in it can, and that difference IS the reason to have built the capacity layer.

    growth(f) for f in 1.00, 0.50, 0.25, 0.10, 0.00 of each gene's enzyme capacity,
    on loop 4's defined medium, with loop 1's capacity constraints, kappa recalibrated for both.

WHAT THIS DOES NOT ANSWER, stated before the results and not after.  It does not predict the effect of
A VARIANT. It has no per-variant feature at all -- no structure, no conservation, no substitution
matrix -- so every mutation in the same gene gets the same number. What it predicts is WHICH GENES
accumulate pathogenic missense variants, which is one necessary component of the question and not the
question. Calling this "answering any point mutation" would be exactly the overstatement this whole
line of work exists to avoid.

PREDECLARED, before any number:

    V1 ENCODE   at least 200 model genes carry both a clinical missense assertion and a dose-response.
                fails -> the join is too thin to test anything and nothing below is reported.
    V2 SIGNAL   dose-response sharpness separates missense-sensitive genes from tolerant ones above the
                95th percentile of 200 label shuffles.
                fails -> the metabolic layer says nothing about mutation sensitivity.
    V3 PARTIAL LOSS IS REAL   sharpness beats the pure knockout score by >= 0.02 AUC.
                This is the whole claim. If knowing what happens at 50% capacity adds nothing over
                knowing what happens at 0%, then the dose-response is decoration and a deletion screen
                was always sufficient.
                fails -> say so; it is the interesting negative.
    V4 BEATS THE CATALOGUE   beats LOEUF, the constraint metric already sitting in cell_complete.
                LOEUF is a strong, honest baseline built from human population data. Losing to it is
                the likely outcome and it does not invalidate V3.

CONTROLS: 200 label shuffles; the knockout score as V3's comparator; LOEUF and mRNA abundance as
independent lookups; benign-only genes as the negative class rather than "everything else".

WHAT HAPPENED, written after the run against the gates above, unedited.

    V1 ENCODE   PASS.  1,659 model genes carry both a clinical missense assertion and a dose-response:
                784 with >= 3 pathogenic missense variants, 875 with benign missense and none
                pathogenic. The slot exists now; it did not this morning.
    V2 SIGNAL   PASS, but barely and it should be read that way.  Best dose-response feature (curve
                area) 0.5513 against a shuffled null of 0.4999 +/- 0.0094, 95th percentile 0.5146.
                Real, and small.
    V3 PARTIAL LOSS IS REAL   FAIL, and this was the whole point.  0.5513 against the pure knockout's
                0.5529 -- a difference of -0.0016 where the gate was +0.02. Knowing what growth does at
                50% of an enzyme's capacity adds NOTHING over knowing what it does at 0%.
    V4 BEATS THE CATALOGUE   FAIL.  LOEUF 0.6393 and mRNA abundance 0.6554 both beat every mechanistic
                feature here. Under cross-validation the dose-response adds +0.0139 on top of LOEUF,
                which is a real but modest increment.

THE NEGATIVE, stated plainly because it was predeclared and it is the finding.  The reasoning behind
this module was that a missense variant causes PARTIAL loss and a deletion causes total loss, so a model
carrying enzyme capacity should see a distinction a knockout screen cannot. It does not. The five-point
dose-response is, for this question, decoration: a deletion screen was always sufficient, and the extra
four linear programs per gene bought -0.0016 AUC.

WHY, most likely, and it is testable rather than asserted.  Flux balance is a linear programme, so
growth as a function of one enzyme's capacity tends to be piecewise linear with a single breakpoint --
either the bound binds or it does not. Real enzyme kinetics are saturating and cooperative, and the
sharp shoulder a missense variant would fall off is a property of the kinetics, not of the
stoichiometry. Building the dose-response on an LP may have been the wrong instrument for the claim
from the start. That is a specific, checkable follow-up: repeat with saturating kinetics rather than
hard bounds and see whether the shoulder appears.

-> outputs/loop_variant.json
"""
import collections
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc, gpr_capacity, GEM, GEMGENES, KDEG, CELL, K_DP, MIN_LINES  # noqa: E402
from loop_deficit import strat_auc, cv_auc  # noqa: E402
from loop_medium import PHYS_SET, NOT_IMPORTABLE_SET, MAX_UPTAKE_CARBONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SCRATCH = Path(os.environ.get("CELL_SCRATCH",
               "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CLINVAR = SCRATCH / "clinvar.vcf.gz"
SEED = 1904
N_SHUFFLE = 200
MU_TARGET = 0.030
FRACTIONS = (1.0, 0.50, 0.25, 0.10, 0.0)
MIN_VARIANTS = 3
GATE_ENCODE = 200
GATE_PARTIAL = 0.02

PATHO = ("Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic")
BENIGN = ("Benign", "Likely_benign", "Benign/Likely_benign")


def read_clinvar(path):
    """Missense variants with a clinical assertion, counted per gene.

    Only MC=missense_variant is kept. A frameshift or nonsense variant is a knockout by another name,
    and counting those would let the deletion signal in through the back door and quietly answer V3
    in the affirmative for the wrong reason."""
    gene_re = re.compile(r"GENEINFO=([^;]+)")
    sig_re = re.compile(r"CLNSIG=([^;]+)")
    mc_re = re.compile(r"MC=([^;]+)")
    P = collections.Counter()
    B = collections.Counter()
    n = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            n += 1
            info = line.rsplit("\t", 1)[-1]
            m = mc_re.search(info)
            if not m or "missense_variant" not in m.group(1):
                continue
            g = gene_re.search(info)
            s = sig_re.search(info)
            if not g or not s:
                continue
            sym = g.group(1).split("|")[0].split(":")[0]
            sig = s.group(1)
            if sig in PATHO:
                P[sym] += 1
            elif sig in BENIGN:
                B[sym] += 1
    return P, B, n


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE FIRST VARIANT-ADDRESSABLE SLOT")
    say("=" * 100)
    say("  capability_audit scored 'any point mutation' worst on every axis: 13 of 16,492 genes with")
    say("  residue-level data, no pipeline, no control. This builds the slot and tests one specific")
    say("  mechanistic claim -- that PARTIAL loss of enzyme capacity, not deletion, is what a missense")
    say("  variant does, and that the difference is measurable.")

    t0 = time.time()
    P, B, nrec = read_clinvar(CLINVAR)
    say(f"\n  ClinVar GRCh38 VCF: {nrec:,} records, {len(P):,} genes with pathogenic missense and "
        f"{len(B):,} with benign missense ({time.time()-t0:.0f}s)")

    import cobra  # noqa: F401
    from cobra.io import read_sbml_model
    M = read_sbml_model(str(GEM))
    M.solver = "glpk"
    bio = str(M.objective.expression).split("*")[1].split(" ")[0]

    # ---- loop 4's medium, rebuilt from the same declared rules ---------------------------------------
    def carbons(met):
        f = met.formula
        if not f:
            return None
        g = re.match(r"C(\d*)", f)
        return int(g.group(1) or 1) if g else 0

    keep = set()
    for r in M.exchanges:
        met = list(r.metabolites)[0]
        nm = (met.name or "").strip().lower()
        if nm in PHYS_SET:
            keep.add(r.id)
    for rid in {"gamma-tocopherol", "lipoic acid"}:
        for r in M.exchanges:
            if (list(r.metabolites)[0].name or "").strip().lower() == rid:
                keep.add(r.id)
    for r in M.exchanges:
        r.lower_bound = 0.0
    for rid in keep:
        M.reactions.get_by_id(rid).lower_bound = -1000.0
    say(f"  medium rebuilt from loop 4's rules: {len(keep)} uptakes "
        f"(the dish, plus gamma-tocopherol and lipoic acid, the two things the network cannot make)")

    # ---- loop 1's capacity layer, on top ------------------------------------------------------------
    gm = pd.read_csv(GEMGENES, sep="\t")
    ens2sym = {a: b for a, b in zip(gm["genes"], gm["geneSymbols"]) if isinstance(b, str)}
    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    kd = k.groupby("feature_ID").agg(kdeg=("avg_kdeg", "median"), n=("cell_line", "size"))
    rp, kd = rp[rp.n >= MIN_LINES], kd[kd.n >= MIN_LINES]
    med_r, med_d = float(rp.rpkm.median()), float(kd.kdeg.median())
    E0 = {}
    for g in M.genes:
        s = ens2sym.get(g.id)
        r = float(rp.rpkm[s]) if s in rp.index else med_r
        d = float(kd.kdeg[s]) if s in kd.index else med_d
        E0[g.id] = (r * (d + MU_TARGET)) / ((d + MU_TARGET) * (K_DP + MU_TARGET))
    gpr = {r.id: r.gpr for r in M.reactions if r.gene_reaction_rule.strip()}
    RX = [M.reactions.get_by_id(rid) for rid in gpr]
    ORIG = {r.id: r.bounds for r in M.reactions}
    default = float(np.median(list(E0.values())))

    def apply_cap(kappa, E):
        for r in RX:
            c = gpr_capacity(gpr[r.id], E, default)
            if c is None:
                continue
            b = kappa * c
            lo, hi = ORIG[r.id]
            r.bounds = (max(lo, -b), min(hi, b))

    lo_k, hi_k = 1e-12, 1e6
    for _ in range(80):
        mid = np.sqrt(lo_k * hi_k)
        apply_cap(mid, E0)
        v = M.slim_optimize()
        if v is None or not np.isfinite(v) or v < MU_TARGET:
            lo_k = mid
        else:
            hi_k = mid
        if hi_k / lo_k < 1.0001:
            break
    KAPPA = float(np.sqrt(lo_k * hi_k))
    apply_cap(KAPPA, E0)
    MU_WT = float(M.slim_optimize() or 0.0)
    CAPPED = {r.id: r.bounds for r in RX}
    say(f"  capacity layer on top: kappa {KAPPA:.4g}, wild-type growth {MU_WT:.5f}/h "
        f"(medium AND enzymes both binding)")

    # ---- the dose-response ---------------------------------------------------------------------------
    gids = [g.id for g in M.genes]
    say(f"\n  DOSE-RESPONSE: growth at {FRACTIONS} of each gene's capacity, {len(gids)} genes")
    # CACHE. The dose-response costs 12 minutes and the first run of this module threw it away on a
    # name collision in the scoring code below -- cobra registers a pandas accessor called `knockout`,
    # so T.knockout resolved to cobra's accessor instead of the column. Expensive state that a cheap
    # downstream bug can destroy should not be recomputed on principle.
    CACHE = SCRATCH / f"_dose_{KAPPA:.6g}_{len(keep)}.json"
    if CACHE.exists():
        dose = {k: v for k, v in json.load(open(CACHE)).items()}
        say(f"    reusing cached dose-response from {CACHE.name}")
        gids = [g for g in gids if g in dose]
    else:
        dose = {}
    t1 = time.time()
    for n, gid in enumerate(list(gids) if not dose else []):
        g = M.genes.get_by_id(gid)
        touched = [r for r in g.reactions if r.id in gpr]
        row = []
        for f in FRACTIONS:
            E = dict(E0)
            E[gid] = E0[gid] * f
            for r in touched:
                c = gpr_capacity(gpr[r.id], E, default)
                if c is None:
                    continue
                b = KAPPA * c
                lo0, hi0 = ORIG[r.id]
                r.bounds = (max(lo0, -b), min(hi0, b))
            v = M.slim_optimize()
            row.append(0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0)))
            for r in touched:
                r.bounds = CAPPED[r.id]
        dose[gid] = row
        if n % 500 == 0:
            say(f"      {n}/{len(gids)}  ({time.time()-t1:.0f}s)")
    if not CACHE.exists():
        json.dump(dose, open(CACHE, "w"))
    say(f"    done in {time.time()-t1:.0f}s ({len(dose)} genes)")

    # ---- assemble ------------------------------------------------------------------------------------
    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    rows = []
    for gid in gids:
        s = ens2sym.get(gid)
        if s is None:
            continue
        p, b = P.get(s, 0), B.get(s, 0)
        if p < MIN_VARIANTS and b < MIN_VARIANTS:
            continue
        d = dose[gid]
        base = d[0] if d[0] > 0 else MU_WT
        rel = [x / base for x in d]
        i = name2i.get(s)
        rows.append({
            "gene": gid, "sym": s, "n_patho": p, "n_benign": b,
            # SHARPNESS: how much is lost by the time capacity is halved. A gene with a flat shoulder
            # scores 0 here however lethal it is at zero, which is exactly the distinction being tested.
            "drop_at_50": 1.0 - rel[1], "drop_at_25": 1.0 - rel[2], "drop_at_10": 1.0 - rel[3],
            "knockout": 1.0 - rel[4],
            "auc_curve": float(np.trapezoid(rel, dx=1.0)) if hasattr(np, "trapezoid")
                         else float(np.trapz(rel, dx=1.0)),
            "loeuf": (genes[i].get("loeuf") if i is not None else None),
            "rpkm": float(rp.rpkm[s]) if s in rp.index else np.nan,
        })
    T = pd.DataFrame(rows)
    # missense-sensitive vs tolerant. The negative class is genes with benign missense variants and NO
    # pathogenic ones -- a real comparator, not "every gene we did not call positive".
    T["y"] = np.where(T.n_patho >= MIN_VARIANTS, 1, np.where(T.n_patho == 0, 0, -1))
    T = T[T.y >= 0].reset_index(drop=True)
    y = T.y.to_numpy()
    say(f"\n  SCORING SET: {len(T)} model genes -- {int(y.sum())} with >= {MIN_VARIANTS} pathogenic "
        f"missense variants, {len(y)-int(y.sum())} with benign missense and none pathogenic")
    v1 = bool(len(T) >= GATE_ENCODE)
    say(f"  V1 ENCODE (>= {GATE_ENCODE} genes)   {'PASS' if v1 else 'FAIL'}")
    if not v1:
        say("    The join is too thin to test anything. Nothing below is reported.")
        json.dump({"test": "loop_variant", "V1": False, "n": len(T), "log": log},
                  open(OUT / "loop_variant.json", "w"), indent=2)
        return 1

    A = {"dose-response at 50%": auc(T["drop_at_50"], y),
         "dose-response at 25%": auc(T["drop_at_25"], y),
         "dose-response at 10%": auc(T["drop_at_10"], y),
         "curve area": auc(-T["auc_curve"], y),
         "knockout only": auc(T["knockout"], y),
         "LOEUF (lookup)": auc(-T["loeuf"].fillna(T["loeuf"].median()), y),
         "mRNA abundance (lookup)": auc(T["rpkm"].fillna(0), y)}
    say(f"\n  {'predictor':<28}{'AUC':>8}")
    for kk, vv in sorted(A.items(), key=lambda x: -(x[1] if np.isfinite(x[1]) else 0)):
        say(f"  {kk:<28}{vv:>8.4f}")

    best_dose = max(("dose-response at 50%", "dose-response at 25%", "dose-response at 10%",
                     "curve area"), key=lambda k: A[k])
    null = np.array([auc(T[{"dose-response at 50%": "drop_at_50", "dose-response at 25%": "drop_at_25",
                            "dose-response at 10%": "drop_at_10", "curve area": "auc_curve"}[best_dose]],
                         rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    v2 = bool(A[best_dose] > p95)
    say(f"\n  V2 SIGNAL -- {N_SHUFFLE} shuffles: null {null.mean():.4f} +/- {null.std():.4f}, "
        f"95th {p95:.4f}")
    say(f"    best dose-response ({best_dose}) {A[best_dose]:.4f}   V2 {'PASS' if v2 else 'FAIL'}")

    d3 = A[best_dose] - A["knockout only"]
    v3 = bool(d3 >= GATE_PARTIAL)
    say(f"\n  V3 IS PARTIAL LOSS REAL -- dose-response minus pure knockout")
    say(f"    {A[best_dose]:.4f} - {A['knockout only']:.4f} = {d3:+.4f}   (gate +{GATE_PARTIAL})   "
        f"V3 {'PASS' if v3 else 'FAIL'}")
    if not v3:
        say("    Knowing what happens at 50% capacity adds nothing over knowing what happens at 0%.")
        say("    The dose-response is decoration for this question and a deletion screen was always")
        say("    sufficient. That is the interesting negative and it is the predeclared one.")

    v4 = bool(A[best_dose] > A["LOEUF (lookup)"])
    say(f"\n  V4 BEATS THE CATALOGUE -- vs LOEUF {A['LOEUF (lookup)']:.4f}   "
        f"{'PASS' if v4 else 'FAIL'}")
    a_look, _ = cv_auc(np.c_[-T["loeuf"].fillna(T["loeuf"].median())], y)
    a_both, _ = cv_auc(np.c_[-T["loeuf"].fillna(T["loeuf"].median()), T["drop_at_50"], T["drop_at_25"],
                             T["knockout"]], y)
    say(f"    5-fold CV: LOEUF {a_look:.4f} -> LOEUF + dose-response {a_both:.4f} ({a_both-a_look:+.4f})")

    say("\n" + "=" * 100)
    gates = {"V1 encode": v1, "V2 signal": v2, "V3 partial loss is real": v3,
             "V4 beats the catalogue": v4}
    for kk, vv in gates.items():
        say(f"  {kk:<32}{'PASS' if vv else 'FAIL'}")
    say("  AND THE LIMIT, restated: this predicts which GENES accumulate pathogenic missense variants.")
    say("  It has no per-variant feature, so every mutation in a gene gets the same number. It is one")
    say("  necessary component of 'any point mutation', not an answer to it.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CLINVAR), str(GEM), str(GEMGENES), str(KDEG), str(CELL)],
                      available=len(gids), used=len(T), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "knockout-only score (V3)",
                                "LOEUF and mRNA lookups (V4)",
                                "benign-only genes as the negative class",
                                "5-fold cross-validation"],
                      note="missense variants only; frameshift and nonsense excluded so the deletion "
                           "signal cannot answer V3 through the back door")
    RM.report(man, emit=say)
    json.dump({"test": "loop_variant", "manifest": man, "gates": gates, "auc": A,
               "kappa": KAPPA, "mu_wt": MU_WT, "fractions": list(FRACTIONS),
               "n_uptakes": len(keep), "n_scored": len(T), "n_positive": int(y.sum()),
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "cv": {"loeuf": float(a_look), "loeuf_plus_dose": float(a_both)},
               "scores": T.to_dict("records"), "log": log},
              open(OUT / "loop_variant.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_variant.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
