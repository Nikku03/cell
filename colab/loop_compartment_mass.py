"""LOOP 102 -- COMPARTMENTS: DOES THE PREDICTED PROTEIN MASS PER COMPARTMENT MATCH MEASUREMENT?

THE CLAIM THIS TESTS. cell_complete.json gives every one of its 16,492 genes a `comp` -- one of
exactly 12 buckets (ER, Golgi, cytoplasm, cytoskeleton, endosome, extracellular, lysosome, membrane,
mitochondrion, nucleus, peroxisome, plasma membrane). That is an ANNOTATION, a GO cellular-component
term collapsed to a single bucket per gene, and it has never been asked to reproduce a physical
quantity. A whole-cell model needs compartments to hold the right amount of stuff: an ER that is 1%
of the protein mass cannot fold the secretome, and a mitochondrion that is 40% of it cannot fit.
Organelle protein mass fractions are one of the few cell-biological numbers that were measured
BEFORE genomics, by subcellular fractionation and stereology, so there is an external answer this
model has never been checked against.

HOW MASS IS FORMED, AND WHY IT IS NOT A RATE. PaxDb abundance is MOLAR ppm -- parts per million of
protein MOLECULES -- so a molar profile is not a mass profile: ribosomal proteins are numerous and
small, titin is one molecule and enormous. Mass needs the molecular weight, computed here from the
UniProt sequence, residue by residue:

        mass_i  =  ppm_i (HeLa, PaxDb)  x  MW_i (UniProt sequence)
        f_c     =  sum of mass_i over genes assigned to c  /  sum over all covered genes

LOOP 92's ABUNDANCE RULE IS OBEYED, AND THE REASON IS WORTH STATING. Nothing here is a quotient of
two abundance datasets. ppm and MW are different KINDS of measurement -- a molar concentration and a
sequence property -- and the sequence is the same molecule in every cell, so multiplying them does
not smuggle in a cell-type difference the way loop 92's HeLa-protein-over-NIH3T3-mRNA did. The one
place a second proteome appears is M4, where the WHOLE profile is recomputed from Schwanhausser
NIH3T3 copies as an independent replicate, never mixed into the HeLa numbers.

THE PUBLISHED REFERENCE, FIXED HERE BEFORE ANY NUMBER WAS COMPUTED. Mammalian somatic cell, protein
mass fraction: cytosol ~0.50 and dominant, mitochondrion ~0.15, nucleus ~0.125, ER ~0.10. These are
the classical fractionation/morphometry values (rat hepatocyte and cultured-cell ranges: cytosol
0.45-0.55, mitochondria 0.15-0.20, nucleus 0.10-0.15, ER 0.08-0.12) and the tolerance everywhere is
a FACTOR OF 2, declared now, not after seeing anything.

TWO BOOKKEEPING FACTS THAT WILL BITE, DECLARED IN ADVANCE:
  - the model's cytosol bucket is called `cytoplasm`, and `cytoskeleton` is a SEPARATE bucket even
    though classical fractionation counts cytoskeletal protein inside the cytosolic fraction. Both
    the strict `cytoplasm` figure and the merged cytoplasm+cytoskeleton figure are reported; the
    GATE is on the strict bucket, because that is what the model actually stores.
  - `extracellular`, `membrane` and `plasma membrane` are model buckets with no clean classical
    counterpart. They are reported and are NOT gated against a published number, because inventing
    a reference for them after the fact is exactly the move this project forbids.

DISCLOSURE OF WHAT WAS LOOKED AT FIRST. Before this docstring was written I ran three STRUCTURAL
checks and no statistic: the 12 bucket names and their gene counts (nucleus 4,007 the largest,
peroxisome 62 the smallest), that PaxDb HeLa covers 7,222 symbols of which 6,463 are model genes,
and that the FASTA carries a GN= field. No mass fraction, no correlation and no gate number was
computed before the gates below were fixed.

PREDECLARED, before any number:

  M1 THE JOIN IS REAL, AND ITS COVERAGE IS DECLARED               THE STRUCTURAL GATE.
       >= 50% of HeLa-measured symbols must receive a protein MW from the UniProt FASTA, AND every
       one of the 12 compartments must retain >= 20 genes carrying both abundance and MW. A
       compartment below that is not a measurement of that compartment and its fraction is reported
       as unsupported rather than quoted.
  M2 THE ORDERING IS RIGHT                                        THE GATE THE TASK ASKS FOR.
       `cytoplasm` must be the single largest compartment by protein mass. If an annotation-derived
       assignment cannot put the cytosol first it cannot be used to size anything.
  M3 THE MITOCHONDRION IS THE RIGHT SIZE                          THE QUANTITATIVE GATE.
       mitochondrial mass fraction within a factor of 2 of 0.15, i.e. in [0.075, 0.300]. Outside
       that window this loop reports a FAIL and the direction of the miss; it does not widen the
       window.
  M4 THE WHOLE PROFILE, NOT ONE LUCKY NUMBER                      THE HARDER GATE.
       of the four compartments with a published value -- cytoplasm 0.50, mitochondrion 0.15,
       nucleus 0.125, ER 0.10 -- at least 3 of 4 must sit within a factor of 2. Reported alongside:
       the same four recomputed from a completely different proteome (Schwanhausser NIH3T3 copy
       numbers), because a profile that only exists in one cell line is a cell line, not a cell.
  M5 THE FAME CONTROL, RUN AS A RIVAL MODEL                       THE RECURRING KILLER.
       `pubs` has beaten the biology in loops 87, 87b, 94 and 96, so it is not enough to report a
       correlation -- it is given its own turn at the problem. A rival profile is built by replacing
       abundance with publication count (pubs_i x MW_i) and scored against the SAME published
       reference by the same distance. PASS requires the measured-abundance profile to be CLOSER to
       the published values than the publication-count profile. If fame reproduces organelle mass
       fractions as well as mass spectrometry does, the profile is a map of attention and this gate
       says so. Per-compartment median pubs and per-compartment abundance coverage are reported
       whatever the verdict.
  M6 THE NULL THAT COULD FIRE, AND IS CHECKED FOR THE ABILITY TO  THE GUARD, GUARDED.
       compartment labels are shuffled across the covered genes 200 times. The statistic is the
       Kullback-Leibler divergence of the MASS fractions from the GENE-COUNT fractions: if
       compartment membership carries mass information, mass is concentrated away from headcount and
       KL is large; under a shuffle every compartment gets an average gene and the distribution
       FLATTENS onto its headcount, so KL -> 0. gate_guard.null_can_move() confirms the shuffle
       actually changes the label vector BEFORE its verdict is used, and gate_guard.survival()
       decides whether the comparison is even defined. Twelve gates in this project fired while
       measuring nothing.
  M7 WHAT THIS DOES NOT GIVE                                      THE HONEST LIMIT.
       one bucket per gene is a lie about biology -- most proteins occupy several compartments and
       many move between them -- and PaxDb HeLa covers only part of the proteome, unevenly. Best-
       and worst-covered compartments are reported, together with a coverage-corrected profile as a
       sensitivity, so nobody reads a bucket count as a localisation model.

POST-HOC ADDITIONS, DISCLOSED. M2, M3 and M4 failed on the first run. Three DIAGNOSTICS were then
added, all inside M7, none of which can change a gate, a threshold or a reference value:
  (i)   the mass concentration -- what share of the total sits in the 20 heaviest contributors, and
        whether the ordering survives deleting them, because ppm x MW has a very heavy tail;
  (ii)  an isoform-choice sensitivity -- the profile recomputed with the SHORTEST UniProt isoform per
        symbol instead of the longest, since the longest is the choice this module made and it
        favours giant scaffolds;
  (iii) the replicate proteome scored by M4's own predeclared rule, so that the same test applied to
        a different cell is visible rather than described.
No gate was re-run after these, no window was widened, and the failures below stand as failures.

-> outputs/loop_compartment_mass.json
"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402
import cell_proteome as CP  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
FASTA = SC / "human_proteome.fasta.gz"

SEED = 10201
NPERM = 200

CYTOSOL_KEY = "cytoplasm"                 # the model's name for the cytosolic bucket
MERGE_INTO_CYTOSOL = ("cytoplasm", "cytoskeleton")

# PUBLISHED MAMMALIAN ORGANELLE PROTEIN MASS FRACTIONS -- fixed before any number was computed.
REFERENCE = {"cytoplasm": 0.50, "mitochondrion": 0.15, "nucleus": 0.125, "ER": 0.10}
FACTOR = 2.0                              # the tolerance, declared once, applied everywhere
MIN_GENES_PER_COMP = 20                   # M1
MIN_MW_COVERAGE = 0.50                    # M1
MIN_WITHIN = 3                            # M4: at least 3 of the 4 referenced compartments

# average residue masses, Da (UniProt/ExPASy), plus water for the free termini
AA = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388, "E": 129.1155,
      "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
      "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051, "W": 186.2132,
      "Y": 163.1760, "V": 99.1326, "U": 150.0388, "O": 237.3018, "X": 110.0000, "B": 114.5962,
      "Z": 128.6231}
WATER = 18.0153

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b, nmin=8):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < nmin:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def load_mw():
    """{symbol: MW in Da} from the UniProt FASTA. Returns the longest AND the shortest isoform per
    symbol; the longest is what the profile uses and the shortest is M7's sensitivity."""
    hi, lo, seq, gene = {}, {}, [], None
    def flush():
        if gene and seq:
            m = sum(AA.get(c, AA["X"]) for c in "".join(seq)) + WATER
            if m > hi.get(gene, 0.0):
                hi[gene] = m
            if m < lo.get(gene, float("inf")):
                lo[gene] = m
    with gzip.open(FASTA, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                flush()
                g = re.search(r"\bGN=(\S+)", ln)
                gene, seq = (g.group(1) if g else None), []
            else:
                seq.append(ln.strip())
    flush()
    return hi, lo


def profile(rows, weight_key):
    """{compartment: mass fraction} over `rows`, weighting each gene by weight_key x MW."""
    tot, per = 0.0, {}
    for r in rows:
        w = r[weight_key] * r["mw"]
        per[r["comp"]] = per.get(r["comp"], 0.0) + w
        tot += w
    if tot <= 0:
        return {c: float("nan") for c in per}
    return {c: v / tot for c, v in per.items()}


def counts(rows):
    per = {}
    for r in rows:
        per[r["comp"]] = per.get(r["comp"], 0) + 1
    n = sum(per.values())
    return {c: v / n for c, v in per.items()}


def kl(p, q):
    """KL(mass || headcount) over the compartments present in both."""
    s = 0.0
    for c, pv in p.items():
        qv = q.get(c, 0.0)
        if pv > 0 and qv > 0:
            s += pv * np.log(pv / qv)
    return float(s)


def entropy(p):
    return float(-sum(v * np.log(v) for v in p.values() if v > 0))


def dist_to_reference(prof):
    """Mean |log2(observed / published)| over the four referenced compartments. Lower is better."""
    d = []
    for c, ref in REFERENCE.items():
        v = prof.get(c, 0.0)
        d.append(abs(np.log2(v / ref)) if v > 0 else float("inf"))
    return float(np.mean(d))


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 102 -- compartments: does the predicted protein mass per compartment match "
        "measurement?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    genes = C["genes"]
    comps = list(C["comps"])
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in genes}
    comp_of = {g["name"]: g.get("comp") for g in genes}
    ppm = CP.hela_ppm()
    mw, mw_short = load_mw()
    say(f"  model genes {len(genes):,}   compartments {len(comps)}   "
        f"PaxDb HeLa symbols {len(ppm):,}   UniProt MWs {len(mw):,}")
    say(f"  reference (fixed in the docstring): " +
        "  ".join(f"{c} {v:.3f}" for c, v in REFERENCE.items()) + f"   tolerance x{FACTOR:.0f}")
    say()

    # ------------------------------------------------------------------ M1
    say("M1 THE JOIN IS REAL, AND ITS COVERAGE IS DECLARED")
    measured = [g["name"] for g in genes if g["name"] in ppm]
    with_mw = [n for n in measured if n in mw]
    mw_cov = len(with_mw) / max(len(measured), 1)
    rows = [{"name": n, "comp": comp_of[n], "ppm": ppm[n], "mw": mw[n], "pubs": pubs.get(n, 0.0)}
            for n in with_mw if comp_of.get(n)]
    per_n = {}
    for r in rows:
        per_n[r["comp"]] = per_n.get(r["comp"], 0) + 1
    thin = [c for c in comps if per_n.get(c, 0) < MIN_GENES_PER_COMP]
    say(f"     model genes with a HeLa ppm: {len(measured):,} of {len(genes):,} "
        f"({len(measured)/len(genes):.1%})")
    say(f"     of those, carrying a UniProt MW: {len(with_mw):,} ({mw_cov:.1%})  "
        f"[gate: >= {MIN_MW_COVERAGE:.0%}]")
    say(f"     genes entering the mass profile: {len(rows):,}")
    say(f"     compartments with < {MIN_GENES_PER_COMP} such genes: "
        f"{', '.join(thin) if thin else 'none'}")
    m1 = bool(mw_cov >= MIN_MW_COVERAGE and not thin)
    say(f"     M1 {'PASS' if m1 else 'FAIL'} -- "
        f"{'every compartment is supported and the sequence join holds' if m1 else 'the join or a compartment is too thin to be a measurement'}")
    say()

    # ------------------------------------------------------------------ the profile
    prof = profile(rows, "ppm")
    cnt = counts(rows)
    molar = {}
    tot_m = sum(r["ppm"] for r in rows)
    for r in rows:
        molar[r["comp"]] = molar.get(r["comp"], 0.0) + r["ppm"] / tot_m
    all_comp_cov = {c: per_n.get(c, 0) / max(sum(1 for g in genes if g.get("comp") == c), 1)
                    for c in comps}
    order = sorted(prof, key=lambda c: -prof[c])

    say("    THE PROFILE  (mass fraction = sum ppm x MW, normalised over covered genes)")
    say(f"    {'compartment':<16}{'mass':>9}{'molar':>9}{'genes':>8}{'headcount':>11}"
        f"{'cover':>8}{'medpubs':>9}   published")
    for c in order:
        ref = REFERENCE.get(c)
        pb = np.median([r["pubs"] for r in rows if r["comp"] == c])
        say(f"    {c:<16}{prof[c]:>9.4f}{molar.get(c,0):>9.4f}{per_n.get(c,0):>8,}"
            f"{cnt.get(c,0):>11.4f}{all_comp_cov[c]:>8.1%}{pb:>9.0f}   "
            f"{(f'{ref:.3f}' if ref is not None else '--')}")
    merged = sum(prof.get(c, 0.0) for c in MERGE_INTO_CYTOSOL)
    say(f"     merged cytoplasm+cytoskeleton (classical 'cytosolic fraction'): {merged:.4f}")
    say()

    # ------------------------------------------------------------------ M2
    say("M2 THE ORDERING IS RIGHT")
    say(f"     largest compartment by protein mass: {order[0]} at {prof[order[0]]:.4f}")
    say(f"     runner-up: {order[1]} at {prof[order[1]]:.4f}")
    m2 = bool(order[0] == CYTOSOL_KEY)
    say(f"     M2 {'PASS' if m2 else 'FAIL'} -- cytosol "
        f"{'is the largest compartment, as it must be' if m2 else f'is NOT largest ({order[0]} is)'}")
    top_by_comp = {}
    for c in order:
        rs = sorted([r for r in rows if r["comp"] == c], key=lambda r: -r["ppm"] * r["mw"])[:5]
        top_by_comp[c] = [r["name"] for r in rs]
    say(f"     sanity, the heaviest contributors per compartment:")
    for c in order[:5]:
        say(f"       {c:<16}{', '.join(top_by_comp[c])}")
    say()

    # ------------------------------------------------------------------ M3
    say("M3 THE MITOCHONDRION IS THE RIGHT SIZE")
    fm = prof.get("mitochondrion", 0.0)
    lo, hi = REFERENCE["mitochondrion"] / FACTOR, REFERENCE["mitochondrion"] * FACTOR
    m3 = bool(lo <= fm <= hi)
    say(f"     mitochondrial protein mass fraction: {fm:.4f}   published {REFERENCE['mitochondrion']:.3f}"
        f"   window [{lo:.4f}, {hi:.4f}]")
    say(f"     ratio observed/published: {fm/REFERENCE['mitochondrion']:.2f}x  "
        f"(log2 {np.log2(fm/REFERENCE['mitochondrion']) if fm > 0 else float('nan'):+.2f})")
    say(f"     M3 {'PASS' if m3 else 'FAIL'} -- "
        f"{'inside a factor of 2' if m3 else ('TOO LOW' if fm < lo else 'TOO HIGH')}")
    say()

    # ------------------------------------------------------------------ M4
    say("M4 THE WHOLE PROFILE, NOT ONE LUCKY NUMBER")
    within = {}
    for c, ref in REFERENCE.items():
        v = prof.get(c, 0.0)
        within[c] = bool(ref / FACTOR <= v <= ref * FACTOR)
        say(f"     {c:<16}observed {v:.4f}   published {ref:.3f}   "
            f"{v/ref if ref else float('nan'):>5.2f}x   {'within' if within[c] else 'OUTSIDE'}")
    n_within = sum(within.values())
    d_real = dist_to_reference(prof)
    say(f"     within a factor of {FACTOR:.0f}: {n_within} of {len(REFERENCE)}   "
        f"[gate: >= {MIN_WITHIN}]")
    say(f"     mean |log2(obs/published)| over the four: {d_real:.3f}")

    # independent proteome replicate -- NOT mixed into the HeLa numbers
    S = json.load(open(SC / "_schwan2011.json"))
    srows = [{"name": g, "comp": comp_of[g], "ppm": float(v["prot_copies"]), "mw": mw[g],
              "pubs": pubs.get(g, 0.0)}
             for g, v in S.items()
             if v.get("prot_copies") and g in mw and comp_of.get(g)]
    sprof = profile(srows, "ppm") if srows else {}
    sorder = sorted(sprof, key=lambda c: -sprof[c]) if sprof else []
    d_sch = dist_to_reference(sprof) if sprof else float("nan")
    say(f"     REPLICATE in a different proteome (Schwanhausser NIH3T3 copies, {len(srows):,} genes):")
    if sprof:
        say(f"       largest {sorder[0]} {sprof[sorder[0]]:.4f}   "
            + "   ".join(f"{c} {sprof.get(c,0):.4f}" for c in REFERENCE))
        say(f"       mean |log2(obs/published)|: {d_sch:.3f}   "
            f"cytosol-first: {sorder[0] == CYTOSOL_KEY}")
        # the SAME predeclared rule, applied to the replicate. Reported, not gated: the gate is on
        # HeLa, and scoring a second dataset would be choosing whichever answer one preferred.
        swithin = {c: bool(REFERENCE[c] / FACTOR <= sprof.get(c, 0.0) <= REFERENCE[c] * FACTOR)
                   for c in REFERENCE}
        s_nwithin = sum(swithin.values())
        say(f"       by M4's own rule the replicate scores {s_nwithin} of {len(REFERENCE)} within a "
            f"factor of {FACTOR:.0f} ({'would PASS' if s_nwithin >= MIN_WITHIN else 'would FAIL'}), "
            f"against {n_within} of {len(REFERENCE)} in HeLa")
        say(f"       the ANNOTATION is identical in both, so the difference is the ABUNDANCE DATA")
        r_two, n_two = spear([prof.get(c, 0.0) for c in comps], [sprof.get(c, 0.0) for c in comps])
        say(f"       HeLa vs NIH3T3 compartment profiles: rho {r_two:+.4f} over {n_two} compartments")
    else:
        r_two, n_two, swithin, s_nwithin = float("nan"), 0, {}, 0
        say(f"       no overlap -- the replicate could not be built")
    m4 = bool(n_within >= MIN_WITHIN)
    say(f"     M4 {'PASS' if m4 else 'FAIL'} -- "
        f"{'the profile as a whole matches measurement' if m4 else 'the profile as a whole does NOT match measurement'}")
    say()

    # ------------------------------------------------------------------ M5
    say("M5 THE FAME CONTROL, RUN AS A RIVAL MODEL")
    fame = profile(rows, "pubs")
    d_fame = dist_to_reference(fame)
    forder = sorted(fame, key=lambda c: -fame[c])
    r_gene, n_gene = spear([r["ppm"] for r in rows], [r["pubs"] for r in rows], nmin=30)
    medpubs = {c: float(np.median([r["pubs"] for r in rows if r["comp"] == c])) for c in comps
               if per_n.get(c, 0)}
    cl = [c for c in comps if c in prof and c in medpubs]
    r_cp, n_cp = spear([prof[c] for c in cl], [medpubs[c] for c in cl])
    r_cc, _ = spear([prof[c] for c in cl], [cnt.get(c, 0.0) for c in cl])
    r_cov, _ = spear([all_comp_cov[c] for c in cl], [medpubs[c] for c in cl])
    say(f"     gene level: ppm vs pubs  rho {r_gene:+.4f}  (n {n_gene:,})")
    say(f"     compartment level: mass fraction vs median pubs  rho {r_cp:+.4f}  (n {n_cp})")
    say(f"     compartment level: mass fraction vs headcount    rho {r_cc:+.4f}")
    say(f"     compartment level: HeLa coverage vs median pubs  rho {r_cov:+.4f}   "
        f"-- does the abundance dataset prefer studied compartments?")
    say(f"     RIVAL PROFILE built from publication count instead of abundance:")
    say(f"       largest {forder[0]} {fame[forder[0]]:.4f}   "
        + "   ".join(f"{c} {fame.get(c,0):.4f}" for c in REFERENCE))
    say(f"       distance to published: fame {d_fame:.3f}  vs  measured abundance {d_real:.3f}")
    m5 = bool(d_real < d_fame)
    say(f"     M5 {'PASS' if m5 else 'FAIL'} -- "
        f"{'measured abundance beats publication count at reproducing the published profile' if m5 else 'PUBLICATION COUNT MATCHES THE PUBLISHED PROFILE AS WELL AS OR BETTER THAN MASS SPECTROMETRY -- this profile is a map of attention'}")
    say()

    # ------------------------------------------------------------------ M6
    say("M6 THE NULL THAT COULD FIRE, AND IS CHECKED FOR THE ABILITY TO")
    rng = np.random.default_rng(SEED)
    labels = [r["comp"] for r in rows]
    real_kl = kl(prof, cnt)
    real_H = entropy(prof)
    real_max = max(prof.values())
    nulls, nH, nmax, moved = [], [], [], []
    for _ in range(NPERM):
        perm = rng.permutation(len(rows))
        shuf = [labels[i] for i in perm]
        moved.append(GG.null_can_move(labels, shuf)["changed"])
        srows2 = [{"comp": shuf[i], "ppm": rows[i]["ppm"], "mw": rows[i]["mw"]}
                  for i in range(len(rows))]
        p2 = profile(srows2, "ppm")
        c2 = counts(srows2)
        nulls.append(kl(p2, c2))
        nH.append(entropy(p2))
        nmax.append(max(p2.values()))
    cap = GG.null_can_move(labels, [labels[i] for i in rng.permutation(len(rows))])
    say(f"     CAPABILITY CHECK: shuffling compartment labels changes {np.mean(moved):.1%} of "
        f"entries -- capable: {cap['capable']}")
    say(f"     statistic = KL(mass fractions || gene-count fractions); 0 means mass follows headcount")
    s = GG.survival(real_kl, nulls)
    GG.report("mass-vs-headcount KL under a compartment-label shuffle", s, emit=say)
    say(f"     the distribution flattens: entropy real {real_H:.4f} -> null "
        f"{np.mean(nH):.4f} +/- {np.std(nH):.4f}   (max of 12 = {np.log(12):.4f})")
    say(f"     largest single compartment: real {real_max:.4f} -> null {np.mean(nmax):.4f} "
        f"+/- {np.std(nmax):.4f}")
    dev = float(np.mean([abs(np.log2(prof[c] / cnt[c])) for c in prof if cnt.get(c, 0) > 0]))
    say(f"     BUT THE MAGNITUDE IS SMALL, and the gate does not measure magnitude: mass departs")
    say(f"     from headcount by only {dev:.3f} in mean |log2(mass/headcount)| across the 12 buckets,")
    say(f"     and entropy falls {1 - real_H/np.mean(nH):.1%} below the shuffle. Knowing a")
    say(f"     compartment's gene count already tells you its protein mass to within ~{2**dev:.2f}x.")
    m6 = bool(cap["capable"] and s.get("defined"))
    say(f"     M6 {'PASS' if m6 else 'FAIL'} -- the null "
        f"{'is capable, fired, and the real value stands clear of it' if m6 else ('is INERT' if not cap['capable'] else 'is capable but the real value is not distinguishable from it')}")
    say()

    # ------------------------------------------------------------------ M7
    say("M7 WHAT THIS DOES NOT GIVE")
    cov_sorted = sorted(comps, key=lambda c: -all_comp_cov[c])
    say(f"     BEST covered by the abundance data:  " +
        ", ".join(f"{c} {all_comp_cov[c]:.0%}" for c in cov_sorted[:4]))
    say(f"     WORST covered by the abundance data: " +
        ", ".join(f"{c} {all_comp_cov[c]:.0%}" for c in cov_sorted[-4:]))
    corr = {}
    for c in comps:
        if per_n.get(c, 0) and all_comp_cov[c] > 0:
            corr[c] = prof.get(c, 0.0) / all_comp_cov[c]
    tc = sum(corr.values())
    corr = {c: v / tc for c, v in corr.items()}
    say(f"     coverage-corrected sensitivity (each compartment's mass divided by its own coverage,")
    say(f"     i.e. unmeasured genes assumed to look like the measured ones in the same bucket):")
    say(f"       " + "   ".join(f"{c} {corr.get(c, 0):.4f}" for c in REFERENCE))
    say(f"       corrected mitochondrion {corr.get('mitochondrion', 0):.4f} vs raw {fm:.4f}; "
        f"distance to published {dist_to_reference(corr):.3f} vs {d_real:.3f}")
    say()
    say(f"     POST-HOC DIAGNOSTICS, added after M2/M3/M4 failed. They change no gate and no window.")
    mass_g = sorted(rows, key=lambda r: -r["ppm"] * r["mw"])
    tot_mass = sum(r["ppm"] * r["mw"] for r in rows)
    top20 = mass_g[:20]
    top20_share = sum(r["ppm"] * r["mw"] for r in top20) / tot_mass
    say(f"     (i)  CONCENTRATION: the 20 heaviest contributors carry {top20_share:.1%} of all "
        f"protein mass")
    say(f"          {', '.join(r['name'] + '/' + r['comp'][:4] for r in top20[:8])}")
    prof_cut = profile(mass_g[20:], "ppm")
    ocut = sorted(prof_cut, key=lambda c: -prof_cut[c])
    say(f"          delete them and the order becomes: {ocut[0]} {prof_cut[ocut[0]]:.4f} > "
        f"{ocut[1]} {prof_cut[ocut[1]]:.4f} > {ocut[2]} {prof_cut[ocut[2]]:.4f}   "
        f"cytosol-first: {ocut[0] == CYTOSOL_KEY}")
    srows_short = [dict(r, mw=mw_short[r["name"]]) for r in rows if r["name"] in mw_short]
    prof_short = profile(srows_short, "ppm")
    oshort = sorted(prof_short, key=lambda c: -prof_short[c])
    say(f"     (ii) ISOFORM CHOICE: with the SHORTEST UniProt isoform instead of the longest, "
        f"{oshort[0]} leads at {prof_short[oshort[0]]:.4f}")
    say(f"          nucleus {prof_short.get('nucleus', 0):.4f}  cytoplasm "
        f"{prof_short.get('cytoplasm', 0):.4f}  mitochondrion {prof_short.get('mitochondrion', 0):.4f}"
        f"   cytosol-first: {oshort[0] == CYTOSOL_KEY}")
    say(f"          -> the nucleus/cytosol verdict is "
        f"{'ROBUST to the isoform choice' if (oshort[0] == order[0]) else 'SENSITIVE to the isoform choice, which is a limitation of this module, not a result'}")
    say(f"     one bucket per gene is a lie about biology: a GO cellular-component term collapsed to")
    say(f"     a single label cannot represent a protein that is nuclear and cytosolic, or one that")
    say(f"     is made on the ER and delivered to the plasma membrane. Every dual-localised protein")
    say(f"     is charged entirely to one compartment here, and there is no way to know from this")
    say(f"     data which. PaxDb HeLa also omits ACTB and POLR2A outright.")
    say()

    gates = {"M1 the join is real and its coverage is declared": m1,
             "M2 the ordering is right (cytosol largest)": m2,
             "M3 the mitochondrion is within a factor of 2 of 0.15": m3,
             "M4 >= 3 of 4 referenced compartments within a factor of 2": m4,
             "M5 measured abundance beats the publication-count rival": m5,
             "M6 the label shuffle is capable, fired, and flattens the profile": m6,
             "M7 the limit is stated": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(CP.HELA), str(CP.GTF), str(FASTA),
                              str(SC / "_schwan2011.json")],
                      available=len(genes), used=len(rows), selection="filtered", seed=SEED,
                      controls=[f"compartment labels shuffled {NPERM}x, checked with null_can_move "
                                f"before the verdict was used",
                                "publication count given its own turn as a rival profile, scored "
                                "against the same published reference by the same distance",
                                "molar ppm multiplied by sequence MW, never one abundance dataset "
                                "divided by another (loop 92's rule)",
                                "the whole profile recomputed in an independent proteome "
                                "(Schwanhausser NIH3T3 copies) rather than only HeLa",
                                "per-compartment abundance coverage reported and a "
                                "coverage-corrected profile computed as a sensitivity",
                                "published reference values and the factor-2 tolerance fixed in the "
                                "docstring before any number was computed"],
                      note="the `comp` annotation has never been asked to reproduce a physical "
                           "quantity; organelle protein mass fractions were measured by "
                           "fractionation long before genomics and are an external answer")
    RM.report(man, emit=say)

    json.dump({"test": "loop_compartment_mass", "manifest": man, "gates": gates,
               "reference": REFERENCE, "factor": FACTOR,
               "m1": {"model_genes": len(genes), "with_ppm": len(measured),
                      "with_mw": len(with_mw), "mw_coverage": mw_cov, "used": len(rows),
                      "per_compartment_n": per_n, "thin_compartments": thin},
               "profile_mass": prof, "profile_molar": molar, "headcount_fraction": cnt,
               "compartment_coverage": all_comp_cov,
               "merged_cytosol": merged, "order": order,
               "top_contributors": top_by_comp,
               "m2": {"largest": order[0], "largest_fraction": prof[order[0]],
                      "runner_up": order[1]},
               "m3": {"mitochondrion": fm, "window": [lo, hi],
                      "ratio_to_published": (fm / REFERENCE["mitochondrion"])},
               "m4": {"within": within, "n_within": n_within, "distance_real": d_real,
                      "schwanhausser_profile": sprof, "schwanhausser_n": len(srows),
                      "schwanhausser_within": swithin, "schwanhausser_n_within": s_nwithin,
                      "distance_schwanhausser": d_sch, "rho_hela_vs_nih3t3": r_two},
               "m5": {"fame_profile": fame, "distance_fame": d_fame, "distance_real": d_real,
                      "rho_ppm_pubs_gene": r_gene, "n_gene": n_gene,
                      "rho_mass_medpubs_comp": r_cp, "rho_mass_headcount_comp": r_cc,
                      "rho_coverage_medpubs_comp": r_cov, "median_pubs": medpubs},
               "m6": {"capability": cap, "survival": s, "real_kl": real_kl,
                      "null_kl_mean": float(np.mean(nulls)), "null_kl_sd": float(np.std(nulls)),
                      "entropy_real": real_H, "entropy_null_mean": float(np.mean(nH)),
                      "entropy_max": float(np.log(12)),
                      "max_frac_real": real_max, "max_frac_null_mean": float(np.mean(nmax)),
                      "mean_abs_log2_mass_over_headcount": dev},
               "m7": {"best_covered": cov_sorted[:4], "worst_covered": cov_sorted[-4:],
                      "coverage_corrected_profile": corr,
                      "distance_coverage_corrected": dist_to_reference(corr),
                      "post_hoc": {"top20_mass_share": top20_share,
                                   "top20": [r["name"] for r in top20],
                                   "profile_without_top20": prof_cut,
                                   "order_without_top20": ocut[:3],
                                   "profile_shortest_isoform": prof_short,
                                   "largest_shortest_isoform": oshort[0]}},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_compartment_mass.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_compartment_mass.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
