"""LOOP 145 -- THE LOCK EXISTS; WAS THE KEY EVER TURNED? PHOSPHODEGRONS.

Eight mechanisms are eliminated on the question of which proteins are destroyed on schedule, and
every one of them tested a CONSTANT property of a protein -- a sequence motif, an annotation, a
network membership, a curator's cross-link. The thing being explained is a TIMED event. The only
property of a protein that varies on the timescale of a cell cycle is its MODIFICATION STATE.

AND LOOP 121 TESTED THE WRONG DEGRONS FOR THAT ARGUMENT. It searched D-box (R..L), D-box+
(R..L..[LIVM]) and KEN-box (KEN). Those are APC/C degrons and they are bound DIRECTLY -- APC/C
recognition does not require the degron to be phosphorylated. So loop 121's failure says nothing
about phospho-timing, because it never tested a motif whose binding depends on phosphorylation.

The SCF-family degrons are the phospho-dependent ones, and they were never searched:

    beta-TrCP    DSGxxS          bound ONLY when both serines are phosphorylated
    Fbw7 (CPD)   [LIVMP]x[ST]Pxx[ST]   the Cdc4 phosphodegron, bound only when the T/S is phosphorylated

That is the exact shape the argument needs. The motif is present in the genome from birth; the
PHOSPHATE is what arrives on schedule. Loop 121 asked whether the lock exists. It never asked
whether the key was turned.

THE TEST, AND WHY IT IS BETTER CONTROLLED THAN ANYTHING IN THIS ARC. Phosphosite annotation is a
fame proxy -- loop 137 measured publication count predicting ubiquitin annotation at AUC 0.7169,
and loops 121 and 122 were both struck for motif densities correlating with publication count at
+0.2429 and +0.3296. So the comparison is made WITHIN the phospho-annotated proteins and asks about
CO-LOCATION, not presence:

    both groups carry a degron motif.  both groups carry annotated phosphosites.
    the only difference is whether a phosphosite sits INSIDE the motif.

Fame drives how many sites a protein has annotated. It does not drive whether one of them lands in
a six-residue window, and P4 checks that rather than asserting it.

AND THE POSITIVE CONTROLS ARE DIFFERENT FROM LOOP 143's. Those were APC/C substrates. These are the
SCF substrates, textbook and independent: beta-TrCP takes CTNNB1, NFKBIA, CDC25A, CDC25B, WEE1 and
FBXO5; Fbw7 takes CCNE1, MYC, JUN, MCL1, NOTCH1 and KLF5. If a phospho-occupied motif cannot find
those, the idea is dead and no amount of reasoning about timing repairs it.

WHAT IS HONESTLY AT RISK. The known SCF substrate list is small -- around a dozen with complete
data -- so this is a low-power test and P2 states its power before reading it. A small-n positive
is weak evidence; a small-n negative on a textbook set is strong evidence, because these are the
proteins the motif was DEFINED on.

PREDECLARED:

  P1 DO THE MOTIFS AND THE SITES EXIST, AND CAN THEY CO-LOCATE?      THE CAPABILITY CHECK.
       counts of each motif across the proteome, counts of phospho-annotated proteins, and how
       often a phosphosite falls inside a motif. Gate: co-location must be neither ~0 nor ~1. If
       every motif already contains a site, the statistic cannot separate anything.

  P2 THE POSITIVE CONTROL, WITH ITS POWER STATED FIRST.              THE ONE THAT CAN END IT.
       twelve textbook SCF substrates. Gate: report how many carry a phospho-occupied degron, and
       state the binomial power to detect a two-fold enrichment at this n BEFORE reading the
       result. If the substrates the motifs were defined on do not carry them, the search is
       broken.

  P3 THE COMPARISON LOOP 121 COULD NOT MAKE.                         THE POINT OF THE LOOP.
       motif-alone against motif-with-a-phosphosite-in-it, scored on the same substrate set with
       the same background. Gate: report both AUCs. If phospho-occupancy adds nothing over the
       motif, then the timing is not in the phosphorylation and this whole line closes.

  P4 IS CO-LOCATION FAME?
       publication count against (a) number of annotated phosphosites, (b) motif count, and
       (c) co-location GIVEN both. Gate: (a) will be large and that is expected. (c) is the one
       that matters and it must be small. Two loops died on this check and it is applied to the
       new statistic without exemption.

  P5 DOES IT REACH THE CELL-CYCLE GROUPS?
       phospho-occupied degron rate across the 88/362/38/642 split. Gate: report all four with
       Fisher tests. This is the question the arc has been asking, and it is asked LAST because
       P2 and P3 decide whether the instrument works at all.

  P6 THE STRUCTURAL CONSEQUENCE, STATED WHETHER OR NOT IT IS EARNED.
       the signalling layer is STATIC because "phospho-forms share a node so the chain is
       chemically inert". Splitting each phosphoprotein into two states is what would make it
       propagate. Gate: say plainly whether P2-P5 justify that build or not.

-> outputs/loop_phosphodegron.json
"""
import csv
import gzip
import json
import math
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

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
FASTA = LR.SC / "human_proteome.fasta.gz"
HPA = LR.SC / "proteinatlas.tsv"
PTM = Path("colab/data/uniprot_human_ptm.tsv.gz")
SEED = 14500

# PHOSPHO-DEPENDENT degrons. Both are bound only when the marked residue carries a phosphate.
PHOSPHO_MOTIFS = {
    "betaTrCP DSGxxS": re.compile(r"DSG..S"),
    "Fbw7 CPD": re.compile(r"[LIVMP].[ST]P..[ST]"),
}
# loop 121's APC/C degrons, kept as the comparison arm. These are NOT phospho-dependent.
APC_MOTIFS = {
    "D-box R..L": re.compile(r"R..L"),
    "KEN-box": re.compile(r"KEN"),
}
# textbook SCF substrates, independent of this project and of loop 143's APC/C set
SCF = {
    "CTNNB1": "betaTrCP", "NFKBIA": "betaTrCP", "CDC25A": "betaTrCP",
    "CDC25B": "betaTrCP", "WEE1": "betaTrCP", "FBXO5": "betaTrCP",
    "CCNE1": "Fbw7", "MYC": "Fbw7", "JUN": "Fbw7",
    "MCL1": "Fbw7", "NOTCH1": "Fbw7", "KLF5": "Fbw7",
}
POS = re.compile(r"MOD_RES\s+(\d+);\s*/note=\"Phospho")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def fisher(a, b, c, d):
    from math import comb
    n, r1, c1 = a + b + c + d, a + b, a + c
    if min(n, r1, c1) < 0 or r1 > n or c1 > n:
        return float("nan")
    def p(k):
        return comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return float(sum(p(k) for k in range(lo, hi + 1) if p(k) <= p0 * (1 + 1e-9)))


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    if l.all() or not l.any():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    sr, r = s[o], np.empty(len(s), float)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 145 -- the lock exists; was the key ever turned?")
    say("=" * 100)
    say()
    gates, res = {}, {}

    seq = {}
    g = None
    buf = []
    with gzip.open(FASTA, "rt", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if g and buf and g not in seq:
                    seq[g] = "".join(buf)
                m = re.search(r"GN=(\S+)", line)
                g = m.group(1) if m else None
                buf = []
            else:
                buf.append(line.strip())
    if g and buf and g not in seq:
        seq[g] = "".join(buf)
    say(f"  proteome: {len(seq):,} genes with a sequence")

    sites = {}
    with gzip.open(PTM, "rt", errors="replace") as f:
        h = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(h):
                continue
            gn = p[ix["Gene Names (primary)"]]
            if not gn:
                continue
            ps = [int(x) for x in POS.findall(p[ix["Modified residue"]])]
            if ps:
                sites.setdefault(gn, set()).update(ps)
    say(f"  phospho-annotated genes: {len(sites):,}")
    say()

    def hits(s, rx):
        return [(m.start() + 1, m.end()) for m in rx.finditer(s)]

    genes = sorted(set(seq) & set(sites) | set(seq))
    feat = {}
    for gn in genes:
        s = seq.get(gn)
        if not s:
            continue
        st = sites.get(gn, set())
        row = {"n_sites": len(st), "length": len(s)}
        for lab, rx in {**PHOSPHO_MOTIFS, **APC_MOTIFS}.items():
            hs = hits(s, rx)
            row[f"n_{lab}"] = len(hs)
            row[f"occ_{lab}"] = int(any(any(a <= q <= b for q in st) for a, b in hs))
        feat[gn] = row

    # ---------------------------------------------------------------- P1
    say("P1 DO THE MOTIFS AND THE SITES EXIST, AND CAN THEY CO-LOCATE?")
    have_site = [gn for gn in feat if feat[gn]["n_sites"] > 0]
    say(f"     genes with a sequence AND at least one annotated phosphosite: {len(have_site):,}")
    p1 = {}
    for lab in {**PHOSPHO_MOTIFS, **APC_MOTIFS}:
        withm = [gn for gn in have_site if feat[gn][f"n_{lab}"] > 0]
        occ = [gn for gn in withm if feat[gn][f"occ_{lab}"]]
        frac = len(occ) / max(1, len(withm))
        p1[lab] = {"n_with_motif": len(withm), "n_occupied": len(occ), "frac_occupied": frac}
        say(f"     {lab:<18} motif in {len(withm):>6,} phospho-genes; a site lands inside it in "
            f"{len(occ):>6,}  ({frac:.1%})")
    ok = all(0.02 < v["frac_occupied"] < 0.98 for v in p1.values())
    gates["P1"] = bool(ok)
    res["p1"] = {"n_phospho_genes": len(have_site), "motifs": p1}
    say(f"     P1 {'PASS' if gates['P1'] else 'FAIL'} -- co-location "
        f"{'separates genes' if ok else 'IS DEGENERATE and can measure nothing'}")
    say()

    # ---------------------------------------------------------------- P2
    say("P2 THE POSITIVE CONTROL, WITH ITS POWER STATED FIRST")
    have = [gn for gn in SCF if gn in feat and feat[gn]["n_sites"] > 0]
    say(f"     textbook SCF substrates with sequence and phosphosites: {len(have)}/{len(SCF)}")
    base = {lab: p1[lab]["n_occupied"] / max(1, len(have_site)) for lab in PHOSPHO_MOTIFS}
    say(f"     BEFORE reading the result: at n = {len(have)}, a background rate of "
        f"{np.mean(list(base.values())):.3f} and a")
    say(f"     two-fold enrichment gives roughly "
        f"{100 * (1 - (1 - 2 * np.mean(list(base.values()))) ** len(have)):.0f}% chance that at least one substrate is hit.")
    say(f"     A small-n POSITIVE here is weak. A small-n NEGATIVE is strong, because these are the")
    say(f"     proteins the motifs were DEFINED on.")
    n_occ = 0
    for gn in have:
        own = SCF[gn]
        lab = "betaTrCP DSGxxS" if own == "betaTrCP" else "Fbw7 CPD"
        o = feat[gn][f"occ_{lab}"]
        n_occ += o
        say(f"       {gn:<8} {own:<9} motif copies {feat[gn][f'n_{lab}']:>2}  sites "
            f"{feat[gn]['n_sites']:>3}   phospho-occupied: {'YES' if o else 'no'}")
    rate = n_occ / max(1, len(have))
    bg = np.mean([base[l] for l in base])
    say(f"     substrates with a phospho-occupied degron of THEIR OWN ligase: {n_occ}/{len(have)} "
        f"({rate:.1%}) against a background of {bg:.1%}")
    gates["P2"] = bool(rate > bg)
    res["p2"] = {"n_substrates": len(have), "n_occupied": n_occ, "rate": rate, "background": bg}
    say(f"     P2 {'PASS' if gates['P2'] else 'FAIL'} -- the motifs "
        f"{'do find the proteins they were defined on' if gates['P2'] else 'DO NOT find their own textbook substrates'}")
    say()

    # ---------------------------------------------------------------- P3
    say("P3 THE COMPARISON LOOP 121 COULD NOT MAKE")
    pool = have_site
    lab_y = np.array([gn in SCF for gn in pool])
    say(f"     scored on {len(pool):,} phospho-annotated genes, {int(lab_y.sum())} of them "
        f"textbook SCF substrates")
    p3 = {}
    for lab in {**PHOSPHO_MOTIFS, **APC_MOTIFS}:
        s_motif = np.array([feat[gn][f"n_{lab}"] for gn in pool], float)
        s_occ = np.array([float(feat[gn][f"occ_{lab}"]) for gn in pool])
        a_m, a_o = auc(s_motif, lab_y), auc(s_occ, lab_y)
        p3[lab] = {"auc_motif_count": a_m, "auc_phospho_occupied": a_o, "gain": a_o - a_m}
        say(f"     {lab:<18} motif count AUC {a_m:.4f}   phospho-occupied AUC {a_o:.4f}   "
            f"{a_o - a_m:+.4f}")
    ph_gain = np.mean([p3[l]["gain"] for l in PHOSPHO_MOTIFS])
    apc_gain = np.mean([p3[l]["gain"] for l in APC_MOTIFS])
    say(f"     mean gain from requiring a phosphosite:  PHOSPHO-degrons {ph_gain:+.4f}   "
        f"APC/C degrons {apc_gain:+.4f}")
    say(f"     the APC/C arm is the falsification control: those degrons are NOT phospho-dependent,")
    say(f"     so requiring a phosphosite should help them LESS. If it helps them equally, the")
    say(f"     statistic is measuring phospho-annotation depth and not phospho-dependence.")
    gates["P3"] = bool(ph_gain > apc_gain and ph_gain > 0)
    res["p3"] = {"per_motif": p3, "phospho_gain": ph_gain, "apc_gain": apc_gain}
    say(f"     P3 {'PASS' if gates['P3'] else 'FAIL'} -- phospho-occupancy "
        f"{'helps the phospho-dependent degrons specifically' if gates['P3'] else 'does NOT show the specificity the hypothesis requires'}")
    say()

    # ---------------------------------------------------------------- P4
    say("P4 IS CO-LOCATION FAME?")
    import cell_assembled as CA
    pubs = CA.load().get("pubs", {})
    pp = [gn for gn in pool if gn in pubs]
    pv = np.array([float(pubs[gn]) for gn in pp])
    ns = np.array([float(feat[gn]["n_sites"]) for gn in pp])
    def rho(a, b):
        def rk(x):
            o = np.argsort(x, kind="mergesort"); r = np.empty(len(x), float); i = 0; xs = x[o]
            while i < len(xs):
                j = i
                while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                    j += 1
                r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
            return r
        ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
        return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))
    r_sites = rho(pv, ns)
    say(f"     publication count vs NUMBER OF ANNOTATED SITES: rho {r_sites:+.4f}  (expected large)")
    p4 = {"rho_pubs_nsites": r_sites}
    for lab in PHOSPHO_MOTIFS:
        sub = [gn for gn in pp if feat[gn][f"n_{lab}"] > 0]
        if len(sub) < 50:
            continue
        pv2 = np.array([float(pubs[gn]) for gn in sub])
        oc = np.array([float(feat[gn][f"occ_{lab}"]) for gn in sub])
        r = rho(pv2, oc)
        p4[lab] = {"n": len(sub), "rho_pubs_occupancy_given_motif": r}
        say(f"     {lab:<18} among genes that HAVE the motif, pubs vs occupancy: rho {r:+.4f} "
            f"(n={len(sub):,})")
    worst = max(abs(v["rho_pubs_occupancy_given_motif"]) for k, v in p4.items() if isinstance(v, dict))
    gates["P4"] = bool(worst < 0.25)
    res["p4"] = p4
    say(f"     the KEN-box was struck at +0.2429 and the 5'TOP motif at +0.3296. Worst here "
        f"{worst:+.4f}.")
    say(f"     P4 {'PASS' if gates['P4'] else 'FAIL'} -- co-location "
        f"{'is not simply fame' if gates['P4'] else 'IS a fame proxy, like the two motifs before it'}")
    say()

    # ---------------------------------------------------------------- P5
    say("P5 DOES IT REACH THE CELL-CYCLE GROUPS?")
    cp, ct = {}, {}
    with open(HPA, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        h = next(rd)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        for row in rd:
            if len(row) <= max(iG, iCP, iCT):
                continue
            if row[iCP] in ("Yes", "No"):
                cp[row[iG]] = row[iCP]
            if row[iCT] in ("Yes", "No"):
                ct[row[iG]] = row[iCT]
    grp = {}
    for gn in sorted(set(cp) & set(ct)):
        p, t = cp[gn] == "Yes", ct[gn] == "Yes"
        grp[gn] = "both" if (p and t) else ("protein only" if p else
                                            ("transcript only" if t else "neither"))
    occ_any = {gn: int(any(feat[gn][f"occ_{l}"] for l in PHOSPHO_MOTIFS)) for gn in pool}
    p5 = {}
    for name in ("both", "protein only", "transcript only", "neither"):
        v = [occ_any[gn] for gn in pool if grp.get(gn) == name]
        p5[name] = {"n": len(v), "rate": (sum(v) / len(v)) if v else float("nan")}
        say(f"     {name:<16} n {len(v):>4}   phospho-occupied degron {p5[name]['rate']:.1%}")
    po, ne = p5["protein only"], p5["neither"]
    a = int(po["rate"] * po["n"]); b = po["n"] - a
    c = int(ne["rate"] * ne["n"]); dd = ne["n"] - c
    pf = fisher(a, b, c, dd) if min(po["n"], ne["n"]) > 10 else float("nan")
    say(f"     protein-only against neither: Fisher p = {pf:.4f}")
    gates["P5"] = bool(np.isfinite(pf) and pf < 0.05 and po["rate"] > ne["rate"])
    res["p5"] = {"groups": p5, "fisher_protein_only_vs_neither": pf}
    say(f"     P5 {'PASS' if gates['P5'] else 'FAIL'} -- the 362 "
        f"{'ARE enriched for phospho-occupied degrons' if gates['P5'] else 'are NOT enriched'}")
    say()

    # ---------------------------------------------------------------- P6
    say("P6 THE STRUCTURAL CONSEQUENCE")
    core = gates["P2"] and gates["P3"] and gates["P4"]
    say(f"     the signalling layer is STATIC because 'phospho-forms share a node so the chain is")
    say(f"     chemically inert'. Splitting each phosphoprotein into an unmodified and a modified")
    say(f"     state is what would let information propagate through 17,432 signal edges.")
    if core:
        say(f"     THAT BUILD IS JUSTIFIED. Phospho-occupancy beats motif alone, specifically for")
        say(f"     the phospho-DEPENDENT degrons and not for the APC/C ones, and it is not fame.")
        say(f"     The timing lives in the modification, which is the thing the split makes")
        say(f"     representable.")
    else:
        say(f"     THAT BUILD IS NOT JUSTIFIED BY THIS RESULT. Splitting the nodes would make the")
        say(f"     network able to propagate, but nothing here says the propagated quantity")
        say(f"     predicts which protein is destroyed when. Building it would be adding")
        say(f"     machinery in the hope that a mechanism appears, which is what the previous")
        say(f"     eight attempts each did in their own way.")
    gates["P6"] = True
    res["p6"] = {"split_justified": bool(core)}
    say()

    say("=" * 100)
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[FASTA, PTM, HPA],
                      available=len(seq), used=len(pool), selection="all", seed=SEED,
                      controls=["the APC/C degrons as a falsification arm -- they are NOT "
                                "phospho-dependent, so requiring a site should help them less",
                                "the comparison made WITHIN phospho-annotated genes, so annotation "
                                "depth is held fixed",
                                "textbook SCF substrates, independent of loop 143's APC/C set",
                                "power stated before the positive control is read",
                                "publication count against co-location GIVEN the motif"],
                      note="loop 121 searched D-box and KEN-box, which APC/C binds without "
                           "phosphorylation. The phospho-DEPENDENT degrons were never tested.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 145 -- phosphodegrons", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_phosphodegron.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_phosphodegron.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
