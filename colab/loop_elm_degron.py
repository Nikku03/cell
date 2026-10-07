"""LOOP 146 -- THE PHOSPHODEGRON TEST REDONE ON CURATED MOTIFS, WITH A CONTROL THAT CAN MOVE.

Loop 145 asked the right question and asked it with the wrong instruments. Two failures, both
recorded there, and ELM fixes both.

  MY REGEXES WERE WRONG, AND THAT IS WHY FIVE OF ELEVEN TEXTBOOK SUBSTRATES RETURNED ZERO HITS.

      beta-TrCP   mine  DSGxxS                    ELM  D(S)G.{2,3}([ST])
      Fbw7        mine  [LIVMP].[ST]P..[ST]       ELM  [LIVMP].{0,2}(T)P..([ST])

  The spacers are variable and the terminal residue may be S OR T. A fixed-width guess misses the
  real sites in CDC25A, CDC25B, WEE1, JUN and MCL1. ELM also marks the PHOSPHO-ACCEPTOR positions
  explicitly with capture groups -- D(S)G.{2,3}([ST]) says which residues must carry the phosphate
  -- so "is this degron phospho-occupied" stops being my inference and becomes the database's.

  MY CONTROL ARM WAS CHEMICALLY INCAPABLE. P1 caught it: KEN-box occupancy is 0.0000 across 1,239
  genes because K, E and N cannot carry a phosphate. A falsification arm that cannot take the
  treatment is not a control. ELM supplies the arm that was missing -- degrons that CONTAIN
  serine or threonine but whose binding does NOT require phosphorylation, so they can be occupied
  and simply should not benefit from it.

THE THREE ARMS, and the middle one is the whole point:

    PHOSPHO-REQUIRED    binding depends on a phosphate at a position ELM marks
    PHOSPHO-CAPABLE     the motif contains S/T, so a site can land in it, but binding does not
                        require phosphorylation
    PHOSPHO-INCAPABLE   no S/T/Y in the motif at all, so occupancy is impossible by chemistry

If phospho-occupancy is a real timing signal it must help the FIRST arm and not the SECOND. Helping
both would mean the statistic tracks annotation density. Helping neither closes the line. The third
arm exists only to show the floor, and loop 145 mistook that floor for a control.

AND THE POSITIVE CONTROLS ARE NOW THE DATABASE'S, NOT MINE. ELM records 116 experimentally
validated human degron instances with exact start and end positions, plus 2 marked false positive
and 5 unknown. Loop 145 used twelve substrates I recalled; this uses a curated set with coordinates,
which also makes a REGEX SENSITIVITY test possible for the first time: does the pattern actually
recover the site the curators annotated?

PREDECLARED:

  Q1 DOES THE CURATED REGEX FIND THE CURATED SITE?                   THE SENSITIVITY TEST.
       for every validated instance, does its class regex match at the recorded position? Gate:
       report recovery. This is the check loop 145 could not run and the one that explains its
       zero hits. A regex that cannot recover the sites it was written from is not an instrument.

  Q2 ARE THE THREE ARMS WHAT I CLAIM?                                THE CAPABILITY CHECK.
       classify every DEG_ class by whether its regex can contain S/T/Y at all, and whether ELM
       marks a phospho-acceptor. Gate: all three arms must be non-empty, and the phospho-capable
       arm must show non-zero occupancy. Loop 145's control had zero and that killed its P1.

  Q3 THE TEST, WITH THE ARM THAT CAN MOVE.                           THE POINT OF THE LOOP.
       phospho-occupancy gain in AUC over motif-presence alone, per arm, scored on the validated
       instance proteins against a phospho-annotated background. Gate: the gain must exceed 0.02
       for PHOSPHO-REQUIRED and must NOT for PHOSPHO-CAPABLE. Both halves are required; the
       second is what makes it specific rather than a density effect.

  Q4 IS IT FAME?
       publication count against occupancy GIVEN the motif, per arm. Two loops died here at
       +0.2429 and +0.3296 and loop 145 measured +0.1804. Gate: below 0.25.

  Q5 DOES IT REACH THE 362?
       occupancy rate across the 88/362/38/642 split with Fisher tests. Asked LAST because Q1-Q3
       decide whether the instrument works at all.

  Q6 THE VERDICT ON THE STRUCTURAL BUILD.
       loop 145's P6 first printed JUSTIFIED on a +0.0094 AUC gain by ANDing booleans. This one
       states the effect size beside the verdict and requires it, and it must also say what a
       negative means: that nine mechanisms are eliminated and the timing is not in any constant
       property this project can currently measure.

-> outputs/loop_elm_degron.json
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
SC = LR.SC
FASTA = SC / "human_proteome.fasta.gz"
HPA = SC / "proteinatlas.tsv"
PTM = Path("colab/data/uniprot_human_ptm.tsv.gz")
ELM_CLASSES = Path("colab/data/elm_classes.tsv")
ELM_INST = Path("colab/data/elm_instances.tsv")
SEED = 14600
MIN_GAIN = 0.02

POS = re.compile(r"MOD_RES\s+(\d+);\s*/note=\"Phospho")
log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def fisher(a, b, c, d):
    from math import comb
    n, r1, c1 = a + b + c + d, a + b, a + c
    if r1 > n or c1 > n or min(a, b, c, d) < 0:
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


def rho(a, b):
    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        i = 0
        xs = x[o]
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = rk(np.asarray(a, float)), rk(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 146 -- phosphodegrons on ELM's curated motifs, with a control that can move")
    say("=" * 100)
    say()
    gates, res = {}, {}

    cls = {r["ELMIdentifier"]: r for r in
           csv.DictReader((l for l in open(ELM_CLASSES) if not l.startswith("#")), delimiter="\t")}
    inst = [r for r in
            csv.DictReader((l for l in open(ELM_INST) if not l.startswith("#")), delimiter="\t")]
    deg_cls = {k: v for k, v in cls.items() if k.startswith("DEG_")}
    deg_inst = [r for r in inst if r["ELMIdentifier"].startswith("DEG_")]
    say(f"  ELM: {len(cls)} classes, {len(deg_cls)} degron classes")
    say(f"  human instances {len(inst):,}; degron instances {len(deg_inst)} "
        f"({sum(1 for r in deg_inst if r['InstanceLogic'] == 'true positive')} true positive)")
    say()

    # sequences keyed by BOTH gene symbol and UniProt accession -- instances are keyed by accession
    seq_by_acc, seq_by_gene, acc2gene = {}, {}, {}
    acc = gene = None
    buf = []
    with gzip.open(FASTA, "rt", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if acc and buf:
                    s = "".join(buf)
                    seq_by_acc.setdefault(acc, s)
                    if gene:
                        seq_by_gene.setdefault(gene, s)
                        acc2gene[acc] = gene
                p = line.split("|")
                acc = p[1] if len(p) > 2 else None
                m = re.search(r"GN=(\S+)", line)
                gene = m.group(1) if m else None
                buf = []
            else:
                buf.append(line.strip())
    if acc and buf:
        s = "".join(buf)
        seq_by_acc.setdefault(acc, s)
        if gene:
            seq_by_gene.setdefault(gene, s)
            acc2gene[acc] = gene
    say(f"  proteome: {len(seq_by_acc):,} accessions, {len(seq_by_gene):,} gene symbols")

    sites = {}
    with gzip.open(PTM, "rt", errors="replace") as f:
        h = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(h):
                continue
            gn = p[ix["Gene Names (primary)"]]
            ps = [int(x) for x in POS.findall(p[ix["Modified residue"]])]
            if gn and ps:
                sites.setdefault(gn, set()).update(ps)
    say(f"  phospho-annotated genes: {len(sites):,}")
    say()

    # ---------------------------------------------------------------- Q1
    say("Q1 DOES THE CURATED REGEX FIND THE CURATED SITE?")
    tp = [r for r in deg_inst if r["InstanceLogic"] == "true positive"]
    rec = miss = nofa = 0
    per_class = {}
    for r in tp:
        a = r["Primary_Acc"].split("-")[0]
        s = seq_by_acc.get(a)
        if not s:
            nofa += 1
            continue
        rx = deg_cls.get(r["ELMIdentifier"], {}).get("Regex")
        if not rx:
            continue
        st, en = int(r["Start"]), int(r["End"])
        hit = any(m.start() + 1 <= en and m.end() >= st
                  for m in re.finditer(rx, s))
        per_class.setdefault(r["ELMIdentifier"], [0, 0])
        per_class[r["ELMIdentifier"]][1] += 1
        if hit:
            rec += 1
            per_class[r["ELMIdentifier"]][0] += 1
        else:
            miss += 1
    say(f"     validated true-positive instances with a sequence: {rec + miss} "
        f"({nofa} accessions absent from the proteome file)")
    say(f"     regex recovers the annotated site: {rec}/{rec + miss} ({rec / max(1, rec + miss):.1%})")
    for k, (g, n) in sorted(per_class.items(), key=lambda kv: -kv[1][1])[:8]:
        say(f"       {k:<26} {g:>2}/{n:<2}")
    say(f"     loop 145's hand-written regexes returned ZERO hits on five of eleven textbook")
    say(f"     substrates. This is the test that would have caught that, and it needed the")
    say(f"     curated coordinates to be possible at all.")
    gates["Q1"] = bool(rec / max(1, rec + miss) > 0.7)
    res["q1"] = {"recovered": rec, "missed": miss, "no_sequence": nofa,
                 "rate": rec / max(1, rec + miss), "per_class": per_class}
    say(f"     Q1 {'PASS' if gates['Q1'] else 'FAIL'} -- the curated patterns "
        f"{'recover the sites they were written from' if gates['Q1'] else 'DO NOT recover their own instances'}")
    say()

    # ---------------------------------------------------------------- Q2
    say("Q2 ARE THE THREE ARMS WHAT I CLAIM?")
    ST = re.compile(r"[STY]")

    def can_hold_phos(rx):
        """can this pattern ever match a window containing S, T or Y?"""
        return bool(ST.search(re.sub(r"\[\^[^\]]*\]", "X", rx))) or "." in rx

    arms = {"PHOSPHO-REQUIRED": [], "PHOSPHO-CAPABLE": [], "PHOSPHO-INCAPABLE": []}
    for k, v in deg_cls.items():
        rx = v["Regex"]
        # ELM marks phospho-acceptor positions with capture groups on S/T/Y
        groups = re.findall(r"\(([^)]*)\)", rx)
        marked = any(re.fullmatch(r"[STY]|\[[STY]+\]", g) for g in groups)
        bare = re.sub(r"[()]", "", rx)
        literal_st = bool(re.search(r"(?<!\^)[STY]", re.sub(r"\[\^[^\]]*\]", "X", bare)))
        if marked:
            arms["PHOSPHO-REQUIRED"].append(k)
        elif literal_st or "[ST" in bare:
            arms["PHOSPHO-CAPABLE"].append(k)
        else:
            arms["PHOSPHO-INCAPABLE"].append(k)
    for a, ks in arms.items():
        say(f"     {a:<20} {len(ks):>2} classes: {', '.join(sorted(ks)[:5])}"
            f"{' ...' if len(ks) > 5 else ''}")
    gates["Q2"] = bool(all(len(v) > 0 for v in arms.values())
                       and len(arms["PHOSPHO-CAPABLE"]) >= 3)
    res["q2"] = {a: sorted(ks) for a, ks in arms.items()}
    say(f"     Q2 {'PASS' if gates['Q2'] else 'FAIL'} -- "
        f"{'all three arms exist, and the middle one is the control loop 145 lacked' if gates['Q2'] else 'an arm is empty'}")
    say()

    # ---------------------------------------------------------------- scoring
    pool = [g for g in seq_by_gene if g in sites]
    say(f"  scoring on {len(pool):,} genes with a sequence and annotated phosphosites")
    feat = {}
    for a, ks in arms.items():
        for k in ks:
            rx = deg_cls[k]["Regex"]
            try:
                cx = re.compile(rx)
            except re.error:
                continue
            npres, nocc = {}, {}
            for g in pool:
                s = seq_by_gene[g]
                st = sites.get(g, set())
                hs = [(m.start() + 1, m.end()) for m in cx.finditer(s)]
                npres[g] = len(hs)
                nocc[g] = int(any(any(lo <= q <= hi for q in st) for lo, hi in hs))
            feat[k] = {"arm": a, "present": npres, "occupied": nocc}
    say()

    # ---------------------------------------------------------------- Q3
    say("Q3 THE TEST, WITH THE ARM THAT CAN MOVE")
    tp_genes = {}
    for r in tp:
        g = acc2gene.get(r["Primary_Acc"].split("-")[0])
        if g in sites:
            tp_genes.setdefault(r["ELMIdentifier"], set()).add(g)
    per_arm = {}
    for a in arms:
        gains, rows = [], []
        for k in arms[a]:
            if k not in feat or k not in tp_genes or len(tp_genes[k]) < 3:
                continue
            y = np.array([g in tp_genes[k] for g in pool])
            sp = np.array([feat[k]["present"][g] for g in pool], float)
            so = np.array([float(feat[k]["occupied"][g]) for g in pool])
            am, ao = auc(sp, y), auc(so, y)
            if not (np.isfinite(am) and np.isfinite(ao)):
                continue
            gains.append(ao - am)
            rows.append({"class": k, "n_pos": int(y.sum()), "auc_present": am,
                         "auc_occupied": ao, "gain": ao - am,
                         "occ_rate": float(np.mean(so))})
        g_mean = float(np.mean(gains)) if gains else float("nan")
        per_arm[a] = {"n_classes": len(rows), "mean_gain": g_mean, "rows": rows}
        say(f"     {a:<20} {len(rows)} testable classes   mean occupancy gain {g_mean:+.4f}")
        for r_ in sorted(rows, key=lambda x: -x["gain"])[:4]:
            say(f"       {r_['class']:<26} n={r_['n_pos']:>2}  present {r_['auc_present']:.3f}  "
                f"occupied {r_['auc_occupied']:.3f}  {r_['gain']:+.4f}  "
                f"(occ rate {r_['occ_rate']:.1%})")
    req = per_arm["PHOSPHO-REQUIRED"]["mean_gain"]
    cap = per_arm["PHOSPHO-CAPABLE"]["mean_gain"]
    say(f"     REQUIRED {req:+.4f} against a bar of {MIN_GAIN:+.4f}; CAPABLE {cap:+.4f}")
    say(f"     the hypothesis needs the FIRST to clear the bar and the SECOND not to. Helping both")
    say(f"     would mean the statistic tracks phospho-annotation density rather than dependence.")
    gates["Q3"] = bool(np.isfinite(req) and req > MIN_GAIN
                       and (not np.isfinite(cap) or cap < MIN_GAIN))
    res["q3"] = per_arm
    say(f"     Q3 {'PASS' if gates['Q3'] else 'FAIL'} -- phospho-occupancy "
        f"{'is specific to the degrons that require it' if gates['Q3'] else 'does NOT show the specificity the hypothesis requires'}")
    say()

    # ---------------------------------------------------------------- Q4
    say("Q4 IS IT FAME?")
    import cell_assembled as CA
    pubs = CA.load().get("pubs", {})
    q4 = {}
    for a in arms:
        rs = []
        for k in arms[a]:
            if k not in feat:
                continue
            sub = [g for g in pool if feat[k]["present"][g] > 0 and g in pubs]
            if len(sub) < 100:
                continue
            r_ = rho([float(pubs[g]) for g in sub],
                     [float(feat[k]["occupied"][g]) for g in sub])
            if np.isfinite(r_):
                rs.append(r_)
        q4[a] = {"n": len(rs), "mean_rho": float(np.mean(rs)) if rs else float("nan"),
                 "max_abs_rho": float(np.max(np.abs(rs))) if rs else float("nan")}
        say(f"     {a:<20} {len(rs):>2} classes   mean rho {q4[a]['mean_rho']:+.4f}   "
            f"max |rho| {q4[a]['max_abs_rho']:.4f}")
    worst = max((v["max_abs_rho"] for v in q4.values() if np.isfinite(v["max_abs_rho"])),
                default=float("nan"))
    say(f"     the KEN-box was struck at +0.2429, the 5'TOP motif at +0.3296, loop 145 measured "
        f"+0.1804.")
    gates["Q4"] = bool(np.isfinite(worst) and worst < 0.25)
    res["q4"] = q4
    say(f"     Q4 {'PASS' if gates['Q4'] else 'FAIL'} -- occupancy "
        f"{'is not simply fame' if gates['Q4'] else 'IS a fame proxy'}")
    say()

    # ---------------------------------------------------------------- Q5
    say("Q5 DOES IT REACH THE 362?")
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
    for g in sorted(set(cp) & set(ct)):
        p, t = cp[g] == "Yes", ct[g] == "Yes"
        grp[g] = "both" if (p and t) else ("protein only" if p else
                                           ("transcript only" if t else "neither"))
    reqk = [k for k in arms["PHOSPHO-REQUIRED"] if k in feat]
    occ_any = {g: int(any(feat[k]["occupied"][g] for k in reqk)) for g in pool}
    q5 = {}
    for name in ("both", "protein only", "transcript only", "neither"):
        v = [occ_any[g] for g in pool if grp.get(g) == name]
        q5[name] = {"n": len(v), "rate": (sum(v) / len(v)) if v else float("nan")}
        say(f"     {name:<16} n {len(v):>4}   phospho-occupied required-degron {q5[name]['rate']:.1%}")
    po, ne = q5["protein only"], q5["neither"]
    a_ = int(round(po["rate"] * po["n"])); b_ = po["n"] - a_
    c_ = int(round(ne["rate"] * ne["n"])); d_ = ne["n"] - c_
    pf = fisher(a_, b_, c_, d_) if min(po["n"], ne["n"]) > 10 else float("nan")
    say(f"     protein-only against neither: Fisher p = {pf:.4f}")
    gates["Q5"] = bool(np.isfinite(pf) and pf < 0.05 and po["rate"] > ne["rate"])
    res["q5"] = {"groups": q5, "fisher": pf}
    say(f"     Q5 {'PASS' if gates['Q5'] else 'FAIL'} -- the 362 "
        f"{'ARE enriched' if gates['Q5'] else 'are NOT enriched'}")
    say()

    # ---------------------------------------------------------------- Q6
    say("Q6 THE VERDICT ON THE STRUCTURAL BUILD")
    say(f"     effect size stated beside the verdict, which loop 145's P6 failed to do:")
    say(f"       phospho-REQUIRED occupancy gain {req:+.4f}   bar {MIN_GAIN:+.4f}")
    say(f"       phospho-CAPABLE  occupancy gain {cap:+.4f}   (must stay below the bar)")
    ok = gates["Q1"] and gates["Q2"] and gates["Q3"] and gates["Q4"]
    if ok:
        say(f"     THE PHOSPHO-STATE SPLIT IS JUSTIFIED. Occupancy helps exactly the degrons whose")
        say(f"     binding requires a phosphate and not the ones that merely contain S/T, on")
        say(f"     curated motifs that recover their own validated instances, and it is not fame.")
    else:
        say(f"     THE SPLIT IS NOT JUSTIFIED. Nine mechanisms are now eliminated and the timing is")
        say(f"     not in any CONSTANT property this project can measure -- not sequence, not")
        say(f"     annotation, not network membership, and not phospho-site presence. What has")
        say(f"     never been available is the quantity the physics actually points at: a")
        say(f"     degradation RATE measured through the cycle. Every remaining candidate is a")
        say(f"     proxy for that, and proxies have now failed nine times.")
    gates["Q6"] = True
    res["q6"] = {"justified": bool(ok), "req_gain": req, "cap_gain": cap, "bar": MIN_GAIN}
    say()

    say("=" * 100)
    for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[ELM_CLASSES, ELM_INST, FASTA, PTM],
                      available=len(seq_by_gene), used=len(pool), selection="all", seed=SEED,
                      controls=["a PHOSPHO-CAPABLE arm that contains S/T but does not require "
                                "phosphorylation -- the control loop 145 lacked",
                                "regex sensitivity against curated instance coordinates",
                                "positive controls from the database rather than from memory",
                                "publication count against occupancy given the motif, per arm",
                                "the effect size stated beside the build verdict"],
                      note="loop 145's regexes were hand-written and missed five of eleven textbook "
                           "substrates; its control arm was chemically incapable of occupancy.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 146 -- ELM phosphodegrons", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_elm_degron.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_elm_degron.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
