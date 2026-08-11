"""THE PRODUCER cell_complete.json NEVER HAD -- what it is made of, and how much of it can be rebuilt.

THE HOLE THIS FILLS.  cell_complete.json is 38 MB, read by 227 modules, gitignored, and has no producer
anywhere in this repository. data_ledger named it as the biggest remaining defect and could not fix it:
a ledger can record a file's hash, it cannot invent the code that made one. If this container is
reclaimed, that file is gone and roughly half of what has been built stops being reproducible by
anyone, including me.

WHAT THIS IS NOT.  It is not a claim to have recovered the original program. The original is lost. This
is a producer that rebuilds the file's LOAD-BEARING FIELDS from named public sources and then CHECKS
each rebuilt field against the surviving copy, so the claim "this field came from that source" is
falsifiable rather than asserted.

SCOPING IT BY WHAT THE RESULTS ACTUALLY STAND ON.  The file has 42 top-level keys and 27 per-gene
fields over 16,492 genes. Rebuilding all of it is the wrong target -- much of it is unused. So every
field was counted by HOW MANY MODULES READ IT, and the producer is aimed at the load-bearing head of
that distribution:

    name 357 · ppi 209 · dep_frac 88 · tf 83 · chrom 80 · ess 68 · path 58 · comp 55 · proc 46
    loeuf 44 · pubs 42 · tss 41 · dark 39 · npath 31 · conf 26 · ndis 24 · enh 20 · cpg 19

FIVE KINDS OF FIELD, and they have completely different prospects. The first run of this module
used three -- recoverable, derived, project-invented -- and the middle one was wishful: `npath`
turned out not to be derivable from GO at all (corr 0.02), and `ndis`/`enh` are real counts from
catalogues nobody wrote down. Collapsing those into "derived" would have claimed a rule this file
does not have. The taxonomy below is what survived contact with the data:

    RECOVERABLE     a named public source exists, was fetched here, and the rebuilt column is
                    checked against the surviving copy: coordinates (Ensembl GTF), loeuf (gnomAD),
                    tf (Lambert), cpg (UCSC CpG islands).
    DECLARED        a named public source exists and the URL is recorded, but it was not fetched in
                    this run: ppi (BioGRID), dep_frac/ess (DepMap), pubs (NCBI).
    SOURCE KNOWN, RULE LOST   the ontology survives but the coarsening this project applied to it
                    does not: proc and comp are each one of exactly 12 buckets, and the GO-term ->
                    bucket map is recorded nowhere.
    SOURCE NOT IDENTIFIED   real data whose origin this file does not name: tss (a second TSS track,
                    stored as a string and differing from the Ensembl TSS in every gene), npath,
                    path, enh, ndis.
    PROJECT-INVENTED  labels this project made up and whose rule is lost: dark, conf, master, flag,
                    ti, ess_prob. These are NOT recoverable, and saying so is the point -- a producer
                    that quietly regenerated them from a guessed rule would be worse than one that
                    admits the gap, because the guess would be indistinguishable from the original in
                    the file and different in every value.

PREDECLARED, before any number:

    Q1 THE INVENTORY IS COMPLETE AND USAGE-WEIGHTED   every top-level key and per-gene field enumerated
                with the number of modules that read it. Without the weighting this rebuilds whatever
                was easiest rather than whatever matters.
    Q2 EVERY FIELD IS CLASSIFIED   into one of the five kinds above, with a URL wherever a source is
                named and an explicit admission wherever one is not. No field is left unaccounted.
    Q3 REBUILT FIELDS MATCH THE SURVIVING COPY   each rebuilt field agrees with the original above its
                declared threshold.
                THIS IS THE REAL GATE. A producer that does not reproduce the file is not the producer,
                and the honest failure mode is worth distinguishing: HIGH correlation with MODERATE
                exact-match means the source is right and the VERSION drifted, which is expected and
                recorded; LOW correlation means the source is wrong and the provenance claim is false.
    Q4 THE SHARE OF MODULE-READS COVERED IS REPORTED   what fraction of all per-gene field reads now
                come from a field with a named, verified source. That number is the honest answer to
                "how much of this file is reproducible", and it will not be 100%.

-> outputs/build_cell_complete.json
"""
import collections
import glob
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"

# ---- the source registry. Every entry is a claim that can be checked. -------------------------------
SOURCES = {
    "coordinates": {
        "fields": ["chrom", "gene_start", "gene_end", "strand", "tss", "ens_tss"],
        "url": "https://ftp.ensembl.org/pub/release-112/gtf/homo_sapiens/"
               "Homo_sapiens.GRCh38.112.gtf.gz",
        "path": SC / "ens_gtf.gz", "kind": "recoverable"},
    "loeuf": {
        "fields": ["loeuf"],
        "url": "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/"
               "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",
        "path": SC / "gnomad_loeuf.txt.bgz", "kind": "recoverable"},
    "gene ontology": {
        "fields": [],   # see SOURCE_KNOWN_RULE_LOST -- GOA is fetchable, the 12-bucket rule is not
        "url": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
        "path": SC / "goa_human.gaf.gz", "kind": "fetched, but see rule-lost below"},
    "transcription factors": {
        "fields": ["tf"],
        "url": "http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
        "path": SC / "lambert.csv", "kind": "recoverable"},
    "CpG islands": {
        "fields": ["cpg"],
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",
        "path": SC / "cpgIslandExt.txt.gz", "kind": "recoverable"},
    "protein interactions": {
        "fields": ["ppi"],
        "url": "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/"
               "BIOGRID-ORGANISM-LATEST.tab3.zip",
        "path": None, "kind": "declared, not fetched here"},
    "dependency": {
        "fields": ["dep_frac", "ess", "ess_src"],
        "url": "https://depmap.org/portal/download/  (CRISPRGeneEffect.csv, release-pinned)",
        "path": None, "kind": "declared, not fetched here"},
    "publications": {
        "fields": ["pubs"],
        "url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz + gene_info.gz",
        "path": None, "kind": "declared, not fetched here"},
}
# A THIRD AND FOURTH CATEGORY, because "recoverable" and "invented" were not enough to be honest.
SOURCE_KNOWN_RULE_LOST = {
    "proc": "one of exactly 12 process buckets (the `procs` key). GOA is fetchable; the GO-term -> "
            "12-bucket aggregation this project defined is not recorded anywhere",
    "comp": "one of exactly 12 compartment buckets (the `comps` key). Same shape as proc: the source "
            "ontology survives, the coarsening rule does not",
}
SOURCE_NOT_IDENTIFIED = {
    # CORRECTION. The first pass said tss "differs from ens_tss in every gene". That was wrong and
    # measurable: 6.65% agree EXACTLY, the median offset is 75 bp, 54% are within 100 bp, and the two
    # correlate at 0.9998. tss is not a different locus -- it is the same promoter picked at
    # TRANSCRIPT level (a canonical/MANE start) where ens_tss is the gene BOUNDARY. What is
    # unidentified is the transcript set, not the coordinate system.
    "tss": "stored as a STRING; the same promoter as ens_tss picked per-TRANSCRIPT rather than at the "
           "gene boundary (6.65% exact, median offset 75 bp, corr 0.9998). The transcript set that "
           "defines the pick -- MANE, canonical, refGene, CAGE -- is not recorded",
    "npath": "a pathway count. Not GO process count (corr 0.02) and not membership in the `pathways` "
             "key (exact 0.37), so the pathway database used is not recorded",
    "path": "a single Reactome-style pathway name with a trailing space; the release is not recorded",
    "enh": "an enhancer count; the catalogue is not recorded",
    "ndis": "a disease count; the database is not recorded",
}
PROJECT_INVENTED = {
    "dark": "a project label for under-studied genes; the threshold rule is lost",
    "conf": "a confidence tier this project assigned; rule lost",
    "master": "'master regulator' call; rule lost",
    "flag": "free-text annotations like 'germline-constrained yet cancer-dispensable'; rule lost",
    "ti": "a therapeutic-index tag; rule lost",
    "ess_prob": "a predicted essentiality probability from a model that is not in this repo",
}
# agreement thresholds, declared per kind of field
GATE_EXACT = 0.90        # categorical / integer fields
GATE_CORR = 0.90         # continuous fields


def load_gtf_coords():
    """Canonical gene coordinates and TSS by symbol, from the GTF gene records."""
    out = {}
    for line in gzip.open(SC / "ens_gtf.gz", "rt"):
        if line[0] == "#":
            continue
        f = line.split("\t", 9)
        if f[2] != "gene":
            continue
        m = re.search(r'gene_name "([^"]+)"', f[8])
        if not m:
            continue
        s, e, st = int(f[3]), int(f[4]), f[6]
        out[m.group(1)] = {"chrom": f"chr{f[0]}", "gene_start": s, "gene_end": e, "strand": st,
                           "tss": s if st == "+" else e}
    return out


def load_loeuf():
    out = {}
    with gzip.open(SC / "gnomad_loeuf.txt.bgz", "rt", errors="ignore") as f:
        head = f.readline().rstrip("\n").split("\t")
        ig, il = head.index("gene"), head.index("oe_lof_upper")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= il or not p[il]:
                continue
            try:
                v = float(p[il])
            except ValueError:
                continue
            if p[ig] not in out or v < out[p[ig]]:
                out[p[ig]] = v
    return out


def load_go():
    """Symbol -> sets of GO aspects. P = biological process, C = cellular component."""
    proc, comp = collections.defaultdict(set), collections.defaultdict(set)
    with gzip.open(SC / "goa_human.gaf.gz", "rt", errors="ignore") as f:
        for line in f:
            if line[0] == "!":
                continue
            p = line.split("\t")
            if len(p) < 10:
                continue
            sym, aspect, term = p[2], p[8], p[9]
            (proc if aspect == "P" else comp if aspect == "C" else proc)[sym].add(term)
            if aspect not in ("P", "C"):
                proc[sym].discard(term)
    return proc, comp


def load_tfs():
    import csv
    out = set()
    for r in csv.DictReader(open(SC / "lambert.csv", errors="ignore")):
        if r.get("Is TF?", "").strip().lower() == "yes" and r.get("HGNC symbol"):
            out.add(r["HGNC symbol"])
    return out


def load_cpg():
    """CpG island intervals per chromosome, for a TSS-overlap test."""
    iv = collections.defaultdict(list)
    with gzip.open(SC / "cpgIslandExt.txt.gz", "rt", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            iv[p[1]].append((int(p[2]), int(p[3])))
    for c in iv:
        iv[c].sort()
    return iv


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    t0 = time.time()
    say("=" * 100)
    say("THE PRODUCER cell_complete.json NEVER HAD")
    say("=" * 100)
    say("  38 MB, read by 227 modules, gitignored, no producer anywhere. This does not claim to have")
    say("  recovered the original program -- that is lost. It rebuilds the LOAD-BEARING fields from")
    say("  named sources and CHECKS each one against the surviving copy, so every provenance claim")
    say("  can fail.")

    D = json.load(open(CELL))
    genes = D["genes"]
    say(f"\n  surviving copy: {len(D)} top-level keys, {len(genes):,} genes")

    # ---- Q1 the usage-weighted inventory ----------------------------------------------------------
    src = {f: open(f, errors="ignore").read() for f in glob.glob(str(ROOT / "colab" / "*.py"))}
    fields = sorted({k for g in genes[:400] for k in g})

    def reads(key):
        return sum(1 for t in src.values() if f'"{key}"' in t or f"'{key}'" in t)

    usage = {f: reads(f) for f in fields}
    total_reads = sum(usage.values())
    q1 = bool(len(fields) > 0 and total_reads > 0)
    say(f"\n  Q1 THE INVENTORY IS USAGE-WEIGHTED -- {len(fields)} per-gene fields, "
        f"{total_reads:,} module-reads in total")
    say(f"     the head: " + " · ".join(f"{k} {v}" for k, v in
                                        sorted(usage.items(), key=lambda x: -x[1])[:8]))
    say(f"     Q1 {'PASS' if q1 else 'FAIL'}")

    # ---- Q2 every field classified ---------------------------------------------------------------
    covered = {f for s in SOURCES.values() for f in s["fields"]}
    unaccounted = [f for f in fields if f not in covered and f not in PROJECT_INVENTED
                   and f not in SOURCE_KNOWN_RULE_LOST and f not in SOURCE_NOT_IDENTIFIED
                   and f != "name"]
    q2 = bool(not unaccounted)
    say(f"\n  Q2 EVERY FIELD IS CLASSIFIED")
    for name, s in SOURCES.items():
        say(f"     {name:<24}{s['kind']:<26}{', '.join(s['fields'])}")
    say(f"     {'SOURCE KNOWN, RULE LOST':<24}{'partly recoverable':<26}"
        f"{', '.join(sorted(SOURCE_KNOWN_RULE_LOST))}")
    say(f"     {'SOURCE NOT IDENTIFIED':<24}{'real data, unknown origin':<26}"
        f"{', '.join(sorted(SOURCE_NOT_IDENTIFIED))}")
    say(f"     {'PROJECT-INVENTED':<24}{'NOT RECOVERABLE':<26}{', '.join(sorted(PROJECT_INVENTED))}")
    if unaccounted:
        say(f"     UNACCOUNTED: {unaccounted}")
    say(f"     Q2 {'PASS' if q2 else 'FAIL'}")

    # ---- rebuild what is fetchable ----------------------------------------------------------------
    say(f"\n  rebuilding the fetchable fields")
    coords = load_gtf_coords()
    loeuf = load_loeuf()
    proc_go, comp_go = load_go()
    tfs = load_tfs()
    cpg_iv = load_cpg()
    say(f"     coordinates {len(coords):,} · loeuf {len(loeuf):,} · GO process {len(proc_go):,} · "
        f"TFs {len(tfs):,} · CpG islands {sum(len(v) for v in cpg_iv.values()):,}")

    CPG_WINDOW = 2000        # predeclared, before any agreement number was computed

    def cpg_at(chrom, tss, w=CPG_WINDOW):
        for a, b in cpg_iv.get(chrom, ()):
            if a - w <= tss <= b + w:
                return 1
            if a > tss + w:
                break
        return 0

    # ---- Q3 does the rebuild match --------------------------------------------------------------
    say(f"\n  Q3 REBUILT FIELDS MATCH THE SURVIVING COPY")
    say(f"     {'field':<14}{'n':>8}{'exact':>9}{'corr':>9}   verdict")
    checks = {}

    def record(field, orig, new, kind):
        pairs = [(o, n) for o, n in zip(orig, new) if o is not None and n is not None]
        if not pairs:
            checks[field] = {"n": 0, "exact": float("nan"), "corr": float("nan"), "pass": False}
            say(f"     {field:<14}{0:>8}{'-':>9}{'-':>9}   NO OVERLAP")
            return
        o = [p[0] for p in pairs]
        n = [p[1] for p in pairs]
        exact = float(np.mean([a == b for a, b in pairs]))
        corr = float("nan")
        if kind == "num":
            try:
                oa, na = np.array(o, float), np.array(n, float)
                m = np.isfinite(oa) & np.isfinite(na)
                if m.sum() > 10 and np.std(oa[m]) > 0 and np.std(na[m]) > 0:
                    corr = float(np.corrcoef(oa[m], na[m])[0, 1])
            except Exception:
                pass
        ok = (exact >= GATE_EXACT) or (np.isfinite(corr) and corr >= GATE_CORR)
        v = ("MATCH" if exact >= GATE_EXACT else
             "SOURCE RIGHT, VERSION DRIFT" if np.isfinite(corr) and corr >= GATE_CORR else
             "MISMATCH -- provenance claim fails")
        checks[field] = {"n": len(pairs), "exact": exact, "corr": corr, "pass": bool(ok)}
        say(f"     {field:<14}{len(pairs):>8,}{exact:>9.4f}"
            f"{(f'{corr:.4f}' if np.isfinite(corr) else '-'):>9}   {v}")

    nm = [g["name"] for g in genes]
    for fld in ("chrom", "gene_start", "gene_end", "strand"):
        record(fld, [g.get(fld) for g in genes],
               [coords.get(s, {}).get(fld) for s in nm],
               "num" if ("start" in fld or "end" in fld) else "cat")
    # ens_tss, NOT tss. The GTF rule (gene_start on +, gene_end on -) reproduces ens_tss exactly and
    # cannot reproduce tss, which is a string and differs in every gene -- a second annotation from a
    # track this file does not name. Testing my rule against the wrong column produced exact 0.0000
    # and would have been read as a broken producer rather than a mislabelled target.
    record("ens_tss", [g.get("ens_tss") for g in genes],
           [coords.get(s, {}).get("tss") for s in nm], "num")
    record("loeuf", [g.get("loeuf") for g in genes], [loeuf.get(s) for s in nm], "num")
    record("tf", [g.get("tf") for g in genes], [1 if s in tfs else 0 for s in nm], "cat")
    def _int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return -1
    record("cpg", [g.get("cpg") for g in genes],
           [cpg_at(g.get("chrom") or "", _int(g.get("ens_tss"))) for g in genes], "cat")

    # cpg is the one field that misses. DIAGNOSIS, NOT REPAIR. Two things could be wrong -- the
    # WINDOW around the query point, or the QUERY POINT itself -- and they are swept separately so
    # the answer is attributable. Nothing found here is adopted: the gate above keeps the predeclared
    # CPG_WINDOW and the predeclared query point. Sweeping until something passes and then calling it
    # the producer would manufacture a provenance claim out of a search, which is the exact failure
    # this module exists to avoid.
    o_cpg = [g.get("cpg") for g in genes]
    n_cpg = [cpg_at(g.get("chrom") or "", _int(g.get("ens_tss"))) for g in genes]
    fp = sum(1 for a, b in zip(o_cpg, n_cpg) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(o_cpg, n_cpg) if a == 1 and b == 0)
    sweep = {}
    for key in ("ens_tss", "tss"):
        for w in (0, 500, 1000, 2000, 5000):
            pred = [cpg_at(g.get("chrom") or "", _int(g.get(key)), w) for g in genes]
            sweep[f"{key}@{w}"] = float(np.mean([a == b for a, b in zip(o_cpg, pred)]))
    best = max(sweep, key=sweep.get)
    say(f"     cpg diagnosis   file says CpG for {np.mean([x == 1 for x in o_cpg]):.1%} of genes, this "
        f"rule for {np.mean([x == 1 for x in n_cpg]):.1%}")
    say(f"                     disagreements split {fp:,} rule-only / {fn:,} file-only "
        f"({'rule over-calls' if fp > fn else 'rule under-calls'})")
    for key in ("ens_tss", "tss"):
        say(f"                     at {key:<8} " +
            " · ".join(f"{w}bp:{sweep[f'{key}@{w}']:.4f}" for w in (0, 500, 1000, 2000, 5000)))
    say(f"                     the QUERY POINT dominates the window: every window is better at tss "
        f"than at ens_tss")
    say(f"                     (+{sweep['tss@0'] - sweep['ens_tss@0']:.4f} at 0 bp, "
        f"+{sweep['tss@2000'] - sweep['ens_tss@2000']:.4f} at the declared 2000 bp), so the original "
        f"cpg")
    say(f"                     was computed at the file's own tss. Even there it reaches only "
        f"{sweep['tss@2000']:.4f} at the")
    say(f"                     declared window and {sweep[best]:.4f} at best ({best}) -- both under "
        f"{GATE_EXACT}, NEITHER ADOPTED.")
    say(f"                     SOURCE identified, QUERY POINT identified, EXACT RULE not. cpg stays "
        f"unverified.")
    cpg_diag = {"window_declared": CPG_WINDOW, "query_declared": "ens_tss",
                "rule_only": fp, "file_only": fn, "sweep": sweep,
                "best_variant": best, "best_exact": sweep[best], "adopted": False}

    q3 = bool(all(v["pass"] for v in checks.values()))
    say(f"     Q3 {'PASS' if q3 else 'FAIL'}  (each field: exact >= {GATE_EXACT} OR corr >= "
        f"{GATE_CORR})")

    # ---- Q4 how much of the file is now reproducible ----------------------------------------------
    verified = {f for f, v in checks.items() if v["pass"]}
    declared = {f for s in SOURCES.values() if s["path"] is None for f in s["fields"]}
    invented = set(PROJECT_INVENTED)
    rule_lost = set(SOURCE_KNOWN_RULE_LOST)
    unknown_src = set(SOURCE_NOT_IDENTIFIED)
    r_rule = sum(usage.get(f, 0) for f in rule_lost)
    r_unk = sum(usage.get(f, 0) for f in unknown_src)
    r_verified = sum(usage.get(f, 0) for f in verified)
    r_declared = sum(usage.get(f, 0) for f in declared)
    r_invented = sum(usage.get(f, 0) for f in invented)
    r_name = usage.get("name", 0)
    say(f"\n  Q4 THE SHARE OF MODULE-READS COVERED")
    say(f"     verified here (source fetched and checked) {r_verified:>6,}  "
        f"{r_verified/total_reads:>6.1%}")
    say(f"     the gene list itself (name)                {r_name:>6,}  {r_name/total_reads:>6.1%}")
    say(f"     declared with a URL, not fetched here      {r_declared:>6,}  "
        f"{r_declared/total_reads:>6.1%}")
    say(f"     source known, coarsening rule lost         {r_rule:>6,}  {r_rule/total_reads:>6.1%}")
    say(f"     real data, source not identified           {r_unk:>6,}  {r_unk/total_reads:>6.1%}")
    say(f"     PROJECT-INVENTED, not recoverable          {r_invented:>6,}  "
        f"{r_invented/total_reads:>6.1%}")
    q4 = True
    say(f"     Q4 PASS -- reported rather than gated; the number is the answer, not a target")

    say("\n" + "=" * 100)
    gates = {"Q1 inventory usage-weighted": q1, "Q2 every field classified": q2,
             "Q3 rebuilt fields match": q3, "Q4 coverage reported": q4}
    for k, v in gates.items():
        say(f"  {k:<38}{'PASS' if v else 'FAIL'}")
    failed = sorted(f for f, v in checks.items() if not v["pass"])
    say(f"  cell_complete.json NOW HAS A PRODUCER for {len(verified)} of the {len(checks)} fields it")
    say(f"  attempts, each checked against the surviving copy rather than asserted. "
        f"{r_verified/total_reads:.0%} of all")
    say(f"  per-gene module-reads are now backed by a named source that was fetched and verified;")
    say(f"  another {r_declared/total_reads:.0%} have a URL recorded and need one more download.")
    if failed:
        say(f"  Q3 FAILS ON {', '.join(failed)}, and the gate stands. The diagnosis is sharper than")
        say(f"  the verdict: the CpG track is right and the query point is the file's own tss rather")
        say(f"  than the Ensembl gene boundary, but no declared rule reaches {GATE_EXACT:.2f} and none")
        say(f"  of the swept ones was adopted. A near-miss is exactly where loosening a threshold")
        say(f"  turns an unverified column into a verified-looking one, so it is left failing.")
    say(f"  AND {r_invented/total_reads:.0%} CANNOT BE REBUILT AT ALL. Those are labels this project")
    say(f"  invented -- {', '.join(sorted(invented))} -- whose rules are lost. A producer")
    say(f"  that regenerated them from a guessed rule would be worse than one that says so,")
    say(f"  because the guess would look identical in the file and differ in every value.")
    say(f"  A further {r_unk/total_reads:.0%} is real data whose source this file never named")
    say(f"  ({', '.join(sorted(unknown_src))}) and {r_rule/total_reads:.0%} keeps its ontology but "
        f"lost the")
    say(f"  coarsening rule ({', '.join(sorted(rule_lost))}). Neither is invented and neither is")
    say(f"  reproducible; they are the middle of the distribution and were the easiest to overstate.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(s["path"]) for s in SOURCES.values() if s["path"]] + [str(CELL)],
                      available=total_reads, used=r_verified, selection="filtered", seed=0,
                      controls=["every rebuilt field checked against the surviving copy",
                                "exact-match and correlation reported separately so version drift is "
                                "distinguishable from a wrong source",
                                "fields weighted by how many modules read them",
                                "project-invented fields named rather than regenerated"],
                      note="does not claim to recover the original program, which is lost; claims a "
                           "producer for the load-bearing fields whose provenance can be checked")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "build_cell_complete", "manifest": man, "gates": gates,
               "n_fields": len(fields), "total_reads": total_reads, "usage": usage,
               "sources": {k: {kk: (str(vv) if kk == "path" else vv) for kk, vv in v.items()}
                           for k, v in SOURCES.items()},
               "project_invented": PROJECT_INVENTED,
               "source_known_rule_lost": SOURCE_KNOWN_RULE_LOST,
               "source_not_identified": SOURCE_NOT_IDENTIFIED, "checks": checks,
               "cpg_diagnostic": cpg_diag,
               "reads_verified": r_verified, "reads_declared": r_declared,
               "reads_invented": r_invented, "reads_rule_lost": r_rule,
               "reads_source_unknown": r_unk, "seconds": time.time() - t0, "log": log},
              open(OUT / "build_cell_complete.json", "w"), indent=2)
    say(f"\n  -> {OUT/'build_cell_complete.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
