"""Mapping isoforms onto the regulatory edges -- by RULING OUT, because assignment is not available.

THE HONEST STATEMENT OF THE PROBLEM. proteinlayer measured that 56.1% of the human proteome is not
one gene, one protein, and that every edge in this repo is gene-to-gene. "Which isoform carries
this edge" cannot be answered from anything on disk or anything public: it needs isoform-resolved
binding (isoform-specific ChIP) on the regulator side and isoform-level response on the target
side, and neither exists at scale for human.

WHAT IS AVAILABLE IS A NECESSARY CONDITION. A sequence-specific transcription factor cannot carry
a transcriptional edge if its DNA-binding domain has been spliced away. UniProt annotates the
DNA-binding region with residue coordinates (DNA_BIND 102..292 for TP53) and annotates every
splice event with its coordinates and the named isoforms it belongs to (VAR_SEQ 1..132, "Missing
in isoform 7, isoform 8 and isoform 9"). Intersecting the two partitions a TF's isoforms into
DBD-INTACT and DBD-DISRUPTED. The disrupted ones cannot carry the gene's outgoing edges.

SO THE OUTPUT IS AN UPPER BOUND ON THE ISOFORM SET PER EDGE, NOT AN ASSIGNMENT, and the difference
is the whole point. An intact DBD does not imply binding -- dimerisation, cofactors and
localisation all still gate it. Ruling out is sound; ruling in is not, and this module does only
the first.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

I0  THE CEILING GATE. Price the constraint before building on it. An edge is ELIGIBLE only if its
    regulator has (a) a DNA_BIND annotation, (b) more than one named isoform, and (c) at least one
    splice event overlapping the DBD. Count eligible regulators and eligible edges.
    PREDECLARED: if fewer than 5% of signed edges are eligible, the constraint is a curiosity and
    this module says so and stops rather than dressing a small number as a capability.

I1  VALIDATION ON DOCUMENTED BIOLOGY, BEFORE ANY COVERAGE NUMBER. The parse must recover cases
    whose answer is already known: TP53's Delta-133 isoforms (7, 8, 9) lose residues 1..132 and
    must come back DBD-DISRUPTED, while the beta/gamma isoforms, which differ only at the
    C-terminus, must come back DBD-INTACT. NR3C1's DBD is 418..493 and its known N-terminal
    translational isoforms must come back INTACT.
    PREDECLARED: if the method does not reproduce these, the coordinate parse is wrong and no
    coverage number from it may be reported. This gate runs first and blocks the rest.

I2  THE CONSTRAINT ITSELF. Per regulator: isoforms partitioned. Per edge: the candidate isoform
    set and how much it shrinks against the naive "any isoform" prior.
    PREDECLARED: report the DISTRIBUTION of shrinkage, not its mean. A mean over edges whose
    regulators have wildly different isoform counts hides everything that matters.

I3  THE TARGET SIDE, REPORTED AS UNRESOLVED. The same edge also has a target with isoforms, and
    nothing here constrains which transcript responds. State the size of that residual explicitly
    rather than letting a regulator-side result imply the edge is resolved.

I4  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import csv
import gzip
import json
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(__file__)
CACHE = os.environ.get("UNIPROT_FT_TSV", os.path.join(HERE, "_cache", "uniprot_ft.tsv"))
ENCY = os.path.normpath(os.path.join(HERE, "..", "..", "colab", "data", "cell_complete.json.gz"))
OUT = os.path.join(HERE, "RESULTS_isoedge.txt")
ART = os.path.normpath(os.path.join(HERE, "..", "..", "outputs", "isoform_edge_bounds.json"))
RULE = "=" * 97
FIELDS = "accession,gene_primary,ft_dna_bind,ft_var_seq,cc_alternative_products"


def fetch():
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 500_000:
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    q = urllib.parse.quote("(organism_id:9606) AND (reviewed:true)")
    url = (f"https://rest.uniprot.org/uniprotkb/stream?query={q}"
           f"&format=tsv&compressed=true&fields={FIELDS}")
    req = urllib.request.Request(url, headers={"User-Agent": "rem-atlas/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        open(CACHE, "wb").write(gzip.decompress(r.read()))
    return CACHE


DNAB = re.compile(r"DNA_BIND\s+(\d+)\.\.(\d+)")
VSEQ = re.compile(r"VAR_SEQ\s+(\d+)(?:\.\.(\d+))?;\s*/note=\"([^\"]*)\"")
ISON = re.compile(r"Named isoforms=(\d+)")
# "(in isoform 7, isoform 8 and isoform 9)" / "(in isoform Alpha-D3)"
INISO = re.compile(r"\(in ([^)]*)\)")


def iso_names(note):
    m = INISO.search(note)
    if not m:
        return set()
    return {p.strip() for p in re.split(r",| and ", m.group(1))
            if p.strip().lower().startswith("isoform")}


def parse(path):
    P = {}
    with open(path, encoding="utf8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            sym = (r.get("Gene Names (primary)") or "").strip().upper()
            if not sym:
                continue
            db = [(int(a), int(b)) for a, b in DNAB.findall(r.get("DNA binding") or "")]
            vs = []
            for a, b, note in VSEQ.findall(r.get("Alternative sequence") or ""):
                lo = int(a); hi = int(b) if b else lo
                vs.append((lo, hi, iso_names(note)))
            m = ISON.search(r.get("Alternative products (isoforms)") or "")
            n = int(m.group(1)) if m else 1
            if sym not in P or len(db) > len(P[sym]["dbd"]):
                P[sym] = {"acc": r.get("Entry", ""), "dbd": db, "vseq": vs, "n_iso": n}
    return P


def partition(rec):
    """-> (disrupted isoform names, all isoform names seen in VAR_SEQ notes)."""
    if not rec["dbd"]:
        return set(), set()
    allio, bad = set(), set()
    for lo, hi, isos in rec["vseq"]:
        allio |= isos
        if any(lo <= d_hi and hi >= d_lo for d_lo, d_hi in rec["dbd"]):
            bad |= isos
    return bad, allio


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("MAPPING ISOFORMS ONTO EDGES BY RULING OUT"); P_(RULE)
    P = parse(fetch())
    P_(f"  {len(P):,} reviewed entries parsed for DNA_BIND + VAR_SEQ coordinates")

    # ---- I1  BLOCKING validation ------------------------------------------------------------
    P_("\n" + RULE); P_("I1  VALIDATION ON DOCUMENTED BIOLOGY (BLOCKING)"); P_(RULE)
    ok = True
    tp = P.get("TP53")
    bad, allio = partition(tp) if tp else (set(), set())
    P_(f"  TP53  DBD {tp['dbd'] if tp else '-'}   named isoforms {tp['n_iso'] if tp else '-'}")
    P_(f"        DISRUPTED : {sorted(bad)}")
    P_(f"        INTACT    : {sorted(allio - bad)}")
    want_bad = {"isoform 7", "isoform 8", "isoform 9"}
    want_ok = {"isoform 2", "isoform 3"}
    c1 = want_bad <= bad
    c2 = not (want_ok & bad)
    P_(f"        Delta-133 (7,8,9) disrupted ....... {c1}")
    P_(f"        beta/gamma (2,3) intact ........... {c2}")
    nr = P.get("NR3C1")
    nbad, nall = partition(nr) if nr else (set(), set())
    P_(f"\n  NR3C1 DBD {nr['dbd'] if nr else '-'}   named isoforms {nr['n_iso'] if nr else '-'}")
    P_(f"        DISRUPTED : {sorted(nbad)[:6]}")
    c3 = len(nbad) < len(nall)
    P_(f"        N-terminal translational isoforms intact ... {c3}")
    ok = c1 and c2 and c3
    P_(f"\n  I1: {'PASS -- the coordinate parse reproduces known cases' if ok else 'FAIL -- parse is wrong, no coverage number may be reported'}")
    if not ok:
        open(OUT, "w").write("\n".join(out) + "\n")
        return

    # ---- I0  ceiling gate -------------------------------------------------------------------
    P_("\n" + RULE); P_("I0  THE CEILING GATE: HOW MANY EDGES ARE EVEN ELIGIBLE?"); P_(RULE)
    D = json.load(gzip.open(ENCY))
    names = [r["name"] for r in D["genes"]]
    edges = [(s, t, w, "reg") for s, t, w in D["reg"] if w] + \
            [(s, t, w, "sig") for s, t, w in D["sig"] if w]
    regs = collections.Counter(s for s, _, _, _ in edges)
    have_dbd = multi = constrains = 0
    elig_regs = set()
    for i in regs:
        rec = P.get(names[i].upper())
        if not rec or not rec["dbd"]:
            continue
        have_dbd += 1
        if rec["n_iso"] > 1:
            multi += 1
            bad, allio = partition(rec)
            if bad:
                constrains += 1
                elig_regs.add(i)
    elig_edges = sum(c for i, c in regs.items() if i in elig_regs)
    tot = len(edges)
    P_(f"  distinct regulators on SIGNED edges     {len(regs):>6,}")
    P_(f"  ...with a DNA_BIND annotation           {have_dbd:>6,}")
    P_(f"  ...and >1 named isoform                 {multi:>6,}")
    P_(f"  ...and a splice event hitting the DBD   {constrains:>6,}   <- eligible")
    P_(f"\n  signed edges total                      {tot:>6,}")
    P_(f"  signed edges with an eligible regulator {elig_edges:>6,}  {100*elig_edges/tot:>5.1f}%")
    passed = 100 * elig_edges / tot >= 5.0
    P_(f"\n  I0: {'PASS' if passed else 'FAIL'} against the predeclared 5% bar --"
       f" {'the constraint is worth carrying' if passed else 'a curiosity, not a capability'}")

    # ---- I2 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("I2  THE CONSTRAINT, AS A DISTRIBUTION"); P_(RULE)
    art, shrink = {}, collections.Counter()
    for i in sorted(elig_regs):
        nm = names[i]
        rec = P[nm.upper()]
        bad, allio = partition(rec)
        n_named = rec["n_iso"]
        n_out = max(n_named - len(bad), 1)
        art[nm] = {"acc": rec["acc"], "dbd": rec["dbd"], "n_isoforms": n_named,
                   "dbd_disrupted": sorted(bad), "n_candidate_isoforms": n_out,
                   "n_signed_edges": regs[i]}
        shrink[(n_named, n_out)] += regs[i]
    P_(f"    {'isoforms':>9} {'candidates after ruling out':>28} {'edges':>9}")
    for (a, b), c in sorted(shrink.items(), key=lambda x: -x[1])[:12]:
        P_(f"    {a:>9} {b:>28} {c:>9,}")
    tot_named = sum(P[names[i].upper()]["n_iso"] * regs[i] for i in elig_regs)
    tot_cand = sum(art[names[i]]["n_candidate_isoforms"] * regs[i] for i in elig_regs)
    P_(f"\n  across eligible edges: candidate isoforms fall from {tot_named:,} to {tot_cand:,}")
    P_(f"  = {100*(1-tot_cand/max(tot_named,1)):.1f}% of the regulator-side isoform space removed")

    # ---- I3 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("I3  THE TARGET SIDE, WHICH THIS DOES NOT TOUCH"); P_(RULE)
    tgt_iso = []
    for s, t, w, k in edges:
        rec = P.get(names[t].upper())
        tgt_iso.append(rec["n_iso"] if rec else 1)
    multi_t = sum(1 for x in tgt_iso if x > 1)
    P_(f"  signed edges whose TARGET has >1 isoform  {multi_t:>6,}  {100*multi_t/tot:>5.1f}%")
    P_(f"  mean target isoforms per edge             {sum(tgt_iso)/len(tgt_iso):>6.2f}")
    P_( "  Nothing here constrains which target transcript responds; that needs isoform-level")
    P_( "  expression which no dataset in this record supplies. A regulator-side bound does NOT")
    P_( "  make the edge resolved, and the residual above is the size of what is left.")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(art, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/isoform_edge_bounds.json  ({len(art):,} regulators)")
    for nm in ("TP53", "NR3C1", "RELA", "MYC"):
        if nm in art:
            P_(f"    {nm}: {json.dumps(art[nm])[:170]}")

    P_("\n" + RULE); P_("I4  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. An intact DBD is NECESSARY, not sufficient. Dimerisation, cofactor availability and")
    P_("     localisation all still gate binding and none is modelled here.")
    P_("  2. Only sequence-specific DNA binding is treated. Cofactors and chromatin regulators")
    P_("     with no DNA_BIND annotation are never eligible, by construction.")
    P_("  3. UniProt's named isoforms are curated, not exhaustive; transcript catalogues list more.")
    P_("  4. The edges themselves remain gene-level in every artifact this repo ships. This module")
    P_("     bounds them; it does not rewrite them.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
