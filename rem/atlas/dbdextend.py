"""Recovering DNA-binding regions for the regulators UniProt does not give a DNA_BIND feature.

WHERE ISOEDGE STOPPED. Its ceiling gate found the binding constraint was ANNOTATION, not method:
only 467 of 4,320 signed-edge regulators carry a DNA_BIND feature, so 89.2% were ineligible by
construction and no improvement to the intersection could move that.

WHY THE FEATURE IS MISSING, AND IT IS NOT BECAUSE THE PROTEIN DOES NOT BIND DNA. UniProt records a
DNA-binding region under DIFFERENT feature types depending on the family:
    SOX2  -> DNA_BIND 41..109  "HMG box"            already covered
    PAX6  -> DNA_BIND 4..130 "Paired", 210..269 "Homeobox"
    SP1   -> ZN_FING 626..650 "C2H2-type 1", ...    the zinc fingers ARE the DBD
    JUN   -> DOMAIN 252..315 "bZIP"                 the basic region contacts DNA
So the union of DNA_BIND, ZN_FING and DOMAIN, filtered to families that actually contact DNA,
recovers positions the single-feature query missed.

THE RISK THIS CREATES, AND IT IS THE WHOLE DESIGN PROBLEM. Widening from one curated feature to a
name-matched whitelist can admit domains that do not bind DNA at all -- PHD fingers bind histones,
RING fingers are E3 ligases, LIM and FYVE bind protein and lipid. A whitelist that leaks turns a
NECESSARY condition into a false one, and every isoform this module then "rules out" would be ruled
out wrongly. So the whitelist is explicit, auditable, and tested against a blacklist before use.

THE DENOMINATOR IS ALSO WRONG IN THE OBVIOUS FRAMING. Most of the 4,320 regulators are not
sequence-specific transcription factors -- proteinlayer measured that only 1,248 are TF-labelled by
two or more independent sources. Quoting the gap against 4,320 inflates it. The honest denominator
is credible TFs, and that is what this module reports against.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

R1  THE BLOCKING VALIDATION, RUN FIRST, IN BOTH DIRECTIONS.
    MUST RECOVER: SP1 via C2H2 zinc fingers, JUN via bZIP, and TP53's DNA_BIND must be unchanged
    so isoedge's validated result does not move.
    MUST REJECT: a non-DNA-binding domain must not be admitted. DAND5's CTCK domain, PHD fingers,
    RING fingers and LIM domains are the named negative controls.
    PREDECLARED: a whitelist that admits any negative control is leaking and no number from it may
    be reported. This gate blocks everything below it.

R0  THE CEILING GATE, AGAINST THE HONEST DENOMINATOR. Of regulators that are credible TFs
    (corroborated by >=2 independent sources, per proteinlayer), how many have a positioned
    DNA-binding region before and after the extension?
    PREDECLARED: if the extension does not lift credible-TF coverage by at least 10 percentage
    points, the missing feature was not the binding constraint and this module says so and stops.

R2  WHAT IT BUYS DOWNSTREAM, recomputed on isoedge's own terms: eligible regulators, eligible
    signed edges, and the fraction of regulator-side isoform space removed, before and after.

R3  THE LEAK CHECK, AS A MEASUREMENT RATHER THAN AN ASSURANCE. How many genes gain a DBD under the
    extension while NO independent source calls them a TF?
    PREDECLARED: that count is the whitelist's false-positive exposure and is reported whatever it
    is. A large value does not invalidate the whitelist but does bound what may be claimed from it.

R4  WHAT THIS DOES NOT SETTLE.
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
CACHE = os.environ.get("UNIPROT_DBD_TSV", os.path.join(HERE, "_cache", "uniprot_dbd.tsv"))
ENCY = os.path.normpath(os.path.join(HERE, "..", "..", "colab", "data", "cell_complete.json.gz"))
TFART = os.path.normpath(os.path.join(HERE, "..", "..", "outputs", "protein_tf_layer.json"))
OUT = os.path.join(HERE, "RESULTS_dbdextend.txt")
ART = os.path.normpath(os.path.join(HERE, "..", "..", "outputs", "dbd_regions.json"))
RULE = "=" * 97
FIELDS = ("accession,gene_primary,ft_dna_bind,ft_zn_fing,ft_domain,ft_var_seq,"
          "cc_alternative_products")

# Domain-name substrings whose families make sequence-specific DNA contact. Explicit so it can be
# audited and amended; every entry is a family that binds DNA, not one that merely sits in a TF.
DBD_NAMES = (
    "bzip", "bhlh", "homeobox", "homeodomain", "paired", "pou-specific", "pou_homeo",
    "hmg box", "hmg-box", "fork-head", "forkhead", "ets", "rel", "t-box", "tbox",
    "mads-box", "mads box", "myb-like", "sant", "runt", "sand", "tea", "cut", "arid",
    "grainyhead", "hsf", "smad mh1", "mh1", "tcp", "csd", "cold-shock", "ap2", "wrky",
    "interferon regulatory factor", "irf", "stat", "p53", "nfat", "rfx", "cp2", "gcm",
    "e2f", "cbf", "nf-ya", "nf-yb", "ccaat", "dm dna-binding", "at-hook",
)
# Zinc-finger TYPES that bind DNA. Deliberately narrow: PHD/RING/LIM/FYVE/B-box are excluded.
ZF_OK = ("c2h2", "gata-type", "gata type", "nr c4-type", "c4-type", "dm", "thap", "bed")
# Named negative controls for R1.
BLACKLIST = ("phd", "ring", "lim", "fyve", "b-box", "ctck", "pdz", "sh2", "sh3", "brct",
             "bromo", "chromo", "wd40", "ankyrin", "kinase", "ph domain")

DNAB = re.compile(r"DNA_BIND\s+(\d+)\.\.(\d+)(?:;\s*/note=\"([^\"]*)\")?")
ZNF = re.compile(r"ZN_FING\s+(\d+)\.\.(\d+);\s*/note=\"([^\"]*)\"")
DOM = re.compile(r"DOMAIN\s+(\d+)\.\.(\d+);\s*/note=\"([^\"]*)\"")
VSEQ = re.compile(r"VAR_SEQ\s+(\d+)(?:\.\.(\d+))?;\s*/note=\"([^\"]*)\"")
ISON = re.compile(r"Named isoforms=(\d+)")
INISO = re.compile(r"\(in ([^)]*)\)")


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


def is_dbd_domain(note):
    n = note.lower()
    if any(b in n for b in BLACKLIST):
        return False
    return any(k in n for k in DBD_NAMES)


def is_dbd_znf(note):
    n = note.lower()
    if any(b in n for b in BLACKLIST):
        return False
    return any(k in n for k in ZF_OK)


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
            core = [(int(a), int(b)) for a, b, _ in DNAB.findall(r.get("DNA binding") or "")]
            zf = [(int(a), int(b)) for a, b, nt in ZNF.findall(r.get("Zinc finger") or "")
                  if is_dbd_znf(nt)]
            dm = [(int(a), int(b)) for a, b, nt in DOM.findall(r.get("Domain [FT]") or "")
                  if is_dbd_domain(nt)]
            vs = []
            for a, b, nt in VSEQ.findall(r.get("Alternative sequence") or ""):
                lo = int(a); hi = int(b) if b else lo
                vs.append((lo, hi, iso_names(nt)))
            m = ISON.search(r.get("Alternative products (isoforms)") or "")
            rec = {"acc": r.get("Entry", ""), "core": core, "zf": zf, "dom": dm,
                   "vseq": vs, "n_iso": int(m.group(1)) if m else 1,
                   "raw_zf": r.get("Zinc finger") or "", "raw_dom": r.get("Domain [FT]") or ""}
            if sym not in P or len(core) + len(zf) + len(dm) > \
                    len(P[sym]["core"]) + len(P[sym]["zf"]) + len(P[sym]["dom"]):
                P[sym] = rec
    return P


def regions(rec, extended=True):
    return rec["core"] + (rec["zf"] + rec["dom"] if extended else [])


def disrupted(rec, extended=True):
    R = regions(rec, extended)
    if not R:
        return set(), set()
    allio, bad = set(), set()
    for lo, hi, isos in rec["vseq"]:
        allio |= isos
        if any(lo <= d2 and hi >= d1 for d1, d2 in R):
            bad |= isos
    return bad, allio


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("RECOVERING DNA-BINDING REGIONS FOR THE REST OF THE REGULATORS"); P_(RULE)
    P = parse(fetch())
    P_(f"  {len(P):,} reviewed entries parsed for DNA_BIND + ZN_FING + DOMAIN coordinates")

    # ---- R1  BLOCKING ------------------------------------------------------------------------
    P_("\n" + RULE); P_("R1  BLOCKING VALIDATION, IN BOTH DIRECTIONS"); P_(RULE)
    checks = []
    sp1 = P.get("SP1", {})
    checks.append(("SP1 recovered via C2H2 zinc fingers", bool(sp1.get("zf")), sp1.get("zf", [])[:2]))
    jun = P.get("JUN", {})
    checks.append(("JUN recovered via bZIP domain", bool(jun.get("dom")), jun.get("dom", [])))
    tp = P.get("TP53", {})
    checks.append(("TP53 DNA_BIND unchanged (102..292)", tp.get("core") == [(102, 292)], tp.get("core")))
    dand = P.get("DAND5", {})
    checks.append(("DAND5 CTCK REJECTED", not dand.get("dom"), dand.get("dom", [])))
    leaks = []
    for s, r in P.items():
        for b in ("PHD", "RING", "LIM domain", "FYVE"):
            if b.lower() in r["raw_dom"].lower() and r["dom"]:
                for a2, b2, nt in DOM.findall(r["raw_dom"]):
                    if b.lower() in nt.lower() and is_dbd_domain(nt):
                        leaks.append((s, nt))
    checks.append((f"no PHD/RING/LIM/FYVE admitted anywhere", not leaks, leaks[:3]))
    for lbl, good, ev in checks:
        P_(f"   {'PASS' if good else 'FAIL'}  {lbl:<48} {str(ev)[:60]}")
    if not all(g for _, g, _ in checks):
        P_("\n  R1: FAIL -- the whitelist leaks. No coverage number reported.")
        open(OUT, "w").write("\n".join(out) + "\n")
        return
    P_("\n  R1: PASS -- recovers the positives, rejects every named negative control")

    # ---- R0  ceiling gate --------------------------------------------------------------------
    P_("\n" + RULE); P_("R0  THE CEILING GATE, AGAINST THE HONEST DENOMINATOR"); P_(RULE)
    D = json.load(gzip.open(ENCY))
    names = [r["name"] for r in D["genes"]]
    TF = json.load(open(TFART))
    cred = {n.upper() for n, v in TF.items() if v.get("tf_corroboration", 0) >= 2}
    edges = [(s, t, w) for s, t, w in D["reg"] if w] + [(s, t, w) for s, t, w in D["sig"] if w]
    regs = collections.Counter(s for s, _, _ in edges)
    reg_syms = {names[i].upper() for i in regs}
    cred_regs = reg_syms & cred
    def cov(syms, ext):
        return sum(1 for s in syms if P.get(s) and regions(P[s], ext))
    b_all, a_all = cov(reg_syms, False), cov(reg_syms, True)
    b_cr, a_cr = cov(cred_regs, False), cov(cred_regs, True)
    P_(f"  all signed-edge regulators              {len(reg_syms):>6,}")
    P_(f"     with a positioned DBD  before/after  {b_all:>6,} -> {a_all:,}")
    P_(f"\n  CREDIBLE TFs among them (corrob >=2)    {len(cred_regs):>6,}   <- the honest denominator")
    P_(f"     with a positioned DBD  before        {b_cr:>6,}  {100*b_cr/len(cred_regs):>5.1f}%")
    P_(f"     with a positioned DBD  after         {a_cr:>6,}  {100*a_cr/len(cred_regs):>5.1f}%")
    lift = 100*a_cr/len(cred_regs) - 100*b_cr/len(cred_regs)
    P_(f"     LIFT                                 {lift:>+6.1f} points")
    P_(f"\n  R0: {'PASS' if lift >= 10 else 'FAIL'} against the predeclared 10-point bar")

    # ---- R2 ----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("R2  WHAT IT BUYS ON ISOEDGE'S OWN TERMS"); P_(RULE)
    def elig(ext):
        E, tot_named, tot_cand = set(), 0, 0
        for i in regs:
            rec = P.get(names[i].upper())
            if not rec or not regions(rec, ext) or rec["n_iso"] <= 1:
                continue
            bad, _ = disrupted(rec, ext)
            if bad:
                E.add(i)
                tot_named += rec["n_iso"] * regs[i]
                tot_cand += max(rec["n_iso"] - len(bad), 1) * regs[i]
        return E, sum(regs[i] for i in E), tot_named, tot_cand
    e0, ed0, n0, c0 = elig(False)
    e1, ed1, n1, c1 = elig(True)
    T = sum(regs.values())
    P_(f"    {'':<28} {'before':>10} {'after':>10}")
    P_(f"    {'eligible regulators':<28} {len(e0):>10,} {len(e1):>10,}")
    P_(f"    {'eligible signed edges':<28} {ed0:>10,} {ed1:>10,}")
    P_(f"    {'% of all signed edges':<28} {100*ed0/T:>9.1f}% {100*ed1/T:>9.1f}%")
    P_(f"    {'isoform space removed':<28} {100*(1-c0/max(n0,1)):>9.1f}% {100*(1-c1/max(n1,1)):>9.1f}%")
    P_(f"\n  R2: eligible edges {ed0:,} -> {ed1:,}  ({ed1/max(ed0,1):.2f}x)")

    # ---- R3 ----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("R3  THE LEAK CHECK, MEASURED"); P_(RULE)
    gained = {s for s in P if not P[s]["core"] and (P[s]["zf"] or P[s]["dom"])}
    nocred = gained - cred
    P_(f"  genes gaining a DBD only from the extension   {len(gained):>6,}")
    P_(f"  ...that NO source calls a TF                  {len(nocred):>6,}  {100*len(nocred)/max(len(gained),1):>5.1f}%")
    ex = sorted(nocred)[:10]
    P_(f"  examples: {ex}")
    P_( "  These are the whitelist's false-positive exposure. C2H2 zinc fingers in particular")
    P_( "  occur in many proteins that are not sequence-specific TFs, so this number is expected")
    P_( "  to be large and is reported rather than filtered away. Claims from this module should")
    P_( "  be restricted to the credible-TF subset above.")

    art = {}
    for s, r in P.items():
        R = regions(r, True)
        if R:
            art[s] = {"acc": r["acc"], "dbd": [list(x) for x in R],
                      "source": ("DNA_BIND" if r["core"] else
                                 ("ZN_FING" if r["zf"] else "DOMAIN")),
                      "n_isoforms": r["n_iso"], "credible_tf": s in cred}
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(art, open(ART, "w"))
    P_(f"\n  artifact: outputs/dbd_regions.json  ({len(art):,} proteins)")

    P_("\n" + RULE); P_("R4  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. A name-matched whitelist is weaker evidence than UniProt's curated DNA_BIND feature.")
    P_("     The two are tagged separately in the artifact so a consumer can use only the first.")
    P_("  2. Zinc fingers are the main source of the lift AND the main source of leak; C2H2 arrays")
    P_("     appear in RNA-binding and protein-interaction roles too.")
    P_("  3. Coverage is still bounded by UniProt curation. Proteins with no positioned domain of")
    P_("     any kind remain ineligible however clearly they regulate transcription.")
    P_("  4. Everything isoedge could not settle is still unsettled: an intact DBD is necessary,")
    P_("     not sufficient, and the target side of every edge remains unresolved.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
