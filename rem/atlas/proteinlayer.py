"""The protein layer and the splicing layer the encyclopedia does not have, plus a labelled TF net.

WHAT IS ACTUALLY MISSING, CHECKED RATHER THAN ASSUMED. The working assumption was "we know the
genes and the mRNA and the splicing patterns, so fetch the protein data". Two thirds of that is
false. cell_complete.json has NO transcript, isoform or exon layer at all -- its `ncrna` key holds
zero entries and there is exactly one record per GENE. Its protein side is `struct` (13 genes) and
`fold` (1 gene), plus abundance and PTM COUNTS. There are no sequences, no domains, no isoforms.

So the encyclopedia silently assumes one gene -> one protein. That assumption is the thing to
measure, not to inherit.

ONE FETCH SUPPLIES BOTH. UniProt's reviewed human proteome carries sequence length and mass, Pfam
domains, evidence-coded subcellular location, modified residues, keywords, NCBI GeneID -- and
`cc_alternative_products`, which names the splicing isoforms. TP53 comes back with 9. So the
splicing layer is not a separate acquisition; it is a column of the protein fetch.

LICENCE, BECAUSE THIS REPO HAS A PROBLEM THERE. UniProt is CC BY 4.0 -- commercially usable with
attribution, unlike the CC-BY-NC SIGNOR data currently redistributed in-tree. This layer is clean.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

P0  THE CEILING GATE. Price the fetch before running it: for each field, how many of the 16,492
    encyclopedia genes hold it today, and how many would hold it after?
    PREDECLARED: a field the encyclopedia already covers is not a gain however large the fetch.
    The fetch is justified only by fields at or near zero coverage now. State the before/after
    per field rather than one headline number.

P1  THE JOIN, WITH SYNONYM RESCUE MEASURED SEPARATELY. Match on primary symbol first, then attempt
    the unmatched against UniProt's synonym field.
    PREDECLARED: the rescue must be reported as its own number. A join that quietly folds synonym
    hits into the primary match rate cannot be audited, and symbol drift is the single most common
    silent failure in gene-level joins. Report collisions too: several accessions may carry one
    symbol.

P2  THE SPLICING LAYER, AND THE ASSUMPTION IT TESTS. Distribution of named isoforms per gene.
    PREDECLARED: the load-bearing quantity is the fraction of the proteome that is NOT one-gene-
    one-protein, because that is the assumption every per-gene record in this repo rests on --
    including the 21,955 signed edges, which are gene-to-gene and cannot address isoforms at all.
    If that fraction is large, every gene-level claim in the record carries an unmeasured
    aggregation error, and this module says so plainly rather than filing the isoform counts away.

P3  LABELLING THE TF NETWORK, FROM THREE SOURCES THAT CAN DISAGREE. UniProt keywords, CollecTRI's
    TF.category, and the encyclopedia's own `tf` flag.
    PREDECLARED: agreement is not the result -- DISAGREEMENT is. A `tf` flag that no independent
    source corroborates is a flag, not a label, and must be reported as such. Cross-tabulate all
    three before assigning any class.

P4  THE LABELLED NETWORK. Every regulator in `reg` and `sig` carries: TF class, corroboration
    count, and the sign PROVENANCE already measured (evidence vs default-activation).

P5  WHAT THIS DOES NOT SETTLE.
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
CACHE = os.environ.get("UNIPROT_TSV", os.path.join(HERE, "_cache", "uniprot_human.tsv"))
ENCY = os.path.normpath(os.path.join(HERE, "..", "..", "colab", "data", "cell_complete.json.gz"))
CTRI = os.path.normpath(os.path.join(HERE, "..", "..", "colab", "data", "networks", "collectri.csv"))
OUT = os.path.join(HERE, "RESULTS_proteinlayer.txt")
ART = os.path.normpath(os.path.join(HERE, "..", "..", "outputs", "protein_tf_layer.json"))
RULE = "=" * 97

FIELDS = ("accession,gene_primary,gene_synonym,length,mass,cc_alternative_products,"
          "xref_pfam,cc_subcellular_location,ft_mod_res,keyword,xref_geneid,protein_existence")


def fetch(force=False):
    """Stream the reviewed human proteome to a local TSV. Cached; re-runs do not refetch."""
    if os.path.exists(CACHE) and not force and os.path.getsize(CACHE) > 1_000_000:
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    q = urllib.parse.quote("(organism_id:9606) AND (reviewed:true)")
    url = (f"https://rest.uniprot.org/uniprotkb/stream?query={q}"
           f"&format=tsv&compressed=true&fields={FIELDS}")
    req = urllib.request.Request(url, headers={"User-Agent": "rem-atlas/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = gzip.decompress(r.read())
    open(CACHE, "wb").write(raw)
    return CACHE


ISO = re.compile(r"Named isoforms=(\d+)")
MOD = re.compile(r"MOD_RES\s+\d+")
LOC = re.compile(r"SUBCELLULAR LOCATION:\s*([^{.]+)")


def parse(path):
    rows = {}
    with open(path, encoding="utf8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            sym = (r.get("Gene Names (primary)") or "").strip().upper()
            if not sym:
                continue
            alt = r.get("Alternative products (isoforms)") or ""
            m = ISO.search(alt)
            kw = [k for k in (r.get("Keywords") or "").split(";") if k]
            pf = [p for p in (r.get("Pfam") or "").split(";") if p]
            loc = LOC.search(r.get("Subcellular location [CC]") or "")
            rec = {
                "acc": r.get("Entry", ""),
                "syn": [s.upper() for s in (r.get("Gene Names (synonym)") or "").split() if s],
                "len": int(r["Length"]) if (r.get("Length") or "").isdigit() else None,
                "mass": int(r["Mass"].replace(",", "")) if (r.get("Mass") or "").replace(",", "").isdigit() else None,
                "isoforms": int(m.group(1)) if m else (1 if alt.strip() else None),
                "pfam": pf,
                "loc": (loc.group(1).strip() if loc else ""),
                "n_ptm": len(MOD.findall(r.get("Modified residue") or "")),
                "kw": kw,
                "geneid": (r.get("GeneID") or "").strip().strip(";"),
                "evidence": r.get("Protein existence", ""),
            }
            # keep the longest entry when a symbol repeats
            if sym not in rows or (rec["len"] or 0) > (rows[sym]["len"] or 0):
                rows[sym] = rec
    return rows


TF_KW = {"DNA-binding", "Transcription regulation", "Activator", "Repressor"}


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("THE PROTEIN LAYER, THE SPLICING LAYER, AND A LABELLED TF NETWORK"); P_(RULE)
    P_("  fetching UniProt reviewed human proteome (CC BY 4.0)...")
    path = fetch()
    U = parse(path)
    P_(f"  {len(U):,} reviewed entries with a primary symbol"
       f"   [{os.path.getsize(path)/1e6:.1f} MB cached]")

    D = json.load(gzip.open(ENCY))
    names = [r["name"] for r in D["genes"]]
    idx = {n.upper(): i for i, n in enumerate(names)}
    N = len(names)
    gene = {r["name"].upper(): r for r in D["genes"]}

    # ---- P1  join ---------------------------------------------------------------------------
    P_("\n" + RULE); P_("P1  THE JOIN, WITH SYNONYM RESCUE REPORTED SEPARATELY"); P_(RULE)
    direct = {s for s in idx if s in U}
    syn2sym = collections.defaultdict(set)
    for s, r in U.items():
        for y in r["syn"]:
            syn2sym[y].add(s)
    rescued = {}
    for s in idx:
        if s in direct:
            continue
        c = syn2sym.get(s)
        if c and len(c) == 1:
            rescued[s] = next(iter(c))
    P_(f"  encyclopedia genes                     {N:>6,}")
    P_(f"  matched on PRIMARY symbol              {len(direct):>6,}  {100*len(direct)/N:>5.1f}%")
    P_(f"  RESCUED via UniProt synonym            {len(rescued):>6,}  {100*len(rescued)/N:>5.1f}%")
    still = N - len(direct) - len(rescued)
    P_(f"  still unmatched                        {still:>6,}  {100*still/N:>5.1f}%")
    amb = sum(1 for s in idx if s not in direct and len(syn2sym.get(s, ())) > 1)
    P_(f"  ambiguous synonyms (>1 accession), NOT rescued  {amb:>6,}")
    M = {s: U[s] for s in direct}
    M.update({s: U[t] for s, t in rescued.items()})

    # ---- P0  ceiling gate -------------------------------------------------------------------
    P_("\n" + RULE); P_("P0  THE CEILING GATE: BEFORE AND AFTER, PER FIELD"); P_(RULE)
    before = {
        "sequence length": 0,
        "molecular mass": 0,
        "isoform count": 0,
        "Pfam domains": 0,
        "subcellular location": N,          # comp, single coarse value
        "PTM sites": len(D["ptm"]),
        "3D structure record": len(D["struct"]),
    }
    after = {
        "sequence length": sum(1 for v in M.values() if v["len"]),
        "molecular mass": sum(1 for v in M.values() if v["mass"]),
        "isoform count": sum(1 for v in M.values() if v["isoforms"]),
        "Pfam domains": sum(1 for v in M.values() if v["pfam"]),
        "subcellular location": sum(1 for v in M.values() if v["loc"]),
        "PTM sites": sum(1 for v in M.values() if v["n_ptm"]),
        "3D structure record": len(D["struct"]),
    }
    P_(f"\n    {'field':<24} {'before':>8} {'after':>8}   {'verdict':<28}")
    for k in before:
        gain = after[k] - before[k]
        v = "NEW LAYER" if before[k] == 0 else ("gain" if gain > 500 else "already covered")
        P_(f"    {k:<24} {before[k]:>8,} {after[k]:>8,}   {v:<28}")
    P_("\n  P0: the fetch is justified by the fields at ZERO before -- sequence, mass, isoforms,")
    P_("  domains. `comp` already gives every gene a location, but a single coarse label with no")
    P_("  evidence code; UniProt's is evidence-attributed. Counted as a quality change, not a gain.")

    # ---- P2  splicing -----------------------------------------------------------------------
    P_("\n" + RULE); P_("P2  THE SPLICING LAYER, AND THE ASSUMPTION IT TESTS"); P_(RULE)
    iso = [v["isoforms"] for v in M.values() if v["isoforms"]]
    dist = collections.Counter(iso)
    multi = sum(1 for x in iso if x > 1)
    P_(f"  genes with an isoform record           {len(iso):>6,}")
    P_(f"  ONE protein  (1 isoform)               {dist.get(1,0):>6,}  {100*dist.get(1,0)/len(iso):>5.1f}%")
    P_(f"  MORE THAN ONE                          {multi:>6,}  {100*multi/len(iso):>5.1f}%")
    P_(f"  mean isoforms per gene                 {sum(iso)/len(iso):>6.2f}")
    P_(f"  max                                    {max(iso):>6,}  ({[s for s,v in M.items() if v['isoforms']==max(iso)][:3]})")
    P_(f"\n    {'isoforms':>9} {'genes':>8}")
    for k in sorted(dist)[:8]:
        P_(f"    {k:>9} {dist[k]:>8,}")
    P_(f"    {'>= 9':>9} {sum(v for k,v in dist.items() if k>=9):>8,}")
    P_(f"\n  P2: {100*multi/len(iso):.1f}% of the proteome is NOT one-gene-one-protein. Every per-gene")
    P_( "  record in this repo -- the 16,492 rows, the 21,955 signed edges, every coverage")
    P_( "  fraction -- aggregates over that silently. The aggregation error is unmeasured, and")
    P_( "  nothing in the record has ever bounded it.")

    # ---- P3  TF labelling -------------------------------------------------------------------
    P_("\n" + RULE); P_("P3  LABELLING THE TF NETWORK FROM THREE SOURCES THAT DISAGREE"); P_(RULE)
    up_tf = {s for s, v in M.items() if TF_KW & set(v["kw"])}
    ct_cat = {}
    with open(CTRI) as fh:
        for r in csv.DictReader(fh):
            ct_cat[r["source"].strip().upper()] = r["TF.category"]
    ct_tf = set(ct_cat)
    ency_tf = {s for s, r in gene.items() if r.get("tf") == 1}
    P_(f"  UniProt keyword TF-like                {len(up_tf):>6,}")
    P_(f"  CollecTRI regulators                   {len(ct_tf):>6,}   {dict(collections.Counter(ct_cat.values()))}")
    P_(f"  encyclopedia `tf` flag                 {len(ency_tf):>6,}")
    allg = set(idx)
    tab = collections.Counter()
    for s in allg:
        tab[(s in up_tf, s in ct_tf, s in ency_tf)] += 1
    P_(f"\n    {'UniProt':>8} {'CollecTRI':>10} {'ency.tf':>8} {'genes':>8}")
    for k in sorted(tab, key=lambda x: -tab[x]):
        if any(k):
            P_(f"    {str(k[0]):>8} {str(k[1]):>10} {str(k[2]):>8} {tab[k]:>8,}")
    solo = tab[(False, False, True)]
    P_(f"\n  P3: the encyclopedia's `tf` flag is corroborated by NEITHER other source on"
       f" {solo:,} genes.")
    P_(f"  All three agree on {tab[(True,True,True)]:,}. Corroboration count is carried per gene in the")
    P_( "  artifact so a consumer can choose a threshold instead of inheriting one.")

    # ---- P4  artifact -----------------------------------------------------------------------
    P_("\n" + RULE); P_("P4  THE LABELLED NETWORK"); P_(RULE)
    outrec = {}
    for s in allg:
        i = idx[s]
        v = M.get(s)
        corr = sum((s in up_tf, s in ct_tf, s in ency_tf))
        rec = {"tf_corroboration": corr}
        if s in ct_cat:
            rec["tf_class"] = ct_cat[s]
        if v:
            rec.update({"acc": v["acc"], "len": v["len"], "mass": v["mass"],
                        "isoforms": v["isoforms"], "pfam": v["pfam"][:8],
                        "loc": v["loc"][:80], "n_ptm_sites": v["n_ptm"],
                        "geneid": v["geneid"], "evidence": v["evidence"]})
        outrec[names[i]] = rec
    sgn = collections.Counter()
    for s, t, w in D["reg"]:
        sgn[("reg", w)] += 1
    for s, t, w in D["sig"]:
        sgn[("sig", w)] += 1
    P_(f"  reg edges  signed {sgn[('reg',1)]+sgn[('reg',-1)]:>7,}  unsigned {sgn[('reg',0)]:>7,}")
    P_(f"  sig edges  signed {sgn[('sig',1)]+sgn[('sig',-1)]:>7,}  unsigned {sgn[('sig',0)]:>7,}")
    regs = {r[0] for r in D["reg"]} | {r[0] for r in D["sig"]}
    lab = sum(1 for i in regs if outrec.get(names[i], {}).get("tf_corroboration", 0) >= 2)
    P_(f"\n  distinct regulators in the network     {len(regs):>6,}")
    P_(f"  ...labelled TF by >=2 independent sources {lab:>6,}  {100*lab/max(len(regs),1):>5.1f}%")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(outrec, open(ART, "w"))
    P_(f"\n  artifact: outputs/protein_tf_layer.json"
       f"  ({os.path.getsize(ART)/1e6:.1f} MB, {len(outrec):,} genes)")
    for nm in ("TP53", "NR3C1", "HK1"):
        P_(f"    {nm}: {json.dumps(outrec.get(nm.upper(), {}))[:190]}")

    # ---- P5 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("P5  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. Isoform COUNTS are not isoform-resolved edges. Knowing TP53 has 9 forms does not tell")
    P_("     us which one carries any of its 1,310 signed targets. The network is still gene-level.")
    P_("  2. A UniProt keyword is a curator's label, not a measurement, and 'DNA-binding' covers")
    P_("     far more than sequence-specific transcription factors.")
    P_("  3. Reviewed/Swiss-Prot only. Unreviewed TrEMBL entries are excluded by construction, so")
    P_("     genes absent here are absent from the REVIEWED proteome, not from biology.")
    P_("  4. Subcellular location is text with evidence codes; it is not parsed into the")
    P_("     encyclopedia's 12-compartment vocabulary, and no mapping is asserted.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
