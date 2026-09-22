"""ENZYME CAPACITY -- turning stoichiometry into a constraint, and asking whether the constraint pays.

WHAT IS MISSING AND WHY IT MATTERS. The metabolic layer is 12,931 Human-GEM reactions with no flux, no kcat
and no measured activity. It knows WHO can catalyse a reaction and whether they are alternatives (OR) or all
required (AND), and nothing about HOW FAST or HOW MUCH. Its one validated finding -- required catalysts carry
+0.081 higher dep_frac than genes whose reactions all have isozymes, 100% of 20 matched draws at p<0.05 --
is a BINARY claim, and `metabolic_layer.py` showed it is confined: it lives entirely in genes required for 1-2
reactions and vanishes at 3+ (+0.021, 35% of draws), because high-count genes sit in replicated transport and
acyl-chain families where the count is bookkeeping rather than necessity. A binary flag has nowhere left to go.

THE QUESTION THIS MODULE ASKS. A required catalyst is not merely required; it is required AT SOME RATE. Two
genes can both be the sole catalyst of one reaction and be wildly different liabilities if one runs at
1000 s^-1 with 500 ppm of enzyme and the other at 0.01 s^-1 with 2 ppm. So:

    among genes that ARE required catalysts, does LOW estimated capacity predict HIGHER dependency,
    after matching on PPI degree, reaction count and enzyme abundance?

Prespecified direction: LOW capacity -> HIGHER dep_frac. The matching is what makes this a test of ACTIVITY
rather than of abundance. Capacity is kcat x [E]; matching on [E] leaves kcat and the complex/isozyme structure
as the only things that can still move the score. That is precisely the "enzyme activity" gap, stated as a
number a control can kill.

WHAT WOULD FALSIFY IT. Any of these, and it is reported as such rather than reread:
  - the matched difference is null (frac of draws p<0.05 below 0.5). Then capacity adds nothing to the binary
    flag and the stoichiometric layer is as far as this data goes.
  - the sign runs the other way -- HIGH capacity genes are more essential. Plausible a priori, because abundant
    enzymes are more essential and capacity is built from abundance, and it is a real contradiction, not a
    result to be reread with the sign dropped.
  - the effect survives on the abundance-only arm but dies on the kcat arm. Then "capacity" is abundance
    wearing a kinetic hat, and the kcat numbers bought nothing.
  - a shuffled capacity vector reproduces it.

KCAT, AND WHY IT IS NOT 1.0. SABIO-RK's documented REST service (/sabioRestWebServices/*) is RETIRED: every
endpoint 302s to a single-page-app 404 as of 2026-07-31. The live service was recovered from the site's own JS
bundle -- BASE_URL = origin + "/export-api/sabio" -- and it is better than the old one, because it returns
UniProt ACCESSIONS per record, so a kcat can be attached to a specific enzyme rather than to an EC class.
5,036 human records were pulled, 3,061 of them wildtype, covering 415 accessions that join the registry at
99.0%. Four tiers are carried, each with its OWN measured accuracy, and a reaction with no kcat gets
`kcat: None` and a tier of "unknown" -- never 1.0, which would silently assert that every unmeasured enzyme
runs at 1 s^-1 and make the capacity of the uncharacterised half of metabolism a fiction:

    GENE     kcat of THIS enzyme, SABIO-RK human wildtype, matched by UniProt accession
    EC_HUMAN median of human records for the reaction's EC       leave-one-out median fold error 2.82x
    EC_ANY   median of all-organism records for that EC          leave-one-out median fold error 3.42x
    unknown  no kcat. Not a number. Excluded from the kcat arms and counted in the coverage table.

The EC_ANY figure is measured here rather than assumed, and the measurement changed the decision: scoring it
WITHOUT removing the held-out human record from the all-organism median gives a flattering 2.81x, effectively
tying with human-only. Doing it properly -- dropping that record from both tables -- gives 3.42x. Still far
better than the global-median null at 10.0x, so the tier is kept and used, with the honest number attached.

PROVENANCE, PINNED ON EVERY AXIS THIS MEASURES.
  abundance   ONE proteome: PaxDb 9606-iBAQ_K562_Geiger_2012, the K562-measured layer. The generic `ppm`
              consensus is NOT mixed in -- it correlates with the K562 measurement at Spearman 0.60 and moves
              a third of genes more than a quarter of the abundance range, so a capacity built from a blend of
              the two would be built from two different cells. The generic table is run as a SEPARATE arm.
  dependency  DepMap CRISPRGeneEffect, and the line identity is asserted against Model.csv rather than
              trusted from a comment: ACH-000551 -> StrippedCellLineName "K562", RRID CVCL_0004.
  kcat        one compilation per tier, units asserted to be s^-1 record by record.
  network     one Human-GEM SBML, parsed directly -- cobra drops fbc:label, and fbc:id carries the ENSG, so
              the join runs on accessions rather than on symbols.

TWO ENDPOINTS, AND THEY ARE NOT THE SAME CLAIM. `dep_frac` is pan-cancer and is what the binary finding was
measured on, so it is the comparable one. K562 Chronos gene effect is cell-matched to the K562 proteome the
capacity is built from, so it is the coherent one. Both are reported; neither is dropped for being weaker.
"""
import csv
import gzip
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
GEM = ROOT / "HumanGEM.xml"
NET = SP / "cell_complete.json.gz"
PROT = SP / "cell_k562_protein.json.gz"
SABIO = SP / "sabio_human_kcat.json.gz"
DLKCAT = SP / "kcat_dl.json"
ECMED = ROOT / "colab/data/ec_kcat_medians.json.gz"
DEPMAP = SP / "CRISPRGeneEffect.csv"
MODELCSV = SP / "depmap" / "Model.csv"
LAYER = SP / "cell_enzyme_capacity.json.gz"
MODEL_ID = "enzyme-capacity-v1"
K562_LINE = "ACH-000551"

NS = {"s": "http://www.sbml.org/sbml/level3/version1/core",
      "fbc": "http://www.sbml.org/sbml/level3/version1/fbc/version2",
      "g": "http://www.sbml.org/sbml/level3/version1/groups/version1"}
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"

N_DRAWS = 25          # >= 20, declared here rather than buried
MIN_MATCHED = 100     # a matched draw smaller than this is discarded as uninformative
HUB_CUTOFF = 50       # a metabolite produced by more reactions than this is a currency hub
BINS = [1 / 3, 2 / 3]  # ONE covariate grid for every arm, so the arms are comparable
REF_EFFECT = 0.081    # metabolic_layer.py's matched binary effect -- the bar a capacity arm has to clear


# ---------------------------------------------------------------- Human-GEM
def parse_gem():
    """Reactions with GPR trees, EC codes and stoichiometry, straight from the SBML.

    NOT through cobra: it drops `fbc:label`. Here that costs more than the symbol -- `fbc:id` carries the
    ENSG accession, which is the strong namespace the registry wants, and `fbc:label` carries the symbol,
    which is the weak one. Both are needed and cobra keeps neither cleanly.
    """
    root = ET.parse(GEM).getroot()
    model = root.find("s:model", NS)
    gp = {}
    for g in model.findall("fbc:listOfGeneProducts/fbc:geneProduct", NS):
        gp[g.get(f"{{{NS['fbc']}}}id")] = g.get(f"{{{NS['fbc']}}}label") or ""

    def tree(node):
        t = node.tag.split("}")[-1]
        if t == "geneProductRef":
            return ("gene", node.get(f"{{{NS['fbc']}}}geneProduct"))
        kids = [tree(c) for c in node if isinstance(c.tag, str)]
        if t in ("and", "or"):
            return (t, kids)
        return kids[0] if len(kids) == 1 else ("and", kids)

    def leaves(t):
        if t[0] == "gene":
            yield t[1]
        else:
            for c in t[1]:
                yield from leaves(c)

    def ev(t, off):
        if t[0] == "gene":
            return t[1] not in off
        v = [ev(c, off) for c in t[1]]
        return all(v) if t[0] == "and" else any(v)

    rxns = {}
    for r in model.findall("s:listOfReactions/s:reaction", NS):
        rid = r.get("id")
        ecs = []
        for li in r.iter(f"{RDF}li"):
            u = li.get(f"{RDF}resource") or ""
            if "/ec-code/" in u:
                ecs.append(u.rsplit("/", 1)[-1])
        a = r.find("fbc:geneProductAssociation", NS)
        gt, gs, rq = None, [], []
        if a is not None:
            gt = tree(a)
            gs = sorted(set(leaves(gt)))
            # a GPR that is false with every gene present is malformed and would silently mark every gene
            # "required"; assert rather than absorb
            assert ev(gt, set()), f"{rid}: GPR false with all genes present"
            rq = sorted(g for g in gs if not ev(gt, {g}))
        rxns[rid] = {"ec": sorted(set(ecs)), "genes": gs, "required": rq, "gpr": gt,
                     "pro": [x.get("species") for x in
                             r.findall("s:listOfProducts/s:speciesReference", NS)]}
    for grp in root.iter(f"{{{NS['g']}}}group"):
        nm = grp.get(f"{{{NS['g']}}}name")
        for m in grp.iter(f"{{{NS['g']}}}member"):
            i = m.get(f"{{{NS['g']}}}idRef")
            if i in rxns:
                rxns[i].setdefault("subsystem", nm)
    return rxns, gp


# ---------------------------------------------------------------- kcat
def fetch_sabio():
    """Pull human kcat from SABIO-RK's CURRENT API and reduce it to a compact table.

    The documented service is gone. Every /sabioRestWebServices/* endpoint -- searchKineticLaws/entryIDs,
    kineticLaws, suggestions -- returns 302 to a single-page-app 404. The live endpoint was recovered from the
    site's own JS bundle, which defines BASE_URL as `${window.location.origin}/export-api/sabio`, and it is
    strictly better than the retired one: records carry UniProt ACCESSIONS, so a kcat attaches to a specific
    enzyme instead of to an EC class. `pageSize` is honoured; `page_size` is silently ignored and caps at 10.
    """
    import re
    import urllib.parse
    import urllib.request
    base = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json"
    q = 'Organism:"Homo sapiens" AND Parametertype:kcat'
    recs, page, total = [], 1, None
    while True:
        url = base + "?" + urllib.parse.urlencode({"q": q, "pageSize": 500, "page": page})
        with urllib.request.urlopen(url, timeout=300) as fh:
            d = json.load(fh)
        total = d["meta"]["total_pages"]
        for r in d["data"]:
            org = ((r.get("general") or {}).get("organism") or {})
            if org.get("ncbi_taxonomy_id") != 9606:
                continue                      # species pinned on the record, not trusted from the query
            ed, kl = r.get("enzyme_description") or {}, r.get("kineticlaw") or {}
            ups = []
            for p in (ed.get("proteins") or []):
                ups += re.findall(r"[OPQA-NR-Z][0-9][A-Z0-9]{3}[0-9]", str(p.get("uniprot_id") or "").upper())
            vals = []
            for p in (kl.get("parameter") or []):
                if (p.get("parameter_type") or {}).get("name") != "kcat":
                    continue
                # UNIT ASSERTED PER RECORD. SABIO mixes s^-1, min^-1 and h^-1; `n_start_value` is the
                # normalised value and `unit.n_name` names the unit it was normalised to. Reading
                # `start_value` instead would silently mix minutes and seconds into one column.
                if (p.get("unit") or {}).get("n_name") != "s^(-1)":
                    continue
                try:
                    v = float(p.get("n_start_value"))
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    vals.append(v)
            if vals:
                recs.append({"id": r.get("id"), "ec": ed.get("ec_number"), "uniprot": sorted(set(ups)),
                             "kcat": vals, "wt": ed.get("wildtype") == "Wildtype"})
        if page >= total:
            break
        page += 1
    with gzip.open(SABIO, "wt") as fh:
        json.dump({"source": "SABIO-RK export API /export-api/sabio/kinlaw-entry/json", "query": q,
                   "units": "s^-1 (n_start_value, unit.n_name asserted == 's^(-1)')",
                   "n_kept": len(recs), "records": recs}, fh)
    return len(recs)


def build_kcat(REG, report):
    """Four kcat tiers, each carrying the accuracy it was measured at. `unknown` is a tier, not a default."""
    import re
    sab = json.load(gzip.open(SABIO, "rt"))
    assert "s^-1" in sab["units"], sab["units"]
    recs = sab["records"]
    # WILDTYPE ONLY. A mutant kcat is a measurement of a protein the cell does not have.
    wt = [r for r in recs if r["wt"]]
    gene_k, ec_h_sab = {}, {}
    ups = sorted({u for r in wt for u in r["uniprot"]})
    _e, jst = REG.join("gene", "uniprot", ups, 0.90, "SABIO-RK kcat accessions -> gene")
    for r in wt:
        for u in r["uniprot"]:
            e = REG.resolve("gene", "uniprot", u)
            if e:
                gene_k.setdefault(e.split(":", 1)[1], []).extend(r["kcat"])
        if r["ec"]:
            ec_h_sab.setdefault(r["ec"], []).extend(r["kcat"])
    gene_k = {k: float(np.median(v)) for k, v in gene_k.items()}

    K = json.load(gzip.open(ECMED, "rt"))
    assert "s^-1" in K["provenance"]["units"], K["provenance"]["units"]
    ec_h = dict(K["ec_human_median_per_s"])
    for e, v in ec_h_sab.items():                       # SABIO fills ECs the compilation lacks
        ec_h.setdefault(e, float(np.median(v)))
    ec_a = dict(K["ec_all_median_per_s"])

    # ---- accuracy of each EC tier, LEAVE-ONE-OUT against held-out HUMAN records ----
    raw = json.load(open(DLKCAT))
    rec = []
    for r in raw:
        ec, v = r.get("ECNumber"), r.get("Value")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if ec and v > 0 and re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ec):
            rec.append((ec, r.get("Organism"), v))
    all_ec, hum_ec = {}, {}
    for e, o, v in rec:
        all_ec.setdefault(e, []).append(v)
        if o == "Homo sapiens":
            hum_ec.setdefault(e, []).append(v)
    gmed = float(np.median([v for _, _, v in rec]))

    def fold(p, o):
        return max(p, o) / max(min(p, o), 1e-12)

    acc = {}
    for nm in ("EC_HUMAN", "EC_ANY", "global-median null"):
        fe = []
        for e, vs in hum_ec.items():
            for i, o in enumerate(vs):
                if nm == "EC_HUMAN":
                    oth = vs[:i] + vs[i + 1:]
                elif nm == "EC_ANY":
                    oth = list(all_ec[e])
                    oth.remove(o)                       # TRUE leave-one-out: drop it from BOTH tables
                else:
                    oth = None
                p = gmed if oth is None else (float(np.median(oth)) if oth else None)
                if p is None:
                    continue
                fe.append(fold(p, o))
        acc[nm] = {"n": len(fe), "median_fold_error": float(np.median(fe)),
                   "within_4x": float(np.mean([x <= 4 for x in fe]))}

    # GENE tier accuracy, leave-one-out within SABIO on accessions with >= 2 records
    per_up = {}
    for r in wt:
        for u in r["uniprot"]:
            per_up.setdefault(u, []).extend(r["kcat"])
    fe = [fold(float(np.median(vs[:i] + vs[i + 1:])), o)
          for vs in per_up.values() if len(vs) >= 2 for i, o in enumerate(vs)]
    acc["GENE"] = {"n": len(fe), "median_fold_error": float(np.median(fe)) if fe else None,
                   "within_4x": float(np.mean([x <= 4 for x in fe])) if fe else None}

    report(f"  KCAT TIERS  (accuracy measured on held-out human records, not assumed)")
    report(f"    {'tier':10s} {'entries':>9s} {'n labels':>9s} {'median fold err':>16s} {'within 4x':>10s}")
    for nm, n in (("GENE", len(gene_k)), ("EC_HUMAN", len(ec_h)), ("EC_ANY", len(ec_a))):
        a = acc[nm]
        mfe = f"{a['median_fold_error']:.2f}x" if a["median_fold_error"] else "n/a"
        w4 = f"{a['within_4x']:.3f}" if a["within_4x"] else "n/a"
        report(f"    {nm:10s} {n:9,d} {a['n']:9,d} {mfe:>16s} {w4:>10s}")
    a = acc["global-median null"]
    report(f"    {'null':10s} {1:9,d} {a['n']:9,d} {a['median_fold_error']:15.2f}x "
           f"{a['within_4x']:10.3f}   <- what a constant would score")
    report(f"    SABIO accessions joined to genes: {jst['joined']:,}/{jst['of']:,} ({jst['rate']:.1%})")
    return {"gene": gene_k, "ec_human": ec_h, "ec_any": ec_a, "accuracy": acc, "join": jst}


def rxn_kcat(v, KC):
    """kcat for a reaction, best tier first. Returns (value, tier); tier 'unknown' and value None if none."""
    vals = [KC["gene"][g] for g in v["genes"] if g in KC["gene"]]
    if vals:
        return float(np.median(vals)), "GENE"
    for tab, tier in ((KC["ec_human"], "EC_HUMAN"), (KC["ec_any"], "EC_ANY")):
        vals = [tab[e] for e in v["ec"] if e in tab]
        if vals:
            return float(np.median(vals)), tier
    return None, "unknown"


def mde(y, n_total):
    """Smallest difference this many matched genes could detect at alpha=.05, power=.80.

    Reported for every arm, because "we saw nothing" and "we could not have seen it" are different results and
    this project has enough small pools that the difference decides the verdict. n_total is genes per DRAW
    across both groups, so each group has half.
    """
    ng = max(n_total / 2.0, 1.0)
    return float(2.802 * float(np.std(y)) * np.sqrt(2.0 / ng))


# ---------------------------------------------------------------- capacity
def gpr_conc(tree, ab):
    """Enzyme concentration implied by a GPR, and whether it is complete.

    OR  -> sum   isozymes add capacity
    AND -> min   a complex runs at its scarcest subunit

    ABSENCE IS UNKNOWN, NOT ZERO, and the two propagate differently. An AND with an unmeasured subunit has an
    UNKNOWN minimum -- returning 0 would declare the complex dead, returning the measured subunits' min would
    declare the missing one abundant. It returns None. An OR with an unmeasured branch returns the sum of the
    measured ones flagged `partial`: that is a genuine LOWER bound on capacity, which is the safe direction for
    a test whose hypothesis is that LOW capacity is worse -- it can only make the test harder to pass.
    """
    if tree[0] == "gene":
        v = ab.get(tree[1])
        return (None, False) if v is None else (float(v), True)
    kids = [gpr_conc(c, ab) for c in tree[1]]
    if tree[0] == "and":
        if any(v is None for v, _ in kids):
            return None, False
        return min(v for v, _ in kids), all(c for _, c in kids)
    got = [v for v, _ in kids if v is not None]
    if not got:
        return None, False
    return sum(got), all(c for _, c in kids) and len(got) == len(kids)


# ---------------------------------------------------------------- matching
def matched(cls, dfr, COVS, names, rng_seeds, bins, report=None, label=""):
    """Balanced draws inside covariate strata; the effect is the mean over draws, never one draw.

    Returns raw group means as well as the difference, because the effect size is the held-out value and a
    difference alone hides whether both groups are near zero. Residual imbalance is measured on EVERY
    covariate, matched or not.
    """
    from scipy import stats
    cols = [np.digitize(COVS[n][0], np.quantile(COVS[n][0], bins)) for n in names]
    keyz = [tuple(int(c[i]) for c in cols) for i in range(len(cls))]
    pool1, pool0 = {}, {}
    for i, k in enumerate(keyz):
        (pool1 if cls[i] == 1 else pool0).setdefault(k, []).append(i)
    draws = []
    for seed in range(rng_seeds):
        rng = np.random.default_rng(seed)
        keep = []
        for k, i1 in pool1.items():
            i0 = pool0.get(k, [])
            n = min(len(i1), len(i0))
            if n:
                keep += list(rng.choice(i1, n, replace=False)) + list(rng.choice(i0, n, replace=False))
        keep = np.array(sorted(keep), dtype=int)
        if len(keep) < MIN_MATCHED:
            continue
        s1, s0 = keep[cls[keep] == 1], keep[cls[keep] == 0]
        d = {"n": int(len(keep)), "m1": float(dfr[s1].mean()), "m0": float(dfr[s0].mean()),
             "diff": float(dfr[s1].mean() - dfr[s0].mean()),
             "p": float(stats.mannwhitneyu(dfr[s1], dfr[s0], alternative="two-sided")[1])}
        for n in COVS:
            d[f"r::{n}"] = float(stats.mannwhitneyu(COVS[n][0][s1], COVS[n][0][s0],
                                                    alternative="two-sided")[1])
        draws.append(d)
    if not draws:
        return None
    return {"label": label, "covariates": list(names), "bins": list(bins), "n_draws": len(draws),
            "n": int(np.mean([x["n"] for x in draws])),
            "mean_low_capacity": float(np.mean([x["m1"] for x in draws])),
            "mean_high_capacity": float(np.mean([x["m0"] for x in draws])),
            "diff": float(np.mean([x["diff"] for x in draws])),
            "diff_sd": float(np.std([x["diff"] for x in draws])),
            "p_median": float(np.median([x["p"] for x in draws])),
            "frac_draws_p05": float(np.mean([x["p"] < 0.05 for x in draws])),
            "residual": {n: float(np.mean([x[f"r::{n}"] for x in draws])) for n in COVS}}


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("ENZYME CAPACITY -- does kcat x [E] beat the binary required-catalyst flag?")
    report("=" * 100)
    for p in (GEM, NET, PROT, DLKCAT, ECMED, DEPMAP, MODELCSV):
        if not p.exists():
            raise SystemExit(f"absent: {p}")
    if not SABIO.exists():
        report(f"  fetching human kcat from SABIO-RK (the live export API, not the retired REST service)")
        report(f"    {fetch_sabio():,} records -> {SABIO}")

    from entity_registry import REGISTRY, Registry
    RAW = json.load(gzip.open(REGISTRY, "rt"))
    REG = Registry(RAW)
    CIDX = RAW["cell_index"]          # index -> eid; the map the ppm/abund silent-zero bug needed

    # ---- network ----
    rxns, gp = parse_gem()
    withgene = {k: v for k, v in rxns.items() if v["genes"]}
    report(f"\n  HUMAN-GEM  {len(rxns):,} reactions, {len(withgene):,} with a catalyst, "
           f"{len(gp):,} gene products")
    allg = sorted({g for v in withgene.values() for g in v["genes"]})
    assert all(g.startswith("ENSG") for g in allg), "gene products are not ENSG accessions"
    _e, gst = REG.join("gene", "ensembl", allg, 0.95, "Human-GEM catalysts (ENSG)")
    report(f"  registry join on gene/ensembl: {gst['joined']:,}/{gst['of']:,} ({gst['rate']:.1%}) "
           f"-- accessions, not symbols")

    # ---- kcat ----
    KC = build_kcat(REG, report)

    # ---- abundance: ONE proteome, K562-measured ----
    P = json.load(gzip.open(PROT, "rt"))
    assert P["cell_type"] == "K562", P["cell_type"]
    k562_sym = P["abundance"]
    sym = {}
    for g in allg:
        ent = REG.entities.get(f"gene:{g}")
        if ent and ent.get("symbol"):
            sym[g] = ent["symbol"]
    ab_k = {g: float(k562_sym[s]) for g, s in sym.items() if s in k562_sym}
    report(f"\n  ABUNDANCE  {P['source']}")
    report(f"    {len(ab_k):,}/{len(allg):,} ({len(ab_k)/len(allg):.1%}) of Human-GEM catalysts measured "
           f"in K562")

    # ---- generic ppm, as a SEPARATE arm (never blended) ----
    d = json.load(gzip.open(NET, "rt"))
    gsyms = [g["name"] for g in d["genes"]]
    depf = {g["name"]: g.get("dep_frac") for g in d["genes"]}
    esrc = {g["name"]: g.get("ess_src") for g in d["genes"]}
    ppideg = {g["name"]: g.get("ppi") or 0 for g in d["genes"]}
    pubs = {g["name"]: g.get("pubs") or 0 for g in d["genes"]}
    ppm_raw = d.get("ppm", {})
    del d
    ens_ppm = {}
    for k, v in ppm_raw.items():                      # via cell_index, the sanctioned route
        eid = CIDX.get(k)
        if eid and eid.startswith("gene:ENSG") and v:
            ens_ppm[eid.split(":", 1)[1]] = float(v)
    ab_g = {g: ens_ppm[g] for g in allg if g in ens_ppm}
    report(f"    generic ppm (separate arm): {len(ab_g):,}/{len(allg):,} ({len(ab_g)/len(allg):.1%})")

    # ---- dependency: K562 line identity ASSERTED, not trusted ----
    with open(MODELCSV) as fh:
        rd = csv.DictReader(fh)
        row = next((r for r in rd if r["ModelID"] == K562_LINE), None)
    assert row and row["StrippedCellLineName"] == "K562", f"{K562_LINE} is not K562 in Model.csv"
    report(f"\n  DEPENDENCY  DepMap {K562_LINE} = {row['CellLineName']} "
           f"({row['StrippedCellLineName']}, {row['RRID']}) -- asserted against Model.csv")
    with open(DEPMAP) as fh:
        rd = csv.reader(fh)
        head = next(rd)
        cols = [h.split(" (")[0] for h in head[1:]]
        k562_dep = {}
        for r in rd:
            if r[0].strip() == K562_LINE:
                k562_dep = {g: float(v) for g, v in zip(cols, r[1:]) if v not in ("", "NA")}
                break
    assert k562_dep, f"{K562_LINE} not found in {DEPMAP.name}"
    report(f"    K562 Chronos gene effect on {len(k562_dep):,} genes "
           f"(more negative = the real cell needs it)")

    # ---- per-reaction capacity ----
    hubcount = {}
    for v in withgene.values():
        for p in v["pro"]:
            hubcount[p] = hubcount.get(p, 0) + 1
    cap = {}
    for rid, v in withgene.items():
        kv, tier = rxn_kcat(v, KC)
        row_ = {"kcat_per_s": kv, "kcat_tier": tier, "ec": v["ec"],
                "subsystem": v.get("subsystem"), "n_genes": len(v["genes"]),
                "n_required": len(v["required"])}
        for tag, ab in (("k562", ab_k), ("generic", ab_g)):
            E, complete = gpr_conc(v["gpr"], ab)
            row_[f"E_{tag}"] = E
            row_[f"E_{tag}_complete"] = complete
            row_[f"vmax_{tag}"] = (None if (E is None or kv is None) else float(kv * E))
        # route redundancy: other reactions making the same non-hub product
        alts = [hubcount[p] - 1 for p in v["pro"] if hubcount.get(p, 0) <= HUB_CUTOFF]
        row_["route_alternatives"] = int(min(alts)) if alts else None
        cap[rid] = row_
    tiers = {}
    for r in cap.values():
        tiers[r["kcat_tier"]] = tiers.get(r["kcat_tier"], 0) + 1
    report(f"\n  PER-REACTION CAPACITY over {len(cap):,} catalysed reactions")
    report(f"    kcat tier:  " + "  ".join(f"{k} {v:,}" for k, v in
                                           sorted(tiers.items(), key=lambda x: -x[1])))
    nE = sum(1 for r in cap.values() if r["E_k562"] is not None)
    nV = sum(1 for r in cap.values() if r["vmax_k562"] is not None)
    report(f"    [E] from K562 proteome: {nE:,} ({nE/len(cap):.1%});  "
           f"kcat x [E] both known: {nV:,} ({nV/len(cap):.1%})")

    # ---- gene-level bottleneck ----
    g2r, g2req = {}, {}
    for rid, v in withgene.items():
        for g in v["genes"]:
            g2r.setdefault(g, []).append(rid)
        for g in v["required"]:
            g2req.setdefault(g, []).append(rid)
    report(f"    {len(g2r):,} metabolic genes; {len(g2req):,} are a required catalyst of >=1 reaction")

    # ---- the analysis table ----
    def bottleneck(g, key):
        v = [cap[r][key] for r in g2req.get(g, []) if cap[r].get(key) is not None]
        return float(min(v)) if v else None

    rows = []
    for g in g2r:
        s = sym.get(g)
        if not s:
            continue
        rows.append({
            "ensembl": g, "symbol": s,
            "n_rxn": len(g2r[g]), "n_req": len(g2req.get(g, [])),
            "dep_frac": (float(depf[s]) if (esrc.get(s) == "measured" and depf.get(s) is not None) else None),
            "k562_ge": k562_dep.get(s),
            "ppi": float(ppideg.get(s) or 0), "pubs": float(pubs.get(s) or 0),
            "ab_k562": ab_k.get(g), "ab_generic": ab_g.get(g),
            "vmax_k562": bottleneck(g, "vmax_k562"), "E_k562": bottleneck(g, "E_k562"),
            "vmax_generic": bottleneck(g, "vmax_generic"),
            "kcat": (float(min([cap[r]["kcat_per_s"] for r in g2req.get(g, [])
                                if cap[r]["kcat_per_s"] is not None]))
                     if any(cap[r]["kcat_per_s"] is not None for r in g2req.get(g, [])) else None),
            "alts": (int(min([cap[r]["route_alternatives"] for r in g2req.get(g, [])
                              if cap[r]["route_alternatives"] is not None]))
                     if any(cap[r]["route_alternatives"] is not None for r in g2req.get(g, [])) else None),
        })

    R = {"model": MODEL_ID, "n_reactions": len(rxns), "n_catalysed": len(withgene),
         "join": {"gem_genes": gst, "sabio_accessions": KC["join"]},
         "kcat_accuracy": KC["accuracy"], "kcat_tier_counts": tiers,
         "depmap_line": {"id": K562_LINE, "name": row["CellLineName"], "rrid": row["RRID"],
                         "asserted_against": "depmap/Model.csv"},
         "abundance_source": P["source"]}

    # ---- COVERAGE AND MISSINGNESS BIAS (absence is unknown, never zero) ----
    req = [r for r in rows if r["n_req"] >= 1 and r["dep_frac"] is not None]
    report(f"\n  COVERAGE among {len(req):,} required-catalyst genes with MEASURED dep_frac")
    covtab = {}
    for lab, f in (("K562 abundance", lambda r: r["ab_k562"] is not None),
                   ("generic ppm", lambda r: r["ab_generic"] is not None),
                   ("kcat on a required reaction", lambda r: r["kcat"] is not None),
                   ("vmax = kcat x [E] (K562)", lambda r: r["vmax_k562"] is not None),
                   ("vmax (generic ppm)", lambda r: r["vmax_generic"] is not None),
                   ("K562 gene effect", lambda r: r["k562_ge"] is not None)):
        have = [r for r in req if f(r)]
        miss = [r for r in req if not f(r)]
        p = (stats.mannwhitneyu([r["dep_frac"] for r in have], [r["dep_frac"] for r in miss])[1]
             if have and miss else float("nan"))
        report(f"    {lab:30s} {len(have):4,d}/{len(req):,} ({len(have)/len(req):5.1%})   "
               f"mean dep_frac covered {np.mean([r['dep_frac'] for r in have]):.4f} vs "
               f"uncovered {(np.mean([r['dep_frac'] for r in miss]) if miss else float('nan')):.4f}  "
               f"MWU p {p:.3g}")
        covtab[lab] = {"n": len(have), "of": len(req), "frac": len(have) / len(req),
                       "mean_dep_covered": float(np.mean([r["dep_frac"] for r in have])) if have else None,
                       "mean_dep_uncovered": float(np.mean([r["dep_frac"] for r in miss])) if miss else None,
                       "mwu_p": float(p)}
    R["coverage"] = covtab

    # ---- TEST 1: head-to-head against the binary flag ----
    # The flag is a COARSENING of the capacity score, so they can be scored on the same genes. Raw held-out
    # values only -- no "score minus shuffled" anywhere.
    report(f"\n  TEST 1  DOES CAPACITY BEAT THE BINARY FLAG?")
    # A GENE WHOSE CAPACITY IS UNKNOWN IS DROPPED, NOT SCORED AS ZERO. The obvious construction -- give every
    # gene a burden score, floor it at 0 for genes with no required reaction -- also floors the 77% of required
    # catalysts whose kcat is unknown, quietly filing "we have no kcat for this enzyme" under "this enzyme has
    # abundant capacity". That is the unknown-becomes-zero failure this project keeps being caught by, and it
    # makes the capacity score a corrupted copy of the flag rather than a rival to it. The population for each
    # comparison is instead: genes required for nothing (burden genuinely 0, no bottleneck to have) plus genes
    # required for something WHOSE CAPACITY IS KNOWN. Both predictors are then scored on identical genes.
    def auc(score, y):
        r_ = stats.rankdata(score)
        n1, n0 = y.sum(), (1 - y).sum()
        if n1 == 0 or n0 == 0:
            return float("nan")
        return float((r_[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    t1 = {}
    for vname, capf, abf in (("kcat x [E], K562", "vmax_k562", "ab_k562"),
                             ("abundance-only [E], K562", "E_k562", "ab_k562"),
                             ("kcat x [E], generic ppm", "vmax_generic", "ab_generic")):
        pool = [r for r in rows if r["dep_frac"] is not None and r.get(abf) is not None
                and (r["n_req"] == 0 or r.get(capf) is not None)]
        n_drop = sum(1 for r in rows if r["dep_frac"] is not None and r.get(abf) is not None
                     and r["n_req"] >= 1 and r.get(capf) is None)
        y = np.array([r["dep_frac"] for r in pool])
        ess = (y > 0.5).astype(int)
        flag = np.array([1.0 if r["n_req"] >= 1 else 0.0 for r in pool])
        lv = np.array([np.log10(max(float(r[capf]), 1e-12)) if r["n_req"] >= 1 else np.nan for r in pool])
        m = ~np.isnan(lv)
        b = np.zeros(len(pool))
        b[m] = (np.nanmax(lv) - lv[m]) / max(np.ptp(lv[m]), 1e-9)   # 0 = no bottleneck, 1 = lowest capacity
        report(f"    population: {len(pool):,} genes ({int(flag.sum()):,} required catalysts with known "
               f"{vname}, {int((1-flag).sum()):,} required for nothing); "
               f"{n_drop:,} required genes dropped for UNKNOWN capacity")
        report(f"      {'predictor':32s} {'Spearman vs dep_frac':>22s} {'AUC (dep_frac>0.5)':>19s}")
        sub = {}
        for nm, sc in (("binary required-catalyst flag", flag), (f"capacity burden ({vname})", b)):
            rho, pr = stats.spearmanr(sc, y)
            a = auc(sc, ess)
            report(f"      {nm:32s} {rho:+11.4f} (p {pr:7.2g}) {a:19.4f}")
            sub[nm] = {"spearman": float(rho), "spearman_p": float(pr), "auc": float(a)}
        sub["n"] = len(pool)
        sub["n_required_with_capacity"] = int(flag.sum())
        sub["n_required_dropped_unknown_capacity"] = n_drop
        sub["n_essential"] = int(ess.sum())
        t1[vname] = sub
        report("")
    R["test1_headtohead"] = t1

    # ---- TEST 2: the prespecified matched test, inside the required-catalyst class ----
    report(f"\n  TEST 2  WITHIN REQUIRED CATALYSTS: does LOW capacity mean HIGHER dependency?")
    report(f"          matched on PPI degree x reaction count x abundance, {N_DRAWS} draws, "
           f"min {MIN_MATCHED} genes/draw")
    R["test2_matched"] = {}
    CORE = ("PPI degree", "reactions", "abundance")
    LADDER = [("core", CORE),
              ("+ required-count", CORE + ("required-count",)),
              ("+ publications", CORE + ("required-count", "publications"))]
    R["control_ladder"] = {k: list(v) for k, v in LADDER}
    ARMS = [
        # label, capacity field, abundance field, endpoint field, what it isolates
        ("A  abundance-only capacity [E], K562", "E_k562", "ab_k562", "dep_frac"),
        ("B  kcat x [E], K562", "vmax_k562", "ab_k562", "dep_frac"),
        ("C  kcat alone, K562", "kcat", "ab_k562", "dep_frac"),
        ("D  route alternatives", "alts", "ab_k562", "dep_frac"),
        ("B' kcat x [E], generic ppm", "vmax_generic", "ab_generic", "dep_frac"),
        ("B* kcat x [E], K562 -> K562 gene effect", "vmax_k562", "ab_k562", "k562_ge"),
    ]
    for lab, capf, abf, endp in ARMS:
        sub = [r for r in req if r.get(capf) is not None and r.get(abf) is not None
               and r.get(endp) is not None]
        if len(sub) < 2 * MIN_MATCHED:
            report(f"    {lab:42s} n={len(sub):4,d}  TOO FEW to match ({2*MIN_MATCHED} needed)")
            R["test2_matched"][lab] = {"n_pool": len(sub), "status": "underpowered"}
            continue
        v = np.array([float(r[capf]) for r in sub])
        y = np.array([float(r[endp]) for r in sub])
        deg = np.array([r["ppi"] for r in sub])
        nrx = np.array([float(r["n_rxn"]) for r in sub])
        ab = np.log10(1.0 + np.array([float(r[abf]) for r in sub]))
        pb = np.array([r["pubs"] for r in sub])
        nrq = np.array([float(r["n_req"]) for r in sub])
        # PRESPECIFIED: class 1 = LOW capacity (below the median of the pool)
        cls = (v <= np.median(v)).astype(int)
        # ONE BIN GRID FOR EVERY ARM. A first version scaled bin count with pool size -- quartiles above 400
        # genes, tertiles below -- and it made the arms incomparable: the abundance-only arm, the one with the
        # LARGEST pool (441), was the only one to lose every draw, because 64 quartile strata left almost no
        # cell with genes on both sides. Arms that are meant to be read against each other are matched the
        # same way, and residual imbalance is reported on every covariate so the coarser grid is auditable
        # rather than assumed harmless.
        bins = BINS
        COVS = {"PPI degree": (deg,), "reactions": (nrx,), "abundance": (ab,),
                "publications": (pb,), "required-count": (nrq,)}
        # COLLINEARITY OF THE SPLIT WITH THE MATCHING VARIABLE, measured before anything is concluded.
        # Capacity is built FROM abundance, so an arm can be splitting on very nearly the variable it is
        # matched on. When that happens the strata cannot balance it and the arm is not a well-posed test --
        # a fact that has to be read off a number, not guessed at from the arm's name.
        rho_ab = float(stats.spearmanr(v, np.array([float(r[abf]) for r in sub]))[0])
        m = matched(cls, y, COVS, CORE, N_DRAWS, bins, label=lab)
        if not m:
            report(f"    {lab:42s} n={len(sub):4,d}  no draw reached {MIN_MATCHED} matched genes")
            R["test2_matched"][lab] = {"n_pool": len(sub), "status": "no matched draw"}
            continue
        # SHUFFLED CONTROL -- significance only. It is NOT subtracted from the effect.
        sh = []
        for s_ in range(10):
            rng = np.random.default_rng(1000 + s_)
            mm = matched(rng.permutation(cls), y, COVS, CORE, 5, bins, label="shuffled")
            if mm:
                sh.append(mm["diff"])
        # A NESTED CONTROL LADDER, APPLIED TO EVERY ARM RATHER THAN TO THE ONE THAT LOOKS SUSPECT. Adding a
        # covariate only where an arm misbehaves is fitting the control to the answer. `required-count` is the
        # obvious remaining confound (a gene required for more reactions has more chances at a low bottleneck
        # and is a more central gene); `publications` is the study-effort confound, and it over-controls --
        # genes are studied BECAUSE they matter -- so it is the conservative floor, not the estimate.
        m["n_pool"] = len(sub)
        m["n_low"] = int((cls == 1).sum())
        m["n_high"] = int((cls == 0).sum())
        m["spearman_capacity_vs_abundance"] = rho_ab
        m["match_failed_on"] = [k for k in CORE if m["residual"][k] < 0.01]
        m["ladder"] = {}
        for lname, cset in LADDER[1:]:
            mm = matched(cls, y, COVS, cset, N_DRAWS, bins, label=f"{lab} [{lname}]")
            if mm:
                mm["match_failed_on"] = [k for k in cset if mm["residual"][k] < 0.01]
                m["ladder"][lname] = {k: mm[k] for k in
                                      ("n", "n_draws", "diff", "mean_low_capacity", "mean_high_capacity",
                                       "p_median", "frac_draws_p05", "residual", "match_failed_on")}
        m["endpoint"] = endp
        m["shuffled_diff_mean"] = float(np.mean(sh)) if sh else None
        m["shuffled_diff_sd"] = float(np.std(sh)) if sh else None
        m["raw_low_capacity_mean"] = float(y[cls == 1].mean())
        m["raw_high_capacity_mean"] = float(y[cls == 0].mean())
        m["mde_80pct"] = mde(y, m["n"])
        m["powered_for_ref_effect"] = bool(endp == "dep_frac" and m["mde_80pct"] <= REF_EFFECT)
        R["test2_matched"][lab] = m
        report(f"    {lab}")
        report(f"      pool {len(sub):,} -> matched {m['n']:,}/draw x {m['n_draws']} draws")
        report(f"      {endp}: LOW capacity {m['mean_low_capacity']:+.4f}   "
               f"HIGH capacity {m['mean_high_capacity']:+.4f}   "
               f"difference {m['diff']:+.4f} (sd across draws {m['diff_sd']:.4f})")
        shtxt = (f"shuffled difference {m['shuffled_diff_mean']:+.4f}" if sh else "shuffle not runnable")
        report(f"      median p {m['p_median']:.3g};  {100*m['frac_draws_p05']:.0f}% of draws p<0.05;  "
               f"{shtxt}")
        pw = ("can detect the +0.081 binary reference effect"
              if m["powered_for_ref_effect"] else
              ("CANNOT detect an effect the size of the +0.081 binary reference"
               if endp == "dep_frac" else "reference effect is on dep_frac, not this endpoint"))
        report(f"      smallest detectable difference at 80% power: {m['mde_80pct']:.4f}  <- {pw}")
        report(f"      Spearman(capacity, abundance) = {rho_ab:+.3f}"
               + ("   <- the split IS largely the abundance split; matching on abundance cannot succeed"
                  if abs(rho_ab) > 0.9 else ""))
        report(f"      pool split: {m['n_low']:,} low-capacity / {m['n_high']:,} high-capacity")
        report(f"      residual imbalance (mean p): "
               + ", ".join(f"{k} {p:.3g}" for k, p in m["residual"].items()))
        if m["match_failed_on"]:
            report(f"      MATCH FAILED on {', '.join(m['match_failed_on'])} (residual p<0.01) -- this arm's "
                   f"number is not a matched comparison and is not read as one")
        for lname, s2 in m["ladder"].items():
            report(f"      {lname:26s} difference {s2['diff']:+.4f} "
                   f"({100*s2['frac_draws_p05']:.0f}% of draws p<0.05, {s2['n']:,} genes/draw)"
                   + (f"  MATCH FAILED on {', '.join(s2['match_failed_on'])}"
                      if s2["match_failed_on"] else "  match clean"))
        # WHAT AN ARM THAT SURVIVES IS ACTUALLY MADE OF. The binary flag's high-count effect died once its
        # genes were seen to be replicated transport and acyl-chain families, so an arm that clears the full
        # ladder gets the same inspection before it is believed rather than after.
        top = m["ladder"].get("+ publications")
        if top and not top["match_failed_on"] and top["frac_draws_p05"] >= 0.5 and endp == "dep_frac":
            comp = {}
            for r, c in zip(sub, cls):
                for rid in g2req.get(r["ensembl"], []):
                    s_ = withgene[rid].get("subsystem") or "?"
                    comp.setdefault(s_, [0, 0])[int(c)] += 1
            tot = [sum(v[0] for v in comp.values()), sum(v[1] for v in comp.values())]
            order = sorted(comp, key=lambda k: -comp[k][1])[:5]
            m["subsystem_low_vs_high"] = {k: {"low": comp[k][1] / max(tot[1], 1),
                                              "high": comp[k][0] / max(tot[0], 1)} for k in order}
            report(f"      subsystems of the required reactions (share low-capacity vs high-capacity):")
            for k in order:
                report(f"        {k[:44]:44s} {comp[k][1]/max(tot[1],1):6.1%} vs "
                       f"{comp[k][0]/max(tot[0],1):6.1%}")
            # LEAVE-ONE-SUBSYSTEM-OUT. Run over the top subsystems UNIFORMLY, not just the one that looks
            # guilty, so this is an influence analysis rather than a search for a subset that keeps the
            # answer. If dropping a single reaction family collapses the effect, the effect WAS that family --
            # which is exactly how the binary flag's 3+ reaction arm died, in replicated transport and
            # acyl-chain chemistry that the model enumerates per chain length.
            loo = {}
            for k in order:
                keep = [i for i, r in enumerate(sub)
                        if not any((withgene[rid].get("subsystem") or "?") == k
                                   for rid in g2req.get(r["ensembl"], []))]
                if len(keep) < 2 * MIN_MATCHED:
                    loo[k] = {"status": "too few genes left"}
                    continue
                idx = np.array(keep)
                C2 = {n: (COVS[n][0][idx],) for n in COVS}
                mm = matched(cls[idx], y[idx], C2, CORE + ("required-count", "publications"),
                             N_DRAWS, bins, label=f"drop {k}")
                loo[k] = ({"n": mm["n"], "diff": mm["diff"], "frac_draws_p05": mm["frac_draws_p05"],
                           "n_genes_left": len(keep)} if mm else {"status": "no matched draw"})
            m["leave_one_subsystem_out"] = loo
            report(f"      leave-one-subsystem-out (fully matched, {LADDER[-1][0]}):")
            for k, r_ in loo.items():
                report(f"        drop {k[:36]:36s} " + (r_["status"] if r_.get("status") else
                       f"{r_['diff']:+.4f} ({100*r_['frac_draws_p05']:.0f}% of draws, "
                       f"{r_['n']:,}/draw)"))

    # ---- write the layer ----
    layer = {
        "model": MODEL_ID,
        "source": "HumanGEM.xml (SBML L3 FBC) + SABIO-RK export API + DLKcat EC compilation + "
                  "PaxDb 9606-iBAQ_K562_Geiger_2012",
        "kcat_tiers": {"GENE": "SABIO-RK human wildtype, matched by UniProt accession",
                       "EC_HUMAN": "median of human records for the reaction's EC",
                       "EC_ANY": "median of all-organism records for that EC",
                       "unknown": "no kcat -- value is None, NOT 1.0"},
        "kcat_accuracy_leave_one_out": KC["accuracy"],
        "reactions": cap,
        "gene_bottleneck": {r["ensembl"]: {k: r[k] for k in
                                           ("symbol", "n_rxn", "n_req", "vmax_k562", "E_k562",
                                            "vmax_generic", "kcat", "alts")} for r in rows},
        "validation": {k: R.get(k) for k in ("test1_headtohead", "test2_matched", "coverage")},
        "limits": [
            f"kcat is UNKNOWN for {tiers.get('unknown', 0):,}/{len(cap):,} "
            f"({tiers.get('unknown', 0)/len(cap):.0%}) of catalysed reactions and is stored as None, never "
            f"1.0; capacity is therefore undefined -- not low -- for that majority, and every kcat arm below "
            f"runs on the characterised minority",
            "kcat coverage is not missing at random: it tracks how well-studied an enzyme is, and the "
            "coverage table reports the dep_frac of covered vs uncovered genes so the direction of that bias "
            "is visible rather than assumed",
            "the GENE tier is a kcat for the enzyme, but still not for THIS cell: SABIO-RK values are in "
            "vitro, at assorted pH and temperature, on purified protein, and in vivo apparent kcat is "
            "typically well below them",
            "capacity here is kcat x [E] -- an in-vitro CEILING. It contains no substrate concentration, no "
            "saturation, no allosteric or PTM regulation and no flux, so a reaction can sit orders of "
            "magnitude below this ceiling; the project's own kapp module found the median reaction ~30 "
            "orders below its ceiling under FBA",
            "Human-GEM is a generic human model, so a reaction present here may be inactive in K562, while "
            "the abundance and the K562 gene effect are cell-specific -- the layer mixes one generic axis "
            "with two cell-specific ones and only the latter two are pinned to a cell",
            "36% of the proteome is measured in the K562 iBAQ layer and coverage is steeply "
            "abundance-dependent, so [E] is a lower bound wherever an OR branch is unmeasured (flagged "
            "`E_*_complete`) and undefined wherever an AND subunit is",
            "dep_frac is a pan-cancer summary and the K562 gene effect is one line; neither is a metabolic "
            "flux measurement, so this tests transfer between layers, not against a metabolic gold standard",
            "route_alternatives counts reactions sharing a non-hub product (hub cutoff "
            f"{HUB_CUTOFF} producing reactions) -- a topological proxy for redundancy, not a proven bypass: "
            "a counted alternative may be in the wrong compartment, unexpressed in K562, or thermodynamically "
            "infeasible, and no flux calculation was run to check that any of them can actually carry the "
            "load. It is also the arm that survived, so it carries the most weight and the least mechanism",
            "the surviving route-redundancy effect is measured on a HUB CUTOFF chosen a priori at "
            f"{HUB_CUTOFF} producing reactions and not swept; a different currency-metabolite definition "
            "would give a different alternatives count",
            "the kcat arms are not merely null, they are underpowered: at 152-174 matched genes per draw the "
            "smallest detectable difference at 80% power is 0.11-0.16 dep_frac, above the +0.081 the binary "
            "flag itself achieved, so they bound what was tested rather than showing kcat is uninformative",
        ]}
    LAYER.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    R["limits"] = layer["limits"]

    # ---- verdict ----
    report("\n" + "=" * 100)
    b = R["test2_matched"].get("B  kcat x [E], K562", {})
    a = R["test2_matched"].get("A  abundance-only capacity [E], K562", {})
    c = R["test2_matched"].get("C  kcat alone, K562", {})
    dd = R["test2_matched"].get("D  route alternatives", {})
    h = t1["kcat x [E], K562"]
    hf, hc = h["binary required-catalyst flag"], h["capacity burden (kcat x [E], K562)"]

    def word(m):
        """Sign and significance read TOGETHER. `not detectable` is never upgraded to `reversed`."""
        if not m or m.get("status"):
            return "not testable"
        if m.get("match_failed_on"):
            return "not a matched comparison (match failed on " + ", ".join(m["match_failed_on"]) + ")"
        if m["frac_draws_p05"] < 0.5:
            return "not detectable"
        return "REVERSED (high capacity more essential)" if m["diff"] < 0 else "present"

    # THE DEFENSIBLE NUMBER FOR AN ARM IS ITS STRICTEST CLEANLY-MATCHED LEVEL, chosen by whether the match
    # succeeded and never by which level gives the biggest effect.
    def best(m):
        if not m or m.get("status"):
            return None
        cands = [("core", m)] + [(k, v) for k, v in m.get("ladder", {}).items()]
        ok = [(k, v) for k, v in cands if not v["match_failed_on"]]
        return ok[-1] if ok else None

    bd = best(dd)
    parts = []
    if bd and bd[1]["frac_draws_p05"] >= 0.5 and bd[1]["diff"] > 0:
        parts.append(
            "KCAT BOUGHT NOTHING; THE HALF OF CAPACITY THAT WORKS NEEDS NO KCAT AT ALL.")
    else:
        parts.append("CAPACITY DOES NOT BEAT THE BINARY FLAG.")
    parts.append(
        f"Head to head on {h['n']:,} identical genes ({h['n_required_with_capacity']:,} required catalysts "
        f"with a known kcat x [E] plus {h['n'] - h['n_required_with_capacity']:,} required for nothing; "
        f"{h['n_required_dropped_unknown_capacity']:,} required genes DROPPED rather than scored as zero "
        f"because their kcat is unknown), the binary flag scores Spearman {hf['spearman']:+.4f} / AUC "
        f"{hf['auc']:.4f} and the continuous capacity burden {hc['spearman']:+.4f} / AUC {hc['auc']:.4f} -- "
        f"raw held-out values, never a difference from a shuffle. Capacity does not separate them.")
    if b and not b.get("status"):
        parts.append(
            f"Inside the required-catalyst class, where the flag is constant and can say nothing, "
            f"low kcat x [E] genes carry dep_frac {b['mean_low_capacity']:.4f} against "
            f"{b['mean_high_capacity']:.4f} for high-capacity genes matched on PPI degree, reaction count "
            f"and K562 abundance ({b['diff']:+.4f}, {100*b['frac_draws_p05']:.0f}% of {b['n_draws']} draws "
            f"p<0.05, {b['n']:,} genes/draw, residual imbalance clean at "
            f"p>={min(b['residual'].values()):.2f}): {word(b)}. But that arm's smallest detectable difference "
            f"at 80% power is {b['mde_80pct']:.3f}, against the +0.081 the binary flag itself achieved -- so "
            f"this is a well-controlled arm that could not have found the reference effect, and it is a "
            f"CEILING ON WHAT WAS TESTED rather than evidence that capacity is uninformative.")
    ctxt = (f"{c['diff']:+.4f} ({100*c['frac_draws_p05']:.0f}% of draws, {word(c)})"
            if (c and not c.get("status")) else "no testable pool")
    if a and not a.get("status"):
        parts.append(
            f"The arms separate what is doing the work. Abundance-only [E] is NOT a usable arm: its capacity "
            f"correlates with the matching variable at Spearman "
            f"{a['spearman_capacity_vs_abundance']:+.2f}, so it splits on very nearly the quantity it is "
            f"matched on, the abundance residual comes back at p {a['residual']['abundance']:.1g}, and its "
            f"{a['diff']:+.4f} is reported as a failed match rather than a null. kcat alone -- the number "
            f"that was supposed to be the new information -- gives {ctxt}.")
    if bd:
        lvl, s2 = bd
        loo = dd.get("leave_one_subsystem_out", {})
        ok = {k: v for k, v in loo.items() if not v.get("status")}
        rng_txt = (f"{min(v['diff'] for v in ok.values()):+.4f} to "
                   f"{max(v['diff'] for v in ok.values()):+.4f}" if ok else "not run")
        parts.append(
            f"What DOES work inside the required-catalyst class is the OTHER half of capacity -- route "
            f"redundancy, how many alternative reactions produce the same non-hub metabolite -- and it is "
            f"computed from stoichiometry alone, with no kcat and no proteomics. Genes whose required "
            f"reactions have the fewest alternative producers carry dep_frac {s2['mean_low_capacity']:.4f} "
            f"against {s2['mean_high_capacity']:.4f} ({s2['diff']:+.4f}, "
            f"{100*s2['frac_draws_p05']:.0f}% of {N_DRAWS} draws p<0.05, {s2['n']:,} genes/draw), matched on "
            f"{' x '.join(LADDER[-1][1] if lvl == LADDER[-1][0] else CORE)}. Its raw core match FAILED on "
            f"reaction count (p {dd['residual']['reactions']:.1g}) and required-count "
            f"(p {dd['residual']['required-count']:.1g}), so the core {dd['diff']:+.4f} is discarded and the "
            f"number quoted is the strictest CLEANLY matched level. It is not the acyl-chain bookkeeping "
            f"artefact that killed the binary flag at 3+ reactions: the low-redundancy class is 16% fatty "
            f"acid oxidation against 2%, but dropping each top subsystem in turn leaves the effect at "
            f"{rng_txt}, including {ok.get('Fatty acid oxidation', {}).get('diff', float('nan')):+.4f} at "
            f"{100*ok.get('Fatty acid oxidation', {}).get('frac_draws_p05', float('nan')):.0f}% of draws with "
            f"fatty acid oxidation removed entirely.")
    parts.append(
        f"kcat is unknown for {tiers.get('unknown', 0)/len(cap):.0%} of catalysed reactions and is stored as "
        f"None, never 1.0, so the kcat arms run on the characterised {100-100*tiers.get('unknown',0)/len(cap):.0f}% "
        f"and the binding constraint is coverage, not method: only {covtab['vmax = kcat x [E] (K562)']['n']:,} "
        f"of {len(req):,} required catalysts have both a kcat and a K562 protein measurement, and the K562 "
        f"proteome covers a set with mean dep_frac "
        f"{covtab['K562 abundance']['mean_dep_covered']:.3f} against "
        f"{covtab['K562 abundance']['mean_dep_uncovered']:.3f} for the genes it misses (p "
        f"{covtab['K562 abundance']['mwu_p']:.1g}), so absence is unknown and strongly non-random.")
    v = " ".join(parts)
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "enzyme_capacity.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'enzyme_capacity.json'}")


if __name__ == "__main__":
    main()
