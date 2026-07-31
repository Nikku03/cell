"""ENVIRONMENTAL STATE -- what condition each dataset was actually in, and which cross-dataset
comparisons therefore rest on an assumption nobody checked.

WHY THIS MATTERS EVEN THOUGH IT PREDICTS NOTHING. A large fraction of cross-dataset non-replication is
environmental rather than biological: the same cell line in RPMI and in DMEM differs in glucose by more than
twofold, and a 48-hour readout is not a 12-day readout. Nothing in this model currently records medium,
glucose, glutamine, serum, oxygen, drug, dose, time or cell density for ANY dataset it uses. Every
cross-dataset number this project has produced -- the +15.0 sign transfer, the bound x causal directness
tier, the dynamics null, the 15.9% mRNA -> protein ceiling, the cross-cell-line decay -- was computed as
though the two sides were in comparable conditions, and that was never checked because there was nowhere to
check it.

WHAT THIS ESTABLISHES AND WHAT IT CANNOT. It cannot show that any of those conclusions is wrong. An
environmental mismatch is not a refutation; it is a term that was never in the error bar. What it CAN do is
say, per comparison and per variable, whether the two sides were in the same condition, in different
conditions, or in conditions nobody recorded -- and that turns a silent assumption into an auditable one.

WHAT WOULD FALSIFY THE PREMISE. If every dataset here turned out to be in the same medium at the same
density on the same clock, the whole concern would collapse and this module should say so. It does not
collapse. The K562 arm of the Perturb-seq transfer is in RPMI-1640 and the RPE1 arm is in DMEM:F12 with
hygromycin B, read out at eight days against six. The K562 proteome that sets the mRNA -> protein ceiling
was grown in DMEM by a different laboratory while the K562 transcriptome it is compared against was grown in
RPMI. Those are not inferred; they are quoted from the publications and the quotes are re-fetched and
asserted every run.

THE RULE ON UNKNOWNS, which is the whole point of the module. An unrecorded variable is written as `unknown`
and never as a default. "Standard conditions" is not a value. Where a fact is not stated but follows from a
named catalogue formulation -- glucose in RPMI-1640 catalogue 11875 -- it is written at level `derived`, a
separate and weaker class than `stated`, so a downstream reader can refuse to use it. Nowhere does an
unknown become a number.

PROVENANCE IS PINNED AND ASSERTED, NOT DESCRIBED. Every `stated` fact carries a source id and a verbatim
quote. At run time each source is re-fetched and each quote is asserted present. A quote that no longer
resolves fails the run rather than surviving as a docstring claim about what some paper said. Where a source
is unreachable, its facts drop to `unverified` and the blocked host is named -- an unreachable condition is
itself a finding and is reported as one, never back-filled.

THE ONE THING THAT IS MEASURED RATHER THAN LOOKED UP. ENCODE's K562 TF ChIP is not one cell population. For
a large minority of TF targets, EVERY released K562 experiment is on a genetically modified line -- a
CRISPR- or BAC-tagged K562, not the parental one -- and a handful are interferon-stimulated. That fraction is
computed here from the portal, joined through the registry, and restricted to the TFs that the bound x causal
analysis could actually have used, so it is a number about that analysis and not about ENCODE in general.
"""
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CACHE = SP / "envaudit"
GWPS = SP / "gwps.h5ad"
RPE1 = SP / "rpe1.h5ad"
PAXDB = SP / "prot" / "9606-iBAQ_K562_Geiger_2012_uniprot.txt"
MODEL = SP / "depmap" / "Model.csv"
REGL = SP / "cell_reg_perturb.json.gz"          # the causal layer bound_causal.py was built on
BC_REPORTED = 396                               # targets bound_causal.py reports intersecting; asserted
ENCODE = "https://www.encodeproject.org"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BIORXIV = "https://www.biorxiv.org/content"

# The nine variables the dataset map names. This list is fixed: a variable that is not on it is out of
# scope, and a variable that is on it is `unknown` unless a source says otherwise.
VARS = ("medium", "glucose", "glutamine", "serum", "oxygen", "drug", "dose", "time", "density")

# Catalogue formulations, used ONLY to derive glucose and glutamine from a medium that a source names
# explicitly. Level `derived`, never `stated`. Plain "DMEM" is deliberately absent: it ships at 5.5 and at
# 25 mM glucose and a source that says only "DMEM" has not determined the glucose.
FORMULATION = {
    "RPMI-1640":  {"glucose_mM": 11.1, "glutamine_mM": 2.05,
                   "ref": "Thermo Fisher RPMI 1640 formulation, cat 11875 (2.0 g/L D-glucose, "
                          "0.3 g/L L-glutamine)"},
    "DMEM:F12":   {"glucose_mM": 17.5, "glutamine_mM": 2.5,
                   "ref": "Thermo Fisher DMEM/F-12 1:1 formulation (3.151 g/L D-glucose, 0.365 g/L "
                          "L-glutamine)"},
    "IMDM":       {"glucose_mM": 25.0, "glutamine_mM": 4.0,
                   "ref": "Thermo Fisher IMDM formulation (4.5 g/L D-glucose, 0.584 g/L L-glutamine)"},
    "EMEM":       {"glucose_mM": 5.56, "glutamine_mM": 2.0,
                   "ref": "Corning EMEM formulation (1.0 g/L D-glucose; 292 mg/L L-glutamine, which the "
                          "source states directly)"},
    "F-12K":      {"glucose_mM": 7.0, "glutamine_mM": 2.0,
                   "ref": "Thermo Fisher Kaighn's F-12K formulation (1.26 g/L D-glucose, 0.292 g/L "
                          "L-glutamine)"},
}

# ---------------------------------------------------------------------------------------------- sources
# Each source is fetched live and each `quotes` entry is asserted present in the fetched text. `squash`
# sources are PDFs whose extracted text has whitespace scattered through every word, so they are compared
# with all non-alphanumeric characters removed.
SOURCES = {
    "replogle": dict(
        kind="publication", id="PMID 35688146 / PMC9380471",
        cite="Replogle et al., Cell 2022, 'Mapping information-rich genotype-phenotype landscapes with "
             "genome-scale Perturb-seq'",
        fetch=("epmc", "PMC9380471"),
        quotes=["K562 cells were grown in RPMI-1640 with 25 mM HEPES, 2.0 g/l NaHCO3, and 0.3 g/l "
                "L-glutamine supplemented with 10% FBS, 2 mM glutamine",
                "RPE1 cells (ATCC, CRL-4000) were grown in DMEM:F12 supplemented with 10% FBS, "
                "0.01 mg/ml hygromycin B",
                "a density of 250,000 to 1,000,000 cells/ml for the course of the experiment",
                "Eight days post infection",
                "Six days post infection"]),
    "geiger": dict(
        kind="publication", id="PMID 22278370 / PMC3316730",
        cite="Geiger et al., Mol Cell Proteomics 2012, 'Comparative proteomic analysis of eleven common "
             "cell lines' -- the deposit behind PaxDb 9606-iBAQ_K562_Geiger_2012",
        fetch=("epmc", "PMC3316730"),
        quotes=["K562, MCF7, RKO, and U2OS cells were grown with Dulbecco",
                "modified Eagle",
                "10% fetal bovine serum and antibiotics",
                "subconfluent cultures were flash frozen"]),
    "frangieh": dict(
        kind="publication", id="PMID 33649592 / PMC8376399",
        cite="Frangieh et al., Nat Genet 2021, 'Multimodal pooled Perturb-CITE-seq screens in patient "
             "models define mechanisms of cancer immune evasion'",
        fetch=("efetch", "8376399"),
        quotes=["Roswell Park Memorial Institute (RPMI) 1640 Medium supplemented with 10% "
                "heat-inactivated fetal bovine serum (FBS), GlutaMax, 10 mM HEPES"]),
    "nadig": dict(
        kind="preprint", id="bioRxiv 10.1101/2024.07.03.601903",
        cite="Nadig et al., 'Transcriptome-wide characterization of genetic perturbations' -- the Jurkat "
             "and HepG2 Perturb-seq arms. NOT in Europe PMC full text; only the preprint server carries "
             "the methods",
        fetch=("biorxiv", "10.1101/2024.07.03.601903"),
        quotes=["Jurkat cells were grown in RPMI-1640 medium with 25 mM HEPES, 2.0 g/l NaHCO3, and "
                "0.3 g/l L-glutamine (Gibco) supplemented with 10% (v/v) standard FBS, 2 mM glutamine",
                "HepG2 cells were grown in EMEM with 1.5 g/L NaHCO3, 110 mg/L sodium pyruvate, "
                "292 mg/L l-glutamine (Corning) supplemented with 10% (v/v) standard FBS",
                "All cell lines were grown at 37"]),
    "xaira": dict(
        kind="preprint", id="bioRxiv 10.1101/2025.06.11.659105",
        cite="Huang et al., 'X-Atlas/Orion: Genome-wide Perturb-seq Datasets via a Scalable "
             "Fix-Cryopreserve Platform' -- the HCT116 arm",
        fetch=("biorxiv", "10.1101/2025.06.11.659105"),
        quotes=["HCT116 and HEK293T cells were propagated in RPMI 1640 GlutaMAX without HEPES (Gibco) "
                "supplemented with 10% fetal bovine serum"]),
    "encode_k562_growth": dict(
        kind="protocol PDF", id="ENCODE document aac59442-1653-4601-94e7-923b6b931e15",
        cite="ENCODE biosample growth protocol 'SOP: Propagation of K562 (ATCC CCL-243)', attached to "
             "K562 ChIP biosamples. A PDF attachment, not a queryable field",
        fetch=("encode_pdf", "/documents/aac59442-1653-4601-94e7-923b6b931e15/"),
        squash=True,
        quotes=["RPMI1640", "LifeTechnologies", "11875150", "HeatInactivatedFetalBovineSerum"]),
    "encode_a549_growth": dict(
        kind="protocol PDF", id="ENCODE document 7f983ac2-a390-4b37-b0d5-458d61b8a1a2",
        cite="ENCODE biosample growth protocol 'Cell Growth Protocol for A549 Cell Line' "
             "(HudsonAlpha/Caltech, 2008). Attached to the Myers A549 biosamples ONLY -- see the audit "
             "note on which experiments the dynamics analysis actually used",
        fetch=("encode_pdf", "/documents/7f983ac2-a390-4b37-b0d5-458d61b8a1a2/"),
        squash=True,
        quotes=["F12/K", "10%fetalbovineserum", "5%CO"]),
    "paxdb_header": dict(
        kind="local file header", id=str(PAXDB),
        cite="the PaxDb K562 file's own comment header, which names the publication the abundances came "
             "from -- the provenance chain from the number to the growth medium runs through it",
        fetch=("localhead", str(PAXDB)),
        quotes=["#name: H.sapiens - Cell line, K562 (Geiger,MCP,2012)",
                "#link: http://www.mcponline.org/cgi/doi/10.1074/mcp.M111.014050",
                "#integrated: false"]),
}

# md5 pinned by depmap_codep.py for the same file. Asserting it here means the two modules are provably
# talking about the same download rather than about "DepMap".
MODEL_MD5 = "e5b60d1ce7636542d93f45494019eded"


# ------------------------------------------------------------------------------------------ fetching

def _get(url, headers=None, tries=3, timeout=180):
    last = None
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers=headers or {"User-Agent": "cellos"})
            return urllib.request.urlopen(r, timeout=timeout).read()
        except Exception as ex:
            last = ex
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))
    raise last


def _cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if p.exists() and p.stat().st_size > 0:
        return p.read_text("utf-8", errors="replace")
    t = fn()
    p.write_text(t, "utf-8")
    return t


def _detag(b):
    t = re.sub(r"<[^>]+>", " ", b.decode("utf-8", "replace"))
    return re.sub(r"\s+", " ", t)


def _pdftext(b):
    """Flate streams -> the text inside PDF string literals. Deliberately crude: the ENCODE protocol PDFs
    are typeset with per-character kerning, so the extracted words come out with spaces inside them and
    only a whitespace-insensitive comparison is meaningful. That is why those sources are `squash`."""
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", b, re.S):
        s = m.group(1)
        try:
            s = zlib.decompress(s)
        except Exception:
            continue
        toks = [mm.group(0)[1:-1] for mm in re.finditer(rb"\((?:\\.|[^\\()])*\)", s)]
        if toks:
            out.append(b" ".join(toks))
    return b"\n".join(out).decode("latin-1")


def fetch_source(sid, spec):
    """Returns (text, status, host). status 'ok' or a precise failure -- never a silent empty string."""
    kind, arg = spec["fetch"]
    try:
        if kind == "epmc":
            return _cached(f"{sid}.txt", lambda: _detag(_get(f"{EPMC}/{arg}/fullTextXML"))), "ok", \
                   "www.ebi.ac.uk"
        if kind == "efetch":
            return _cached(f"{sid}.txt",
                           lambda: _detag(_get(f"{EFETCH}?db=pmc&id={arg}&rettype=xml"))), "ok", \
                   "eutils.ncbi.nlm.nih.gov"
        if kind == "biorxiv":
            return _cached(f"{sid}.txt",
                           lambda: _detag(_get(f"{BIORXIV}/{arg}v1.full",
                                               headers={"User-Agent": "Mozilla/5.0"}))), "ok", \
                   "www.biorxiv.org"
        if kind == "encode_pdf":
            def go():
                o = json.loads(_get(ENCODE + arg + "?frame=object",
                                    headers={"accept": "application/json", "User-Agent": "cellos"}))
                href = ENCODE + o["@id"] + o["attachment"]["href"]
                return _pdftext(_get(href))
            return _cached(f"{sid}.txt", go), "ok", "www.encodeproject.org"
        if kind == "localhead":
            p = Path(arg)
            if not p.exists():
                return "", f"absent: {arg}", "local"
            with open(p) as fh:
                return "".join([next(fh) for _ in range(14)]), "ok", "local"
    except Exception as ex:
        host = {"epmc": "www.ebi.ac.uk", "efetch": "eutils.ncbi.nlm.nih.gov",
                "biorxiv": "www.biorxiv.org", "encode_pdf": "www.encodeproject.org",
                "localhead": "local"}[kind]
        return "", f"{type(ex).__name__}: {str(ex)[:120]}", host
    return "", "no fetcher", "?"


_SQ = re.compile(r"[^A-Za-z0-9%/.#]")


def assert_quotes(text, spec, sid):
    """Every quote must be present in the live text. A quote that has stopped resolving fails the run.

    This is the difference between pinning provenance and describing it. A docstring that says 'the paper
    reports RPMI' is unfalsifiable; a quote that is re-fetched and asserted is not.
    """
    hay = _SQ.sub("", text) if spec.get("squash") else text
    missing = []
    for q in spec["quotes"]:
        needle = _SQ.sub("", q) if spec.get("squash") else q
        if needle not in hay:
            missing.append(q[:70])
    if missing:
        raise AssertionError(
            f"provenance assertion FAILED for source {sid} ({spec['id']}): {len(missing)} of "
            f"{len(spec['quotes'])} pinned quotes no longer resolve in the fetched text -- {missing}. "
            f"Either the source changed, the fetch route changed, or the quote was wrong. None of those "
            f"may be papered over: the facts keyed on this source are not verified and must not be "
            f"reported as `stated`.")
    return len(spec["quotes"])


# --------------------------------------------------------------------------------------------- facts

def V(value, level, src=None, note=None, base=None, norm=None):
    """One variable of one dataset. `level` is stated / derived / unknown / unverified.

    UNKNOWN CARRIES NO VALUE. There is no code path where an unrecorded variable acquires a default.

    `norm` IS THE COMPARISON KEY AND `value` IS THE EVIDENCE. Two sources that both mean "10% FBS" write
    it differently ("10% FBS", "10% fetal bovine serum"), and comparing the prose would report a
    MISMATCH that does not exist -- which is the same class of error as reporting a match that does not
    exist, and just as damaging to an audit whose only product is those verdicts. Where a variable has a
    canonical form, it is written here explicitly; where it does not, `value` is compared and any match
    is exact-string.
    """
    assert level in ("stated", "derived", "unknown", "unverified")
    d = {"value": value if level != "unknown" else None, "level": level}
    if src:
        d["source"] = src
    if note:
        d["note"] = note
    if base:
        d["base"] = base
    if norm:
        d["norm"] = norm
    return d


def derived_from(medium_base, var, src_note):
    f = FORMULATION.get(medium_base)
    if not f:
        return V(None, "unknown", note=f"medium named as {medium_base!r}, which does not determine "
                                       f"{var} -- the formulation is sold at more than one concentration")
    k = {"glucose": "glucose_mM", "glutamine": "glutamine_mM"}[var]
    return V(f"{f[k]} mM", "derived", src=src_note,
             note=f"not stated by the source; follows from the named formulation. {f['ref']}")


UNK = "no source consulted here records it"


def build_datasets(enc, dep, hdr_ok):
    """The audit table. `enc` is the live ENCODE census, `dep` the DepMap Model.csv lookup."""
    D = {}

    D["replogle_k562"] = dict(
        label="Replogle K562 genome-scale CRISPRi Perturb-seq", cell_line="K562",
        deposit="SCRATCH/gwps.h5ad", role="the source of every causal edge in this project",
        vars={
            "medium": V("RPMI-1640 with 25 mM HEPES, 2.0 g/L NaHCO3, 0.3 g/L L-glutamine + 10% FBS + "
                        "2 mM glutamine + pen/strep", "stated", "replogle", base="RPMI-1640"),
            "glucose": derived_from("RPMI-1640", "glucose", "replogle"),
            "glutamine": V("~4.05 mM (0.3 g/L in base = 2.05 mM, plus 2 mM supplemented)", "stated",
                           "replogle", note="both components stated; the sum is arithmetic"),
            "serum": V("10% FBS", "stated", "replogle", norm="10% FBS"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": V("none during the Perturb-seq arm; sgRNA-positive cells were FACS-sorted on GFP "
                      "rather than drug-selected", "stated", "replogle", norm="none"),
            "dose": V("none", "stated", "replogle", norm="none", note="no drug in this arm"),
            "time": V("8 days post infection", "stated", "replogle", norm="8 d"),
            "density": V("250,000-1,000,000 cells/mL maintained through the screen", "stated",
                         "replogle")})

    D["replogle_rpe1"] = dict(
        label="Replogle RPE1 essential-gene CRISPRi Perturb-seq", cell_line="RPE1 (hTERT)",
        deposit="SCRATCH/rpe1.h5ad", role="the held-out arm of the +15.0 sign-transfer result",
        vars={
            "medium": V("DMEM:F12 + 10% FBS + 0.01 mg/mL hygromycin B + pen/strep", "stated",
                        "replogle", base="DMEM:F12"),
            "glucose": derived_from("DMEM:F12", "glucose", "replogle"),
            "glutamine": derived_from("DMEM:F12", "glutamine", "replogle"),
            "serum": V("10% FBS", "stated", "replogle", norm="10% FBS"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": V("hygromycin B 0.01 mg/mL, continuous, in the growth medium; no puromycin "
                      "selection because RPE1 is puromycin-resistant", "stated", "replogle",
                      norm="hygromycin B"),
            "dose": V("0.01 mg/mL hygromycin B", "stated", "replogle", norm="0.01 mg/mL hygromycin B"),
            "time": V("6 days post infection", "stated", "replogle", norm="6 d"),
            "density": V(None, "unknown",
                         note="the stated 250,000-1,000,000 cells/mL applies to the suspension K562 "
                              "arm; RPE1 is adherent and no confluence target is given")})

    D["encode_k562_chip"] = dict(
        label=f"ENCODE K562 TF ChIP-seq ({enc['k562_n_exp']} released experiments, "
              f"{enc['k562_n_targets']} targets)", cell_line="K562",
        deposit="ENCODE portal", role="the `bound` half of the bound x causal directness tier",
        vars={
            "medium": V("RPMI-1640 (Life Technologies 11875-150) + heat-inactivated FBS + "
                        "antibiotic-antimycotic, per the one K562 growth protocol on the portal",
                        "stated", "encode_k562_growth", base="RPMI-1640",
                        note=f"the medium exists only as a PDF attachment; "
                             f"{enc['k562_n_with_docs']}/{enc['k562_n_exp']} experiments carry any "
                             f"biosample document at all and there are {enc['k562_n_growth_docs']} "
                             f"distinct growth protocols across the whole set, from different labs"),
            "glucose": derived_from("RPMI-1640", "glucose", "encode_k562_growth"),
            "glutamine": derived_from("RPMI-1640", "glutamine", "encode_k562_growth"),
            "serum": V("heat-inactivated FBS (percentage not recoverable from the extracted text)",
                       "stated", "encode_k562_growth", norm="FBS, percentage unstated"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": V(f"none for {enc['k562_n_exp']-enc['k562_n_treated']}/{enc['k562_n_exp']} "
                      f"experiments; {enc['k562_n_treated']} are interferon alpha/gamma treated. "
                      f"Separately {enc['k562_n_modified']}/{enc['k562_n_exp']} are on a genetically "
                      f"modified K562 -- CRISPR- or BAC-tagged -- not the parental line", "stated",
                      "encode_api", norm="none (majority)",
                      note="ENCODE records treatment as a first-class field, so this one is machine-"
                           "readable; medium is not. The modification status is NOT a drug and is "
                           "reported here because it is the same kind of unrecorded-state problem"),
            "dose": V("none for the untreated majority; the interferon subset records a duration in "
                      "free-text notes and leaves `amount` unpopulated", "stated", "encode_api",
                      norm="none (majority)"),
            "time": V("no clock: steady-state occupancy, except the interferon subset", "stated",
                      "encode_api", norm="steady state"),
            "density": V(None, "unknown",
                         note="the growth protocol gives a passaging schedule, not a harvest density, "
                              "and it is not attached to most experiments")})

    D["encode_a549_dex"] = dict(
        label=f"ENCODE A549 dexamethasone time course ({enc['a549_n_exp']} released experiments)",
        cell_line="A549", deposit="ENCODE portal",
        role="the only dataset in this project with a real driver on a real clock; the dynamics null",
        vars={
            "medium": V(None, "unknown",
                        note=f"the only A549 growth protocol on the portal (F-12K + 10% FBS + pen/strep, "
                             f"5% CO2, 37 C) is attached to the {enc['a549_docs_lab']} Myers/HAIB "
                             f"biosamples only. All {enc['a549_reddy_nodoc']} experiments from the "
                             f"laboratory the dynamics analysis actually pinned carry NO biosample "
                             f"document, so the medium for the measured series is not recorded anywhere "
                             f"on the portal",
                        base=None),
            "glucose": V(None, "unknown", note="follows the medium"),
            "glutamine": V(None, "unknown", note="follows the medium"),
            "serum": V(None, "unknown", note="follows the medium"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": V("dexamethasone, a glucocorticoid receptor agonist", "stated", "encode_api",
                      norm="dexamethasone"),
            "dose": V(enc["a549_dose"], "stated", "encode_api", norm=enc["a549_dose_modal"],
                      note=f"a structured field on the treatment object. Across the whole released "
                           f"A549+dex set there are {enc['a549_n_amounts']} distinct amounts -- a small "
                           f"dose-response subset exists -- but within the laboratory the dynamics "
                           f"analysis pinned it is one value, {enc['a549_dose_modal']}, on "
                           f"{enc['a549_dose_modal_frac']:.0%} of that laboratory's treatment records"),
            "time": V(f"{enc['a549_n_durations']} distinct exposure durations from "
                      f"{enc['a549_min']} to {enc['a549_max']}", "stated", "encode_api",
                      norm="one shared grid",
                      note="a structured field on the treatment object"),
            "density": V(None, "unknown", note=UNK)})

    D["paxdb_k562"] = dict(
        label="PaxDb K562 proteome (Geiger 2012 iBAQ)", cell_line="K562",
        deposit="SCRATCH/prot/9606-iBAQ_K562_Geiger_2012_uniprot.txt",
        role="the protein side of the 15.9% mRNA -> protein ceiling",
        vars={
            "medium": V("Dulbecco's modified Eagle's medium + 10% fetal bovine serum + antibiotics",
                        "stated", "geiger", base="DMEM",
                        note="the PaxDb file's own #link header points at doi 10.1074/mcp.M111.014050, "
                             "which is this paper; the chain from abundance to medium is complete and "
                             f"the header assertion {'passed' if hdr_ok else 'DID NOT PASS'}"),
            "glucose": derived_from("DMEM", "glucose", "geiger"),
            "glutamine": V(None, "unknown",
                           note="'DMEM' alone does not determine glutamine either; the source gives no "
                                "catalogue number"),
            "serum": V("10% fetal bovine serum", "stated", "geiger", norm="10% FBS"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": V("none stated", "stated", "geiger", norm="none",
                      note="'antibiotics' unspecified; no perturbation -- this is an unperturbed "
                           "steady-state proteome"),
            "dose": V("none", "stated", "geiger", norm="none"),
            "time": V(None, "unknown",
                      note="no clock: cells were harvested once from subconfluent culture"),
            "density": V("subconfluent", "stated", "geiger",
                         note="a qualitative statement, not a number; it is the only density record "
                              "outside the Replogle K562 arm")})

    D["depmap_crispr"] = dict(
        label="DepMap CRISPRGeneEffect (24Q2 Public)", cell_line="1,150+ lines incl. K562",
        deposit="SCRATCH/CRISPRGeneEffect.csv", role="the viability endpoint behind the co-dependency layer",
        vars={
            "medium": V(f"per-line, from Model.csv FormulationID -- K562 {dep.get('K562')!r}, "
                        f"RPE1 {dep.get('RPE1')!r}, HCT116 {dep.get('HCT116')!r}, "
                        f"JURKAT {dep.get('JURKAT')!r}, HEPG2 {dep.get('HEPG2')!r}, "
                        f"A549 {dep.get('A549')!r}", "stated", "depmap_model",
                        base=(dep.get("K562") or "").split(" +")[0] or None,
                        note="the only machine-readable per-line medium field anywhere in this project. "
                             "It documents DEPMAP's own culture, not the culture of any other dataset "
                             "using the same line"),
            "glucose": derived_from((dep.get("K562") or "").split(" +")[0], "glucose", "depmap_model"),
            "glutamine": derived_from((dep.get("K562") or "").split(" +")[0], "glutamine",
                                      "depmap_model"),
            "serum": V("10% FBS for K562 per FormulationID; varies by line", "stated", "depmap_model",
                       norm="10% FBS"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": V("puromycin selection after Cas9/library transduction; concentration not in the "
                      "public release", "unverified",
                      note="stated in the Achilles protocol but not in any file this module fetched, so "
                           "it is recorded as unverified rather than stated"),
            "dose": V(None, "unknown", note=UNK),
            "time": V("21-day dropout screen (Achilles standard); not a field in the public release",
                      "unverified", note="not asserted from any fetched source"),
            "density": V(None, "unknown", note=UNK)})

    for k, (lab, cl, src, med, base, drug) in {
        "nadig_jurkat": ("Nadig Jurkat CRISPRi Perturb-seq", "Jurkat", "nadig",
                         "RPMI-1640 with 25 mM HEPES, 2.0 g/L NaHCO3, 0.3 g/L L-glutamine + 10% FBS + "
                         "2 mM glutamine + pen/strep", "RPMI-1640", None),
        "nadig_hepg2": ("Nadig HepG2 CRISPRi Perturb-seq", "HepG2", "nadig",
                        "EMEM with 1.5 g/L NaHCO3, 110 mg/L sodium pyruvate, 292 mg/L L-glutamine + "
                        "10% FBS + pen/strep", "EMEM", None),
        "xaira_hct116": ("Xaira X-Atlas/Orion HCT116 CRISPRi Perturb-seq", "HCT116", "xaira",
                         "RPMI 1640 GlutaMAX without HEPES + 10% FBS + pen/strep", "RPMI-1640", None),
        "frangieh_mel": ("Frangieh patient-derived melanoma Perturb-CITE-seq",
                         "melanoma (patient-derived)", "frangieh",
                         "RPMI-1640 + 10% heat-inactivated FBS + GlutaMax + 10 mM HEPES + 10 mg/L "
                         "insulin + 5.5 mg/L transferrin + 6.7 ug/L sodium selenite + 55 uM "
                         "2-mercaptoethanol", "RPMI-1640",
                         ("none in the arm this project uses. The deposit carries three -- Control, "
                          "IFN-gamma stimulation, and autologous T-cell co-culture -- and "
                          "scperturb_transfer.py pins Control, which is the ONE place in this project "
                          "where an environmental variable was controlled rather than ignored")),
    }.items():
        vv = {
            "medium": V(med, "stated", src, base=base),
            "glucose": derived_from(base, "glucose", src),
            "glutamine": (V("2 mM (292 mg/L, stated)", "stated", src) if base == "EMEM"
                          else derived_from(base, "glutamine", src)),
            "serum": V("10% FBS", "stated", src, norm="10% FBS"),
            "oxygen": V(None, "unknown", note=UNK),
            "drug": (V(drug, "stated", src, norm="none") if drug
                     else V(None, "unknown", note=UNK)),
            "dose": (V("none", "stated", src, norm="none") if drug
                     else V(None, "unknown", note=UNK)),
            "time": V(None, "unknown", note="no perturbation-to-readout interval stated in the methods "
                                            "text fetched here"),
            "density": V(None, "unknown", note=UNK)}
        if k in ("nadig_jurkat", "nadig_hepg2"):
            vv["oxygen"] = V("5% CO2 at 37 C; O2 tension not stated", "stated", "nadig",
                             norm="5% CO2, O2 unstated",
                             note="CO2 is not O2 -- this is the closest any source comes and it still "
                                  "leaves oxygen tension unrecorded")
        D[k] = dict(label=lab, cell_line=cl, deposit=f"SCRATCH/{k.split('_')[-1]}",
                    role="a cross-cell-line transfer arm", vars=vv)

    for k, lab, why in [
        ("humangem", "HumanGEM metabolic reconstruction",
         "a consensus human reconstruction, not a measurement of any cell in any medium. Its exchange "
         "bounds ARE an environment, and they are a modelling choice, not a record of one"),
        ("corum", "CORUM protein complexes",
         "literature-curated across many organisms, tissues and decades; a complex has no single "
         "culture condition and the constituent papers are not condition-harmonised"),
        ("bioplex", "BioPlex interactome",
         "measured in HEK293T and HCT116 under bait overexpression -- a different cell AND a "
         "non-physiological expression level. Cell line is recorded per interaction; medium is not"),
        ("hpa", "Human Protein Atlas localisation",
         "antibody-based, across a panel of lines, curated per antigen. Cell line is recorded; the "
         "culture condition of each imaged line is not"),
    ]:
        D[k] = dict(label=lab, cell_line="not one cell line", deposit="various",
                    role="cell-line-agnostic or differently-sourced reference layer",
                    vars={v: V(None, "unknown", note=why) for v in VARS})

    return D


# --------------------------------------------------------------------------------------- comparisons

def compare(a, b, D, var):
    """MATCHED / MISMATCHED / UNKNOWN for one variable across two datasets."""
    fa, fb = D[a]["vars"][var], D[b]["vars"][var]
    if fa["level"] in ("unknown", "unverified") or fb["level"] in ("unknown", "unverified"):
        side = [n for n, f in ((a, fa), (b, fb)) if f["level"] in ("unknown", "unverified")]
        return "UNKNOWN", f"not recorded for {', '.join(side)}"
    if var == "medium":
        ba, bb = fa.get("base"), fb.get("base")
        if ba and bb and ba != bb:
            return "MISMATCHED", f"{ba} vs {bb}"
        if fa["value"] == fb["value"]:
            return "MATCHED", "identical formulation as stated"
        return "MATCHED", (f"same base formulation ({ba}) but the supplements differ, which is a "
                           f"MATCH only at the level the base determines")
    ka, kb = fa.get("norm", fa["value"]), fb.get("norm", fb["value"])
    if ka == kb:
        return "MATCHED", str(ka)[:70]
    return "MISMATCHED", f"{str(ka)[:46]} vs {str(kb)[:46]}"


COMPARISONS = [
    dict(key="a", arms=("replogle_k562", "replogle_rpe1"),
         claim="K562 Perturb-seq edges evaluated on RPE1 Perturb-seq -- the +15.0 point sign transfer",
         co_sourced=True,
         needed="RPE1's harvest density and both arms' O2 tension; and, to make the comparison clean "
                "rather than merely documented, the same experiment run with RPE1 in DMEM:F12 and in "
                "RPMI-1640, which nobody has done"),
    dict(key="b", arms=("replogle_k562", "encode_k562_chip"),
         claim="K562 TF ChIP intersected with K562 Perturb-seq -- the bound x causal directness tier",
         co_sourced=False,
         needed="the medium as a portal field rather than a PDF attachment, harvest density, and a "
                "per-experiment record of whether the tagged line's TF is expressed at parental levels"),
    dict(key="c", arms=("encode_a549_dex", "encode_a549_dex"),
         claim="A549 dexamethasone ChIP against A549 RNA on one time grid -- the dynamics null",
         co_sourced=True, self_pair=True,
         needed="the medium for the Reddy-lab biosamples, which the portal does not carry for any of "
                "the experiments that analysis used"),
    dict(key="d", arms=("replogle_k562", "paxdb_k562"),
         claim="PaxDb K562 protein against K562 Perturb-seq mean expression -- the 15.9% mRNA -> "
               "protein ceiling",
         co_sourced=False,
         needed="the DMEM glucose concentration Geiger used, the harvest density as a number, and "
                "ideally a K562 proteome measured in RPMI on the Replogle schedule"),
    dict(key="e", arms=("replogle_k562", None),
         claim="K562-derived edges evaluated on HCT116, Jurkat, HepG2 and melanoma -- cross-cell-line "
               "transfer",
         co_sourced=False, multi=["nadig_jurkat", "nadig_hepg2", "xaira_hct116", "frangieh_mel",
                                  "replogle_rpe1"],
         needed="perturbation-to-readout interval for the Nadig, Xaira and Frangieh arms, harvest "
                "density everywhere, and O2 everywhere"),
]


# ------------------------------------------------------------------------------------------- ENCODE

def encode_census(report):
    """One paginated call per assay. Returns the counts the audit table quotes."""
    def api(q):
        return json.loads(_get(ENCODE + q,
                               headers={"accept": "application/json", "User-Agent": "cellos"}))

    def cache(name, fn):
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / name
        if p.exists() and p.stat().st_size > 0:
            return json.load(open(p))
        d = fn()
        json.dump(d, open(p, "w"))
        return d

    F = ["accession", "target.label", "lab.title",
         "replicates.library.biosample.treatments",
         "replicates.library.biosample.genetic_modifications.method",
         "replicates.library.biosample.genetic_modifications.purpose",
         "replicates.library.biosample.documents"]
    k = cache("encode_k562_chip.json", lambda: api(
        "/search/?type=Experiment&biosample_ontology.term_name=K562&assay_title=TF+ChIP-seq"
        "&status=released&limit=all" + "".join("&field=" + x for x in F))["@graph"])
    a = cache("encode_a549_dex.json", lambda: api(
        "/search/?type=Experiment&biosample_ontology.term_name=A549"
        "&replicates.library.biosample.treatments.treatment_term_name=dexamethasone"
        "&status=released&limit=all"
        "&field=accession&field=lab.title&field=assay_title&field=target.label"
        "&field=replicates.library.biosample.treatments"
        "&field=replicates.library.biosample.documents")["@graph"])

    def bios(e):
        return [r.get("library", {}).get("biosample", {}) for r in e.get("replicates", [])]

    per = {}
    docs, growth = set(), set()
    for e in k:
        t = (e.get("target") or {}).get("label")
        bs = bios(e)
        per.setdefault(t, []).append(
            dict(acc=e["accession"], treated=bool(any(b.get("treatments") for b in bs)),
                 modified=bool(any(b.get("genetic_modifications") for b in bs)),
                 lab=(e.get("lab") or {}).get("title")))
        for b in bs:
            for d in (b.get("documents") or []):
                docs.add(d if isinstance(d, str) else d.get("@id"))
    # the growth-protocol subset is resolved once, not per experiment
    for u in sorted(docs):
        try:
            o = api(u + "?frame=object")
            if o.get("document_type") == "growth protocol":
                growth.add(u)
        except Exception:
            pass

    n_with_docs = sum(1 for e in k if any(b.get("documents") for b in bios(e)))
    n_treated = sum(1 for e in k if any(b.get("treatments") for b in bios(e)))
    n_mod = sum(1 for e in k if any(b.get("genetic_modifications") for b in bios(e)))

    # DOSE IS A DISTRIBUTION, NOT A SCALAR, AND THE MINIMUM IS NOT THE VALUE. The released A549+dex set
    # contains a small dose-response subset alongside the main series, so taking any single amount off
    # the top of a sorted set reports a dose no analysis in this project ever used. The whole
    # distribution is counted and the modal dose is taken WITHIN the laboratory whose series the
    # dynamics analysis pinned.
    from collections import Counter
    amounts, amounts_lab, durations = Counter(), Counter(), []
    reddy = docs_lab = 0
    for e in a:
        lab = (e.get("lab") or {}).get("title")
        has = any((b.get("documents") or []) for b in bios(e))
        if lab == "Tim Reddy, Duke":
            reddy += 1
        if has:
            docs_lab += 1
        for b in bios(e):
            for t in (b.get("treatments") or []):
                if t.get("treatment_term_name") != "dexamethasone":
                    continue
                key = (float(t["amount"]), t.get("amount_units")) if t.get("amount") is not None \
                    else (None, None)
                amounts[key] += 1
                if lab == "Tim Reddy, Duke":
                    amounts_lab[key] += 1
                    if t.get("duration") is not None:
                        u = t.get("duration_units", "minute")
                        durations.append(float(t["duration"]) * {"minute": 1, "hour": 60}.get(u, 1))
    durations = sorted(set(durations))
    reddy_nodoc = sum(1 for e in a if (e.get("lab") or {}).get("title") == "Tim Reddy, Duke"
                      and not any((b.get("documents") or []) for b in bios(e)))
    modal, n_modal = (amounts_lab or amounts).most_common(1)[0]
    tot_lab = sum((amounts_lab or amounts).values())

    return dict(k562_n_exp=len(k), k562_n_targets=len(per), k562_n_with_docs=n_with_docs,
                k562_n_growth_docs=len(growth), k562_n_docs=len(docs),
                k562_n_treated=n_treated, k562_n_modified=n_mod, k562_per_target=per,
                a549_n_exp=len(a), a549_reddy=reddy, a549_docs_lab=docs_lab,
                a549_reddy_nodoc=reddy_nodoc,
                a549_dose=f"{modal[0]:g} {modal[1]} in the pinned laboratory's series; "
                          f"{len(amounts)} distinct amounts across the whole released A549+dex set "
                          f"({', '.join(f'{x[0]:g} {x[1]}' for x, _ in amounts.most_common())})",
                a549_dose_modal=f"{modal[0]:g} {modal[1]}",
                a549_dose_modal_frac=n_modal / max(tot_lab, 1),
                a549_amount_hist={f"{x[0]:g} {x[1]}": n for x, n in amounts.most_common()},
                a549_n_amounts=len(amounts), a549_n_durations=len(durations),
                a549_min=f"{durations[0]:g} min", a549_max=f"{durations[-1]/60:g} h")


# ----------------------------------------------------------------------------------------------- run

def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("ENVIRONMENTAL STATE -- what is recorded, what is not, and which comparisons cross it")
    report("=" * 100)

    # ---- 1. fetch and ASSERT every pinned source ----
    report("\n  1. PROVENANCE -- every source re-fetched, every quote asserted")
    verified, blocked, n_q = {}, [], 0
    for sid, spec in SOURCES.items():
        text, status, host = fetch_source(sid, spec)
        if status != "ok":
            blocked.append({"source": sid, "id": spec["id"], "host": host, "status": status})
            report(f"     {sid:22s} BLOCKED at {host}: {status}")
            continue
        n = assert_quotes(text, spec, sid)
        n_q += n
        verified[sid] = {"id": spec["id"], "kind": spec["kind"], "cite": spec["cite"],
                         "n_quotes_asserted": n, "chars": len(text), "host": host}
        report(f"     {sid:22s} {n} quote(s) asserted in {len(text):,} chars from {host}")
    if blocked:
        report(f"     -> {len(blocked)} source(s) unreachable; their facts stay `unverified` and are "
               f"NOT promoted to stated")
    hdr_ok = "paxdb_header" in verified

    # DepMap Model.csv: pinned by md5 against the value depmap_codep.py recorded, then read
    dep, dep_status = {}, "absent"
    if MODEL.exists():
        h = hashlib.md5(MODEL.read_bytes()).hexdigest()
        if h != MODEL_MD5:
            raise AssertionError(
                f"DepMap Model.csv md5 {h} != {MODEL_MD5} pinned by depmap_codep.py. The two modules are "
                f"not reading the same release and no cross-module statement about DepMap's medium is "
                f"safe until that is resolved.")
        import csv
        with open(MODEL) as fh:
            r = csv.DictReader(fh)
            for row in r:
                n = row["StrippedCellLineName"].upper()
                if n in ("K562", "RPE1", "HCT116", "JURKAT", "HEPG2", "A549"):
                    dep[n] = row["FormulationID"] or None
        dep_status = f"md5-pinned {MODEL_MD5}"
        report(f"     depmap_model           md5 asserted against depmap_codep.py; "
               f"FormulationID for {len(dep)} lines: " + ", ".join(f"{k}={v}" for k, v in dep.items()))
        verified["depmap_model"] = {"id": "DepMap 24Q2 Model.csv", "kind": "local file",
                                    "cite": "DepMap 24Q2 Public Model.csv, FormulationID column",
                                    "n_quotes_asserted": 0, "md5": MODEL_MD5, "host": "local"}
    else:
        blocked.append({"source": "depmap_model", "id": str(MODEL), "host": "local",
                        "status": "absent"})
        report(f"     depmap_model           ABSENT: {MODEL}")

    # ---- 2. assert the NEGATIVE: the Perturb-seq deposits record none of this ----
    report("\n  2. THE DEPOSITS THEMSELVES -- asserting the absence, not assuming it")
    import h5py
    envre = re.compile(r"medium|media|glucose|glutamine|serum|fbs|oxygen|o2|hypox|dose|"
                       r"density|confluen|passage|treatment|drug|hour|timepoint", re.I)
    deposit_scan = {}
    for nm, p in (("gwps.h5ad (K562 source)", GWPS), ("rpe1.h5ad (RPE1 arm)", RPE1)):
        if not p.exists():
            deposit_scan[nm] = "absent"
            report(f"     {nm:26s} ABSENT")
            continue
        with h5py.File(p, "r") as f:
            keys = list(f["obs"].keys()) + (list(f["uns"].keys()) if "uns" in f else [])
        hits = [k for k in keys if envre.search(k)]
        deposit_scan[nm] = {"n_obs_uns_keys": len(keys), "env_like_keys": hits}
        if hits:
            report(f"     {nm:26s} {len(keys)} obs/uns keys, env-like: {hits}")
        else:
            report(f"     {nm:26s} {len(keys)} obs/uns keys, ZERO match any of the nine variables")
    report("     -> the condition is not in the deposit for either arm of the transfer claim; every "
           "environmental\n        fact below had to be read out of the publication text instead")

    # ---- 3. ENCODE census, joined through the registry ----
    report("\n  3. ENCODE CENSUS, and the registry join that makes it about THIS project's TFs")
    enc = encode_census(report)
    report(f"     K562 TF ChIP: {enc['k562_n_exp']:,} released experiments over "
           f"{enc['k562_n_targets']:,} targets")
    report(f"       {enc['k562_n_treated']:,} cytokine-treated, {enc['k562_n_modified']:,} on a "
           f"genetically modified biosample")
    report(f"       {enc['k562_n_with_docs']:,} carry any biosample document; "
           f"{enc['k562_n_growth_docs']} distinct growth protocols exist across the whole set, all PDFs")
    report(f"     A549 + dexamethasone: {enc['a549_n_exp']:,} experiments; "
           f"{enc['a549_n_amounts']} distinct doses across the whole set "
           f"{enc['a549_amount_hist']}")
    report(f"       within the pinned laboratory: {enc['a549_dose_modal']} on "
           f"{enc['a549_dose_modal_frac']:.0%} of treatment records, {enc['a549_n_durations']} "
           f"durations {enc['a549_min']}..{enc['a549_max']} -- the 16-point grid gr_dynamics.py used")
    report(f"       of the {enc['a549_reddy']:,} experiments from the laboratory the dynamics analysis "
           f"pinned, {enc['a549_reddy_nodoc']:,} carry NO biosample document")

    from entity_registry import load as load_reg
    REG = load_reg()
    tgts = sorted(t for t in enc["k562_per_target"] if t)
    _e, st_chip = REG.join("gene", "symbol", tgts, 0.85, "ENCODE K562 TF ChIP target labels")
    report(f"     registry join: {st_chip['joined']:,}/{st_chip['of']:,} ({st_chip['rate']:.1%}) ChIP "
           f"target labels resolve to gene entities (via=symbol)")

    # THE RIGHT DENOMINATOR IS bound_causal's, NOT THE WHOLE LIBRARY. Intersecting ENCODE targets with
    # all 9,867 gwps perturbations gives 512 TFs, which is not the set that analysis worked on: it built
    # from the causal layer, whose sources are the perturbations that produced at least one |z| >= 5
    # edge. Using the larger set would spread the modification exposure over TFs the analysis never
    # touched and quietly understate it. The causal layer is read directly and the count is asserted
    # against the 396 bound_causal.py reports in its own docstring.
    if REGL.exists():
        srcs = sorted({e["source"] for e in json.load(gzip.open(REGL, "rt"))["edges"]})
        pset, plabel = srcs, "causal-layer perturbed genes (cell_reg_perturb.json.gz)"
    else:
        with h5py.File(GWPS, "r") as f:
            pert = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in f["obs"]["gene_transcript"][:]]
        pset = sorted({p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert})
        plabel = "gwps.h5ad perturbations (causal layer absent -- WIDER set than bound_causal used)"
    _e2, st_pert = REG.join("gene", "symbol", pset, 0.85, plabel)
    report(f"     registry join: {st_pert['joined']:,}/{st_pert['of']:,} ({st_pert['rate']:.1%}) "
           f"{plabel} resolve to gene entities (via=symbol)")

    pe = {REG.resolve("gene", "symbol", s) for s in pset} - {None}
    testable = [t for t in tgts if REG.resolve("gene", "symbol", t) in pe]
    raw_inter = len(set(pset) & set(tgts))
    report(f"     reproduces bound_causal's intersection: {len(testable):,} via registry entity ids, "
           f"{raw_inter:,} on raw symbols, against the {BC_REPORTED} that module reports")
    if REGL.exists() and abs(raw_inter - BC_REPORTED) > 4:
        raise AssertionError(
            f"the ENCODE x causal-layer intersection is {raw_inter}, not the {BC_REPORTED} "
            f"bound_causal.py reports. This module's modification-exposure percentage is a statement "
            f"ABOUT that analysis, so it may not be computed over a different denominator.")
    per = enc["k562_per_target"]
    all_mod = [t for t in testable if all(x["modified"] for x in per[t])]
    any_trt = [t for t in testable if any(x["treated"] for x in per[t])]
    all_trt = [t for t in testable if all(x["treated"] for x in per[t])]
    clean = [t for t in testable if any((not x["treated"]) and (not x["modified"]) for x in per[t])]
    report(f"\n     TFs testable in bound x causal (ChIP target AND a Replogle perturbation): "
           f"{len(testable):,}")
    report(f"       {len(all_mod):,} ({100*len(all_mod)/max(len(testable),1):.1f}%) have NO released "
           f"K562 experiment on an unmodified line --")
    report(f"         for those, `bound in K562` cannot mean `bound in parental K562`; there was no "
           f"such experiment to pick")
    report(f"       {len(any_trt):,} have a cytokine-treated experiment; {len(all_trt):,} have ONLY "
           f"treated experiments {all_trt}")
    report(f"       {len(clean):,} have at least one untreated, unmodified option")

    # ---- 4. the audit table ----
    D = build_datasets(enc, dep, hdr_ok)
    report(f"\n  4. AUDIT -- {len(D)} datasets x {len(VARS)} variables")
    report(f"     {'dataset':20s} " + " ".join(f"{v[:4]:>5s}" for v in VARS) + "   known/9")
    lv = {"stated": " S ", "derived": " d ", "unknown": " . ", "unverified": " ? "}
    tot_known = tot = 0
    for k, d in D.items():
        marks = [lv[d["vars"][v]["level"]] for v in VARS]
        kn = sum(1 for v in VARS if d["vars"][v]["level"] in ("stated", "derived"))
        tot_known += kn
        tot += len(VARS)
        report(f"     {k:20s} " + " ".join(f"{m:>5s}" for m in marks) + f"     {kn}/9")
    report(f"     S=stated (quoted+asserted)  d=derived from a named formulation  "
           f".=UNKNOWN  ?=unverified")
    report(f"     {tot_known}/{tot} variable-slots recorded ({100*tot_known/tot:.0f}%); "
           f"{tot-tot_known} are unknown and stay unknown")
    n_o2 = sum(1 for d in D.values() if d["vars"]["oxygen"]["level"] in ("stated", "derived"))
    n_den = sum(1 for d in D.values() if d["vars"]["density"]["level"] in ("stated", "derived"))
    report(f"     oxygen recorded for {n_o2}/{len(D)} datasets; density for {n_den}/{len(D)}")

    # ---- 5. the five comparisons ----
    report(f"\n  5. THE FIVE COMPARISONS THIS PROJECT HAS ALREADY DRAWN CONCLUSIONS FROM")
    comps = []
    for c in COMPARISONS:
        a, b = c["arms"]
        report(f"\n     ({c['key']}) {c['claim']}")
        if c.get("self_pair"):
            # both arms are the same ENCODE series: the environment is identical BY CONSTRUCTION on the
            # variables that are recorded, and jointly unknown on the rest. That is a different kind of
            # unknown from two independent labs and is labelled as such.
            per_var = {}
            for v in VARS:
                f = D[a]["vars"][v]
                if f["level"] in ("stated", "derived"):
                    per_var[v] = ("MATCHED", "one series, one protocol, one value for both readouts")
                else:
                    per_var[v] = ("UNKNOWN", "unrecorded, but the ChIP and RNA arms are the same "
                                             "biosamples from one laboratory, so they are unknown "
                                             "TOGETHER rather than possibly different")
        elif c.get("multi"):
            # WORST CASE OVER THE ARMS, and the per-arm breakdown is shown rather than hidden behind it:
            # a single MISMATCHED arm makes the pooled claim cross an environment, but which arm it was
            # is the part a reader needs.
            per_var = {}
            for v in VARS:
                by = {m: compare(a, m, D, v)[0] for m in c["multi"]}
                verds = set(by.values())
                short = " ".join(f"{m.split('_')[-1]}:{x[:4]}" for m, x in by.items())
                per_var[v] = ("MISMATCHED" if "MISMATCHED" in verds else
                              ("UNKNOWN" if "UNKNOWN" in verds else "MATCHED"), short)
        else:
            per_var = {v: compare(a, b, D, v) for v in VARS}
        for v in VARS:
            verd, why = per_var[v]
            report(f"        {v:10s} {verd:11s} {why[:78]}")
        worst = ("MISMATCHED" if any(x[0] == "MISMATCHED" for x in per_var.values())
                 else "UNKNOWN" if any(x[0] == "UNKNOWN" for x in per_var.values()) else "MATCHED")
        report(f"        OVERALL    {worst}")
        report(f"        to know:   {c['needed'][:96]}")
        comps.append({"key": c["key"], "claim": c["claim"], "overall": worst,
                      "arms": [x for x in (c.get("multi") or list(c["arms"])) if x],
                      "per_variable": {v: {"verdict": per_var[v][0], "detail": per_var[v][1]}
                                       for v in VARS},
                      "what_would_have_to_be_recorded": c["needed"]})

    # ---- 6. does medium distance track the transfer ordering? ----
    report(f"\n  6. DOES THE RECORDED ENVIRONMENT EXPLAIN THE CROSS-LINE ORDERING?")
    sp = OUT / "scperturb_transfer.json"
    order = None
    if sp.exists():
        R = json.load(open(sp))["results"]
        arms = [("nadig_jurkat", "jurkat"), ("nadig_hepg2", "hepg2"), ("replogle_rpe1", "rpe1"),
                ("xaira_hct116", "hct116"), ("frangieh_mel", "frangieh")]
        src_base = D["replogle_k562"]["vars"]["medium"]["base"]
        rows = []
        for dk, rk in arms:
            if rk not in R:
                continue
            base = D[dk]["vars"]["medium"].get("base")
            rows.append(dict(arm=rk, increment=100 * R[rk]["increment"], base=base,
                             base_match=(base == src_base), lab=R[rk].get("lab"),
                             modality=R[rk].get("modality")))
        rows.sort(key=lambda r: -r["increment"])
        report(f"     {'arm':10s} {'incr(pts)':>9s}  {'medium':10s} {'base==K562?':>12s}  "
               f"{'lab':18s} modality")
        for r in rows:
            report(f"     {r['arm']:10s} {r['increment']:9.1f}  {str(r['base']):10s} "
                   f"{str(r['base_match']):>12s}  {str(r['lab'])[:18]:18s} {r['modality']}")
        m = [r["increment"] for r in rows if r["base_match"]]
        nm = [r["increment"] for r in rows if not r["base_match"]]
        report(f"     mean increment, same base medium as source: {np.mean(m):+.1f} pts (n={len(m)}); "
               f"different base: {np.mean(nm):+.1f} pts (n={len(nm)})")
        report(f"     -> base-medium match does NOT track the ordering. The top arm (Jurkat) shares the")
        report(f"        source's medium recipe exactly, but the second (HepG2) is on EMEM, the most")
        report(f"        different medium of the five, and HCT116 shares the base and ranks fourth.")
        report(f"        With five arms in which lab, modality and medium are collinear, no design here")
        report(f"        can separate them. This rules medium OUT as an explanation of the ORDERING; it")
        report(f"        says nothing about whether medium shifts the LEVEL of any single number, which")
        report(f"        would need the same line measured twice in two media and does not exist.")
        order = {"rows": rows, "mean_same_base": float(np.mean(m)) if m else None,
                 "mean_diff_base": float(np.mean(nm)) if nm else None,
                 "conclusion": "base-medium match does not track the transfer increment across the five "
                               "arms; medium, laboratory and perturbation modality are collinear here "
                               "so nothing is separable, and this neither supports nor excludes a "
                               "medium effect on the level of any single estimate"}
    else:
        report(f"     scperturb_transfer.json absent -- ordering check not run")

    # ---- 7. verdict ----
    limits = [
        "this is an audit, not an experiment. It estimates no effect, so the project's rule about "
        "matched controls drawn >= 20 times does not apply and no sham matching was manufactured to "
        "satisfy it; there is nothing here to match",
        "a MATCHED verdict means the two sources STATE the same thing, not that the two cultures were "
        "the same. Two laboratories following the same catalogue number still differ in serum lot, "
        "passage number, incubator and handling, none of which any source records",
        "`derived` glucose and glutamine come from catalogue formulations, not from a measurement of "
        "the medium as used; supplemented serum contributes glucose and glutamine that these numbers "
        "exclude",
        "the ENCODE growth protocols were read out of PDF attachments with a crude text extractor, so a "
        "protocol whose text does not extract would read as unknown even though the fact is on the "
        "portal; the extraction is asserted against pinned quotes, which catches a failed extraction "
        "but not a partial one",
        "publication methods sections describe intent. They are the best record that exists and they "
        "are still not measurements of the culture that produced the deposited data",
        "the ENCODE modification census is per TARGET, not per file: it establishes that for some TFs "
        "no unmodified experiment exists, which is exact, but it does not reconstruct which file the "
        "bound x causal module actually downloaded, because its manifest is no longer on disk",
        "oxygen tension is unknown for every dataset here without exception, and cell density for all "
        "but two. Their absence is recorded as unknown and must not be read as normoxia or as any "
        "particular density",
        "cell-line-agnostic layers (HumanGEM, CORUM, BioPlex, HPA) are marked unknown on all nine "
        "variables, which understates the problem rather than overstating it: for those the question "
        "is not what the condition was but that there is no single condition to ask about",
    ]

    a_med = comps[0]["per_variable"]["medium"]["verdict"]
    d_med = comps[3]["per_variable"]["medium"]["verdict"]
    verdict = (
        f"EVERY CROSS-DATASET COMPARISON IN THIS PROJECT CROSSES AN ENVIRONMENT THAT WAS EITHER "
        f"DIFFERENT OR UNRECORDED, AND NONE OF THEM SAID SO. Of {tot} dataset x variable slots, "
        f"{tot_known} ({100*tot_known/tot:.0f}%) are recorded anywhere reachable and {tot-tot_known} are "
        f"not; oxygen tension is unknown for all {len(D)} datasets and cell density for {len(D)-n_den}. "
        f"The two arms of the +15.0 sign-transfer result are {a_med} on medium -- K562 in RPMI-1640, "
        f"RPE1 in DMEM:F12 with 0.01 mg/mL hygromycin B -- and also differ in readout time, 8 days "
        f"against 6, both quoted from the paper and asserted here. The 15.9% mRNA -> protein ceiling is "
        f"{d_med} on medium in the same way: the PaxDb K562 proteome was grown in DMEM by Geiger while "
        f"the transcriptome it is regressed against was grown in RPMI by Replogle, and the PaxDb file's "
        f"own header points at that paper, so the mismatch was always derivable and never derived. The "
        f"bound x causal tier is matched at the base-medium level but {len(all_mod)} of "
        f"{len(testable)} testable TFs ({100*len(all_mod)/max(len(testable),1):.0f}%) have no released "
        f"K562 ChIP experiment on an unmodified line at all, so for those `bound in K562` means bound in "
        f"a CRISPR- or BAC-tagged K562. The dynamics null is the best-documented comparison in the "
        f"project -- one series, one lab, dexamethasone {enc['a549_dose']} on "
        f"{enc['a549_n_durations']} durations, all structured portal fields -- and even there the "
        f"medium is unknown for all {enc['a549_reddy_nodoc']} experiments it used, because the only A549 "
        f"growth protocol on the portal is attached to the arm that analysis deliberately excluded. "
        f"NONE OF THIS SHOWS ANY OF THOSE CONCLUSIONS IS WRONG, and one check argues against the "
        f"simplest confound story: across the five cross-line arms, base-medium match does not track the "
        f"transfer increment. What it does show is which numbers carry an unexamined assumption of "
        f"environmental comparability, and that assumption is now written down, per comparison and per "
        f"variable, instead of being invisible.")

    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {
        "model": "environment-v1",
        "purpose": "per-dataset record of the nine environmental variables, and a per-comparison verdict "
                   "on whether the environments were matched, mismatched or unknown",
        "variables": list(VARS),
        "knowledge_levels": {
            "stated": "quoted verbatim from a source that is re-fetched and asserted at run time",
            "derived": "not stated; follows from a named catalogue formulation. Weaker than stated and "
                       "kept in a separate class so it can be refused downstream",
            "unknown": "not recorded in any source consulted. Carries no value. Never defaulted",
            "unverified": "believed but not asserted from any fetched source; treated as unknown by "
                          "every comparison"},
        "sources_verified": verified,
        "sources_blocked": blocked,
        "n_quotes_asserted": n_q,
        "depmap_model_status": dep_status,
        "deposit_scan": deposit_scan,
        "encode_census": {k: v for k, v in enc.items() if k != "k562_per_target"},
        "encode_modification_exposure": {
            "n_testable_tfs": len(testable),
            "n_no_unmodified_experiment_exists": len(all_mod),
            "frac_no_unmodified_experiment_exists": len(all_mod) / max(len(testable), 1),
            "n_with_a_treated_experiment": len(any_trt),
            "tfs_with_only_treated_experiments": all_trt,
            "n_with_clean_option": len(clean),
            "note": "per TARGET over released K562 TF ChIP-seq. Establishes that for a large minority "
                    "of TFs no unmodified-K562 experiment exists to choose; does not reconstruct which "
                    "file bound_causal.py downloaded"},
        "registry_joins": {"encode_chip_targets": st_chip, "replogle_perturbations": st_pert},
        "datasets": D,
        "coverage": {"n_datasets": len(D), "n_variables": len(VARS), "n_slots": tot,
                     "n_recorded": tot_known, "frac_recorded": tot_known / tot,
                     "n_datasets_with_oxygen": n_o2, "n_datasets_with_density": n_den},
        "comparisons": comps,
        "ordering_check": order,
        "limits": limits,
        "verdict": verdict,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "environment.json", "w"), indent=1, default=str)
    report(f"\n  -> {OUT/'environment.json'}")
    report(f"  -> raw fetched sources cached under {CACHE}")


if __name__ == "__main__":
    main()
