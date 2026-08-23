"""Which transcription factors regulate their own gene, from curation rather than from assumption.

WHY THIS IS SEPARATE FROM THE CLAIM IT SERVES. The proposal is that around half of transcription
factors are self-activating or self-repressing, that such factors do not need a distal element to
influence transcription, and that removing them should clean up an enhancer search. The first half
of that has a source and the second half needs one, so the two are kept apart here.

WHERE THE ~50% COMES FROM. It is an ESCHERICHIA COLI number. Thieffry, Huerta, Perez-Rueda and
Collado-Vides (BioEssays 1998) found that about half of E. coli's characterised transcription
factors bind their own promoter, and Rosenfeld, Elowitz & Alon (JMB 2002) showed why negative
autoregulation is so common there -- it speeds the response time of the gene. Neither result is a
statement about human transcription factors, and this module does not assume it transfers. It
measures the human number from curation and prints both side by side.

WHAT IS USED. TRRUST v2 (Han et al., Nucleic Acids Res 2018), a literature-curated human
TF-target network with a direction on each edge. A self-loop -- an edge whose regulator and target
are the same gene symbol -- is a curated statement that the factor regulates its own gene, with a
PubMed identifier behind it.

THE THREE-WAY SPLIT, WHICH MATTERS MORE THAN THE FRACTION. A factor absent from TRRUST is not
evidence of anything. So matrices are sorted into:

    SELF        curated self-loop, with the mode (activation / repression / unknown) kept
    NO_SELF     present in TRRUST as a regulator, with curated targets, and no self-loop
    UNCURATED   not present in TRRUST as a regulator at all

Only SELF against NO_SELF is a comparison between two measured things. Any test that lumps
UNCURATED in with NO_SELF is partly measuring how well studied a factor is, and that is recorded
here so the loop can gate on it.

THE CONFOUND THIS MODULE MAKES MEASURABLE. Self-loops are found in factors that have been studied,
and studied factors have more curated edges of every kind. So the out-degree of each regulator is
carried alongside the label, which is what lets a loop draw a size- and degree-matched control set
instead of comparing "autoregulatory" against "obscure".

Output: colab/data/tf_autoregulation.json
"""
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from enh import tf_domains as TD             # noqa: E402

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "tf_autoregulation.json"
TRRUST = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
ECOLI_FRACTION = 0.50        # the figure the plan cites, and it is an E. coli figure


def fetch(report=print):
    p = SP / "trrust_rawdata.human.tsv"
    if not p.exists():
        SP.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TRRUST, timeout=300) as r:
            p.write_bytes(r.read())
    rows = []
    for line in p.read_text().splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            rows.append((f[0].upper(), f[1].upper(), f[2], f[3] if len(f) > 3 else ""))
    report(f"    TRRUST v2 human: {len(rows):,} curated edges")
    return rows


def build(report=print):
    rows = fetch(report)
    regulators = sorted({r[0] for r in rows})
    outdeg = Counter(r[0] for r in rows)
    self_mode = {}
    for tf, tgt, mode, pmid in rows:
        if tf == tgt:
            self_mode.setdefault(tf, []).append(mode)
    report(f"    {len(regulators):,} distinct regulators; {len(self_mode):,} carry a curated "
           f"self-loop ({len(self_mode)/len(regulators):.1%})")
    modes = Counter(m for v in self_mode.values() for m in v)
    report(f"    self-loop modes: {dict(modes)}")

    dom = TD.load()
    lab, hits = {}, Counter()
    for mid, rec in dom.items():
        name = (rec.get("name") or "").upper().split("::")[0].replace("(VAR.2)", "").strip()
        if name in self_mode:
            cls = "SELF"
        elif name in outdeg:
            cls = "NO_SELF"
        else:
            cls = "UNCURATED"
        hits[cls] += 1
        lab[mid] = dict(name=name, cls=cls, outdeg=int(outdeg.get(name, 0)),
                        self_modes=self_mode.get(name, []))
    n = len(lab)
    report(f"    {n} JASPAR matrices: SELF {hits['SELF']} ({hits['SELF']/n:.1%}), "
           f"NO_SELF {hits['NO_SELF']} ({hits['NO_SELF']/n:.1%}), "
           f"UNCURATED {hits['UNCURATED']} ({hits['UNCURATED']/n:.1%})")
    curated = hits["SELF"] + hits["NO_SELF"]
    if curated:
        report(f"    among the {curated} CURATED matrices, the self-regulating fraction is "
               f"{hits['SELF']/curated:.1%}; the plan's figure of {ECOLI_FRACTION:.0%} is an "
               f"E. coli number (Thieffry et al. 1998), not a human one")
    for cls in ("SELF", "NO_SELF", "UNCURATED"):
        d = [v["outdeg"] for v in lab.values() if v["cls"] == cls]
        if d:
            d = sorted(d)
            report(f"      {cls:10} out-degree median {d[len(d)//2]:5d}  mean {sum(d)/len(d):8.1f}")

    DATA.mkdir(parents=True, exist_ok=True)
    json.dump(dict(matrices=lab, counts=dict(hits), n_regulators=len(regulators),
                   n_self_loops=len(self_mode), self_loop_modes=dict(modes),
                   ecoli_reference_fraction=ECOLI_FRACTION,
                   source="TRRUST v2 (Han et al., NAR 2018)"), open(OUT, "w"), indent=1)
    report(f"  -> {OUT}")


def load():
    if not OUT.exists():
        raise SystemExit(f"{OUT} missing -- run `python colab/enh/autoregulation.py` first")
    return json.load(open(OUT))


if __name__ == "__main__":
    print("=" * 100)
    print("SELF-REGULATING TRANSCRIPTION FACTORS, from TRRUST curation")
    print("=" * 100)
    build()
