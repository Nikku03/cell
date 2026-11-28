"""PROTEINPAGE -- data for the "Proteins" and "Tests" views of the cell-map page (descriptive; no claims).

Asked by the user: "Do the first 3" -- (3) update the cell-map page with the address tags, the partner finder, the
ESM C results, the measured splicing controllers and the ER RNA results.
Per protein (UniProt reviewed, Ensembl): rooms by imaging (HPA, OpenCell), address tags (UniProt); likely partners =
the first 10 candidates of the G9 partner finder, re-ordered with ESM C fit (G10-P1 rule) when esmcpartners.py has run (tested: G9-P1 PASS -- fit (Pfam domain pairs learned on DEV
interactions + similar behaviour in DepMap / ProCan / HPA / K562 knockdowns), then same destination, then made
together), known partners (HPA, OpenCell, Complex Portal) marked; RNA versions that change the address with the ESM C
room prediction (G10 catalogue), where they are used (HPA tissues) and the knockdowns that moved them (ENCODE, G11).
Only shareable sources are written (BioPlex, MitoCarta and APEX-seq stay local; MitoCarta does feed the finder's
destination step as in G9). Proteins with neither domain nor behaviour data get no partner list (the order would be
arbitrary).

Run: python3 -I rem/atlas/proteinpage.py OUTDIR
"""

from __future__ import annotations
import collections
import csv
import gzip
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


eb = _load("esmcbundle")
at, gx, cf = eb.at, eb.gx, eb.cf
cbb = gx.cbb
TAGS = ["signal", "transit", "tm", "gpi", "NLS", "NES", "ER_RETENTION", "PTS1"]


def tests():
    """plain lines for each test since G9, from the recorded result files."""
    a = json.load(open(R("outputs", "address", "address_tests.json"))); e = json.load(open(R("outputs", "esmc", "esmc_tests.json")))
    s = json.load(open(R("outputs", "splice", "splice_tests.json"))); r = json.load(open(R("outputs", "erloc", "erloc_tests.json")))
    p1 = a["G9-P1"]; ep = e["G10-P1"]; e1 = e["G10-E1"]["exam_auroc"]
    rows = [
        ("G9-A1", a["G9-A1"]["gate"]["verdict"], "Address tags predict the room", "Signal tag: 79% in the secretory system vs 41% without. Mitochondria tag: 85% vs 6%. Nucleus tag: 89% vs 55%."),
        ("G9-P1", p1["gate"]["verdict"], "Common-sense partner finder", f"Fit, then same destination, then made together: a known partner in the first 10 tries for {p1['within_10_tries']['full']:.0%} of held-out proteins (random {p1['random_within_10']:.1%})."),
        ("G10-E1", e["G10-E1"]["gate"]["verdict"], "ESM C reads the address from the whole sequence", f"Better than tags alone for {e['G10-E1']['esmc_vs_tags']['share_positive']:.0%} of held-out proteins; nucleus AUROC {e1['tags']['nuclear']:.2f} → {e1['tags+esmc']['nuclear']:.2f}."),
        ("G10-P1", e["G10-P1"]["gate"]["verdict"], "ESM C helps the partner finder", f"Partner within 10 tries {ep['within_10']['g9']:.0%} → {ep['within_10']['esmc']:.0%}; median tries {ep['median_tries']['g9']:.0f} → {ep['median_tries']['esmc']:.0f} (proteins whose shortlist holds a known partner)."),
        ("G11-K1", s["G11-K1"]["gate"]["verdict"], "Splicing proteins decide the RNA version", f"Switching one off shifts the address-changing versions more than other RNA proteins do ({s['G11-K1']['splicing_vs_other']['share_positive']:.0%} of genes); true of all versions, not only address ones."),
        ("G10-E2", e["G10-E2"]["gate"]["verdict"], "ESM C finds addresses with no tag", f"Real but just under the bar: AUROC {e['G10-E2']['esmc']['mean_auroc']:.3f} (bar 0.70)."),
        ("G12-A2", r["G12-A2"]["gate"]["verdict"], "RNA of ER-made proteins sits at the ER", f"Clearly enriched at the ER membrane (APEX-seq) but just under the bar: AUROC {r['G12-A2']['erm']['auroc']:.2f} (bar 0.70)."),
        ("G10-S1", e["G10-S1"]["gate"]["verdict"], "ESM C sees RNA versions that move the protein", f"Real but under the bar: AUROC {e['G10-S1']['esmc']['auroc']:.2f} (bar 0.60)."),
        ("G12-T1", r["G12-T1"]["gate"]["verdict"], "The growing protein pulls its RNA to the mitochondria", f"Real but small: AUROC {r['G12-T1']['puromycin']['auroc']:.2f} (bar 0.60)."),
        ("G12-A3", r["G12-A3"]["gate"]["verdict"], "RNA of mitochondrial proteins waits at the mitochondria", f"No: AUROC {r['G12-A3']['omm']['auroc']:.2f} in HEK293T cells."),
        ("G11-K2", s["G11-K2"]["gate"]["verdict"], "Tissue patterns point to the controlling splicing protein", "No: which splicing protein follows a version across tissues does not predict which one controls it."),
        ("G9-A2/A3", "FAIL", "RNA in ENCODE's membrane fraction", "The fraction is not an ER read-out (ribosome and skeleton RNAs dominate); re-tested as G12 with APEX-seq."),
        ("G9-S1", a["G9-S1"]["gate"]["verdict"], "Address-changing versions make dual-room proteins", "Imaging sees one location per gene, so it cannot tell versions apart."),
    ]
    pc = R("outputs", "address", "partner_checks.json")
    if os.path.exists(pc):
        c = json.load(open(pc)); b, x = c["G13-B1"], c["G13-X1"]
        rows[4:4] = [("G13-B1", b["gate"]["verdict"], "ESM C helps outside the big machines", f"Partners that are contacts, not parts of the same machine: within 10 tries {b['within_10']['g9']:.0%} → {b['within_10']['esmc']:.0%}; median tries {b['median_tries']['g9']:.0f} → {b['median_tries']['esmc']:.0f}."),
                     ("G13-X1", x["gate"]["verdict"], "The partner lists find partners nobody gave them", f"BioPlex partners never used to build the lists: found in the first 10 tries for {x['within_10']:.1%} of held-out proteins, {x['fold_over_random']:.0f}x chance; ESM C beats the finder alone (median tries {x['median_tries']['g9']:.0f} → {x['median_tries']['esmc']:.0f}).")]
    return [dict(id=i, verdict=v, title=t, text=x) for i, v, t, x in rows]


def build(outdir):
    t0 = time.time(); D = at.load_all(); P, MC, ppi, rooms, aclass = D
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e)
    U = sorted(P); ui = {g: i for i, g in enumerate(U)}
    truth = collections.defaultdict(set)
    for p in ppi:
        x, y = tuple(p); truth[x].add(y); truth[y].add(x)
    sug = R("outputs", "address", "partner_suggestions_esmc.json.gz"); only = set()
    if os.path.exists(sug):                                                   # ESM C fit added (G10-P1 rule; esmcpartners.py)
        S = json.load(gzip.open(sug, "rt")); partners = {v["gene"]: v["suggested"] for v in S.values()}
        only = {v["gene"] for v in S.values() if v["how"] == "ESM C fit only"}
        print(f"partner lists with ESM C for {len(partners)} proteins ({len(only)} from ESM C alone)", flush=True)
    else:
        ev = cf.evidences(); with_beh = set().union(*[set(map(str, v[0])) for v in ev.values()])
        has = [g for g in U if P[g]["pfam"] or g in with_beh]
        print(f"partner lists for {len(has)} of {len(U)} proteins", flush=True)
        o = eb.g9_orders(D, U, has, truth, k=10)
        partners = {g: [[sym(U[j]), int(U[j] in truth.get(g, ()))] for j in o["cand"][k_]] for k_, g in enumerate(has)}
    cat = collections.defaultdict(list)
    for r in csv.DictReader(open(R("outputs", "esmc", "G10_isoform_catalogue.csv"))):
        cat[r["gene"]].append([r["isoform"], r["tag_changed"], r["canonical_class"], r["isoform_class"], float(r["shift"]), int(r["moves"])])
    when = {v["gene"]: v for v in json.load(open(R("outputs", "address", "isoform_when_and_governors_v2.json"))).values()}
    gov = {v["gene"]: v for v in json.load(open(R("outputs", "splice", "governors_measured.json"))).values()}
    genes = []
    for g in U:
        v = P[g]; mot = set(v["motifs"]); s_ = sym(g)
        tags = [t for t in TAGS if (v.get(t) if t in ("signal", "transit", "gpi") else (v["tm"] > 0 if t == "tm" else t in mot))]
        row = dict(s=s_, e=g, r=sorted(rooms.get(g, ())), t=tags, tm=int(v["tm"]), n=int(len(truth.get(g, ()))))
        if g in partners:
            row["p"] = partners[g]
            if g in only:
                row["po"] = 1
        if g in cat:
            row["v"] = cat[g]
        if g in when:
            w = when[g]; row["w"] = dict(most=w["tissues_most_tag_changed"], least=w["tissues_least_tag_changed"], follow=w["splicing_proteins_following_across_tissues"][:3])
        if g in gov:
            row["g"] = [[x["line"], x["knocked_down"], x["change"], int(x["splicing_protein"])] for x in gov[g]["governors"][:6]]
        genes.append(row)
    genes.sort(key=lambda r: r["s"])
    os.makedirs(outdir, exist_ok=True)
    json.dump(dict(genes=genes, tests=tests()), open(os.path.join(outdir, "proteins.json"), "w"), separators=(",", ":"))
    if not only and not os.path.exists(sug):
        with gzip.open(R("outputs", "address", "partner_suggestions.json.gz"), "wt") as f:
            json.dump({sym(g): dict(gene=g, suggested=p, known_partners=len(truth.get(g, ()))) for g, p in partners.items()}, f, separators=(",", ":"))
    print(f"{len(genes)} proteins; {len(partners)} partner lists; {len(cat)} with ESM C version predictions; {len(when)} with tissue use; "
          f"{len(gov)} with measured controllers; {os.path.getsize(os.path.join(outdir, 'proteins.json')) / 1e6:.1f} MB ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    build(sys.argv[1])
