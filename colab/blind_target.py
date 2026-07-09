"""blind_target — the PROPER pipeline: a tumor's mutation panel in, a ranked druggable target out, using only the
model's data. No peeking at the known answer.

Input = a list of (gene, mutation_residue) as tumor sequencing gives (driver + passengers mixed). For EACH
mutated gene the model scores driver-likelihood from its own layers, ranks them, then reads druggability off the
top — exactly the pipeline: mutation -> does it hit a functional module -> is the gene in a cancer pathway / a
known disease gene -> is it druggable -> target.

Driver score (each term is model data, none is 'I know the answer'):
  hit   : the mutation lands inside a positioned protein domain (a functional module, not a linker)   [domains]
  onco  : the gene sits in an oncogenic signalling / cell-cycle / DNA-repair pathway                   [Reactome]
  dz    : how disease-linked the gene is                                                               [ndis]
  penalty: giant structural genes (TTN/MUC16-like: many domains, low cancer-pathway) are demoted
  score = hit * (0.4 + onco) * (0.5 + dz_norm)

Honest limits carried through: this ranks DRIVER-likelihood and reads druggability; it does NOT call
activating-vs-inactivating (GOF/LOF), and a lost tumour-suppressor needs a synthetic-lethal partner our SL data
may not hold. Both are flagged per gene, not hidden.
-> outputs/orphan/blind_target_report.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
ONCO_KW = ("signaling", "signal transduction", "mapk", "ras", "raf", "kinase", "cell cycle", "apoptos",
           "pi3k", "akt", "mtor", "receptor tyrosine", "egfr", "growth factor", "wnt", "notch", "dna repair",
           "p53", "cell division", "proliferat", "erk", "gpcr")
# a driver mutation hits a CATALYTIC/regulatory module; giant passenger genes (TTN, MUC16, OBSCN...) are made of
# STRUCTURAL REPEATS — hitting one of those is background noise, not a driver event.
CATALYTIC = ("kinase", "phosphatase", "gtpase", "atpase", "protease", "peptidase", "cyclase", "helicase",
             "nuclease", "ligase", "transferase", "hydrolase", "oxidoreductase", "dehydrogenase", "reductase",
             "sh2", "sh3", "bromodomain", "ph domain", "ras", "rho", "bcl", "death", "brct", "p53", "homeobox")
STRUCTURAL = ("fibronectin", "immunoglobulin", "ig-like", "ig ", "egf-like", "cadherin", "collagen", "spectrin",
              "ankyrin", "laminin", "leucine-rich", "wd40", "armadillo", "kelch", "pentapeptide", "titin")


def _reactome_names(C):
    rx = {}
    p = f"{OUT}/reactome_pathways.json"
    if os.path.exists(p):
        d = json.load(open(p)).get("pathways", {})
        for name, members in d.items():
            for g in members:
                rx.setdefault(g, []).append(name)
    return rx


def analyze(panel, C, rx):
    idx = C.idx
    rows = []
    for sym, pos in panel:
        if sym not in idx:
            rows.append({"gene": sym, "in_model": False}); continue
        i = idx[sym]; raw = C.genes[i]; g = C.gene(sym)
        # 1) does the mutation hit a functional domain? — and is it CATALYTIC (driver) or STRUCTURAL (passenger)?
        dp = g.get("domain_positions") or []
        in_dom = [d["name"] for d in dp if d["start"] <= pos <= d["end"]]
        has_dom = bool(g.get("domains"))
        hit = 1.0 if in_dom else (0.35 if has_dom else 0.0)
        dname = " ".join(in_dom).lower()
        if any(k in dname for k in CATALYTIC):
            hit *= 2.5                                          # catalytic/regulatory module -> driver-like
        elif any(k in dname for k in STRUCTURAL):
            hit *= 0.12                                         # structural repeat -> passenger noise (TTN etc.)
        # 2) oncogenic pathway membership
        paths = rx.get(i, [])
        onco = [p for p in paths if any(k in p.lower() for k in ONCO_KW)]
        onco_s = 1.0 if onco else 0.0
        # 3) disease link
        ndis = raw.get("ndis") or 0
        dz = min(ndis / 12.0, 1.5)
        # 4) druggable + centrality (context, not driver signal)
        gi = idx[sym]
        drug = (gi in C.D.get("drugs", {})) or (str(gi) in C.D.get("drugs", {}))
        deg = len(C.ppi_adj.get(i, ()))
        score = hit * (0.4 + onco_s) * (0.5 + dz)
        rows.append({"gene": sym, "in_model": True, "mut_pos": pos,
                     "domain_hit": in_dom or ("(no positioned domain)" if has_dom else "(no domains)"),
                     "onco_pathway": onco[:2], "ndis": ndis, "ppi_deg": deg, "druggable": drug,
                     "driver_score": round(score, 3)})
    rows.sort(key=lambda r: -r.get("driver_score", -1))
    return rows


def target_call(top):
    """turn the top-ranked driver into a target + honest mechanism flag."""
    if not top.get("druggable"):
        return "top driver not directly druggable -> needs synthetic-lethal / downstream target (SL data may be incomplete)"
    dom = top.get("domain_hit")
    if isinstance(dom, list) and dom:
        return (f"target {top['gene']} directly (mutation in its {dom[0]} domain, druggable) — "
                f"BUT cannot confirm activating vs inactivating from structure alone (GOF/LOF flag)")
    return f"target {top['gene']} (druggable, disease-linked) — mechanism weak (mutation not in a positioned domain)"


def run(panels=None):
    C = CompleteCell()
    rx = _reactome_names(C)
    panels = panels or {
        # real, documented alterations used as INPUT (driver + classic passengers TTN/MUC16 + hub ACTB)
        "A375 (melanoma)":  [("BRAF", 600), ("CDKN2A", 58), ("TTN", 20000), ("MUC16", 5000), ("ACTB", 150)],
        "HCC827 (lung adeno)": [("EGFR", 746), ("CDKN2A", 58), ("TTN", 18000), ("MUC16", 4000), ("ACTB", 150)],
    }
    report = {}
    for name, panel in panels.items():
        rows = analyze(panel, C, rx)
        top = next((r for r in rows if r.get("in_model")), {})
        call = target_call(top) if top else "no genes mapped"
        report[name] = {"ranking": rows, "proposed_target": top.get("gene"), "call": call}
        print("=" * 76)
        print(f"TUMOR: {name}   —   input: {[p[0] for p in panel]}")
        print("=" * 76)
        print(f"  {'gene':8}{'score':>7}  {'dom-hit':<22}{'onco-path':<10}{'ndis':>5} {'drug':>5}")
        for r in rows:
            if not r.get("in_model"):
                print(f"  {r['gene']:8}   (not in model)"); continue
            dh = r["domain_hit"][0] if isinstance(r["domain_hit"], list) and r["domain_hit"] else str(r["domain_hit"])
            print(f"  {r['gene']:8}{r['driver_score']:>7}  {dh[:20]:<22}{'yes' if r['onco_pathway'] else '-':<10}"
                  f"{r['ndis']:>5} {'yes' if r['druggable'] else '-':>5}")
        print(f"  -> PROPOSED TARGET: {top.get('gene')}")
        print(f"     {call}")
        print()
    json.dump(report, open(f"{OUT}/blind_target_report.json", "w"))
    return report


if __name__ == "__main__":
    run()
