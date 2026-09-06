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
import os, sys, json, re, urllib.request, ssl
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
_CA = "/root/.ccr/ca-bundle.crt"
_SITE_RE = re.compile(r"(?:ACT_SITE|BINDING|SITE)\s+(\d+)(?:\.\.(\d+))?")
SITE_WINDOW = 30                                    # residues: a driver mutation sits near the catalytic machinery


def fetch_functional_sites(genes):
    """UniProt per-residue functional sites (active/binding/site) -> {gene: sorted[positions]}. The principled
    driver signal: a mutation NEAR the catalytic machinery vs a structural passenger far from any functional site."""
    ctx = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None
    out = {}
    for g in genes:
        q = (f"https://rest.uniprot.org/uniprotkb/search?query=gene:{g}+AND+organism_id:9606+AND+reviewed:true"
             f"&fields=gene_primary,ft_act_site,ft_binding,ft_site&format=tsv&size=1")
        try:
            d = urllib.request.urlopen(q, timeout=60, context=ctx).read().decode().splitlines()
            if len(d) < 2:
                out[g] = []; continue
            text = "\t".join(d[1].split("\t")[1:])
            pos = set()
            for a, b in _SITE_RE.findall(text):
                a = int(a); b = int(b) if b else a
                pos.update(range(a, b + 1))
            out[g] = sorted(pos)
        except Exception:
            out[g] = []
    return out
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


def load_ml_prob(C):
    """Load the cell WITH the trained ML model, exactly the earlier-times way (CompleteCell + DepMap
    co-essentiality + the trained signal_combiner). Returns prob(i, j) -> calibrated P(the two genes interact),
    or None if the ML artifacts are absent. This is the SAME edge model phase2_loop heals the network with —
    here it replaces the hand-set disease-popularity term with an ML view of how a gene wires into the cancer
    machinery."""
    try:
        import numpy as np, signal_combiner
        m = signal_combiner.load(f"{OUT}/signal_combiner.pkl")
        if not m:
            print("  (no trained combiner -> ML lens off, hand-scored fallback)")
            return None
        try:
            from phase3_depmap import DepMapEdges
            dm = DepMapEdges()                                     # DepMap co-essentiality (from the cached matrix)
        except Exception as e:
            dm = None; print("  (co-essentiality off:", str(e)[:50], ")")
        feats = set(m.get("features_all", m.get("features")))
        sc = signal_combiner.SignalCombiner(C=C, dm=dm, only=feats)
        kept = [sc.features_list.index(f) for f in m["features"]]
        def prob(u, v):
            full = sc.features(u, v)
            x = np.array([[full[i] for i in kept]])
            if m.get("uses_scaler"):
                x = m["scaler"].transform(x)
            return float(m["model"].predict_proba(x)[0, 1])
        print(f"  ML combiner loaded (features {m['features']}, thr {round(m['threshold'],3)})")
        return prob
    except Exception as e:
        print("  (ML combiner unavailable:", str(e)[:80], ") -> hand-scored fallback")
        return None


def oncogenic_core(C, rx):
    """the cancer machinery, as model indices: genes sitting in an oncogenic Reactome pathway. Defined from the
    pathway layer alone (blind to which gene is any tumour's answer)."""
    core = set()
    for i, paths in rx.items():
        if any(k in p.lower() for p in paths for k in ONCO_KW):
            core.add(i)
    return core


def ml_wiring(i, C, prob, core, cache):
    """ML network-corroboration for gene index i: how tightly the trained combiner wires it into the cancer
    machinery. mean of its top-3 calibrated edge probabilities to oncogenic-core neighbours — a driver sits in
    the signalling core; a giant structural passenger does not. Blind to the answer (uses network + ML only)."""
    if prob is None:
        return None
    if i in cache:
        return cache[i]
    partners = [j for j in C.ppi_adj.get(i, ()) if j in core and j != i][:60]   # cap for cost
    ps = sorted((prob(i, j) for j in partners), reverse=True)[:3]
    val = round(sum(ps) / len(ps), 3) if ps else 0.0
    cache[i] = val
    return val


def analyze(panel, C, rx, sites, prob=None, core=None, mlcache=None):
    idx = C.idx
    core = core if core is not None else set()
    mlcache = mlcache if mlcache is not None else {}
    rows = []
    for sym, pos in panel:
        if sym not in idx:
            rows.append({"gene": sym, "in_model": False}); continue
        i = idx[sym]; raw = C.genes[i]; g = C.gene(sym)
        # 1) PRIMARY: is the mutation NEAR the catalytic machinery? (functional-site proximity — principled)
        sp = sites.get(sym, [])
        dp = g.get("domain_positions") or []
        in_dom = [d["name"] for d in dp if d["start"] <= pos <= d["end"]]
        if sp:
            dmin = min(abs(pos - s) for s in sp)
            if dmin <= SITE_WINDOW:
                hit, site_note = 2.5, f"{dmin} aa from a functional site"          # near active site -> driver-like
            else:
                hit, site_note = 0.12, f"far ({dmin} aa) from any functional site"  # structural region -> passenger
        else:
            # FALLBACK (no site annotation): catalytic vs structural domain type
            has_dom = bool(g.get("domains"))
            hit = 1.0 if in_dom else (0.35 if has_dom else 0.0)
            dname = " ".join(in_dom).lower()
            if any(k in dname for k in CATALYTIC):
                hit *= 2.5
            elif any(k in dname for k in STRUCTURAL):
                hit *= 0.12
            site_note = "no functional-site data (domain-type fallback)"
        # 2) oncogenic pathway membership
        paths = rx.get(i, [])
        onco = [p for p in paths if any(k in p.lower() for k in ONCO_KW)]
        onco_s = 1.0 if onco else 0.0
        # 3) NETWORK term. ML-loaded: the trained combiner's wiring into the cancer core (earlier-times cell+ML).
        #    Falls back to the hand disease-popularity term (ndis) only when the ML artifacts are absent.
        ndis = raw.get("ndis") or 0
        mlw = ml_wiring(i, C, prob, core, mlcache) if prob is not None else None
        if mlw is not None:
            net_term = mlw
            net_note = f"ML-wire {mlw:.2f}"
        else:
            net_term = min(ndis / 12.0, 1.5)
            net_note = f"dz {net_term:.2f} (hand)"
        # 4) druggable + centrality (context, not driver signal)
        gi = idx[sym]
        drug = (gi in C.D.get("drugs", {})) or (str(gi) in C.D.get("drugs", {}))
        deg = len(C.ppi_adj.get(i, ()))
        score = hit * (0.4 + onco_s) * (0.5 + net_term)
        rows.append({"gene": sym, "in_model": True, "mut_pos": pos, "site_note": site_note,
                     "domain_hit": in_dom or "(none)", "net_note": net_note, "ml_wiring": mlw,
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
    prob = load_ml_prob(C)                                     # load the cell WITH the ML model (earlier-times way)
    core = oncogenic_core(C, rx)
    mlcache = {}
    print(f"  oncogenic core (pathway layer): {len(core):,} genes | ML lens: {'ON' if prob else 'off'}")
    panels = panels or {
        # real, documented alterations used as INPUT (driver + classic passengers TTN/MUC16 + hub ACTB)
        "A375 (melanoma)":  [("BRAF", 600), ("CDKN2A", 58), ("TTN", 20000), ("MUC16", 5000), ("ACTB", 150)],
        "HCC827 (lung adeno)": [("EGFR", 858), ("CDKN2A", 58), ("TTN", 18000), ("MUC16", 4000), ("ACTB", 150)],
    }
    allgenes = sorted({g for panel in panels.values() for g, _ in panel})
    print(f"  fetching UniProt functional sites for {len(allgenes)} genes ...")
    sites = fetch_functional_sites(allgenes)
    report = {}
    for name, panel in panels.items():
        rows = analyze(panel, C, rx, sites, prob=prob, core=core, mlcache=mlcache)
        top = next((r for r in rows if r.get("in_model")), {})
        call = target_call(top) if top else "no genes mapped"
        report[name] = {"ranking": rows, "proposed_target": top.get("gene"), "call": call}
        print("=" * 78)
        print(f"TUMOR: {name}   —   input: {[p[0] for p in panel]}")
        print("=" * 78)
        print(f"  {'gene':8}{'score':>7}  {'functional-site test':<32}{'network(ML)':<14}{'onco':<5}{'drug':>5}")
        for r in rows:
            if not r.get("in_model"):
                print(f"  {r['gene']:8}   (not in model)"); continue
            print(f"  {r['gene']:8}{r['driver_score']:>7}  {r['site_note'][:30]:<32}{r.get('net_note',''):<14}"
                  f"{'yes' if r['onco_pathway'] else '-':<5}{'yes' if r['druggable'] else '-':>5}")
        print(f"  -> PROPOSED TARGET: {top.get('gene')}")
        print(f"     {call}")
        print()
    json.dump(report, open(f"{OUT}/blind_target_report.json", "w"))
    return report


if __name__ == "__main__":
    run()
