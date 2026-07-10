"""reasoning_chain_test — a GENUINE, MULTI-PROTEIN test of the chain's one new claim: that interface localisation
adds pathogenicity signal on top of ESM. The demonstration cases were hand-picked; this is blind, and pooled
across proteins so it has the power one protein (VHL) lacked.

The claim under test: "a variant at a partner interface is more likely to be pathogenic." Two honest framings:
  (1) DIRECT   — among in-domain ClinVar variants, is at-interface associated with pathogenic? (interface as a
                 pathogenicity marker at all)
  (2) RESCUE   — among the variants ESM MISSES (functional_score < 0.35), does at-interface separate path/benign?
                 (the specific thing the chain used it for)

Both are pooled across proteins with the COCHRAN-MANTEL-HAENSZEL test, which STRATIFIES by protein — so a protein
with a high base rate can't fake an association (the Simpson's-paradox confound that produced a spurious positive
on VHL alone). Numbering is VERIFIED per protein (PDB residue names must match the UniProt sequence at the same
numbers); a protein that fails self-excludes instead of poisoning the pool.

Panel (numbering-verified, spanning interface density 9-50%):
  VHL/1LM8, HRAS/1WQ1, TP53/1YCS, BRCA1/1T29, MSH2/2O8B, RB1/1O9K   (SMAD4/1U7F dropped: 3% numbering match)
-> outputs/orphan/reasoning_chain_multitest.json
"""
import os, sys, json, urllib.request, math, random
sys.path.insert(0, os.path.dirname(__file__))
import molecular_engine as me
import interface_analysis as ia
import blind_variant_test as bvt

OUT = "outputs/orphan"
_ESM_CACHE = f"{OUT}/rc_test_esm_cache.json"
_3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
         "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
         "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# (gene, acc, pdb, chain) -- all numbering-verified 100% except where noted
PANEL = [
    ("VHL",   "P40337", "1LM8", "V"),
    ("HRAS",  "P01112", "1WQ1", "R"),
    ("TP53",  "P04637", "1YCS", "A"),
    ("BRCA1", "P38398", "1T29", "A"),
    ("MSH2",  "P43246", "2O8B", "A"),
    ("RB1",   "P06400", "1O9K", "A"),
]


def fetch_clinvar_reviewed(acc):
    """STRICT labels: ClinVar-sourced calls only (drop ensembl-only computational). path={pathogenic,likely
    pathogenic}; benign={benign,likely benign}; skip VUS/conflicting. Deduped by (pos,wt,mut)."""
    d = json.load(urllib.request.urlopen(f"https://www.ebi.ac.uk/proteins/api/variation?accession={acc}",
                                          timeout=90, context=bvt._ctx()))
    feats = d["features"] if isinstance(d, dict) else d[0]["features"]
    seen = {}
    for f in feats:
        wt = f.get("wildType", ""); mut = f.get("alternativeSequence", "")
        if len(wt) != 1 or len(mut) != 1 or mut not in "ACDEFGHIKLMNPQRSTVWY":
            continue
        try:
            pos = int(f.get("begin"))
        except (TypeError, ValueError):
            continue
        lab = None
        for sig in (f.get("clinicalSignificances") or []):
            srcs = [s.lower() for s in (sig.get("sources") or [])]
            if "clinvar" not in srcs:
                continue
            t = str(sig.get("type", "")).lower()
            if "conflicting" in t or "uncertain" in t:
                continue
            if "pathogenic" in t:
                lab = "path"
            elif "benign" in t and lab != "path":
                lab = "benign"
        if lab:
            seen[(pos, wt, mut)] = lab
    return [(p, w, m, lab) for (p, w, m), lab in seen.items()]


def _resnames(path, chain):
    out = {}
    for ln in open(path):
        if not ln.startswith("ATOM") or ln[21] != chain:
            continue
        rn = ln[17:20].strip()
        try:
            res = int(ln[22:26])
        except ValueError:
            continue
        if rn in _3to1:
            out.setdefault(res, _3to1[rn])
    return out


def _fs(llr, ent):
    """replicate molecular_engine.functional_score so the ESM-missed gate matches the chain."""
    if llr is None:
        return None
    f = max(0.0, -llr)
    if ent is not None and ent < 1.5:
        f *= 1.4
    return f


def _esm(cache, acc, seq, pos, wt, mut):
    k = f"{acc}:{wt}{pos}{mut}"
    if k in cache:
        return cache[k]
    llr, ent = me.esm_scores(seq, pos, wt, mut)
    cache[k] = [llr, ent]
    return cache[k]


def _binom_tail(k, n, p):
    if n == 0:
        return None
    return round(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)), 4)


def _fisher(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return None
    r1 = a + b
    c1 = a + c
    def hg(x):
        return (math.comb(c1, x) * math.comb(n - c1, r1 - x)) / math.comb(n, r1)
    p_obs = hg(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return round(sum(hg(x) for x in range(lo, hi + 1) if hg(x) <= p_obs * 1.0000001), 4)


def cmh(strata):
    """Cochran-Mantel-Haenszel: pooled association of at-interface x pathogenic, STRATIFIED by protein.
    strata: list of (a,b,c,d) = (path&iface, path&no-iface, benign&iface, benign&no-iface). Immune to per-protein
    base-rate confounding. Returns common odds ratio + continuity-corrected chi2 + p (chi2, 1 df)."""
    or_n = or_d = S = SE = 0.0
    used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        used += 1
        or_n += a * d / n
        or_d += b * c / n
        r1, r2, k1, k2 = a + b, c + d, a + c, b + d
        S += a - r1 * k1 / n
        SE += (r1 * r2 * k1 * k2) / (n * n * (n - 1))
    OR = round(or_n / or_d, 3) if or_d > 0 else None
    chi = ((abs(S) - 0.5) ** 2 / SE) if SE > 0 else 0.0
    p = round(math.erfc((chi / 2) ** 0.5), 4) if chi > 0 else 1.0
    return {"strata_used": used, "common_OR": OR, "cmh_chi2": round(chi, 3), "p": p}


def collect(gene, acc, pdb, chain, cache, cap=160, seed=0):
    """per-protein: verify numbering, pull ClinVar, restrict to structurally-mapped in-domain variants, score
    ESM + at-interface. Returns rows + per-protein stats, or {'skip': reason}."""
    path = ia.fetch_pdb(pdb)
    cx = ia.parse_complex(path)
    rn = _resnames(path, chain)
    acc2, seq = me.uniprot_seq(gene, acc)
    if not seq or not rn:
        return {"gene": gene, "skip": "no seq/chain"}
    match = sum(1 for r, aa in rn.items() if 1 <= r <= len(seq) and seq[r - 1] == aa)
    if match / len(rn) < 0.9:
        return {"gene": gene, "skip": f"numbering mismatch {match}/{len(rn)}"}
    at_iface_pos = {r for r in cx.get(chain, {}) if ia.interface_partners(cx, chain, r)}
    dens = round(len(at_iface_pos) / len(cx.get(chain, {})), 3) if cx.get(chain) else None

    variants = fetch_clinvar_reviewed(acc)
    # keep only variants structurally mapped: in the chain AND the PDB residue name equals the UniProt WT
    mapped = [(p, w, m, lab) for (p, w, m, lab) in variants
              if p in rn and rn[p] == w and 1 <= p <= len(seq) and seq[p - 1] == w]
    random.Random(seed).shuffle(mapped)
    mapped = mapped[:cap]                                     # bound CPU cost; deterministic sample
    rows = []
    for pos, wt, mut, lab in mapped:
        llr, ent = _esm(cache, acc, seq, pos, wt, mut)
        rows.append({"pos": pos, "sub": f"{wt}{pos}{mut}", "label": lab, "llr": llr, "fs": _fs(llr, ent),
                     "at_interface": pos in at_iface_pos})
    P = [r for r in rows if r["label"] == "path"]
    B = [r for r in rows if r["label"] == "benign"]
    esm = bvt.auc([-r["llr"] for r in P if r["llr"] is not None],
                  [-r["llr"] for r in B if r["llr"] is not None])
    p_if = sum(r["at_interface"] for r in P)
    b_if = sum(r["at_interface"] for r in B)
    return {"gene": gene, "pdb": pdb, "chain": chain, "interface_density": dens,
            "n_path": len(P), "n_benign": len(B), "esm_auc": esm,
            "path_iface": p_if, "benign_iface": b_if,
            "path_iface_rate": round(p_if / len(P), 3) if P else None,
            "benign_iface_rate": round(b_if / len(B), 3) if B else None,
            "rows": rows}


def run_panel(panel=None):
    panel = panel or PANEL
    cache = {}
    if os.path.exists(_ESM_CACHE):
        try: cache = json.load(open(_ESM_CACHE))
        except Exception: cache = {}
    per, skipped = [], []
    for gene, acc, pdb, chain in panel:
        r = collect(gene, acc, pdb, chain, cache)
        (skipped if r.get("skip") else per).append(r)
        json.dump(cache, open(_ESM_CACHE, "w"))               # checkpoint ESM cache after each protein
        if r.get("skip"):
            msg = "SKIP " + r["skip"]
        else:
            au = round(r["esm_auc"], 3) if r["esm_auc"] is not None else None
            msg = (f"path={r['n_path']} benign={r['n_benign']} esmAUC={au} "
                   f"iface: path {r['path_iface_rate']} vs benign {r['benign_iface_rate']}")
        print(f"  {gene:6} {msg}")

    # ---- pooled stats, stratified by protein (CMH) ----
    def strata(filter_fn):
        S = []
        for r in per:
            rows = [x for x in r["rows"] if filter_fn(x)]
            a = sum(x["at_interface"] and x["label"] == "path" for x in rows)
            b = sum((not x["at_interface"]) and x["label"] == "path" for x in rows)
            c = sum(x["at_interface"] and x["label"] == "benign" for x in rows)
            d = sum((not x["at_interface"]) and x["label"] == "benign" for x in rows)
            S.append((a, b, c, d))
        return S

    direct = cmh(strata(lambda x: True))                                      # framing (1): all in-domain
    missed = cmh(strata(lambda x: x["fs"] is None or x["fs"] < 0.35))         # framing (2): ESM-missed only

    # pooled ESM-missed rescue precision vs the pooled missed base rate (unstratified reference)
    all_missed = [x for r in per for x in r["rows"] if (x["fs"] is None or x["fs"] < 0.35)]
    resc = [x for x in all_missed if x["at_interface"]]
    resc_path = sum(x["label"] == "path" for x in resc)
    missed_base = round(sum(x["label"] == "path" for x in all_missed) / len(all_missed), 3) if all_missed else None

    pooled_p = sum(r["n_path"] for r in per)
    pooled_b = sum(r["n_benign"] for r in per)
    out = {"panel": [f"{r['gene']}/{r['pdb']}" for r in per], "skipped": [f"{r['gene']}:{r['skip']}" for r in skipped],
           "per_protein": [{k: v for k, v in r.items() if k != "rows"} for r in per],
           "pooled_n_path": pooled_p, "pooled_n_benign": pooled_b,
           "direct_cmh_interface_x_path": direct,          # is at-interface associated with pathogenic (all in-domain)?
           "esm_missed_cmh_interface_x_path": missed,      # ... among ESM-missed (the rescue framing)?
           "rescue_pooled": {"n_missed": len(all_missed), "missed_base_rate": missed_base,
                             "rescued_total": len(resc), "rescued_are_path": resc_path,
                             "rescue_precision": round(resc_path / len(resc), 3) if resc else None}}
    out["verdict"] = _verdict(out)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/reasoning_chain_multitest.json", "w"), indent=2, default=str)
    _print(out)
    return out


def _verdict(o):
    v = []
    d = o["direct_cmh_interface_x_path"]; m = o["esm_missed_cmh_interface_x_path"]
    def call(c, name):
        if c["common_OR"] is None:
            return f"{name}: no data"
        sig = c["p"] < 0.05
        direction = "ENRICHED at interface" if c["common_OR"] > 1 else "DEPLETED at interface" if c["common_OR"] < 1 else "flat"
        return (f"{name}: pathogenic {direction} (stratified OR {c['common_OR']}, CMH p={c['p']}) "
                f"-> {'SIGNIFICANT' if sig else 'not significant'} across {c['strata_used']} proteins")
    v.append(call(d, "DIRECT (all in-domain)"))
    v.append(call(m, "RESCUE (ESM-missed only)"))
    r = o["rescue_pooled"]
    v.append(f"pooled rescue precision {r['rescue_precision']} vs missed base rate {r['missed_base_rate']} "
             f"(n rescued {r['rescued_total']})")
    return v


def _print(o):
    print("=" * 92)
    print(f"REASONING-CHAIN MULTI-PROTEIN TEST   panel={o['panel']}  skipped={o['skipped']}")
    print(f"pooled path={o['pooled_n_path']} benign={o['pooled_n_benign']}")
    print("=" * 92)
    print(f"{'protein':16}{'iface_dens':11}{'esmAUC':8}{'path@iface':12}{'benign@iface':13}")
    for r in o["per_protein"]:
        print(f"  {r['gene']+'/'+r['pdb']:14}{str(r['interface_density']):11}"
              f"{str(round(r['esm_auc'],2) if r['esm_auc'] else None):8}"
              f"{str(r['path_iface_rate'])+' ('+str(r['path_iface'])+'/'+str(r['n_path'])+')':12}"
              f"  {str(r['benign_iface_rate'])+' ('+str(r['benign_iface'])+'/'+str(r['n_benign'])+')'}")
    print("-" * 92)
    for line in o["verdict"]:
        print(f"  -> {line}")


if __name__ == "__main__":
    run_panel()
