"""reasoning_chain_test — a GENUINE test of the chain's one new claim: that structural localisation ADDS signal
on top of ESM, rather than just flipping everything to 'significant'. The demonstration (VHL 115/167) was
hand-picked; this is blind.

The method's headline move is: when ESM misses a variant (functional_score < 0.35), the experimental complex can
RESCUE it if the residue sits at a partner interface. That is only a real gain if, AMONG the variants ESM misses,
being at an interface separates pathogenic from benign. If path-missed and benign-missed are equally at
interfaces, the rescue is noise (pure sensitivity, no specificity) and must be reported as such.

Test set: VHL (P40337) — the protein behind the 115 claim, with an experimental complex (1LM8, chain V =
VHL, contacting ElonginB/C + HIF-1a) and enough ClinVar path/benign missense to score. Numbering in 1LM8
chain V is UniProt numbering (validated: 115->HIF, 157->ElonginC).

Reports, honestly:
  - interface DENSITY (what fraction of VHL positions are 'at interface' at all — if ~everything is, the gate is
    vacuous);
  - ESM-alone AUC on path vs benign;
  - the rescue table: among ESM-missed variants, path-at-interface vs benign-at-interface;
  - ESM-alone significance accuracy vs chain (ESM OR interface) significance accuracy.
-> outputs/orphan/reasoning_chain_test.json
"""
import os, sys, json, urllib.request, math
sys.path.insert(0, os.path.dirname(__file__))
import molecular_engine as me
import interface_analysis as ia
import blind_variant_test as bvt

OUT = "outputs/orphan"


def fetch_clinvar_reviewed(acc):
    """STRICT labelled set: only ClinVar-sourced, criteria-provided calls (drop ensembl-only / no-review /
    uncertain). path = {pathogenic, likely pathogenic}; benign = {benign, likely benign}. Deduped by (pos,wt,mut).
    This is the defensible clinical set — noisy predictor calls removed."""
    d = json.load(urllib.request.urlopen(f"https://www.ebi.ac.uk/proteins/api/variation?accession={acc}",
                                          timeout=60, context=bvt._ctx()))
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
                continue                                   # drop ensembl-only computational calls
            t = str(sig.get("type", "")).lower()
            if "conflicting" in t or "uncertain" in t:
                continue                                   # skip VUS / conflicting
            if "pathogenic" in t:
                lab = "path"                               # strict + likely pathogenic
            elif "benign" in t and lab != "path":
                lab = "benign"                             # path dominates if both ever seen
        if lab:
            seen[(pos, wt, mut)] = lab                     # dedupe by (pos,wt,mut)
    return [(p, w, m, lab) for (p, w, m), lab in seen.items()]


def _binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p) -- one-sided; is the pathogenic count in the rescued set above chance?"""
    if n == 0:
        return None
    return round(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)), 4)


def _fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]] -- is interface associated with pathogenicity among ESM-missed?"""
    n = a + b + c + d
    if n == 0:
        return None
    r1, r2, c1 = a + b, c + d, a + c
    def hg(x):
        return (math.comb(c1, x) * math.comb(n - c1, r1 - x)) / math.comb(n, r1)
    p_obs = hg(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return round(sum(hg(x) for x in range(lo, hi + 1) if hg(x) <= p_obs * 1.0000001), 4)


def _fs(llr, ent):
    """replicate molecular_engine's functional_score exactly (so the gate matches the chain)."""
    if llr is None:
        return None
    f = max(0.0, -llr)
    if ent is not None and ent < 1.5:
        f *= 1.4
    return f


def interface_density(pdb, chain, labels, seq_len):
    """fraction of chain positions that contact ANY other chain within 5A (is the 'at interface' gate vacuous?)."""
    cx = ia.parse_complex(ia.fetch_pdb(pdb))
    present = sorted(cx.get(chain, {}).keys())
    at = 0
    for r in present:
        if ia.interface_partners(cx, chain, r):
            at += 1
    return {"structure_positions": len(present), "at_interface": at,
            "density": round(at / len(present), 3) if present else None,
            "covered_range": [min(present), max(present)] if present else None}


def run(gene="VHL", acc="P40337", pdb="1LM8", chain="V", max_each=200):
    labels = {"B": "ElonginB", "C": "ElonginC", "V": "VHL", "H": "HIF1a"}
    dens = interface_density(pdb, chain, labels, 0)
    lo, hi = dens["covered_range"]

    variants = fetch_clinvar_reviewed(acc)
    acc2, seq = me.uniprot_seq(gene, acc)
    rows = []
    for pos, wt, mut, lab in variants:
        if not seq or pos > len(seq) or seq[pos - 1] != wt:
            continue                                   # numbering/sequence mismatch -> skip (honest)
        in_struct = lo <= pos <= hi
        r = ia.analyze(pdb, chain, pos, labels) if in_struct else {"at_interface": False, "contacts": {}}
        llr, ent = me.esm_scores(seq, pos, wt, mut)
        fs = _fs(llr, ent)
        rows.append({"pos": pos, "sub": f"{wt}{pos}{mut}", "label": lab, "llr": llr, "fs": fs,
                     "in_structure": in_struct, "at_interface": bool(r["at_interface"]),
                     "partners": list(r["contacts"])})

    path = [x for x in rows if x["label"] == "path"]
    ben = [x for x in rows if x["label"] == "benign"]

    def esm_auc(rr):
        ps = [-x["llr"] for x in rr if x["label"] == "path" and x["llr"] is not None]
        ns = [-x["llr"] for x in rr if x["label"] == "benign" and x["llr"] is not None]
        return bvt.auc(ps, ns)

    esm = esm_auc(rows)
    # HONEST SCOPE: the rescue test must compare LIKE FOR LIKE -- pathogenic vs benign that are BOTH in the folded
    # domain (where an interface is even defined). Including out-of-domain benign (VHL's disordered N-term, which is
    # never at an interface) fakes a positive by comparing in-domain path against out-of-domain benign.
    in_struct = [x for x in rows if x["in_structure"]]
    esm_in = esm_auc(in_struct)

    # the rescue test: among ESM-MISSED IN-DOMAIN variants (fs < 0.35), does interface separate path from benign?
    missed = [x for x in in_struct if (x["fs"] is None or x["fs"] < 0.35)]
    mp = [x for x in missed if x["label"] == "path"]
    mb = [x for x in missed if x["label"] == "benign"]
    p_at, b_at = sum(x["at_interface"] for x in mp), sum(x["at_interface"] for x in mb)
    rescue = {
        "esm_missed_total": len(missed),
        "path_missed": len(mp), "path_missed_at_interface": p_at,
        "benign_missed": len(mb), "benign_missed_at_interface": b_at,
        # the CORRECT null: pathogenic fraction AMONG the ESM-missed pool (we select the rescued set from it)
        "missed_pool_path_rate": round(len(mp) / len(missed), 3) if missed else None,
        "path_missed_iface_rate": round(p_at / len(mp), 3) if mp else None,
        "benign_missed_iface_rate": round(b_at / len(mb), 3) if mb else None,
    }
    # rescue precision: of ESM-missed variants the structure rescues (flips to significant), how many are truly path?
    rescued = [x for x in missed if x["at_interface"]]
    rescue["rescued_total"] = len(rescued)
    rescue["rescued_are_path"] = sum(x["label"] == "path" for x in rescued)
    rescue["rescue_precision"] = round(rescue["rescued_are_path"] / len(rescued), 3) if rescued else None
    # is the pathogenic count in the rescued set above the missed-pool rate? (binomial) + interface~pathogenicity (Fisher)
    if missed:
        rescue["binom_p_vs_missed_pool"] = _binom_tail(rescue["rescued_are_path"], len(rescued),
                                                       rescue["missed_pool_path_rate"] or 0)
    rescue["fisher_p_interface_x_path"] = _fisher(p_at, len(mp) - p_at, b_at, len(mb) - b_at)
    in_path = sum(x["label"] == "path" for x in in_struct)
    base_rate = round(in_path / len(in_struct), 3) if in_struct else None   # in-domain path prevalence (the honest null)

    def sig_acc(use_struct):
        """accuracy of significant<->pathogenic under two gates -- IN-DOMAIN only (honest scope)."""
        tp = fp = tn = fn = 0
        for x in in_struct:
            sig = (x["fs"] is not None and x["fs"] >= 0.35) or (use_struct and x["at_interface"])
            if x["label"] == "path":
                tp += sig; fn += (not sig)
            else:
                fp += sig; tn += (not sig)
        n = tp + fp + tn + fn
        return {"acc": round((tp + tn) / n, 3) if n else None, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else None,
                "specificity": round(tn / (tn + fp), 3) if (tn + fp) else None}

    esm_gate = sig_acc(False)
    chain_gate = sig_acc(True)

    out = {"gene": gene, "pdb": pdb, "interface_density": dens,
           "n_path": len(path), "n_benign": len(ben), "path_base_rate_all": round(len(path) / len(rows), 3),
           "n_in_domain": len(in_struct), "in_domain_path_base_rate": base_rate,
           "esm_alone_auc_all": esm, "esm_alone_auc_in_domain": esm_in, "rescue_test": rescue,
           "esm_gate": esm_gate, "chain_gate": chain_gate,
           "verdict": _verdict(dens, rescue, base_rate, esm_gate, chain_gate, esm_in),
           "rows": rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/reasoning_chain_test.json", "w"), indent=2, default=str)
    _print(out)
    return out


def _verdict(dens, rescue, base_rate, esm_gate, chain_gate, esm_in=None):
    v = []
    if dens["density"] and dens["density"] > 0.6:
        v.append(f"WARNING interface gate near-vacuous: {dens['density']:.0%} of positions are at some interface")
    if esm_in is not None:
        v.append(f"ESM in-domain AUC {esm_in:.2f} — {'at chance (ESM uninformative here)' if abs(esm_in-0.5)<0.06 else 'informative' if esm_in>0.5 else 'anti-predictive'}")
    rp = rescue["rescue_precision"]
    null = rescue.get("missed_pool_path_rate")          # correct null = pathogenic rate WITHIN the ESM-missed pool
    fp = rescue.get("fisher_p_interface_x_path")
    if rp is not None and null is not None:
        if rescue["rescued_total"] < 5:
            v.append(f"too few ESM-missed-at-interface variants ({rescue['rescued_total']}) to conclude")
        elif rp > null + 0.05:
            v.append(f"rescue ADDS signal: rescued {rp:.0%} pathogenic vs {null:.0%} within-missed-pool null "
                     f"(interface~path Fisher p={fp}); ESM-missed pathogenic hit an interface "
                     f"{rescue['path_missed_iface_rate']:.0%} vs benign {rescue['benign_missed_iface_rate']:.0%}")
        elif rp < null - 0.05:
            v.append(f"rescue HURTS: rescued {rp:.0%} pathogenic, below {null:.0%} within-missed-pool null")
        else:
            v.append(f"rescue NEUTRAL: rescued {rp:.0%} ~ {null:.0%} within-missed-pool null (no clear gain)")
    d_sens = (chain_gate["sensitivity"] or 0) - (esm_gate["sensitivity"] or 0)
    d_spec = (chain_gate["specificity"] or 0) - (esm_gate["specificity"] or 0)
    v.append(f"gate change: sensitivity {d_sens:+.2f}, specificity {d_spec:+.2f} (ESM->chain)")
    return v


def _print(o):
    print("=" * 84)
    print(f"REASONING-CHAIN TEST  {o['gene']}  ({o['pdb']})   path={o['n_path']} benign={o['n_benign']} "
          f"| in-domain {o['n_in_domain']} (base-rate {o['in_domain_path_base_rate']})")
    print("=" * 84)
    d = o["interface_density"]
    print(f"  interface density: {d['at_interface']}/{d['structure_positions']} positions at interface "
          f"= {d['density']:.0%}  (range {d['covered_range']})")
    print(f"  ESM-alone AUC: all {round(o['esm_alone_auc_all'],3)}  |  in-domain {round(o['esm_alone_auc_in_domain'],3)} "
          f"(the honest number — N-term benign removed)")
    r = o["rescue_test"]
    print(f"  ESM-missed (fs<0.35): {r['esm_missed_total']}  | path-missed {r['path_missed']} "
          f"(at-iface {r['path_missed_at_interface']})  benign-missed {r['benign_missed']} "
          f"(at-iface {r['benign_missed_at_interface']})")
    print(f"  RESCUE precision: {r['rescued_are_path']}/{r['rescued_total']} rescued are pathogenic "
          f"= {r['rescue_precision']}  (within-missed-pool null {r.get('missed_pool_path_rate')}, "
          f"binom p={r.get('binom_p_vs_missed_pool')})")
    print(f"  interface rate among ESM-missed:  pathogenic {r['path_missed_iface_rate']}  "
          f"benign {r['benign_missed_iface_rate']}  (Fisher p={r.get('fisher_p_interface_x_path')})")
    print(f"  ESM gate : acc {o['esm_gate']['acc']}  sens {o['esm_gate']['sensitivity']}  "
          f"spec {o['esm_gate']['specificity']}")
    print(f"  chain gate: acc {o['chain_gate']['acc']}  sens {o['chain_gate']['sensitivity']}  "
          f"spec {o['chain_gate']['specificity']}")
    for line in o["verdict"]:
        print(f"  -> {line}")


if __name__ == "__main__":
    run()
