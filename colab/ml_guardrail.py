"""ml_guardrail — a trust boundary for the learned edge-scorer (signal_combiner), from THIS session's test.

Tested this session (three-lens TP53 run): the ML combiner FAILS on HUB genes. Asked for TP53's partners, its
top-15 predictions were mitochondrial ribosomal proteins (MRPS/MRPL — co-essential HOUSEKEEPING genes), and it
ranked TP53's true partner set BELOW random (AUC ~0.43). It confuses co-essentiality with interaction on
high-degree nodes. The curated (fixed) and structural (chain) lenses carried the correct answer (MDM2, EP300, the
apoptosis programme); the ML lens actively misled.
=> For high-degree hubs, DO NOT trust the ML combiner alone -> flag it and defer to curated + structural evidence.
This does not retrain the model; it fences off the region where we PROVED it is unreliable.
"""


def node_degree(C, gene):
    i = C.idx.get(gene)
    return None if i is None else len(C.ppi_adj.get(i, ()))


def hub_threshold(C, pct=0.90):
    """degree at the given percentile of the PPI degree distribution (default top-decile = hub)."""
    degs = sorted(len(v) for v in C.ppi_adj.values())
    return degs[int(pct * len(degs))] if degs else 50


def ml_reliability(C, gene, thr=None):
    """(reliable: bool, note: str). Hubs (>= p90 degree) -> ML combiner UNTRUSTED (validated on TP53)."""
    d = node_degree(C, gene)
    if d is None:
        return False, "gene absent from model"
    thr = hub_threshold(C) if thr is None else thr
    if d >= thr:
        return (False, f"HUB (degree {d} >= p90 {thr}): learned combiner unreliable here "
                       f"(validated on TP53 -> ribosomal hallucination). Defer to curated + structural lenses.")
    return True, f"degree {d} < hub threshold {thr}: learned combiner usable (corroborate with curated when possible)"


def guarded_proba(combiner_proba, C, a_gene, b_gene, thr=None):
    """wrap a combiner.proba(i,j) result with the guardrail: returns {prob, trust, note}. If EITHER endpoint is a
    hub, the ML probability is flagged low-trust (not deleted — the caller decides how to weight it)."""
    thr = hub_threshold(C) if thr is None else thr
    ra, na = ml_reliability(C, a_gene, thr)
    rb, nb = ml_reliability(C, b_gene, thr)
    trust = ra and rb
    note = na if not ra else nb if not rb else "both endpoints below hub threshold"
    return {"ml_prob": combiner_proba, "trust": trust, "note": note}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from complete_cell import CompleteCell
    C = CompleteCell()
    thr = hub_threshold(C)
    print(f"hub threshold (p90 PPI degree) = {thr}")
    for g in ["TP53", "BRAF", "VHL", "MDM2", "MRPS25", "PAH", "GBA"]:
        ok, note = ml_reliability(C, g, thr)
        print(f"  {g:7} degree={node_degree(C,g)}  ML-trusted={ok}")
