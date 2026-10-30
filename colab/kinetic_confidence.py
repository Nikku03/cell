"""kinetic_confidence — honest uncertainty on kcat-derived rates, from THIS session's measurements.

Tested this session (in-vitro vs in-vivo kcat audit + kcat->flux mismatch):
  - in-vitro kcat OVERESTIMATES the in-vivo apparent rate by ~1 order of magnitude (literature: Davidi 2016).
  - our own `davidi_*` in-vivo field is NOT a trustworthy human reference: measured-kcat vs it gave log-log
    r=0.22 with a ~30x systematic offset -> it behaves like a CROSS-SPECIES proxy, not human in-vivo ground truth.
  - enzymes operate at a median ~60% of catalytic capacity in-cell, and the kcat->flux mismatch spans ~8.6 orders
    of magnitude (p5..p95 of log_excess = -2.0 .. +6.6). A single crowding constant fixes the average, not the
    per-enzyme scatter.
=> No kcat-derived rate is precise. Attach a log10 sigma (fold-uncertainty) and a provenance caveat so nothing
   downstream mistakes an order-of-magnitude estimate for a measured value.
"""

# log10 uncertainty floor by tier (measured in-vitro is tighter than ML-predicted; none beats ~0.5 log)
_SIGMA = {"measured": 0.5, "EC-measured": 0.5,
          "catpred(calibrated)": 0.8, "catpred+ECprior": 0.8,
          "EC-class": 1.0, "network-propagated": 1.0}
_DEFAULT_SIGMA = 1.0                      # priors / unknown tier -> ~1 order of magnitude
_INVITRO_INVIVO = 0.5                     # extra ~0.5 log from the in-vitro->in-vivo gap (Davidi 2016)


def rate_uncertainty(entry):
    """(log10_sigma, [caveats]) for one kinetics_refined entry. sigma combines the tier floor with the
    in-vitro->in-vivo gap in quadrature."""
    base = _SIGMA.get(entry.get("tier"), _DEFAULT_SIGMA)
    sig = (base ** 2 + _INVITRO_INVIVO ** 2) ** 0.5
    caveats = []
    if entry.get("davidi_kcat_max_per_s") is not None or entry.get("invivo_apparent_kcat_per_s") is not None:
        caveats.append("in-vivo field is a CROSS-SPECIES proxy (validated unreliable this session: r=0.22, "
                       "~30x offset) — a weak prior, not human ground truth")
    if entry.get("tier") not in ("measured", "EC-measured"):
        caveats.append("kcat is ML-PREDICTED, not measured")
    return round(sig, 2), caveats


def annotate(kinetics_refined):
    """attach honest uncertainty to every kcat entry, in place. Returns the dict."""
    for g, v in kinetics_refined.items():
        s, caveats = rate_uncertainty(v)
        v["kcat_log10_sigma"] = s
        v["kcat_fold_uncertainty"] = round(10 ** s, 1)     # e.g. 3.5 -> "good to ~3.5x"; 5.6 -> "~5.6x"
        if caveats:
            v["kcat_caveat"] = caveats
    return kinetics_refined


if __name__ == "__main__":
    import json, os, statistics as st
    p = "outputs/orphan/kinetics_refined_corrected.json"
    d = json.load(open(p))["kinetics_refined"]
    annotate(d)
    folds = [v["kcat_fold_uncertainty"] for v in d.values()]
    n_proxy = sum(1 for v in d.values() if any("CROSS-SPECIES" in c for c in v.get("kcat_caveat", [])))
    print(f"annotated {len(d)} enzymes with honest kcat uncertainty")
    print(f"  median fold-uncertainty: {st.median(folds)}x   (range {min(folds)}-{max(folds)}x)")
    print(f"  flagged cross-species in-vivo proxy on {n_proxy} enzymes")
