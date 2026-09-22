"""connectivity — is the software CONNECTED like a cell? Test it the way you'd test a cell: knock out a component
and watch what propagates. For each knockout classify the response:
  PROPAGATES  — a downstream metric changes (the piece is really wired in; removing it has a direct consequence)
  ROBUST      — the dependent syscall still runs, degraded, via a fallback (a 'backup pathway')
  HARD-FAIL   — the dependent syscall crashes (tightly coupled, no backup)
  SILENT      — nothing changes (siloed; not actually used — the dangerous case)

We temporarily hide an artifact, run the dependent syscall in a FRESH kernel (so it re-reads), classify, restore.
-> outputs/orphan/connectivity.json
"""
import os, sys, json, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def fresh_reason(gene):
    """run reason() in a brand-new kernel so artifact caches are re-read after a knockout."""
    import importlib, cellos
    importlib.reload(cellos)
    return cellos.CellKernel(quiet=True).reason(gene)


def _auc_without(core_only=False, drop=None):
    """essentiality reasoning AUC, optionally dropping a science line — measures the metric consequence."""
    import reason as R, reasoner_core as rc
    from complete_cell import CompleteCell
    C = CompleteCell(); N = len(C.genes)
    truth = np.array([float(C.genes[i].get("dep_frac")) if str(C.genes[i].get("dep_frac")).replace(".", "", 1)
                      .replace("-", "", 1).isdigit() else np.nan for i in range(N)])
    L = R.evidence_lines(C)
    if drop:
        L = {k: v for k, v in L.items() if k != drop}
    return rc.reason_over(L, truth)["weighted_reasoning_auc"]


def knockout(artifact, syscall_gene, present_marker, degraded_marker):
    """hide `artifact`, run reason() on a gene, classify the response, restore."""
    path = f"{OUT}/{artifact}"
    baseline = fresh_reason(syscall_gene)
    had = present_marker in baseline
    bak = path + ".ko"
    result = {"artifact": artifact}
    if not os.path.exists(path):
        result["class"] = "ABSENT"; return result
    shutil.move(path, bak)
    try:
        out = fresh_reason(syscall_gene)
        crashed = "kernel fault" in out or "Traceback" in out
        still_runs = "CONCLUSION" in out or "chaining independent" in out
        line_gone = had and (present_marker not in out)
        if crashed or not still_runs:
            result["class"] = "HARD-FAIL"
        elif line_gone:
            result["class"] = "ROBUST"      # ran without the line = graceful degradation / backup pathway
        elif out == baseline:
            result["class"] = "SILENT"
        else:
            result["class"] = "ROBUST"
        result["still_runs"] = bool(still_runs and not crashed)
        result["line_dropped"] = bool(line_gone)
    finally:
        shutil.move(bak, path)
    return result


def main():
    print("=" * 88)
    print("CONNECTIVITY / ROBUSTNESS — knock out a component, watch what propagates (like a cell)")
    print("=" * 88)

    # (1) METRIC PROPAGATION — does removing a science line change the engine's accuracy? (is it wired in?)
    base = _auc_without()
    d_loc = _auc_without(drop="loc")
    d_str = _auc_without(drop="string_hub")
    print("\n  METRIC PROPAGATION (does the piece actually change the answer?)")
    print(f"    full engine ........................ AUC {base:.3f}")
    print(f"    − localization line ................ AUC {d_loc:.3f}   (Δ {d_loc-base:+.3f})  {'PROPAGATES' if abs(d_loc-base)>0.005 else 'silent'}")
    print(f"    − STRING line ...................... AUC {d_str:.3f}   (Δ {d_str-base:+.3f})  {'PROPAGATES' if abs(d_str-base)>0.005 else 'silent'}")

    # (2) KNOCKOUT — hide the artifact, does the syscall crash (hard-coupled) or degrade (robust/backup)?
    print("\n  ARTIFACT KNOCKOUT (crash = no backup; runs-degraded = backup pathway)")
    kos = [
        knockout("localization.json", "TP53", "localization", "no compartment"),
        knockout("string_degree.json", "TP53", "STRING hub", "no STRING"),
        knockout("reason.json", "TP53", "calibrated P", "P"),
    ]
    for r in kos:
        tag = {"ROBUST": "ROBUST (backup pathway — runs degraded)", "HARD-FAIL": "HARD-FAIL (no backup)",
               "SILENT": "SILENT (siloed — not used!)", "ABSENT": "absent"}.get(r["class"], r["class"])
        print(f"    knock out {r['artifact']:22} → {tag}")

    robust = all(r["class"] in ("ROBUST", "ABSENT") for r in kos)
    propagates = abs(d_loc - base) > 0.005 and abs(d_str - base) > 0.005
    verdict = (
        f"CONNECTED AND ROBUST — like the cell. The layers ARE wired in: removing localization or STRING directly "
        f"drops the engine ({base:.3f}→{d_loc:.3f}/{d_str:.3f}), so a change propagates to the answer. But knocking "
        f"out an artifact does NOT crash the software — the dependent syscall runs DEGRADED via a fallback (a backup "
        f"pathway), exactly like a cell buffering a gene knockout. The honest catch is the SAME as the cell's: this "
        f"robustness can HIDE damage — a corrupted layer silently lowers accuracy unless you RE-RUN the audit (which "
        f"regenerates and re-checks the number). Connectivity is real at the reasoning layer; it is NOT a live "
        f"signal — most modules are batch scripts writing artifacts, wired through the reason engine + ID backbone, "
        f"not a continuously-propagating network."
        if robust and propagates else
        f"PARTIAL: propagation={propagates}, robust={robust} — inspect the knockout classes.")
    print("\n" + "=" * 88 + f"\nVERDICT: {verdict}\n" + "=" * 88)
    json.dump({"metric_propagation": {"base": base, "minus_localization": d_loc, "minus_string": d_str},
               "knockouts": kos, "connected": bool(propagates), "robust": bool(robust), "verdict": verdict},
              open(f"{OUT}/connectivity.json", "w"), indent=1)
    return verdict


if __name__ == "__main__":
    main()
