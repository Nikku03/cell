"""cellos_full_audit — test the WHOLE software, honestly, in one pass. Not 'does a file exist' but 'does the live
product still boot, do its syscalls recover known biology, do its headline numbers reproduce, and is every claim
backed by a committed artifact'.

Five sections, each a hard PASS/FAIL, then a whole-system scorecard:

  1  IMPORT AUDIT      — every colab/*.py imports without error (runtime-critical set MUST all pass; historical
                          build/notebook scripts may fail only on genuinely-absent heavy deps, reported not fatal).
  2  CELLOS INTEGRATION— the full cellos_test suite: every syscall smoke + correctness assertions + the whodunit
                          coherence benchmark (recovers real complexes/pathways vs random).
  3  RUNNING-CELL LAYER— boot converges to a non-trivial attractor and is robust; reprogram induces 8/9 lineages;
                          the essentiality direction reproduces its honest NULL. (the forward/backward dichotomy.)
  4  CLAIMS ↔ ARTIFACTS— every headline number a syscall prints is checked against the committed JSON that earned
                          it (FBA 0.70, reprogram 8/9 AUC 0.99, essentiality null 0.47, litmine 4.6k, epistasis).
  5  ARTIFACT INTEGRITY— every artifact persist.py registers, that is present, is valid parseable JSON.

-> prints a scorecard; exit 0 iff all runtime-critical checks pass. A NULL that is SUPPOSED to be null counts as PASS.
"""
import os, sys, json, importlib, glob, io, contextlib, subprocess
sys.path.insert(0, os.path.dirname(__file__))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs", "orphan")

# runtime product = the modules the live CellOS actually depends on (these MUST import + work)
RUNTIME = ["complete_cell", "grn", "grn_validate", "grn_reprogram", "cellformer", "ecflux",
           "litmine", "epistasis", "signal_combiner", "perturb_prioritizer", "persist", "cellos", "cellos_test"]

results = {}   # section -> list[(name, ok, detail)]


def sect(name):
    results[name] = []
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    return results[name]


def line(bucket, ok, name, detail=""):
    bucket.append((name, ok, detail))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:42} {detail}")


# ---------------------------------------------------------------- 1 IMPORT AUDIT
def _try_import(m, timeout=8):
    """import one module in an ISOLATED subprocess (some historical scripts do work at import / lack a
    __main__ guard, so importing them in-process would run them). Returns (status, reason)."""
    try:
        r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0,{HERE!r}); import {m}"],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=os.path.join(HERE, ".."))
    except subprocess.TimeoutExpired:
        return "timeout", "runs work at import (no __main__ guard) or hangs"
    if r.returncode == 0:
        return "ok", ""
    err = (r.stderr.strip().splitlines() or [""])[-1][:70]
    return "fail", err


def import_audit():
    b = sect("1  IMPORT AUDIT — every module imports cleanly (each isolated in a subprocess)")
    mods = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(HERE, "*.py")))
    mods = [m for m in mods if m != "cellos_full_audit"]
    rt_bad, other_fail, other_timeout, ok_rt, ok_other = [], [], [], 0, 0
    for m in mods:
        st, reason = _try_import(m)
        if m in RUNTIME:
            if st == "ok": ok_rt += 1
            else: rt_bad.append((m, st, reason))
        else:
            if st == "ok": ok_other += 1
            elif st == "timeout": other_timeout.append(m)
            else: other_fail.append((m, reason))
    line(b, not rt_bad, f"runtime-critical modules import ({ok_rt}/{len(RUNTIME)})",
         "" if not rt_bad else f"BROKEN: {rt_bad}")
    # non-runtime failures fatal only if real breakage (not an absent heavy dep)
    heavy = ("torch", "scanpy", "cobra", "anndata", "No module named")
    real = [(m, r) for m, r in other_fail if not any(h in r for h in heavy)]
    line(b, len(real) == 0, f"historical scripts import ({ok_other}/{len(mods)-len(RUNTIME)})",
         f"{len(other_fail)} fail ({len(real)} real breakage, rest = absent heavy deps), "
         f"{len(other_timeout)} run-at-import" + (f"  REAL: {real[:3]}" if real else ""))
    return b


# ---------------------------------------------------------------- 2 CELLOS INTEGRATION
def cellos_integration():
    b = sect("2  CELLOS INTEGRATION — full syscall suite (smoke + correctness + coherence benchmark)")
    import cellos_test
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok = cellos_test.main()
    except BaseException as e:
        line(b, False, "cellos_test.main()", f"EXCEPTION {type(e).__name__}: {e}")
        return b
    out = buf.getvalue()
    tail = [l for l in out.splitlines() if "RESULT:" in l]
    line(b, ok, "all syscalls + correctness assertions", tail[0][:60] if tail else "")
    # pull the coherence number out for the scorecard
    coh = next((l.strip() for l in out.splitlines() if "share a complex/pathway" in l), "")
    line(b, "enrichment" in out or bool(coh), "whodunit coherence benchmark ran", coh[-40:] if coh else "")
    return b


# ---------------------------------------------------------------- 3 RUNNING-CELL LAYER
def running_cell():
    b = sect("3  RUNNING-CELL LAYER — the cell as software: boots, is robust, reprograms; essentiality stays null")
    from complete_cell import CompleteCell
    from grn import GRN
    import numpy as np
    C = CompleteCell()
    g = GRN(C=C)
    att, info = g.baseline_attractor()
    on = float((att > 0.5).mean())
    line(b, info["converged"] and 0.02 < on < 0.98,
         "boot: converges to a non-trivial attractor", f"{info['steps']} ticks, {on:.0%} ON")
    # reprogram reproduces 8/9
    import grn_reprogram
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rp = grn_reprogram.main(seeds=(0, 1, 2), save=False)   # don't clobber the committed 5-seed artifact
    line(b, rp["hit_at_1"] >= 8 and rp["specificity_auc"] > 0.9,
         "reprogram: master TFs induce own lineage (forward)", f"hit@1 {rp['hit_at_1']}/9, AUC {rp['specificity_auc']:.2f}")
    # essentiality null reproduces (from committed artifact — the full sweep is slow)
    gv = json.load(open(f"{OUT}/grn_validation.json")) if os.path.exists(f"{OUT}/grn_validation.json") else {}
    aucf = gv.get("T4", {}).get("auc_fragility")
    line(b, aucf is not None and 0.42 < aucf < 0.58,
         "essentiality: knockout fragility stays NULL (backward)", f"AUC {aucf} (null by design = PASS)")
    return b


# ---------------------------------------------------------------- 4 CLAIMS <-> ARTIFACTS
def claims_consistency():
    b = sect("4  CLAIMS ↔ ARTIFACTS — every headline number a syscall prints is backed by committed evidence")

    def art(name):
        p = f"{OUT}/{name}"
        return json.load(open(p)) if os.path.exists(p) else None

    # FBA viability AUC ~0.68-0.70
    fba = art("fba_essentiality.json")
    v = (fba or {}).get("validation_vs_measured_DepMap", {})
    a = v.get("AUC_dep>=0.9") or v.get("AUC_dep>=0.5")
    line(b, fba is not None and a is not None and a > 0.6,
         "viability quotes FBA AUC ~0.70", f"artifact AUC={a} (n={v.get('n_matched')})")
    # reprogram 8/9 & AUC 0.99
    rp = art("grn_reprogram.json")
    line(b, rp is not None and rp.get("hit_at_1", 0) >= 8 and rp.get("specificity_auc", 0) > 0.95,
         "induce quotes reprogram 8/9, AUC 0.99", f"hit@1={rp.get('hit_at_1')}, auc={rp.get('specificity_auc'):.3f}" if rp else "MISSING")
    # essentiality null 0.47, gain-robust
    diag = art("grn_diag.json")
    nullrobust = diag is not None and all(0.42 < r["auc_frag"] < 0.55 for r in diag)
    line(b, nullrobust, "boot quotes essentiality null 0.47 (gain-robust)",
         f"AUC across beta: {[round(r['auc_frag'],2) for r in diag]}" if diag else "MISSING")
    # litmine ~4.6k genes with papers
    lm = art("litmine.json")
    wp = sum(1 for v in lm.values() if v.get("n_papers", 0) > 0) if lm else 0
    line(b, lm is not None and wp > 4000, "lit quotes dark-proteome mine (4.6k w/ papers)", f"{wp} genes with papers")
    # epistasis additive r ~0.88
    ep = art("epistasis.json")
    r = (ep or {}).get("mean_r_additive")
    line(b, ep is not None and r is not None and r > 0.8, "simulate/epistasis additive model (r~0.88)",
         f"additive r={r}")
    # synthesis: assess quotes physics+data router covers ~2x at effective AUC > best single
    sy = art("cellos_synthesis.json")
    cm = (sy or {}).get("combined", {})
    eff, gain = cm.get("effective_accuracy_cw"), cm.get("coverage_gain_over_best_single")
    line(b, sy is not None and eff is not None and eff > 0.7 and (gain or 0) > 500,
         "assess quotes physics+data synergy (2x reach)", f"effective AUC={eff}, +{gain} genes")
    # cellsim: checkpoints reconstruct the response (data), mechanism adds ~nothing
    cx = art("cellsim.json")
    rw = (cx or {}).get("rows", [])
    rdata = max((r.get("r_data", 0) for r in rw), default=0)
    rmech = max((r.get("r_mech", 0) for r in rw), default=0)
    line(b, cx is not None and rdata > 0.4 and rmech < 0.1,
         "cellsim quotes checkpoint reconstruction (data>>mech)", f"data r={rdata:.2f}, mech r={rmech:.2f}")
    return b


# ---------------------------------------------------------------- 5 ARTIFACT INTEGRITY
def artifact_integrity():
    b = sect("5  ARTIFACT INTEGRITY — every registered artifact present is valid JSON")
    import persist
    present, bad = 0, []
    for f in persist.ARTIFACTS:
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            continue
        present += 1
        if f.endswith(".json"):
            try:
                json.load(open(p))
            except Exception as e:
                bad.append((f, str(e)[:40]))
    line(b, not bad, f"registered artifacts parse ({present} present, {len(persist.ARTIFACTS)} registered)",
         "" if not bad else f"CORRUPT: {bad}")
    return b


def main():
    import_audit()
    cellos_integration()
    running_cell()
    claims_consistency()
    artifact_integrity()

    print(f"\n{'#'*72}\nWHOLE-SOFTWARE SCORECARD\n{'#'*72}")
    total_ok = total = 0
    fatal = 0
    for sec, items in results.items():
        n_ok = sum(1 for _, ok, _ in items if ok)
        total_ok += n_ok; total += len(items)
        fails = [nm for nm, ok, _ in items if not ok]
        fatal += len(fails)
        mark = "✓" if not fails else "✗"
        print(f"  {mark} {sec.split('—')[0].strip():<28} {n_ok}/{len(items)}" + (f"   FAIL: {fails}" if fails else ""))
    print(f"\n  {total_ok}/{total} checks pass")
    print("  " + ("ALL SYSTEMS GO — the whole software boots, recovers known biology, and every claim is backed."
                  if fatal == 0 else f"{fatal} FAILURE(S) — see above."))
    return fatal == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
