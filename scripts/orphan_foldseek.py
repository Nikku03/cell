"""PATH-ORPHAN, STEP 4 (Phase A) -- structural homology via AFDB + Foldseek.

A sequence orphan is often NOT a structure orphan. We pull AlphaFold structures
(no prediction needed -- they exist) for the org's proteins and Foldseek them
against AFDB-cluster reps. A hit to a CHARACTERIZED fold can rescue an orphan;
a hit to another DUF/hypothetical cannot.

This script does the SCORING half (parse + grade), which is the bug-prone part
and is fully testable; the heavy half (UniProt idmapping, AFDB pull, foldseek
easy-search) is run in the Colab notebook and feeds this its `.m8` output.

GRADING:
  fold_best_tm     best alntmscore over hits passing evalue<=EVAL & pLDDT>=70
  fold_hit_named   1 if the best qualifying hit's target is CHARACTERIZED
                   (target annotation is not 'hypothetical'/'DUF'/'uncharacterized')
  fold_hit_any     1 if any qualifying hit at all (even orphan<->orphan)

INPUT (real): foldseek .m8 (query,target,fident,evalue,bits,alntmscore),
              target_annotation.tsv (target_id <tab> product), pLDDT per query
OUTPUT: outputs/orphan/foldhit_<org>.parquet  locus_tag, fold_best_tm,
        fold_hit_named, fold_hit_any, fold_target, fold_target_product

--smoke : validate parsing + the named/loose grading on a synthetic .m8
--real  : read the foldseek .m8 produced on Colab
"""
from __future__ import annotations
import argparse, sys, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
EVAL_MAX = 1e-3
TM_MIN = 0.5
PLDDT_MIN = 70.0
_DARK = re.compile(r'hypothetical|uncharacteri|DUF\d|domain of unknown|putative protein',
                   re.I)


def is_characterized(product):
    return bool(product) and not _DARK.search(product)


def grade_hits(hits, plddt, target_products,
               eval_max=EVAL_MAX, tm_min=TM_MIN, plddt_min=PLDDT_MIN):
    """hits: list of (query, target, fident, evalue, tm). plddt: query->float.
    Returns per-query dict of fold features."""
    by_q = {}
    for q, t, fid, ev, tm in hits:
        if ev > eval_max or tm < tm_min:
            continue
        if plddt.get(q, 100.0) < plddt_min:      # query fold unreliable
            continue
        if q == t:                                # self
            continue
        by_q.setdefault(q, []).append((tm, t))
    out = {}
    for q, lst in by_q.items():
        lst.sort(reverse=True)                    # best tm first
        best_tm, best_t = lst[0]
        named = next(((tm, t) for tm, t in lst
                      if is_characterized(target_products.get(t, ""))), None)
        out[q] = {
            "fold_best_tm": best_tm,
            "fold_hit_any": 1,
            "fold_hit_named": int(named is not None),
            "fold_target": (named[1] if named else best_t),
            "fold_target_product": target_products.get(
                named[1] if named else best_t, ""),
        }
    return out


def _read_m8(path):
    hits = []
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            q, t, fid, ev = c[0], c[1], float(c[2]), float(c[3])
            tm = float(c[5])
            hits.append((q, t, fid, ev, tm))
    return hits


def run_real(args):
    import pandas as pd
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    if not Path(args.m8).exists():
        print(f"ERROR: foldseek .m8 not found at {args.m8} (Colab step)"); return 2
    hits = _read_m8(args.m8)
    plddt = {}
    if args.plddt and Path(args.plddt).exists():
        for line in open(args.plddt):
            k, v = line.split()[:2]; plddt[k] = float(v)
    prods = {}
    if args.target_anno and Path(args.target_anno).exists():
        for line in open(args.target_anno):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                prods[p[0]] = p[1]
    graded = grade_hits(hits, plddt, prods)
    df = pd.read_parquet(bridge)
    rows = []
    for lt in df.locus_tag:
        g = graded.get(lt, {"fold_best_tm": float("nan"), "fold_hit_any": 0,
                            "fold_hit_named": 0, "fold_target": "",
                            "fold_target_product": ""})
        g["locus_tag"] = lt; rows.append(g)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"foldhit_{args.org}.parquet", index=False)
    print(f"  graded {len(hits):,} hits -> {(out.fold_hit_any==1).sum():,} genes "
          f"with a fold hit, {(out.fold_hit_named==1).sum():,} to a NAMED fold")
    print(f"  wrote foldhit_{args.org}.parquet")
    return 0


def run_smoke():
    print("=" * 64)
    print("STEP 4 foldseek SMOKE -- parse + named/loose grading")
    print("=" * 64)
    # is_characterized
    assert is_characterized("ATP synthase subunit beta")
    assert not is_characterized("hypothetical protein")
    assert not is_characterized("DUF484 domain-containing protein")
    assert not is_characterized("uncharacterized protein")
    print("  characterized-vs-dark classification: OK")
    # synthetic hits: g1 best tm is to a DARK target but also has a weaker NAMED
    # hit -> fold_hit_named=1 but target should be the NAMED one, not best tm.
    hits = [
        ("g1", "AF_dark", 0.9, 1e-9, 0.92),     # best tm, but hypothetical
        ("g1", "AF_kinase", 0.4, 1e-5, 0.71),   # weaker, characterized
        ("g2", "AF_dark2", 0.8, 1e-8, 0.85),    # only a dark hit
        ("g3", "AF_x", 0.9, 1e-1, 0.95),        # fails evalue
        ("g4", "AF_y", 0.9, 1e-9, 0.40),        # fails tm
        ("g5", "AF_z", 0.9, 1e-9, 0.95),        # query pLDDT too low
    ]
    plddt = {"g5": 40.0}
    prods = {"AF_dark": "hypothetical protein", "AF_kinase": "serine kinase",
             "AF_dark2": "DUF999 protein", "AF_x": "enzyme", "AF_y": "enzyme",
             "AF_z": "enzyme"}
    g = grade_hits(hits, plddt, prods)
    print(f"  g1: named={g['g1']['fold_hit_named']} target={g['g1']['fold_target']} "
          f"best_tm={g['g1']['fold_best_tm']:.2f}")
    assert g["g1"]["fold_hit_named"] == 1 and g["g1"]["fold_target"] == "AF_kinase"
    assert abs(g["g1"]["fold_best_tm"] - 0.92) < 1e-9      # best_tm still the dark
    assert g["g2"]["fold_hit_any"] == 1 and g["g2"]["fold_hit_named"] == 0
    assert "g3" not in g, "evalue filter"
    assert "g4" not in g, "tm filter"
    assert "g5" not in g, "pLDDT filter"
    print("  g2 dark-only (any=1,named=0), g3 evalue-cut, g4 tm-cut, g5 pLDDT-cut: OK")
    print("\n=== STEP 4 SMOKE PASS ===")
    print("  .m8 grading: evalue/tm/pLDDT filters + named-vs-dark rescue logic")
    print("  validated. Real reads the Colab foldseek output.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--m8", default=str(OUT / "foldseek_result.m8"))
    ap.add_argument("--plddt", default="")
    ap.add_argument("--target_anno", default="")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
