"""PATH-ORPHAN, NOVELTY #1 -- the acute x evolutionary 2x2 lens.

Cross the two independent perturbation experiments:
  ACUTE axis        = does knockout hurt NOW (FB-measured essentiality)
  EVOLUTIONARY axis = is the gene retained or labile across lineages
                      (ortholog breadth across the panel)

The DISAGREEMENT cells are the discovery (no one has crossed measured fitness x
pangenome retention to partition essentiality this way):
  essential + retained  -> CORE / housekeeping
  essential + LABILE    -> CONDITIONAL / niche-essential  (the prize)
  non-ess  + RETAINED   -> BUFFERED / redundant           (kept despite dispensable)
  non-ess  + labile     -> accessory / dispensable

Non-circular validation (enrichments that must hold if the lens is real):
  - BUFFERED quadrant enriched for in-genome PARALOGS (redundancy explains why a
    retained gene is dispensable)
  - CONDITIONAL quadrant enriched for transport/metabolic + has FEW paralogs
    (no backup -> essential in its niche) and is dark-rich
  - CORE quadrant under strongest purifying selection (lowest dN/dS)

--smoke : synthetic data with planted enrichments -> lens recovers them
--real  : run on outputs/orphan/bridge_<org>.parquet (+ dnds)
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
RETAINED_MIN = 24          # present in >= half the 48-org panel = retained
DARK = re.compile(r'hypothetical|uncharacteri|DUF\d|domain of unknown|putative protein', re.I)
META = re.compile(r'transport|permease|abc|efflux|synth|transferase|reductase|'
                  r'dehydrogenase|kinase|hydrolase|oxidase|lyase|isomerase|'
                  r'carboxylase|biosynth|metabol|channel', re.I)


def quadrant(essential, n_orgs, retained_min=RETAINED_MIN):
    retained = n_orgs >= retained_min
    if essential and retained:   return "core"
    if essential and not retained: return "conditional"
    if not essential and retained: return "buffered"
    return "accessory"


def enrich(sub, full, mask_fn, label):
    import numpy as np
    a = np.mean([mask_fn(r) for r in sub]) if len(sub) else float("nan")
    b = np.mean([mask_fn(r) for r in full]) if len(full) else float("nan")
    lift = a / b if b else float("nan")
    return f"{label}: {a:.2f} vs {b:.2f} (lift {lift:.2f}x)"


def run_real(args):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    df = pd.read_parquet(bridge)
    dn = OUT / f"dnds_{args.org}.parquet"
    if dn.exists():
        df = df.merge(pd.read_parquet(dn)[["locus_tag", "dnds"]], on="locus_tag",
                      how="left")
    else:
        df["dnds"] = np.nan
    df["n_orgs"] = pd.to_numeric(df["family_n_organisms"], errors="coerce").fillna(0)
    df["npar"] = pd.to_numeric(df["n_paralogs_in_genome"], errors="coerce").fillna(0)
    df["quad"] = [quadrant(e == 1, n) for e, n in zip(df.essential, df.n_orgs)]

    print("=" * 70)
    print(f"ACUTE x EVOLUTIONARY 2x2 -- {args.org} (retained = >={RETAINED_MIN} orgs)")
    print("=" * 70)
    rows = [r._asdict() if hasattr(r, "_asdict") else r for r in df.to_dict("records")]
    def mask_paralog(r): return r["npar"] > 0
    def mask_meta(r): return bool(META.search(str(r["product"])))
    def mask_dark(r): return bool(DARK.search(str(r["product"])))
    counts = df.quad.value_counts().to_dict()
    print(f"\n  {'quadrant':<12}{'n':>6}{'%ess':>6}{'has_paralog':>12}"
          f"{'metabolic':>10}{'dark':>7}{'medianω':>9}")
    order = ["core", "conditional", "buffered", "accessory"]
    res = {}
    for q in order:
        sub = df[df.quad == q]
        if not len(sub): continue
        par = (sub.npar > 0).mean()
        met = sub["product"].fillna("").str.contains(META).mean()
        drk = sub["product"].fillna("").str.contains(DARK).mean()
        om = sub["dnds"].replace([np.inf, -np.inf], np.nan).median()
        res[q] = {"n": int(len(sub)), "pct_ess": float((sub.essential==1).mean()),
                  "has_paralog": float(par), "metabolic": float(met),
                  "dark": float(drk), "median_omega": None if pd.isna(om) else float(om)}
        print(f"  {q:<12}{len(sub):>6}{100*(sub.essential==1).mean():>5.0f}%"
              f"{par:>12.2f}{met:>10.2f}{drk:>7.2f}"
              f"{('  nan' if pd.isna(om) else f'{om:>9.3f}')}")

    # the decisive, non-circular checks
    print("\n  --- non-circular predictions ---")
    par_buf = res.get("buffered",{}).get("has_paralog",float('nan'))
    par_cond = res.get("conditional",{}).get("has_paralog",float('nan'))
    om_core = res.get("core",{}).get("median_omega",float('nan'))
    om_acc = res.get("accessory",{}).get("median_omega",float('nan'))
    print(f"  [1] BUFFERED has more paralogs than CONDITIONAL (redundancy=why kept): "
          f"{par_buf:.2f} vs {par_cond:.2f}  -> {'PASS' if par_buf>par_cond else 'FAIL'}")
    drk_cond = res.get("conditional",{}).get("dark",float('nan'))
    drk_core = res.get("core",{}).get("dark",float('nan'))
    print(f"  [2] CONDITIONAL is darker than CORE (novel niche genes): "
          f"{drk_cond:.2f} vs {drk_core:.2f}  -> {'PASS' if drk_cond>drk_core else 'FAIL'}")
    if om_core==om_core and om_acc==om_acc:
        print(f"  [3] CORE under stronger purifying selection than ACCESSORY: "
              f"omega {om_core:.3f} vs {om_acc:.3f}  -> {'PASS' if om_core<om_acc else 'FAIL'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"twobytwo_{args.org}.json").write_text(json.dumps(
        {"org": args.org, "retained_min": RETAINED_MIN, "quadrants": res}, indent=2))
    print(f"\n  wrote twobytwo_{args.org}.json")
    return 0


def run_smoke():
    import pandas as pd, numpy as np
    print("=" * 70)
    print("2x2 SMOKE -- lens recovers planted enrichments")
    print("=" * 70)
    assert quadrant(True, 40) == "core"
    assert quadrant(True, 5) == "conditional"
    assert quadrant(False, 40) == "buffered"
    assert quadrant(False, 5) == "accessory"
    print("  quadrant assignment: OK")
    rng = np.random.RandomState(0)
    rows = []
    # plant: buffered genes have paralogs; conditional genes are dark, no paralog
    for _ in range(200): rows.append(dict(essential=1, n=40, npar=0, product="ribosomal protein"))     # core
    for _ in range(200): rows.append(dict(essential=1, n=5,  npar=0, product="hypothetical protein"))  # conditional (dark)
    for _ in range(200): rows.append(dict(essential=0, n=40, npar=2, product="MFS transporter"))       # buffered (paralog)
    for _ in range(200): rows.append(dict(essential=0, n=5,  npar=0, product="DUF protein"))           # accessory
    df = pd.DataFrame(rows)
    df["quad"] = [quadrant(bool(e), n) for e, n in zip(df.essential, df.n)]
    par_buf = (df[df.quad=="buffered"].npar>0).mean()
    par_cond = (df[df.quad=="conditional"].npar>0).mean()
    drk_cond = df[df.quad=="conditional"]["product"].str.contains(DARK).mean()
    drk_core = df[df.quad=="core"]["product"].str.contains(DARK).mean()
    print(f"  buffered paralog rate {par_buf:.2f} > conditional {par_cond:.2f}: "
          f"{'PASS' if par_buf>par_cond else 'FAIL'}")
    print(f"  conditional dark {drk_cond:.2f} > core dark {drk_core:.2f}: "
          f"{'PASS' if drk_cond>drk_core else 'FAIL'}")
    assert par_buf > par_cond and drk_cond > drk_core
    print("\n=== 2x2 SMOKE PASS ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
