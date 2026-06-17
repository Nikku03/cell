"""PATH-ORPHAN, NOVELTY #2 -- the integration layer (per-gene fused atlas).

The white space the prior-art survey found: every channel exists in a silo
(Price=phenotype, Barrio-Hernandez=structure, gapseq=metabolism, Humphreys=PPI),
but no one fuses them into ONE per-gene call with confidence tiers + an honest
unknown bucket + the leave-one-clade-out discipline. This assembles whatever
channels are present into that single record.

Per gene it emits:
  function_tier   : annotated | fold_named | phenotype_module | live_ghost | unknown
  conditional     : the 2x2 quadrant (core/conditional/buffered/accessory)
  live_ghost      : dark BUT functional (essential | dnds<0.3 | conserved>=10 orgs)
  priority        : rank for experimental characterization (the hit-list)
  evidence/confidence per call

Channels consumed if present: bridge (conservation+product), dnds, cofit,
foldhit (structure), esm. Degrades gracefully -- runs today on what exists,
strengthens as foldseek/gem land.

KILL-GATE (honest, runnable now): the live-ghost flag must enrich for genuine
functional reality -- flagged dark genes should have broader orthology and
stronger purifying selection than unflagged dark genes. If not, the flag is noise.

--smoke : validate tiering + priority + the enrichment gate on synthetic data
--real  : build the atlas for outputs/orphan/bridge_<org>.parquet (+ channels)
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
DARK = re.compile(r'hypothetical|uncharacteri|DUF\d|domain of unknown|putative protein', re.I)
RETAINED_MIN = 24


def function_tier(product, fold_named, cofit_cons, is_live_ghost):
    """Assign the most informative tier available for this gene."""
    dark = bool(DARK.search(str(product or "")))
    if not dark:
        return "annotated"
    if fold_named == 1:
        return "fold_named"          # structure rescued the orphan
    if cofit_cons == cofit_cons and cofit_cons >= 0.5:
        return "phenotype_module"    # couples to a conserved-essential module
    if is_live_ghost:
        return "live_ghost"          # functional but unexplained -> characterize
    return "unknown"


def build(df):
    import pandas as pd, numpy as np
    n_orgs = pd.to_numeric(df.get("family_n_organisms"), errors="coerce").fillna(0)
    dnds = pd.to_numeric(df.get("dnds"), errors="coerce")
    ess = df["essential"].astype(int)
    cons = pd.to_numeric(df["conservation"], errors="coerce")
    fold_named = pd.to_numeric(df.get("fold_hit_named"), errors="coerce").fillna(0) \
        if "fold_hit_named" in df else pd.Series(0, index=df.index)
    cofit = pd.to_numeric(df.get("cofit_cons"), errors="coerce") \
        if "cofit_cons" in df else pd.Series(np.nan, index=df.index)
    dark = df["product"].fillna("").str.contains(DARK)
    live_ghost = dark & ((ess == 1) | (dnds < 0.3) | (n_orgs >= 10))

    # 2x2 conditional quadrant
    retained = n_orgs >= RETAINED_MIN
    quad = np.where(ess == 1, np.where(retained, "core", "conditional"),
                    np.where(retained, "buffered", "accessory"))
    tier = [function_tier(p, fn, cc, lg) for p, fn, cc, lg in
            zip(df["product"], fold_named, cofit, live_ghost)]
    # priority for the hit-list: live ghosts, ranked by functional-reality signal
    # (essential > selected > conserved), strongest first
    pr = (live_ghost.astype(int) * 4
          + (ess == 1).astype(int) * 3
          + (dnds < 0.3).fillna(False).astype(int) * 2
          + (n_orgs >= 10).astype(int))
    out = pd.DataFrame({
        "locus_tag": df["locus_tag"], "product": df["product"],
        "function_tier": tier, "conditional": quad,
        "essential": ess, "conservation": cons.round(3),
        "dnds": dnds.round(3), "n_orgs": n_orgs.astype(int),
        "live_ghost": live_ghost.astype(int), "priority": pr.astype(int),
    })
    return out


def run_real(args):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    df = pd.read_parquet(bridge)
    for ch in ["dnds", "cofit", "foldhit", "esm"]:
        p = OUT / f"{ch}_{args.org}.parquet"
        if p.exists():
            fd = pd.read_parquet(p)
            df = df.merge(fd, on="locus_tag", how="left", suffixes=("", f"_{ch}"))
    present = [c for c in ["dnds", "cofit_cons", "fold_hit_named", "esm_0"]
              if c in df.columns]
    atlas = build(df)

    print("=" * 70)
    print(f"FUSED ATLAS -- {args.org}   (channels present: {present or ['conservation only']})")
    print("=" * 70)
    print("\n  function tiers:")
    for t, n in atlas.function_tier.value_counts().items():
        print(f"    {t:<18}{n:>6}  ({100*n/len(atlas):.0f}%)")
    print("\n  conditional quadrants:")
    for q, n in atlas.conditional.value_counts().items():
        print(f"    {q:<18}{n:>6}  ({100*n/len(atlas):.0f}%)")

    # KILL-GATE: live-ghost flag must capture functional reality
    dark = atlas["product"].fillna("").str.contains(DARK)
    lg = atlas[dark & (atlas.live_ghost == 1)]
    nlg = atlas[dark & (atlas.live_ghost == 0)]
    print("\n  --- KILL-GATE: does the live-ghost flag capture real genes? ---")
    if len(lg) and len(nlg):
        o_lg, o_nlg = lg.n_orgs.mean(), nlg.n_orgs.mean()
        d_lg = lg.dnds.replace([np.inf,-np.inf],np.nan).median()
        d_nlg = nlg.dnds.replace([np.inf,-np.inf],np.nan).median()
        print(f"    ortholog breadth: live-ghost {o_lg:.1f} vs unflagged {o_nlg:.1f}"
              f"  -> {'PASS' if o_lg>o_nlg else 'FAIL'}")
        if d_lg==d_lg and d_nlg==d_nlg:
            print(f"    median omega:     live-ghost {d_lg:.3f} vs unflagged {d_nlg:.3f}"
                  f"  -> {'PASS' if d_lg<d_nlg else 'FAIL'} (stronger purifying selection)")

    # the deliverable hit-list: top-priority live ghosts to characterize
    hit = atlas[atlas.live_ghost == 1].sort_values(
        ["priority", "n_orgs"], ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    atlas.to_parquet(OUT / f"atlas_{args.org}.parquet", index=False)
    hit.to_csv(OUT / f"live_ghost_hitlist_{args.org}.csv", index=False)
    print(f"\n  LIVE-GHOST HIT-LIST: {len(hit)} genes (-> live_ghost_hitlist_*.csv)")
    print("  top 10 to characterize first:")
    for r in hit.head(10).itertuples():
        print(f"    {r.locus_tag:<13} pri={r.priority} cons={r.conservation:.2f} "
              f"orgs={r.n_orgs:<3}| {str(r.product)[:42]}")
    (OUT / f"atlas_{args.org}_summary.json").write_text(json.dumps({
        "org": args.org, "n_genes": int(len(atlas)),
        "channels_present": present,
        "function_tiers": atlas.function_tier.value_counts().to_dict(),
        "conditional_quadrants": atlas.conditional.value_counts().to_dict(),
        "n_live_ghosts": int((atlas.live_ghost == 1).sum())}, indent=2))
    print(f"\n  wrote atlas_{args.org}.parquet + hitlist + summary")
    return 0


def run_smoke():
    import pandas as pd, numpy as np
    print("=" * 70)
    print("ATLAS SMOKE -- tiering, priority, live-ghost enrichment gate")
    print("=" * 70)
    assert function_tier("ATP synthase", 0, np.nan, False) == "annotated"
    assert function_tier("hypothetical protein", 1, np.nan, True) == "fold_named"
    assert function_tier("DUF999 protein", 0, 0.8, True) == "phenotype_module"
    assert function_tier("hypothetical protein", 0, 0.1, True) == "live_ghost"
    assert function_tier("hypothetical protein", 0, np.nan, False) == "unknown"
    print("  function-tier assignment: OK")
    # synthetic frame: live ghosts have broad orthology + low omega (planted)
    rng = np.random.RandomState(0)
    rows = []
    for _ in range(100):  # real live ghosts: dark, essential, broad, low omega
        rows.append(dict(locus_tag=f"g{len(rows)}", product="hypothetical protein",
                         essential=1, conservation=0.05, dnds=0.1,
                         family_n_organisms=20, n_paralogs_in_genome=0))
    for _ in range(100):  # spurious dark: dark, non-ess, narrow, neutral omega
        rows.append(dict(locus_tag=f"g{len(rows)}", product="hypothetical protein",
                         essential=0, conservation=0.02, dnds=0.9,
                         family_n_organisms=2, n_paralogs_in_genome=0))
    df = pd.DataFrame(rows)
    atlas = build(df)
    lg = atlas[atlas.live_ghost == 1]; nlg = atlas[atlas.live_ghost == 0]
    print(f"  live-ghosts {len(lg)}, unflagged {len(nlg)}")
    print(f"  ortholog breadth {lg.n_orgs.mean():.1f} vs {nlg.n_orgs.mean():.1f}")
    assert lg.n_orgs.mean() > nlg.n_orgs.mean(), "live-ghost flag must enrich real genes"
    assert (atlas[atlas.essential==1].conditional.isin(["core","conditional"])).all()
    print("\n=== ATLAS SMOKE PASS ===")
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
