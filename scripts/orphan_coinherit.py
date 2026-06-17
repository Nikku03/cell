"""PATH-ORPHAN, missing channel (cheap half) -- co-inheritance / functional link.

The cheap, scale-appropriate version of the PPI channel. True residue coevolution
needs hundreds of sequences per pair; we have <=59 organisms, so instead we use
GENE-level co-inheritance: two genes whose orthologous groups are present/absent
TOGETHER across the organism panel are functionally linked (classic phylogenetic
profiling -- the signal STRING is built on). It doubles as:
  (a) a 6th de-orphaning channel: a ghost co-inherited with a known module
      inherits that module's process;
  (b) the cheap NOMINATOR of candidate interacting pairs for later AF-Multimer
      confirm (gated, expensive, deferred).

"IS IT EVEN USEFUL" kill-gate (runs real in sandbox): genomically ADJACENT genes
(operon members, from GFF gene order) should have HIGHER co-inheritance than
random pairs. If adjacent ~ random, the signal is too weak at this panel size ->
drop the channel. If clearly higher -> it carries real functional signal.

--smoke : validate phi co-inheritance + adjacency-enrichment logic on synthetic
--real  : build OG presence matrix, score GMI1000 pairs, run the kill-gate
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "drive_import"
OUT = REPO / "outputs" / "orphan"
DARK = re.compile(r'hypothetical|uncharacteri|DUF\d|domain of unknown|putative protein', re.I)


def phi(a, b):
    """phi coefficient (binary correlation) of two presence vectors. The
    standard co-inheritance score; >0 = co-occur, <0 = anti-occur."""
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    sa, sb = a.sum(), b.sum()
    both = (a * b).sum()
    denom = (sa * sb * (n - sa) * (n - sb)) ** 0.5
    if denom == 0:
        return 0.0
    return (n * both - sa * sb) / denom


def adjacency_enrichment(sim_adjacent, sim_random):
    """mean co-inheritance of adjacent vs random pairs + fold."""
    import numpy as np
    ma = float(np.mean(sim_adjacent)) if len(sim_adjacent) else float("nan")
    mr = float(np.mean(sim_random)) if len(sim_random) else float("nan")
    fold = ma / mr if mr else float("nan")
    return ma, mr, fold


def _gff_order(acc):
    """GMI1000 gene order by (contig,start) -> list of locus_tags (operon proxy)."""
    gff = DATA / "genome_cache" / acc / "genomic.gff"
    feats = []
    with open(gff) as f:
        for line in f:
            if "\tCDS\t" not in line:
                continue
            c = line.split("\t")
            lt = re.search(r'locus_tag=([^;\n]+)', c[8])
            if lt:
                feats.append((c[0], int(c[3]), lt.group(1)))
    feats.sort()
    seen = set(); ordered = []
    for ctg, start, lt in feats:
        if lt not in seen:
            seen.add(lt); ordered.append((ctg, lt))
    return ordered


def run_real(args):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    # 1) OG -> set of orgs present (the pangenome presence matrix, panel-scale)
    og_orgs = {}
    all_orgs = set()
    with open(DATA / "labels" / "orthology_features.csv") as f:
        for r in csv.DictReader(f):
            og_orgs.setdefault(r["og_id"], set()).add(r["organism"])
            all_orgs.add(r["organism"])
    orgs = sorted(all_orgs); oidx = {o: i for i, o in enumerate(orgs)}
    print("=" * 70)
    print(f"CO-INHERITANCE channel -- {args.org}  (panel: {len(orgs)} organisms)")
    print("=" * 70)

    df = pd.read_parquet(bridge)
    # presence vector per GMI1000 gene (via its OG)
    def vec(og):
        v = np.zeros(len(orgs))
        for o in og_orgs.get(og, ()):
            v[oidx[o]] = 1.0
        return v
    df = df[df.og_id.isin(og_orgs)].copy()
    prof = {lt: vec(og) for lt, og in zip(df.locus_tag, df.og_id)}

    # 2) kill-gate: adjacent (operon) vs random pair co-inheritance
    from collections import OrderedDict
    acc = None
    with open(DATA / "labels" / "genome_cache_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["organism"] == args.org and r["accession"]:
                acc = r["accession"]
    ordered = [lt for ctg, lt in _gff_order(acc) if lt in prof]
    adj_pairs = [(ordered[i], ordered[i+1]) for i in range(len(ordered)-1)]
    rng = np.random.RandomState(0)
    lts = list(prof)
    rand_pairs = [(lts[rng.randint(len(lts))], lts[rng.randint(len(lts))])
                  for _ in range(len(adj_pairs))]
    sim_adj = [phi(prof[a], prof[b]) for a, b in adj_pairs]
    sim_rnd = [phi(prof[a], prof[b]) for a, b in rand_pairs]
    ma, mr, fold = adjacency_enrichment(sim_adj, sim_rnd)
    # fraction of adjacent pairs with strong co-inheritance
    strong_adj = float(np.mean([s > 0.5 for s in sim_adj]))
    strong_rnd = float(np.mean([s > 0.5 for s in sim_rnd]))
    print(f"\n  adjacent (operon) pairs : {len(adj_pairs):,}")
    print(f"  mean co-inheritance  adjacent {ma:.3f}  vs random {mr:.3f}  "
          f"-> {fold:.1f}x")
    print(f"  strong (phi>0.5) rate adjacent {strong_adj:.2f}  vs random "
          f"{strong_rnd:.2f}  -> {(strong_adj/strong_rnd if strong_rnd else float('inf')):.1f}x")
    useful = fold >= 2.0 or (strong_rnd > 0 and strong_adj / strong_rnd >= 2.0)
    print(f"\n  VERDICT: {'USEFUL -- carries real functional signal at panel scale' if useful else 'WEAK -- too little signal at this panel size; defer'}")

    # 3) de-orphaning use: top co-inherited partner per live ghost
    dn = OUT / f"dnds_{args.org}.parquet"
    dnds = (pd.read_parquet(dn).set_index("locus_tag")["dnds"].to_dict()
            if dn.exists() else {})
    # Emit confound-guarded hints for ALL genes (atlas filters to ghosts for tier
    # assignment; validator uses the broader file to measure recovery on knowns).
    ghosts = df
    # product lookup for naming the partner's function
    prod = dict(zip(df.locus_tag, df["product"]))
    P = np.array([prof[l] for l in df.locus_tag])           # genes x orgs
    Pn = P - P.mean(1, keepdims=True)
    norm = np.linalg.norm(Pn, axis=1, keepdims=True); norm[norm == 0] = 1
    Pn = Pn / norm
    idx = {l: i for i, l in enumerate(df.locus_tag)}
    nominated = 0; rows = []
    present_count = {l: int(prof[l].sum()) for l in df.locus_tag}
    partner_use = {}
    for g in ghosts.locus_tag:
        # CONFOUND GUARD: phi over a narrow distribution is trivially 1.0 (shared
        # lineage, not function). Require the ghost present in >= min_orgs so the
        # co-inheritance is statistically meaningful, not a 4-organism accident.
        if present_count[g] < args.min_orgs:
            continue
        i = idx[g]
        sims = Pn @ Pn[i]
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        partner = df.locus_tag.iloc[j]; ppr = str(prod.get(partner, ""))
        if present_count[partner] < args.min_orgs:
            continue
        if sims[j] > 0.5 and not DARK.search(ppr):
            nominated += 1
            partner_use[partner] = partner_use.get(partner, 0) + 1
            rows.append({"ghost": g, "ghost_product": str(prod.get(g, "")),
                         "ghost_orgs": present_count[g],
                         "top_partner": partner, "partner_product": ppr,
                         "phi": round(float(sims[j]), 3)})
    # drop degenerate partners (one partner for many ghosts = residual confound)
    if rows:
        deg = {p for p, c in partner_use.items() if c > args.max_partner_reuse}
        rows = [r for r in rows if r["top_partner"] not in deg]
        nominated = len(rows)
    print(f"\n  de-orphaning (CONFOUND-GUARDED, ghost present in >={args.min_orgs} "
          f"orgs, partner reuse <={args.max_partner_reuse}):")
    print(f"  {nominated} live ghosts co-inherited (phi>0.5) with a CHARACTERIZED "
          f"partner -> trustworthy function hint")
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).sort_values("phi", ascending=False).to_csv(
            OUT / f"coinherit_hints_{args.org}.csv", index=False)
        for r in sorted(rows, key=lambda x: -x["phi"])[:8]:
            print(f"    {r['ghost']:<13} phi={r['phi']} <-> {r['top_partner']} "
                  f"| {r['partner_product'][:40]}")
    (OUT / f"coinherit_{args.org}.json").write_text(json.dumps({
        "org": args.org, "panel_orgs": len(orgs),
        "adjacent_mean": ma, "random_mean": mr, "fold": fold,
        "useful": bool(useful), "ghosts_with_named_partner": nominated}, indent=2))
    print(f"\n  wrote coinherit_{args.org}.json"
          + (f" + coinherit_hints_{args.org}.csv" if rows else ""))
    return 0


def run_smoke():
    import numpy as np
    print("=" * 70)
    print("CO-INHERITANCE SMOKE -- phi score + adjacency-enrichment logic")
    print("=" * 70)
    # perfectly co-occurring vectors -> phi = 1; anti -> -1; independent -> ~0
    a = [1, 1, 1, 0, 0, 0]; b = [1, 1, 1, 0, 0, 0]
    c = [0, 0, 0, 1, 1, 1]
    print(f"  phi(co-occur)={phi(a,b):.2f}  phi(anti)={phi(a,c):.2f}")
    assert abs(phi(a, b) - 1.0) < 1e-9 and abs(phi(a, c) + 1.0) < 1e-9
    rng = np.random.RandomState(0)
    ind = rng.randint(0, 2, 200)
    assert abs(phi(rng.randint(0,2,200), ind)) < 0.3
    print("  phi coefficient: OK")
    # adjacency enrichment: adjacent pairs co-inherited, random not
    sim_adj = [0.8, 0.7, 0.9, 0.6]; sim_rnd = [0.05, -0.1, 0.0, 0.1]
    ma, mr, fold = adjacency_enrichment(sim_adj, sim_rnd)
    print(f"  adjacent {ma:.2f} vs random {mr:.2f} -> fold {fold:.1f}")
    assert ma > mr and (fold > 2 or mr <= 0)
    print("\n=== CO-INHERITANCE SMOKE PASS ===")
    print("  phi co-inheritance + operon-adjacency kill-gate validated.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--min_orgs", type=int, default=10)
    ap.add_argument("--max_partner_reuse", type=int, default=5)
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
