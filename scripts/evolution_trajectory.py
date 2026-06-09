"""Evolutionary trajectory of essentiality.

Pick the organism with the fewest labeled genes as the reference point
(=most minimal genome we have). Then walk outward: at each step, the
'closest relative' = the not-yet-visited organism with the highest OG
overlap with the *current* organism.

At every step report:
  * how many gene families (OGs) are in the new organism but NOT in the
    reference (= 'novel gene families gained since the reference')
  * of those novel families, how many are ESSENTIAL in the new organism
    (= novel essential gene families)
  * what FUNCTIONS those novel essentials encode (product word counts
    -> the biological category of the gain)
  * examples: a handful of named novel-essential genes with their
    products and immediate neighbours (operon context = often clue to
    why the gain became essential)
  * a side-by-side comparison with the previous step organism too,
    so we can see WHICH gains stuck around vs got lost again

The output is a readable markdown narrative -- the goal is biology,
not metrics.
"""
from __future__ import annotations
import argparse, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_flanking_regions import parse_gff_cds


def norm_product(p):
    p = p.lower().strip()
    p = re.sub(r"\b(putative|probable|possible|predicted|conserved)\b", "", p)
    return re.sub(r"\s+", " ", p).strip()


# words too generic to be informative
STOP = set("a an and or of for the in to with by from on at as is are be "
           "domain-containing domain containing protein putative hypothetical "
           "uncharacterized predicted family member type protein-coding small "
           "large subunit conserved related signal binding sequence "
           "function functional product n-terminal c-terminal".split())


def function_words(products: list[str]) -> Counter:
    c = Counter()
    for p in products:
        if not p: continue
        toks = re.split(r"[^a-z0-9]+", p.lower())
        c.update(t for t in toks if len(t) >= 3 and t not in STOP)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT/"data/drive_import/genome_cache")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT/"outputs/evolution_trajectory.md")
    args = p.parse_args()

    import pandas as pd
    D = REPO_ROOT/"data/drive_import/labels"
    labels = pd.read_csv(D/"labels.csv")
    orth = pd.read_csv(D/"orthology_features.csv")
    mf = pd.read_csv(D/"genome_cache_manifest.csv")

    # eligible = our orgs with a genome cached and a clean labels join
    elig = mf[(mf.kind=="ours") & mf.accession.notna() &
              mf.join_rate.notna() & (mf.join_rate>=0.3)].copy()
    # restrict to organisms that ALSO have OG assignments (BERIL only)
    has_og = set(orth.organism.unique())
    elig = elig[elig.organism.isin(has_og)]
    eligible_orgs = set(elig.organism)
    accs = dict(zip(elig.organism, elig.accession))
    print(f"eligible organisms ({len(eligible_orgs)}):")
    for o in sorted(eligible_orgs): print(f"  {o}  ({accs[o]})")

    # join labels x orth
    j = labels.merge(orth[["organism","locus_tag","og_id"]],
                     on=["organism","locus_tag"], how="inner")
    j = j[j.organism.isin(eligible_orgs) & j.og_id.notna()]

    # per-organism OG sets and essential-OG sets
    og_set, ess_og_set = {}, {}
    for org, g in j.groupby("organism"):
        og_set[org] = set(g.og_id.unique())
        ess_og_set[org] = set(g[g.essential==1].og_id.unique())

    # reference = smallest gene count among eligibles
    sizes = j.groupby("organism").size().sort_values()
    print(f"\n=== gene-count ranking ===")
    for o, n in sizes.items(): print(f"  {o:<28s} {n:>5d} labeled genes  "
                                      f"({len(og_set[o]):>5d} OGs)")
    ref = sizes.index[0]
    print(f"\n>>> REFERENCE = {ref}  ({sizes.iloc[0]} genes) <<<")

    # similarity = Jaccard of OG sets
    def jacc(a, b):
        u = len(og_set[a] | og_set[b]); return len(og_set[a] & og_set[b])/u if u else 0

    print(f"\n=== Jaccard OG similarity to {ref} ===")
    for o in sorted(eligible_orgs):
        if o == ref: continue
        print(f"  {o:<28s} {jacc(ref, o):.3f}")

    # walk outward: at each step pick the closest unvisited org to the CURRENT step's org
    walk = [ref]
    while len(walk) < args.steps + 1:
        cur = walk[-1]
        candidates = [(jacc(cur, o), o) for o in eligible_orgs if o not in walk]
        if not candidates: break
        candidates.sort(reverse=True)
        walk.append(candidates[0][1])
    print(f"\n=== walk ===")
    for i, o in enumerate(walk):
        print(f"  step {i}: {o}")

    # for each step beyond reference, characterize gains
    # GFF/product lookup per org (lazy)
    gff_cache = {}
    def org_products_by_og(org):
        """Map og_id -> list of (locus_tag, gene, product) in this organism."""
        if org in gff_cache: return gff_cache[org]
        feats = parse_gff_cds(args.cache_dir / accs[org] / "genomic.gff")
        lt2feat = {}
        for f in feats:
            for k in (f["locus_tag"], f["old_locus_tag"]):
                if k: lt2feat[k] = f
        # join to (organism, locus_tag) -> og_id
        sub = j[j.organism == org]
        out = defaultdict(list)
        for _, r in sub.iterrows():
            f = lt2feat.get(str(r.locus_tag))
            if f is None: continue
            out[r.og_id].append((str(r.locus_tag), f.get("gene",""),
                                  f.get("product",""), int(r.essential),
                                  f.get("start",0), f.get("seqid","")))
        gff_cache[org] = out
        return out

    md = []
    md.append(f"# Evolutionary trajectory of essentiality\n\n")
    md.append(f"**Reference point:** `{ref}` — smallest cached genome "
              f"({sizes.iloc[0]} labeled genes, {len(og_set[ref])} OGs, "
              f"{len(ess_og_set[ref])} essential OGs)\n\n")
    md.append(f"**Walk:** closest-OG-relative chain (Jaccard).\n\n")

    # universal essentials = essential in reference (a baseline)
    print(f"\n=== REFERENCE: {ref} essential profile ===")
    ref_essentials = ess_og_set[ref]
    print(f"  essential OGs: {len(ref_essentials)}")
    ref_products = org_products_by_og(ref)
    ref_ess_prods = [t[2] for og in ref_essentials for t in ref_products[og]]
    fw = function_words(ref_ess_prods).most_common(15)
    print(f"  top function words in reference essentials: "
          f"{', '.join(w for w,_ in fw)}")
    md.append(f"## Reference: `{ref}`\n\n")
    md.append(f"Essential gene families: **{len(ref_essentials)}**\n\n")
    md.append(f"Top function words in essentials: "
              f"`{', '.join(w + f'×{n}' for w,n in fw)}`\n\n")

    # iterate
    cumulative_visited = set()
    cumulative_visited.add(ref)
    for step in range(1, len(walk)):
        cur = walk[step]; prev = walk[step-1]
        # GAINS = OGs in current not in reference (the "biological gain" set)
        gains_vs_ref = og_set[cur] - og_set[ref]
        # NOVEL essentials (essential in cur, not present in ref)
        novel_ess_vs_ref = ess_og_set[cur] & gains_vs_ref
        # gains versus previous step (incremental)
        gains_vs_prev = og_set[cur] - og_set[prev]
        novel_ess_vs_prev = ess_og_set[cur] & gains_vs_prev

        cur_prods = org_products_by_og(cur)
        novel_ess_products = [t[2] for og in novel_ess_vs_ref for t in cur_prods.get(og,[])]
        fw_novel = function_words(novel_ess_products).most_common(20)

        # "what changed" -- count of various categories
        hyp_count = sum(1 for p in novel_ess_products
                        if "hypothetical" in p.lower() or "uncharacterized" in p.lower())
        characterized = len(novel_ess_products) - hyp_count

        # examples: pick a handful of named novel-essential genes
        examples = []
        for og in list(novel_ess_vs_ref)[:200]:
            for lt, gene, product, ess, start, seqid in cur_prods.get(og, []):
                if ess and gene and "hypothetical" not in product.lower():
                    examples.append((og, lt, gene, product, start, seqid))
        examples = examples[:8]

        # check whether each novel-essential is ALSO in the previous walk org
        # (= gained-but-lost-already vs gained-and-kept)
        shared_with_prev = len(novel_ess_vs_ref & og_set[prev])
        unique_to_cur    = len(novel_ess_vs_ref - og_set[prev])

        print(f"\n=== STEP {step}: {cur}  (closest to {prev}, jacc={jacc(prev,cur):.3f}) ===")
        print(f"  total OGs in {cur}:  {len(og_set[cur]):>5d}")
        print(f"  OGs ALSO in {ref}:    {len(og_set[cur] & og_set[ref]):>5d}")
        print(f"  GAINS vs reference: {len(gains_vs_ref):>5d}  "
              f"({len(novel_ess_vs_ref)} essential)")
        print(f"  GAINS vs {prev}:    {len(gains_vs_prev):>5d}  "
              f"({len(novel_ess_vs_prev)} essential)")
        print(f"  of the novel essentials: characterized={characterized}, "
              f"hypothetical={hyp_count}")
        print(f"  top novel-essential function words:")
        for w, n in fw_novel[:12]:
            print(f"    {w:<24s} {n}x")
        print(f"  examples (named novel essentials):")
        for og, lt, gene, product, start, seqid in examples:
            print(f"    {gene:<10s} {product[:70]:<70s} ({lt})")

        # ---- markdown ----
        md.append(f"\n---\n\n## Step {step}: `{cur}`\n\n")
        md.append(f"Closest relative of `{prev}` (Jaccard {jacc(prev,cur):.3f}).  \n")
        md.append(f"Total OGs in this genome: **{len(og_set[cur])}**  \n")
        md.append(f"Shared with reference `{ref}`: **{len(og_set[cur] & og_set[ref])}**  \n")
        md.append(f"### Gains vs reference\n\n")
        md.append(f"- Novel OGs (not in {ref}): **{len(gains_vs_ref)}**  \n")
        md.append(f"- Of those, **{len(novel_ess_vs_ref)}** are essential in {cur}  \n")
        md.append(f"- Characterized: {characterized}  &nbsp; hypothetical: {hyp_count}\n\n")
        md.append(f"### Gains *incrementally* vs previous step (`{prev}`)\n\n")
        md.append(f"- Novel OGs: **{len(gains_vs_prev)}**  &nbsp; "
                  f"novel essentials: **{len(novel_ess_vs_prev)}**  \n")
        md.append(f"- Of the novel essentials vs reference, **{shared_with_prev}** were "
                  f"already in `{prev}` (continued lineage), "
                  f"**{unique_to_cur}** are uniquely added at this step\n\n")
        md.append(f"### What biological functions are the gains?\n\n")
        md.append("Top function words in novel essentials:\n\n")
        md.append("| word | count |\n|---|---|\n")
        for w, n in fw_novel[:15]: md.append(f"| `{w}` | {n} |\n")
        md.append(f"\n### Example named novel essentials\n\n")
        md.append("| gene | product | locus_tag |\n|---|---|---|\n")
        for og, lt, gene, product, start, seqid in examples:
            md.append(f"| {gene} | {product[:90]} | {lt} |\n")

    # final cumulative view
    all_essentials = set()
    for o in walk: all_essentials |= ess_og_set[o]
    universal = ref_essentials.copy()
    for o in walk[1:]:
        universal &= ess_og_set[o]
    md.append(f"\n---\n\n## Summary across walk\n\n")
    md.append(f"- Total distinct essential gene families seen anywhere on the "
              f"walk: **{len(all_essentials)}**\n")
    md.append(f"- Essential in EVERY step (universal core, including reference): "
              f"**{len(universal)}**\n")
    md.append(f"- Step-by-step novel-essentials growth: "
              f"{', '.join(str(len(ess_og_set[walk[i]] & (og_set[walk[i]] - og_set[ref]))) for i in range(1,len(walk)))}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(md))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
