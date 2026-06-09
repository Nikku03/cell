"""Evolutionary trajectory of essentiality -- LOSS side.

Same walk as evolution_trajectory.py (smallest-genome reference,
closest-relative chain), but reports what was LOST at each step:

  1. FAMILY LOSSES   = OGs present in PREV but absent in CURRENT
                       (gene family discarded from genome)
  2. LOST ESSENTIALS = family losses that WERE essential in PREV
                       (function abandoned entirely)
  3. ESSENTIALITY LOSSES = OG still present, was essential in PREV,
                            now non-essential in CURRENT
                       (function retained but no longer required --
                        a backup pathway emerged, environment changed,
                        or it became conditional)

For each step we report both 'vs previous step' (incremental) and
'vs reference' (cumulative -- what the most-derived lineage no
longer carries from the archaeal starting point).

Output: outputs/evolution_trajectory_losses.md
"""
from __future__ import annotations
import argparse, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT/"scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "afr", REPO_ROOT/"scripts"/"analyze_flanking_regions.py")
_afr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_afr)
parse_gff_cds = _afr.parse_gff_cds

STOP = set("a an and or of for the in to with by from on at as is are be "
           "domain-containing domain containing protein putative hypothetical "
           "uncharacterized predicted family member type protein-coding small "
           "large subunit conserved related signal binding sequence "
           "function functional product n-terminal c-terminal".split())


def function_words(products):
    c = Counter()
    for p in products:
        if not p: continue
        toks = re.split(r"[^a-z0-9]+", p.lower())
        c.update(t for t in toks if len(t) >= 3 and t not in STOP)
    return c


def main():
    pp = argparse.ArgumentParser()
    pp.add_argument("--steps", type=int, default=7)
    pp.add_argument("--cache-dir", type=Path,
                    default=REPO_ROOT/"data/drive_import/genome_cache")
    pp.add_argument("--out", type=Path,
                    default=REPO_ROOT/"outputs/evolution_trajectory_losses.md")
    args = pp.parse_args()

    import pandas as pd
    D = REPO_ROOT/"data/drive_import/labels"
    labels = pd.read_csv(D/"labels.csv")
    orth   = pd.read_csv(D/"orthology_features.csv")
    mf     = pd.read_csv(D/"genome_cache_manifest.csv")

    elig = mf[(mf.kind=="ours") & mf.accession.notna() &
              mf.join_rate.notna() & (mf.join_rate>=0.3)].copy()
    has_og = set(orth.organism.unique())
    elig = elig[elig.organism.isin(has_og)]
    accs = dict(zip(elig.organism, elig.accession))
    eligible = set(elig.organism)

    j = labels.merge(orth[["organism","locus_tag","og_id"]],
                     on=["organism","locus_tag"], how="inner")
    j = j[j.organism.isin(eligible) & j.og_id.notna()]
    og_set, ess_og_set, non_og_set = {}, {}, {}
    for org, g in j.groupby("organism"):
        og_set[org] = set(g.og_id.unique())
        ess_og_set[org] = set(g[g.essential==1].og_id.unique())
        non_og_set[org] = set(g[g.essential==0].og_id.unique())

    sizes = j.groupby("organism").size().sort_values()
    ref = sizes.index[0]
    print(f">>> REFERENCE = {ref}  ({sizes.iloc[0]} genes, "
          f"{len(og_set[ref])} OGs, {len(ess_og_set[ref])} ess OGs)\n")

    def jacc(a, b):
        u = len(og_set[a]|og_set[b])
        return len(og_set[a]&og_set[b])/u if u else 0

    walk = [ref]
    while len(walk) < args.steps + 1:
        cur = walk[-1]
        cands = sorted([(jacc(cur,o), o) for o in eligible if o not in walk],
                        reverse=True)
        if not cands: break
        walk.append(cands[0][1])
    print("walk:")
    for i,o in enumerate(walk):
        print(f"  {i}: {o}")

    # Cache GFF lookups per org so we can name lost genes by their
    # appearance in the *previous* organism's GFF
    gff_cache = {}
    def org_lookup(org):
        if org in gff_cache: return gff_cache[org]
        feats = parse_gff_cds(args.cache_dir/accs[org]/"genomic.gff")
        lt2feat = {}
        for f in feats:
            for k in (f["locus_tag"], f["old_locus_tag"]):
                if k: lt2feat[k] = f
        sub = j[j.organism == org]
        og_to_genes = defaultdict(list)
        for _, r in sub.iterrows():
            f = lt2feat.get(str(r.locus_tag))
            if f is None: continue
            og_to_genes[r.og_id].append({"locus_tag": str(r.locus_tag),
                                           "gene": f.get("gene",""),
                                           "product": f.get("product",""),
                                           "essential": int(r.essential)})
        gff_cache[org] = og_to_genes
        return og_to_genes

    md = []
    md.append(f"# Evolutionary trajectory — losses side\n\n")
    md.append(f"**Reference:** `{ref}` — {len(og_set[ref])} OGs, "
              f"{len(ess_og_set[ref])} essential\n\n")
    md.append(f"At each step beyond the reference we ask three questions:\n\n")
    md.append("1. **Family losses** — OGs present in the previous step that "
              "are absent in this organism (gene family discarded)\n")
    md.append("2. **Lost essentials** — those family losses that WERE essential "
              "in the previous organism (function abandoned)\n")
    md.append("3. **Essentiality losses** — OG still present, was essential "
              "before but is no longer (function retained, requirement gone)\n")

    for step in range(1, len(walk)):
        cur = walk[step]; prev = walk[step-1]
        # vs PREVIOUS
        fam_lost_prev = og_set[prev] - og_set[cur]
        lost_ess_prev = ess_og_set[prev] & fam_lost_prev   # was essential, family gone
        ess_lost_kept_family = (ess_og_set[prev] & og_set[cur]) - ess_og_set[cur]
        # ^ essential in prev, still here, but not essential now

        # vs REFERENCE (cumulative since archaeon)
        fam_lost_ref = og_set[ref] - og_set[cur]
        lost_ess_ref = ess_og_set[ref] & fam_lost_ref

        prev_lookup = org_lookup(prev)

        print(f"\n=== STEP {step}: {cur}  (closest to {prev}, J={jacc(prev,cur):.3f}) ===")
        print(f"  OGs in {prev}:        {len(og_set[prev]):>5d}")
        print(f"  OGs in {cur}:         {len(og_set[cur]):>5d}")
        print(f"  Family LOSSES vs prev:           {len(fam_lost_prev):>5d}  "
              f"(of which essential in prev: {len(lost_ess_prev)})")
        print(f"  Essentiality LOSSES (family kept): {len(ess_lost_kept_family):>5d}")
        print(f"  Family LOSSES vs reference:      {len(fam_lost_ref):>5d}  "
              f"(of which essential in reference: {len(lost_ess_ref)})")

        # function words on lost essentials (vs prev)
        prods_lost_ess_prev = []
        examples_lost_prev = []
        for og in lost_ess_prev:
            for r in prev_lookup.get(og, []):
                prods_lost_ess_prev.append(r["product"])
                if r["gene"] and "hypothetical" not in r["product"].lower():
                    examples_lost_prev.append((r["gene"], r["product"], r["locus_tag"]))
        # also for essentiality-loss (family kept, no longer essential)
        prods_ess_lost_kept = []
        examples_ess_lost_kept = []
        for og in ess_lost_kept_family:
            for r in prev_lookup.get(og, []):
                prods_ess_lost_kept.append(r["product"])
                if r["gene"] and "hypothetical" not in r["product"].lower():
                    examples_ess_lost_kept.append((r["gene"], r["product"], r["locus_tag"]))
        fw_lost = function_words(prods_lost_ess_prev).most_common(15)
        fw_kept = function_words(prods_ess_lost_kept).most_common(15)
        print(f"  TOP function words in lost-essentials (family-gone): ", end="")
        print(", ".join(f"{w}×{n}" for w,n in fw_lost[:10]))
        print(f"  TOP function words in essentiality-relaxed (family-here): ", end="")
        print(", ".join(f"{w}×{n}" for w,n in fw_kept[:10]))
        print(f"  EXAMPLES of lost-essential (named, family-gone):")
        for gene, product, lt in examples_lost_prev[:6]:
            print(f"    {gene:<8s}  {product[:65]:<65s} (was {lt} in {prev})")
        print(f"  EXAMPLES of essentiality-relaxed (still here, not needed):")
        for gene, product, lt in examples_ess_lost_kept[:6]:
            print(f"    {gene:<8s}  {product[:65]:<65s} (was essential in {prev})")

        # markdown
        md.append(f"\n---\n\n## Step {step}: `{cur}` ← `{prev}`  (J={jacc(prev,cur):.3f})\n\n")
        md.append(f"**vs previous step:**\n")
        md.append(f"- Family losses (OG discarded): **{len(fam_lost_prev)}**, "
                  f"of which **{len(lost_ess_prev)}** were essential in `{prev}`\n")
        md.append(f"- Essentiality losses (OG kept, no longer essential): "
                  f"**{len(ess_lost_kept_family)}**\n\n")
        md.append(f"**vs reference (`{ref}`):**\n")
        md.append(f"- Family losses since reference: **{len(fam_lost_ref)}**, "
                  f"of which **{len(lost_ess_ref)}** were essential in `{ref}`\n\n")
        md.append(f"### Functions abandoned (family GONE, was essential in `{prev}`)\n\n")
        md.append("| word | count |\n|---|---|\n")
        for w,n in fw_lost[:15]: md.append(f"| `{w}` | {n} |\n")
        md.append(f"\n#### Named examples\n\n")
        md.append("| gene | was-essential product | locus_tag in prev |\n|---|---|---|\n")
        for gene, product, lt in examples_lost_prev[:10]:
            md.append(f"| {gene} | {product[:80]} | {lt} |\n")
        md.append(f"\n### Functions retained but no longer required "
                  f"(family kept, essential→non-essential)\n\n")
        md.append("| word | count |\n|---|---|\n")
        for w,n in fw_kept[:15]: md.append(f"| `{w}` | {n} |\n")
        md.append(f"\n#### Named examples\n\n")
        md.append("| gene | product | locus_tag in prev |\n|---|---|---|\n")
        for gene, product, lt in examples_ess_lost_kept[:10]:
            md.append(f"| {gene} | {product[:80]} | {lt} |\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(md))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
