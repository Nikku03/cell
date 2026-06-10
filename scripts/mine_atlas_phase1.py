"""Phase 1: mine the conditional vulnerability atlas for one validated surprise.

Reads the atlas built earlier (conditional_vulnerability_atlas.csv on Drive),
applies the discipline from PHASE1_BRIEF.md:

  1. Cleanup: drop motility (Agar=measurement-artifact), drop blank conditions,
     set ecology aside.
  2. Stratify into therapeutic buckets:
       A. Antibiotic-potentiation (highest-value — adjuvant targets)
       B. Nutrient-niche (host-environment relevance)
       C. Metal/oxidative stress (less drug-relevant, more biology)
  3. Cross-organism consistency: same (gene, condition) lethal in N orgs
     is reproducibility evidence; one-offs are not.
  4. Rank by consistency × magnitude. Top 100 per bucket -> CSVs the
     literature-hunt step will operate on.

Output: outputs/atlas_phase1_<bucket>.csv (small, repo-friendly).
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ATLAS = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/conditional_atlas/conditional_vulnerability_atlas.csv")

# Clinically-used antibiotics (substring match, lowercase) — this is the
# surprise-hunt pool: a gene knockout that sensitizes the bug to one of
# these is a drug-adjuvant candidate.
ANTIBIOTICS = [
    "polymyxin","bacitracin","fusidic","sisomicin","cycloserine","cisplatin",
    "vancomycin","tobramycin","gentamicin","amikacin","kanamycin","neomycin",
    "tetracycline","doxycycline","minocycline","tigecycline","ampicillin",
    "carbenicillin","penicillin","cefoxitin","cefotaxime","ceftriaxone",
    "ciprofloxacin","norfloxacin","levofloxacin","nalidixic","rifampicin",
    "rifampin","chloramphenicol","erythromycin","azithromycin","clindamycin",
    "trimethoprim","sulfamethoxazole","spectinomycin","streptomycin",
    "novobiocin","benzalkonium","benzethonium","fosfomycin","mupirocin",
    "colistin","daptomycin","linezolid","tigecycline","cerulenin","triclosan",
]
# stress conditions that are *biological*, not therapeutic (we keep but
# separate them out from the antibiotic bucket)
STRESS_KW = ["copper","cobalt","nickel","zinc","cadmium","iron","manganese",
             "peroxide","paraquat","menadione","thallium","arsenate","arsenite",
             "chromate","selenite","silver","mercury","cisplatin"]


def norm_cond(c: str) -> str:
    s = str(c or "").lower().strip()
    # strip common salt / hydrate suffixes so "Cobalt chloride hexahydrate" ==
    # "Cobalt chloride" for cross-org matching
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


def main(atlas_path: Path):
    atlas = pd.read_csv(atlas_path)
    print(f"loaded {len(atlas):,} entries from {atlas_path}")

    # === CLEANUP ===
    motility = atlas.expGroup.fillna("") == "motility"
    print(f"  dropping motility: {motility.sum()}")
    atlas = atlas[~motility].copy()

    blank = (atlas.condition_1.fillna("").str.strip() == "")
    uninform_grp = atlas.expGroup.fillna("").isin(
        ["","lb","control","mixed community","magnetic pulldown"])
    drop = blank & uninform_grp
    print(f"  dropping blank/uninformative: {drop.sum()}")
    atlas = atlas[~drop].copy()

    ecology = atlas.expGroup.fillna("").isin(
        ["plant","in planta","mixed community","marine broth"])
    ecology_set = atlas[ecology].copy()
    atlas = atlas[~ecology].copy()
    print(f"  set aside ecology: {len(ecology_set)}")
    print(f"  -> working with {len(atlas):,} entries")

    # === BUCKETS ===
    def bucket(row):
        c = str(row.get("condition_1") or "").lower()
        g = str(row.get("expGroup") or "").lower()
        if any(ab in c for ab in ANTIBIOTICS):
            return "A_antibiotic"
        if g in ("nitrogen source","carbon source","vitamin","starvation","nutrient",
                  "nutrient t1","nutrient t2","sulfur source"):
            return "B_nutrient_niche"
        if g == "stress" and any(m in c for m in STRESS_KW):
            return "C_stress_metal_oxidative"
        if g in ("ph","temperature","anaerobic","respiratory growth","iron"):
            return "D_physical_stress"
        return "Z_other"
    atlas["bucket"] = atlas.apply(bucket, axis=1)
    print(f"\n  buckets:")
    for b,n in atlas.bucket.value_counts().items():
        print(f"    {b:<30s} {n:>5d}")

    atlas["gene_n"]  = atlas.gene.fillna("").astype(str).str.strip().str.lower()
    atlas["cond_n"]  = atlas.condition_1.apply(norm_cond)
    atlas["desc_short"] = atlas.desc.fillna("").astype(str).str[:90]

    out_dir = REPO / "outputs"; out_dir.mkdir(exist_ok=True)
    for blabel in ["A_antibiotic","B_nutrient_niche","C_stress_metal_oxidative"]:
        sub = atlas[atlas.bucket == blabel].copy()
        # named genes only (so we can search literature)
        named = sub[sub.gene_n != ""].copy()
        # group across organisms
        grouped = named.groupby(["gene_n","cond_n"]).agg(
            n_orgs=("organism","nunique"),
            organisms=("organism", lambda x: ",".join(sorted(set(x)))),
            n_entries=("organism","size"),
            median_fit=("fit","median"),
            worst_fit=("fit","min"),
            example_locus=("locusId","first"),
            example_org=("organism","first"),
            example_condition=("condition_1","first"),
            description=("desc_short","first"),
        ).reset_index()
        # CORE RANKING: cross-org consistency × magnitude
        grouped["score"] = grouped.n_orgs * (-grouped.median_fit)
        # surprise bias: prefer pairs that recur across 2+ organisms
        grouped = grouped.sort_values(
            ["n_orgs","score"], ascending=[False,False]
        ).head(100)
        path = out_dir / f"atlas_phase1_{blabel}.csv"
        grouped.to_csv(path, index=False)
        print(f"\n  {blabel}: top 100 -> {path.name}")
        # show top 10 with consistency >=2 orgs (the surprise candidates)
        top = grouped[grouped.n_orgs >= 2].head(10)
        if len(top):
            print(f"    cross-org candidates (n_orgs>=2):")
            for _,r in top.iterrows():
                print(f"      {r.gene_n:<10s} {r.cond_n[:30]:<30s} "
                      f"n_orgs={r.n_orgs}  med_fit={r.median_fit:+.1f}  "
                      f"{r.description[:50]}")
    # save cleaned atlas + ecology subset
    atlas.to_csv(out_dir/"atlas_phase1_clean.csv", index=False)
    ecology_set.to_csv(out_dir/"atlas_phase1_ecology.csv", index=False)
    print(f"\n  cleaned atlas: {len(atlas):,} -> outputs/atlas_phase1_clean.csv")
    print(f"  ecology: {len(ecology_set):,} -> outputs/atlas_phase1_ecology.csv")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    args = p.parse_args()
    sys.exit(main(args.atlas))
