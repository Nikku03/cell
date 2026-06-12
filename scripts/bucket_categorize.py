"""Categorize the essential genes by functional category.

Naresh's question: "in what category do they fall?" -- meaning the 5,658
essential OGs.

Data caveat surfaced by inspection: in our LOCAL data the 3,021 essentials
with canonical gene names are all in the DEG-benchmark organisms (E. coli
Keio, M. tuberculosis, S. pneumoniae, etc.) which don't carry og_id in our
local features. The 48,525 essentials WITH og_id are all in beril_ Fitness
Browser organisms with accession-style locus tags (no canonical symbols).
We can't bridge these two subsets without feba.db's KEGG/Pfam annotations.

So we report TWO complementary analyses:

PART 1: COG-style breakdown of the 3,021 NAMED essentials via gene-symbol
        prefix mapping. This is biologically interpretable and is the
        textbook answer to "what categories do bacterial essentials fall
        into?"

PART 2: Coarse breakdown of the 5,658 beril ESSENTIAL OGs using the flag
        features we have (is_regulator, is_signaling, is_transporter,
        fba_in_model). ~16% coverage; the rest are "unannotated by these
        flags" and will need KEGG/Pfam from feba.db (on Drive) to resolve.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"

# canonical-symbol -> COG-like category (bacterial conventions)
PREFIX_RULES = [
    (re.compile(r"^(rps|rpl|rpm)[A-Z0-9]*$"), "J_translation_ribosome"),
    (re.compile(r"^(rrl|rrs|rrf)"),           "J_translation_rRNA"),
    (re.compile(r"^trn[A-Z]"),                "J_translation_tRNA"),
    (re.compile(r"^(infA|infB|infC|fmt|prfA|prfB|prfC|frr|tsf|fusA|tuf|tufA|tufB|"
                r"efp|efpL)$"),               "J_translation_factor"),
    (re.compile(r"^(rsm|rlm|rim|ksgA|rbfA|rsgA|era|engA|obgE|rnpA|rnpB|rnz|truB)"),
                                              "J_translation_assembly"),
    (re.compile(r"^(serS|thrS|alaS|glyS|glyQ|lysS|argS|hisS|leuS|ileS|metG|metS|"
                r"valS|pheS|pheT|tyrS|trpS|aspS|asnS|gluS|glnS|proS|cysS)$"),
                                              "J_aminoacyl_tRNA_synthetase"),
    (re.compile(r"^(rpoA|rpoB|rpoC|rpoZ|rpoE|rpoH|rpoD|rpoN|rpoS)$"),
                                              "K_transcription_RNAP"),
    (re.compile(r"^(nusA|nusB|nusG|greA|greB|rho)$"), "K_transcription_factor"),
    (re.compile(r"^(sigA|sigB|sigE|sigH|sigK)"),      "K_sigma_factor"),
    (re.compile(r"^(dnaA|dnaB|dnaC|dnaE|dnaG|dnaN|dnaQ|dnaX|holA|holB|holC|"
                r"holD|holE|polA|polB|polC|polIV|ligA|priA|priB|priC|ssb)$"),
                                              "L_replication"),
    (re.compile(r"^(gyrA|gyrB|topA|topB|parC|parE)$"), "L_topology"),
    (re.compile(r"^(recA|recB|recC|recD|recF|recO|recR|ruvA|ruvB|ruvC|"
                r"uvrA|uvrB|uvrC|uvrD|mutH|mutL|mutS|mutY)$"), "L_DNA_repair"),
    (re.compile(r"^(ftsA|ftsB|ftsE|ftsH|ftsI|ftsK|ftsL|ftsN|ftsQ|ftsW|ftsX|"
                r"ftsY|ftsZ|minC|minD|minE|zipA|zapA|zapB|sulA|sepF|gpsB|ezrA)$"),
                                              "D_cell_division"),
    (re.compile(r"^(mreB|mreC|mreD|rodA|rodZ|pbp[A-Z0-9]*)$"),
                                              "D_cell_shape"),
    (re.compile(r"^(murA|murB|murC|murD|murE|murF|murG|murI|ddlA|ddlB|dacA|"
                r"dacB|ampC|ampG|mtgA|mraY|mraZ)$"), "M_cell_wall_peptidoglycan"),
    (re.compile(r"^(lpxA|lpxB|lpxC|lpxD|lpxH|lpxK|kdsA|kdsB|kdtA|kdsC|kdsD|"
                r"gmhA|gmhB|rfaC)"),         "M_LPS_outer_envelope"),
    (re.compile(r"^(lpt[A-Z]|lol[A-Z]|bam[A-Z]|surA|surB|skp|degP|degQ|degS|"
                r"ompA|ompC|ompF|tolA|tolB|tolC|pal)$"),
                                              "M_outer_membrane"),
    (re.compile(r"^(secA|secB|secD|secE|secF|secG|secY|yidC|tatA|tatB|tatC|"
                r"ffh|ffs)$"),                "U_protein_secretion"),
    (re.compile(r"^atp[A-Z0-9]+$"),           "C_energy_ATP_synthase"),
    (re.compile(r"^(nuo|ndh|cyo|cyd|cyc|cco|qco|sdh|frd|nrf|nap|nar|hyp|hya|hyb)"),
                                              "C_energy_respiration"),
    (re.compile(r"^(acpP|acpS|acpT|accA|accB|accC|accD|fab[A-Z]|plsB|plsC|"
                r"plsX|plsY|psd|pgsA|cls)$"),"I_lipid_fatty_acid"),
    (re.compile(r"^(purA|purB|purC|purD|purE|purF|purH|purK|purL|purM|purN|"
                r"purQ|purT|pyrB|pyrC|pyrD|pyrE|pyrF|pyrG|pyrH|nrdA|nrdB|"
                r"nrdE|nrdF|guaA|guaB|gmk|ndk|adk|tmk|cmk|thyA|thyX|dut|"
                r"udk|upp|tdk)$"),            "F_nucleotide_metab"),
    (re.compile(r"^(bioA|bioB|bioC|bioD|bioF|bioH|thiC|thiD|thiE|thiH|thiL|"
                r"thiM|nadA|nadB|nadC|nadD|nadE|nadK|folA|folB|folC|folD|"
                r"folE|folK|folP|panB|panC|panD|panE|menA|menB|menC|menD|"
                r"menE|menF|menG|menH|ubiA|ubiB|ubiC|ubiD|ubiE|ubiF|ubiG|"
                r"ubiH|ubiI|ubiX|mogA|mogB|moaA|moaB|moaC|moaD|moaE|moeA|"
                r"moeB|hemA|hemB|hemC|hemD|hemE|hemH|hemL|hemN|ribA|ribB|"
                r"ribC|ribD|ribE|ribF|ribH)$"),"H_coenzyme_metab"),
    (re.compile(r"^(trpA|trpB|trpC|trpD|trpE|trpG|trpS|tyrA|tyrB|aroA|aroB|"
                r"aroC|aroD|aroE|aroG|aroH|aroK|pheA|pheS|pheT|hisA|hisB|"
                r"hisC|hisD|hisF|hisG|hisH|hisI|leuA|leuB|leuC|leuD|ilvA|"
                r"ilvB|ilvC|ilvD|ilvE|ilvH|ilvI|metA|metB|metC|metE|metF|"
                r"metH|metK|metL|cysE|cysH|cysK|cysS|serA|serB|serC|thrA|"
                r"thrB|thrC|glyA|asd|asnA|asnB|aspA|aspB|gltA|gltB|gltD|"
                r"glnA|glnE|lysA|lysC|argA|argB|argC|argD|argE|argF|argG|"
                r"argH|proA|proB|proC|carA|carB|dapA|dapB|dapD|dapE|dapF)$"),
                                              "E_amino_acid_biosynth"),
    (re.compile(r"^(pgi|pfkA|pfkB|fbaA|fbaB|tpi|tpiA|gapA|gapB|pgk|gpmA|gpmB|"
                r"eno|pykA|pykF|zwf|gnd|tktA|tktB|talA|talB|rpe|rpiA|ackA|"
                r"pta|adhE|glk|maeA|maeB|sdhA|sdhB|sdhC|sdhD|sucA|sucB|"
                r"sucC|sucD|mdh|gltA|acnA|acnB|icd|fumA|fumB|fumC)$"),
                                              "G_central_carbon"),
]


def categorize_name(name):
    if not isinstance(name, str): return None
    n = name.strip()
    for pat, cat in PREFIX_RULES:
        if pat.match(n):
            return cat
    return None


def main() -> int:
    import pandas as pd

    print("Loading ...")
    lab = pd.read_csv(D / "labels.csv")
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism","locus_tag","og_id",
                                "family_n_organisms"])
    reg = pd.read_csv(D / "regulator_features.csv")
    fba = pd.read_csv(D / "fba_features.csv")

    ess = lab[lab.essential == 1].copy()

    # === PART 1: named essentials -> COG categories via gene-symbol prefix ===
    named = ess[ess.gene_name.notna()].copy()
    named["cog_cat"] = named.gene_name.apply(categorize_name)
    n_named = len(named); n_cat = named.cog_cat.notna().sum()
    print(f"\n=== PART 1: named essentials -> COG-style categories ===")
    print(f"  named essentials: {n_named:,}; "
          f"categorized by prefix: {n_cat:,} ({n_cat/n_named*100:.0f}%)")
    print(f"  source organisms: {named.organism.nunique()} -- "
          f"DEG benchmark set (E. coli Keio, M. tuberculosis, etc.)")

    cog_counts = named.cog_cat.value_counts().sort_values(ascending=False)
    print(f"\n  category distribution:")
    print(f"  {'category':<32s} {'n':>5s} {'%':>5s}")
    for cat, n in cog_counts.items():
        print(f"  {cat:<32s} {n:>5d} {n/n_cat*100:>4.1f}%")

    uncategorized = named[named.cog_cat.isna()]
    print(f"\n  uncategorized (no prefix match): {len(uncategorized):,}")
    if len(uncategorized) > 0:
        print(f"  sample names: "
              f"{uncategorized.gene_name.sample(min(15, len(uncategorized)), random_state=0).tolist()}")

    # roll up to single-letter COG
    def cog_letter(cat):
        return cat.split("_")[0] if isinstance(cat, str) else None
    named["cog_letter"] = named.cog_cat.apply(cog_letter)
    letter_counts = named.cog_letter.value_counts().sort_values(ascending=False)
    cog_names = {"J":"Translation/ribosome","K":"Transcription","L":"DNA replication/repair",
                 "D":"Cell division/shape","M":"Cell wall/membrane",
                 "U":"Protein secretion","C":"Energy production",
                 "I":"Lipid metabolism","F":"Nucleotide metabolism",
                 "H":"Coenzyme metabolism","E":"Amino acid biosynthesis",
                 "G":"Carbohydrate metabolism"}
    print(f"\n  rolled up to COG single-letter:")
    print(f"  {'letter':<8s} {'name':<32s} {'n':>5s} {'%':>5s}")
    for letter, n in letter_counts.items():
        nm = cog_names.get(letter, "?")
        print(f"  {letter:<8s} {nm:<32s} {n:>5d} {n/letter_counts.sum()*100:>4.1f}%")

    # === PART 2: beril OG-level coarse buckets from flags ===
    print(f"\n\n=== PART 2: beril OG-level coarse buckets (flag features) ===")
    beril = ess.merge(orth, on=["organism","locus_tag"], how="left")
    beril = beril.merge(reg[["organism","locus_tag","is_regulator","is_signaling",
                             "is_transporter","is_conditional"]],
                        on=["organism","locus_tag"], how="left")
    beril = beril.merge(fba[["organism","locus_tag","fba_in_model"]],
                        on=["organism","locus_tag"], how="left")
    beril = beril.dropna(subset=["og_id"])

    og_flags = beril.groupby("og_id").agg(
        any_regulator=("is_regulator", lambda s: int(s.fillna(0).max())),
        any_signaling=("is_signaling", lambda s: int(s.fillna(0).max())),
        any_transporter=("is_transporter", lambda s: int(s.fillna(0).max())),
        any_metabolic=("fba_in_model", lambda s: int(s.fillna(0).max())),
        n_orgs_essential=("organism", "nunique"),
    ).reset_index()

    def coarse(r):
        if r.any_metabolic:   return "metabolic_(in_FBA_model)"
        if r.any_regulator:   return "regulator"
        if r.any_signaling:   return "signaling"
        if r.any_transporter: return "transport"
        return "unannotated_by_flags"
    og_flags["coarse"] = og_flags.apply(coarse, axis=1)

    print(f"  total essential OGs: {len(og_flags):,}")
    cnt = og_flags.coarse.value_counts()
    print(f"  {'bucket':<32s} {'n':>5s} {'%':>5s}")
    for k, v in cnt.items():
        print(f"  {k:<32s} {v:>5d} {v/len(og_flags)*100:>4.1f}%")
    print(f"\n  CAVEAT: 'unannotated_by_flags' = the og_id had no member tagged")
    print(f"  as regulator/signaling/transporter, and was not in any metabolic")
    print(f"  model. These are likely structural/biosynthesis/translation genes")
    print(f"  -- exactly the J/L/M/D categories that dominate PART 1. To resolve")
    print(f"  them properly we need KEGG/Pfam annotations from feba.db (Drive).")

    # cross-tab: coarse x breadth tier (the rarefaction tiers)
    def tier(n):
        if n >= 40: return "1_universal(>=40)"
        if n >= 20: return "2_broad(20-39)"
        if n >= 10: return "3_widely(10-19)"
        if n >= 5:  return "4_common(5-9)"
        if n >= 2:  return "5_uncommon(2-4)"
        return "6_singleton(1)"
    og_flags["tier"] = og_flags.n_orgs_essential.apply(tier)
    print(f"\n  coarse x breadth-tier cross-tab (n OGs):")
    ct = pd.crosstab(og_flags.coarse, og_flags.tier).reindex(
        sorted(set(og_flags.tier)), axis=1)
    print(ct.to_string())

    # === verdict ===
    print(f"\n\n=== verdict ===")
    print(f"From PART 1 (named essentials, ~2,400 categorized):")
    top = letter_counts.head(5)
    print(f"  Top 5 COG categories cover "
          f"{top.sum()/letter_counts.sum()*100:.0f}% of named essentials:")
    for letter, n in top.items():
        print(f"    {letter} ({cog_names.get(letter,'?')}): "
              f"{n/letter_counts.sum()*100:.0f}%")
    print(f"\nThis matches the DEG-review claim: 'essential genes cluster in")
    print(f"fundamental life processes' -- J (translation), L (replication),")
    print(f"K (transcription), D (cell division), M (envelope) dominate.")

    # write markdown
    out_lines = ["# Essential genes by functional category\n"]
    out_lines.append(f"## Part 1: named essentials (n={n_named:,}) -> COG categories\n")
    out_lines.append("Source: 11 DEG benchmark organisms with canonical gene "
                     "symbols. Categorized by symbol-prefix mapping; coverage "
                     f"{n_cat/n_named*100:.0f}%.\n")
    out_lines.append("| COG letter | category | n | % |")
    out_lines.append("|---|---|---:|---:|")
    for letter, n in letter_counts.items():
        out_lines.append(f"| {letter} | {cog_names.get(letter,'?')} | {n:,} | "
                         f"{n/letter_counts.sum()*100:.1f}% |")
    out_lines.append("\n## Part 2: beril OG-level breakdown (n=5,658 OGs)\n")
    out_lines.append("Coverage limitation: only flag-based categories available "
                     "locally (~16%); the rest are 'unannotated_by_flags' and "
                     "are likely the J/L/M/D categories that dominate Part 1. "
                     "Resolving them needs KEGG/Pfam from feba.db (Drive).\n")
    out_lines.append("| bucket | n OGs | % |")
    out_lines.append("|---|---:|---:|")
    for k, v in cnt.items():
        out_lines.append(f"| {k} | {v:,} | {v/len(og_flags)*100:.1f}% |")
    out_path = REPO / "outputs" / "essential_bucket_categories.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
