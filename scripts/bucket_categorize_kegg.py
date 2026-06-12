"""KEGG-resolved functional category breakdown of essential OGs.

The companion to bucket_categorize.py for environments where feba.db is
available (Colab with Drive). bucket_categorize.py left 83.7% of the
5,658 beril essential OGs as 'unannotated_by_flags' because we couldn't
read feba.db's BestHitKEGG -> KEGGMember -> KgroupDesc chain locally.
This script resolves them.

Pipeline:
  1. Load labels.csv from the repo (the 52,940 essential genes).
  2. Join with orthology_features.csv for og_id.
  3. Pull BestHitKEGG + KEGGMember from feba.db -> (orgId, locusId) -> KO.
  4. Pull KgroupDesc to get the KO's human-readable description.
  5. Categorize each description by keyword regex into COG-style buckets.
  6. Roll up to OG level: each OG gets the modal category across its members.
  7. Cross-tab category x breadth tier; report.

Robust to schema variation: introspects column names via PRAGMA.

OUTPUT:
  outputs/essential_bucket_categories_kegg.md
  outputs/essential_bucket_categories_kegg.csv   (per-OG: og_id, category,
                                                   n_orgs_essential, ko, desc)
"""
from __future__ import annotations
import argparse, re, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"

DEFAULT_DB = Path("/content/feba.db")
DRIVE_DB = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                "fitness_browser/feba.db")

# Description keyword -> category. Order matters: more specific first.
CATEGORIES = [
    (r"ribosomal protein|ribosome (biogenesis|small|large)|rRNA",
     "J_translation_ribosome"),
    (r"aminoacyl-tRNA|--?tRNA ligase|tRNA synthetase",
     "J_aminoacyl_tRNA_synthetase"),
    (r"\btRNA\b|tRNA (uridine|modification|methylase|pseudouridine)",
     "J_tRNA_modification"),
    (r"elongation factor|translation initiation|peptide chain release|"
     r"translation termination|EF-Tu|EF-G|EF-Ts|IF-?[123]|RF-?[123]",
     "J_translation_factor"),
    (r"DNA-directed RNA polymerase|RNA polymerase subunit|sigma (factor|-?\d+)",
     "K_transcription_RNAP_sigma"),
    (r"transcription (elongation|termination|antitermination)|NusA|NusB|"
     r"NusG|GreA|GreB|Rho factor",
     "K_transcription_factor"),
    (r"transcriptional (regulator|repressor|activator)|response regulator|"
     r"DNA-binding|LysR|TetR|GntR|MarR|LuxR|AraC",
     "K_regulator"),
    (r"DNA polymerase|DNA primase|DNA helicase|DNA ligase|"
     r"DNA replication|primosome|replication initiator|"
     r"single-strand DNA-binding|SSB protein",
     "L_replication"),
    (r"DNA gyrase|DNA topoisomerase|topology",
     "L_DNA_topology"),
    (r"excinuclease|nucleotide excision|mismatch repair|recombinase|"
     r"recombination protein|RecA|RecBCD|RuvAB|UvrABC|MutHLS|"
     r"endonuclease|exonuclease",
     "L_DNA_repair"),
    (r"cell division|septum (formation|site)|FtsZ|FtsA|FtsW|FtsI|FtsQ|"
     r"divisome|Min[CDE]|ZapA|ZipA",
     "D_cell_division"),
    (r"cell shape|MreB|MreC|MreD|RodA|rod shape",
     "D_cell_shape"),
    (r"penicillin-binding protein|PBP\d|D-alanyl-D-alanine",
     "M_PBP_crosslink"),
    (r"UDP-N-acetyl|peptidoglycan|murein|MurA|MurB|MurC|MurD|MurE|MurF|"
     r"MraY|D-alanine--D-alanine ligase",
     "M_peptidoglycan"),
    (r"lipopolysaccharide|LPS|Kdo|lipid A|KdsA|KdsB|LpxA|LpxB|LpxC|LpxD|"
     r"GmhA|GmhB|heptose",
     "M_LPS"),
    (r"outer membrane (protein|biogenesis|assembly)|BamA|BamB|BamD|"
     r"LolA|LolB|LolC|LolD|LptA|LptB|LptC|LptD|LptE|porin",
     "M_outer_membrane"),
    (r"Sec translocase|SecA|SecB|SecY|SecE|SecG|YidC|TatA|TatB|TatC|"
     r"signal recognition particle|Ffh|signal peptidase|preprotein",
     "U_protein_secretion"),
    (r"ATP synthase|F-type ATPase|F0F1|F1F0|H\+-transporting ATP synthase",
     "C_ATP_synthase"),
    (r"NADH (dehydrogenase|:quinone)|cytochrome [cdo]|cytochrome bd|"
     r"cytochrome bc1|succinate dehydrogenase|fumarate reductase|"
     r"formate dehydrogenase|hydrogenase|terminal oxidase|"
     r"electron transport|quinol oxidase",
     "C_respiration"),
    (r"acyl carrier|acetyl-CoA carboxylase|fatty acid (synthase|biosynthesis|"
     r"elongation)|3-oxoacyl|beta-ketoacyl|enoyl-ACP|FabA|FabB|FabD|FabF|"
     r"FabG|FabH|FabI|FabZ",
     "I_fatty_acid_biosynth"),
    (r"phospholipid (biosynthesis|synthesis)|phosphatidyl|cardiolipin|"
     r"PlsB|PlsC|PlsX|PlsY|PgsA|Psd",
     "I_phospholipid_membrane"),
    (r"ribonucleotide reductase|thymidylate (synthase|kinase)|"
     r"dUTPase|thymidine kinase|nucleoside diphosphate kinase|"
     r"adenylate kinase|guanylate kinase|cytidylate kinase|uridylate",
     "F_nucleotide_kinases"),
    (r"purine (biosynthesis|metabolism)|IMP cyclohydrolase|GMP synthase|"
     r"phosphoribosyl|PurA|PurB|PurC|PurD|PurE|PurF|PurH|PurK|PurL|PurM|"
     r"PurN|PurQ|PurT",
     "F_purine_biosynth"),
    (r"pyrimidine (biosynthesis|metabolism)|orotate|carbamoyl-phosphate|"
     r"orotidine|UMP|CTP synthase|aspartate carbamoyltransferase|"
     r"PyrB|PyrC|PyrD|PyrE|PyrF|PyrG|PyrH",
     "F_pyrimidine_biosynth"),
    (r"folate|tetrahydrofolate|dihydrofolate|methylenetetrahydrofolate",
     "H_folate"),
    (r"riboflavin|FAD|FMN biosynthesis", "H_riboflavin_FMN"),
    (r"biotin biosynthesis|BioA|BioB|BioC|BioD|BioF|BioH",
     "H_biotin"),
    (r"thiamine|ThiC|ThiD|ThiE|ThiF|ThiG|ThiH",
     "H_thiamine"),
    (r"nicotinamide|nicotinic acid|NAD biosynthesis|NadA|NadB|NadC|NadD|"
     r"NadE|NadK",
     "H_NAD"),
    (r"coenzyme A|pantothenate|PanB|PanC|PanD|PanE",
     "H_CoA_pantothenate"),
    (r"menaquinone|ubiquinone|MenA|MenB|MenC|MenD|MenE|UbiA|UbiB|UbiD|"
     r"UbiE|UbiG|UbiH|UbiX",
     "H_quinone"),
    (r"heme biosynthesis|porphyrin|protoporphyrin|HemA|HemB|HemC|HemD|"
     r"HemE|HemH|HemL|HemN",
     "H_heme_porphyrin"),
    (r"pyridoxal|pyridoxine|vitamin B6", "H_pyridoxal"),
    (r"molybdopterin|molybdenum cofactor|MoaA|MoaB|MoaC|MoaD|MoaE|MoeA|MoeB|"
     r"MogA|MogB",
     "H_molybdopterin"),
    (r"cobalamin|vitamin B12|CobA|CobB|CobC|CobI|CobJ|CobK|CobL|CobM|CobN|"
     r"CobO|CobP|CobQ|CobR|CobS|CobT|CobU|CobV|CobW",
     "H_cobalamin_B12"),
    (r"tryptophan biosynthesis|TrpA|TrpB|TrpC|TrpD|TrpE|TrpF|TrpG|TrpS",
     "E_tryptophan"),
    (r"tyrosine biosynthesis|chorismate|prephenate|TyrA|TyrB|AroA|AroB|"
     r"AroC|AroD|AroE|AroG|AroH|AroK",
     "E_aromatic_aa"),
    (r"phenylalanine biosynthesis|PheA",
     "E_phenylalanine"),
    (r"histidine biosynthesis|HisA|HisB|HisC|HisD|HisF|HisG|HisH|HisI",
     "E_histidine"),
    (r"branched-chain amino acid|leucine biosynthesis|valine biosynthesis|"
     r"isoleucine biosynthesis|LeuA|LeuB|LeuC|LeuD|IlvA|IlvB|IlvC|IlvD|"
     r"IlvE|IlvH|IlvI",
     "E_BCAA"),
    (r"methionine biosynthesis|cobalamin-(dependent|independent) methionine|"
     r"MetA|MetB|MetC|MetE|MetF|MetH|MetK|MetL",
     "E_methionine"),
    (r"cysteine biosynthesis|CysE|CysH|CysK", "E_cysteine"),
    (r"serine biosynthesis|SerA|SerB|SerC", "E_serine"),
    (r"threonine biosynthesis|homoserine|ThrA|ThrB|ThrC",
     "E_threonine_homoserine"),
    (r"glycine cleavage|glycine biosynthesis|glycine hydroxymethyl|GlyA",
     "E_glycine"),
    (r"aspartate|asparagine|AsnA|AsnB|AspA|AspB", "E_asp_asn"),
    (r"glutamate|glutamine synthetase|GlnA|GlnE|GltA|GltB|GltD",
     "E_glu_gln"),
    (r"lysine biosynthesis|LysA|LysC|diaminopimelate|DapA|DapB|DapD|DapE|"
     r"DapF",
     "E_lysine"),
    (r"arginine biosynthesis|ornithine|carbamoyl phosphate|ArgA|ArgB|ArgC|"
     r"ArgD|ArgE|ArgF|ArgG|ArgH",
     "E_arginine"),
    (r"proline biosynthesis|ProA|ProB|ProC", "E_proline"),
    (r"alanine racemase|alanine aminotransferase|Alr|DadX",
     "E_alanine"),
    (r"glycolysis|gluconeogenesis|fructose-bisphosphate aldolase|"
     r"phosphoglucose isomerase|phosphofructokinase|glyceraldehyde 3-phosphate|"
     r"phosphoglycerate (kinase|mutase)|enolase|pyruvate kinase|"
     r"Pgi|PfkA|PfkB|FbaA|FbaB|Gap[AB]|Pgk|GpmA|GpmB|PykA|PykF|Tpi",
     "G_glycolysis"),
    (r"pyruvate dehydrogenase|2-oxoglutarate dehydrogenase|2-oxoacid",
     "G_pyruvate_dh"),
    (r"TCA cycle|citrate synthase|aconitase|isocitrate dehydrogenase|"
     r"succinyl-CoA|fumarase|malate dehydrogenase|GltA|AcnA|AcnB|Icd|"
     r"SucA|SucB|SucC|SucD|FumA|FumB|FumC|Mdh",
     "G_TCA_cycle"),
    (r"pentose phosphate|6-phosphogluconate|ribulose-phosphate|transketolase|"
     r"transaldolase|Zwf|Gnd|Rpe|RpiA|TktA|TktB|TalA|TalB",
     "G_pentose_phosphate"),
    (r"ABC transporter|transporter|permease|ATP-binding cassette|"
     r"major facilitator|MFS|symporter|antiporter|uniporter|channel|"
     r"phosphotransferase system|PTS",
     "P_transport"),
    (r"two-component system|sensor histidine kinase|histidine kinase|"
     r"signal transduction histidine",
     "T_signal_transduction"),
    (r"chaperone|GroEL|GroES|DnaK|DnaJ|GrpE|HtpG|ClpX|ClpA|ClpB|ClpP",
     "O_protein_quality"),
    (r"protease|peptidase|Lon|FtsH|HslV|HslU",
     "O_proteolysis"),
]


def categorize(desc):
    if not isinstance(desc, str) or not desc: return None
    d = desc.lower()
    for pat, cat in CATEGORIES:
        if re.search(pat, d, re.IGNORECASE):
            return cat
    return None


def pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--labels", type=Path, default=D / "labels.csv")
    ap.add_argument("--orth", type=Path, default=D / "orthology_features.csv")
    args = ap.parse_args()

    if not args.db.exists():
        if DRIVE_DB.exists():
            print(f"--db not at {args.db}; using Drive copy {DRIVE_DB}")
            args.db = DRIVE_DB
        else:
            print(f"ERROR: feba.db not found at {args.db} or {DRIVE_DB}",
                  file=sys.stderr); return 1

    import pandas as pd

    print(f"Loading labels + orthology ...")
    lab = pd.read_csv(args.labels)
    orth = pd.read_csv(args.orth,
                       usecols=["organism", "locus_tag", "og_id",
                                "family_n_organisms"])
    ess = lab[lab.essential == 1].merge(
        orth, on=["organism", "locus_tag"], how="left").dropna(subset=["og_id"])
    # strip "beril_" prefix to match feba.db orgIds
    ess["fb_org"] = ess.organism.str.replace(r"^beril_", "", regex=True)
    ess["locusId"] = ess.locus_tag.astype(str)
    print(f"  essentials with og_id: {len(ess):,}; orgs: "
          f"{ess.fb_org.nunique()}; OGs: {ess.og_id.nunique():,}")

    con = sqlite3.connect(str(args.db))
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    cur = con.cursor()
    tabs = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("BestHitKEGG", "KEGGMember", "KgroupDesc"):
        if t not in tabs:
            print(f"ERROR: table {t} missing from feba.db", file=sys.stderr)
            return 1

    # ---- BestHitKEGG: (orgId, locusId) -> (keggOrg, keggId) ----
    bhk_cols = [r[1] for r in cur.execute(
        "PRAGMA table_info(BestHitKEGG)").fetchall()]
    print(f"BestHitKEGG cols: {bhk_cols}")
    lo = pick_col(bhk_cols, "locusId"); or_ = pick_col(bhk_cols, "orgId")
    kid = pick_col(bhk_cols, "keggId"); korg = pick_col(bhk_cols, "keggOrg")
    bhk = pd.read_sql(f"SELECT {or_} AS orgId, {lo} AS locusId, "
                      f"{korg} AS keggOrg, {kid} AS keggId FROM BestHitKEGG "
                      f"WHERE {kid} IS NOT NULL AND {kid}!=''", con)
    bhk["locusId"] = bhk.locusId.astype(str)

    # ---- KEGGMember: (keggOrg, keggId) -> kgroup (KO) ----
    km_cols = [r[1] for r in cur.execute(
        "PRAGMA table_info(KEGGMember)").fetchall()]
    print(f"KEGGMember cols: {km_cols}")
    kg = pick_col(km_cols, "kgroup", "ko")
    mkid = pick_col(km_cols, "keggId"); mkorg = pick_col(km_cols, "keggOrg")
    km = pd.read_sql(f"SELECT {kg} AS ko, {mkid} AS keggId, "
                     f"{mkorg} AS keggOrg FROM KEGGMember "
                     f"WHERE {kg} IS NOT NULL AND {kg}!=''", con)

    bhk = bhk.merge(km, on=["keggId", "keggOrg"], how="inner")
    print(f"  (orgId, locusId, ko) rows: {len(bhk):,}; "
          f"distinct KOs: {bhk.ko.nunique():,}")

    # ---- KgroupDesc: ko -> description ----
    kd_cols = [r[1] for r in cur.execute(
        "PRAGMA table_info(KgroupDesc)").fetchall()]
    print(f"KgroupDesc cols: {kd_cols}")
    dko = pick_col(kd_cols, "kgroup", "ko")
    ddesc = pick_col(kd_cols, "desc", "description", "name", "definition")
    if not dko or not ddesc:
        print(f"ERROR: KgroupDesc missing expected cols", file=sys.stderr)
        return 1
    kd = pd.read_sql(f"SELECT {dko} AS ko, {ddesc} AS desc FROM KgroupDesc",
                     con)
    print(f"  KO descriptions: {len(kd):,}")
    con.close()

    bhk = bhk.merge(kd, on="ko", how="left")
    bhk["category"] = bhk.desc.apply(categorize)
    cat_cov = bhk.category.notna().sum() / max(len(bhk), 1)
    print(f"  description-categorized fraction: {cat_cov*100:.0f}%")

    # ---- join: essential locus -> category ----
    ess = ess.merge(bhk[["orgId", "locusId", "ko", "desc", "category"]],
                    left_on=["fb_org", "locusId"],
                    right_on=["orgId", "locusId"], how="left")
    print(f"  essentials with KEGG KO: "
          f"{ess.ko.notna().sum():,} / {len(ess):,} "
          f"({ess.ko.notna().mean()*100:.0f}%)")
    print(f"  essentials with category: "
          f"{ess.category.notna().sum():,} "
          f"({ess.category.notna().mean()*100:.0f}%)")

    # ---- roll up to OG level: dominant category ----
    def modal_cat(s):
        s = s.dropna()
        if len(s) == 0: return None
        return s.value_counts().index[0]
    og_summary = ess.groupby("og_id").agg(
        category=("category", modal_cat),
        ko=("ko", modal_cat),
        desc=("desc", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        n_orgs_essential=("organism", "nunique"),
        n_genes=("locus_tag", "size"),
    ).reset_index()

    cnt = og_summary.category.value_counts()
    print(f"\n=== OG-level category breakdown ({len(og_summary):,} OGs) ===")
    print(f"  categorized: {og_summary.category.notna().sum():,} "
          f"({og_summary.category.notna().mean()*100:.0f}%)")
    print(f"\n  category                                  n_OGs    %")
    for c, n in cnt.head(40).items():
        print(f"  {c:<40s} {n:>5d} {n/len(og_summary)*100:>4.1f}%")
    nc = og_summary.category.isna().sum()
    print(f"  (uncategorized)                          {nc:>5d} "
          f"{nc/len(og_summary)*100:>4.1f}%")

    # ---- roll up to COG single-letter ----
    og_summary["cog_letter"] = og_summary.category.str.split("_").str[0]
    letter_counts = og_summary.cog_letter.value_counts()
    cog_names = {"J":"Translation/ribosome","K":"Transcription/regulation",
                 "L":"DNA replication/repair","D":"Cell division/shape",
                 "M":"Cell wall/envelope","U":"Protein secretion",
                 "C":"Energy production","I":"Lipid metabolism",
                 "F":"Nucleotide metabolism","H":"Coenzyme metabolism",
                 "E":"Amino acid biosynthesis","G":"Carbohydrate metabolism",
                 "P":"Transport","T":"Signal transduction",
                 "O":"Protein turnover/chaperones"}
    print(f"\n=== COG single-letter roll-up ===")
    print(f"  letter   category                          n_OGs    %")
    total_cat = letter_counts.sum()
    for L, n in letter_counts.items():
        print(f"  {L:<8s} {cog_names.get(L,'?'):<32s} {n:>5d} "
              f"{n/total_cat*100:>4.1f}%")

    # ---- cross-tab: category x breadth tier ----
    def tier(n):
        if n >= 40: return "1_universal(>=40)"
        if n >= 20: return "2_broad(20-39)"
        if n >= 10: return "3_widely(10-19)"
        if n >= 5:  return "4_common(5-9)"
        if n >= 2:  return "5_uncommon(2-4)"
        return "6_singleton(1)"
    og_summary["tier"] = og_summary.n_orgs_essential.apply(tier)
    ct = pd.crosstab(og_summary.cog_letter.fillna("(none)"), og_summary.tier)
    ct = ct.reindex(sorted(ct.columns), axis=1)
    print(f"\n=== COG-letter x breadth-tier cross-tab (n OGs) ===")
    print(ct.to_string())

    # ---- write outputs ----
    out_md = REPO / "outputs" / "essential_bucket_categories_kegg.md"
    out_csv = REPO / "outputs" / "essential_bucket_categories_kegg.csv"
    out_md.parent.mkdir(exist_ok=True)
    og_summary.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    lines = ["# Essential OGs by KEGG-derived functional category\n",
             f"5,658 essential OGs from 48 beril organisms, categorized via "
             f"BestHitKEGG -> KEGGMember -> KgroupDesc -> regex-on-description.\n",
             f"**Coverage:** {og_summary.category.notna().sum():,} of "
             f"{len(og_summary):,} OGs "
             f"({og_summary.category.notna().mean()*100:.0f}%) got a category. "
             f"The rest are either non-KEGG-annotated (hypothetical) or didn't "
             f"match any keyword pattern.\n",
             "## COG single-letter breakdown\n",
             "| letter | name | n OGs | % of categorized |",
             "|---|---|---:|---:|"]
    for L, n in letter_counts.items():
        lines.append(f"| {L} | {cog_names.get(L,'?')} | {n:,} | "
                     f"{n/total_cat*100:.1f}% |")
    lines.append("\n## Fine-grained sub-categories (top 25)\n")
    lines.append("| category | n OGs |")
    lines.append("|---|---:|")
    for c, n in cnt.head(25).items():
        lines.append(f"| {c} | {n:,} |")
    lines.append("\n## COG letter x breadth tier (n OGs)\n")
    lines.append(ct.to_string().replace("\n", "\n    "))
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
