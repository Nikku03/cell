"""Manually-transcribed Kuhner 2009 SOM Table S5 complex membership.

Transcribed from rendered PNGs of pages 89-95 of
memory_bank/data/multiorg_essentiality/raw/mpneumoniae/kuhner_2009/
kuhner_2009_som.pdf. Source: Kuhner et al. 2009 Science 326:1235
(DOI 10.1126/science.1176343), Table S5 "List of literature curated
protein complexes".

Each entry: (name, [(mpn_id, protein_name)], default_stoichiometry,
            source_pmid_or_db, organism_evidence)
For heteromultimers, default stoichiometry = 1 per subunit.
For homomultimers, the listed copy count from the paper.
"""

HETEROMULTIMERS = [
    {"name": "DNA Gyrase complex",
     "members": [("MPN003", "GyrB"), ("MPN004", "GyrA")],
     "evidence": "Expasy", "organism": "M. pneumoniae"},
    {"name": "Phenylalanine-tRNA synthetase complex",
     "members": [("MPN105", "SyfA"), ("MPN106", "SyfB")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "RNA polymerase complex",
     "members": [("MPN024", "RpoE"), ("MPN191", "RpoA"),
                 ("MPN352", "RpoD"), ("MPN516", "RpoB"),
                 ("MPN515", "RpoC")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Transcription antitermination factor complex",
     "members": [("MPN067", "NusG"), ("MPN024", "RpoE"),
                 ("MPN191", "RpoA"), ("MPN352", "RpoD"),
                 ("MPN516", "RpoB"), ("MPN515", "RpoC")],
     "evidence": "Expasy", "organism": "by homology to Bacillus subtilis"},
    {"name": "Pyruvate dehydrogenase complex",
     "members": [("MPN390", "PdhD"), ("MPN391", "PdhC"),
                 ("MPN392", "PdhB"), ("MPN393", "PdhA")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Ribonucleoside-diphosphate reductase complex",
     "members": [("MPN322", "NrdF"), ("MPN324", "NrdE")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "DNA Topoisomerase IV complex",
     "members": [("MPN122", "ParE"), ("MPN123", "ParC")],
     "evidence": "Expasy", "organism": "M. pneumoniae"},
    {"name": "DNA polymerase III complex",
     "members": [("MPN001", "DnaN"), ("MPN007", "Mpn007"),
                 ("MPN034", "PolC"), ("MPN378", "DnaE")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Translation elongation factor complex",
     "members": [("MPN450", "Tsf"), ("MPN618", "Tsf"),
                 ("MPN631", "Tsf"), ("MPN665", "Tuf"), ("MPN573", "GroL")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Ribosome complex",
     "members": [
         # 50S subunit (L1-L36)
         ("MPN069", "RpmG2"), ("MPN116", "RpmI"), ("MPN117", "RplT"),
         ("MPN164", "RplI"), ("MPN165", "RpsR"), ("MPN166", "RpsF"),
         ("MPN167", "RplC"), ("MPN168", "RplD"), ("MPN169", "RplW"),
         ("MPN170", "RplB"), ("MPN171", "RpsS"), ("MPN172", "RplV"),
         ("MPN173", "RpsC"), ("MPN174", "RplP"), ("MPN175", "RpmC"),
         ("MPN176", "RpsQ"), ("MPN177", "RplN"), ("MPN178", "RplX"),
         ("MPN179", "RplE"), ("MPN180", "RpsN"), ("MPN181", "RpsH"),
         ("MPN182", "RplF"), ("MPN183", "RplR"), ("MPN188", "RpsE"),
         ("MPN189", "RpmD"), ("MPN190", "RplO"), ("MPN192", "RpmJ"),
         ("MPN208", "RplQ"), ("MPN219", "RplM"), ("MPN220", "RpsI"),
         ("MPN225", "RpsB"), ("MPN226", "RpsK"), ("MPN228", "RpsM"),
         ("MPN230", "RpsR"), ("MPN231", "RpoA"), ("MPN296", "RpsR"),
         ("MPN325", "RpsT"), ("MPN327", "RpsP"), ("MPN360", "RpmA"),
         ("MPN446", "RplK"), ("MPN471", "RplA"), ("MPN538", "RplJ"),
         ("MPN539", "RplL"), ("MPN540", "RpsR"), ("MPN541", "RpsR"),
         ("MPN616", "RpsR"), ("MPN617", "RpmF"), ("MPN622", "RpsR"),
         ("MPN624", "RpmG2"), ("MPN658", "RplS"), ("MPN660", "RpsP"),
         ("MPN682", "RpmH"),
     ],
     "evidence": "Expasy", "organism": "M. pneumoniae"},
    {"name": "Cohesin-like complex",
     "members": [("MPN300", "ScpA"), ("MPN301", "ScpB"), ("MPN426", "Smc")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "RbfA-30S-ribosomal subunit complex",
     "members": [("MPN156", "RbfA")],
     "evidence": "PMID 15866174", "organism": "by homology to Thermus thermophilus",
     "notes": "binds the 30S ribosomal subunit; the 30S subunit's components are listed under 'Ribosome complex'"},
    {"name": "Glycolytic enzyme complex 1",
     "members": [("MPN430", "GapA"), ("MPN674", "Ldh"),
                 ("MPN303", "Pyk"), ("MPN302", "Pfk")],
     "evidence": "PMID 15701694", "organism": "by homology to Mus musculus"},
    {"name": "DnaK-GrpE-DnaJ complex",
     "members": [("MPN434", "DnaK"), ("MPN021", "DnaJ"), ("MPN120", "GrpE")],
     "evidence": "PMID 15632130", "organism": "by homology to Escherichia coli"},
    {"name": "Glycolytic enzyme complex 2",
     "members": [("MPN025", "Fba"), ("MPN429", "Pgk"),
                 ("MPN606", "Eno"), ("MPN303", "Pyk")],
     "evidence": "PMID 8613464", "organism": "by homology to MCF-7 cells (eukaryotes)"},
    {"name": "Era-30S-ribosomal subunit complex",
     "members": [("MPN568", "Era")],
     "evidence": "PMID 15866174", "organism": "by homology to Thermus thermophilus",
     "notes": "binds the 30S ribosomal subunit"},
    {"name": "GreA-RNA polymerase complex",
     "members": [("MPN401", "GreA"), ("MPN024", "RpoE"),
                 ("MPN191", "RpoA"), ("MPN352", "RpoD"),
                 ("MPN516", "RpoB"), ("MPN515", "RpoC")],
     "evidence": "PMID 17207814", "organism": "M. pneumoniae by similarity"},
    {"name": "Dihydrofolate reductase-thymidylate synthase complex",
     "members": [("MPN321", "FolA"), ("MPN320", "ThyA")],
     "evidence": "Expasy", "organism": "other"},
    {"name": "Holliday junction ATP-dependent helicase complex",
     "members": [("MPN535", "RuvB"), ("MPN536", "RuvA")],
     "evidence": "Expasy", "organism": "M. pneumoniae"},
    {"name": "Aspartyl-glutamyl-tRNA(Asn/Gln) amidotransferase complex",
     "members": [("MPN236", "GatB"), ("MPN237", "GatA"), ("MPN238", "GatC")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "UvrABC complex",
     "members": [("MPN125", "UvrC"), ("MPN619", "UvrA"), ("MPN211", "UvrB")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Phosphotransfer system maltose complex",
     "members": [("MPN053", "PtsH"), ("MPN651", "MtlA"), ("MPN653", "MtlF")],
     "evidence": "PMID 17803963", "organism": "by homology to Escherichia coli"},
    {"name": "DnaB-DnaG complex",
     "members": [("MPN232", "DnaB"), ("MPN353", "DnaG")],
     "evidence": "PMID 17947583", "organism": "by homology to Escherichia coli"},
    {"name": "Phosphotransfer system glucose complex",
     "members": [("MPN053", "PtsH"), ("MPN207", "PtsG")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Phosphotransfer system fructose complex",
     "members": [("MPN053", "PtsH"), ("MPN078", "FruA")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "Phosphotransfer system Ula (ascorbate) complex",
     "members": [("MPN053", "PtsH"), ("MPN494", "UlaC"),
                 ("MPN495", "UlaB"), ("MPN496", "UlaA")],
     "evidence": "Expasy", "organism": "M. pneumoniae by similarity"},
    {"name": "tRNA modification complex",
     "members": [("MPN557", "MnmG"), ("MPN008", "TrmE")],
     "evidence": "PMID 17062623", "organism": "by homology to Escherichia coli"},
]


HOMOMULTIMERS = [
    # (mpn_id, protein_name, copies_per_complex, source, organism, function)
    ("MPN314", "MraZ", 8, "Expasy", "M. pneumoniae", "Cell division MraZ"),
    ("MPN303", "Pyk", 4, "Expasy", "M. pneumoniae by similarity", "Pyruvate kinase"),
    ("MPN320", "ThyA", 2, "Expasy", "M. pneumoniae by similarity", "Thymidylate synthase"),
    ("MPN674", "Ldh", 4, "Expasy", "M. pneumoniae by similarity", "L-lactate dehydrogenase"),
    ("MPN025", "Fba", 2, "Expasy", "M. pneumoniae by similarity", "Fructose-biphosphate aldolase"),
    ("MPN317", "FtsZ", None, "PMID 17977836", "M. pneumoniae by similarity", "Cell division FtsZ; multiple, aggregates to form a ring-like structure"),
    ("MPN223", "HprK", 6, "Expasy", "M. pneumoniae", "HPr kinase/phosphorylase"),
    ("MPN302", "PfkA", None, "PMID 16289704", "by homology to Escherichia coli", "6-phosphofructokinase / Phosphohexokinase; multiple"),
    ("MPN479", "AzoR", 2, "PMID 16664776", "by homology to Escherichia coli", "FMN-dependent NADH-azoreductase"),
    ("MPN430", "GapA", 4, "Expasy", "M. pneumoniae by similarity", "Glyceraldehyde-3-phosphate dehydrogenase"),
    ("MPN533", "AckA", 4, "PMID 21667429037", "by homology to Veillonella alcalescens", "Acetate kinase; 2 to 4 copies"),
    ("MPN419", "AlaS", 2, "Expasy / PMID 9207019", "by homology to Thermus thermophilus", "Alanyl-tRNA synthetase"),
    ("MPN267", "PpnK", 2, "PMID 11006082", "by homology to Mycobacterium flavus / tuberculosis", "Probable inorganic polyphosphate/ATP-NAD kinase"),
    ("MPN490", "RecA", 6, "PMID 17955055", "by homology to Escherichia coli", "Recombinase A"),
    ("MPN531", "ClpB", 6, "Expasy", "M. pneumoniae by similarity", "Chaperone protein ClpB"),
    ("MPN232", "DnaB", 6, "PMID 17947583", "by homology to Escherichia coli", "Replicative DNA helicase"),
    ("MPN050", "GlpK", 4, "PMID 9930671", "by homology to Escherichia coli", "Glycerol kinase; 2 to 4 copies"),
    ("MPN022", "Pip", None, "PMID 8885412", "by homology to Xanthomonas campestris pv. citri", "Putative proline iminopeptidase; multiple"),
    ("MPN023", "MetG", 2, "PMID 2126467", "by homology to Escherichia coli", "Methionyl-tRNA synthetase"),
    ("MPN606", "Eno", 2, "Expasy", "M. pneumoniae by similarity", "Enolase"),
    ("MPN126", "Mpn126", 4, "PMID 17586769", "by homology to Escherichia coli", "Putative Metallophosphoesterase"),
    ("MPN229", "Ssb", 4, "PMID 16041080", "by homology to Mycobacterium smegmatis", "Single stranded DNA binding protein"),
    ("MPN560", "Mpn560", 3, "PMID 1722335911911465", "by homology to Lactococcus lactis", "Arginine deiminase-like protein; 2 to 3 copies"),
    ("MPN001", "DnaN", 2, "Expasy", "by homology to Streptococcus pneumoniae", "DNA polymerase III subunit beta"),
    ("MPN003", "GyrB", 2, "PMID 11850422", "by homology to Thermus thermophilus", "DNA gyrase subunit B (homomultimer)"),
    ("MPN006", "Tmk", 2, "PMID 11071809", "by homology to Homo sapiens", "Thymidylate kinase"),
    ("MPN007", "Mpn007", 6, "PMID 11545747", "by homology to Pyrococcus furiosus", "Uncharacterized protein MG007 homolog"),
    ("MPN034", "PolC", 2, "Expasy", "M. pneumoniae by similarity", "DNA polymerase III polC-type"),
    ("MPN063", "DeoC", 2, "PMID 7675789", "by homology to Escherichia coli", "Deoxyribose-phosphate aldolase; 1 or 2 copies"),
    ("MPN066", "ManB", 2, "PMID 16540464", "by homology to Sulfolobus tokodaii", "Phosphomannomutase"),
    ("MPN079", "FruK", 8, "PMID 2527305", "by homology to Homo sapiens", "Putative 1-phosphofructokinase"),
    ("MPN105", "PheS", 2, "PMID 7664121", "by homology to Thermus thermophilus", "Phenylalanyl-tRNA synthetase alpha chain"),
    ("MPN106", "PheT", 2, "PMID 7664121", "by homology to Thermus thermophilus", "Phenylalanyl-tRNA synthetase beta chain"),
    ("MPN191", "RpoA", 2, "Expasy", "M. pneumoniae by similarity", "DNA-directed RNA polymerase subunit alpha"),
    ("MPN322", "NrdF", 2, "PMID 16301799", "by homology to Salmonella typhimurium", "Ribonucleoside-diphosphate reductase subunit beta"),
    ("MPN324", "NrdE", 2, "PMID 8052308", "by homology to Salmonella typhimurium", "Ribonucleoside-diphosphate reductase subunit alpha"),
    ("MPN328", "Nfo", 2, "PMID 10458614", "by homology to Escherichia coli", "Probable endonuclease 4"),
    ("MPN391", "PdhC", 60, "PMID 9990008", "M. pneumoniae by similarity", "Dihydrolipoyllysine-residue acetyltransferase"),
    ("MPN392", "PdhB", 2, "Expasy", "M. pneumoniae by similarity", "Pyruvate dehydrogenase E1 component beta"),
    ("MPN393", "PdhA", 2, "Expasy", "M. pneumoniae by similarity", "Pyruvate dehydrogenase E1 component alpha"),
    ("MPN470", "PepP", 2, "PMID 12377124", "by homology to Pyrococcus horikoshii", "Putative Xaa-Pro aminopeptidase"),
    ("MPN492", "UlaE", 4, "PMID 11732895", "by homology to Pseudomonas cichorii", "Probable L-ribulose-5-phosphate 3-epimerase UlaE"),
    ("MPN550", "ThiL", 2, "PMID 16343540", "by homology to Bacillus anthracis", "Probable thiamine biosynthesis protein"),
    ("MPN558", "RsmG", 2, "PMID 18667428", "by homology to Geobacter sulfurreducens", "Ribosomal RNA small subunit methyltransferase G"),
    ("MPN562", "NadE", 2, "PMID 15645437", "by homology to Bacillus subtilis", "Probable NH3-dependent NAD+ synthetase"),
    ("MPN573", "GroL", 14, "Expasy", "M. pneumoniae by similarity", "60 kDa chaperonin (GroEL); 2x7 = 14 copies"),
    ("MPN638", "Mpn638", 2, "PMID 16038930", "by homology to Methanocaldococcus jannaschii", "Putative type I restriction enzyme specificity protein"),
    ("MPN665", "Tuf", 2, "PMID 16257965", "by homology to Escherichia coli", "Elongation factor Tu"),
    ("MPN390", "PdhD", 2, "Expasy", "M. pneumoniae by similarity", "Dihydrolipoyl dehydrogenase"),
    ("MPN574", "GroS", 7, "Expasy", "M. pneumoniae by similarity", "10 kDa chaperonin (GroES)"),
    ("MPN122", "ParE", 2, "Expasy", "M. pneumoniae by similarity", "DNA topoisomerase 4 subunit B (homomultimer)"),
    ("MPN332", "Lon", 4, "Expasy", "M. pneumoniae by similarity", "ATP-dependent protease Lon"),
    ("MPN008", "TrmE", 2, "Expasy", "by homology to Thermotoga maritima", "Probable tRNA modification GTPase (homomultimer)"),
    ("MPN278", "Glf", 2, "PMID 10089346", "by homology to Escherichia coli", "UDP-galactopyranose mutase"),
    ("MPN632", "PyrH", 6, "PMID 17297917", "by homology to Sulfolobus solfataricus", "Uridylate kinase"),
    ("MPN595", "RpiB", 4, "PMID 15162497", "by homology to Thermotoga maritima", "Ribose-5-phosphate isomerase B"),
    ("MPN186", "Map", 2, "PMID 12297034", "by homology to Bacillus stearothermophilus", "Methionine aminopeptidase Peptidase M"),
    ("MPN158", "RibF", 2, "PMID 15468322", "by homology to Thermotoga maritima", "FAD synthetase"),
    ("MPN245", "Def", 2, "PMID 17012781", "by homology to Staphylococcus aureus", "GMP kinase"),
    ("MPN298", "AcpS", 2, "PMID 16788183", "M. pneumoniae", "Acyl carrier protein (ACP) synthetase"),
    ("MPN395", "Apt", 2, "PMID 1039317011535055", "by homology to Leishmania donovani", "Adenine phosphoribosyltransferase"),
    ("MPN553", "ThrS", 2, "Expasy", "M. pneumoniae by similarity", "Threonyl-tRNA synthetase"),
    ("MPN064", "DeoA", None, "Expasy", "M. pneumoniae by similarity", "Thymidine phosphorylase; multiple"),
    ("MPN073", "Prs", None, "PMID 2169413", "by homology to Bacillus subtilis", "Ribose-phosphate pyrophosphokinase; multiple"),
    ("MPN120", "GrpE", 2, "Expasy", "M. pneumoniae by similarity", "Protein GrpE"),
    ("MPN257", "GalE", 2, "Expasy", "M. pneumoniae by similarity", "UDP-glucose 4-epimerase"),
    ("MPN265", "TrpS", 2, "Expasy", "M. pneumoniae by similarity", "Tryptophanyl-tRNA synthetase"),
    ("MPN686", "DnaA", None, "PMID 16829961", "by homology to Aquifex aeolicus", "Replication initiator protein DnaA; multiple"),
    ("MPN629", "TpiA", 2, "Expasy", "M. pneumoniae by similarity", "Triosephosphate isomerase"),
    ("MPN576", "GlyA", 4, "Expasy", "M. pneumoniae by similarity", "Serine hydroxymethyltransferase"),
    ("MPN627", "PtsI", 2, "Expasy", "M. pneumoniae by similarity", "Phosphoenolpyruvate-protein phosphotransferase"),
    ("MPN354", "GlyS", 2, "Expasy", "M. pneumoniae by similarity", "Glycyl-tRNA synthetase"),
    ("MPN005", "SerS", 2, "Expasy", "M. pneumoniae by similarity", "Seryl-tRNA synthetase"),
    ("MPN017", "FolD", 2, "Expasy", "M. pneumoniae by similarity", "Bifunctional protein FolD"),
    ("MPN277", "LysS", 2, "Expasy", "M. pneumoniae by similarity", "Lysyl-tRNA synthetase"),
    ("MPN045", "HisS", 2, "Expasy", "M. pneumoniae by similarity", "Histidyl-tRNA synthase"),
    ("MPN240", "TrxB", 2, "Expasy", "M. pneumoniae by similarity", "Thioredoxin reductase"),
    ("MPN021", "DnaJ", 2, "Expasy", "M. pneumoniae by similarity", "Chaperone protein DnaJ (homomultimer)"),
]


def write_complex_formation_xlsx():
    """Update cell_sim/data/Mycoplasma_pneumoniae/input_data/complex_formation.xlsx
    with Kuhner-derived complexes alongside the existing Ribosome row."""
    import pandas as pd
    from pathlib import Path

    OUT = Path('cell_sim/data/Mycoplasma_pneumoniae/input_data/complex_formation.xlsx')

    rows = []
    # Heteromultimers
    for c in HETEROMULTIMERS:
        members = c['members']
        rows.append({
            'Name': c['name'],
            'Genes Products': ','.join(m[0] for m in members),
            'Stoichiometries': ','.join('1' for _ in members),
            'Init. Count': None,
            'Assembly Pathway': c['evidence'],
            'PDB Structures': '',
            'How Cal Count': f"Kuhner 2009 SOM Table S5 i) heteromultimers; {c['organism']}",
        })
    # Homomultimers
    for mpn, prot, copies, src, org, fn in HOMOMULTIMERS:
        rows.append({
            'Name': fn,
            'Genes Products': mpn,
            'Stoichiometries': str(copies) if copies is not None else 'multiple',
            'Init. Count': None,
            'Assembly Pathway': src,
            'PDB Structures': '',
            'How Cal Count': f"Kuhner 2009 SOM Table S5 ii) homomultimers; {org}",
        })

    complexes_df = pd.DataFrame(rows, columns=[
        'Name', 'Genes Products', 'Stoichiometries', 'Init. Count',
        'Assembly Pathway', 'PDB Structures', 'How Cal Count',
    ])
    print(f"Kuhner heteromultimers: {len(HETEROMULTIMERS)}")
    print(f"Kuhner homomultimers:   {len(HOMOMULTIMERS)}")
    print(f"Total new rows:         {len(complexes_df)}")

    # Read existing complex_formation.xlsx (which has Ribosome from Maier S7)
    # and merge — Kuhner Ribosome supersedes Maier (more accurate subunits).
    # Drop the old Ribosome row, keep all other sheets unchanged.
    existing_complexes = pd.read_excel(OUT, sheet_name='Complexes')
    print(f"Existing Complexes sheet: {len(existing_complexes)} rows")

    # Drop any pre-existing 'Ribosome' so Kuhner's version wins
    existing_complexes = existing_complexes[existing_complexes['Name'].astype(str) != 'Ribosome']
    merged = pd.concat([existing_complexes, complexes_df], ignore_index=True)
    print(f"Merged Complexes sheet:   {len(merged)} rows")

    # Read all other sheets to preserve them
    other_sheets = {}
    xl = pd.ExcelFile(OUT)
    for sh in xl.sheet_names:
        if sh != 'Complexes':
            other_sheets[sh] = pd.read_excel(xl, sh)

    # Write back
    with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
        merged.to_excel(writer, sheet_name='Complexes', index=False)
        for sh, df in other_sheets.items():
            df.to_excel(writer, sheet_name=sh, index=False)
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    write_complex_formation_xlsx()
