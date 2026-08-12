# Dark genes: how far out do they reach?

The 235 genes in `highlight_ancient_dark_genes.tsv` have no curated human function but a characterized fly/worm/yeast ortholog. Neither ortholog source used upstream can see past those three species, so this is a direct sequence search against 26 species that appear in neither: bacteria, archaea, plants, algae, protists, and the basal metazoans and invertebrate chordates that NCBI's vertebrate-only set skips.

| quantity | value |
|:---|---:|
| dark genes searched | 234 |
| panel species | 26 |
| panel proteins searched | 283,294 |
| genes with ≥1 homolog in the panel | 231 |
| genes with no hit anywhere | 3 |
| **genes reaching bacteria or archaea** | **94** |
| ↳ surviving the reciprocal-best-hit test | 7 |
| genes reaching bacteria | 75 |
| genes reaching archaea | 56 |
| total hit rows | 68,182 |

## Deepest domain reached

| deepest group | genes | share |
|:---|---:|---:|
| Bacteria | 75 | 32.1% |
| Excavata (Giardia, Trypanosoma) | 42 | 17.9% |
| Amoebozoa (Dictyostelium) | 31 | 13.2% |
| Archaea | 19 | 8.1% |
| Sponges (Amphimedon) | 15 | 6.4% |
| Plants (Arabidopsis) | 12 | 5.1% |
| Choanoflagellates (Monosiga) | 8 | 3.4% |
| Apicomplexa (Plasmodium) | 7 | 3.0% |
| Fission yeast (S. pombe) | 7 | 3.0% |
| Green algae (Chlamydomonas) | 5 | 2.1% |
| Echinoderms (sea urchin) | 4 | 1.7% |
| Cnidaria (Nematostella) | 3 | 1.3% |
| no hit anywhere in the panel | 3 | 1.3% |
| Cephalochordates (amphioxus) | 2 | 0.9% |
| Placozoa (Trichoplax) | 1 | 0.4% |

## Genes that reach bacteria or archaea

`RBH` marks a reciprocal best hit: searching the prokaryotic protein back against the reviewed human proteome returns this same gene on top. The `reverse hit` column shows what it returns instead when it is not this gene — and that column is the most informative one in the table.

For 87 of the 94 genes, the reverse search lands on a *different human gene*, and it is almost always the query's better-studied paralog: ABCF2-H2BK1's bacterial hit maps back to ABCF1, ATAD3C's maps to VCP, ATP13A5's to ATP2C1, MGAM2's to MGAM. The prokaryotic homology is real and often overwhelming (E-values to 1e-206), but the human family member that best represents it is not the dark gene. Read that as *the family is ancient, and this gene is a young duplicate of it* — not as evidence against the homology. Only 3 genes have their best prokaryotic hit come back as a reciprocal best hit; 7 have at least one RBH hit somewhere in their hit list. For 4 genes the reverse search found no human hit at all, which means the match exists only at profile level and is the weakest evidence in the table.

| human gene | best prokaryotic hit | species | E-value | found by | RBH | reverse hit |
|:---|:---|:---|:---|:---|:---|:---|
| ATAD3C | Cell division cycle protein 48 homolog MJ11… | Methanocaldococcus jann… | 6.49e-234 | profile | — | VCP |
| ABCF2-H2BK1 | Probable ATP-binding protein YheS | Escherichia coli K-12 | 1.26e-206 | profile | — | ABCF3 |
| PTGES3L-AARSD1 | Alanine--tRNA ligase | Synechocystis sp. PCC 6… | 1.76e-181 | profile | — | AARS1 |
| ATP13A5 | Calcium-transporting ATPase | Bacillus subtilis 168 | 1.19e-175 | profile | — | ATP2C1 |
| SERPINA11 | Uncharacterized serpin-like protein TK1782 | Thermococcus kodakarens… | 7.87e-129 | profile | — | SERPINB3 |
| SLC6A16 | Uncharacterized sodium-dependent transporte… | Methanocaldococcus jann… | 5.35e-113 | profile | — | SLC6A11 |
| MGAM2 | Alpha-glucosidase | Thermus thermophilus HB8 | 9.12e-110 | profile | — | MGAM |
| WDR89 | Uncharacterized WD repeat-containing protei… | Synechocystis sp. PCC 6… | 3.97e-109 | profile | — | DAW1 |
| DCAF12L2 | Uncharacterized WD repeat-containing protei… | Synechocystis sp. PCC 6… | 2.59e-99 | profile | — | DAW1 |
| TLE7 | Uncharacterized WD repeat-containing protei… | Synechocystis sp. PCC 6… | 1.70e-93 | profile | — | DAW1 |
| SDR42E1 | 3 beta-hydroxysteroid dehydrogenase/Delta 5… | Mycobacterium tuberculo… | 2.39e-74 | profile | **yes** | SDR42E1 |
| NME1-NME2 | Nucleoside diphosphate kinase | Synechocystis sp. PCC 6… | 5.07e-73 | profile | — | NME3 |
| ASAH2B | Neutral ceramidase | Mycobacterium tuberculo… | 6.08e-59 | profile | — | ASAH2 |
| RPS4Y2 | Small ribosomal subunit protein eS4 | Thermococcus kodakarens… | 1.76e-57 | profile | — | RPS4Y1 |
| DIP2C | 4-hydroxyphenylalkanoate adenylyltransferase | Mycobacterium tuberculo… | 9.92e-57 | profile | **yes** | DIP2B |
| GPX6 | Hydroperoxy fatty acid reductase gpx1 | Synechocystis sp. PCC 6… | 1.43e-53 | profile | — | GPX4 |
| MSANTD1 | Amidase | Saccharolobus solfatari… | 2.14e-53 | profile | — | QRSL1 |
| SBK2 | Serine/threonine-protein kinase PrkC | Bacillus subtilis 168 | 7.27e-48 | profile | — | RPS6KA5 |
| DCLK3 | non-specific serine/threonine protein kinase | Thermus thermophilus HB8 | 1.66e-47 | profile | — | RPS6KA3 |
| DNAJC25-GNG10 | Chaperone protein DnaJ 1 | Synechocystis sp. PCC 6… | 5.26e-44 | profile | — | DNAJA3 |
| TANGO6 | cysteine desulfurase | Synechocystis sp. PCC 6… | 1.41e-43 | profile | — | NFS1 |
| PSD2 | PROTEIN TRANSPORT PROTEIN SEC7 (Sec7) | Rickettsia prowazekii | 1.55e-41 | profile | — | ARFGEF2 |
| CPA5 | Sll0236 protein | Synechocystis sp. PCC 6… | 1.53e-40 | profile | — | CPD |
| MCTS2 | Uncharacterized protein MJ1432 | Methanocaldococcus jann… | 3.03e-31 | profile | **yes** | MCTS1 |
| SLC22A31 | Transport protein | Thermus thermophilus HB8 | 3.53e-31 | profile | — | SVOP |
| ANKHD1-EIF4EBP3 | Erythroid ankyrin | Synechocystis sp. PCC 6… | 7.17e-30 | phmmer | — | ANK2 |
| CDKL4 | Serine/threonine-protein kinase PrkC | Bacillus subtilis 168 | 1.10e-27 | profile | — | RPS6KA5 |
| DNAJC25 | DnAJ-like protein slr0093 | Synechocystis sp. PCC 6… | 8.60e-27 | profile | — | DNAJB4 |
| FAM86B1 | Putative branched-chain-amino-acid aminotra… | Methanocaldococcus jann… | 9.30e-27 | profile | — | BCAT1 |
| DNAJC5G | DnAJ-like protein slr0093 | Synechocystis sp. PCC 6… | 1.04e-26 | profile | — | DNAJB4 |
| SATL1 | PPE family protein PPE8 | Mycobacterium tuberculo… | 4.44e-25 | profile | — | QRICH2 |
| MAST4 | Serine/threonine-protein kinase PknA | Mycobacterium tuberculo… | 7.16e-24 | profile | — | RPS6KA5 |
| AMMECR1L | Protein MJ0810 | Methanocaldococcus jann… | 2.76e-22 | profile | **yes** | AMMECR1L |
| WDR7 | Uncharacterized WD repeat-containing protei… | Synechocystis sp. PCC 6… | 2.88e-21 | profile | — | POC1B |
| NWD2 | Uncharacterized WD repeat-containing protei… | Synechocystis sp. PCC 6… | 2.39e-20 | profile | — | DAW1 |
| IGLON5 | Serine/threonine-protein kinase G | Synechocystis sp. PCC 6… | 1.44e-19 | profile | — | RPS6KA2 |
| ACYP1 | Acylphosphatase | Thermococcus kodakarens… | 1.57e-19 | profile | **yes** | ACYP2 |
| TMEM132C | Fat protein | Synechocystis sp. PCC 6… | 7.20e-19 | profile | — | FAT4 |
| TPTEP2-CSNK1E | Serine/threonine-protein kinase PrkC | Bacillus subtilis 168 | 3.32e-18 | profile | — | RPS6KA5 |
| TTC14 | Sll0910 protein | Synechocystis sp. PCC 6… | 2.36e-17 | profile | — | TTC6 |
| ERI2 | Probable 3'-5' exonuclease KapD | Bacillus subtilis 168 | 4.25e-17 | profile | — | ERI1 |
| DMXL1 | Uncharacterized WD repeat-containing protei… | Synechocystis sp. PCC 6… | 1.50e-16 | profile | — | POC1B |
| TTC39C | TPR repeat-containing protein MJ1345 | Methanocaldococcus jann… | 1.86e-15 | profile | — | OGT |
| GAB4 | Ribosomal RNA small subunit methyltransfera… | Escherichia coli K-12 | 6.47e-14 | profile | — | NOP2 |
| LRRN2 | Leucine-rich repeat domain-containing prote… | Escherichia coli K-12 | 1.67e-13 | profile | — | LGR6 |
| TRIM67 | Kelch domain-containing protein SSO1033 | Saccharolobus solfatari… | 4.19e-13 | profile | — | TTN |
| HNRNPCL4 | Putative RNA-binding protein RbpA | Synechocystis sp. PCC 6… | 1.07e-12 | profile | — | PABPC1 |
| TPBGL | Leucine-rich repeat domain-containing prote… | Escherichia coli K-12 | 4.60e-12 | profile | — | LGR6 |
| KLF18 | Serine/threonine-protein kinase PrkC | Bacillus subtilis 168 | 9.26e-12 | profile | — | RPS6KA5 |
| PLCXD2 | Protein sll1483 | Synechocystis sp. PCC 6… | 1.43e-11 | profile | — | TGFBI |
| PPM1N | Protein phosphatase PrpC | Bacillus subtilis 168 | 8.62e-11 | profile | **yes** | PPM1K |
| ADAMTS16 | Sll0499 protein | Synechocystis sp. PCC 6… | 4.00e-10 | profile | — | TTC28 |
| RGPD3 | TPR repeat-containing protein MJ0940 | Methanocaldococcus jann… | 4.40e-10 | profile | — | OGT |
| ADAMTS6 | Sll0499 protein | Synechocystis sp. PCC 6… | 5.67e-10 | profile | — | TTC28 |
| RGPD4 | TPR repeat-containing protein MJ0940 | Methanocaldococcus jann… | 7.57e-10 | profile | — | OGT |
| PLPPR3 | Probable conserved integral membrane protein | Mycobacterium tuberculo… | 9.13e-10 | profile | — | PLPP7 |
| RGPD2 | TPR repeat-containing protein MJ0940 | Methanocaldococcus jann… | 1.68e-09 | profile | — | OGT |
| RGPD1 | TPR repeat-containing protein MJ0940 | Methanocaldococcus jann… | 1.70e-09 | profile | — | OGT |
| RGPD8 | TPR repeat-containing protein MJ0940 | Methanocaldococcus jann… | 1.88e-09 | profile | — | OGT |
| LRRC24 | Leucine-rich repeat domain-containing prote… | Escherichia coli K-12 | 1.97e-09 | profile | — | LGR6 |
| SLC36A3 | Aromatic amino acid permease | Thermococcus kodakarens… | 5.95e-09 | profile | — | SLC38A1 |
| LRIT2 | Leucine-rich repeat domain-containing prote… | Escherichia coli K-12 | 1.55e-08 | profile | — | LGR6 |
| CCDC186 | S-layer protein B | Saccharolobus solfatari… | 2.74e-07 | profile | — | MYH2 |
| UAP1L1 | Ribosome maturation factor RimM | Synechocystis sp. PCC 6… | 2.75e-07 | profile | — | — |
| GOLGA8N | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 3.30e-07 | profile | — | APOA4 |
| GOLGA8H | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 5.12e-07 | profile | — | APOA4 |
| PPM1M | PPM-type phosphatase domain-containing prot… | Thermus thermophilus HB8 | 5.32e-07 | profile | — | PPM1N |
| GOLGA8K | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 7.24e-07 | profile | — | APOA4 |
| GOLGA8T | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 7.75e-07 | profile | — | APOA4 |
| GOLGA8J | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 1.06e-06 | profile | — | APOA4 |
| GOLGA8M | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 1.11e-06 | profile | — | APOA4 |
| GOLGA8Q | Apolipoprotein A1/A4/E | Thermus thermophilus HB8 | 6.09e-06 | profile | — | APOA4 |
| METTL25B | Probable trans-aconitate 2-methyltransferase | Mycobacterium tuberculo… | 1.35e-05 | profile | **yes** | METTL25B |
| ARL13A | Probable macrolide-transport ATP-binding pr… | Mycobacterium tuberculo… | 2.64e-05 | profile | — | ABCF1 |
| C6orf136 | Sll0364 protein | Synechocystis sp. PCC 6… | 3.37e-05 | profile | — | — |
| TMEM245 | Hypothetical membrane protein, conserved, D… | Thermococcus kodakarens… | 3.57e-05 | profile | — | HLA-DPB1 |
| TIGD2 | Second ORF in transposon ISC1395 | Saccharolobus solfatari… | 4.19e-05 | profile | — | SETMAR |
| LRRC3C | Leucine-rich repeat domain-containing prote… | Escherichia coli K-12 | 8.45e-05 | phmmer | — | LGR6 |
| CTAGE1 | ATP SYNTHASE B CHAIN (AtpX) | Rickettsia prowazekii | 8.82e-05 | phmmer | — | — |
| MYO1H | Replication factor C large subunit | Methanocaldococcus jann… | 9.26e-05 | profile | — | RFC1 |
| HDGFL1 | 2-hydroxy-3-oxopropionate reductase | Escherichia coli K-12 | 1.73e-04 | profile | — | HIBADH |
| CDV3 | Conserved hypothetical transmembrane protein | Mycoplasma mycoides SC… | 3.15e-04 | phmmer | — | SHOX2 |
| PAPOLB | Polymerase nucleotidyl transferase domain-c… | Saccharolobus solfatari… | 3.19e-04 | phmmer | — | PAPOLA |
| TEX261 | Putative NADH-ubiquinone oxidoreductase MJ0… | Methanocaldococcus jann… | 3.61e-04 | phmmer | — | MT-ND1 |
| JAKMIP3 | Sll1424 protein | Synechocystis sp. PCC 6… | 3.76e-04 | profile | — | KANK4 |
| DISP2 | Inner membrane protein YabI | Escherichia coli K-12 | 5.11e-04 | phmmer | — | — |
| YIPF7 | Uncharacterized protein YebC | Bacillus subtilis 168 | 5.58e-04 | profile | — | SLC50A1 |
| GOLGA6D | Prespore-specific transcriptional regulator… | Bacillus subtilis 168 | 7.58e-04 | phmmer | — | MAFA |
| CCDC149 | Peptidoglycan DL-endopeptidase CwlO | Bacillus subtilis 168 | 7.60e-04 | profile | — | SHQ1 |
| CTAGE6 | DUF3782 domain-containing protein | Saccharolobus solfatari… | 7.99e-04 | profile | — | C17orf100 |
| GOLGA6C | Prespore-specific transcriptional regulator… | Bacillus subtilis 168 | 8.06e-04 | phmmer | — | MAFA |
| GOLGA6B | Prespore-specific transcriptional regulator… | Bacillus subtilis 168 | 8.34e-04 | phmmer | — | MAFA |
| SMTNL2 | Murein hydrolase activator EnvC | Escherichia coli K-12 | 8.54e-04 | profile | — | TEX12 |
| LIX1L | Uncharacterized protein RP084 | Rickettsia prowazekii | 9.43e-04 | phmmer | — | MICAL2 |

## Prokaryotic coverage by species

| species | dark genes with a homolog |
|:---|---:|
| Synechocystis sp. PCC 6803 | 47 |
| Thermus thermophilus HB8 | 41 |
| Bacillus subtilis 168 | 38 |
| Saccharolobus solfataricus P2 | 35 |
| Mycobacterium tuberculosis H37Rv | 31 |
| Escherichia coli K-12 | 28 |
| Thermococcus kodakarensis | 26 |
| Halobacterium salinarum NRC-1 | 25 |
| Methanocaldococcus jannaschii | 22 |
| Mycoplasma mycoides SC PG1 | 19 |
| Mycoplasmoides genitalium G37 | 18 |
| Rickettsia prowazekii | 17 |

Method attribution, counted on the 1,397 distinct gene–protein pairs rather than on hit rows (the two passes overlap, so summing rows double-counts): 910 pairs were found by both passes, 448 by the profile search alone, and 39 by direct search alone. At gene level, 38 of the 94 genes that reach prokaryotes would have been missed entirely without the profile stage — which is the point of building one.

## Caveats

- **Homology is not orthology.** Most prokaryotic hits here are the human protein landing in an ancient, widely-shared family (P-loop NTPases, ABC transporter ATPase domains, Rossmann folds). Rows without an RBH should be read as "this gene contains an ancient domain", not "this gene has a bacterial ortholog".
- **But RBH is conservative in exactly the wrong direction here.** These are dark genes, and a large share of them are recent duplicates or readthrough products of well-studied human genes. When a bacterial protein's best human match is the parent rather than the duplicate, RBH fails even though the ancestry is genuine. A failed RBH on a young paralog is not evidence against ancient origin; it is evidence that the query is not the family's representative member. The 7 RBH genes are the safest calls, not the complete set of true ones.
- **Multidomain proteins can pass RBH on one domain.** DIP2C is 1,500+ residues and its hit is to an adenylyltransferase; the reciprocal verdict reflects that domain, not the whole protein. Check `query_coverage` in the hits file before reading a whole-protein claim into a domain-level match.
- **A profile search trades specificity for reach.** Profiles built from a gene's eukaryotic homologs find real remote homology that single-sequence search misses, and they also drift toward generic domain models when the seed set is large and divergent. Seed counts are in the `profile_seeds` column of `dark_gene_deep_homology.tsv`.
- **The panel is a sample, not a census.** 8 bacteria and 4 archaea stand in for two entire domains of life. A gene with no hit here may still have homologs in lineages not sampled; absence of evidence is weak evidence at this depth.
- **Eukaryotic parasites have reduced genomes.** Giardia, Plasmodium and Trypanosoma have lost many ancestral genes, so a missing hit in those species reflects their biology as much as the human gene's age.
