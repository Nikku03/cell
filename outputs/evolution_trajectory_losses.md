# Evolutionary trajectory — losses side

**Reference:** `beril_Methanococcus_JJ` — 1521 OGs, 354 essential

At each step beyond the reference we ask three questions:

1. **Family losses** — OGs present in the previous step that are absent in this organism (gene family discarded)
2. **Lost essentials** — those family losses that WERE essential in the previous organism (function abandoned)
3. **Essentiality losses** — OG still present, was essential before but is no longer (function retained, requirement gone)

---

## Step 1: `beril_Methanococcus_S2` ← `beril_Methanococcus_JJ`  (J=0.970)

**vs previous step:**
- Family losses (OG discarded): **14**, of which **4** were essential in `beril_Methanococcus_JJ`
- Essentiality losses (OG kept, no longer essential): **57**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **14**, of which **4** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_Methanococcus_JJ`)

| word | count |
|---|---|
| `restriction` | 2 |
| `endonuclease` | 2 |
| `class` | 1 |
| `sam` | 1 |
| `dependent` | 1 |
| `dna` | 1 |
| `methyltransferase` | 1 |
| `duf5655` | 1 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `synthase` | 6 |
| `abc` | 5 |
| `transporter` | 5 |
| `substrate` | 5 |
| `factor` | 4 |
| `helix` | 4 |
| `recombinase` | 4 |
| `dependent` | 3 |
| `complex` | 3 |
| `hydrogenase` | 3 |
| `tyrosine` | 3 |
| `integrase` | 3 |
| `phosphate` | 3 |
| `elongation` | 2 |
| `50s` | 2 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| endA | tRNA-intron lyase | MMJJ_RS08505 |
| porA | pyruvate synthase subunit PorA | MMJJ_RS06620 |
| vhuA | F420-non-reducing hydrogenase Vhu subunit A | MMJJ_RS05635 |
| selB | selenocysteine-specific translation elongation factor | MMJJ_RS07460 |
| lonB | ATP-dependent protease LonB | MMJJ_RS08245 |
| ilvC | ketol-acid reductoisomerase | MMJJ_RS01620 |
| tpiA | triose-phosphate isomerase | MMJJ_RS01455 |
| hisE | phosphoribosyl-ATP diphosphatase | MMJJ_RS05180 |
| ilvD | dihydroxy-acid dehydratase | MMJJ_RS03675 |
| hisB | imidazoleglycerol-phosphate dehydratase HisB | MMJJ_RS02175 |

---

## Step 2: `beril_Magneto` ← `beril_Methanococcus_S2`  (J=0.101)

**vs previous step:**
- Family losses (OG discarded): **1210**, of which **348** were essential in `beril_Methanococcus_S2`
- Essentiality losses (OG kept, no longer essential): **16**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **1195**, of which **252** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_Methanococcus_S2`)

| word | count |
|---|---|
| `ribosomal` | 45 |
| `synthase` | 34 |
| `50s` | 26 |
| `trna` | 21 |
| `dna` | 19 |
| `30s` | 17 |
| `kinase` | 15 |
| `polymerase` | 14 |
| `ligase` | 14 |
| `factor` | 13 |
| `reductase` | 12 |
| `atp` | 12 |
| `dehydrogenase` | 12 |
| `methyltransferase` | 10 |
| `phosphate` | 10 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|
| mvk | mevalonate kinase | MMP_RS06890 |
| cfbE | coenzyme F430 synthase | MMP_RS00970 |
| thiI | tRNA uracil 4-sulfurtransferase ThiI | MMP_RS06985 |
| hypF | carbamoyltransferase HypF | MMP_RS00805 |
| polC | DNA polymerase II large subunit | MMP_RS00160 |
| guaA | glutamine-hydrolyzing GMP synthase | MMP_RS04645 |
| speD | adenosylmethionine decarboxylase | MMP_RS08140 |
| map | type II methionyl aminopeptidase | MMP_RS07430 |
| eif1A | translation initiation factor eIF-1A | MMP_RS03175 |
| cmk | (d)CMP kinase | MMP_RS03290 |

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `iron` | 3 |
| `hydrogenase` | 3 |
| `formation` | 2 |
| `nitrogenase` | 2 |
| `synthase` | 2 |
| `ferrous` | 1 |
| `transport` | 1 |
| `fructose` | 1 |
| `phosphate` | 1 |
| `aldolase` | 1 |
| `hypd` | 1 |
| `nickel` | 1 |
| `incorporation` | 1 |
| `hypb` | 1 |
| `agmatinase` | 1 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| feoB | ferrous iron transport protein B | MMP_RS03310 |
| fsa | fructose-6-phosphate aldolase | MMP_RS06740 |
| hypD | hydrogenase formation protein HypD | MMP_RS01555 |
| hypB | hydrogenase nickel incorporation protein HypB | MMP_RS07810 |
| speB | agmatinase | MMP_RS08150 |
| nifH | nitrogenase iron protein | MMP_RS00840 |
| nifH | nitrogenase iron protein | MMP_RS04440 |
| dcd | dCTP deaminase | MMP_RS07340 |
| ppsA | phosphoenolpyruvate synthase | MMP_RS05655 |
| hypE | hydrogenase expression/formation protein HypE | MMP_RS01480 |

---

## Step 3: `beril_HerbieS` ← `beril_Magneto`  (J=0.331)

**vs previous step:**
- Family losses (OG discarded): **912**, of which **228** were essential in `beril_Magneto`
- Essentiality losses (OG kept, no longer essential): **220**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **1183**, of which **251** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_Magneto`)

| word | count |
|---|---|
| `synthase` | 13 |
| `transporter` | 12 |
| `regulator` | 10 |
| `atp` | 8 |
| `helix` | 8 |
| `glycosyltransferase` | 8 |
| `dependent` | 7 |
| `methyltransferase` | 7 |
| `cytochrome` | 7 |
| `beta` | 7 |
| `cobalt` | 7 |
| `transcriptional` | 7 |
| `sam` | 6 |
| `abc` | 6 |
| `phosphate` | 6 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|
| ftsZ | cell division protein FtsZ | AMB_RS19505 |
| pseG | UDP-2%2C4-diacetamido-2%2C4%2C6-trideoxy-beta-L-altropyranose hydrolase | AMB_RS03665 |
| ccmB | heme exporter protein CcmB | AMB_RS21150 |
| tusB | sulfurtransferase complex subunit TusB | AMB_RS17055 |
| rnd | ribonuclease D | AMB_RS14275 |
| ykgO | type B 50S ribosomal protein L36 | AMB_RS11600 |
| cbiM | cobalt transporter CbiM | AMB_RS21930 |
| cobJ | precorrin-3B C(17)-methyltransferase | AMB_RS01515 |
| ccmA | heme ABC exporter ATP-binding protein CcmA | AMB_RS21155 |
| purQ | phosphoribosylformylglycinamidine synthase subunit PurQ | AMB_RS10665 |

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `transporter` | 51 |
| `abc` | 41 |
| `synthase` | 37 |
| `coa` | 20 |
| `phosphate` | 19 |
| `dependent` | 17 |
| `regulator` | 16 |
| `permease` | 16 |
| `atp` | 14 |
| `substrate` | 14 |
| `dna` | 10 |
| `dehydrogenase` | 10 |
| `isomerase` | 9 |
| `acid` | 9 |
| `methyltransferase` | 9 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| nhaA | Na+/H+ antiporter NhaA | AMB_RS12285 |
| asnB | asparagine synthase (glutamine-hydrolyzing) | AMB_RS20340 |
| asnB | asparagine synthase (glutamine-hydrolyzing) | AMB_RS00215 |
| asnB | asparagine synthase (glutamine-hydrolyzing) | AMB_RS06425 |
| asnB | asparagine synthase (glutamine-hydrolyzing) | AMB_RS00530 |
| asnB | asparagine synthase (glutamine-hydrolyzing) | AMB_RS00600 |
| grxC | glutaredoxin 3 | AMB_RS08110 |
| recJ | single-stranded-DNA-specific exonuclease RecJ | AMB_RS18875 |
| leuC | 3-isopropylmalate dehydratase large subunit | AMB_RS20560 |
| cobT | nicotinate-nucleotide--dimethylbenzimidazole phosphoribosyltransferase | AMB_RS02805 |

---

## Step 4: `beril_RalstoniaPSI07` ← `beril_HerbieS`  (J=0.447)

**vs previous step:**
- Family losses (OG discarded): **753**, of which **57** were essential in `beril_HerbieS`
- Essentiality losses (OG kept, no longer essential): **152**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **1181**, of which **250** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_HerbieS`)

| word | count |
|---|---|
| `helix` | 10 |
| `toxin` | 7 |
| `antitoxin` | 7 |
| `transcriptional` | 6 |
| `regulator` | 6 |
| `turn` | 5 |
| `system` | 4 |
| `transporter` | 4 |
| `isomerase` | 3 |
| `abc` | 3 |
| `beta` | 2 |
| `synthase` | 2 |
| `chain` | 2 |
| `factor` | 2 |
| `higa` | 2 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|
| prfB | peptide chain release factor 2 | HSERO_RS13990 |
| ribA | GTP cyclohydrolase II | HSERO_RS23850 |
| ylqF | ribosome biogenesis GTPase YlqF | HSERO_RS04965 |
| folE | GTP cyclohydrolase I | HSERO_RS23745 |
| ybeY | rRNA maturation RNase YbeY | HSERO_RS03605 |
| ftsL | cell division protein FtsL | HSERO_RS01630 |
| nifT | putative nitrogen fixation protein NifT | HSERO_RS14345 |

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `regulator` | 149 |
| `transcriptional` | 147 |
| `lysr` | 95 |
| `transporter` | 76 |
| `abc` | 61 |
| `permease` | 41 |
| `amino` | 33 |
| `acid` | 33 |
| `atp` | 30 |
| `gntr` | 21 |
| `membrane` | 20 |
| `substrate` | 19 |
| `outer` | 18 |
| `branched` | 18 |
| `chain` | 18 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| sucC | ADP-forming succinate--CoA ligase subunit beta | HSERO_RS03280 |
| rlmN | dual-specificity RNA methyltransferase RlmN | HSERO_RS14810 |
| zapD | cell division protein ZapD | HSERO_RS01750 |
| pnp | polyribonucleotide nucleotidyltransferase | HSERO_RS08755 |
| lolB | lipoprotein insertase outer membrane protein LolB | HSERO_RS19555 |
| gltK | glutamate/aspartate ABC transporter permease GltK | HSERO_RS17560 |
| glnP | glutamine ABC transporter permease GlnP | HSERO_RS17545 |
| mlaE | lipid asymmetry maintenance ABC transporter permease subunit MlaE | HSERO_RS20395 |
| lplT | lysophospholipid transporter LplT | HSERO_RS10735 |
| pxpA | 5-oxoprolinase subunit PxpA | HSERO_RS08780 |

---

## Step 5: `beril_RalstoniaGMI1000` ← `beril_RalstoniaPSI07`  (J=0.827)

**vs previous step:**
- Family losses (OG discarded): **271**, of which **18** were essential in `beril_RalstoniaPSI07`
- Essentiality losses (OG kept, no longer essential): **111**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **1180**, of which **251** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_RalstoniaPSI07`)

| word | count |
|---|---|
| `chemotaxis` | 2 |
| `chew` | 2 |
| `chromosome` | 1 |
| `segregation` | 1 |
| `smc` | 1 |
| `phosphate` | 1 |
| `acetyltransferase` | 1 |
| `atp` | 1 |
| `phosphoribosyltransferase` | 1 |
| `beta` | 1 |
| `2fe` | 1 |
| `ybbc` | 1 |
| `yhhh` | 1 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|
| smc | chromosome segregation protein SMC | RPSI07_RS17140 |
| hisG | ATP phosphoribosyltransferase | RPSI07_RS10385 |

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `transporter` | 20 |
| `efflux` | 18 |
| `rnd` | 17 |
| `periplasmic` | 17 |
| `adaptor` | 17 |
| `synthase` | 9 |
| `dna` | 5 |
| `iii` | 5 |
| `regulator` | 5 |
| `cytochrome` | 4 |
| `oxidase` | 4 |
| `ycei` | 4 |
| `hydratase` | 4 |
| `iron` | 4 |
| `dependent` | 4 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| trpD | anthranilate phosphoribosyltransferase | RPSI07_RS02570 |
| trpD | anthranilate phosphoribosyltransferase | RPSI07_RS10740 |
| trpE | anthranilate synthase component I | RPSI07_RS10755 |
| aroG | 3-deoxy-7-phosphoheptulonate synthase AroG | RPSI07_RS11845 |
| rdgB | RdgB/HAM1 family non-canonical purine NTP pyrophosphatase | RPSI07_RS13835 |
| recX | recombination regulator RecX | RPSI07_RS21255 |
| gmhB | D-glycero-beta-D-manno-heptose 1%2C7-bisphosphate 7-phosphatase | RPSI07_RS21435 |
| def | peptide deformylase | RPSI07_RS17125 |
| apbC | iron-sulfur cluster carrier protein ApbC | RPSI07_RS12975 |
| crcB | fluoride efflux transporter CrcB | RPSI07_RS17460 |

---

## Step 6: `beril_RalstoniaBSBF1503` ← `beril_RalstoniaGMI1000`  (J=0.768)

**vs previous step:**
- Family losses (OG discarded): **376**, of which **58** were essential in `beril_RalstoniaGMI1000`
- Essentiality losses (OG kept, no longer essential): **174**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **1180**, of which **250** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_RalstoniaGMI1000`)

| word | count |
|---|---|
| `like` | 8 |
| `transposase` | 8 |
| `element` | 7 |
| `acyl` | 3 |
| `carrier` | 3 |
| `antitoxin` | 2 |
| `system` | 2 |
| `coa` | 2 |
| `helix` | 2 |
| `is630` | 2 |
| `isrso5` | 2 |
| `integrase` | 2 |
| `phage` | 2 |
| `is5` | 2 |
| `conjugative` | 1 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|
| trbJ | P-type conjugative transfer protein TrbJ | RS_RS12935 |

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `regulator` | 57 |
| `transporter` | 54 |
| `transcriptional` | 37 |
| `response` | 23 |
| `abc` | 21 |
| `helix` | 18 |
| `amino` | 16 |
| `acid` | 16 |
| `factor` | 16 |
| `permease` | 15 |
| `system` | 14 |
| `substrate` | 14 |
| `dna` | 13 |
| `dmt` | 13 |
| `secretion` | 12 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| cueR | Cu(I)-responsive transcriptional regulator | RS_RS16770 |
| cadR | Cd(II)/Pb(II)-responsive transcriptional regulator | RS_RS18685 |
| cobT | nicotinate-nucleotide--dimethylbenzimidazole phosphoribosyltransferase | RS_RS12035 |
| ehuD | ectoine/hydroxyectoine ABC transporter permease subunit EhuD | RS_RS17085 |
| gltK | glutamate/aspartate ABC transporter permease GltK | RS_RS02395 |
| rpmG | 50S ribosomal protein L33 | RS_RS12280 |
| gspG | type II secretion system major pseudopilin GspG | RS_RS11590 |
| gspG | type II secretion system major pseudopilin GspG | RS_RS17870 |
| gspG | type II secretion system major pseudopilin GspG | RS_RS19405 |
| gspG | type II secretion system major pseudopilin GspG | RS_RS15600 |

---

## Step 7: `beril_Dda3937` ← `beril_RalstoniaBSBF1503`  (J=0.264)

**vs previous step:**
- Family losses (OG discarded): **2161**, of which **356** were essential in `beril_RalstoniaBSBF1503`
- Essentiality losses (OG kept, no longer essential): **212**

**vs reference (`beril_Methanococcus_JJ`):**
- Family losses since reference: **1174**, of which **252** were essential in `beril_Methanococcus_JJ`

### Functions abandoned (family GONE, was essential in `beril_RalstoniaBSBF1503`)

| word | count |
|---|---|
| `helix` | 18 |
| `factor` | 11 |
| `like` | 10 |
| `is21` | 8 |
| `regulator` | 8 |
| `porin` | 7 |
| `turn` | 7 |
| `membrane` | 6 |
| `phosphate` | 6 |
| `synthase` | 6 |
| `cytochrome` | 5 |
| `atp` | 5 |
| `assembly` | 5 |
| `toxin` | 5 |
| `antitoxin` | 5 |

#### Named examples

| gene | was-essential product | locus_tag in prev |
|---|---|---|
| gshA | glutamate--cysteine ligase | RALBFv3_RS10855 |
| istA | IS21 family transposase | RALBFv3_RS01925 |
| ubiD | 4-hydroxy-3-polyprenylbenzoate decarboxylase | RALBFv3_RS12790 |
| ompA | outer membrane protein OmpA | RALBFv3_RS13740 |
| gatB | Asp-tRNA(Asn)/Glu-tRNA(Gln) amidotransferase subunit GatB | RALBFv3_RS09685 |
| tolA | cell envelope integrity protein TolA | RALBFv3_RS12920 |
| petA | ubiquinol-cytochrome c reductase iron-sulfur subunit | RALBFv3_RS06730 |
| ybeY | rRNA maturation RNase YbeY | RALBFv3_RS11815 |
| ilvC | ketol-acid reductoisomerase | RALBFv3_RS03040 |
| scpB | SMC-Scp complex subunit ScpB | RALBFv3_RS15660 |

### Functions retained but no longer required (family kept, essential→non-essential)

| word | count |
|---|---|
| `transporter` | 52 |
| `regulator` | 45 |
| `abc` | 38 |
| `atp` | 35 |
| `transcriptional` | 34 |
| `helix` | 34 |
| `dehydrogenase` | 28 |
| `synthase` | 27 |
| `dependent` | 25 |
| `factor` | 19 |
| `phosphate` | 18 |
| `turn` | 17 |
| `winged` | 15 |
| `sigma` | 14 |
| `coa` | 12 |

#### Named examples

| gene | product | locus_tag in prev |
|---|---|---|
| plsY | glycerol-3-phosphate 1-O-acyltransferase PlsY | RALBFv3_RS05150 |
| argG | argininosuccinate synthase | RALBFv3_RS05180 |
| rlmN | 23S rRNA (adenine(2503)-C(2))-methyltransferase RlmN | RALBFv3_RS15260 |
| acpS | holo-ACP synthase | RALBFv3_RS14505 |
| rpoE | RNA polymerase sigma factor RpoE | RALBFv3_RS14445 |
| sdhD | succinate dehydrogenase%2C hydrophobic membrane anchor protein | RALBFv3_RS02645 |
| mlaE | lipid asymmetry maintenance ABC transporter permease subunit MlaE | RALBFv3_RS06895 |
| rnhA | ribonuclease HI | RALBFv3_RS02225 |
| rng | ribonuclease G | RALBFv3_RS03575 |
| rimI | ribosomal protein S18-alanine N-acetyltransferase | RALBFv3_RS00190 |
