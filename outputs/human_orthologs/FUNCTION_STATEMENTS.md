# From ortholog evidence to a function statement

`REPORT.md` establishes which human genes have orthologs and what GO terms those orthologs carry. A GO term list is evidence, not an answer. This layer fetches the curated UniProt FUNCTION text of the orthologs themselves and assigns every human gene the best statement available, tagged with where it came from.

| evidence tier | genes | meaning |
|:---|---:|:---|
| human_curated | 16,779 | the human gene already has curated function text — nothing to transfer |
| ortholog_curated | 455 | statement taken from a model-organism ortholog's UniProt record |
| ortholog_experimental | 489 | no ortholog function text, but ortholog GO terms with experimental evidence |
| none | 2,899 | no usable evidence anywhere |

**3,843 human genes have no curated function text of their own.** Orthologs supply a statement for **1,004** of them (455 as curated prose, 489 as experimental GO terms). 2,839 genes still have nothing.

## Which species actually rescues a gene

| transfer distance | genes | read the statement how |
|:---|---:|:---|
| high_mammal (mouse, rat) | 38 | safe to use nearly verbatim |
| medium_vertebrate (zebrafish, Xenopus) | 9 | molecular function transfers, tissue context may not |
| low_invertebrate (fly, worm) | 240 | molecular half transfers; organismal half is that animal's biology |
| low_fungal (yeast) | 168 | molecular/complex-level only |

The distribution is the interesting part: only 38 of the 455 prose transfers come from mouse or rat, while 408 come from fly, worm or yeast. That is not an accident of coverage — mouse curation largely mirrors human curation, so a gene nobody has characterized in human is usually uncharacterized in mouse too. The genes that get rescued are rescued by classical invertebrate and yeast genetics, which is exactly the literature a human-only search never surfaces.

## Examples

| human gene | from | confidence | shared with | statement |
|:---|:---|:---|:---|:---|
| C1QTNF5 | mouse Membrane frizzled-rel… | high_mammal | 2 genes | May play a role in eye development. |
| CR1L | mouse Complement component… | high_mammal | unique | Acts as a cofactor for complement factor I, a serine protease which protects autologous cells against complement-mediated injury… |
| CRISP3 | mouse Cysteine-rich secreto… | high_mammal | unique | This protein is supposed to help spermatozoa undergo functional maturation while they move from the testis to the ductus deferens. |
| CIMAP1B | xenopus_tropicalis Ciliary microtubule a… | medium_vertebrate | unique | Outer dense fibers are filamentous structures located on the outside of the axoneme in the midpiece and principal piece of the ma… |
| CNPY1 | zebrafish Protein canopy-1 | medium_vertebrate | unique | Involved in the maintenance of the midbrain-hindbrain boundary (MHB) organizer. Contributes to a positive-feedback loop of FGF si… |
| CXXC4 | xenopus_tropicalis CXXC-type zinc finger… | medium_vertebrate | unique | Acts as a negative regulator of the Wnt signaling pathway required for anterior neural structure formation (By similarity). Binds… |
| ACTL7B | fly Actin-like protein 53D | low_invertebrate | 4 genes | Required for optimal embryo development, particularly under heat stress conditions. Also appears to have a role in negatively reg… |
| ADAMTS16 | worm A disintegrin and met… | low_invertebrate | 2 genes | Regulates body size probably independently of the TGF beta-like dbl-1 pathway. However, may regulate some dbl-1-mediated transcri… |
| ADAMTS19 | worm A disintegrin and met… | low_invertebrate | 5 genes | Plays a role in ray morphogenesis in the male tail, probably by remodeling the extracellular matrix (ECM) in the cuticle.. |
| ABCF2 | yeast ABC transporter ATP-b… | low_fungal | 2 genes | ATPase that stimulates 40S and 60S ribosome biogenesis. Also involved in ribosome-associated quality control (RQC) pathway, a pat… |
| ABHD1 | yeast Medium-chain fatty ac… | low_fungal | unique | Displays enzymatic activity both for medium-chain fatty acid (MCFA) ethyl ester synthesis and hydrolysis (esterase activity). MCF… |
| ANKHD1-EIF4EBP3 | yeast Protein HOS4 | low_fungal | 2 genes | Unknown. Component of the Set3C complex, which is required to repress early/middle sporulation genes during meiosis. |

## The deep-homolog columns are context, not function

94 of the 235 dark genes have a bacterial, archaeal or protist homolog whose UniProt record carries curated function text. Those columns are deliberately kept out of `recommended_function`, because at that distance the shared part is the domain, not the pathway:

| dark gene | deep homolog | its curated function | why you cannot copy it |
|:---|:---|:---|:---|
| ACYP1 | *M. jannaschii* HypF | matures [NiFe] hydrogenases via carbamoyl transfer | humans have no hydrogenases; the shared part is the acylphosphatase domain |
| DIP2C | *M. tuberculosis* FadD32 | activates long-chain fatty acids for mycolic acid synthesis | humans make no mycolic acids; the shared part is the fatty-acyl AMP ligase fold |
| DMXL1 | *S. pombe* Rav1 | RAVE complex subunit required for V-ATPase assembly | this one *does* transfer — DMXL1/2 are the human RAVE homologs |

The lesson is that a prokaryotic hit tells you what biochemistry the protein is built for, and occasionally — as with DMXL1 — the whole complex-level role survives. Deciding which case you are in needs the reciprocal-best-hit flag, the query coverage, and a human reading the pathway.

## Caveats

- **Only reviewed UniProt entries carry FUNCTION text**, so an ortholog annotated only in TrEMBL is invisible here. That is why some genes fall through to a worm or yeast source when a mouse ortholog exists: the mouse entry is unreviewed, not absent.
- **Transferred prose describes the source species.** ADAMTS16's statement comes from worm and talks about cuticle collagen and body size. The metalloprotease/ECM-remodelling core is the transferable part; the cuticle is not. Always read `ortholog_species` and `transfer_confidence` beside the statement.
- **168 of the 455 prose transfers are shared with at least one human paralog**, because the ortholog relationship is many-to-one: ACTL7B, ACTL8, ACTRT2 and ACTRT3 all inherit the same statement from fly Act53D, and several ADAMTS genes inherit the same one from worm. Identical sentences on paralogs are one piece of evidence, not four — the `source_shared_with_n_human_genes` column makes that countable, and the 361 distinct sources are the real denominator.
- **A statement is not a validation.** These are the best available priors for what a gene does, generated to be checked, not cited. The `human_curated` tier is the only one where someone has actually done the human experiment.
