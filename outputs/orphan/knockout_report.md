# Knockout simulator -- demo reports (E. coli Keio)


## FtsZ (cell division GTPase)  --  `14241` (?)

- **product**: cell division protein FtsZ
- **slot**: cell_division    **essential (truth)**: True    **universal core**: False
- **operon neighbours**: 14240 (?, cell division protein FtsA, essential), 14242 (?, UDP-3-O-acyl-N-acetylglucosami, essential)
    polar effect hits 2 downstream essential(s)
- **phyletic co-evolving partners**: 14810 (?, glutamine--tRNA ligase/YqeY domain ); 15054 (?, lysine--tRNA ligase); 15387 (?, L-threonylcarbamoyladenylate syntha); 16970 (?, lysine--tRNA ligase); 17345 (?, L-threonylcarbamoyladenylate syntha)
- **slot cell_division**: had 8 essentials, after knockout = 5, viable min = 1 -> slot OK
- **cascade reach**: 47 additional essentials in 2-hop network neighbourhood
- **VERDICT**: **DEAD** -- essential gene knockout (AND gate: 1 essential lost -> 0 viability)

## GyrA (DNA gyrase A)  --  `16338` (?)

- **product**: DNA gyrase subunit A
- **slot**: replication    **essential (truth)**: True    **universal core**: True
- **operon neighbours**: 16337 (?, DUF2138 domain-containing prot, non-essential), 16339 (?, bifunctional 2-polyprenyl-6-hy, essential)
    polar effect hits 1 downstream essential(s)
- **slot replication**: had 19 essentials, after knockout = 17, viable min = 5 -> slot OK
- **cascade reach**: 58 additional essentials in 2-hop network neighbourhood
- **VERDICT**: **DEAD** -- essential gene knockout (AND gate: 1 essential lost -> 0 viability)

## RpoB (RNA polymerase beta)  --  `18018` (?)

- **product**: DNA-directed RNA polymerase subunit beta
- **slot**: transcription    **essential (truth)**: True    **universal core**: True
- **operon neighbours**: 18017 (?, 50S ribosomal protein L7/L12, essential), 18019 (?, DNA-directed RNA polymerase su, essential)
    polar effect hits 2 downstream essential(s)
- **slot transcription**: had 61 essentials, after knockout = 58, viable min = 6 -> slot OK
- **cascade reach**: 3 additional essentials in 2-hop network neighbourhood
- **VERDICT**: **DEAD** -- essential gene knockout (AND gate: 1 essential lost -> 0 viability)

## FolA (DHFR, folate)  --  `14194` (?)

- **product**: dihydrofolate reductase
- **slot**: cofactor    **essential (truth)**: True    **universal core**: False
- **operon neighbours**: 14193 (?, glutathione-regulated potassiu, non-essential), 14195 (?, symmetrical bis(5'-nucleosyl)-, non-essential)
    polar effect hits 1 downstream essential(s)
- **phyletic co-evolving partners**: 14302 (?, iron-sulfur cluster insertion prote); 14316 (?, translation elongation factor Ts); 14334 (?, tRNA lysidine(34) synthetase TilS); 14355 (?, DNA polymerase III subunit epsilon); 14612 (?, ferrochelatase)
- **slot cofactor**: had 5 essentials, after knockout = 3, viable min = n/a -> slot OK
- **cascade reach**: 87 additional essentials in 2-hop network neighbourhood
- **VERDICT**: **DEAD** -- essential gene knockout (AND gate: 1 essential lost -> 0 viability)

## non-essential metabolic gene  --  `15565` (?)

- **product**: aldehyde dehydrogenase family protein
- **slot**: other    **essential (truth)**: False    **universal core**: True
- **operon neighbours**: 15564 (?, , non-essential), 15566 (?, , essential)
    polar effect hits 1 downstream essential(s)
- **slot other**: had 327 essentials, after knockout = 326, viable min = n/a -> slot OK
- **cascade reach**: 1 additional essentials in 2-hop network neighbourhood
- **VERDICT**: **IMPAIRED** -- universal-core gene removal; conserved everywhere despite being non-essential here