# Essential genes by functional category

## Part 1: named essentials (n=3,021) -> COG categories

Source: 11 DEG benchmark organisms with canonical gene symbols. Categorized by symbol-prefix mapping; coverage 18%.

| COG letter | category | n | % |
|---|---|---:|---:|
| J | Translation/ribosome | 199 | 36.7% |
| L | DNA replication/repair | 75 | 13.8% |
| H | Coenzyme metabolism | 39 | 7.2% |
| I | Lipid metabolism | 38 | 7.0% |
| M | Cell wall/membrane | 36 | 6.6% |
| F | Nucleotide metabolism | 35 | 6.5% |
| G | Carbohydrate metabolism | 26 | 4.8% |
| D | Cell division/shape | 23 | 4.2% |
| K | Transcription | 21 | 3.9% |
| U | Protein secretion | 17 | 3.1% |
| C | Energy production | 17 | 3.1% |
| E | Amino acid biosynthesis | 16 | 3.0% |

## Part 2: beril OG-level breakdown (n=5,658 OGs)

Coverage limitation: only flag-based categories available locally (~16%); the rest are 'unannotated_by_flags' and are likely the J/L/M/D categories that dominate Part 1. Resolving them needs KEGG/Pfam from feba.db (Drive).

| bucket | n OGs | % |
|---|---:|---:|
| unannotated_by_flags | 4,738 | 83.7% |
| regulator | 300 | 5.3% |
| transport | 290 | 5.1% |
| metabolic_(in_FBA_model) | 288 | 5.1% |
| signaling | 42 | 0.7% |