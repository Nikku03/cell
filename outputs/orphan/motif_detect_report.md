# TF network-motif layer: detect from the edge graph (validated on E. coli)

Proposal: motifs are topological patterns in the TF->gene EDGE graph (which we
can get), NOT something to predict from operator sequence. Recover the graph,
detect motifs vs a degree-preserving null, attach signs/logic, instantiate
dynamics with the Gillespie engine.

## Validation (RegulonDB: 1909 genes, 211 TFs, 4670 edges; 60 randomizations)

| motif | observed | null | Z | function (Alon) |
|---|---|---|---|---|
| negative autoregulation | 93 | 1.4 | +93 | fast response, noise buffering |
| positive autoregulation | 27 | 1.4 | +25 | bistability / memory |
| coherent FFL | 1056 | 411 | +12 | persistence detector / delay |
| incoherent FFL | 881 | 383 | +10 | pulse / speed-up |
| bi-fan | 61924 | 28186 | +23 | combinatorial control |
| SIM | 51 | 42 | +3 | temporal ordering |

Reproduces the canonical Milo/Alon result -> detection layer is correct.

## Pipeline (all components now exist)
1. edge graph: measured (RegulonDB) or inferred (co-expression + co-fitness, AUC 0.63)
2. motif detection: enumerate + degree-preserving null Z-scores (this script)
3. signs + logic: activator/repressor; coherent vs incoherent FFL
4. dynamics: instantiate each motif in the Gillespie engine (FFL delay shown)

## Dovetails with the specific/global split
- motif TOP node = high-fan-out GLOBAL regulator (CRP/FNR) -> operator
  unpredictable from sequence -> take edge from FUNCTIONAL route.
- second-layer TF = LOCAL/specific -> recoverable from SEQUENCE.
Motifs weld both methods: global edges from data, specific edges from sequence.

## Honest limit
Motif detection is only as good as the edge graph. Clean on RegulonDB; on an
inferred graph (0.63 AUC) false edges create spurious motifs -> unreliable
Z-scores. Solid where the network is measured, degrades where the edge graph does.

Files: colab/motif_detect.py, outputs/orphan/motif_detect.json.
