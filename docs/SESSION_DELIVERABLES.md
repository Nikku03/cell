# Session deliverables — map, run, status

## Notebooks (Colab)
| Notebook | Does | GPU |
|---|---|---|
| `colab/build_complete_cell.ipynb` | the full model build (+ all validations now wired in) | yes |
| `colab/transformer_dynamics.ipynb` | LINCS perturbation transformer (trained; repurposed for perturbation prediction) | yes |
| `colab/reverse_inference_test.ipynb` | **the real reverse-inference test** (recover a KD gene from its LINCS signature) | no |
| `colab/ask_the_cell.ipynb` | **the agent** — ask macro questions, cited answers | no |

## The agent — `cell_agent.py`
`CellAgent().ask(q)` (deterministic, grounded) or `.chat(q)` (LLM phrasing if ANTHROPIC/OPENAI key set;
falls back to deterministic). Recipes: target dossier ("will disabling it work?"), disease→target (reverse
inference), perturbation effect. Every fact sourced + confidence-tagged; nothing invented.

## Validated (real numbers)
- Kinetics error **9.6× → 3.9×** (CatPred, leak-verified) · Km 1,487 · **measured Ki** (new).
- Metabolite concentrations: 94 measured; thermo cross-check 2.19×.
- Dynamics: response times for 18k genes; time-resolution equation 0.29, learned 0.25.
- **Dark-gene hypotheses beat chance 2.18×** (z=31.6); **3.11× / 84% when ≥2 datasets agree**.
- **Approved-drug-target prediction from biology: AUC 0.83**; shortlist enriched, surfaced MCL1/TMPRSS2/SNAI1.
- Agent outputs biologically correct (TP53 KO cascade, EGFR drugs, MYC→MAX SL).

## Pending real-data numbers (run on Colab)
- `lincs_reverse_test.py` — **reverse-inference accuracy** (the discovery engine's true test). Sandbox self-check
  is circular; this is the honest number. Wired into `reverse_inference_test.ipynb`.
- `build_target_gnn.py` — GNN over the full dense network (the +0.011 local gain should grow with co-expression
  + Perturb-seq lenses populated).
- `build_propagation_therapy.py` — real self-amplifying ligand→receptor circuits (needs the lr layer).

## Honest boundaries
Target-level, cellular, confidence-scored. NOT molecule design, PK/ADMET, or clinical outcome. Reverse
inference is a hypothesis until the LINCS test returns a non-circular number.
