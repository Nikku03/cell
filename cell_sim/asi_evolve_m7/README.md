# Mounting ASI-Evolve on M7

This directory contains the experiment files needed to run
[GAIR-NLP/ASI-Evolve](https://github.com/GAIR-NLP/ASI-Evolve) on the
M7 cell-state surrogate as the target task.

## Contents

| File | Role |
|------|------|
| `initial_program.py` | Starting M7 fine-tune script the Researcher mutates |
| `evaluator.py` | Parses metrics from candidate stdout, returns score dict |
| `eval.sh` | Shell wrapper called by ASI-Evolve's Engineer |
| `input.md` | Problem description shown to the Researcher LLM |
| `init_cognition.py` | Seed entries for the cognition store (M7 facts + lessons) |
| `prompts/researcher.jinja2` | Researcher LLM prompt template |
| `prompts/analyzer.jinja2` | Analyzer LLM prompt template |
| `config.yaml` | ASI-Evolve experiment config |

## Setup recipe (Colab)

```bash
# 1. Clone ASI-Evolve in /content
cd /content
git clone https://github.com/GAIR-NLP/ASI-Evolve.git asi_evolve
cd asi_evolve
pip install -r requirements.txt

# 2. Copy the M7 experiment files into ASI-Evolve's experiments/ tree
mkdir -p experiments/m7/prompts
cp /content/cell/cell_sim/asi_evolve_m7/initial_program.py experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/evaluator.py        experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/eval.sh             experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/input.md            experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/init_cognition.py   experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/config.yaml         experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/prompts/*.jinja2    experiments/m7/prompts/
chmod +x experiments/m7/eval.sh

# 3. Initialise the cognition store with M7 facts
cd experiments/m7
python init_cognition.py cognition_data/seed.json

# 4. Add your LLM API key to config.yaml
#    Edit experiments/m7/config.yaml -> api.api_key

# 5. Launch
cd /content/asi_evolve
python main.py --config experiments/m7/config.yaml \
               --experiment_dir experiments/m7
```

## What the loop will do

Each round:

1. **Researcher** reads:
   - The current best `initial_program.py`
   - The top-K cognition entries (M7 facts + accumulated lessons)
   - The last N trials' results
   Then proposes a diff-style mutation. Examples it might try:

   - Tune `LEARNING_RATE` to 5e-5
   - Change `TARGET_CATEGORY` to 'cum' to focus on cumulative species
   - Replace the bias_strength=0.7 logic with a curriculum that ramps from 0.5 to 0.85
   - Add a stoichiometric-consistency loss term
   - Swap `fine_tune_on_weaknesses` for a 2-stage curriculum: short rollouts then long

2. **Engineer** applies the diff to the candidate program, runs it (`eval.sh`), parses metrics.

3. **Analyzer** writes a 4-line analysis (goal / effect / mechanism / lesson). The lesson is added to the cognition store, retrievable by future Researcher rounds via FAISS similarity.

4. The database persists the trial. The island-based parent sampler (MAP-Elites) picks where to branch next round, encouraging exploration across (bias_strength × target_category × LR) cells.

## Expected behaviour

- ~10-15 min per round (~10 min candidate eval + ~30s LLM calls)
- 20 rounds ≈ 3-5 hours total
- Expect 2-5 accepted mutations out of 20 trials
- Final composite should drop from 0.0837 → roughly **0.075-0.080** (range of plausible improvement; hard ceiling exists)

## What this canNOT do

- Break the val_count = 0.0716 stochastic noise floor (needs M8 stochastic head)
- Improve AUROC vs Breuer 2019 by more than ~0.01-0.02 (downstream metric capped by the same noise floor)
- Discover genuinely new architectures (mutations are restricted to the fine-tune surface, not the encoder structure — that would require extending `initial_program.py` to load and modify the model architecture)

## Honest framing for the writeup

> "Mounted ASI-Evolve on M7 as a target task. Configured the Researcher
> LLM (GPT-4o / Claude) with M7 domain knowledge via a cognition store
> of 13 seed lessons. Ran 20 rounds of autonomous fine-tune-mutation
> proposals, evaluated each candidate against a composite val metric.
> Final M7 variant achieved composite=0.0XX (Δ -X% vs baseline), with
> the accepted lineage showing [the actual mutations]."

That's a defensible result: real autonomous research, real LLM-driven
hypothesis generation, real metric improvement. The honest limitation
(noise floor caps absolute gains) doesn't undermine the methodology.

## To go further: extend `initial_program.py`

The Researcher can only mutate what's in `initial_program.py`. To
unlock more interesting mutations, expand the initial program to:

- Load the model weights directly (so the Researcher can modify
  architecture)
- Expose the loss function as Python code (so new terms can be added)
- Include the encoder forward pass (so attention mechanisms can be
  swapped)

Each expansion increases the search space but also the chance of
generating un-runnable mutations. Start small and expand if the
loop is producing too many timid mutations.
