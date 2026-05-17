# Mounting ASI-Evolve on M7 (v3)

This directory contains the experiment files needed to run
[GAIR-NLP/ASI-Evolve](https://github.com/GAIR-NLP/ASI-Evolve) on the
M7 cell-state surrogate as the target task.

## What's in v3 (corrections from v2)

v3 was rewritten after auditing ASI-Evolve's actual source. v2 used
fictional prompt-template variables and assumed multi-file mutation;
neither matched the framework. v3 fixes are:

1. **Prompt templates use real variables** — `task_description`,
   `context_nodes` (with `.name`, `.motivation`, `.analysis`,
   `.results`, `.code`), `cognition_items` (with `.content`),
   `base_code`, `diff_based`. Matches ASI-Evolve's Researcher /
   Analyzer interfaces exactly.

2. **Single-file mutation** — ASI-Evolve only mutates `base_code`
   (the contents of `initial_program.py`). The multi-target
   ambition is now expressed via mutation hooks inside that file:
   `patch_model(model)`, `custom_loss_addon(...)`, plus toggles
   like `USE_TORCH_COMPILE`, `INFERENCE_DTYPE`, etc.

3. **Correct CLI** — `--experiment`, `--steps`, `--sample-n`,
   `--eval-script`. Not `--config` or `--experiment_dir`.

4. **Corrected cost estimate** — `sample-n: 3` means "show 3
   historical nodes to the Researcher", not "make 3 LLM calls".
   Per-round cost is ~3× lower than v2 docs claimed.

## Contents

| File | Role |
|------|------|
| `initial_program.py` | Single mutable file with hyperparams + inference toggles + 3 mutation hooks |
| `full_benchmark.py` | Multi-objective benchmark (accuracy + speed + memory) |
| `snapshot_tests.py` | 4 safety checks before full benchmark |
| `evaluator.py` | Parses metrics, returns Pareto-ready score dict |
| `eval.sh` | Three-phase wrapper: run candidate → snapshot tests → full benchmark |
| `input.md` | Problem description (becomes `task_description` in prompts) |
| `init_cognition.py` | Seed entries for the cognition store (M7 facts + lessons) |
| `prompts/researcher.jinja2` | Uses real ASI-Evolve variables |
| `prompts/analyzer.jinja2` | Uses real ASI-Evolve variables |
| `config.yaml` | ASI-Evolve experiment config (API + sampling + cognition) |

## Setup recipe (Colab)

```bash
# 1. Clone ASI-Evolve in /content
cd /content
git clone https://github.com/GAIR-NLP/ASI-Evolve.git asi_evolve
cd asi_evolve
pip install -r requirements.txt

# 2. Drop the M7 experiment files into experiments/m7/
mkdir -p experiments/m7/prompts experiments/m7/cognition_data
cp /content/cell/cell_sim/asi_evolve_m7/initial_program.py    experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/evaluator.py          experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/eval.sh               experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/input.md              experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/init_cognition.py     experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/config.yaml           experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/full_benchmark.py     experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/snapshot_tests.py     experiments/m7/
cp /content/cell/cell_sim/asi_evolve_m7/prompts/*.jinja2      experiments/m7/prompts/
chmod +x experiments/m7/eval.sh

# 3. Initialise the cognition store
cd experiments/m7
python init_cognition.py cognition_data/seed.json

# 4. Edit config.yaml: set api.api_key and api.model
#    For GPT-5:
#      provider: openai
#      base_url: https://api.openai.com/v1
#      api_key: sk-...
#      model: gpt-5

# 5. Launch
cd /content/asi_evolve
python main.py \
  --experiment m7 \
  --steps 20 \
  --sample-n 3 \
  --eval-script /content/asi_evolve/experiments/m7/eval.sh
```

## How the loop works

Each round, ASI-Evolve:

1. **Researcher** receives:
   - `task_description` (from `input.md`)
   - `context_nodes` — top-N (3 by default with `--sample-n 3`) historical trials, each with name/motivation/analysis/results/code
   - `cognition_items` — top-K retrieved cognition entries (FAISS similarity, see `cognition.retrieval.top_k` in config)
   - `base_code` — the current best `initial_program.py` content
   Then produces a diff via `<<<<<<< SEARCH / >>>>>>> REPLACE` blocks.

2. **Engineer** applies the diff, writes the candidate to a temp file, calls `eval.sh` with the candidate path. eval.sh runs:
   - Phase 1: execute candidate (~5–12 min); writes `m7_candidate.pt`
   - Phase 2: snapshot_tests.py against `m7_candidate.pt` (~30 sec)
   - Phase 3: full_benchmark.py for accuracy + speed + memory (~3–5 min)
   - Returns a JSON `{score, composite_relative, accuracy_ratio, speed_ratio, memory_ratio, ...}`

3. **Analyzer** receives the candidate's `code`, `results`, and the `best_sampled_node`. Writes a 5-section analysis. The LESSON line gets added to the cognition store and indexed for future retrieval.

## What the Researcher can actually do (within `initial_program.py`)

The single-file mutation surface covers three improvement axes by exposing knobs and hooks:

**Accuracy mutations** (the dominant weight in composite_relative):
- Tune hyperparameters: `LEARNING_RATE`, `BIAS_STRENGTH`, `FINE_TUNE_STEPS`, `WEIGHT_COUNT/FLUX/CUM`, `TARGET_CATEGORY`
- Rewrite `filter_weakness_by_category` to use different criteria
- Add a regularization or auxiliary loss via the `custom_loss_addon` hook

**Speed mutations** (weight 0.3):
- Set `USE_TORCH_COMPILE = True` (compile delay ~30s but ~1.5–2× faster forward)
- Set `INFERENCE_DTYPE = 'bf16'` (half the memory bandwidth per forward)
- Set `INFERENCE_DISABLE_CHECKPOINT = True` (eliminate recompute overhead at inference)
- Mutate `apply_inference_optimizations` to apply additional speedups

**Architecture mutations** (combined weight):
- Monkey-patch model attrs via `patch_model(model)` hook
  - `model.rate_clip = 5.0` (tighter rate cap)
  - `model.use_checkpoint = False` (memory-vs-speed tradeoff)
- Replace `patch_model` body to swap submodules (riskier — must remain loadable by snapshot tests)

## Expected cost (GPT-5)

With GPT-5 at $1.25/$0.125/$10 per 1M tokens (input/cached/output):

| Run | Realistic cost | 99% upper bound |
|-----|--------------:|----------------:|
| 20 rounds | **$0.40–1.00** | $5 |
| 50 rounds | $1.00–2.50 | $13 |
| 100 rounds | $2.00–5.00 | $25 |

Per-round wall clock: ~10–15 min (candidate + snapshot + benchmark).

Set a $10 hard limit on the API key for insurance (Settings → Limits → Usage).

## Honest expectations

What this will deliver:
- Real autonomous research loop with reasoning chains in the trial log
- Specific mutations you didn't think of, with mechanistic justifications
- Composite metric improvements on accuracy and/or speed

What this will NOT deliver:
- Breaking the val_count = 0.0716 stochastic noise floor (need M8 stochastic head)
- AUROC jumps beyond ~0.65 (downstream cap)
- Architecture-level changes (those need to be expressed as monkey-patches inside `patch_model`)

## Writeup framing

> "Mounted ASI-Evolve (GAIR-NLP, 2024) on the M7 surrogate as a
> target task. Seeded the cognition store with 13 M7-specific facts.
> Ran 20 rounds of LLM-driven research with FAISS-retrieval cognition
> + island-sampling parent selection, scoring on a 3-objective
> composite (accuracy + speed + memory). The autonomous loop accepted
> N of 20 proposed mutations, lowering composite_relative from 1.0 →
> 0.XX. Best mutation: [...]"

That's a defensible autonomous-research result.
