# ASI-Evolve mount for snn_scaling reversal benchmark

Drop-in experiment that lets ASI-Evolve evolve the snn_scaling memory architecture's hyperparameters on a 5-class × 3-action reversal-learning task. GPU-capable (A100 ≈ 2-3 min per round), ~$0.05 in LLM tokens per round. Target: push the best integrated agent (MemRec or CLS) past the hand-tuned baseline of +15 cumulative reward while keeping naive ≪ best integrated.

## Quick launch (Colab)

```python
# Pull the snn_scaling branch
%cd /content
!if [ -d cell ]; then cd cell && git fetch origin && git checkout claude/bio-inspired-neural-network-6dFAZ && git pull; \
   else git clone --branch claude/bio-inspired-neural-network-6dFAZ \
              https://github.com/nikku03/cell.git; fi
!pip install torch -q

# Clone ASI-Evolve if needed
!if [ -d asi_evolve ]; then echo 'asi_evolve already cloned'; \
   else git clone https://github.com/GAIR-NLP/ASI-Evolve.git asi_evolve; fi
%cd /content/asi_evolve
!pip install -r requirements.txt -q
!pip install --upgrade openai pyyaml -q

# Apply the GPT-5.x compat patch to llm.py (one-time)
import re, subprocess, shutil
llm_path = '/content/asi_evolve/utils/llm.py'
code = open(llm_path).read()
if 'GPT-5.x compatibility' not in code:
    pattern = re.compile(
        r'( *)params = \{\s*\n( *)"model": model or self\.model,\s*\n'
        r'( *)"messages": messages,\s*\n( *)\*\*self\.extra_params,\s*\n'
        r'( *)\*\*kwargs,\s*\n( *)\}', re.MULTILINE)
    m = pattern.search(code)
    indent = m.group(1)
    shim = (f'{indent}# GPT-5.x compatibility\n'
            f'{indent}_mn = (model or self.model or "")\n'
            f'{indent}_is_gpt5 = _mn.startswith("gpt-5") or _mn.startswith("o3") or _mn.startswith("o1")\n'
            f'{indent}_extra = dict(self.extra_params)\n'
            f'{indent}_kw = dict(kwargs)\n'
            f'{indent}if _is_gpt5:\n'
            f'{indent}    if "max_tokens" in _extra: _extra["max_completion_tokens"] = _extra.pop("max_tokens")\n'
            f'{indent}    if "max_tokens" in _kw: _kw["max_completion_tokens"] = _kw.pop("max_tokens")\n'
            f'{indent}    for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):\n'
            f'{indent}        _extra.pop(k, None); _kw.pop(k, None)\n'
            f'{indent}params = {{"model": model or self.model, "messages": messages, **_extra, **_kw}}')
    open(llm_path, 'w').write(code[:m.start()] + shim + code[m.end():])
    shutil.rmtree('/content/asi_evolve/utils/__pycache__', ignore_errors=True)
    print('llm.py patched')

# Build the experiment dir
import os
EXP = 'snn_scaling_reversal'
EXP_DIR = f'/content/asi_evolve/experiments/{EXP}'
shutil.rmtree(EXP_DIR, ignore_errors=True)
for sub in ('prompts', 'cognition_data', 'database_data'):
    os.makedirs(f'{EXP_DIR}/{sub}', exist_ok=True)
SRC = '/content/cell/snn_scaling/asi_evolve'
for f in ['initial_program.py', 'evaluator.py', 'eval.sh', 'input.md',
          'init_cognition.py', 'config.yaml']:
    shutil.copy(f'{SRC}/{f}', f'{EXP_DIR}/{f}')
for f in os.listdir(f'{SRC}/prompts'):
    shutil.copy(f'{SRC}/prompts/{f}', f'{EXP_DIR}/prompts/{f}')
os.chmod(f'{EXP_DIR}/eval.sh', 0o755)

# Patch config with your API key
OPENAI_KEY = 'sk-proj-PASTE_YOUR_REAL_KEY_HERE'   # REPLACE
assert OPENAI_KEY != 'sk-proj-PASTE_YOUR_REAL_KEY_HERE' and len(OPENAI_KEY) > 100, \
    'paste your real OpenAI API key (~165 chars)'
cfg_text = open(f'{EXP_DIR}/config.yaml').read()
cfg_text = re.sub(r'^(\s*)api_key:.*', f'\\1api_key: "{OPENAI_KEY}"',
                  cfg_text, count=1, flags=re.MULTILINE)
open(f'{EXP_DIR}/config.yaml', 'w').write(cfg_text)

# Seed cognition
subprocess.run(['python', f'{EXP_DIR}/init_cognition.py',
                f'{EXP_DIR}/cognition_data/seed.json'])

# Smoke test: run initial_program directly to confirm it works (~3 min on A100)
print('\n=== Smoke test: python initial_program.py ===')
env = dict(os.environ); env['PYTHONPATH'] = '/content/cell:' + env.get('PYTHONPATH', '')
r = subprocess.run(['python', f'{EXP_DIR}/initial_program.py'],
                   env=env, capture_output=True, text=True, timeout=900)
print(r.stdout[-2500:])
assert '===EVAL_METRICS===' in r.stdout, 'smoke test failed'
print('\nsmoke test passed; ready for ASI-Evolve')

# Launch 1-step (~$0.05)
%cd /content/asi_evolve
!python main.py --experiment snn_scaling_reversal --steps 1 --sample-n 3 \
    --eval-script /content/asi_evolve/experiments/snn_scaling_reversal/eval.sh
```

## Files

| File | Role |
|------|------|
| `initial_program.py` | Pipeline: build pool, build reservoir, extract centered features, train naive/memrec/cls agents on reversal task, print metrics. Mutable section exposes 20 hyperparameters. |
| `evaluator.py` | Reads candidate stdout, writes results.json with `eval_score = -composite` |
| `eval.sh` | Runs candidate with 15-min timeout, sets PYTHONPATH, dispatches to evaluator |
| `input.md` | Problem description with composite formula + safe mutation ranges |
| `init_cognition.py` | 13 seed lessons covering hand-tuning experience, dead ends, and safe knob ranges |
| `prompts/researcher.jinja2` | Researcher prompt: enforces diff blocks, includes worked example |
| `prompts/analyzer.jinja2` | Analyzer prompt: 5-section structured analysis with LESSON line for cognition |
| `config.yaml` | ASI-Evolve config with loose diff regex (4-10 brackets) |

## What the Researcher can mutate (20 knobs)

**Reservoir**: `N_RESERVOIR, P_RECURRENT, G_EXC, G_INH, TAU_SYN, RES_NOISE_STD, EXC_FRACTION`

**Task / features**: `T_TIMESTEPS, TASK_NOISE_STD, N_PER_CLASS_POOL`

**MemRec**: `MEMREC_TAU, MEMREC_TOP_K, MEMREC_EPS`

**CLS**: `CLS_HIPPO_TAU, CLS_HIPPO_TOP_K, CLS_HIPPO_CAPACITY, CLS_CORTEX_THRESHOLD, CLS_CORTEX_LR, CLS_ALPHA, CLS_EPS`

**Naive baseline**: `NAIVE_LR, NAIVE_EPS`

## Score formula

```
composite = -best_integrated_cum + 0.1 * (wall_time / 60)
          + 200 * (1 if naive_cum > best_integrated_cum else 0)
eval_score = -composite          # ASI-Evolve maximises eval_score
```

- **+1 reward = +1 score**: every cumulative reward point matters
- **Wall time penalty = 0.1 / minute** (gentle; don't sacrifice quality for speed)
- **Sanity penalty +200** if naive beats best integrated (catastrophic; avoid)

## Expected behavior

- **Baseline** (defaults from initial_program.py): MemRec ≈ +15 cumulative, naive ≈ −70, composite ≈ −14.7
- **First successful trial** (~3 min): likely tunes one safe knob like `T_TIMESTEPS=140` (wall-time win, neutral on reward) or `MEMREC_TAU` in [40, 70] (small reward gain). Expect +5 score from baseline.
- **20-round full run** (~1 hour): likely finds a Pareto frontier over (best_integrated_cum, wall_time). Lessons accumulate showing which knobs each architecture is sensitive to. Best score I'd expect: composite around −30 to −50 (i.e. best_integrated_cum ≈ +30 to +50, gap over naive widened).
