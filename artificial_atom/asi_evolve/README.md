# ASI-Evolve mount for artificial_atom v3

Drop-in experiment that lets ASI-Evolve evolve the artificial_atom v3
molecular force-field surrogate. CPU-only, ~10 min per round, ~$0.05
in LLM tokens per round. Target: maintain 8/8 verification checks
while reducing wall time / improving metric margins.

## Quick launch (Colab)

```python
# Pull the artificial_atom branch
%cd /content
!if [ -d cell ]; then cd cell && git fetch origin && git checkout claude/bio-inspired-neural-network-6dFAZ && git pull; \
   else git clone --branch claude/bio-inspired-neural-network-6dFAZ \
              https://github.com/nikku03/cell.git; fi
!pip install "numpy<2" rdkit -q

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
    print('✓ llm.py patched')

# Build the experiment dir
import os
EXP = 'artificial_atom_v3'
EXP_DIR = f'/content/asi_evolve/experiments/{EXP}'
shutil.rmtree(EXP_DIR, ignore_errors=True)
for sub in ('prompts', 'cognition_data', 'database_data'):
    os.makedirs(f'{EXP_DIR}/{sub}', exist_ok=True)
SRC = '/content/cell/artificial_atom/asi_evolve'
for f in ['initial_program.py', 'evaluator.py', 'eval.sh', 'input.md',
          'init_cognition.py', 'config.yaml']:
    shutil.copy(f'{SRC}/{f}', f'{EXP_DIR}/{f}')
for f in os.listdir(f'{SRC}/prompts'):
    shutil.copy(f'{SRC}/prompts/{f}', f'{EXP_DIR}/prompts/{f}')
os.chmod(f'{EXP_DIR}/eval.sh', 0o755)

# Patch config with your API key (use string substitution, not yaml round-trip)
OPENAI_KEY = 'sk-proj-PASTE_YOUR_REAL_KEY_HERE'   # ← REPLACE
assert OPENAI_KEY != 'sk-proj-PASTE_YOUR_REAL_KEY_HERE' and len(OPENAI_KEY) > 100, \
    'paste your real OpenAI API key (~165 chars)'
cfg_text = open(f'{EXP_DIR}/config.yaml').read()
cfg_text = re.sub(r'^(\s*)api_key:.*', f'\\1api_key: "{OPENAI_KEY}"',
                  cfg_text, count=1, flags=re.MULTILINE)
open(f'{EXP_DIR}/config.yaml', 'w').write(cfg_text)

# Seed cognition
subprocess.run(['python', f'{EXP_DIR}/init_cognition.py',
                f'{EXP_DIR}/cognition_data/seed.json'])

# Smoke test (~8 min, $0): run initial_program directly to confirm it works
print('\n=== Smoke test: python initial_program.py (~8 min, $0) ===')
env = dict(os.environ); env['PYTHONPATH'] = '/content/cell:' + env.get('PYTHONPATH', '')
r = subprocess.run(['python', f'{EXP_DIR}/initial_program.py'],
                   env=env, capture_output=True, text=True, timeout=1200)
print(r.stdout[-2500:])
assert '===EVAL_METRICS===' in r.stdout, 'smoke test failed'
print('\n✓ smoke test passed; ready for ASI-Evolve')

# Launch 1-step (~$0.05)
%cd /content/asi_evolve
!python main.py --experiment artificial_atom_v3 --steps 1 --sample-n 3 \
    --eval-script /content/asi_evolve/experiments/artificial_atom_v3/eval.sh
```

## Files

| File | Role |
|------|------|
| `initial_program.py` | Pipeline: build dataset → train primary → train ensemble → run 8-check verify → print metrics. Mutable section exposes 18 hyperparameters. |
| `evaluator.py` | Reads candidate stdout, writes results.json with `eval_score = -composite` |
| `eval.sh` | Runs candidate with 15-min timeout, sets PYTHONPATH, dispatches to evaluator |
| `input.md` | Problem description with the 8-check contract + safe mutation ranges |
| `init_cognition.py` | 13 seed lessons covering check margins, safe ranges, common failure modes |
| `prompts/researcher.jinja2` | Researcher prompt: uses real ASI-Evolve variables, enforces diff blocks, includes worked example |
| `prompts/analyzer.jinja2` | Analyzer prompt: 5-section structured analysis with LESSON line for cognition |
| `config.yaml` | ASI-Evolve config with loose diff regex (4-10 brackets) |

## What the Researcher can mutate (18 knobs)

Architecture: `D_H, D_RBF, D_BOND_EXTRA, N_LAYERS, R_CUTOFF, R_CUTOFF_ELECTRO`

Training: `EPOCHS_PRIMARY, EPOCHS_ENSEMBLE, LR_PRIMARY, BATCH_SIZE, ENERGY_WEIGHT, FORCE_WEIGHT, POSE_WEIGHT, BAD_POSE_WEIGHT, GRAD_CLIP, SEED_PRIMARY, SEED_ENSEMBLE`

Data: `TRAIN_FRAC, VAL_FRAC`

## Why this is faster to mount than M7

| | M7 | artificial_atom v3 |
|---|---|---|
| Hardware required | A100 40GB+ | CPU enough |
| Per-candidate train | 5-10 min | 7-10 min |
| Failure modes per round | many (OOM, dtype, device, NaN) | few (mostly clean Python) |
| Cost per round | $0.05 LLM + GPU credits | $0.05 LLM, no GPU |
| Score range | -∞ to 0 (composite_relative ratio) | clear: -∞ to -0.70 (composite) |

## Expected behavior

- **First successful trial** (~10 min): probably tunes one safe knob like `EPOCHS_ENSEMBLE = 8` or `FORCE_WEIGHT = 0.05`. Should produce 8/8 still, lower wall time, slightly lower composite than baseline.
- **20-round full run** (~3-5 hr): likely finds a Pareto frontier in (wall_time, metric_margins). Lessons accumulate showing which knobs are safe to push.
