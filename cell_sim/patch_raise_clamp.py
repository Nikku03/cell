"""
Diagnostic patch: raise the enzyme-count clamp from 100 to 10000.

Context: the original code clamped `n_effective` (number of enzyme
"tokens" contributing to a rule's propensity) at 100. This was fine
at 2% scale where high-abundance enzymes have <100 copies. At 50-100%
scale, enzymes like PGI (427 copies at 100%) hit the ceiling, which
artificially throttles propensity.

Hypothesis: this clamp is one of the reasons the wildtype decays at
100% scale, and also why the knockout test showed no divergence
between essential KOs and wildtype in ATP/pyruvate — because every
condition's ATP-producing enzymes were capped at the same ceiling.

This script modifies THREE files in place to raise the clamp:
  - cell_sim/layer3_reactions/reversible.py   (MAX_TOKENS = 100 -> 10000)
  - cell_sim/layer2_field/fast_dynamics.py    (np.clip(..., 1, 100) -> 10000)

If you're using the Rust backend, also edit cell_sim_rust/src/lib.rs
(MAX_N_EFFECTIVE) and rebuild with `maturin build --release`.

Run from the cell_sim root:
    python patch_raise_clamp.py

Output: three files updated, with sanity check that changes were made.
Re-run `tests/test_knockouts.py` afterward to see if the signal appears.

To revert: `git checkout cell_sim/layer3_reactions/reversible.py
           cell_sim/layer2_field/fast_dynamics.py`
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NEW_CAP = 10000

TARGETS = [
    dict(
        path=REPO_ROOT / 'layer3_reactions' / 'reversible.py',
        replacements=[
            ('    MAX_TOKENS = 100\n',
             '    MAX_TOKENS = 10000   # raised from 100 to allow high-abundance enzymes to scale at full cell sizes\n'),
        ],
    ),
    dict(
        path=REPO_ROOT / 'layer2_field' / 'fast_dynamics.py',
        replacements=[
            # The exact two-line block — preserve indentation and comment
            ('            # n_effective = max(1, min(100, round(E * sat))) — matches\n'
             '            # reversible.py:177 exactly.\n'
             '            n_eff = np.clip(np.round(self.C_enzyme_counts * saturation),\n'
             '                             1, 100).astype(np.int64)\n',
             '            # n_effective = max(1, min(10000, round(E * sat))) — matches\n'
             '            # reversible.py:MAX_TOKENS exactly. Raised from 100 so high-abundance\n'
             '            # enzymes (PGI ~400 copies, PTAr ~300 copies) don\'t throttle at full scale.\n'
             '            n_eff = np.clip(np.round(self.C_enzyme_counts * saturation),\n'
             '                             1, 10000).astype(np.int64)\n'),
        ],
    ),
]


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> int:
    """Apply each (old, new) pair. Return count of successful substitutions."""
    if not path.is_file():
        print(f'  SKIP: {path} does not exist')
        return 0
    text = path.read_text()
    n_applied = 0
    for old, new in replacements:
        if old not in text:
            print(f'  SKIP: pattern not found in {path}')
            print(f'    looking for: {old!r:.120s}...')
            continue
        text = text.replace(old, new, 1)
        n_applied += 1
    if n_applied > 0:
        path.write_text(text)
        print(f'  OK:   patched {path} ({n_applied} substitutions)')
    return n_applied


def main():
    print(f'Patching enzyme clamp: 100 -> {NEW_CAP}')
    print('')
    total = 0
    for t in TARGETS:
        n = patch_file(t['path'], t['replacements'])
        total += n
    print('')
    print(f'Done. {total} substitutions applied.')
    if total == 0:
        print('WARNING: zero changes made. File layouts may have drifted; '
              'patch manually or re-examine strings in this script.')
    else:
        print('')
        print('Next steps:')
        print('  1. If using Rust backend: edit cell_sim_rust/src/lib.rs,')
        print('     change MAX_N_EFFECTIVE from 100 to 10000, rebuild:')
        print('       cd cell_sim_rust && maturin build --release')
        print('       pip install --force-reinstall target/wheels/cell_sim_rust-*.whl')
        print('')
        print('  2. Re-run the knockout test:')
        print('       cd cell_sim')
        print('       KO_SCALE=0.5 KO_T_END=0.5 python tests/test_knockouts.py')
        print('')
        print('  3. If ATP trajectories diverge between essential vs non-essential')
        print('     KOs, the clamp was the bottleneck. If not, move on to ADK1')
        print('     direction audit.')


if __name__ == '__main__':
    main()
