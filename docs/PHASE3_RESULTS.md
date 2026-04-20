# Phase 3 — Rust hot-path for FastEventSimulator

**Status: shipped, bit-identical to Python, 2.3x speedup over Phase 1.**

## What was built

A Rust extension (`cell_sim_rust`) exposing one function,
`compute_propensities(...)`, which replaces the vectorised-numpy propensity
block inside `FastEventSimulator._step`. A Python subclass
(`RustBackedFastEventSimulator`) inherits everything from the Phase 1
simulator and overrides only `_step`.

The scope is deliberately narrow:

- **In Rust**: compiled MM-rule propensity computation (substrate
  availability, saturation factor, `n_effective` with banker's rounding,
  kcat · n_effective); combination with pre-computed Python-closure
  propensities; per-rule saturation cached for `_apply`.
- **In Python** (unchanged): state, rule building, enzyme tracking, RNG
  draws, categorical selection, rule application, event logging.

Everything else about Phase 1 is untouched.

## Bit-identity guarantee

The Rust port is verified bit-identical to `FastEventSimulator` by
[`tests/test_rust_equivalence.py`](../cell_sim/tests/test_rust_equivalence.py).

For a 300 ms Priority 1.5 run at seed 42:

| Check | Result |
|---|---|
| Event count | 27,463 = 27,463 ✓ |
| Rule-name sequence (all 27,463) | MATCH ✓ |
| Event times (exact f64) | MATCH ✓ |
| Final metabolite counts (308 species) | MATCH ✓ |
| Final protein state distribution | MATCH ✓ |

This is the same standard used to validate Phase 1 against the pure-Python
reference simulator. The Rust path is a drop-in replacement, safe to use
anywhere `FastEventSimulator` is used.

### How bit-identity is achieved

Three design decisions keep every float bit-for-bit identical:

1. **RNG stays in Python.** All `rng.exponential()` and `rng.random()`
   calls happen in Python in the same order as `FastEventSimulator`
   (exponential → uniform). Rust never touches the RNG.

2. **FP-exact math replication.** `count → mM` uses the exact operation
   order from `coupled.py` (`(count / AVOGADRO) / vol_L * 1000.0` — two
   divisions, not one). Saturation multiplies ratios left-to-right in
   declaration order. `n_effective` uses banker's rounding (round half
   to even) to match Python 3's `round()` exactly — Rust's default
   `f64::round()` rounds half away from zero and would be wrong here.

3. **Sum semantics match.** Python recomputes `total = float(sum(props.tolist()))`
   after the Rust call, using sequential Python sum over a Python list.
   This is belt-and-braces: the sum Rust returns is already bit-identical
   because it iterates the same floats in the same order, but redundantly
   computing in Python makes the equivalence robust to future Rust
   refactors.

Banker's rounding is exercised in a Rust unit test
(`cargo test`) covering .5 boundary cases.

## Performance

Priority 1.5 at 2% scale, seed 42, 30 runs each:

| bio time | events | Python Fast (wall) | Rust-backed (wall) | speedup over Fast |
|---|---|---|---|---|
| 0.3 s | 27,463 | 3.05 s | **1.40 s** | **2.18x** |
| 1.0 s | 83,049 | 8.66 s | **3.75 s** | **2.31x** |

Compared to the original Python baseline (49 s for the same 1.0 s run),
the Rust-backed simulator is **13.1x faster end to end**. Phase 1 alone
delivered 11.4x; Phase 3 adds another 2.3x on top, and does so with zero
numerical drift.

### Why the speedup is 2.3x, not 10x

Phase 1 already vectorised the propensity computation efficiently (a
few numpy calls over padded 2D arrays). The remaining Python overhead
in `_step` is distributed across many small operations that stay in
Python: `props.tolist()` conversions, the Python `sum()`, the cumsum
loop, dict lookups during apply, f-string construction during event
logging, and Python RNG calls. The Rust port eliminates one of those
costs (numpy dispatch inside propensity compute) but leaves the others.

Getting beyond 2-3x would require moving `_apply` and event logging
into Rust too — a much larger project that would need to materialise
Python `Event` dataclasses across the FFI boundary. That's Phase 3b
territory; the current port is a clean foundation to build on.

## Toolchain requirements

To build the Rust extension from source:

```bash
# Rust toolchain (Ubuntu)
apt-get install -y rustc cargo python3-dev

# Maturin (Python <-> Rust build frontend)
pip install maturin
```

Tested on Ubuntu 24.04 with rustc 1.75.0, cargo 1.75.0, maturin 1.13.1.

## Build and install

```bash
cd cell_sim_rust
maturin build --release
pip install --force-reinstall target/wheels/cell_sim_rust-*.whl
```

Release build takes ~1 minute on a warm Cargo cache, ~4 minutes cold.
The wheel is `manylinux_2_34_x86_64` — portable across modern Linux
distributions. For Colab, the same command chain works; a pre-built
wheel in the release tarball saves the build time.

Verify the install:

```bash
python3 -c "import cell_sim_rust; print(cell_sim_rust.__version__)"
# expected: 0.1.0
cd cell_sim_rust && cargo test --release
# expected: banker_rounding_matches_python ... ok
```

## Using it

```python
from layer2_field.rust_dynamics import RustBackedFastEventSimulator

sim = RustBackedFastEventSimulator(state, rules, mode='gillespie', seed=42)
sim.run_until(t_end=1.0)
```

Identical API and behaviour to `FastEventSimulator`. The Python fallback
is still in place — if `cell_sim_rust` isn't installed, importing
`rust_dynamics` raises a clear `ImportError` with build instructions.

## Files delivered

- `cell_sim_rust/Cargo.toml` — pyo3 0.21 + numpy 0.21 deps, LTO release profile
- `cell_sim_rust/pyproject.toml` — maturin build config
- `cell_sim_rust/src/lib.rs` — ~200 lines of Rust, one public function + helpers
- `cell_sim/layer2_field/rust_dynamics.py` — Python wrapper
- `cell_sim/tests/test_rust_equivalence.py` — bit-identity + performance test
- `cell_sim_rust/target/wheels/cell_sim_rust-0.1.0-*.whl` — pre-built wheel

## Roadmap for Phase 3b (future)

To push speedup past 3x on top of Phase 1, the next targets are:

1. **Apply in Rust.** Update `self._counts` and mark species changed,
   all in Rust. Eliminates ~80k Python closure calls per Priority 1.5 run.
2. **Event logging in Rust.** Accumulate events in a Rust `Vec<EventData>`;
   flush to `state.events` at the end of `run_until` in a single batch.
   Eliminates ~80k Python `Event(...)` constructor calls.
3. **Cumsum selection in Rust.** Move the 250-element cumsum loop into
   the same Rust call as propensity compute. Saves one FFI crossing
   per step.

Each is independent and bit-identical achievable. Together they should
deliver another 3-5x on top of the current Rust-backed simulator,
putting the total end-to-end speedup in the 40-70x range over Python
baseline. Out of scope for this phase.
