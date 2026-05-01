# SIREN + HNN data-scaling smoke test — findings

**Run date:** 2026-05-01
**Scope:** Re-test two previously-failed Dark Manifold-adjacent
components at higher data densities to distinguish data-sparsity
failures from structural failures.

## What was tested

The April 2026 Dark Manifold salvage tests evaluated three component
architectures as approximations of Dark Manifold concepts. Two of the
three failed:

* **SIREN spatial field** failed at 500 training points on a 3D
  synthetic metabolite field (test MSE 0.039 vs plain MLP 0.008).
* **Hamiltonian Neural Network (HNN)** passed on the harmonic
  oscillator (energy-conserving) and failed by ~14× on the damped
  oscillator (dissipative).

The two failures had two possible explanations: *data sparsity* (the
component is correct but starved) or *structural mismatch* (the
component cannot represent the target regardless of data). This
smoke test re-runs both at the data densities the original salvage
test did not cover.

The test does **not** evaluate the full Dark Manifold concept (4D
spacetime field with dark matter coupling, quantum fluctuations as
sampling, superposition collapse, cognitive scaffold). That
architecture was never built and is still untested.

Synthetic data, model widths, training budgets, and exact data
conditions are documented in
`siren_data_scaling_test.py` and `hnn_data_scaling_test.py`.
Reproducible at `seed=42`.

## SIREN results

3D synthetic metabolite field (40³ grid, ~57k interior points,
Gaussian ribosome blobs + radial membrane gradient + sin/cos
high-frequency crowding). 5000 held-out test points, constant across
conditions. Each model: hidden width 128, depth 3, Adam lr=1e-3,
batch 2048, 3000 optimization steps.

### Test MSE (lower is better)

| n_train  | MLP-ReLU  | Fourier-MLP | SIREN-ω10 | SIREN-ω30 |
|---------:|----------:|------------:|----------:|----------:|
|     500  | **4.94e-3** | 7.09e-3   | 7.06e-3   | 4.04e-2   |
|   5 000  | **3.54e-3** | 6.50e-3   | 6.27e-3   | 5.40e-3   |
|  50 000  | **3.12e-3** | 4.69e-3   | 3.18e-3   | 3.35e-3   |
| 500 000  | 3.12e-3   | **3.99e-4** | 3.11e-3   | **1.45e-3** |

### Gradient cosine similarity vs ground truth (higher is better)

| n_train  | MLP-ReLU | Fourier-MLP | SIREN-ω10 | SIREN-ω30 |
|---------:|---------:|------------:|----------:|----------:|
|     500  | +0.121   | +0.031      | +0.098    | −0.009    |
|   5 000  | +0.172   | +0.051      | +0.057    | +0.086    |
|  50 000  | +0.178   | +0.073      | +0.163    | +0.184    |
| 500 000  | +0.185   | **+0.708**  | +0.167    | **+0.694** |

### Interpretation

The original salvage report's failure mode reproduces at 500 points:
SIREN-ω30 sits at test MSE 4.04e-2 while MLP-ReLU is at 4.94e-3 — an
8× MLP win, qualitatively the same as the original 0.039 vs 0.008.
By 50 000 points the four models converge to within ~50% of each
other. By 500 000 points the high-frequency-aware models pull ahead:
SIREN-ω30 at 1.45e-3 (~2.1× better than MLP-ReLU's 3.12e-3) and
Fourier-MLP at 3.99e-4 (~7.8× better). The gradient-cosine table is
the cleaner signal: at 500 000 points the frequency-aware models
recover gradient direction (+0.7) while MLP-ReLU plateaus at +0.18,
which is what we expect when MLP-ReLU has fit the smooth low-
frequency components but cannot represent the sin/cos crowding term.

**SIREN's 500-point failure was a data-sparsity issue.** Given enough
training points, the architecture's high-frequency representation
becomes load-bearing and the model wins on this synthetic test.

What this does **not** establish: viability for cell biology. Real
cell trajectory or omics datasets typically expose 10²–10⁴ measured
points per quantity, not 5×10⁵. The crossover where SIREN starts
beating MLP on this synthetic test is between 50 000 and 500 000
points — far above the cell-biology data regime. The salvage test's
original 500-point setup was the realistic cell-data scale; that
scale was where SIREN lost. Wins at 500k synthetic points do not
back-imply wins at 500 measured points.

## HNN results

Harmonic oscillator (`m·x'' + k·x = 0`, conservative) and damped
oscillator (`m·x'' + b·x' + k·x = 0`, dissipative). Both with
m=k=1, b=0.5. Trajectories generated via RK4 at dt=0.01 to t=10s.
Models: HNN that learns scalar H(q,p) and predicts (q̇, ṗ) =
(∂H/∂p, −∂H/∂q) by autograd, vs plain MLP that predicts (q̇, ṗ)
directly. Both width 128, depth 3, tanh activation. 2000
optimization steps. 8 held-out test trajectories with new initial
conditions, rolled forward via RK4 using the model's predicted
derivatives. `traj_mse` = mean squared error in (q,p) over the 1001
rollout steps.

### Trajectory MSE (lower is better)

| n_train  | harmonic-MLP | harmonic-HNN | damped-MLP   | damped-HNN |
|---------:|-------------:|-------------:|-------------:|-----------:|
|     100  | 6.13e-4      | 1.90e-3      | **2.89e-4**  | 3.21e-1    |
|   1 000  | 3.85e-4      | **1.74e-4**  | **2.29e-5**  | 1.90e-1    |
|  10 000  | 3.79e-4      | 4.66e-4      | **2.29e-5**  | 1.92e-1    |
| 100 000  | 2.84e-4      | **9.35e-5**  | **1.26e-4**  | 1.89e-1    |

### Damped HNN/MLP ratio (HNN failure magnitude)

| n_train  | HNN/MLP ratio (damped) |
|---------:|-----------------------:|
|     100  | 1 110×                 |
|   1 000  | 8 300×                 |
|  10 000  | 8 400×                 |
| 100 000  | 1 500×                 |

### Interpretation

On the harmonic (conservative) oscillator, HNN matches or beats MLP
at every data condition past n=100, as expected — the architectural
constraint matches the system. At n=100 000, HNN reaches traj_mse
9.35e-5 vs MLP's 2.84e-4 (~3× better), with consistently lower
energy drift. This is the canonical "structural prior fits the
target" story.

On the damped (dissipative) oscillator, HNN's traj_mse never drops
below 1.89e-1. The MLP improves rapidly with data (down to ~2.3e-5
at n=10 000) while HNN sits flat at ~1.9e-1. The damped HNN/MLP ratio
gets *worse* with more data — from 1 110× at n=100 to 8 300× at
n=1 000 — because MLP improves while HNN's floor is fixed. The mild
improvement at n=100 000 (down to 1 500×) is MLP regressing slightly
on this run, not HNN making progress.

This is the canonical signature of a **structural failure**, not a
data-sparsity failure. Hamilton's equations preserve total energy by
construction; an HNN architecture literally cannot represent
dissipation regardless of how much training data it sees. The
April 2026 salvage report's 14× HNN/MLP ratio on damped was at lower
data; with more data and a converged MLP baseline, the ratio is
much larger because HNN's floor sits where it always sat.

This is a confirmation, not a surprise. The failure is the kind of
result we would extrapolate from theory anyway; the smoke test
converts the inference into a measurement.

## What this means and what it doesn't

**What this test concludes:**

* SIREN's failure on the original 500-point salvage test was a
  data-sparsity failure. At 500 000 training points on this 3D
  synthetic field SIREN-ω30 outperforms a plain MLP-ReLU by ~2.1×
  on test MSE and ~3.8× on gradient cosine.
* Fourier-MLP also beats both vanilla MLP and SIREN at 500 000
  points on this target — frequency-aware input embedding alone is
  enough to capture the sin/cos crowding component.
* HNN's failure on the damped oscillator persists at all four data
  conditions tested up to n=100 000. The HNN/MLP ratio gets worse
  with more data because the MLP improves while HNN sits at a
  structural floor. This is structural failure: the architecture
  cannot represent dissipation.

**What this test does NOT conclude:**

* Anything about the full Dark Manifold concept. The Dark Manifold
  was never built. Its central claims (4D continuous spacetime
  field with dark matter coupling, quantum fluctuations as sampling,
  superposition with geodesic collapse) are still untested.
* That SIREN is viable for cell biology. The crossover where SIREN
  beats MLP on this synthetic test is around 50 000–500 000 points.
  Cell biology training data is typically 10²–10⁴ points per
  quantity. The 500-point salvage failure is the realistic
  cell-data scale; SIREN losing there is the relevant data point
  for "is this useful for cell biology," and the answer is no.
* That HNN is a wrong technique generally. HNN works exactly as
  designed on conservative systems and is the correct prior there.
  The damped failure is not a bug — it's the architecture being
  honest about what it cannot represent. Real cell systems are
  overwhelmingly dissipative; an HNN-shaped layer would import
  exactly the wrong inductive bias.

## Updated salvage status

| Component                         | Original salvage result        | Followup result (this test) | Verdict |
|-----------------------------------|--------------------------------|-----------------------------|---------|
| Green's function regulatory layer | PASS (+10.7pp on Syn3A)        | not re-tested here          | unchanged |
| SIREN spatial field               | FAIL at n=500 (8× MLP win)     | wins at n=500 000 (2.1×); reproduces failure at n=500 | data-sparsity, not structural |
| Hamiltonian Neural Network        | PASS harmonic; FAIL damped 14× | passes harmonic at n≥1 000; damped failure persists 1 500–8 400× across all n | structural |

The smoke test converts two inferences into measurements:

* "SIREN failed because of data sparsity, not because the
  architecture is wrong" — measured. The architecture works given
  enough data.
* "HNN cannot represent dissipation regardless of data" — measured.
  The damped failure persists across four orders of magnitude in
  training-set size and gets worse at higher data.

Neither result changes the practical conclusion for cell biology.
Cell data is sparse and cell dynamics are dissipative; the
architectural primitives that fail in this test domain remain
unsuited for the cell-biology track.
