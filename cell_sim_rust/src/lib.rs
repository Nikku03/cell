//! Rust hot-path for the cell_sim FastEventSimulator — Phase 3b.
//!
//! Provides `SimCore`, a stateful PyClass that owns:
//!   - All compiled rule tables (substrate / product / Km / kcat / etc.)
//!   - Per-rule enzyme pools (refreshed from Python on invalidation)
//!   - An event accumulator (flushed to Python Event objects on demand)
//!   - Per-rule last saturation (for cand reconstruction compat)
//!
//! Exposed methods (all bit-identical to the Python FastEventSimulator
//! when driven from the matching Python wrapper):
//!
//!   - `new(...)` — build SimCore from numpy rule tables
//!   - `set_enzyme_pool(k, ids)` — Python supplies per-rule enzyme IDs
//!   - `compute_propensities(counts, py_props)` -> (total, props, sat)
//!   - `apply_mm(k, enzyme_id, counts)` — update counts, record event
//!   - `pending_events()` -> count of unflushed events
//!   - `drain_events()` -> Vec of (time, full_rule_idx, enzyme_id)
//!
//! RNG remains in Python. The enzyme-choice draw happens in Python via
//! `rng.integers(0, n_pool)` and the chosen ID is passed to apply_mm.
//! Bit-identity is verified by `tests/test_rust_equivalence_full.py`.
//!
//! FP-exact replication:
//!   - AVOGADRO = 6.022e23 (matches coupled.py, not CODATA)
//!   - count → mM uses (count / AVOGADRO) / vol_L * 1000 exactly
//!   - saturation multiplies ratios left-to-right in declaration order
//!   - n_effective uses banker's rounding (Python 3 round() semantics)
//!   - final sum is a sequential accumulate over the propensity array

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyArray1, IntoPyArray, PyUntypedArrayMethods};
use numpy::ndarray;

const AVOGADRO: f64 = 6.022e23;
const MAX_N_EFFECTIVE: i64 = 100;


// ---------------------------------------------------------------------
// Standalone free function (unchanged from Phase 3a) — kept for callers
// that just want the propensity speedup without adopting the full core.
// ---------------------------------------------------------------------

/// Compute per-rule propensities and total (stateless).
#[pyfunction]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn compute_propensities<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<i64>,
    is_infinite: PyReadonlyArray1<bool>,
    inf_value: i64,
    vol_L: f64,
    sub_idx: PyReadonlyArray2<i64>,
    sub_stoich: PyReadonlyArray2<f64>,
    sub_mask: PyReadonlyArray2<bool>,
    km_idx: PyReadonlyArray2<i64>,
    km_val: PyReadonlyArray2<f64>,
    km_mask: PyReadonlyArray2<bool>,
    kcat: PyReadonlyArray1<f64>,
    include_sat: PyReadonlyArray1<bool>,
    enzyme_counts: PyReadonlyArray1<i64>,
    rule_idx: PyReadonlyArray1<i64>,
    py_props: PyReadonlyArray1<f64>,
) -> PyResult<(f64, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let counts = counts.as_slice()?;
    let is_inf = is_infinite.as_slice()?;
    let sub_idx_a = sub_idx.as_array();
    let sub_stoich_a = sub_stoich.as_array();
    let sub_mask_a = sub_mask.as_array();
    let km_idx_a = km_idx.as_array();
    let km_val_a = km_val.as_array();
    let km_mask_a = km_mask.as_array();
    let kcat = kcat.as_slice()?;
    let include_sat = include_sat.as_slice()?;
    let enzyme_counts = enzyme_counts.as_slice()?;
    let rule_idx = rule_idx.as_slice()?;
    let py_props_s = py_props.as_slice()?;

    let n_compiled = kcat.len();
    let n_rules = py_props_s.len();
    let max_sub = sub_idx_a.shape()[1];
    let max_km = km_idx_a.shape()[1];

    let mut props: Vec<f64> = py_props_s.to_vec();
    let mut saturation: Vec<f64> = vec![1.0; n_compiled];

    for k in 0..n_compiled {
        let ri = rule_idx[k] as usize;
        let n_enz = enzyme_counts[k];
        if n_enz <= 0 {
            props[ri] = 0.0;
            continue;
        }
        let (min_avail, have_sub) = min_availability(
            &sub_idx_a, &sub_stoich_a, &sub_mask_a, k, max_sub, counts, is_inf, inf_value);
        if !have_sub || min_avail < 1.0 {
            props[ri] = 0.0;
            continue;
        }
        let sat = saturation_factor(
            include_sat[k], &km_idx_a, &km_val_a, &km_mask_a, k, max_km,
            counts, is_inf, vol_L);
        saturation[k] = sat;
        let n_eff = clamp_n_effective(n_enz, sat);
        props[ri] = kcat[k] * (n_eff as f64);
    }

    let mut total = 0.0_f64;
    for i in 0..n_rules {
        total += props[i];
    }
    Ok((total, props.into_pyarray_bound(py).unbind(),
        saturation.into_pyarray_bound(py).unbind()))
}


// ---------------------------------------------------------------------
// Stateful SimCore (Phase 3b)
// ---------------------------------------------------------------------

/// Event record accumulated in Rust; materialised to Python Event objects
/// at flush time.
#[derive(Clone)]
struct EventRec {
    time: f64,
    rule_idx: usize,   // full rule index
    enzyme_id: i64,    // -1 when unknown (e.g. tests)
}


#[pyclass]
struct SimCore {
    // Shape
    n_species: usize,
    n_rules: usize,
    n_compiled: usize,
    max_sub: usize,
    max_prd: usize,
    max_km: usize,

    // Rule tables (flattened row-major: [k * max_X + j])
    sub_idx: Vec<i64>,
    sub_stoich: Vec<f64>,
    sub_mask: Vec<bool>,
    prd_idx: Vec<i64>,
    prd_stoich: Vec<f64>,
    prd_mask: Vec<bool>,
    km_idx: Vec<i64>,
    km_val: Vec<f64>,
    km_mask: Vec<bool>,

    kcat: Vec<f64>,
    include_sat: Vec<bool>,
    rule_idx: Vec<usize>,   // k -> full rule index

    // Species-level state
    is_infinite: Vec<bool>,
    inf_value: i64,
    vol_L: f64,

    // Per-rule enzyme pools. Each Vec<i64> is the concrete list of
    // protein instance IDs eligible to fire this rule. Refreshed from
    // Python via `set_enzyme_pool` whenever protein states change.
    enzyme_pools: Vec<Vec<i64>>,

    // Per-rule saturation from last compute_propensities call.
    last_saturation: Vec<f64>,

    // Event accumulator. Flushed to Python on demand.
    events: Vec<EventRec>,
}


#[pymethods]
#[allow(non_snake_case)]
impl SimCore {
    /// Construct a SimCore from numpy rule tables.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_species: usize,
        n_rules: usize,
        inf_value: i64,
        vol_L: f64,
        is_infinite: PyReadonlyArray1<bool>,
        sub_idx: PyReadonlyArray2<i64>,
        sub_stoich: PyReadonlyArray2<f64>,
        sub_mask: PyReadonlyArray2<bool>,
        prd_idx: PyReadonlyArray2<i64>,
        prd_stoich: PyReadonlyArray2<f64>,
        prd_mask: PyReadonlyArray2<bool>,
        km_idx: PyReadonlyArray2<i64>,
        km_val: PyReadonlyArray2<f64>,
        km_mask: PyReadonlyArray2<bool>,
        kcat: PyReadonlyArray1<f64>,
        include_sat: PyReadonlyArray1<bool>,
        rule_idx: PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        let n_compiled = kcat.as_slice()?.len();
        let max_sub = sub_idx.shape()[1];
        let max_prd = prd_idx.shape()[1];
        let max_km = km_idx.shape()[1];

        // Sanity checks
        if sub_idx.shape()[0] != n_compiled
            || prd_idx.shape()[0] != n_compiled
            || km_idx.shape()[0] != n_compiled
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rule-table dimension mismatch: n_compiled inconsistent",
            ));
        }

        Ok(Self {
            n_species,
            n_rules,
            n_compiled,
            max_sub,
            max_prd,
            max_km,
            sub_idx: sub_idx.as_array().iter().copied().collect(),
            sub_stoich: sub_stoich.as_array().iter().copied().collect(),
            sub_mask: sub_mask.as_array().iter().copied().collect(),
            prd_idx: prd_idx.as_array().iter().copied().collect(),
            prd_stoich: prd_stoich.as_array().iter().copied().collect(),
            prd_mask: prd_mask.as_array().iter().copied().collect(),
            km_idx: km_idx.as_array().iter().copied().collect(),
            km_val: km_val.as_array().iter().copied().collect(),
            km_mask: km_mask.as_array().iter().copied().collect(),
            kcat: kcat.as_slice()?.to_vec(),
            include_sat: include_sat.as_slice()?.to_vec(),
            rule_idx: rule_idx.as_slice()?.iter().map(|&x| x as usize).collect(),
            is_infinite: is_infinite.as_slice()?.to_vec(),
            inf_value,
            vol_L,
            enzyme_pools: vec![Vec::new(); n_compiled],
            last_saturation: vec![1.0; n_compiled],
            events: Vec::new(),
        })
    }

    /// Set the enzyme pool for compiled-rule `k`. Called from Python when
    /// protein states change (e.g. after a folding or assembly event).
    fn set_enzyme_pool(&mut self, k: usize, ids: Vec<i64>) -> PyResult<()> {
        if k >= self.n_compiled {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("k={k} out of range for n_compiled={}", self.n_compiled),
            ));
        }
        self.enzyme_pools[k] = ids;
        Ok(())
    }

    /// Set ALL enzyme pools in one call. Expects a list of n_compiled lists.
    fn set_all_enzyme_pools(&mut self, pools: &Bound<'_, PyList>) -> PyResult<()> {
        if pools.len() != self.n_compiled {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("expected {} pools, got {}", self.n_compiled, pools.len()),
            ));
        }
        for k in 0..self.n_compiled {
            let item = pools.get_item(k)?;
            let ids: Vec<i64> = item.extract()?;
            self.enzyme_pools[k] = ids;
        }
        Ok(())
    }

    /// Get the current pool size for compiled-rule `k`.
    fn pool_size(&self, k: usize) -> usize {
        self.enzyme_pools.get(k).map(|v| v.len()).unwrap_or(0)
    }

    /// Compute propensities over all compiled rules, combine with pre-
    /// computed Python-closure propensities, and return (total, props, sat).
    ///
    /// Propensity depends on enzyme_counts = len(enzyme_pools[k]) — the
    /// pools must be up to date before calling.
    fn compute_propensities<'py>(
        &mut self,
        py: Python<'py>,
        counts: PyReadonlyArray1<i64>,
        py_props: PyReadonlyArray1<f64>,
    ) -> PyResult<(f64, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let counts = counts.as_slice()?;
        let py_props_s = py_props.as_slice()?;
        if counts.len() != self.n_species {
            return Err(pyo3::exceptions::PyValueError::new_err("counts len mismatch"));
        }
        if py_props_s.len() != self.n_rules {
            return Err(pyo3::exceptions::PyValueError::new_err("py_props len mismatch"));
        }

        let mut props: Vec<f64> = py_props_s.to_vec();
        let mut saturation: Vec<f64> = vec![1.0; self.n_compiled];

        for k in 0..self.n_compiled {
            let ri = self.rule_idx[k];
            let n_enz = self.enzyme_pools[k].len() as i64;
            if n_enz <= 0 {
                props[ri] = 0.0;
                continue;
            }

            // Substrate availability
            let mut min_avail = f64::INFINITY;
            let mut have_any = false;
            for j in 0..self.max_sub {
                let off = k * self.max_sub + j;
                if !self.sub_mask[off] {
                    continue;
                }
                have_any = true;
                let s = self.sub_idx[off] as usize;
                let raw = if self.is_infinite[s] { self.inf_value } else { counts[s] };
                let cap = (raw as f64) / self.sub_stoich[off];
                if cap < min_avail {
                    min_avail = cap;
                }
            }
            if !have_any || min_avail < 1.0 {
                props[ri] = 0.0;
                continue;
            }

            // Saturation factor
            let sat = if self.include_sat[k] {
                let mut prod = 1.0_f64;
                for j in 0..self.max_km {
                    let off = k * self.max_km + j;
                    if !self.km_mask[off] {
                        continue;
                    }
                    let s = self.km_idx[off] as usize;
                    let c_mM = if self.is_infinite[s] {
                        1000.0
                    } else {
                        (counts[s] as f64 / AVOGADRO) / self.vol_L * 1000.0
                    };
                    let km = self.km_val[off];
                    prod *= c_mM / (c_mM + km);
                }
                prod
            } else {
                1.0
            };
            saturation[k] = sat;

            let n_eff = clamp_n_effective(n_enz, sat);
            props[ri] = self.kcat[k] * (n_eff as f64);
        }

        // Sequential sum (matches Python sum())
        let mut total = 0.0_f64;
        for i in 0..self.n_rules {
            total += props[i];
        }

        self.last_saturation = saturation.clone();

        Ok((total,
            props.into_pyarray_bound(py).unbind(),
            saturation.into_pyarray_bound(py).unbind()))
    }

    /// Apply a compiled MM rule. Updates `counts` in place (clamped to 0
    /// on the substrate side, matching `update_species_count`) and
    /// records an event.
    ///
    /// `enzyme_id` is the protein instance ID chosen by the Python RNG
    /// via `rng.integers(0, pool_size)` then indexed into the pool.
    /// Passing the ID rather than the index keeps this call tolerant to
    /// concurrent pool updates on the Python side.
    ///
    /// `time` is the current state.time (after exponential step).
    fn apply_mm(
        &mut self,
        k: usize,
        enzyme_id: i64,
        time: f64,
        mut counts: PyReadwriteArray1<i64>,
    ) -> PyResult<()> {
        if k >= self.n_compiled {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("k={k} out of range for n_compiled={}", self.n_compiled),
            ));
        }
        let counts = counts.as_slice_mut()?;

        // Substrates: decrement, clamp to 0, skip infinite species
        for j in 0..self.max_sub {
            let off = k * self.max_sub + j;
            if !self.sub_mask[off] {
                continue;
            }
            let s = self.sub_idx[off] as usize;
            if self.is_infinite[s] {
                continue;
            }
            let stoich = self.sub_stoich[off] as i64;
            counts[s] -= stoich;
            if counts[s] < 0 {
                counts[s] = 0;
            }
        }

        // Products: increment, skip infinite species
        for j in 0..self.max_prd {
            let off = k * self.max_prd + j;
            if !self.prd_mask[off] {
                continue;
            }
            let s = self.prd_idx[off] as usize;
            if self.is_infinite[s] {
                continue;
            }
            let stoich = self.prd_stoich[off] as i64;
            counts[s] += stoich;
        }

        self.events.push(EventRec {
            time,
            rule_idx: self.rule_idx[k],
            enzyme_id,
        });
        Ok(())
    }

    /// Number of unflushed events.
    fn pending_events(&self) -> usize {
        self.events.len()
    }

    /// Drain all accumulated events as a list of (time, rule_idx, enzyme_id)
    /// tuples. The internal buffer is emptied.
    fn drain_events(&mut self) -> Vec<(f64, usize, i64)> {
        let mut out = Vec::with_capacity(self.events.len());
        for ev in self.events.drain(..) {
            out.push((ev.time, ev.rule_idx, ev.enzyme_id));
        }
        out
    }

    /// Expose last_saturation for the Python `_apply` fallback path.
    fn get_last_saturation<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        self.last_saturation.clone().into_pyarray_bound(py).unbind()
    }

    /// Update vol_L after construction (in case state.metabolite_volume_L changes)
    fn set_volume(&mut self, vol_L: f64) {
        self.vol_L = vol_L;
    }
}


// ---------------------------------------------------------------------
// Shared helpers (used by both the free function and SimCore methods)
// ---------------------------------------------------------------------

#[inline]
fn min_availability(
    sub_idx_a: &ndarray::ArrayView2<i64>,
    sub_stoich_a: &ndarray::ArrayView2<f64>,
    sub_mask_a: &ndarray::ArrayView2<bool>,
    k: usize, max_sub: usize,
    counts: &[i64], is_inf: &[bool], inf_value: i64,
) -> (f64, bool) {
    let mut min_avail = f64::INFINITY;
    let mut have_any = false;
    for j in 0..max_sub {
        if !sub_mask_a[[k, j]] { continue; }
        have_any = true;
        let s = sub_idx_a[[k, j]] as usize;
        let raw = if is_inf[s] { inf_value } else { counts[s] };
        let cap = (raw as f64) / sub_stoich_a[[k, j]];
        if cap < min_avail { min_avail = cap; }
    }
    (min_avail, have_any)
}

#[inline]
#[allow(non_snake_case)]
fn saturation_factor(
    include_sat: bool,
    km_idx_a: &ndarray::ArrayView2<i64>,
    km_val_a: &ndarray::ArrayView2<f64>,
    km_mask_a: &ndarray::ArrayView2<bool>,
    k: usize, max_km: usize,
    counts: &[i64], is_inf: &[bool], vol_L: f64,
) -> f64 {
    if !include_sat { return 1.0; }
    let mut prod = 1.0_f64;
    for j in 0..max_km {
        if !km_mask_a[[k, j]] { continue; }
        let s = km_idx_a[[k, j]] as usize;
        let c_mM = if is_inf[s] {
            1000.0
        } else {
            (counts[s] as f64 / AVOGADRO) / vol_L * 1000.0
        };
        let km = km_val_a[[k, j]];
        prod *= c_mM / (c_mM + km);
    }
    prod
}

#[inline]
fn clamp_n_effective(n_enz: i64, sat: f64) -> i64 {
    let raw = (n_enz as f64) * sat;
    let mut n = banker_round_to_i64(raw);
    if n < 1 { n = 1; }
    if n > MAX_N_EFFECTIVE { n = MAX_N_EFFECTIVE; }
    n
}


/// Banker's rounding (round half to even) — matches Python 3 `round()`.
#[inline]
fn banker_round_to_i64(x: f64) -> i64 {
    if !x.is_finite() { return 0; }
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor as i64
    } else if diff > 0.5 {
        (floor + 1.0) as i64
    } else {
        let f_i = floor as i64;
        if f_i % 2 == 0 { f_i } else { f_i + 1 }
    }
}


// ---------------------------------------------------------------------
// Lennard-Jones force kernel for the AtomUnit MD engine (Session 13+)
// ---------------------------------------------------------------------
//
// Takes pre-computed pair index arrays (iu, ju) and per-atom LJ params
// (sigma, epsilon, element_code), returns the (N, 3) force array. Pair
// exclusion by bonded-set is handled by the caller before passing in
// the pair arrays, which is cheap with np.isin.
//
// Units are the same as the Python path: nm, kJ/mol, Da. Elementwise
// pair modifiers (tail-tail boost, tail-water penalty, head-tail
// reduction) match _effective_lj in force_field.py.

const COARSE_TAIL_CODE: i32 = 101;
const COARSE_SOLVENT_CODE: i32 = 102;
const COARSE_HEAD_CODE: i32 = 100;
const TAIL_TAIL_BOOST: f64 = 1.4;
const TAIL_WATER_PENALTY: f64 = 0.1;
const HEAD_TAIL_FACTOR: f64 = 0.3;

#[pyfunction]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn lj_forces<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<f64>,
    iu: PyReadonlyArray1<i64>,
    ju: PyReadonlyArray1<i64>,
    sigmas: PyReadonlyArray1<f64>,
    epsilons: PyReadonlyArray1<f64>,
    elem_codes: PyReadonlyArray1<i32>,
    cutoff: f64,
    has_coarse: bool,
    bonded_codes_sorted: Option<PyReadonlyArray1<i64>>,
) -> PyResult<Py<numpy::PyArray2<f64>>> {
    let pos_a = pos.as_array();
    let iu_s = iu.as_slice()?;
    let ju_s = ju.as_slice()?;
    let sig_s = sigmas.as_slice()?;
    let eps_s = epsilons.as_slice()?;
    let codes_s = elem_codes.as_slice()?;
    // Bonded pairs are encoded as (min(i,j) * n + max(i,j)) and passed in
    // sorted; Rust does a binary search per candidate pair to exclude.
    let bonded_codes_owned = bonded_codes_sorted.as_ref().map(|x| x.as_slice().unwrap());

    let n = pos_a.shape()[0];
    let m = iu_s.len();
    let cutoff2 = cutoff * cutoff;

    // Output forces, zero-initialised.
    let mut forces = ndarray::Array2::<f64>::zeros((n, 3));

    for p in 0..m {
        let i = iu_s[p] as usize;
        let j = ju_s[p] as usize;
        if let Some(codes) = bonded_codes_owned {
            let (lo_atom, hi_atom) = if i < j { (i, j) } else { (j, i) };
            let code = (lo_atom as i64) * (n as i64) + (hi_atom as i64);
            if codes.binary_search(&code).is_ok() {
                continue;
            }
        }
        let dx = pos_a[[j, 0]] - pos_a[[i, 0]];
        let dy = pos_a[[j, 1]] - pos_a[[i, 1]];
        let dz = pos_a[[j, 2]] - pos_a[[i, 2]];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 <= 1e-8 || r2 >= cutoff2 {
            continue;
        }
        let r = r2.sqrt();
        let sig = 0.5 * (sig_s[i] + sig_s[j]);
        let mut eps = (eps_s[i].max(0.0) * eps_s[j].max(0.0)).sqrt();

        if has_coarse {
            let ci = codes_s[i];
            let cj = codes_s[j];
            if ci == COARSE_TAIL_CODE && cj == COARSE_TAIL_CODE {
                eps *= TAIL_TAIL_BOOST;
            } else if (ci == COARSE_TAIL_CODE && cj == COARSE_SOLVENT_CODE)
                || (ci == COARSE_SOLVENT_CODE && cj == COARSE_TAIL_CODE)
            {
                eps *= TAIL_WATER_PENALTY;
            } else if (ci == COARSE_HEAD_CODE && cj == COARSE_TAIL_CODE)
                || (ci == COARSE_TAIL_CODE && cj == COARSE_HEAD_CODE)
            {
                eps *= HEAD_TAIL_FACTOR;
            }
        }

        let sr = sig / r;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let mag = 24.0 * eps * (2.0 * sr12 - sr6) / r;
        let factor = mag / r;
        let fx = factor * dx;
        let fy = factor * dy;
        let fz = factor * dz;
        // Force on j: + (mag / r) * dvec; force on i: -that.
        forces[[j, 0]] += fx;
        forces[[j, 1]] += fy;
        forces[[j, 2]] += fz;
        forces[[i, 0]] -= fx;
        forces[[i, 1]] -= fy;
        forces[[i, 2]] -= fz;
    }

    Ok(forces.into_pyarray_bound(py).unbind())
}


// ---------------------------------------------------------------------
// Extension module entrypoint
// ---------------------------------------------------------------------

#[pymodule]
fn cell_sim_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_propensities, m)?)?;
    m.add_function(wrap_pyfunction!(lj_forces, m)?)?;
    m.add_class::<SimCore>()?;
    m.add("__version__", "0.2.1")?;
    m.add("AVOGADRO", AVOGADRO)?;
    m.add("MAX_N_EFFECTIVE", MAX_N_EFFECTIVE)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banker_rounding_matches_python() {
        assert_eq!(banker_round_to_i64(0.5), 0);
        assert_eq!(banker_round_to_i64(1.5), 2);
        assert_eq!(banker_round_to_i64(2.5), 2);
        assert_eq!(banker_round_to_i64(3.5), 4);
        assert_eq!(banker_round_to_i64(-0.5), 0);
        assert_eq!(banker_round_to_i64(-1.5), -2);
        assert_eq!(banker_round_to_i64(0.4), 0);
        assert_eq!(banker_round_to_i64(0.6), 1);
        assert_eq!(banker_round_to_i64(100.5), 100);
    }

    #[test]
    fn clamp_n_effective_lo_hi() {
        assert_eq!(clamp_n_effective(0, 1.0), 1);      // floor clamp
        assert_eq!(clamp_n_effective(1000, 1.0), 100); // ceiling clamp
        assert_eq!(clamp_n_effective(10, 0.001), 1);   // round-then-clamp
        assert_eq!(clamp_n_effective(10, 0.75), 8);    // 7.5 → 8 (banker: odd)
        assert_eq!(clamp_n_effective(10, 0.85), 8);    // 8.5 → 8 (banker: even)
    }
}
