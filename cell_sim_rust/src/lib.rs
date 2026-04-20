//! Rust hot-path for the cell_sim FastEventSimulator.
//!
//! Exposes `compute_propensities(...)` which:
//!   1. Reads the vectorised rule tables (padded 2D numpy arrays).
//!   2. Computes Michaelis-Menten propensities for every compiled MM rule
//!      using the same FP-exact math as the reference Python simulator.
//!   3. Combines with pre-computed Python-closure rule propensities.
//!   4. Returns (total_propensity, full_propensity_array, saturation_array).
//!
//! All RNG draws and the categorical selection stay in Python so the
//! simulator remains bit-identical to `FastEventSimulator` (same seed →
//! same event sequence, metabolite trajectory, and event log).
//!
//! FP-exact replication notes:
//!   - AVOGADRO = 6.022e23 (matches coupled.py exactly; not CODATA)
//!   - count → mM uses (count / AVOGADRO) / vol_L * 1000 (two divisions
//!     in this exact order)
//!   - saturation multiplies ratios left-to-right in declaration order
//!   - n_effective uses banker's rounding (matches Python's `round()`)
//!   - final sum is a sequential accumulate (matches `sum(list)`)

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, IntoPyArray};

const AVOGADRO: f64 = 6.022e23;
const MAX_N_EFFECTIVE: i64 = 100;


/// Compute per-rule propensities and total.
///
/// Returns `(total_propensity, propensities[n_rules], saturation[n_compiled])`.
/// The propensities array is a fresh allocation each call; saturation
/// is needed by Python `_apply` to reconstruct cands identically.
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
    let is_infinite = is_infinite.as_slice()?;
    let sub_idx_arr = sub_idx.as_array();
    let sub_stoich_arr = sub_stoich.as_array();
    let sub_mask_arr = sub_mask.as_array();
    let km_idx_arr = km_idx.as_array();
    let km_val_arr = km_val.as_array();
    let km_mask_arr = km_mask.as_array();
    let kcat = kcat.as_slice()?;
    let include_sat = include_sat.as_slice()?;
    let enzyme_counts = enzyme_counts.as_slice()?;
    let rule_idx = rule_idx.as_slice()?;
    let py_props_slice = py_props.as_slice()?;

    let n_compiled = kcat.len();
    let n_rules = py_props_slice.len();
    let max_sub = sub_idx_arr.shape()[1];
    let max_km = km_idx_arr.shape()[1];

    // Working propensity vector: start from the python-closure cache,
    // then overwrite compiled-rule slots below.
    let mut props: Vec<f64> = py_props_slice.to_vec();

    // Per-rule saturation factor (returned for Python `_apply`).
    let mut saturation: Vec<f64> = vec![1.0; n_compiled];

    // ------------------------------------------------------------------
    // Compiled Michaelis-Menten propensities
    // ------------------------------------------------------------------
    for k in 0..n_compiled {
        let ri = rule_idx[k] as usize;
        let n_enz = enzyme_counts[k];

        // No enzyme → this rule is inactive. Zero out any stale python value.
        if n_enz <= 0 {
            props[ri] = 0.0;
            continue;
        }

        // Substrate availability: min capacity across substrates.
        // Padded slots are skipped via the mask.
        let mut min_avail = f64::INFINITY;
        let mut have_any_substrate = false;
        for j in 0..max_sub {
            if !sub_mask_arr[[k, j]] {
                continue;
            }
            have_any_substrate = true;
            let s = sub_idx_arr[[k, j]] as usize;
            let raw = if is_infinite[s] { inf_value } else { counts[s] };
            let cap = (raw as f64) / sub_stoich_arr[[k, j]];
            if cap < min_avail {
                min_avail = cap;
            }
        }
        if !have_any_substrate || min_avail < 1.0 {
            props[ri] = 0.0;
            continue;
        }

        // MM saturation factor. Matches Python `mm_saturation_factor` exactly:
        // iterate Km substrates in their declaration order, multiply ratios
        // `c_mM / (c_mM + Km)` into a running product initialised at 1.0.
        let sat = if include_sat[k] {
            let mut prod = 1.0_f64;
            for j in 0..max_km {
                if !km_mask_arr[[k, j]] {
                    continue;
                }
                let s = km_idx_arr[[k, j]] as usize;
                let c_mM = if is_infinite[s] {
                    1000.0
                } else {
                    (counts[s] as f64 / AVOGADRO) / vol_L * 1000.0
                };
                let km = km_val_arr[[k, j]];
                prod *= c_mM / (c_mM + km);
            }
            prod
        } else {
            1.0
        };
        saturation[k] = sat;

        // n_effective = clamp(round(n_enz * sat), 1, 100) using banker's
        // rounding to match Python's `round()` semantics.
        let raw = (n_enz as f64) * sat;
        let mut n_eff = banker_round_to_i64(raw);
        if n_eff < 1 { n_eff = 1; }
        if n_eff > MAX_N_EFFECTIVE { n_eff = MAX_N_EFFECTIVE; }

        props[ri] = kcat[k] * (n_eff as f64);
    }

    // ------------------------------------------------------------------
    // Total — sequential accumulate (matches Python `sum(list)`)
    // ------------------------------------------------------------------
    let mut total = 0.0_f64;
    for i in 0..n_rules {
        total += props[i];
    }

    let props_arr = props.into_pyarray_bound(py).unbind();
    let sat_arr = saturation.into_pyarray_bound(py).unbind();
    Ok((total, props_arr, sat_arr))
}


/// Banker's rounding (round half to even) — matches Python 3 `round()`.
#[inline]
fn banker_round_to_i64(x: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor as i64
    } else if diff > 0.5 {
        (floor + 1.0) as i64
    } else {
        // Exactly .5: pick the even neighbour.
        let f_i = floor as i64;
        if f_i % 2 == 0 { f_i } else { f_i + 1 }
    }
}


/// Python extension module entrypoint.
#[pymodule]
fn cell_sim_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_propensities, m)?)?;
    m.add("__version__", "0.1.0")?;
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
        assert_eq!(banker_round_to_i64(2.7), 3);
        assert_eq!(banker_round_to_i64(-2.7), -3);
        assert_eq!(banker_round_to_i64(99.5), 100);
        assert_eq!(banker_round_to_i64(100.5), 100);  // banker: to even
    }
}
