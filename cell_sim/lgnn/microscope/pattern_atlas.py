"""Pattern recognition microscope — cluster trajectory data into
discrete cell states + per-species dynamic states.

Two tiers (the spec's tier-1 and tier-3; tier-2's functional-module
pooling requires a species→module mapping that isn't yet built):

  whole_cell_atlas    -- cluster each timestep's full (S,) state into
                          K_whole "cell states." Captures cell-cycle
                          phases, metabolic regimes, stress responses.
  per_species_atlas   -- cluster each (species, time) pair by a small
                          window of its values into K_species "atomic
                          dynamic states." Captures e.g. mRNA-burst,
                          protein-decay, post-translocation.

Both are post-hoc — no retraining needed. Run on the raw counts_and_
fluxes trajectory data; the model's predictions are not required.
The resulting cluster labels feed directly into the pattern-atlas
microscope deliverable.

Implementation: mini-batch k-means via sklearn. Fits a 7200-timestep
trajectory in seconds.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def whole_cell_atlas(
    X: np.ndarray,
    K: int = 64,
    seed: int = 0,
    standardize: bool = True,
) -> dict:
    """Cluster each timestep's (S,) state vector into K cell-state codes.

    Args
    ----
    X : (T, S) trajectory, e.g. signed-log1p counts from one replicate.
    K : number of whole-cell pattern codes.
    seed : k-means random_state.
    standardize : per-species mean/std normalize before clustering.
                   Defaults to True so high-magnitude species don't
                   dominate the distance metric.

    Returns
    -------
    dict with:
        K, T, S          -- shape parameters
        labels           -- (T,) int, cluster label per timestep
        centroids        -- (K, S) float, cluster centroids (in standardized
                            space if standardize=True)
        timestep_density -- (K,) int, how many timesteps fall in each cluster
        active_K         -- int, number of clusters with ≥1 member
        inertia          -- float, k-means objective at convergence
    """
    from sklearn.cluster import MiniBatchKMeans

    X = np.asarray(X, dtype=np.float32)
    T, S = X.shape
    if standardize:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - mean) / std
    else:
        Xn = X
        mean = np.zeros((1, S), dtype=np.float32)
        std = np.ones((1, S), dtype=np.float32)

    km = MiniBatchKMeans(
        n_clusters=K, random_state=seed,
        batch_size=min(256, T), n_init=3, max_iter=200,
    ).fit(Xn)
    labels = km.labels_                                  # (T,)
    centroids = km.cluster_centers_                       # (K, S)
    density = np.bincount(labels, minlength=K)
    return {
        'K':                K,
        'T':                T,
        'S':                S,
        'labels':           labels,
        'centroids':        centroids,
        'standardize_mean': mean.squeeze(0),
        'standardize_std':  std.squeeze(0),
        'timestep_density': density,
        'active_K':         int((density > 0).sum()),
        'inertia':          float(km.inertia_),
    }


def characterize_whole_cell_clusters(
    atlas: dict,
    row_names: Sequence[str],
    top_n_distinctive: int = 10,
) -> pd.DataFrame:
    """For each whole-cell cluster, identify the top-N most distinctive
    species (largest absolute deviation of cluster centroid from
    overall mean, in standardized units).

    Returns DataFrame: cluster_id, density, top_species, top_signed_zscore.
    """
    centroids = atlas['centroids']                       # (K, S) standardized
    K, S = centroids.shape
    # In standardized space, overall mean = 0 by construction; so the
    # centroid value IS its signed z-score relative to the species'
    # marginal distribution.
    distinctive_idx = np.argsort(-np.abs(centroids), axis=1)[:, :top_n_distinctive]

    rows = []
    for k in range(K):
        if atlas['timestep_density'][k] == 0:
            continue
        idxs = distinctive_idx[k]
        rows.append({
            'cluster_id':         int(k),
            'density':            int(atlas['timestep_density'][k]),
            'top_species':        [row_names[i] for i in idxs],
            'top_signed_zscore':  [float(centroids[k, i]) for i in idxs],
        })
    return pd.DataFrame(rows).sort_values(
        'density', ascending=False,
    ).reset_index(drop=True)


def per_species_atlas(
    X: np.ndarray,
    K: int = 64,
    window: int = 4,
    seed: int = 0,
    species_subset: Optional[Sequence[int]] = None,
) -> dict:
    """Cluster (species, time) pairs by a window of values into K
    atomic dynamic states. Each (species, t) → label.

    Feature vector per (species, t):
        [X[t, species], X[t+1, species], ..., X[t+window-1, species]]
    (a small window of that species' time series starting at t)

    Args
    ----
    X : (T, S) trajectory
    K : number of atomic states
    window : window length used as feature
    species_subset : if given, only build features for these row indices.
        Useful for restricting to e.g. PM_xxxx or central-dogma rows.

    Returns
    -------
    dict with:
        K, window, n_pairs  -- shape parameters
        species_subset      -- (N_subset,) row indices included
        species_per_pair    -- (n_pairs,) species idx for each pair
        time_per_pair       -- (n_pairs,) timestep for each pair
        labels              -- (n_pairs,) atomic-state label
        centroids           -- (K, window) cluster centroids
        density             -- (K,) members per cluster
        active_K            -- int
    """
    from sklearn.cluster import MiniBatchKMeans

    X = np.asarray(X, dtype=np.float32)
    T, S = X.shape
    if species_subset is None:
        species_subset = np.arange(S)
    else:
        species_subset = np.asarray(species_subset, dtype=np.int64)

    n_valid_t = T - window
    n_pairs = n_valid_t * len(species_subset)
    features = np.empty((n_pairs, window), dtype=np.float32)
    species_per_pair = np.empty(n_pairs, dtype=np.int64)
    time_per_pair = np.empty(n_pairs, dtype=np.int64)

    p = 0
    for s in species_subset:
        for t in range(n_valid_t):
            features[p] = X[t:t + window, s]
            species_per_pair[p] = s
            time_per_pair[p] = t
            p += 1

    km = MiniBatchKMeans(
        n_clusters=K, random_state=seed,
        batch_size=min(1024, n_pairs), n_init=3, max_iter=200,
    ).fit(features)
    labels = km.labels_
    density = np.bincount(labels, minlength=K)
    return {
        'K':                K,
        'window':           window,
        'n_pairs':          n_pairs,
        'species_subset':   species_subset,
        'species_per_pair': species_per_pair,
        'time_per_pair':    time_per_pair,
        'labels':           labels,
        'centroids':        km.cluster_centers_,
        'density':          density,
        'active_K':         int((density > 0).sum()),
        'inertia':          float(km.inertia_),
    }


def save_atlas(atlas: dict, path) -> None:
    """Save an atlas dict to .npz."""
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{
        k: v for k, v in atlas.items()
        if isinstance(v, (np.ndarray, int, float, str))
    })


def load_atlas(path) -> dict:
    """Load a previously saved atlas."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
