"""Datasets for the count-dynamics task.

`CountsTransitionDataset` streams (x_t, Δx) pairs one replicate at a time
— minimal RAM, batches are highly correlated within a replicate.

`BufferedShuffleDataset` pre-loads several replicates into a ring buffer
and samples uniformly across the buffer, so consecutive batches see
transitions from different stochastic realisations of the same starting
state. The decorrelation matters: with one-replicate-at-a-time, SGD's
gradient is averaged over ~28 batches of correlated samples before any
new replicate is seen, which biases AdamW's running statistics.

The transform is `signed_log1p` rather than plain `log1p` because the
8572 rows of counts_and_fluxes are a mix of species counts (nonneg ints)
and reaction fluxes (signed floats, NaN-when-idle).
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset


def replicate_to_log1p_array(df) -> np.ndarray:
    """(n_species, n_timepoints) DataFrame -> (n_timepoints, n_species)
    float32 ndarray, signed-log1p-transformed.

    nan_to_num first (idle flux is zero), then sign(x)·log1p(|x|) — the
    standard symmetric extension that handles negatives without
    artefacts at zero. Always copies; in-place transform would otherwise
    alias back into the DataFrame buffer.
    """
    arr = np.ascontiguousarray(df.to_numpy(dtype=np.float32, copy=True).T)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    sign = np.sign(arr)
    np.abs(arr, out=arr)
    np.log1p(arr, out=arr)
    np.multiply(arr, sign, out=arr)
    return arr


class CountsTransitionDataset(IterableDataset):
    """One replicate at a time, sequential or shuffled within."""

    def __init__(self, replicate_indices: Sequence[int],
                 lsdata_module,
                 species_filter: Optional[Sequence[str]] = None,
                 shuffle_within_replicate: bool = True,
                 seed: int = 0):
        super().__init__()
        self.replicate_indices = list(replicate_indices)
        self.lsdata = lsdata_module
        self.species_filter = (list(species_filter)
                                if species_filter is not None else None)
        self.shuffle = shuffle_within_replicate
        self.seed = seed

    def __iter__(self) -> Iterator:
        rng = np.random.default_rng(self.seed)
        for rep_i in self.replicate_indices:
            df = self.lsdata.load_replicate(rep_i, species=self.species_filter)
            arr = replicate_to_log1p_array(df)             # (T, S) float32
            order = np.arange(arr.shape[0] - 1)
            if self.shuffle:
                rng.shuffle(order)
            for j in order:
                x_t = arr[j]
                dx  = arr[j + 1] - arr[j]
                yield (torch.from_numpy(x_t), torch.from_numpy(dx))


class BufferedShuffleDataset(IterableDataset):
    """Cross-replicate shuffler with an in-memory ring buffer.

    Holds `buffer_size` replicates' worth of (T, S) arrays in RAM at
    once (~250 MB each on the Luthey-Schulten data). On every yield it
    picks a uniformly random buffer slot, pops one transition, and when
    a slot empties refills it with the next replicate. Net effect:
    consecutive batches mix transitions from different replicates, so
    SGD sees IID-ish samples instead of 28-batch-correlated runs of one
    replicate.

    Memory: buffer_size × ~250 MB (replicate array) + ~60 MB
    (transition-index list). Default buffer_size=3 → ~750 MB RAM.
    """

    def __init__(self, replicate_indices: Sequence[int],
                 lsdata_module,
                 species_filter: Optional[Sequence[str]] = None,
                 buffer_size: int = 3,
                 seed: int = 0):
        super().__init__()
        self.replicate_indices = list(replicate_indices)
        self.lsdata = lsdata_module
        self.species_filter = (list(species_filter)
                                if species_filter is not None else None)
        self.buffer_size = max(1, int(buffer_size))
        self.seed = seed

    def _load(self, rep_i: int):
        df = self.lsdata.load_replicate(rep_i, species=self.species_filter)
        return replicate_to_log1p_array(df)

    def __iter__(self) -> Iterator:
        rng = np.random.default_rng(self.seed)
        rep_order = list(rng.permutation(self.replicate_indices))
        rep_iter = iter(rep_order)
        buffer: list = []                                  # [arr, [indices...]]

        for _ in range(min(self.buffer_size, len(rep_order))):
            try:
                rep_i = next(rep_iter)
            except StopIteration:
                break
            arr = self._load(rep_i)
            idx = list(range(arr.shape[0] - 1))
            rng.shuffle(idx)
            buffer.append([arr, idx])

        while buffer:
            slot = int(rng.integers(0, len(buffer)))
            arr, idx_list = buffer[slot]
            j = idx_list.pop()
            yield (torch.from_numpy(arr[j]),
                   torch.from_numpy(arr[j + 1] - arr[j]))
            if not idx_list:
                try:
                    rep_i = next(rep_iter)
                except StopIteration:
                    buffer.pop(slot)
                    continue
                arr = self._load(rep_i)
                new_idx = list(range(arr.shape[0] - 1))
                rng.shuffle(new_idx)
                buffer[slot] = [arr, new_idx]
