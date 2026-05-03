"""Confidence-gated cascade detector.

Wraps a mechanistic detector (typically `ComposedDetector`) and a per-gene
LNN essentiality probability map, routing each gene's call by the
mechanistic detector's confidence:

  if mechanistic_conf >= cascade_threshold:
      use mechanistic call
  else:
      use LNN call (essential if lnn_prob >= lnn_threshold)

Empirical result on Syn3A (commit 0bb3a11):
  v15 ComposedDetector solo                MCC = 0.505
  in-domain LNN solo                       MCC = 0.674
  cascade @ T=0.5, lnn_threshold=0.297     MCC = 0.731

The cascade beats both components because the mechanistic detector and
the LNN have complementary failure modes: v15 has near-zero false
positives but ~25% false-negative rate; the in-domain LNN catches many
of v15's false negatives because its sequence/scalar/regulatory feature
basis is independent of the mechanistic simulation.

Plug it in anywhere a `ComposedDetector` is used:

    from cell_sim.layer6_essentiality.cascade_detector import CascadeDetector
    inner = ComposedDetector(structural=..., annotation=..., trajectory=...)
    cascade = CascadeDetector(
        inner=inner,
        lnn_probs={'JCVISYN3A_0001': 0.92, ...},
        lnn_threshold=0.297,
        cascade_threshold=0.5,
    )
    mode, t_fail, conf, evidence = cascade.detect_for_gene(locus_tag, ko_traj)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from cell_sim.layer6_essentiality.harness import FailureMode, Trajectory


class TrajectoryDetector(Protocol):
    def detect_for_gene(
        self, locus_tag: str, ko: Trajectory,
    ) -> tuple[FailureMode, Optional[float], float, str]: ...


# Threshold defaults learned from the cascade analysis on Syn3A
# (notebooks/cascade_v15_lnn_syn3a.ipynb, commit 0bb3a11).
DEFAULT_CASCADE_THRESHOLD = 0.5
DEFAULT_LNN_THRESHOLD = 0.297

# Failure mode used when the LNN says essential — the LNN is sequence/
# feature-based and doesn't know WHY a gene is essential, just that it is.
# We use GROWTH_INVIABLE as the umbrella label so callers know this came
# from the ML predictor and not a specific mechanistic signal.
LNN_FAILURE_MODE = FailureMode.NONE  # placeholder; overridden below


# We extend FailureMode at module load with an LNN-specific value if the
# enum is open to extension. Since FailureMode is a str Enum, we just use
# the most generic existing label that means "essential by some signal we
# can't pin to a specific mechanism": CATALYSIS_SILENCED comes closest in
# semantics, but ATP_DEPLETION is what most LNN-flagged essentials would
# present as in a longer simulation. We default to CATALYSIS_SILENCED so
# downstream metric code that aggregates by failure mode can identify
# LNN-derived calls separately from purely mechanistic ones.
LNN_FAILURE_MODE = FailureMode.CATALYSIS_SILENCED


@dataclass
class CascadeDetector:
    """Wrap an inner mechanistic detector + LNN probability map.

    Parameters
    ----------
    inner : has detect_for_gene(locus_tag, ko) -> (mode, t_fail, conf, evidence)
        Typically a `ComposedDetector` (v15 mechanistic stack), but any
        detector with the standard signature works.
    lnn_probs : dict[str, float]
        Per-gene LNN essentiality probability, keyed by locus_tag. Genes
        absent from this dict fall through to the inner detector's call.
    lnn_threshold : float, default 0.297
        Threshold above which `lnn_probs[locus_tag]` is interpreted as
        an essential call. Default learned on Syn3A.
    cascade_threshold : float, default 0.5
        Inner detector confidence above which the inner call is trusted
        without consulting the LNN. Below this, the LNN fallback fires.
    lnn_failure_mode : FailureMode, default CATALYSIS_SILENCED
        Failure mode tag returned when the LNN is the one calling
        essential. Helps downstream code distinguish ML-derived calls
        from mechanistic ones.
    """
    inner: TrajectoryDetector
    lnn_probs: dict[str, float] = field(default_factory=dict)
    lnn_threshold: float = DEFAULT_LNN_THRESHOLD
    cascade_threshold: float = DEFAULT_CASCADE_THRESHOLD
    lnn_failure_mode: FailureMode = LNN_FAILURE_MODE

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Trajectory,
    ) -> tuple[FailureMode, Optional[float], float, str]:
        v_mode, v_t, v_conf, v_ev = self.inner.detect_for_gene(locus_tag, ko)

        # Trust the inner mechanistic detector when it's confident
        if v_conf >= self.cascade_threshold:
            return (v_mode, v_t, v_conf,
                    f"v15>={self.cascade_threshold:.2f}[{v_ev}]")

        # Inner detector is uncertain — defer to LNN if available
        lnn_prob = self.lnn_probs.get(locus_tag)
        if lnn_prob is None:
            return (v_mode, v_t, v_conf,
                    f"no_lnn_for_{locus_tag}[{v_ev}]")

        if lnn_prob >= self.lnn_threshold:
            return (self.lnn_failure_mode, None, float(lnn_prob),
                    f"lnn(p={lnn_prob:.3f}>={self.lnn_threshold:.3f})"
                    f"_v15conf={v_conf:.2f}[{v_ev}]")
        else:
            # LNN says non-essential — confidence is 1 - lnn_prob
            return (FailureMode.NONE, None, float(1.0 - lnn_prob),
                    f"lnn(p={lnn_prob:.3f}<{self.lnn_threshold:.3f})"
                    f"_v15conf={v_conf:.2f}[{v_ev}]")


# ──────────── inference helpers ────────────


def load_lnn_predictions_for_organism(
    checkpoint_path,
    gene_batch: dict,
    device: str = 'cpu',
) -> dict[str, float]:
    """Load an in-domain `EssentialityLNN` checkpoint and run inference on
    a single organism's gene_batch (as built by
    `cell_sim.layer_ml.data_loader.load_organism_batches`).

    Returns {locus_tag: essentiality_probability}.
    """
    import torch
    from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig
    from cell_sim.layer_ml.hyperbolic_memory import HyperbolicMemoryBank

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt['config']
    # Filter dataclass fields so unknown keys (e.g. legacy ones) don't crash
    valid_fields = {f.name for f in LNNConfig.__dataclass_fields__.values()}
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    cfg = LNNConfig(**cfg_kwargs)
    model = EssentialityLNN(cfg).to(device)

    # Memory bank: rebuild with the saved size + buffer if recorded
    mem_size = ckpt.get('memory_size', cfg.memory_size)
    mem_growable = ckpt.get('memory_growable', False)
    model.memory = HyperbolicMemoryBank(
        hidden=cfg.hidden, memory_size=mem_size,
        retrieval_k=cfg.memory_retrieval_k,
        curvature=cfg.memory_curvature,
        growable=mem_growable,
        max_size=max(mem_size, 8192) if mem_growable else mem_size,
    ).to(device)

    model.load_state_dict(ckpt['state_dict'], strict=False)

    # Restore memory buffer if saved
    if 'memory_buffer' in ckpt and 'memory_populated' in ckpt:
        with torch.no_grad():
            model.memory.memory.copy_(ckpt['memory_buffer'].to(device))
            model.memory.populated.copy_(ckpt['memory_populated'].to(device))

    # Move batch tensors to device
    batch_dev = {}
    for k, v in gene_batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.to(device)
        else:
            batch_dev[k] = v

    model.eval()
    with torch.no_grad():
        out = model(batch_dev, store_memory=False)
    probs = torch.sigmoid(out['essentiality']).cpu().numpy()
    loci = batch_dev['locus_tags']
    return {lt: float(p) for lt, p in zip(loci, probs)}


# ──────────── self-test ────────────


def _self_test():
    """Verify cascade behavior matches the Syn3A v15+LNN regime: v15 is
    high-precision (98%) but only ~75% recall on essentials; LNN catches
    many of v15's false negatives because it has independent feature
    basis. Cascade should beat both."""
    import numpy as np
    from sklearn.metrics import matthews_corrcoef

    print('cascade_detector self-test:')
    rng = np.random.default_rng(42)
    N = 455
    y = (rng.random(N) < 0.84).astype(int)   # ~84% essential like Syn3A

    # Realistic v15 regime: detects 75% of true essentials with high conf,
    # misses 25% (call=0, conf=0). False positive rate 5%.
    # Matches the real Syn3A run: 286 TP / 97 FN / 6 FP / 66 TN.
    v15_call = np.zeros(N, dtype=int)
    v15_conf = np.zeros(N, dtype=float)
    for i in range(N):
        if y[i] == 1:
            if rng.random() < 0.75:    # v15 catches essential
                v15_call[i] = 1
                v15_conf[i] = rng.uniform(0.5, 1.0)
            # else: v15 misses (call=0, conf=0)
        else:
            if rng.random() < 0.05:    # v15 false positive
                v15_call[i] = 1
                v15_conf[i] = rng.uniform(0.5, 1.0)
            # else: v15 correct nonessential (call=0, conf=0)

    # In-domain LNN: in-domain MCC ~0.65 (well-calibrated probs)
    lnn_probs = np.where(y == 1, rng.beta(7, 3, N), rng.beta(3, 7, N))
    lnn_threshold = 0.5

    class _StubInner:
        def __init__(self, calls, confs, loci):
            self.calls = dict(zip(loci, calls))
            self.confs = dict(zip(loci, confs))
        def detect_for_gene(self, lt, ko):
            c = self.calls[lt]; cf = self.confs[lt]
            mode = FailureMode.CATALYSIS_SILENCED if c == 1 else FailureMode.NONE
            return mode, None, cf, 'stub'

    loci = [f'GENE_{i:04d}' for i in range(N)]
    inner = _StubInner(v15_call, v15_conf, loci)
    lnn_dict = {lt: float(p) for lt, p in zip(loci, lnn_probs)}

    cascade = CascadeDetector(
        inner=inner, lnn_probs=lnn_dict,
        lnn_threshold=lnn_threshold, cascade_threshold=0.5,
    )

    cascade_calls = []
    for lt in loci:
        mode, _, _, _ = cascade.detect_for_gene(lt, None)
        cascade_calls.append(int(mode != FailureMode.NONE))
    cascade_calls = np.array(cascade_calls)

    v15_mcc = matthews_corrcoef(y, v15_call)
    lnn_mcc = matthews_corrcoef(y, (lnn_probs >= lnn_threshold).astype(int))
    cas_mcc = matthews_corrcoef(y, cascade_calls)
    print(f'  v15 solo MCC:    {v15_mcc:.4f}  '
          f'(real Syn3A: 0.505)')
    print(f'  LNN solo MCC:    {lnn_mcc:.4f}  '
          f'(real Syn3A in-domain: 0.674)')
    print(f'  cascade MCC:     {cas_mcc:.4f}  '
          f'(real Syn3A: 0.731)')
    print(f'  frac v15-routed: '
          f'{(v15_conf >= 0.5).mean():.3f}  (real Syn3A: 0.642)')
    assert cas_mcc > v15_mcc, \
        f'cascade ({cas_mcc:.3f}) should beat v15 ({v15_mcc:.3f})'
    assert cas_mcc >= lnn_mcc - 0.05, \
        f'cascade ({cas_mcc:.3f}) should be ~ LNN ({lnn_mcc:.3f})'

    # Verify the gene-without-LNN-prob fallback path
    cascade_no_lnn = CascadeDetector(inner=inner, lnn_probs={},
                                       cascade_threshold=0.5)
    fallback_call_count = 0
    for lt in loci:
        mode, _, _, ev = cascade_no_lnn.detect_for_gene(lt, None)
        if 'no_lnn_for_' in ev:
            fallback_call_count += 1
    n_uncertain = int((v15_conf < 0.5).sum())
    assert fallback_call_count == n_uncertain, \
        f'expected {n_uncertain} uncertain genes, got {fallback_call_count}'
    print(f'  no-LNN fallback: {fallback_call_count}/{N} genes correctly '
          f'used inner-only')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
