"""Head-to-head: bio_neural_net SNN vs a tiny MLP, both pure-Python.

Result (documented honestly):

    MLP   16->8->2  backprop      train 1.00  test 0.94
    SNN   16->2     STDP+teacher  train ~0.50 test ~0.50  (chance)

On a 16-bit binary classification task with class-conditional Bernoulli
features, the MLP solves it cleanly. The single-layer STDP-trained SNN
fails: a strong teacher current forces both outputs to fire at high
rates, every active input -> output pair gets potentiated, all weights
converge to the cap, and the classifier predicts at chance. Adding WTA
inhibition between the outputs causes symmetry breaking - one output
dominates training and silences the other. Tighter caps or lower
learning rates exchange one failure mode for the other.

The honest takeaway: gradient-trained ANNs solve this task in <1 s.
Supervised STDP on a single layer cannot - the local plasticity rule
lacks a way to push weight *differences* between the correct and
incorrect output without a global error signal. Reward-modulated STDP
or hidden-layer feature learning would be needed; both are their own
research projects. Run as `python -m bio_neural_net.benchmark`.
"""

from __future__ import annotations

import math
import random
import time

from .network import NeuralCluster
from .neuron import NeuronParams
from .plasticity import STDP
from .synapse import SynapseKind


# ----------------------- dataset -----------------------

def make_dataset(rng: random.Random, n_per_class: int, p_diff: float):
    """Class 0: left half (bits 0-7) at P(1)=0.5+p_diff/2, right half at 0.5-p_diff/2.
    Class 1: swapped. Larger p_diff = easier."""
    p_hi = 0.5 + p_diff / 2.0
    p_lo = 0.5 - p_diff / 2.0
    X: list[list[int]] = []
    y: list[int] = []
    for label in (0, 1):
        for _ in range(n_per_class):
            ps = ([p_hi] * 8 + [p_lo] * 8) if label == 0 else ([p_lo] * 8 + [p_hi] * 8)
            X.append([1 if rng.random() < p else 0 for p in ps])
            y.append(label)
    order = list(range(len(X)))
    rng.shuffle(order)
    return [X[i] for i in order], [y[i] for i in order]


# ----------------------- MLP baseline -----------------------

def _sigmoid(z: float) -> float:
    if z >= 30.0:
        return 1.0
    if z <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


class MLP:
    """16 -> n_hid -> 2 sigmoid MLP, trained by SGD on cross-entropy."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, rng: random.Random):
        s = 0.5
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        self.W1 = [[rng.gauss(0.0, s) for _ in range(n_in)] for _ in range(n_hid)]
        self.b1 = [0.0] * n_hid
        self.W2 = [[rng.gauss(0.0, s) for _ in range(n_hid)] for _ in range(n_out)]
        self.b2 = [0.0] * n_out

    def forward(self, x):
        h = [
            _sigmoid(sum(self.W1[j][i] * x[i] for i in range(self.n_in)) + self.b1[j])
            for j in range(self.n_hid)
        ]
        y = [
            _sigmoid(sum(self.W2[k][j] * h[j] for j in range(self.n_hid)) + self.b2[k])
            for k in range(self.n_out)
        ]
        return h, y

    def train_one(self, x, target_idx: int, lr: float = 0.5):
        h, y = self.forward(x)
        target = [0.0] * self.n_out
        target[target_idx] = 1.0
        d2 = [(y[k] - target[k]) * y[k] * (1.0 - y[k]) for k in range(self.n_out)]
        d1 = [
            h[j] * (1.0 - h[j]) * sum(self.W2[k][j] * d2[k] for k in range(self.n_out))
            for j in range(self.n_hid)
        ]
        for k in range(self.n_out):
            for j in range(self.n_hid):
                self.W2[k][j] -= lr * d2[k] * h[j]
            self.b2[k] -= lr * d2[k]
        for j in range(self.n_hid):
            for i in range(self.n_in):
                self.W1[j][i] -= lr * d1[j] * x[i]
            self.b1[j] -= lr * d1[j]

    def predict(self, x) -> int:
        _, y = self.forward(x)
        return max(range(self.n_out), key=lambda k: y[k])


def train_mlp(X, y, epochs: int, rng: random.Random) -> MLP:
    net = MLP(n_in=16, n_hid=8, n_out=2, rng=rng)
    order = list(range(len(X)))
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            net.train_one(X[i], y[i])
    return net


def evaluate(predict_fn, X, y) -> float:
    return sum(1 for xi, yi in zip(X, y) if predict_fn(xi) == yi) / len(X)


# ----------------------- SNN classifier -----------------------

class SNNClassifier:
    """16 input + 2 output LIF neurons, STDP-trained with a teacher current."""

    PRESENT_MS = 80.0
    DT = 1.0
    INPUT_DRIVE = 4.0    # nA, slow input firing so synaptic drive is moderate
    TEACHER_DRIVE = 12.0 # nA, target fires immediately and dominates the WTA

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.net = NeuralCluster(rng=rng)
        self.inputs = self.net.add_neurons(16)
        self.outputs = self.net.add_neurons(2)

        # Start weights very low so inputs alone cannot drive outputs.
        self.in_syns: list[list] = []
        for i in self.inputs:
            row = []
            for o in self.outputs:
                row.append(
                    self.net.connect(
                        i, o, SynapseKind.EXCITATORY, g_max=0.01, tau_syn=5.0
                    )
                )
            self.in_syns.append(row)

        # Strong lateral inhibition between outputs. When the teacher drives
        # output k, output (1-k) is suppressed and its incoming synapses
        # are not potentiated. Without this, once teacher-direction weights
        # grow, synaptic drive alone fires the non-teacher output too and
        # STDP starts potentiating everything.
        for a in self.outputs:
            for b in self.outputs:
                if a != b:
                    self.net.connect(
                        a, b, SynapseKind.INHIBITORY, g_max=6.0, tau_syn=12.0
                    )

        self.net.enable_plasticity(
            STDP(A_plus=0.002, A_minus=0.0005, g_max_cap=1.0)
        )

    def _reset_state(self) -> None:
        for n in self.net.neurons:
            n.reset()
        for s in self.net.synapses:
            s.g = 0.0
            s._pending.clear()
        if self.net.stdp is not None:
            self.net.stdp._pre_traces.clear()
            self.net.stdp._post_traces.clear()

    def _present(self, x, teacher: int | None):
        active = [self.inputs[i] for i, b in enumerate(x) if b == 1]
        out_ids = self.outputs

        def I(_t: float):
            row = [0.0] * (16 + 2)
            for nid in active:
                row[nid] = self.INPUT_DRIVE
            if teacher is not None:
                row[out_ids[teacher]] = self.TEACHER_DRIVE
            return row

        return self.net.run(self.PRESENT_MS, dt=self.DT, external_current=I)

    def fit(self, X, y, epochs: int) -> None:
        order = list(range(len(X)))
        for _ in range(epochs):
            self.rng.shuffle(order)
            for i in order:
                self._reset_state()
                self._present(X[i], teacher=y[i])

    def predict(self, x) -> int:
        save = self.net.stdp
        self.net.stdp = None
        self._reset_state()
        rec = self._present(x, teacher=None)
        self.net.stdp = save
        counts = [len(rec.for_neuron(o)) for o in self.outputs]
        if counts[0] == counts[1]:
            return self.rng.randrange(2)
        return max(range(2), key=lambda k: counts[k])

    def summarize_weights(self) -> tuple[list[float], list[float]]:
        c0 = [self.in_syns[i][0].g_max for i in range(16)]
        c1 = [self.in_syns[i][1].g_max for i in range(16)]
        return c0, c1


# ----------------------- main -----------------------

def main() -> None:
    rng = random.Random(42)
    X_train, y_train = make_dataset(rng, n_per_class=30, p_diff=0.6)
    X_test, y_test = make_dataset(rng, n_per_class=50, p_diff=0.4)

    print("Task: 16-bit binary classification, class-conditional Bernoulli features.")
    print(f"  train: {len(X_train)} examples (P(1)=0.8 in correct half, 0.2 in other)")
    print(f"  test:  {len(X_test)} examples (harder: 0.7 vs 0.3, more overlap)")
    print()

    print("Training MLP (16 -> 8 -> 2, backprop, 200 epochs)...", flush=True)
    t0 = time.time()
    mlp = train_mlp(X_train, list(y_train), epochs=200, rng=random.Random(1))
    t_mlp = time.time() - t0
    mlp_train = evaluate(mlp.predict, X_train, y_train)
    mlp_test = evaluate(mlp.predict, X_test, y_test)
    print(f"  done in {t_mlp:.2f}s   train={mlp_train:.3f}  test={mlp_test:.3f}")

    print("Training SNN (16 -> 2, STDP + teacher + WTA, 5 epochs)...", flush=True)
    t0 = time.time()
    snn = SNNClassifier(rng=random.Random(1))
    snn.fit(X_train, y_train, epochs=5)
    t_snn = time.time() - t0
    snn_train = evaluate(snn.predict, X_train, y_train)
    snn_test = evaluate(snn.predict, X_test, y_test)
    print(f"  done in {t_snn:.2f}s   train={snn_train:.3f}  test={snn_test:.3f}")

    print()
    print("=" * 64)
    print(f"{'model':<8} {'params':>8} {'train_s':>8} {'train acc':>11} {'test acc':>10}")
    print("-" * 64)
    mlp_params = 16 * 8 + 8 + 8 * 2 + 2
    snn_params = 16 * 2   # plastic synapses, ignoring the lateral inhibition
    print(f"{'MLP':<8} {mlp_params:>8} {t_mlp:>8.2f} {mlp_train:>11.3f} {mlp_test:>10.3f}")
    print(f"{'SNN':<8} {snn_params:>8} {t_snn:>8.2f} {snn_train:>11.3f} {snn_test:>10.3f}")
    print()

    c0, c1 = snn.summarize_weights()
    print("SNN learned synapse strengths (g_max, input -> output):")
    print("  in:  " + " ".join(f"{i:>4d}" for i in range(16)))
    print("  ->A: " + " ".join(f"{w:>4.2f}" for w in c0))
    print("  ->B: " + " ".join(f"{w:>4.2f}" for w in c1))


if __name__ == "__main__":
    main()
