"""spliceai_torch — the REAL pretrained SpliceAI (Jaganathan et al., Cell 2019) run in PyTorch, no TensorFlow/Keras.

Loads Illumina's 5 trained SpliceAI-10k models (spliceai1..5.h5, bundled in the official `spliceai` PyPI wheel) by parsing
the Keras functional-model graph and re-executing it with torch ops + the original weights. Architecture (confirmed from the
h5 model_config): pre-activation dilated-ResNet -- Conv1D(32,1) stem + a skip Conv1D, 16 residual units in 4 dilation groups
(W=[11,11,11,11,21,21,21,21,41,41,41,41,...], dilation=[1,4,10,25] per group of 4), skip Conv1D every 4 units summed, then
Cropping1D(5000/side) and Conv1D(3,1)+softmax -> per-base P(null,acceptor,donor). Ensembles the 5 models by averaging.

This is a faithful re-host of the published weights (NOT a retrain and NOT an approximation): every Conv1D/BatchNorm weight is
the original, loaded by name; only the framework changed (Keras->torch) so it runs in this torch-only sandbox.
"""
import json, zipfile
from pathlib import Path
import numpy as np

WHL = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/spliceai_dl/spliceai-1.3.1-py2.py3-none-any.whl")
MODELS_DIR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/spliceai_dl/spliceai/models")
CONTEXT = 10000                     # SpliceAI-10k total flanking (5000 each side)
IN_MAP = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
_BASE = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}


def _ensure_models():
    """make the 5 trained SpliceAI models available. Uses the extracted .h5 if present; else the local wheel; else
    downloads the official `spliceai` wheel from PyPI (scratchpad is ephemeral, so a fresh session self-heals)."""
    if all((MODELS_DIR / f"spliceai{i}.h5").exists() for i in range(1, 6)):
        return
    MODELS_DIR.parent.mkdir(parents=True, exist_ok=True)
    whl = WHL
    if not whl.exists():
        import subprocess
        WHL.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["python3", "-m", "pip", "download", "spliceai==1.3.1", "--no-deps", "-d", str(WHL.parent)],
                       check=True, capture_output=True)
        cands = list(WHL.parent.glob("spliceai-*.whl"))
        whl = cands[0] if cands else WHL
    with zipfile.ZipFile(whl) as z:
        for n in z.namelist():
            if n.endswith(".h5"):
                z.extract(n, MODELS_DIR.parent.parent)   # wheel path is spliceai/models/*.h5


def one_hot(seq):
    """A,C,G,T,N string -> (len,4) one-hot in SpliceAI's channel order (A,C,G,T)."""
    codes = np.fromiter((_BASE.get(c, 0) for c in seq.upper()), dtype=np.int64, count=len(seq))
    return IN_MAP[codes]


class SpliceAITorch:
    """A single SpliceAI model re-executed in torch from its Keras h5. Call predict(one_hot) -> (L,3) probabilities."""
    def __init__(self, h5path):
        import h5py, torch
        self.torch = torch
        f = h5py.File(h5path, "r")
        cfg = f.attrs["model_config"]
        self.graph = json.loads(cfg if isinstance(cfg, str) else cfg.decode())["config"]["layers"]
        mw = f["model_weights"]
        self.W = {}          # layer_name -> dict of numpy arrays
        for lname in mw.keys():
            g = mw[lname]
            d = {}
            def collect(gg):
                for k in gg.keys():
                    it = gg[k]
                    if isinstance(it, h5py.Dataset):
                        d[k.split("/")[-1]] = np.array(it)
                    else:
                        collect(it)
            collect(g)
            if d:
                self.W[lname] = d
        f.close()

    def predict(self, oh):
        """oh: (L+CONTEXT, 4) numpy one-hot. Returns (L, 3) numpy probabilities (null, acceptor, donor)."""
        torch = self.torch
        import torch.nn.functional as F
        x0 = torch.from_numpy(oh.T[None].astype(np.float32))     # (1, 4, Lfull)  channels-first for torch conv
        tensors = {}
        out_name = None
        for layer in self.graph:
            cls = layer["class_name"]; name = layer["config"]["name"]
            inb = layer.get("inbound_nodes", [])
            if cls == "InputLayer":
                tensors[name] = x0; out_name = name; continue
            # resolve input tensor(s)
            src = inb[0]
            ins = [tensors[n[0]] for n in src] if isinstance(src, list) else [tensors[src["args"][0]["config"]["keras_history"][0]]]
            x = ins[0]
            if cls == "Conv1D":
                w = self.W[name]
                k = torch.from_numpy(np.transpose(w["kernel:0"], (2, 1, 0)).copy())   # (out,in,k)
                b = torch.from_numpy(w["bias:0"].copy())
                dil = layer["config"]["dilation_rate"][0]; ks = layer["config"]["kernel_size"][0]
                pad = dil * (ks - 1) // 2                                             # 'same' for odd kernels
                x = F.conv1d(x, k, b, padding=pad, dilation=dil)
                if layer["config"].get("activation") == "softmax":
                    x = F.softmax(x, dim=1)
            elif cls == "BatchNormalization":
                w = self.W[name]
                x = F.batch_norm(x, torch.from_numpy(w["moving_mean:0"].copy()), torch.from_numpy(w["moving_variance:0"].copy()),
                                 torch.from_numpy(w["gamma:0"].copy()), torch.from_numpy(w["beta:0"].copy()),
                                 training=False, eps=1e-3)
            elif cls == "Activation":
                x = F.relu(x) if layer["config"]["activation"] == "relu" else x
            elif cls == "Add":
                x = sum(ins)
            elif cls == "Cropping1D":
                c = layer["config"]["cropping"]; a, b2 = c[0], c[1]
                x = x[:, :, a:x.shape[2] - b2]
            else:
                raise ValueError(f"unhandled layer {cls}")
            tensors[name] = x; out_name = name
        y = tensors[out_name][0].detach().numpy().T                                  # (L,3)
        return y


_ENSEMBLE = None
def load_ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ensure_models()
        _ENSEMBLE = [SpliceAITorch(MODELS_DIR / f"spliceai{i}.h5") for i in range(1, 6)]
    return _ENSEMBLE


def predict_probs(seq):
    """seq must be length L+CONTEXT (5000 flank each side already included). Returns (L,3) ensemble-averaged probs."""
    oh = one_hot(seq)
    models = load_ensemble()
    ys = [m.predict(oh) for m in models]
    return np.mean(ys, axis=0)


def splice_sites(seq_with_flank, thresh=0.5):
    """convenience: positions (0-based in the L target) predicted as acceptor/donor above thresh."""
    p = predict_probs(seq_with_flank)
    acc = [(i, float(p[i, 1])) for i in range(len(p)) if p[i, 1] > thresh]
    don = [(i, float(p[i, 2])) for i in range(len(p)) if p[i, 2] > thresh]
    return {"acceptor": acc, "donor": don, "probs": p}


if __name__ == "__main__":
    import sys
    # sanity: build model, run on a synthetic sequence, confirm valid probabilities
    L = 200
    rng = np.random.RandomState(0)
    seq = "".join(rng.choice(list("ACGT"), size=L + CONTEXT))
    p = predict_probs(seq)
    print(f"SpliceAI-torch OK: output shape {p.shape}, each row sums to ~1 (mean {p.sum(1).mean():.4f})")
    print(f"  max acceptor prob {p[:,1].max():.3f}, max donor prob {p[:,2].max():.3f} (random seq -> should be low)")
