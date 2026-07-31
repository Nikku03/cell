"""DOES A TRANSFORMER USE THE SIGN? Attention over a knockout's interaction context, with and without it.

WHAT IS ALREADY KNOWN HERE, so this test starts from the right place. `transformer_ko.py` established two
things on held-out K562, and they point in opposite directions:

    transformer over the knockout's interaction neighbourhood   0.490
    plain MLP on the knockout's OWN features                    0.400
    transformer on its own features, no neighbourhood           0.332   <- LOSES to the MLP
    same transformer with RANDOM partners                       0.155

So the architecture won nothing on its own -- it lost to a multilayer perceptron. What won was WHAT IT
ATTENDED OVER, and the random-partner control puts a number on that: +0.335. A transformer here is a device
for reading interaction context, not a better function approximator.

WHAT IS NEW AND WHY IT IS WORTH ONE MORE TEST. Every context layer that model attended over was UNSIGNED --
PPI, co-dependency and complex membership say two genes are related, not which way one moves the other. This
project's admission gates then found exactly that shape of failure: the curated regulatory layer carries
sign = 0 on 91.2% of its edges and moved a bounded task by -0.0002, and the nuisance ladder stalls because
nothing in it supplies a direction.

`reg_perturb_k562` is the first layer in this model with a validated, transferable SIGN. So the question is
narrow and answerable: given that attention over context is the thing that worked, does a SIGNED context beat
an UNSIGNED one of the same size?

THE ARMS, each differing from its neighbour by one thing:
    1  MLP, own features            the bar the architecture has to clear at all
    2  transformer, self-only       isolates the architecture from the context
    3  transformer + unsigned       PPI / complex / co-dependency partners, no direction
    4  transformer + signed         the same budget, but partners carry the edge's sign and magnitude
    5  transformer + shuffled sign  arm 4 with the signs permuted -- the decisive control

Arm 5 is what makes arm 4 readable. A signed token carries an extra number, and an extra number can help a
model for reasons having nothing to do with what the number means. Permuting the signs while keeping the same
partners, the same token count and the same magnitudes destroys only the DIRECTION. If arm 4 does not beat
arm 5, then the sign is decoration.

NOT CIRCULAR, and the design is bent to make sure. `reg_perturb_k562` is derived from Replogle K562, so
scoring it on K562 would be reporting its own training data -- which is exactly why the RPE1 gate exists. The
target here is therefore RPE1 perturbation response, from a different cell line the edges were not built
from. Folds are grouped by perturbation, so a perturbation never appears in training and test at once.

THE TARGET IS PROFILE SIMILARITY, NOT A POINT PREDICTION. `neural_ko.py` established why: regressing a sparse
high-dimensional response directly collapses to the per-gene mean, because that is the unique optimum of MSE
against such a target, and the network stops using perturbation identity at all (0.214 against a 0.264 floor,
with a feature-shuffle that scored identically). Metric learning on profile similarity avoids that failure
mode, and it is the formulation that worked before.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
PB = SP / "rpe1_pseudobulk.npz"
NET = SP / "cell_complete.json.gz"
REGL = SP / "cell_reg_perturb.json.gz"
MAX_TOK = 16              # partners per knockout; identical across arms so capacity is held constant
DIM = 64
EPOCHS = 60
FOLDS = 5
SEEDS = (0, 1, 2)


def build(report):
    z = np.load(PB, allow_pickle=True)
    L = z["lfc"].astype(np.float32)
    perts = np.array([str(x) for x in z["perts"]])
    genes = np.array([str(x) for x in z["genes"]])
    report(f"  RPE1: {L.shape[0]:,} perturbations x {L.shape[1]:,} genes")

    # per-gene robust z removes the "responds to everything" column effect before similarity is computed
    med = np.median(L, 0)
    mad = np.median(np.abs(L - med), 0) * 1.4826
    Z = (L - med) / np.where(mad < 1e-6, np.nan, mad)
    ok = np.isfinite(Z).all(0)
    Z = Z[:, ok]
    report(f"  {int(ok.sum()):,} genes with a usable robust z")

    from entity_registry import load as load_reg
    REG = load_reg()
    d = json.load(gzip.open(NET, "rt"))
    name = [g["name"] for g in d["genes"]]
    idx = {n: i for i, n in enumerate(name)}
    ppm = d.get("ppm", {})
    feat = {}
    for i, g in enumerate(d["genes"]):
        feat[g["name"]] = [float(g.get("dep_frac") or 0), np.log10(1 + float(ppm.get(str(i), 0) or 0)),
                           float(g.get("loeuf") or 1), float(g.get("tf") or 0),
                           np.log1p(float(g.get("ppi") or 0)), np.log1p(float(g.get("pubs") or 0))]
    unsigned = {}
    for a, b in [(e[0], e[1]) for e in d["ppi"]]:
        unsigned.setdefault(name[int(a)], set()).add(name[int(b)])
        unsigned.setdefault(name[int(b)], set()).add(name[int(a)])
    for a, v in (d.get("codep") or {}).items():
        for b, _s in v:
            unsigned.setdefault(name[int(a)], set()).add(name[int(b)])
    del d

    rp = json.load(gzip.open(REGL, "rt"))
    signed = {}
    for e in rp["edges"]:
        s = e["source"]
        eid = REG.resolve("gene", "ensembl", e["target_ensembl"])
        ent = REG.entities.get(eid) if eid else None
        t = ent.get("symbol") if ent else None
        if t and t in feat:
            signed.setdefault(s, []).append((t, float(e["sign"]), abs(float(e["z"]))))
    report(f"  context: {len(unsigned):,} genes with unsigned partners, "
           f"{len(signed):,} with signed perturbational targets")

    keep = [i for i, p in enumerate(perts) if p in feat]
    report(f"  {len(keep):,} RPE1 perturbations join the gene universe")
    return dict(Z=Z[keep], perts=perts[keep], feat=feat, unsigned=unsigned, signed=signed, REG=REG)


def tokens(B, arm, rng):
    """(n_perturbations, MAX_TOK, F) token tensor. Token 0 is always the perturbed gene itself."""
    F = 6 + 3                                   # gene features + [is_self, sign, |z|]
    P = B["perts"]
    X = np.zeros((len(P), MAX_TOK, F), np.float32)
    M = np.zeros((len(P), MAX_TOK), np.float32)
    for i, p in enumerate(P):
        X[i, 0, :6] = B["feat"][p]
        X[i, 0, 6] = 1.0
        M[i, 0] = 1.0
        if arm == "self":
            continue
        if arm == "unsigned":
            part = [(q, 0.0, 0.0) for q in sorted(B["unsigned"].get(p, ()))]
        elif arm == "random":
            allg = list(B["feat"])
            part = [(allg[int(rng.integers(0, len(allg)))], 0.0, 0.0) for _ in range(MAX_TOK)]
        else:
            part = sorted(B["signed"].get(p, []), key=lambda x: -x[2])
            if arm == "shuffled_sign" and part:
                # SAME partners, SAME magnitudes, permuted DIRECTIONS. The only thing destroyed is meaning.
                sg = rng.permutation([s for _t, s, _z in part])
                part = [(t, float(sg[k]), z) for k, (t, _s, z) in enumerate(part)]
        for k, (q, s, zz) in enumerate(part[:MAX_TOK - 1]):
            if q not in B["feat"]:
                continue
            X[i, k + 1, :6] = B["feat"][q]
            X[i, k + 1, 7] = s
            X[i, k + 1, 8] = np.log1p(zz)
            M[i, k + 1] = 1.0
    return X, M


def run_arm(B, arm, seed, report):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, M = tokens(B, arm, rng)
    Z = B["Z"]
    Zn = (Z - Z.mean(1, keepdims=True)) / np.maximum(Z.std(1, keepdims=True), 1e-9)
    n = len(X)
    fold = np.random.default_rng(7 + seed).permutation(n) % FOLDS
    Xt, Mt = torch.tensor(X), torch.tensor(M)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(X.shape[2], DIM)
            if arm == "mlp":
                self.body = nn.Sequential(nn.ReLU(), nn.Linear(DIM, DIM), nn.ReLU(), nn.Linear(DIM, DIM))
            else:
                lay = nn.TransformerEncoderLayer(DIM, 4, DIM * 2, batch_first=True, dropout=0.1)
                self.body = nn.TransformerEncoder(lay, 2)
            self.out = nn.Linear(DIM, DIM)

        def forward(self, x, m):
            h = self.inp(x)
            if arm == "mlp":
                h = self.body(h[:, 0])           # own features only, no context, no attention
            else:
                h = self.body(h, src_key_padding_mask=(m < 0.5))
                h = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            return torch.nn.functional.normalize(self.out(h), dim=-1)

    scores = []
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        if len(te) < 20:
            continue
        mdl = Enc()
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        Ztr = torch.tensor(Zn[tr])
        for _ep in range(EPOCHS):
            b = torch.tensor(rng.choice(len(tr), min(256, len(tr)), replace=False))
            emb = mdl(Xt[tr][b], Mt[tr][b])
            tgt = (Ztr[b] @ Ztr[b].T) / Zn.shape[1]
            pred = emb @ emb.T
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            e = mdl(Xt[te], Mt[te]).numpy()
        true = (Zn[te] @ Zn[te].T) / Zn.shape[1]
        pred = e @ e.T
        iu = np.triu_indices(len(te), 1)
        from scipy import stats
        scores.append(float(stats.spearmanr(pred[iu], true[iu])[0]))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("DOES A TRANSFORMER USE THE SIGN? attention over knockout context, RPE1 held out")
    report("=" * 100)
    for p in (PB, NET, REGL):
        if not p.exists():
            raise SystemExit(f"absent: {p}")
    B = build(report)

    ARMS = [("mlp", "MLP, own features (the bar)"),
            ("self", "transformer, self-only"),
            ("unsigned", "transformer + UNSIGNED context"),
            ("signed", "transformer + SIGNED context"),
            ("shuffled_sign", "transformer + shuffled signs (control)"),
            ("random", "transformer + random partners (control)")]
    R = {"task": "predict RPE1 perturbation profile similarity; edges are K562-derived so this is "
                 "out-of-sample", "max_tokens": MAX_TOK, "dim": DIM, "epochs": EPOCHS,
         "folds": FOLDS, "seeds": list(SEEDS), "arms": {}}
    report(f"\n  held-out Spearman(predicted, true) profile similarity, {FOLDS} folds x {len(SEEDS)} seeds")
    report(f"  {'arm':38s} {'score':>9s} {'sd':>8s}")
    for key, label in ARMS:
        vals = [run_arm(B, key, s, report)[0] for s in SEEDS]
        m, sd = float(np.mean(vals)), float(np.std(vals))
        report(f"  {label:38s} {m:9.4f} {sd:8.4f}")
        R["arms"][key] = {"label": label, "score": m, "sd": sd, "per_seed": vals}

    a = R["arms"]
    ctx = a["unsigned"]["score"] - a["mlp"]["score"]
    sgn = a["signed"]["score"] - a["unsigned"]["score"]
    shuf = a["signed"]["score"] - a["shuffled_sign"]["score"]
    rnd = a["signed"]["score"] - a["random"]["score"]
    arch = a["self"]["score"] - a["mlp"]["score"]
    R["deltas"] = {"architecture_alone": arch, "context_over_mlp": ctx, "sign_over_unsigned": sgn,
                   "sign_over_shuffled": shuf, "over_random_partners": rnd}
    report(f"\n  {'architecture alone (self - MLP)':42s} {arch:+.4f}")
    report(f"  {'context (unsigned - MLP)':42s} {ctx:+.4f}")
    report(f"  {'SIGN (signed - unsigned)':42s} {sgn:+.4f}")
    report(f"  {'SIGN vs shuffled sign (the real control)':42s} {shuf:+.4f}")
    report(f"  {'vs random partners':42s} {rnd:+.4f}")

    report("\n" + "=" * 100)
    tol = max(a["signed"]["sd"], a["shuffled_sign"]["sd"])
    if shuf > tol and sgn > 0:
        v = (f"THE SIGN IS USED. Attention over signed context beats the same context with permuted signs by "
             f"{shuf:+.4f}, above the {tol:.4f} seed spread, and beats unsigned context by {sgn:+.4f}. "
             f"Since the shuffled arm has identical partners, token count and magnitudes, the only thing "
             f"separating them is direction. Context itself is worth {ctx:+.4f} over the MLP bar while the "
             f"architecture alone is worth {arch:+.4f} -- consistent with the earlier finding that a "
             f"transformer here is a device for reading interaction context, not a better approximator.")
    elif ctx > tol:
        v = (f"CONTEXT YES, SIGN NO. Attention over interaction context beats the MLP bar by {ctx:+.4f}, "
             f"reproducing the earlier result on a cell line the edges were not built from. But signed "
             f"context beats PERMUTED-sign context by only {shuf:+.4f} against a seed spread of {tol:.4f}, "
             f"so the extra number is carrying capacity rather than direction. The architecture alone is "
             f"worth {arch:+.4f}. A signed layer helps a model that can use signs; nothing here shows this "
             f"one does.")
    else:
        v = (f"NEITHER CONTEXT NOR SIGN SEPARATES on this task. Context is worth {ctx:+.4f} over the MLP "
             f"bar and sign {shuf:+.4f} over shuffled signs, both inside the {tol:.4f} seed spread. Note "
             f"this is a HARDER setting than the earlier K562 result: the target is a different cell line "
             f"from the one the edges were built on, and RPE1 profile similarity carries a measured "
             f"reliability ceiling of about 0.48.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "transformer_signed.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'transformer_signed.json'}")


if __name__ == "__main__":
    main()
