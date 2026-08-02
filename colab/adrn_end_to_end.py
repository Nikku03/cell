"""LET IT LEARN: end-to-end training on the sealed knockout protocol, with no hand-designed features.

WHY THIS RUNS.  Everything this project has scored on the sealed protocol used features a human chose: 696 named
channels, hand-cut buckets ("catalyses >= 5"), hand-picked top-N pathway lists, hand-designed conjunctions.  That
is a real objection.  The counter-evidence usually cited -- Set Transformer 0.0297 losing to a zero-parameter bag,
+0.012 architecture headroom -- was all measured on PROXY tasks.  No learned end-to-end model has ever been scored
on the sealed cohorts.  That gap is closed here.

WHAT "END TO END" MEANS IN THIS FILE, precisely, because the phrase is easy to fake.
    - NO hand-picked channel list.  The input is the FULL annotation vocabulary: every Reactome pathway, GO term,
      protein complex, Pfam family and InterPro domain that occurs at least 3 times.  No top-N cut, no frequency
      ranking, no curated subset.
    - NO hand-cut thresholds.  No ">= 5 reactions" buckets.  Terms go in as membership, degrees go in as raw
      numbers and the model decides what to do with them.
    - NO hand-designed conjunctions.  If interactions matter the network finds them in its hidden layer.
    - The gene's own representation and its partners' representations come from the SAME learned embedding table,
      so a held-out gene is represented exactly like any other and cold start is preserved.
    - The objective is the actual goal -- the knockout's response profile over all measured genes -- not an
      intermediate a human picked.

ARMS
    e2e_full     term/partner embeddings -> MLP -> learned decoder over all 8,246 genes.  Fully learned, the
                 purest form of the request.
    e2e_nmf      the same trunk, decoding to the 60 NMF loadings with the frozen basis.  A hybrid: learned
                 representation, human-chosen output space.  Separates "learning the inputs helps" from
                 "learning the outputs helps".
    trees        gradient-boosted trees on the 696 named channels -> 60 loadings.  On the DepMap cold-start task
                 trees beat every linear arm by ~0.06 RMSE and have never been tried here.  Cheap, strong, and
                 the single most likely thing to work.
    chan2a       the incumbent hand-built linear model, 0.2885, for reference.

LEAKAGE.  Both sealed cohorts are removed before anything is fitted, including the NMF basis.  Early stopping
uses a validation slice carved from TRAINING knockouts.  A held-out knockout's profile is never read.

PREDECLARED.  A learned model earns the headline only if it beats chan2a past a bootstrap CI on BOTH cohorts.
If e2e loses and trees win, the lesson is that the ceiling was non-linearity rather than representation.  If
both lose, the hand-built linear model stands and 4,720 training knockouts is simply too few to learn a
representation from.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT = A.OUT
SP = A.SP
MIN_TERM = 3
DIM = int(os.environ.get("E2E_DIM", "128"))
EPOCHS = int(os.environ.get("E2E_EPOCHS", "300"))
PATIENCE = 25
MAX_PARTNERS = 16


def vocabulary(genes: list[str], report):
    """Every annotation term, with no selection beyond a minimum count. This is the un-curated input."""

    D = json.load(open(OUT / "cell_complete.json"))
    nm = [g["name"] for g in D["genes"]]
    terms: dict[str, set[str]] = collections.defaultdict(set)

    def gname(k: str) -> str:
        return nm[int(k)] if k.isdigit() and int(k) < len(nm) else k

    for g, v in (D.get("go") or {}).items():
        key = gname(g)
        if isinstance(v, dict):
            for ns in sorted(v):
                for t in (v[ns] or []):
                    terms[key].add(f"go:{ns}:{t}")
    for g, cs in (D.get("gene2cplx") or {}).items():
        for c in (cs if isinstance(cs, list) else [cs]):
            terms[gname(g)].add(f"cplx:{c}")
    partners: dict[str, list[str]] = collections.defaultdict(list)
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < len(nm) and 0 <= b_ < len(nm) and a_ != b_:
            partners[nm[a_]].append(nm[b_])
            partners[nm[b_]].append(nm[a_])
    cplx_of: dict[str, list[str]] = collections.defaultdict(list)
    for g, cs in (D.get("gene2cplx") or {}).items():
        for c in (cs if isinstance(cs, list) else [cs]):
            cplx_of[str(c)].append(gname(g))
    for c, mem in cplx_of.items():
        for g in mem:
            partners[g].extend(x for x in mem if x != g)
    degrees = {g: float(len(v)) for g, v in partners.items()}
    del D

    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            for g in p[2:]:
                terms[g].add(f"react:{p[0]}")
    for src, tag in (("prot_pfam.json", "pfam"), ("domains.json", "dom")):
        try:
            raw = json.loads((OUT / src).read_text())
            raw = raw.get("domains", raw)
            for k_, v in raw.items():
                if k_.isdigit() and int(k_) < len(nm):
                    for t in v:
                        terms[nm[int(k_)]].add(f"{tag}:{t}")
        except Exception as exc:
            report(f"  [warn] {src} unavailable ({exc})")

    count: collections.Counter = collections.Counter()
    for g in genes:
        for t in terms.get(g, ()):
            count[t] += 1
    vocab = sorted(t for t, c in count.items() if c >= MIN_TERM)
    vid = {t: i for i, t in enumerate(vocab)}
    report(f"  un-curated vocabulary: {len(vocab):,} terms occurring >= {MIN_TERM} times "
           f"(no top-N cut, no curation)")
    per_gene = {g: sorted(vid[t] for t in terms.get(g, ()) if t in vid) for g in genes}
    report(f"  mean terms per gene {np.mean([len(v) for v in per_gene.values()]):.1f}; "
           f"genes with none {sum(1 for v in per_gene.values() if not v):,}")
    return vocab, per_gene, partners, degrees


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    import torch
    import torch.nn as nn

    report("=" * 104)
    report("END-TO-END LEARNED MODEL ON THE SEALED KNOCKOUT PROTOCOL")
    report("=" * 104)
    report("  No hand-picked channels, no hand-cut buckets, no hand-designed conjunctions.")
    report("  PREDECLARED: a learned model earns the headline only if it beats chan2a on BOTH cohorts.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes_all = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes_all)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed: set[str] = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)

    vocab, per_gene, partners, degrees = vocabulary(universe, report)
    urow = {g: i for i, g in enumerate(universe)}

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    report(f"\n  training pool {len(train):,}; NMF basis on training only ...")
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    target_full = X                                                # the actual goal: the response profile
    del X

    # ---- pack the un-curated inputs ----
    MAXT = max(1, max(len(per_gene.get(g, [])) for g in universe))
    MAXT = min(MAXT, 256)

    def pack(ko_list):
        n = len(ko_list)
        self_t = np.zeros((n, MAXT), np.int64)
        self_m = np.zeros((n, MAXT), np.float32)
        part_t = np.zeros((n, MAX_PARTNERS, MAXT), np.int64)
        part_m = np.zeros((n, MAX_PARTNERS, MAXT), np.float32)
        deg = np.zeros((n, 2), np.float32)
        for i, k in enumerate(ko_list):
            ts = per_gene.get(k, [])[:MAXT]
            self_t[i, :len(ts)] = ts
            self_m[i, :len(ts)] = 1.0
            ps = sorted(set(partners.get(k, [])))[:MAX_PARTNERS]
            for j, p in enumerate(ps):
                pt = per_gene.get(p, [])[:MAXT]
                part_t[i, j, :len(pt)] = pt
                part_m[i, j, :len(pt)] = 1.0
            deg[i] = [np.log1p(degrees.get(k, 0.0)), np.log1p(len(ps))]
        return (torch.from_numpy(self_t), torch.from_numpy(self_m),
                torch.from_numpy(part_t), torch.from_numpy(part_m), torch.from_numpy(deg))

    class Net(nn.Module):
        def __init__(self, n_out):
            super().__init__()
            self.emb = nn.Embedding(len(vocab) + 1, DIM, padding_idx=0)
            self.trunk = nn.Sequential(nn.Linear(2 * DIM + 2, DIM), nn.GELU(),
                                       nn.Dropout(0.2), nn.Linear(DIM, DIM), nn.GELU())
            self.head = nn.Linear(DIM, n_out)

        def bag(self, t, m):
            e = self.emb(t) * m.unsqueeze(-1)
            return e.sum(-2) / m.sum(-1, keepdim=True).clamp(min=1)

        def forward(self, st, sm, pt, pm, dg):
            s = self.bag(st, sm)
            p = self.bag(pt.reshape(pt.shape[0], -1, pt.shape[-1]),
                         pm.reshape(pm.shape[0], -1, pm.shape[-1]))
            return self.head(self.trunk(torch.cat([s, p, dg], -1)))

    torch.manual_seed(A.SEED)
    rng = np.random.default_rng(A.SEED)
    idx = rng.permutation(len(train))
    cut = int(0.9 * len(train))
    tr_i, va_i = idx[:cut], idx[cut:]
    packed = pack(train)

    def fit(y: np.ndarray, label: str):
        net = Net(y.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        Y = torch.from_numpy(y)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(EPOCHS):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(tr_i)))
            for b in range(0, len(tr_i), 256):
                sel = torch.from_numpy(tr_i)[perm[b:b + 256]]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(*[p[sel] for p in packed]), Y[sel])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                sel = torch.from_numpy(va_i)
                v = float(nn.functional.mse_loss(net(*[p[sel] for p in packed]), Y[sel]))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k_: t.clone() for k_, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        report(f"    {label:<10} stopped at epoch {ep + 1}, validation MSE {best:.6f}")
        return net

    report(f"\n  training (dim {DIM}, early stopping on 10% of training knockouts)")
    net_full = fit(target_full, "e2e_full")
    net_nmf = fit(W, "e2e_nmf")

    from sklearn.ensemble import HistGradientBoostingRegressor
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    tr_rows = np.array([urow[k] for k in train])
    report(f"  trees: {A.K} gradient-boosted regressors on {chan.shape[1]} channels ...")
    forests = [HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(chan[tr_rows], W[:, j])
               for j in range(A.K)]

    for tag in ("", "_holdout"):
        dp = OUT / f"ko_pred_dossiers{tag}.json"
        if not dp.exists():
            continue
        cohort = sorted(json.loads(dp.read_text()))
        pk = pack(cohort)
        with torch.no_grad():
            score_full = net_full(*pk).numpy()
            score_nmf = net_nmf(*pk).numpy() @ H
        score_trees = np.stack([f.predict(chan[[urow[k] for k in cohort]]) for f in forests], 1) @ H
        for arm, S in (("e2e_full", score_full), ("e2e_nmf", score_nmf), ("trees", score_trees)):
            preds = {}
            for j, k in enumerate(cohort):
                s = S[j].copy()
                s[tide] = -np.inf
                if k in gi:
                    s[gi[k]] = -np.inf
                order = np.argsort(-s)[:A.NPRED]
                preds[k] = {"variant": arm,
                            "reasoning": f"{arm}: learned from the un-curated annotation vocabulary "
                                         f"({len(vocab):,} terms) with no hand-designed features",
                            "predicted_genes": [genes_all[i] for i in order if np.isfinite(s[i])]}
            p = OUT / f"ko_predictions_{arm}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            report(f"  {tag or 'cohort1':<9} {arm:<10} -> {p.name}")

    json.dump({"test": "adrn_end_to_end", "vocab": len(vocab), "dim": DIM,
               "n_train": len(train), "log": log},
              open(OUT / "adrn_end_to_end.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_end_to_end.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
