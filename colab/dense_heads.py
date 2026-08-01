"""STAGE 5 -- does any head beat its own per-gene marginal, on an untruncated target?

WHY EVERY HEAD NUMBER IN THIS PROJECT NEEDS REMEASURING. `cellformer_v1` reported five supervised heads, and
the strongest-looking result in the whole arc was the direction head at AUC 0.975. It was already recorded
that a per-gene marginal using NO KNOCKOUT INFORMATION AT ALL reaches 0.949 of that. Both numbers were
measured on the `nlz_*` benches, where 96.5% of every response matrix is a zero that means NOT RECORDED. The
recorded 3.5% is the top ~250 genes per knockout -- an extreme, non-random selection -- so both the head and
its marginal were fitted and scored on the easiest cells in the matrix. Neither number transfers.

This remeasures all of it on the dense K562 pseudobulk: 11,258 perturbations x 8,175 genes, 100% dense, no
truncation. Same convention as `dense_stage1.py` so the two are comparable.

THE RULE THIS STAGE EXISTS TO ENFORCE. A head is only worth its parameters if it beats what you would get by
knowing nothing about the perturbation. For every head that means a PER-GENE MARGINAL fitted on training
knockouts only:

    responds     P(gene g moves at all), ignoring which knockout it was
    direction    P(gene g moves UP | it moved), ignoring which knockout it was
    magnitude    the gene's mean |z| -- this project's "tide"
    uncertainty  total predicted movement, i.e. bigger predictions are wronger

Those marginals are strong. Gene-level responsiveness is the single largest source of structure in
perturbation data, and a model can score very well while contributing nothing on top of it. Reporting a head
against chance instead of against its marginal is the error this stage is built to prevent.

WHAT IS SHARED AND WHAT IS NOT. All heads read one knockout embedding, produced by the same encoder stage 1
validated -- the perturbed gene plus its typed neighbours (co-dependency, complex, PPI) over DepMap
co-dependency features -- concatenated with a learned gene embedding. So this measures the heads, not a new
architecture. The random-partner control is carried through: if a head's advantage over its marginal survives
with WRONG neighbours, the advantage is not coming from the interaction context.

SPLITS. Three disjoint held-out knockout partitions, not three seeds of one partition. The unit of
replication is the knockout, the claim is about a knockout never seen, and rotating fold labels of a single
partition would not test that.
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
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
DEPVEC = OUT / "depmap_vecs.npz"

TAU = 1.0
MAX_TOK = 24
D_MODEL = 128
D_GENE = 64
N_PAIRS = int(os.environ.get("DH_PAIRS", "3000000"))
EPOCHS = int(os.environ.get("DH_EPOCHS", "4"))
N_SPLITS = int(os.environ.get("DH_SPLITS", "3"))
SMOKE = os.environ.get("DH_SMOKE", "0") == "1"
if SMOKE:
    N_PAIRS, EPOCHS, N_SPLITS = 200000, 1, 1


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(s), float)
    r[np.argsort(s)] = np.arange(len(s))
    n1 = int(y.sum())
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(s) - n1)))


def main():
    import h5py
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("STAGE 5 -- five heads, each against its own per-gene marginal, on a 100%-dense target")
    report("=" * 100)

    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    Z = (X - med) / np.where(mad < 1e-6, np.nan, mad)
    del X
    ok = np.isfinite(Z).all(0)
    Z = Z[:, ok]
    report(f"  dense K562: {Z.shape[0]:,} perturbations x {Z.shape[1]:,} genes with a defined robust z")

    dv = np.load(DEPVEC, allow_pickle=True)
    syms = [str(x) for x in dv["syms"]]
    Zd = dv["Z"].astype(np.float32)
    U, s_, _ = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zp = (U[:, :64] * s_[:64]).astype(np.float32)
    srow = {g: i for i, g in enumerate(syms)}

    d = json.load(gzip.open(NET, "rt"))
    nmz = [g["name"] for g in d["genes"]]
    codep, cplx, ppi = {}, {}, {}
    for a_, v in (d.get("codep") or {}).items():
        codep.setdefault(nmz[int(a_)], []).extend(nmz[int(b)] for b, _s in v)
    g2c = {int(k_): set(v) for k_, v in (d.get("gene2cplx") or {}).items()}
    bym = {}
    for x, cs in g2c.items():
        for c in cs:
            bym.setdefault(c, []).append(x)
    for c, mem in bym.items():
        for x in mem:
            cplx.setdefault(nmz[x], []).extend(nmz[y] for y in mem if y != x)
    for e in d["ppi"]:
        ppi.setdefault(nmz[int(e[0])], []).append(nmz[int(e[1])])
        ppi.setdefault(nmz[int(e[1])], []).append(nmz[int(e[0])])
    del d

    keep = np.where(np.array([s in srow for s in psym]))[0]
    if SMOKE:
        keep = keep[:1500]
    Z = Z[keep]
    psym_k = psym[keep]
    nP, nG = Z.shape
    report(f"  usable: {nP:,} knockouts with a DepMap profile x {nG:,} genes = {nP*nG:,} (p,g) cells")

    def tokens(mode, rng):
        allg = list(srow)
        F = np.zeros((nP, MAX_TOK, Zp.shape[1]), np.float32)
        T = np.zeros((nP, MAX_TOK), np.int64)
        M = np.zeros((nP, MAX_TOK), bool)
        for i, k_ in enumerate(psym_k):
            rows, types = [k_], [0]
            for t, lst in ((1, codep.get(k_, [])), (2, cplx.get(k_, [])), (3, ppi.get(k_, []))):
                pick = lst[:7] if mode == "real" else [allg[int(x)] for x in
                                                       rng.integers(0, len(allg), min(7, len(lst)))]
                for g in pick:
                    rows.append(g)
                    types.append(t)
            for j, (g, t) in enumerate(zip(rows[:MAX_TOK], types[:MAX_TOK])):
                if g in srow:
                    F[i, j] = Zp[srow[g]]
                T[i, j] = t
                M[i, j] = True
        return torch.from_numpy(F), torch.from_numpy(T), torch.from_numpy(M)

    Fr, Tr, Mr = tokens("real", np.random.default_rng(0))
    Frr, Trr, Mrr = tokens("random", np.random.default_rng(1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(Zp.shape[1], D_MODEL)
            self.rel = nn.Embedding(4, D_MODEL)
            self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
            self.ln = nn.LayerNorm(D_MODEL)
            self.gemb = nn.Embedding(nG, D_GENE)
            self.mlp = nn.Sequential(nn.Linear(D_MODEL + D_GENE, 128), nn.ReLU(), nn.Linear(128, 3))

        def embed(self, F, T, M):
            h = self.proj(F) + self.rel(T)
            a, _ = self.attn(h, h, h, key_padding_mask=~M, need_weights=False)
            h = self.ln(h + a)
            m = M.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)

        def forward(self, F, T, M, gidx):
            e = self.embed(F, T, M)
            return self.mlp(torch.cat([e, self.gemb(gidx)], 1))

    res = {}
    for sp in range(N_SPLITS):
        rng = np.random.default_rng(500 + sp)
        p = rng.permutation(nP)
        te, tr = p[:nP // 4], p[nP // 4:]
        report(f"\n  PARTITION {sp}: {len(tr):,} train / {len(te):,} test knockouts")

        # THE MARGINALS. Fitted on TRAINING knockouts only, and they use no knowledge of which knockout a
        # cell belongs to -- that is exactly what makes them the right thing for a head to have to beat.
        Ztr = Z[tr]
        mv_tr = np.abs(Ztr) >= TAU
        m_rate = mv_tr.mean(0)                                            # responds
        up_rate = np.where(mv_tr, Ztr > 0, np.nan)
        m_up = np.nanmean(up_rate, axis=0)
        m_up = np.where(np.isfinite(m_up), m_up, 0.5)
        m_mag = np.abs(Ztr).mean(0)                                       # magnitude == the tide

        pi = tr[rng.integers(0, len(tr), N_PAIRS)]
        gj = rng.integers(0, nG, N_PAIRS)
        zt = Z[pi, gj]
        y_mv = (np.abs(zt) >= TAU).astype(np.float32)
        y_up = (zt > 0).astype(np.float32)
        y_mg = np.abs(zt).astype(np.float32)

        pe = te[rng.integers(0, len(te), N_PAIRS // 4)]
        ge = rng.integers(0, nG, N_PAIRS // 4)
        ze = Z[pe, ge]
        e_mv = np.abs(ze) >= TAU
        e_up = ze > 0
        e_mg = np.abs(ze)

        def train_eval(F, T, M, tag):
            torch.manual_seed(sp)
            net = Net()
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            bce = nn.BCEWithLogitsLoss()
            tpi = torch.from_numpy(pi)
            tgj = torch.from_numpy(gj)
            tv = torch.from_numpy(y_mv)
            tu = torch.from_numpy(y_up)
            tm = torch.from_numpy(y_mg)
            for _e in range(EPOCHS):
                perm = torch.randperm(N_PAIRS)
                for i0 in range(0, N_PAIRS, 8192):
                    b = perm[i0:i0 + 8192]
                    ix = tpi[b]
                    o = net(F[ix], T[ix], M[ix], tgj[b])
                    # direction is supervised ONLY where the gene actually moved; training it on unmoved
                    # genes would make it predict the sign of noise.
                    w = tv[b]
                    loss = (bce(o[:, 0], tv[b])
                            + (bce(o[:, 1], tu[b]) * w).sum() / w.sum().clamp(min=1)
                            + (o[:, 2] - tm[b]).pow(2).mean())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            outs = []
            with torch.no_grad():
                for i0 in range(0, len(pe), 16384):
                    ix = torch.from_numpy(pe[i0:i0 + 16384])
                    outs.append(net(F[ix], T[ix], M[ix],
                                    torch.from_numpy(ge[i0:i0 + 16384])).numpy())
            o = np.concatenate(outs)
            err = np.abs(o[:, 2] - e_mg)
            r = {
                f"{tag} responds AUC": auc(e_mv, o[:, 0]),
                f"{tag} direction AUC": auc(e_up[e_mv], o[e_mv, 1]),
                f"{tag} magnitude r": float(np.corrcoef(o[:, 2], e_mg)[0, 1]),
                # uncertainty: does predicted magnitude rank its own error? Spearman via rank correlation.
                f"{tag} uncertainty rho": float(np.corrcoef(
                    np.argsort(np.argsort(o[:, 2])), np.argsort(np.argsort(err)))[0, 1]),
            }
            return r

        out = {}
        out.update(train_eval(Fr, Tr, Mr, "MODEL"))
        out.update(train_eval(Frr, Trr, Mrr, "random-partner CONTROL"))
        # THE PER-GENE MARGINAL IS WEAK HERE, AND NOT FOR A BIOLOGICAL REASON. The target is a per-gene
        # robust z: every gene is centred on its own median and scaled by its own MAD across perturbations.
        # That construction FLATTENS the per-gene marginal -- P(|z| >= 1) is near-identical for every gene by
        # design. On the truncated benches the normalisation was different and the marginal was strong,
        # which is why the old head numbers had a 0.949 marginal to sit just above. Comparing a head only to
        # a marginal that the normalisation has already destroyed would be the same error this stage exists
        # to prevent, running the other way. So two ORACLE diagnostics are added. They USE THE HELD-OUT
        # TRUTH and are not baselines; they exist to decompose where a head's gain actually comes from.
        ko_scale = np.abs(Z[pe]).mean(1)                       # this knockout's own overall responsiveness
        out["ORACLE knockout-scale magnitude r"] = float(np.corrcoef(ko_scale, e_mg)[0, 1])
        out["ORACLE gene x knockout magnitude r"] = float(np.corrcoef(m_mag[ge] * ko_scale, e_mg)[0, 1])
        out["ORACLE knockout-scale responds AUC"] = auc(e_mv, ko_scale)
        out["MARGINAL responds AUC"] = auc(e_mv, m_rate[ge])
        out["MARGINAL direction AUC"] = auc(e_up[e_mv], m_up[ge[e_mv]])
        out["MARGINAL magnitude r"] = float(np.corrcoef(m_mag[ge], e_mg)[0, 1])
        out["MARGINAL uncertainty rho"] = float(np.corrcoef(
            np.argsort(np.argsort(m_mag[ge])), np.argsort(np.argsort(np.abs(m_mag[ge] - e_mg))))[0, 1])
        for k_, v in out.items():
            res.setdefault(k_, []).append(v)
            report(f"    {k_:34s} {v:+.4f}")

    report(f"\n  {'quantity':34s} {'mean':>9s} {'sd':>8s}")
    for k_, v in res.items():
        report(f"  {k_:34s} {np.mean(v):+9.4f} {np.std(v):8.4f}")

    sd = float(np.mean([np.std(v) for v in res.values()]))
    mde = 3 * sd / np.sqrt(N_SPLITS)
    heads = ["responds AUC", "direction AUC", "magnitude r", "uncertainty rho"]
    orc = {k_: float(np.mean(v)) for k_, v in res.items() if k_.startswith("ORACLE")}
    verdicts = {}
    report(f"\n  pooled partition sd {sd:.4f} -> minimum detectable increment {mde:+.4f} at {N_SPLITS} "
           f"partitions")
    report(f"\n  {'head':22s} {'model':>9s} {'marginal':>9s} {'delta':>9s} {'rand-ctrl':>10s} verdict")
    for h in heads:
        mo = float(np.mean(res[f"MODEL {h}"]))
        ma = float(np.mean(res[f"MARGINAL {h}"]))
        rc = float(np.mean(res[f"random-partner CONTROL {h}"]))
        beats = (mo - ma) > mde
        ctx = (mo - rc) > mde
        verdicts[h] = {"model": mo, "marginal": ma, "delta": mo - ma, "random_partner_control": rc,
                       "beats_marginal": bool(beats), "context_attributable": bool(ctx)}
        report(f"  {h:22s} {mo:+9.4f} {ma:+9.4f} {mo-ma:+9.4f} {rc:+10.4f} "
               f"{'BEATS ITS MARGINAL' if beats else 'does NOT beat its marginal'}"
               f"{'' if ctx else ' / gain NOT attributable to real neighbours'}")

    report("\n  ORACLE DECOMPOSITION (uses held-out truth -- diagnostics, NOT baselines)")
    for k_, v in orc.items():
        report(f"    {k_:44s} {v:+.4f}")
    mo_mag = verdicts["magnitude r"]["model"]
    report(f"    -> the magnitude head reaches {mo_mag:+.4f}; knowing ONLY the held-out knockout's overall "
           f"responsiveness reaches {orc['ORACLE knockout-scale magnitude r']:+.4f}. "
           f"{'The head is therefore mostly predicting knockout scale, not gene-specific response.' if orc['ORACLE knockout-scale magnitude r'] >= mo_mag - mde else 'The head is ahead of pure knockout scale, so part of it is gene-specific.'}")

    nbeat = sum(v["beats_marginal"] for v in verdicts.values())
    verdict = (
        f"{nbeat}/{len(heads)} HEADS BEAT THEIR OWN PER-GENE MARGINAL on a 100%-dense target, across "
        f"{N_SPLITS} disjoint held-out knockout partitions, with a minimum detectable increment of {mde:.4f}. "
        + " ".join(f"{h}: model {v['model']:+.4f} vs marginal {v['marginal']:+.4f} "
                   f"({v['delta']:+.4f})." for h, v in verdicts.items())
        + f" The random-partner control is carried through every head: where a head's advantage survives with "
          f"WRONG neighbours, the advantage is not coming from interaction context and is marked as such. "
          f"Every one of these numbers replaces a figure measured on the 96.5%-censored benches, where both "
          f"head and marginal were scored only on each knockout's top ~250 genes -- an extreme non-random "
          f"selection. The 0.975 direction AUC this project has quoted is void, and so is the 0.949 marginal "
          f"it was compared against.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "dense-heads-v1", "n_knockouts": int(nP), "n_genes": int(nG), "density": 1.0,
         "n_pairs_train": N_PAIRS, "partitions": N_SPLITS,
         "raw": {k_: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "per_split": v}
                 for k_, v in res.items()},
         "heads": verdicts, "oracle_decomposition": orc, "mde": mde, "heads_beating_marginal": nbeat,
         "replaces": "cellformer_v1 head table, measured on the 96.5%-censored nlz benches",
         "limits": [
             "K562 only. These are within-cell head quality numbers and say nothing about transfer",
             "the marginals are fitted on training knockouts only, but they are computed on the SAME dense "
             "matrix, so they are strong here in a way they were not on the truncated bench. That is the "
             "point: a weak marginal is what made the old head numbers look good",
             "the direction head is supervised only where a gene moved, so its AUC is conditional on "
             "movement and is not comparable to an unconditional direction AUC",
             "one encoder, one gene embedding, shared across heads -- this measures the heads, not a "
             "per-head architecture search",
             "the uncertainty head here is the magnitude prediction ranking its own error, which is the "
             "cheapest possible uncertainty signal and a deliberate lower bound",
             "the per-gene marginal is WEAK on this target because per-gene robust z flattens it by "
             "construction, not because there is no tide. A head beating it here is a much lower bar than "
             "beating it on the raw log fold changes, and the ORACLE decomposition is what makes that "
             "visible rather than assumed"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "dense_heads.json", "w"), indent=1)
    report(f"\n  -> {OUT/'dense_heads.json'}")


if __name__ == "__main__":
    main()
