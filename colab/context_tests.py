"""CONTEXT TESTS -- can cell state fix the transfer failure, and does the model use context at all?

WHY. `cellformer_v1` reached recall@50 0.4301 on held-out K562 knockouts and then LOST to the tide on every
well-powered held-out cell line: RPE1 -0.049, HepG2 -0.027, Jurkat -0.035. It beats a random ranking tenfold,
so it is not predicting noise -- it is confidently predicting K562's answer in a cell that is not K562. The
model has no cell-context input of any kind, so that is exactly what it should do.

Rather than jump to a multimodal architecture, this runs a graded sequence of cheap tests. Each adds ONE
thing, so a gain is attributable.

  A  NO CONTEXT               the current model, cell-blind. The thing to beat.
  B  CELL-LINE TOKEN          a learned embedding per line. DESCRIPTIVE ONLY -- an unseen line has no
                              embedding, so this cannot answer the transfer question by construction.
  C  GLOBAL BASELINE VECTOR   one encoded cell-state vector conditioning every prediction.
  D  GENE-SPECIFIC CONTEXT    each token carries that cell's baseline expression of that gene.
  E  CONTEXT-GATED NEIGHBOURS the same real graph, but a neighbour not expressed in this cell is
                              down-weighted. The most biologically motivated of the five.
  F  COUNTERFACTUAL SWAP      feed D and E the WRONG cell's context. THE DECISIVE ONE.

WHY F DECIDES IT. A model can carry a context channel and ignore it. Accuracy cannot detect that: an
ignored channel costs nothing and shows up as a wash, which reads identically to "context does not help".
The swap separates them. A model that USES context must degrade when handed another cell's; a model that
ignores it is invariant. Invariance under swap is a positive finding about the model, not a null.

THE POWER SITUATION, MEASURED BEFORE THE RUN AND IT IS NOT SYMMETRIC.
The rectangle here is the TRANSPOSE of what this task wants: 4 usable cell lines x ~1,398 shared
perturbations, where the useful shape is many cells x fewer perturbations. Consequences, stated in advance:

  * Tests B and C are UNIDENTIFIABLE, not merely underpowered. Under leave-one-cell-out they fit a cell
    encoder from THREE training points. Any number they produce is a statement about three cells and is
    reported as descriptive.
  * Tests D, E and F keep full power. They act at the GENE level inside each cell, so effective n is genes
    x perturbations (~6,550 x 1,398), not cells.

LEAKAGE RULES, because this is the leave-one-cell-line-out task and it is easy to cheat at.
Allowed for the held-out cell: baseline RNA (from colab/baseline_state.py -- control/all-cell expression,
never response-derived), the cell-line-agnostic biological graph, and metadata. FORBIDDEN: any perturbation
response from that cell, its tide vector, its mover frequency, its per-gene mean |z|, or anything fitted on
its labels. The tide baseline it is SCORED against is computed from the held-out cell and is used only as a
scoring reference, never as a model input.

ACCESSIBILITY IS ABSENT for these four lines. Tests D and E therefore run on RNA only, and the ATAC channel
is declared missing rather than substituted with a proxy.
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
BASE = OUT / "baseline_state.json.gz"
NET = SP / "cell_complete.json.gz"
DEPVEC = OUT / "depmap_vecs.npz"
LINES = ["K562", "RPE1", "HepG2", "Jurkat"]
K = 50
MAX_TOK = 16
D_MODEL = 96
EPOCHS = int(os.environ.get("CTX_EPOCHS", "12"))
SMOKE = os.environ.get("CTX_SMOKE", "0") == "1"


def load_baseline(report):
    if not BASE.exists():
        raise SystemExit(f"absent: {BASE} -- run colab/baseline_state.py first")
    d = json.load(gzip.open(BASE, "rt"))
    prof = d["profiles"]
    report(f"  baseline state: {len(prof)} lines, derivations {d.get('derivation_per_line')}")
    return prof


def main():
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("CONTEXT TESTS -- leave-one-CELL-LINE-out, and a counterfactual swap that can falsify the rest")
    report("=" * 100)

    from eval_harness import Harness
    prof = load_baseline(report)

    H = {}
    for ln in LINES:
        try:
            H[ln] = Harness(ln)
        except Exception as e:
            report(f"  {ln}: bench unavailable ({type(e).__name__}) -- excluded, not substituted")
    lines = [ln for ln in LINES if ln in H and ln in prof]
    if len(lines) < 3:
        raise SystemExit(f"only {len(lines)} lines have both a bench and a baseline; leave-one-out needs "
                         f"at least 3 and this is a blocked input, not a result")

    genes = sorted(set.intersection(*[set(H[ln].genes) for ln in lines]) & set.intersection(
        *[set(prof[ln]) for ln in lines]))
    kos = sorted(set.intersection(*[set(H[ln].kos) for ln in lines]))
    report(f"  rectangle: {len(lines)} cell lines x {len(kos):,} shared perturbations x {len(genes):,} "
           f"shared genes")
    if SMOKE:
        kos, genes = kos[:120], genes[:1200]
        report(f"  SMOKE: {len(kos)} perturbations x {len(genes)} genes")
    gi = {g: i for i, g in enumerate(genes)}

    # ---- per-line response matrices on the shared grid, and the per-line tide (SCORING ONLY) ----
    A, TIDE, NONTIDE = {}, {}, {}
    for ln in lines:
        h = H[ln]
        col = np.array([h.gi[g] for g in genes])
        row = np.array([h.ki[k] for k in kos])
        A[ln] = h.A[np.ix_(row, col)].astype(np.float32)
        TIDE[ln] = h.mover_freq[col].astype(np.float32)
        NONTIDE[ln] = h.nontide[col]
        report(f"    {ln:8s} {A[ln].shape[0]:,} x {A[ln].shape[1]:,}; "
               f"{int(NONTIDE[ln].sum()):,} non-tide genes")

    B = np.stack([np.array([prof[ln].get(g, 0.0) for g in genes], dtype=np.float32) for ln in lines])
    B = (B - B.mean()) / (B.std() + 1e-6)
    lidx = {ln: i for i, ln in enumerate(lines)}

    # ---- tokens: perturbed gene + typed neighbours (cell-line-agnostic graph) ----
    d = json.load(gzip.open(NET, "rt"))
    nm = [g["name"] for g in d["genes"]]
    codep, cplx, ppi = {}, {}, {}
    for a_, v in (d.get("codep") or {}).items():
        codep.setdefault(nm[int(a_)], []).extend(nm[int(b)] for b, _s in v)
    g2c = {int(k_): set(v) for k_, v in (d.get("gene2cplx") or {}).items()}
    bym = {}
    for x, cs in g2c.items():
        for c in cs:
            bym.setdefault(c, []).append(x)
    for c, mem in bym.items():
        for x in mem:
            cplx.setdefault(nm[x], []).extend(nm[y] for y in mem if y != x)
    for e in d["ppi"]:
        ppi.setdefault(nm[int(e[0])], []).append(nm[int(e[1])])
        ppi.setdefault(nm[int(e[1])], []).append(nm[int(e[0])])
    del d

    dv = np.load(DEPVEC, allow_pickle=True)
    syms = [str(x) for x in dv["syms"]]
    Zd = dv["Z"].astype(np.float32)
    U, s_, _Vt = np.linalg.svd(Zd - Zd.mean(0), full_matrices=False)
    Zp = (U[:, :48] * s_[:48]).astype(np.float32)
    srow = {g: i for i, g in enumerate(syms)}
    report(f"  DepMap profile -> {Zp.shape[1]} components/token; {len(srow):,} genes")

    tok_g, tok_t = [], []
    for k_ in kos:
        rows, types = [k_], [0]
        for t, lst in ((1, codep.get(k_, [])), (2, cplx.get(k_, [])), (3, ppi.get(k_, []))):
            for g in lst[:5]:
                rows.append(g)
                types.append(t)
        tok_g.append(rows[:MAX_TOK])
        tok_t.append(types[:MAX_TOK])
    L = max(len(t) for t in tok_g)
    NT = len(kos)
    TOKZ = np.zeros((NT, L, Zp.shape[1]), np.float32)
    TOKT = np.zeros((NT, L), np.int64)
    TOKM = np.zeros((NT, L), bool)
    TOKGI = np.full((NT, L), -1, np.int64)          # index into `genes`, for the cell-specific channel
    for i, (gs, ts) in enumerate(zip(tok_g, tok_t)):
        for j, (g, t) in enumerate(zip(gs, ts)):
            if g in srow:
                TOKZ[i, j] = Zp[srow[g]]
            TOKT[i, j] = t
            TOKM[i, j] = True
            TOKGI[i, j] = gi.get(g, -1)
    report(f"  tokens: {NT:,} perturbations x up to {L} (mean "
           f"{np.mean([len(t) for t in tok_g]):.1f}); {100*np.mean(TOKGI >= 0):.0f}% resolve a baseline")

    tZ = torch.from_numpy(TOKZ)
    tT = torch.from_numpy(TOKT)
    tM = torch.from_numpy(TOKM)
    tGI = torch.from_numpy(TOKGI)
    tB = torch.from_numpy(B)

    class Net(nn.Module):
        """One encoder, four context modes. The mode is a constructor flag so an arm cannot silently
        acquire a channel it is not supposed to have."""

        def __init__(self, mode, n_lines, n_genes):
            super().__init__()
            self.mode = mode
            din = Zp.shape[1] + (1 if mode in ("gene", "gate") else 0)
            self.proj = nn.Linear(din, D_MODEL)
            self.rel = nn.Embedding(4, D_MODEL)
            self.line_emb = nn.Embedding(n_lines + 1, D_MODEL)      # +1 = the unseen-line slot
            self.ctx = nn.Linear(n_genes, D_MODEL)
            self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
            self.ln = nn.LayerNorm(D_MODEL)
            self.dec = nn.Linear(D_MODEL, n_genes)

        def forward(self, zi, li, bvec):
            z, t, m, g = tZ[zi], tT[zi], tM[zi], tGI[zi]
            if self.mode in ("gene", "gate"):
                bg = torch.where(g >= 0, bvec.gather(1, g.clamp(min=0)), torch.zeros_like(z[..., 0]))
                z = torch.cat([z, bg.unsqueeze(-1)], -1)
            h = self.proj(z) + self.rel(t)
            if self.mode == "line":
                h = h + self.line_emb(li).unsqueeze(1)
            if self.mode == "global":
                h = h + self.ctx(bvec).unsqueeze(1)
            mask = m.clone()
            if self.mode == "gate":
                # CONTEXT GATING: a neighbour not expressed in THIS cell is masked out. The graph is the
                # same real graph; only which of its edges are live changes with the cell.
                expressed = torch.where(g >= 0, bvec.gather(1, g.clamp(min=0)), torch.zeros_like(bg))
                mask = mask & ((expressed > -0.5) | (t == 0))
            a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
            h = self.ln(h + a)
            mm = mask.unsqueeze(-1).float()
            p = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            return self.dec(p)

    def recall(pred, truth_mag, nontide):
        out = []
        for i in range(pred.shape[0]):
            t = np.argsort(-np.where(nontide, truth_mag[i], -np.inf))[:K]
            t = [x for x in t if nontide[x] and truth_mag[i][x] > 0]
            if len(t) < 5:
                continue
            top = set(np.argsort(-pred[i])[:K].tolist())
            out.append(len(set(t) & top) / len(t))
        return float(np.mean(out)) if out else float("nan"), len(out)

    def run_fold(held, mode, seed=0):
        torch.manual_seed(seed)
        tr = [ln for ln in lines if ln != held]
        net = Net(mode, len(lines), len(genes))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        Y = {ln: torch.from_numpy(A[ln]) for ln in tr}
        idx = np.arange(NT)
        for _ep in range(EPOCHS):
            np.random.default_rng(_ep + seed).shuffle(idx)
            for ln in tr:
                li = torch.full((0,), 0)
                for i0 in range(0, NT, 128):
                    b = torch.from_numpy(idx[i0:i0 + 128])
                    if len(b) < 8:
                        continue
                    li = torch.full((len(b),), lidx[ln], dtype=torch.long)
                    out = net(b, li, tB[lidx[ln]].expand(len(b), -1))
                    loss = (out - Y[ln][b]).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        return net

    def predict(net, ctx_line, held):
        with torch.no_grad():
            li = torch.full((NT,), lidx.get(ctx_line, 0), dtype=torch.long)
            if ctx_line is None:                      # unseen-line slot for the cell-token arm
                li = torch.full((NT,), len(lines), dtype=torch.long)
            bv = tB[lidx[ctx_line]] if ctx_line else torch.zeros(len(genes))
            return net(torch.arange(NT), li, bv.expand(NT, -1)).numpy()

    MODES = [("A no context", "none"), ("B cell-line token", "line"), ("C global baseline", "global"),
             ("D gene-specific context", "gene"), ("E context-gated neighbours", "gate")]
    R = {"model": "context-tests-v1", "lines": lines, "n_perturbations": len(kos), "n_genes": len(genes),
         "rectangle_note": "4 cell lines x ~1,398 shared perturbations -- the TRANSPOSE of what this task "
                           "wants; tests B and C fit a cell encoder from 3 points and are DESCRIPTIVE",
         "accessibility": "ABSENT for these lines; tests D and E run on RNA only, not substituted",
         "folds": {}, "swap": {}}

    for held in lines:
        report(f"\n  ===== HELD-OUT CELL LINE: {held} "
               f"(train on {', '.join([l for l in lines if l != held])}) =====")
        tmag, tnon = A[held], NONTIDE[held]
        r_tide, n_ok = recall(np.tile(TIDE[held], (NT, 1)), tmag, tnon)
        rng = np.random.default_rng(0)
        r_rand, _ = recall(rng.standard_normal((NT, len(genes))).astype(np.float32), tmag, tnon)
        report(f"    {'reference: that cell line own tide':38s} {r_tide:.4f}  ({n_ok} scorable)")
        report(f"    {'reference: random':38s} {r_rand:.4f}")
        fold = {"tide": r_tide, "random": r_rand, "n_scorable": n_ok, "arms": {}}
        nets = {}
        for label, mode in MODES:
            net = run_fold(held, mode)
            nets[mode] = net
            p = predict(net, held if mode != "line" else None, held)
            r_, _ = recall(p, tmag, tnon)
            fold["arms"][label] = {"recall": r_, "delta_vs_tide": r_ - r_tide}
            report(f"    {label:38s} {r_:.4f}   vs tide {r_-r_tide:+.4f}")
        # ---- F: counterfactual swap, on the two arms that HAVE a context to swap ----
        sw = {}
        for mode in ("gene", "gate"):
            base = fold["arms"]["D gene-specific context" if mode == "gene"
                                else "E context-gated neighbours"]["recall"]
            row = {"correct": base}
            for other in [l for l in lines if l != held]:
                p = predict(nets[mode], other, held)
                row[f"swapped:{other}"] = recall(p, tmag, tnon)[0]
            perm = np.random.default_rng(3).permutation(len(genes))
            with torch.no_grad():
                li = torch.full((NT,), lidx[held], dtype=torch.long)
                p = nets[mode](torch.arange(NT), li,
                               tB[lidx[held]][perm].expand(NT, -1)).numpy()
            row["shuffled"] = recall(p, tmag, tnon)[0]
            drop = base - np.mean([v for k_, v in row.items() if k_.startswith("swapped")])
            row["drop_when_swapped"] = float(drop)
            row["uses_context"] = bool(drop > 0.005)
            sw[mode] = row
            report(f"    F swap [{mode}]  correct {base:.4f} | " +
                   " | ".join(f"{k_.split(':')[1]} {v:.4f}" for k_, v in row.items()
                              if k_.startswith("swapped")) +
                   f" | shuffled {row['shuffled']:.4f} | drop {drop:+.4f} -> "
                   f"{'USES context' if row['uses_context'] else 'IGNORES context'}")
        fold["swap"] = sw
        R["folds"][held] = fold

    # ---- verdict ----
    def agg(label, key="delta_vs_tide"):
        return float(np.mean([R["folds"][ln]["arms"][label][key] for ln in R["folds"]]))
    best = max(MODES, key=lambda m: agg(m[0]))
    uses = [ln for ln in R["folds"] if any(R["folds"][ln]["swap"][m]["uses_context"]
                                           for m in ("gene", "gate"))]
    report("\n" + "=" * 100)
    report(f"  MEAN DELTA VS THE HELD-OUT CELL'S OWN TIDE, over {len(R['folds'])} leave-one-cell-out folds")
    for label, _m in MODES:
        report(f"    {label:38s} {agg(label):+.4f}")
    verdict = (
        f"ACROSS {len(R['folds'])} LEAVE-ONE-CELL-LINE-OUT FOLDS, the best context arm is '{best[0]}' at "
        f"{agg(best[0]):+.4f} against the held-out cell's own tide, and the cell-blind arm is at "
        f"{agg('A no context'):+.4f}. "
        + (f"CONTEXT IS USED where the swap says so: {len(uses)}/{len(R['folds'])} folds degrade when handed "
           f"another cell's baseline. " if uses else
           f"THE SWAP SAYS THE CONTEXT CHANNEL IS IGNORED: no fold degrades materially when handed another "
           f"cell's baseline, so the context arms are not failing to help -- they are not being used at "
           f"all, which accuracy alone could never have distinguished from a null. ")
        + f"Tests B and C are reported but are NOT identifiable at this rectangle: leave-one-cell-out fits "
          f"a cell encoder from 3 training points. Accessibility is absent for all four lines, so D and E "
          f"are RNA-only.")
    report(f"\n  VERDICT: {verdict}")
    R["verdict"] = verdict
    R["mean_delta_vs_tide"] = {label: agg(label) for label, _m in MODES}
    R["limits"] = [
        "4 cell lines is 4 folds. Every aggregate here is a mean over 4 numbers",
        "tests B and C are unidentifiable at this rectangle and are descriptive only",
        "accessibility and protein are absent for these lines; the context is RNA baseline alone",
        "the shared-perturbation set is ~1,398 knockouts common to all lines, which is a biased subset -- "
        "the perturbations chosen to be screened everywhere are not a random sample of perturbations",
        "the tide reference is computed FROM the held-out cell. It is a scoring reference only and never "
        "an input, but it means the bar each fold must clear is that cell's own answer",
    ]
    R["log"] = log
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "context_tests.json", "w"), indent=1)
    report(f"\n  -> {OUT/'context_tests.json'}")


if __name__ == "__main__":
    main()
