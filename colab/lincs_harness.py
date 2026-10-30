"""Shared, verified harness for testing what explains the LINCS gene x line interaction.

Every hypothesis tested against this interaction must go through the SAME data loading, the SAME
residual definition and the SAME held-out-by-cell-line protocol, or the numbers are not comparable.
Loop 254's G1 is the reason this file exists: loop 253's F1 subtracted only the gene mean while its
F3-F5 targeted the full additive residual, so the loop compared two different quantities and
reported the difference as a finding.

THE TARGET. For a held-out cell line c, with gene means taken over the TRAINING lines only:

    R[g, c]  =  P[g, c]  -  ( genemean_train[g] + linemean[c] - grand )

This is what loop 252's best arm (A3_ADDITIVE, 0.4477) cannot express, and it is 68.7% of the
variance. Its reproducibility across DISJOINT shRNA constructs is 0.2487 (loop 254 G2), which is
the ceiling any explanation can reach: nothing can explain more of it than two independent
reagents agree on.

A hypothesis supplies `features(gene, line, ctx) -> (n_landmarks, k)` and nothing else. The
harness fits ridge on the training lines and scores the held-out line, so a hypothesis cannot
accidentally leak, choose its own split, or score against a different residual.
"""
import collections, gzip, json
from pathlib import Path
import numpy as np

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
LX = SCR / "lincs"
LINES = ["PC3", "MCF7", "VCAP", "A375", "HA1E", "A549", "HT29", "HEPG2", "HCC515"]
MIN_LINES = 6
CONSTRUCT_CEILING = 0.2487          # loop 254 G2: what two disjoint hairpins agree on


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def load():
    """Everything a hypothesis might need, loaded once and verified."""
    X = np.load(LX / "shrna_landmark.npy", mmap_mode="r")
    S = np.load(LX / "select2.npz", allow_pickle=True)
    gene = np.array([str(x) for x in S["gene"]]); cell = np.array([str(x) for x in S["cell"]])
    lmids = np.array([str(x) for x in S["lm_gene_ids"]])
    keep = np.isin(cell, LINES)
    Xk, gk, ck = np.asarray(X[keep]), gene[keep], cell[keep]
    idxs = collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(gk, ck)): idxs[(g, c)].append(i)
    pairs = sorted(idxs)
    nlc = collections.Counter(g for g, c in pairs)
    pairs = [p for p in pairs if nlc[p[0]] >= MIN_LINES]
    Pm = np.stack([Xk[idxs[p]].mean(0) for p in pairs])
    pg = np.array([p[0] for p in pairs]); pc = np.array([p[1] for p in pairs])

    sym = {}
    with gzip.open(LX / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt", errors="replace") as fh:
        h = fh.readline().rstrip("\n").split("\t"); ix = {k: i for i, k in enumerate(h)}
        for ln in fh:
            q = ln.rstrip("\n").split("\t")
            if len(q) >= len(h): sym[q[ix["pr_gene_id"]]] = q[ix["pr_gene_symbol"]]
    lmsym = np.array([sym.get(g, "?") for g in lmids])

    lmap = json.load(open(LX / "line_map.json"))
    ez = np.load(SCR / "depmap_expr_aligned.npz", allow_pickle=True)
    XE = ez["XE"]; el = np.array([str(x) for x in ez["lines"]])
    eg = np.array([str(x) for x in ez["genes"]])
    E = np.stack([XE[int(np.where(el == lmap[l])[0][0])] for l in LINES])
    Ez = ((E - E.mean(0)) / (E.std(0) + 1e-6)).astype(np.float32)

    return dict(Pm=Pm, pg=pg, pc=pc, lmsym=lmsym, LINES=LINES, lmap=lmap,
                li={l: i for i, l in enumerate(LINES)}, expr=E, exprz=Ez,
                expr_genes=eg, expr_gpos={g: i for i, g in enumerate(eg)},
                genes=sorted(set(pg.tolist())), NL=Pm.shape[1],
                lm_gpos={s: i for i, s in enumerate(lmsym)})


def residuals(D, hold):
    """(indices into Pm, residual matrix) for the held-out line, gene means from training only."""
    Pm, pg, pc = D["Pm"], D["pg"], D["pc"]
    tr = pc != hold
    gm = {}
    for g in D["genes"]:
        m = tr & (pg == g)
        if m.sum(): gm[g] = Pm[m].mean(0)
    lmv = Pm[pc == hold].mean(0)
    grand = Pm[tr].mean(0)
    te = np.where(pc == hold)[0]
    keep, R = [], []
    for j in te:
        if pg[j] not in gm: continue
        keep.append(j); R.append(Pm[j] - (gm[pg[j]] + lmv - grand))
    return np.array(keep), (np.stack(R) if R else np.zeros((0, D["NL"]), np.float32))


def evaluate(D, features, lines=None, max_train_per_line=500, lam=1e-2, shuffle_line=False,
             rng=None):
    """Ridge on training lines, scored on the held-out line. Returns per-(gene, line) correlations.

    `features(gene, line, D) -> (NL, k)`. The hypothesis never sees the target and never chooses
    the split, so it cannot leak or self-select a protocol."""
    lines = lines or D["LINES"]
    rng = rng or np.random.default_rng(0)
    out = []
    for hold in lines:
        Xtr, ytr = [], []
        for l in lines:
            if l == hold: continue
            te2, R2 = residuals(D, l)
            for i2, j2 in enumerate(te2[:max_train_per_line]):
                f = features(D["pg"][j2], l, D)
                if f is None or not np.isfinite(f).all(): continue
                Xtr.append(f); ytr.append(R2[i2])
        if not Xtr: continue
        Xa = np.concatenate(Xtr, 0); ya = np.concatenate(ytr)
        Z = np.concatenate([Xa, np.ones((len(Xa), 1), np.float32)], 1)
        A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
        b = np.linalg.solve(A, Z.T @ ya)
        te, R = residuals(D, hold)
        src = hold if not shuffle_line else str(rng.choice([l for l in lines if l != hold]))
        for i2, j2 in enumerate(te):
            f = features(D["pg"][j2], src, D)
            if f is None or not np.isfinite(f).all(): continue
            p = np.concatenate([f, np.ones((len(f), 1), np.float32)], 1) @ b
            out.append(pear(p, R[i2]))
    return np.asarray(out)


def expression_baseline(g, line, D):
    """The reference hypothesis every other one must beat: landmark expression in this line, the
    knocked-down gene's expression in this line, and their product. Loop 253's F3 scored 0.0037."""
    i = D["li"][line]
    col = np.array([D["expr_gpos"].get(s, -1) for s in D["lmsym"]])
    ok = col >= 0
    lm = D["exprz"][i][col] * ok
    gz = D["exprz"][i, D["expr_gpos"][g]] if g in D["expr_gpos"] else 0.0
    return np.stack([lm, np.full(D["NL"], gz, np.float32), lm * gz], 1).astype(np.float32)
