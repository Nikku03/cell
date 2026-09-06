"""Loop 253. Signed edges and chromatin, rerun where the interaction term actually exists.

WHY THESE TWO ARE BEING RERUN AT ALL. Loop 242 found signed OmniPath edges worth +0.0001 on
held-out prediction and loop 245 found K562 chromatin worth +0.0328 beyond expression level. Both
were recorded as failures. Loop 252 then measured, on LINCS, that the response to a knockdown
splits as

    gene main effect        27.6%
    cell-line main effect    3.8%
    gene x line INTERACTION  68.7%

while the same decomposition on DepMap (loop 240's X2) put 85% BETWEEN GENE with the context term
near-vacuous. So loops 242 and 245 were asking mechanisms to explain a context effect on data whose
context term barely existed. Their numbers stand for that data. They are not statements about the
mechanisms, and this loop is what actually tests those.

THE TARGET IS THE INTERACTION, NOT THE RESPONSE. Loop 252's best arm, A3_ADDITIVE = gene mean plus
line mean, reached 0.4477 with no interaction in it at all. Anything a mechanism explains must be
what A3 cannot, so the target here is

    R[g, c]  =  P[g, c]  -  ( gene mean over TRAINING lines  +  held-out line mean  -  grand mean )

held out BY CELL LINE, exactly as loop 252 held it out. On DepMap this residual was 15% of the
variance; here it is 69%, which is the whole reason to run this.

WHAT EACH MECHANISM PREDICTS ABOUT THE INTERACTION, stated as a mechanism rather than as a feature:

    EXPRESSION   a gene that is not transcribed in this cell line cannot be knocked down in it, so
                 its knockdown should do less here than the gene average predicts. This is the
                 premise loop 240's Z1 tested on DepMap and FAILED (+0.0228 against a 0.05 bar,
                 within-gene correlation -0.0169). F2 re-asks it with a transcriptional readout
                 instead of a growth phenotype.
    SIGNED EDGE  a target of the knocked-down gene moves in the direction the edge sign says -- but
                 only where that target is itself expressed. The edge is a property of the genome;
                 whether it can fire is a property of the cell. That product is an interaction and
                 it is the form in which a signed graph can explain context at all.
    CHROMATIN    a target sitting in closed chromatin in this line cannot respond in it. Available
                 for 4 of the 9 lines -- A549, MCF7, HEPG2, PC3 -- with ATAC and H3K27ac from
                 ENCODE, GRCh38. The other five are reported as excluded, not quietly dropped.

PREDECLARED, BEFORE ANY NUMBER.

  F1 IS THERE INTERACTION STRUCTURE TO PREDICT, OR ONLY NOISE?
     69% of the variance being interaction does not make it predictable: replicate noise lands in
     the same term. The signatures for each (gene, line) are split in half and the residual is
     computed independently in each.
     Gate: PASS iff the split-half correlation of the residual exceeds 0.30. Everything requires
     this, because a mechanism cannot explain noise and a gate that tried would be measuring the
     assay.

  F2 CAN YOU KNOCK DOWN A GENE THAT IS NOT EXPRESSED?      -- requires F1
     The root premise, failed once on DepMap. For each knocked-down gene, its signatures are split
     by whether the gene is in the top or bottom third of its own expression across the nine lines.
     Gate: PASS iff the response magnitude is at least 20% larger in the top third than the bottom
     third, paired within gene so the comparison is not across different genes.

  F3 DOES EXPRESSION CONTEXT PREDICT THE INTERACTION?      -- requires F1
     Ridge on the held-out line's expression of the knocked-down gene and of each landmark.
     Gate: PASS iff held-out correlation with the residual exceeds 0.05.

  F4 DO SIGNED EDGES PREDICT THE INTERACTION?      -- requires F1. The loop 242 rerun.
     Signed OmniPath edge from the knocked-down gene to the landmark, crossed with whether that
     landmark is expressed in the held-out line.
     Gate: PASS iff it adds at least 0.02 over F3's expression-only model, paired over held-out
     (gene, line) pairs. Loop 242's equivalent number was +0.0001.

  F5 DOES CHROMATIN PREDICT THE INTERACTION?      -- requires F1. The loop 245 rerun.
     ATAC and H3K27ac at each landmark's promoter in the held-out line, on the four lines that
     have both.
     Gate: PASS iff it adds at least 0.02 over expression alone on those four lines. Loop 245's
     equivalent was +0.0328 beyond expression, on a different quantity in a different cell line.

  F6 CONTROL: THE WRONG CELL LINE.      -- requires F3, VOID if F3 is under 0.02
     Every cell-line-specific input taken from a different line, the graph and the gene identity
     unchanged.
     Gate: PASS iff the best arm's advantage over the gene-only model collapses to under 25%.

  F7 WHAT THIS CANNOT SHOW -- written before the run.
     978 landmarks. A signed edge whose target is not among them is invisible here, and OmniPath's
     coverage of the landmark set is what it is rather than what the biology is.
     Chromatin covers four of nine lines, and those four are the best-studied ones, so the
     mechanism is being tested where the data is best rather than where it is representative.
     shRNA seed effects put construct-specific signal into every signature; that inflates the
     apparent gene term and therefore makes the interaction term harder, not easier, to explain.
     DepMap expression and LINCS signatures come from different laboratories and different stocks
     of nominally the same lines.
     A mechanism failing here still means it failed on the best data this project has. It does not
     mean the mechanism is absent from biology.
"""
import os, sys, json, time, gzip, csv, collections, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_interaction_mech.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
LX = SCR / "lincs"

SEED = 253253
LINES = ["PC3", "MCF7", "VCAP", "A375", "HA1E", "A549", "HT29", "HEPG2", "HCC515"]
CHROM_LINES = ["A549", "MCF7", "HEPG2", "PC3"]
MIN_LINES, TSS_WIN = 6, 2000
F1_BAR, F2_BAR, F3_BAR, F4_BAR, F5_BAR, F6_MAX = 0.30, 0.20, 0.05, 0.02, 0.02, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


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


def ridge_fit(X, y, lam=1e-2):
    Z = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Z.T @ Z + lam * len(X) * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y)


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "signed edges and chromatin against the gene x line interaction"}
    say("=" * 104)
    say("LOOP 253 -- SIGNED EDGES AND CHROMATIN, WHERE THE INTERACTION TERM ACTUALLY EXISTS")
    say("=" * 104)
    say("     Loop 242 put signed edges at +0.0001 and loop 245 put chromatin at +0.0328. Both ran")
    say("     on data whose gene x line interaction was near-vacuous (DepMap: 85% between-gene).")
    say("     LINCS is 68.7% interaction. The target here is the residual A3_ADDITIVE cannot")
    say("     explain -- 69% of the variance rather than 15%.")

    X = np.load(LX / "shrna_landmark.npy", mmap_mode="r")
    S = np.load(LX / "select.npz", allow_pickle=True)
    gene = np.array([str(x) for x in S["gene"]]); cell = np.array([str(x) for x in S["cell"]])
    lmids = np.array([str(x) for x in S["lm_gene_ids"]])
    keep = np.isin(cell, LINES)
    Xk, gk, ck = np.asarray(X[keep]), gene[keep], cell[keep]

    key = collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(gk, ck)): key[(g, c)].append(i)
    pairs = sorted(key)
    Pm = np.zeros((len(pairs), Xk.shape[1]), np.float32)
    Ha = np.zeros_like(Pm); Hb = np.zeros_like(Pm)
    for j, k_ in enumerate(pairs):
        idx = key[k_]; Pm[j] = Xk[idx].mean(0)
        h = max(1, len(idx) // 2)
        Ha[j] = Xk[idx[:h]].mean(0); Hb[j] = Xk[idx[h:]].mean(0) if len(idx) > 1 else np.nan
    pg = np.array([p[0] for p in pairs]); pc = np.array([p[1] for p in pairs])
    nl = collections.Counter(pg)
    good = np.array([nl[g] >= MIN_LINES for g in pg])
    Pm, Ha, Hb, pg, pc = Pm[good], Ha[good], Hb[good], pg[good], pc[good]
    genes = sorted(set(pg.tolist()))
    NL = Pm.shape[1]
    say(f"     {len(pg):,} (gene, line) profiles, {len(genes):,} genes, {NL} landmarks")

    # landmark symbols
    sym = {}
    with gzip.open(LX / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt", errors="replace") as fh:
        h = fh.readline().rstrip("\n").split("\t"); ix = {k: i for i, k in enumerate(h)}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= len(h): sym[p[ix["pr_gene_id"]]] = p[ix["pr_gene_symbol"]]
    lmsym = np.array([sym.get(g, "?") for g in lmids])

    # ---------------------------------------------------------------- residual
    grand = Pm.mean(0)
    def residual(hold):
        tr = pc != hold
        gm = {}
        for g in genes:
            m = tr & (pg == g)
            if m.sum(): gm[g] = Pm[m].mean(0)
        lmv = Pm[pc == hold].mean(0)
        te = np.where(pc == hold)[0]
        R = np.full((len(te), NL), np.nan, np.float32)
        for i, j in enumerate(te):
            if pg[j] in gm: R[i] = Pm[j] - (gm[pg[j]] + lmv - grand)
        return te, R

    # ---------------------------------------------------------------- F1
    say("F1 IS THERE INTERACTION STRUCTURE TO PREDICT, OR ONLY NOISE?")
    ok = np.isfinite(Hb).all(1)
    ga = {}; gb = {}
    for g in genes:
        m = (pg == g) & ok
        if m.sum() >= 3: ga[g] = Ha[m].mean(0); gb[g] = Hb[m].mean(0)
    sh = []
    for j in np.where(ok)[0]:
        g = pg[j]
        if g not in ga: continue
        ra = Ha[j] - ga[g]; rb = Hb[j] - gb[g]
        sh.append(pear(ra, rb))
    r1 = float(np.nanmean(sh))
    say(f"     signatures split in half within each (gene, line); residual computed independently")
    say(f"     split-half correlation of the interaction residual: {r1:.4f}  (n={len(sh):,})")
    G.add("F1", bool(r1 >= F1_BAR), stat=float(r1),
          if_true=lambda: f"F1 PASS -- the interaction reproduces at {r1:.4f}; there is structure "
                          f"to explain, not only replicate noise",
          if_false=lambda: f"F1 FAIL -- {r1:.4f} against a {F1_BAR} bar; the interaction term is "
                           f"largely assay noise and no mechanism can explain it")
    res["F1"] = {"split_half": r1, "n": len(sh)}

    # ---------------------------------------------------------------- expression
    ez = np.load(SCR / "depmap_expr_aligned.npz", allow_pickle=True)
    XE = ez["XE"]; elines = np.array([str(x) for x in ez["lines"]])
    egenes = np.array([str(x) for x in ez["genes"]])
    lmap = json.load(open(LX / "line_map.json"))
    epos = {l: int(np.where(elines == lmap[l])[0][0]) for l in LINES if lmap.get(l) in set(elines)}
    gpos = {g: i for i, g in enumerate(egenes)}
    E = np.zeros((len(LINES), len(egenes)), np.float32)
    for i, l in enumerate(LINES): E[i] = XE[epos[l]]
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-6)          # z across the nine lines
    li = {l: i for i, l in enumerate(LINES)}
    lmcol = np.array([gpos.get(s, -1) for s in lmsym])
    have_lm = lmcol >= 0
    say(f"     expression matched for all {len(epos)} lines; {int(have_lm.sum())} of {NL} "
        f"landmarks resolve to a DepMap gene")

    # ---------------------------------------------------------------- F2
    say("F2 CAN YOU KNOCK DOWN A GENE THAT IS NOT EXPRESSED?")
    say("     loop 240's Z1 asked this on DepMap fitness and FAILED (+0.0228, within-gene corr")
    say("     -0.0169). Here the readout is transcriptional.")
    hi_l, lo_l = [], []
    for g in genes:
        if g not in gpos: continue
        m = pg == g
        if m.sum() < 5: continue
        ez_g = Ez[[li[c] for c in pc[m]], gpos[g]]
        mag = np.linalg.norm(Pm[m], axis=1)
        if np.nanstd(ez_g) < 1e-6: continue
        thi = np.nanpercentile(ez_g, 66); tlo = np.nanpercentile(ez_g, 33)
        a = mag[ez_g >= thi]; b = mag[ez_g <= tlo]
        if len(a) and len(b): hi_l.append(a.mean()); lo_l.append(b.mean())
    hi_l, lo_l = np.array(hi_l), np.array(lo_l)
    ratio = float(np.mean(hi_l / np.maximum(lo_l, 1e-9))) - 1.0
    d2, se2, z2 = paired(hi_l, lo_l)
    say(f"     {len(hi_l):,} genes; response magnitude where the gene is in its TOP expression")
    say(f"     third {hi_l.mean():.3f} vs BOTTOM third {lo_l.mean():.3f}")
    say(f"     paired within gene: {d2:+.4f} +/- {se2:.4f} ({z2:+.1f} se); relative {ratio:+.1%}")
    G.add("F2", bool(ratio >= F2_BAR and z2 > 0), stat=float(ratio), requires=("F1",),
          if_true=lambda: f"F2 PASS -- a knockdown does {ratio:+.0%} more where the gene is "
                          f"expressed; the premise holds on a transcriptional readout",
          if_false=lambda: f"F2 FAIL -- {ratio:+.1%} against a {F2_BAR:.0%} bar")
    res["F2"] = {"hi": float(hi_l.mean()), "lo": float(lo_l.mean()), "relative": ratio,
                 "paired": d2, "se": se2, "z": z2, "n": len(hi_l)}

    # ---------------------------------------------------------------- graphs
    act, inh = collections.defaultdict(set), collections.defaultdict(set)
    with open(SCR / "reg" / "op_2022.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s_, t_ = r["source_genesymbol"], r["target_genesymbol"]
            if not s_ or not t_ or s_ == t_ or r["is_directed"] != "1": continue
            if r["is_stimulation"] == "1": act[s_].add(t_)
            if r["is_inhibition"] == "1": inh[s_].add(t_)
    lmset = {s: i for i, s in enumerate(lmsym)}
    nsig = sum(1 for g in genes if len((act.get(g, set()) | inh.get(g, set())) & set(lmsym)) >= 1)
    say(f"     OmniPath: {nsig:,} of {len(genes):,} knocked-down genes have a signed edge to at "
        f"least one landmark")

    # ---------------------------------------------------------------- chromatin
    def load_marks(line):
        out = {}
        for mk in ("ATAC", "H3K27ac"):
            p = LX / "chrom" / f"{line}_{mk}.bed.gz"
            if not p.exists(): continue
            by = collections.defaultdict(list)
            with gzip.open(p, "rt") as fh:
                for ln in fh:
                    q = ln.rstrip("\n").split("\t")
                    if len(q) < 7: continue
                    by[q[0]].append((int(q[1]), int(q[2]), float(q[6])))
            out[mk] = {c: (np.array([x[0] for x in sorted(v)]),
                           np.array([x[1] for x in sorted(v)])) for c, v in by.items()}
        return out
    tss = {}
    with open(SCR / "epi" / "tss.tsv") as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4 or not p[0] or len(p[1]) > 2 or p[1] == "MT": continue
            try: tss.setdefault(p[0], (f"chr{p[1]}", int(p[2])))
            except ValueError: pass
    CH = {}
    for l in CHROM_LINES:
        mk = load_marks(l)
        M = np.zeros((NL, 2), np.float32)
        for i, s_ in enumerate(lmsym):
            if s_ not in tss: continue
            ch, po = tss[s_]
            for k2, name in enumerate(("ATAC", "H3K27ac")):
                d = mk.get(name, {}).get(ch)
                if d is None: continue
                st, en = d
                a = np.searchsorted(en, po - TSS_WIN, "right"); b = np.searchsorted(st, po + TSS_WIN, "left")
                M[i, k2] = 1.0 if b > a else 0.0
        CH[l] = M
        say(f"       {l:<7} landmarks with ATAC at TSS: {int(M[:,0].sum()):3d}, "
            f"H3K27ac: {int(M[:,1].sum()):3d}")

    # ---------------------------------------------------------------- feature builder
    def feats(j, src, use_sig, use_chrom):
        g = pg[j]; c = src
        f = [Ez[li[c]][lmcol] * have_lm]                       # landmark expression in this line
        gz = Ez[li[c], gpos[g]] if g in gpos else 0.0
        f.append(np.full(NL, gz, np.float32))                  # the knocked-down gene's expression
        f.append(np.full(NL, gz, np.float32) * f[0])           # their product: the interaction form
        if use_sig:
            a = np.zeros(NL, np.float32); i_ = np.zeros(NL, np.float32)
            for t_ in act.get(g, ()):
                k2 = lmset.get(t_)
                if k2 is not None: a[k2] = 1.0
            for t_ in inh.get(g, ()):
                k2 = lmset.get(t_)
                if k2 is not None: i_[k2] = 1.0
            f += [-a, +i_, -a * f[0], +i_ * f[0]]              # edge, and edge x expressed-here
        if use_chrom:
            M = CH.get(c)
            f += [M[:, 0], M[:, 1], M[:, 0] * f[0]] if M is not None else [np.zeros(NL, np.float32)] * 3
        return np.stack(f, 1)

    def run(use_sig, use_chrom, lines, shuffle_line=False):
        sc = []
        for hold in lines:
            te, R = residual(hold)
            tr_lines = [l for l in lines if l != hold]
            Xtr, ytr = [], []
            for l in tr_lines:
                te2, R2 = residual(l)
                for i2, j2 in enumerate(te2[:600]):
                    if not np.isfinite(R2[i2]).all(): continue
                    Xtr.append(feats(j2, l, use_sig, use_chrom)); ytr.append(R2[i2])
            if not Xtr: continue
            b = ridge_fit(np.concatenate(Xtr, 0), np.concatenate(ytr))
            src = hold if not shuffle_line else str(rng.choice([l for l in lines if l != hold]))
            for i2, j2 in enumerate(te):
                if not np.isfinite(R[i2]).all(): continue
                Xf = feats(j2, src, use_sig, use_chrom)
                p = np.concatenate([Xf, np.ones((NL, 1), np.float32)], 1) @ b
                sc.append(pear(p, R[i2]))
        return np.asarray(sc)

    # ---------------------------------------------------------------- F3
    say("F3 DOES EXPRESSION CONTEXT PREDICT THE INTERACTION?")
    S3 = run(False, False, LINES)
    r3 = float(np.nanmean(S3))
    say(f"     expression only, held out by cell line: {r3:.4f}  (n={int(np.isfinite(S3).sum()):,})")
    G.add("F3", bool(r3 >= F3_BAR), stat=float(r3), requires=("F1",),
          if_true=lambda: f"F3 PASS -- expression context explains {r3:.4f} of the interaction",
          if_false=lambda: f"F3 FAIL -- {r3:.4f} against a {F3_BAR} bar")
    res["F3"] = {"r": r3}

    # ---------------------------------------------------------------- F4
    say("F4 DO SIGNED EDGES PREDICT THE INTERACTION?  (the loop 242 rerun)")
    S4 = run(True, False, LINES)
    d4, se4, z4 = paired(S4, S3)
    say(f"     expression + signed edges {np.nanmean(S4):.4f} vs expression alone {r3:.4f}")
    say(f"     paired {d4:+.4f} +/- {se4:.4f}  ({z4:+.1f} se)")
    say(f"     loop 242's equivalent, on data with almost no interaction term: +0.0001")
    G.add("F4", bool(d4 >= F4_BAR), stat=float(d4), requires=("F1",),
          if_true=lambda: f"F4 PASS -- signed edges add {d4:+.4f} to the interaction",
          if_false=lambda: f"F4 FAIL -- signed edges add {d4:+.4f} against a {F4_BAR} bar, even "
                           f"where the interaction term is 69% of the variance")
    res["F4"] = {"delta": d4, "se": se4, "z": z4, "with": float(np.nanmean(S4))}

    # ---------------------------------------------------------------- F5
    say("F5 DOES CHROMATIN PREDICT THE INTERACTION?  (the loop 245 rerun)")
    say(f"     restricted to {CHROM_LINES}; the other five lines have no ENCODE ATAC + H3K27ac")
    Sb = run(False, False, CHROM_LINES)
    Sc = run(False, True, CHROM_LINES)
    d5, se5, z5 = paired(Sc, Sb)
    say(f"     expression + chromatin {np.nanmean(Sc):.4f} vs expression alone "
        f"{np.nanmean(Sb):.4f}   paired {d5:+.4f} +/- {se5:.4f} ({z5:+.1f} se)")
    G.add("F5", bool(d5 >= F5_BAR), stat=float(d5), requires=("F1",),
          if_true=lambda: f"F5 PASS -- chromatin adds {d5:+.4f} to the interaction",
          if_false=lambda: f"F5 FAIL -- chromatin adds {d5:+.4f} against a {F5_BAR} bar")
    res["F5"] = {"delta": d5, "se": se5, "z": z5, "base": float(np.nanmean(Sb)),
                 "with": float(np.nanmean(Sc)), "lines": CHROM_LINES}

    # ---------------------------------------------------------------- F6
    say("F6 CONTROL: THE WRONG CELL LINE")
    if r3 < 0.02:
        G.add("F6", False, stat=float(r3), requires=("F3",), void_if=True,
              void_reason=f"F3 is {r3:.4f}; there is nothing to collapse")
    else:
        Sh = run(True, False, LINES, shuffle_line=True)
        rs = float(np.nanmean(Sh))
        f6 = rs / max(np.nanmean(S4), 1e-9)
        say(f"     cell-line inputs taken from a different line: {rs:.4f} against a real "
            f"{np.nanmean(S4):.4f}  ({f6:.0%})")
        G.add("F6", bool(f6 <= F6_MAX), stat=float(f6), requires=("F3",),
              if_true=lambda: f"F6 PASS -- collapses to {f6:.0%} on the wrong line",
              if_false=lambda: f"F6 FAIL -- {f6:.0%} survives the wrong line")
        res["F6"] = {"real": float(np.nanmean(S4)), "shuffled": rs, "fraction": f6}

    say("F7 WHAT THIS CANNOT SHOW")
    say("     978 landmarks: a signed edge whose target is not among them is invisible here.")
    say("     Chromatin covers four of nine lines, and they are the best-studied ones, so the")
    say("     mechanism is tested where the data is best rather than where it is representative.")
    say("     shRNA seed effects put construct signal into every signature, which inflates the")
    say("     gene term and makes the interaction HARDER to explain, not easier.")
    say("     DepMap expression and LINCS signatures come from different labs and stocks of")
    say("     nominally the same lines.")
    say("     A mechanism failing here failed on the best data this project has; that is not the")
    say("     same as being absent from biology.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
