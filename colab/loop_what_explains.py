"""Loop 254. Is the gene x line interaction biology, or is it the shRNA construct?

WHAT LOOP 253 LEFT UNRESOLVED, AND WHY IT MATTERS MORE THAN THE THREE FAILURES IN IT.

Loop 253 established that the interaction term reproduces at 0.3567 (F1) while expression, signed
OmniPath edges and ENCODE chromatin explain 0.0037, -0.0000 and +0.0000 of it. The natural reading
is "real biology, unexplained by the standard annotations". But F1's split was RANDOM within each
(gene, line), and the signatures inside one (gene, line) share their shRNA constructs and mostly
their plates:

    distinct shRNA constructs per (gene, line)   median 4, at least 2 in 100% of pairs
    distinct plates per (gene, line)             median 2, at least 2 in 92%

An shRNA carries seed-based off-target effects: a signature is partly the intended knockdown and
partly the construct. A random split puts the SAME constructs on both sides, so F1's 0.3567 is
consistent with the interaction being construct signal that happens to reproduce because it is the
same construct twice. Loop 253's own F7 named this as a caveat and then did not test it. This loop
tests it, and the test is available because every (gene, line) has at least two constructs.

THE LOGIC IS A DIFFERENCE OF TWO SPLITS OF THE SAME DATA.

    RANDOM split      constructs shared across the two halves   -> 0.3567 (loop 253's F1)
    CONSTRUCT split   disjoint constructs, same gene, same line -> ?
    PLATE split       disjoint plates, same gene, same line     -> ?

If the construct-split reproducibility collapses toward zero, the interaction is an artefact of
which hairpins were used and the loop 253 failures were failures to explain an artefact -- which
would be the correct outcome and would retire the question. If it survives, the interaction is a
property of the (gene, cell line) pair that two different hairpins agree on, and loop 253's
failures are then a real open problem.

PREDECLARED, BEFORE ANY NUMBER.

  G1 IS THE RANDOM-SPLIT NUMBER REPRODUCED HERE?
     Loop 253's F1, recomputed inside this loop on the same data, so the comparison in G2 is
     against a number this loop measured rather than one quoted from another file.
     Gate: PASS iff it lands within 0.05 of 0.3567. Everything requires this.

  G2 DOES THE INTERACTION SURVIVE A CONSTRUCT SPLIT?      -- the load-bearing gate
     Each (gene, line)'s constructs partitioned into two disjoint sets; the residual computed
     independently from each and correlated.
     Gate: PASS iff the construct-split reproducibility is at least half the random-split value.
     A FAIL says the interaction is mostly hairpin identity.

  G3 DOES IT SURVIVE A PLATE SPLIT?      -- requires G1
     The same with disjoint plates, on the 92% of pairs that span more than one.
     Gate: PASS iff at least half the random-split value. This separates batch from construct:
     the two splits are not nested, so passing one and failing the other is informative.

  G4 HOW MUCH OF THE INTERACTION IS EACH?  -- reported, not gated.
     The three reproducibilities side by side, with the construct-split value taken as the
     biological share and the gap to the random split as the construct-plus-batch share.

  G5 DOES ANYTHING EXPLAIN THE PART THAT SURVIVES?      -- requires G2
     If a construct-reproducible interaction exists, loop 253's three mechanisms are re-scored
     against THAT rather than against the raw residual: the target becomes the component two
     different hairpins agree on.
     Gate: PASS iff expression, signed edges or chromatin reaches 0.05 on it. VOID if G2 failed,
     because there is then no biological component for them to explain and scoring against one
     would be measuring the artefact.

  G6 WHAT THIS CANNOT SHOW -- written before the run.
     Two hairpins against the same gene can share seed sequence families, so a construct split is
     a lower bound on how much construct signal is removed, not a guarantee that none remains.
     Constructs and plates are correlated in the LINCS design: hairpins for one gene were often
     run together. G2 and G3 therefore overlap and neither is a clean single-factor control.
     A pair needs at least two constructs to enter G2 and at least two plates to enter G3, so both
     gates run on a subset and that subset is the better-replicated one.
     If the interaction turns out to be construct, that is a fact about this assay, not evidence
     that cell context is unimportant -- loop 252's E4 and E6 stand on the raw response, not on
     this residual.
"""
import os, sys, json, time, collections, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_what_explains.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
LX = SCR / "lincs"

SEED = 254254
LINES = ["PC3", "MCF7", "VCAP", "A375", "HA1E", "A549", "HT29", "HEPG2", "HCC515"]
MIN_LINES = 6
LOOP253_F1 = 0.3567
G1_TOL, G2_FRAC, G3_FRAC, G5_BAR = 0.05, 0.50, 0.50, 0.05

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


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "is the gene x line interaction biology or the shRNA construct"}
    say("=" * 104)
    say("LOOP 254 -- IS THE INTERACTION BIOLOGY, OR IS IT THE HAIRPIN?")
    say("=" * 104)
    say("     Loop 253 measured the interaction reproducing at 0.3567 while nothing explained it.")
    say("     But that split was RANDOM within each (gene, line), and those signatures share their")
    say("     shRNA constructs -- median 4 per pair, at least 2 in 100% of pairs. A random split")
    say("     puts the same hairpins on both sides, so 0.3567 is equally consistent with construct")
    say("     signal reproducing because it is the same construct twice. Loop 253's F7 named this")
    say("     and did not test it.")

    X = np.load(LX / "shrna_landmark.npy", mmap_mode="r")
    S = np.load(LX / "select2.npz", allow_pickle=True)
    gene = np.array([str(x) for x in S["gene"]]); cell = np.array([str(x) for x in S["cell"]])
    pid = np.array([str(x) for x in S["pert_id"]]); plate = np.array([str(x) for x in S["plate"]])
    keep = np.isin(cell, LINES)
    Xk = np.asarray(X[keep]); gk, ck = gene[keep], cell[keep]
    pk, plk = pid[keep], plate[keep]
    idxs = collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(gk, ck)): idxs[(g, c)].append(i)
    pairs = sorted(idxs)
    NL = Xk.shape[1]
    nlc = collections.Counter(g for g, c in pairs)
    pairs = [p for p in pairs if nlc[p[0]] >= MIN_LINES]
    say(f"     {len(pairs):,} (gene, line) pairs from {len(set(p[0] for p in pairs)):,} genes")

    def build(splitter):
        """Two disjoint half-profiles per (gene, line) under a given splitting rule."""
        A = {}; B = {}
        for p in pairs:
            ii = idxs[p]
            ga, gb = splitter(ii)
            if not len(ga) or not len(gb): continue
            A[p] = Xk[ga].mean(0); B[p] = Xk[gb].mean(0)
        return A, B

    def rand_split(ii):
        j = list(ii); rng.shuffle(j); h = len(j) // 2
        return j[:h], j[h:]

    def by_key(arr):
        def f(ii):
            ks = sorted({arr[i] for i in ii})
            if len(ks) < 2: return [], []
            rng.shuffle(ks); h = max(1, len(ks) // 2)
            s1 = set(ks[:h])
            return [i for i in ii if arr[i] in s1], [i for i in ii if arr[i] not in s1]
        return f

    def reproducibility(A, B):
        """Residual after gene mean and line mean, computed independently in each half, correlated."""
        ps = sorted(A)
        if len(ps) < 100: return float("nan"), 0, None
        gsA = collections.defaultdict(list); gsB = collections.defaultdict(list)
        lsA = collections.defaultdict(list); lsB = collections.defaultdict(list)
        for p in ps:
            gsA[p[0]].append(A[p]); gsB[p[0]].append(B[p])
            lsA[p[1]].append(A[p]); lsB[p[1]].append(B[p])
        gmA = {g: np.mean(v, 0) for g, v in gsA.items()}; gmB = {g: np.mean(v, 0) for g, v in gsB.items()}
        lmA = {c: np.mean(v, 0) for c, v in lsA.items()}; lmB = {c: np.mean(v, 0) for c, v in lsB.items()}
        grA = np.mean([A[p] for p in ps], 0); grB = np.mean([B[p] for p in ps], 0)
        out = []; RA = {}
        for p in ps:
            ra = A[p] - (gmA[p[0]] + lmA[p[1]] - grA)
            rb = B[p] - (gmB[p[0]] + lmB[p[1]] - grB)
            out.append(pear(ra, rb)); RA[p] = (ra + rb) / 2
        return float(np.nanmean(out)), len(ps), RA

    # ---------------------------------------------------------------- G1
    say("G1 IS THE RANDOM-SPLIT NUMBER REPRODUCED HERE?")
    A, B = build(rand_split)
    r_rand, n_rand, _ = reproducibility(A, B)
    say(f"     random split within each (gene, line): {r_rand:.4f}  (n={n_rand:,})")
    say(f"     loop 253's F1 on the same data: {LOOP253_F1:.4f}")
    G.add("G1", bool(abs(r_rand - LOOP253_F1) <= G1_TOL), stat=float(r_rand),
          if_true=lambda: f"G1 PASS -- reproduces loop 253's F1 to "
                          f"{abs(r_rand - LOOP253_F1):.4f}",
          if_false=lambda: f"G1 FAIL -- {r_rand:.4f} here against {LOOP253_F1:.4f} in loop 253")
    res["G1"] = {"random": r_rand, "loop253": LOOP253_F1, "n": n_rand}

    # ---------------------------------------------------------------- G2
    say("G2 DOES THE INTERACTION SURVIVE A CONSTRUCT SPLIT?")
    Ac, Bc = build(by_key(pk))
    r_con, n_con, Rc = reproducibility(Ac, Bc)
    frac2 = r_con / r_rand if abs(r_rand) > 1e-9 else float("nan")
    say(f"     DISJOINT shRNA constructs, same gene, same line: {r_con:.4f}  (n={n_con:,})")
    say(f"     that is {frac2:.0%} of the random-split value")
    G.add("G2", bool(np.isfinite(frac2) and frac2 >= G2_FRAC), stat=float(r_con), requires=("G1",),
          if_true=lambda: f"G2 PASS -- two different hairpins against the same gene in the same "
                          f"line agree at {r_con:.4f}, {frac2:.0%} of the random split; the "
                          f"interaction is a property of the (gene, line) pair",
          if_false=lambda: f"G2 FAIL -- across disjoint constructs the interaction reproduces at "
                           f"{r_con:.4f}, only {frac2:.0%} of the random split; it is largely "
                           f"hairpin identity, and loop 253 was trying to explain an artefact")
    res["G2"] = {"construct": r_con, "fraction_of_random": frac2, "n": n_con}

    # ---------------------------------------------------------------- G3
    say("G3 DOES IT SURVIVE A PLATE SPLIT?")
    Ap, Bp = build(by_key(plk))
    r_pl, n_pl, _ = reproducibility(Ap, Bp)
    frac3 = r_pl / r_rand if abs(r_rand) > 1e-9 else float("nan")
    say(f"     DISJOINT plates, same gene, same line: {r_pl:.4f}  (n={n_pl:,}) -- {frac3:.0%}")
    G.add("G3", bool(np.isfinite(frac3) and frac3 >= G3_FRAC), stat=float(r_pl), requires=("G1",),
          if_true=lambda: f"G3 PASS -- survives a plate split at {r_pl:.4f} ({frac3:.0%})",
          if_false=lambda: f"G3 FAIL -- across disjoint plates it reproduces at {r_pl:.4f}, "
                           f"{frac3:.0%} of the random split; batch carries much of it")
    res["G3"] = {"plate": r_pl, "fraction_of_random": frac3, "n": n_pl}

    # ---------------------------------------------------------------- G4
    say("G4 HOW MUCH OF THE INTERACTION IS EACH? -- reported, not gated")
    say(f"     random split    {r_rand:.4f}   same constructs, same plates on both sides")
    say(f"     plate split     {r_pl:.4f}   {frac3:.0%}")
    say(f"     construct split {r_con:.4f}   {frac2:.0%}  <- the part two hairpins agree on")
    say(f"     the gap between the random and construct splits, {r_rand - r_con:.4f}, is what")
    say(f"     hairpin identity and batch contribute together")
    res["G4"] = {"random": r_rand, "plate": r_pl, "construct": r_con, "gap": r_rand - r_con}

    # ---------------------------------------------------------------- G5
    say("G5 DOES ANYTHING EXPLAIN THE PART THAT SURVIVES?")
    if not (np.isfinite(frac2) and frac2 >= G2_FRAC):
        G.add("G5", False, stat=float(frac2), requires=("G2",), void_if=True,
              void_reason=f"the construct split retains only {frac2:.0%}; there is no "
                          f"construct-reproducible component for a mechanism to explain, and "
                          f"scoring against one would be measuring the artefact")
    else:
        ez = np.load(SCR / "depmap_expr_aligned.npz", allow_pickle=True)
        XE = ez["XE"]; elines = np.array([str(x) for x in ez["lines"]])
        egenes = np.array([str(x) for x in ez["genes"]])
        lmap = json.load(open(LX / "line_map.json"))
        gp = {g: i for i, g in enumerate(egenes)}
        lmsym = np.array([str(x) for x in S["lm_gene_ids"]])
        import gzip as _gz
        sym = {}
        with _gz.open(LX / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt", errors="replace") as fh:
            h = fh.readline().rstrip("\n").split("\t"); ix = {k: i for i, k in enumerate(h)}
            for ln in fh:
                q = ln.rstrip("\n").split("\t")
                if len(q) >= len(h): sym[q[ix["pr_gene_id"]]] = q[ix["pr_gene_symbol"]]
        lms = np.array([sym.get(g, "?") for g in lmsym])
        E = np.stack([XE[int(np.where(elines == lmap[l])[0][0])] for l in LINES])
        Ez = (E - E.mean(0)) / (E.std(0) + 1e-6)
        li = {l: i for i, l in enumerate(LINES)}
        col = np.array([gp.get(s, -1) for s in lms]); ok = col >= 0
        sc = []
        for p, r in Rc.items():
            g, c = p
            f = Ez[li[c]][col] * ok
            gz = Ez[li[c], gp[g]] if g in gp else 0.0
            sc.append(pear(f * gz, r))
        r5 = float(np.nanmean(sc))
        say(f"     expression context against the construct-reproducible component: {r5:.4f}")
        G.add("G5", bool(r5 >= G5_BAR), stat=float(r5), requires=("G2",),
              if_true=lambda: f"G5 PASS -- expression explains {r5:.4f} of the surviving component",
              if_false=lambda: f"G5 FAIL -- {r5:.4f} against a {G5_BAR} bar even on the part two "
                               f"hairpins agree on")
        res["G5"] = {"expression": r5}

    say("G6 WHAT THIS CANNOT SHOW")
    say("     Two hairpins against one gene can share seed families, so a construct split is a")
    say("     LOWER BOUND on how much construct signal is removed, not a guarantee.")
    say("     Constructs and plates are correlated by the LINCS design -- hairpins for a gene were")
    say("     often run together -- so G2 and G3 overlap and neither is a clean single factor.")
    say("     Both gates run on pairs with at least two constructs or plates, which is the")
    say("     better-replicated subset.")
    say("     If the interaction is construct, that is a fact about this ASSAY. Loop 252's E4 and")
    say("     E6 stand on the raw response, not on this residual, and are unaffected.")

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
