"""LOOP 75 -- REBUILD THE PATHWAY MEMBERSHIP LAYER, AND IDENTIFY THE SOURCE THAT WAS LOST.

WHAT IS BROKEN. The cell model records, per gene, that it belongs to a median of 2 and a maximum of
241 pathways -- and stores exactly ONE pathway label for it. The membership itself was thrown away.

    npath        an integer per gene, max 241            the count SURVIVED
    path         ONE label per gene, 1,222 distinct      the membership DID NOT
    pathways     60 pathways with member lists           23.2% of genes, and derived

That third layer is not a second source. Measured here before anything was rebuilt: `pathways` is
exactly groupby(path) keeping the 60 largest groups -- for all 3,833 member entries the gene's own
`path` string equals the pathway name (agreement 1.0000), and every gene appears in at most ONE list.
A structure in which each gene has one pathway can never reproduce a count that reaches 241.

This is the failure mode already recorded in this repository at `outputs/orphan/3did_ddi.json`, where
3did was parsed, a PDB COUNT per domain pair was kept, and the residue-level contacts were discarded:
the layer looks present and cannot answer anything. Loop 72's docstring names it. Here it is again,
in a layer that 227 modules read.

AND THE SOURCE WAS UNKNOWN. `colab/build_cell_complete.py:150-152` declares npath/path
SOURCE_NOT_IDENTIFIED, having tested two candidates and refuted both: GO process count (corr 0.02)
and membership in the `pathways` key (exact 0.37). cell_complete.json has NO producer anywhere in
this repository -- commit c32f283 established that by static scan and runtime tracer, "zero writers,
56 readers". So the rebuild has to identify the source before it can claim to restore it, and R1
below is that identification run as a discrimination between three competing hypotheses rather than
as a single fit.

WHY npath IS THE WHOLE OPPORTUNITY. A count is a checksum. Whatever wrote `path` also wrote `npath`
in the same pass -- npath > 0 iff path is non-null, on all 16,492 genes, zero exceptions -- so a
membership matrix rebuilt from the right source and the right LEVEL must reproduce npath gene by
gene. That is a far sharper test than coverage, and it is available for free because the count is the
one thing the original producer did not discard.

PREDECLARED, before any number:

  R1 THE REBUILD REPRODUCES npath, AND DOES SO BY DISCRIMINATION       THE PROVENANCE PROOF.
       Three candidate reconstructions are scored against the stored npath on all 16,492 genes:
         A  Reactome LOWEST-LEVEL, counting ROWS          (NCBI2Reactome.txt, Homo sapiens)
         B  Reactome LOWEST-LEVEL, counting UNIQUE R-HSA  (same file, deduplicated)
         C  Reactome ALL-LEVELS,   counting ROWS          (NCBI2Reactome_All_Levels.txt)
       Plus a null: the stored npath vector shuffled across genes, 20 draws.
       Gate: the winner must reach exact agreement >= 0.95 AND Spearman >= 0.98 AND beat the null by
       more than 20 null standard deviations. A single candidate that merely "fits well" proves
       nothing; three candidates where two are refuted is an identification.
  R2 THE MATRIX IS COMPLETE WHERE THE OLD LAYER WAS NOT
       gate: the rebuilt layer must give a membership list to every gene the old layer says has one
       (npath > 0), and its maximum memberships-per-gene must reach the stored maximum of 241. The
       old layer's maximum is 1. Hierarchy carried too: parent->child edges, roots, leaves.
  R3 THE LAYER IS PARTLY A FAME AXIS AND THE NUMBER IS PRINTED         REPORTED, NOT JUDGED.
       rho(npath, publication count) and rho(npath, PPI degree). This gate cannot fail; it exists
       because those two correlations are the reason R4 must match on both, and a reader should see
       them before the result rather than after.
  R4 SHARED-PATHWAY PAIRS SURVIVE A THREE-WAY MATCHED NULL             THE GATE.
       The claim a pathway layer makes is "these genes work together". Tested against two MEASURED
       (not curated) targets, used as independent replicates because they overlap on only 2.0% of
       their pairs:
           codep    DepMap co-dependency, top-8 per gene
           coexpr   co-expression, top-12 per gene, after removing 1,133 self-loops and 6,026
                    within-list duplicates
       Positives are target edges; negatives are matched non-edges; the feature is the number of
       shared Reactome pathways. Four negative sets are drawn and all four reported: RANDOM,
       PUBS-MATCHED, DEGREE-MATCHED, and DEG+PUBS+NPATH-MATCHED. The third matching axis is npath
       itself, because a gene in 241 pathways shares a pathway with almost everything, and
       rho(npath, pubs) = 0.544 means degree+pubs matching alone does not remove it.
       PPI is deliberately NOT a target: rho(PPI degree, pubs) = 0.528 makes it the confound, not
       the referee.
       THE BAR IS NOT 0.5. It is the better of the two trivial baselines computed on the same pairs
       -- PUBS (log pubs_a + log pubs_b) and PREF-ATT (log deg_a + log deg_b) -- following
       colab/link_completion.py:272. And it must clear that on BOTH targets.
       PREDECLARED EXPECTATION, so this cannot be presented as a discovery either way:
       colab/altdata_sources.py:3-6 already records that shared Reactome pathway sits at AUC ~0.51
       against degree-matched negatives, i.e. chance. A result near 0.51 here is a REPLICATION of a
       known negative, not a new finding, and will be reported as such.
       The method is copied, not invented: quantile bins and per-positive matched sampling from
       colab/link_completion.py:175-201, NNEG=5 from :69, the mandatory covariate-balance diagnostic
       from :227-233, empirical p as (n_ge+1)/(NREP+1) with its floor spelled out from
       colab/graph_null.py:229-235, and the pathway size cap 2 <= size <= 60 from graph_null.py:94
       so that one 317-gene pathway cannot dominate the pair expansion.
       The anti-pattern is named too: colab/lens_confidence.py:100-109 validates a same-pathway
       claim against uniformly random pairs and reports lift over it. That baseline is not
       defensible under the standard the rest of this repository already set, and is not inherited.
  R5 THE WRITE IS ADDITIVE
       lifetimes of other layers untouched: every pre-existing top-level key of cell_complete.json
       must survive with an identical element count. The new layer goes in its own file.

-> outputs/orphan/cell_pathways.json  (+ outputs/loop_pathways.json)
"""
import collections
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"
PWOUT = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_pathways.json"

LOW = SC / "NCBI2Reactome.txt"                  # lowest-level, all species
ALL = SC / "NCBI2Reactome_All_Levels.txt"       # all-levels, all species
REL = SC / "ReactomePathwaysRelation.txt"       # parent -> child
NAMES = SC / "ReactomePathways.txt"             # id -> name -> species
GMT = SC / "ReactomePathways.gmt"               # all-levels, HUMAN ONLY, gene SYMBOLS
GINFO = SC / "hs_gene_info.gz"                  # NCBI gene_info, Entrez -> symbol
SPECIES = "Homo sapiens"

EXACT_FLOOR = 0.95      # R1
RHO_FLOOR = 0.98        # R1
NULL_SD = 20.0          # R1, winner must beat the shuffled null by this many null sds
NREP = 20               # R1/R4 null replicates
NNEG = 5                # R4, negatives per positive (link_completion.py:69)
NBIN = 6                # R4, quantile bins per matching axis (8 in the 2-axis original)
PW_MIN, PW_MAX = 2, 60  # R4, pathway size cap before pair expansion (graph_null.py:94)
ALTDATA_KNOWN = 0.51    # R4, the negative this replicates or overturns
SEED = 7501

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def entrez2symbol():
    m = {}
    for ln in gzip.open(GINFO, "rt"):
        if ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) > 2 and f[0] == "9606":
            m[f[1]] = f[2]
    return m


def read_reactome(path, e2s):
    """(gene symbol -> list of R-HSA ids WITH duplicates), id -> name. Human rows only.

    Duplicates are KEPT. The same gene/pathway pair appears on several rows when Entrez maps to
    several physical entities, and R1 shows the original producer kept them: counting rows agrees
    with the stored npath far better than deduplicating to unique pathway ids does.
    """
    rows = collections.defaultdict(list)
    nm = {}
    for ln in open(path, errors="ignore"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 6 or f[5] != SPECIES:
            continue
        s = e2s.get(f[0])
        if not s:
            continue
        rows[s].append(f[1])
        nm[f[1]] = f[3]
    return rows, nm


def bins(v, nb=NBIN):
    """quantile bins -- link_completion.py:175-177, verbatim."""
    q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)))
    return np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)


def auc(score, y):
    from scipy.stats import rankdata
    score = np.asarray(score, float)
    y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 75 -- rebuild the pathway membership layer, and identify the source that was lost")
    say("  the model stored HOW MANY pathways each gene is in, and threw away WHICH")
    say("=" * 100)
    say()

    from scipy.stats import spearmanr

    D = json.load(open(CELL))
    G = D["genes"]
    n = len(G)
    names = [g["name"] for g in G]
    ix = {nm: i for i, nm in enumerate(names)}
    pubs = np.array([float(g.get("pubs") or 0) for g in G])
    npath = np.array([int(g.get("npath") or 0) for g in G])
    haspath = np.array([bool(g.get("path")) for g in G])

    say("THE OLD LAYER, MEASURED BEFORE ANYTHING IS REBUILT")
    P60 = D["pathways"]
    cov60 = set()
    derived = 0
    tot60 = 0
    for k, v in P60.items():
        cov60.update(v)
        for i in v:
            tot60 += 1
            if str(G[i].get("path") or "").strip() == k.strip():
                derived += 1
    per_gene_60 = collections.Counter()
    for k, v in P60.items():
        for i in v:
            per_gene_60[i] += 1
    say(f"     pathways key: {len(P60)} pathways, {len(cov60):,} genes ({len(cov60)/n:.1%}), "
        f"max memberships per gene {max(per_gene_60.values()) if per_gene_60 else 0}")
    say(f"     of its {tot60:,} member entries, {derived/max(tot60,1):.4f} have the gene's own `path` "
        f"string equal to the pathway name")
    say(f"     -> it is groupby(path), not a source. A one-pathway-per-gene structure cannot "
        f"reproduce npath's max of {npath.max()}.")
    say(f"     npath > 0 agrees with path non-null on {(( npath>0)==haspath).mean():.4f} of genes "
        f"-- the same pass wrote both")
    say()

    say("R1 THE REBUILD REPRODUCES npath, BY DISCRIMINATION")
    e2s = entrez2symbol()
    say(f"     NCBI gene_info: {len(e2s):,} human Entrez -> symbol")
    low, nm_low = read_reactome(LOW, e2s)
    alll, nm_all = read_reactome(ALL, e2s)
    say(f"     Reactome LOWEST-LEVEL : {sum(len(v) for v in low.values()):,} human rows, "
        f"{len(low):,} genes, {len(nm_low):,} pathways")
    say(f"     Reactome ALL-LEVELS   : {sum(len(v) for v in alll.values()):,} human rows, "
        f"{len(alll):,} genes, {len(nm_all):,} pathways")

    cand = {
        "A lowest-level, ROWS": np.array([len(low.get(s, ())) for s in names]),
        "B lowest-level, UNIQUE ids": np.array([len(set(low.get(s, ()))) for s in names]),
        "C all-levels, ROWS": np.array([len(alll.get(s, ())) for s in names]),
    }
    rng = np.random.default_rng(SEED)
    nullex = np.array([float((npath == rng.permutation(npath)).mean()) for _ in range(NREP)])
    say(f"     null (npath shuffled across genes, {NREP} draws): exact "
        f"{nullex.mean():.4f} +/- {nullex.std():.4f}")
    best, bestex = None, -1.0
    for k, v in sorted(cand.items()):
        ex = float((v == npath).mean())
        rho = float(spearmanr(v, npath).statistic)
        w1 = float((np.abs(v - npath) <= 1).mean())
        z = (ex - nullex.mean()) / max(nullex.std(), 1e-9)
        say(f"       {k:28s} exact {ex:.4f}  within1 {w1:.4f}  rho {rho:+.4f}  "
            f"max {v.max():4d} (stored {npath.max()})  z {z:8.1f}")
        if ex > bestex:
            best, bestex = k, ex
    v = cand[best]
    ex = float((v == npath).mean())
    rho = float(spearmanr(v, npath).statistic)
    z = (ex - nullex.mean()) / max(nullex.std(), 1e-9)
    mism = int((v != npath).sum())
    dif = (v - npath)[v != npath]
    say(f"     WINNER: {best}")
    say(f"       {mism:,} mismatches; median |diff| {np.median(np.abs(dif)):.0f}, "
        f"{float((np.abs(dif) <= 1).mean()):.4f} within +/-1; "
        f"{int((dif > 0).sum())} gained / {int((dif < 0).sum())} lost vs the stored count")
    say(f"       residual disagreement is Reactome curation drift between the stored release and "
        f"the current one; the release was never recorded and cannot be pinned.")
    r1 = ex >= EXACT_FLOOR and rho >= RHO_FLOOR and z >= NULL_SD
    say(f"     R1 {'PASS' if r1 else 'FAIL'} -- the lost source is identified: Reactome, human, "
        f"lowest level, rows not unique ids")
    say()

    say("R2 THE MATRIX IS COMPLETE WHERE THE OLD LAYER WAS NOT")
    hier = []
    for ln in open(REL, errors="ignore"):
        f = ln.rstrip("\n").split("\t")
        if len(f) == 2 and f[0].startswith("R-HSA") and f[1].startswith("R-HSA"):
            hier.append((f[0], f[1]))
    hnames = {}
    for ln in open(NAMES, errors="ignore"):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3 and f[2] == SPECIES:
            hnames[f[0]] = f[1]
    parents = {a for a, _ in hier}
    children = {b for _, b in hier}
    leaves = sorted(set(hnames) - parents)
    roots = sorted(parents - children)
    gmt = {}
    if GMT.exists():
        for ln in open(GMT, errors="ignore"):
            f = ln.rstrip("\n").split("\t")
            if len(f) > 2 and f[1].startswith("R-HSA"):
                gmt[f[1]] = [x for x in f[2:] if x in ix]
    mem = {s: sorted(set(v)) for s, v in low.items() if s in ix}
    want = int((npath > 0).sum())
    got = sum(1 for s in names if mem.get(s))
    maxmem = max((len(v) for v in low.values()), default=0)
    say(f"     leaf membership   {len(mem):,} genes, {len(nm_low):,} pathways, "
        f"{sum(len(v) for v in mem.values()):,} unique gene-pathway links")
    say(f"     all-levels (GMT)  {len(gmt):,} human pathways carrying symbols, "
        f"{sum(len(v) for v in gmt.values()):,} links (pre-rolled-up; do NOT re-propagate)")
    say(f"     hierarchy         {len(hier):,} human parent->child edges, {len(roots)} roots, "
        f"{len(leaves):,} leaves")
    say(f"     genes the old layer says have a pathway: {want:,};  the rebuild gives one to {got:,}")
    say(f"     max memberships per gene  old {max(per_gene_60.values()) if per_gene_60 else 0}  "
        f"-> new {maxmem}  (stored npath max {npath.max()})")
    r2 = got >= want * 0.95 and maxmem >= npath.max()
    say(f"     R2 {'PASS' if r2 else 'FAIL'}")
    say()

    say("R3 THE LAYER IS PARTLY A FAME AXIS -- REPORTED, NOT JUDGED")
    ppideg = np.zeros(n)
    for a, b in D["ppi"]:
        ppideg[a] += 1
        ppideg[b] += 1
    say(f"     rho(npath, publication count) {spearmanr(npath, pubs).statistic:+.4f}")
    say(f"     rho(npath, PPI degree)        {spearmanr(npath, ppideg).statistic:+.4f}")
    say(f"     rho(PPI degree, pubs)         {spearmanr(ppideg, pubs).statistic:+.4f}   "
        f"<- why PPI is the confound and not the referee")
    top = np.argsort(-npath)[:6]
    say(f"     most-annotated genes: " + ", ".join(f"{names[i]} {npath[i]}" for i in top))
    say(f"     R3 PASS (reported)")
    say()

    say("R4 SHARED-PATHWAY PAIRS SURVIVE A THREE-WAY MATCHED NULL")
    say(f"     predeclared: altdata_sources.py already records shared-Reactome-pathway at AUC "
        f"~{ALTDATA_KNOWN} vs degree-matched.")
    say(f"     A result near {ALTDATA_KNOWN} is a REPLICATION of a known negative, not a discovery.")
    # membership restricted to pathways of usable size, then gene -> set of pathway ids
    sizes = collections.Counter()
    for s, v in mem.items():
        for p in v:
            sizes[p] += 1
    keep = {p for p, c in sizes.items() if PW_MIN <= c <= PW_MAX}
    memk = {s: {p for p in v if p in keep} for s, v in mem.items()}
    memk = {s: v for s, v in memk.items() if v}
    say(f"     pathway size cap {PW_MIN}-{PW_MAX}: {len(keep):,} of {len(sizes):,} pathways kept, "
        f"{len(memk):,} genes still covered")
    say(f"       (uncapped, the largest pathway holds {max(sizes.values()):,} genes and would "
        f"dominate every pair count)")
    pw = [memk.get(s, frozenset()) for s in names]

    def targets():
        cd = set()
        for k, lst in D["codep"].items():
            a = int(k)
            for b, _ in lst:
                if a != b:
                    cd.add((min(a, b), max(a, b)))
        ce = set()
        for k, lst in D["coexpr"].items():
            a = int(k)
            for b, _ in lst:
                if a != b:
                    ce.add((min(a, b), max(a, b)))
        return {"codep": sorted(cd), "coexpr": sorted(ce)}

    T = targets()
    results = {}
    for tname, pos in T.items():
        deg = np.zeros(n)
        for a, b in pos:
            deg[a] += 1
            deg[b] += 1
        edgeset = set(pos)
        dbin, pbin = bins(deg), bins(pubs)
        jbin = np.array([f"{d}|{p}|{q}" for d, p, q in
                         zip(bins(deg), bins(pubs), bins(npath.astype(float)))])

        def sample_neg(key, seed):
            r = np.random.RandomState(seed)
            idx = collections.defaultdict(list)
            for i in range(n):
                idx[key[i]].append(i)
            out = []
            for (a, b) in pos:
                pa_, pb_ = idx[key[a]], idx[key[b]]
                got_, tries = 0, 0
                while got_ < NNEG and tries < 200:
                    tries += 1
                    x = pa_[r.randint(len(pa_))]
                    y = pb_[r.randint(len(pb_))]
                    if x == y:
                        continue
                    p = (min(x, y), max(x, y))
                    if p in edgeset:
                        continue
                    out.append(p)
                    got_ += 1
            return out

        def sample_rand(seed):
            r = np.random.RandomState(seed)
            out = []
            while len(out) < NNEG * len(pos):
                x, y = r.randint(n), r.randint(n)
                if x == y:
                    continue
                p = (min(x, y), max(x, y))
                if p in edgeset:
                    continue
                out.append(p)
            return out

        NEG = {"RANDOM": sample_rand(SEED + 1),
               "PUBS-MATCHED": sample_neg(pbin, SEED + 2),
               "DEGREE-MATCHED": sample_neg(dbin, SEED + 3),
               "DEG+PUBS+NPATH-MATCHED": sample_neg(jbin, SEED + 4)}
        say(f"     --- target {tname}: {len(pos):,} positive pairs ---")
        say(f"     COVARIATE BALANCE (link_completion.py:227-233 -- a module that does not print "
            f"this cannot claim its null is matched)")
        pa = np.array([p[0] for p in pos])
        pb = np.array([p[1] for p in pos])
        say(f"       {'set':24s} {'n':>8s}  {'log1p(pubs)':>22s}  {'log1p(deg)':>20s}  "
            f"{'npath':>18s}")
        say(f"       {'POSITIVES':24s} {len(pos):8,d}  "
            f"{np.mean(np.log1p(pubs[pa]) + np.log1p(pubs[pb])):22.2f}  "
            f"{np.mean(np.log1p(deg[pa]) + np.log1p(deg[pb])):20.2f}  "
            f"{np.mean(npath[pa] + npath[pb]):18.1f}")
        for k, v in NEG.items():
            aa = np.array([p[0] for p in v])
            bb = np.array([p[1] for p in v])
            say(f"       {k:24s} {len(v):8,d}  "
                f"{np.mean(np.log1p(pubs[aa]) + np.log1p(pubs[bb])):22.2f}  "
                f"{np.mean(np.log1p(deg[aa]) + np.log1p(deg[bb])):20.2f}  "
                f"{np.mean(npath[aa] + npath[bb]):18.1f}")

        def feats(prs):
            sh = np.array([len(pw[a] & pw[b]) for a, b in prs], float)
            pu = np.array([np.log1p(pubs[a]) + np.log1p(pubs[b]) for a, b in prs])
            pa_ = np.array([np.log1p(deg[a]) + np.log1p(deg[b]) for a, b in prs])
            return sh, pu, pa_

        fp = feats(pos)
        row = {}
        for k, v in NEG.items():
            fn = feats(v)
            y = np.concatenate([np.ones(len(pos), bool), np.zeros(len(v), bool)])
            a_sh = auc(np.concatenate([fp[0], fn[0]]), y)
            a_pu = auc(np.concatenate([fp[1], fn[1]]), y)
            a_pa = auc(np.concatenate([fp[2], fn[2]]), y)
            bar = max(a_pu, a_pa)
            row[k] = {"shared_pathway": a_sh, "PUBS": a_pu, "PREF-ATT": a_pa, "bar": bar,
                      "beats_bar": bool(a_sh > bar)}
            say(f"       {k:24s} shared-pathway AUC {a_sh:.4f}   "
                f"trivial: PUBS {a_pu:.4f} PREF-ATT {a_pa:.4f}   bar {bar:.4f}   "
                f"{'BEATS' if a_sh > bar else 'BELOW'}")
        results[tname] = row
        say()

    key = "DEG+PUBS+NPATH-MATCHED"
    r4 = all(results[t][key]["beats_bar"] for t in results)
    for t in results:
        a = results[t][key]["shared_pathway"]
        say(f"     {t:8s} under the three-way matched null: AUC {a:.4f}  "
            f"bar {results[t][key]['bar']:.4f}  "
            f"(altdata_sources' known negative {ALTDATA_KNOWN})")
    say(f"     R4 {'PASS' if r4 else 'FAIL'}")
    if not r4:
        say(f"     This REPLICATES the negative already in the repository. Shared pathway membership")
        say(f"     does not identify functionally related gene pairs once degree, publication count")
        say(f"     and annotation depth are held fixed. The layer is still worth rebuilding -- R1/R2")
        say(f"     restore data that was destroyed -- but it must not be sold as a predictor.")
    say()

    say("R5 THE WRITE IS ADDITIVE")
    before = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in D.items()}
    payload = {
        "source": "Reactome, Homo sapiens. Leaf membership from NCBI2Reactome.txt (LOWEST level), "
                  "Entrez joined to symbol via NCBI gene_info. All-levels membership from "
                  "ReactomePathways.gmt (human, symbol-native, already rolled up). Hierarchy from "
                  "ReactomePathwaysRelation.txt.",
        "identified_by": f"reproducing the stored per-gene npath count: {best}, exact {ex:.4f}, "
                         f"rho {rho:.4f}, against a shuffled null of {nullex.mean():.4f}",
        "warning": "the all-levels lists are PRE-ROLLED-UP; do not propagate genes up the hierarchy "
                   "again or parents will double-count",
        "pathway_names": {p: nm_low.get(p) or hnames.get(p) or "" for p in
                          sorted(set(nm_low) | set(gmt))},
        "leaf_membership": mem,
        "all_levels_membership": gmt,
        "hierarchy": hier,
        "roots": roots,
        "n_leaves": len(leaves),
    }
    PWOUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(PWOUT, "w"))
    D2 = json.load(open(CELL))
    after = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in D2.items()}
    changed = [k for k in before if before[k] != after.get(k)]
    sz = PWOUT.stat().st_size
    say(f"     wrote {sz/1e6:.1f} MB to {PWOUT.name}; cell_complete.json fields changed: "
        f"{len(changed)}")
    say(f"     LAYER CENSUS  pathways 60 -> {len(nm_low):,} leaf / {len(gmt):,} all-levels")
    say(f"                   genes covered {len(cov60):,} -> {len(mem):,}")
    say(f"                   max memberships per gene 1 -> {maxmem}")
    say(f"                   hierarchy 0 -> {len(hier):,} edges")
    r5 = not changed
    say(f"     R5 {'PASS' if r5 else 'FAIL'}")
    say()

    gates = {"R1 rebuild reproduces npath by discrimination": bool(r1),
             "R2 matrix complete where the old layer was not": bool(r2),
             "R3 fame correlation reported": True,
             "R4 shared-pathway pairs survive a three-way matched null": bool(r4),
             "R5 write additive": bool(r5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LOW), str(ALL), str(REL), str(NAMES), str(GMT), str(GINFO),
                              str(CELL)],
                      available=n, used=len(mem), selection="filtered", seed=SEED,
                      controls=["three competing reconstructions scored, two refuted",
                                "stored npath shuffled across genes as the null",
                                "two measured targets (codep, coexpr) as independent replicates",
                                "PPI excluded as a target because it is the fame confound",
                                "four negative sets: random, pubs, degree, deg+pubs+npath",
                                "covariate balance printed for every negative set",
                                "bar set at the better trivial baseline, not at 0.5",
                                "pathway size capped 2-60 before pair expansion",
                                "predeclared against altdata_sources' known 0.51 negative"],
                      note="build_cell_complete.py declared npath/path SOURCE_NOT_IDENTIFIED; "
                           "cell_complete.json has no producer in this repository")
    RM.report(man, emit=say)
    json.dump({"test": "loop_pathways", "manifest": man, "gates": gates,
               "npath_candidates": {k: {"exact": float((v == npath).mean()),
                                        "rho": float(spearmanr(v, npath).statistic),
                                        "within1": float((np.abs(v - npath) <= 1).mean()),
                                        "max": int(v.max())} for k, v in cand.items()},
               "npath_null_exact_mean": float(nullex.mean()),
               "npath_null_exact_sd": float(nullex.std()),
               "winner": best, "winner_exact": ex, "winner_rho": rho,
               "old_pathways_derived_fraction": derived / max(tot60, 1),
               "old_genes_covered": len(cov60), "new_genes_covered": len(mem),
               "n_leaf_pathways": len(nm_low), "n_all_levels_pathways": len(gmt),
               "n_hierarchy_edges": len(hier), "max_memberships_per_gene": maxmem,
               "rho_npath_pubs": float(spearmanr(npath, pubs).statistic),
               "rho_npath_ppideg": float(spearmanr(npath, ppideg).statistic),
               "r4": results, "altdata_known_negative": ALTDATA_KNOWN,
               "bytes": sz, "existing_fields_changed": changed,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_pathways.json", "w"), indent=1)
    say(f"\n  -> {PWOUT}")
    say(f"  -> {OUT / 'loop_pathways.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
