"""LOOP 49 -- CLOSE THE DECLARED GAP, AND FIND OUT WHETHER LOOP 48'S MATCHES MEANT ANYTHING.

Loop 48 gave cell_complete.json a producer for 7 fields covering 17% of all module-reads, and left
30% in a category called "declared with a URL, not fetched here": ppi (210 reads), dep_frac (89),
ess (69), pubs (42). A URL is not a source. Recording one and moving on is exactly the move this
project keeps catching itself making -- it converts an unverified column into a documented-looking
one at zero cost, and the documentation is never wrong because it is never checked.

So: fetch them, rebuild them, check them. Two of the four turn out not to need a download at all,
which is its own small finding -- depmap_vecs.npz has been sitting in outputs/orphan the whole time
with 18,443 genes x 1,150 cell lines of gene-effect data, z-scored with mu and sd kept, so the raw
matrix reconstructs as Z*sd+mu and dep_frac is a threshold away.

AND THE PART THAT MATTERS MORE. Loop 48 had no negative control. It reported that `tf` agrees at
0.9383 and called that a verified provenance claim, but never asked what a WRONG rebuild would have
scored. For a column that is 90% zeros, a rebuild that assigns every gene a random other gene's
value still agrees with the file 82% of the time, and 0.9383 stops looking like evidence. That
question is asked here, for the new fields AND retroactively for loop 48's, and any field whose
real agreement does not clear its own shuffled twin is withdrawn.

WHAT EACH COLUMN IS CLAIMED TO BE, declared before the numbers:

    ppi       count of DISTINCT human interaction partners in BioGRID, PHYSICAL evidence only,
              self-loops excluded. Physical because the field is called protein interactions;
              the genetic-inclusive variant is reported as a diagnostic and is not the claim.
    dep_frac  fraction of DepMap cell lines where the gene-effect score is below -0.5, the
              conventional dependency threshold.
    ess       1 where dep_frac > 0.5. This rule was READ OFF the file rather than remembered --
              the minimum dep_frac among ess=1 genes and the maximum among ess=0 genes are both
              exactly 0.5, which is a hard deterministic boundary and not a fitted threshold.
    pubs      count of distinct PubMed ids linked to the gene in NCBI gene2pubmed.

PREDECLARED, before any number:

    D1 THE DECLARED SOURCES ARRIVE AND COVER THE GENES   each of the four has a source present that
                parses to a gene-keyed table covering at least 80% of the file's 16,492 genes. A
                source covering a tenth of the genes cannot check the column, and a check run on the
                overlap alone would report agreement on the easy genes only.
    D2 EACH REBUILT COLUMN MATCHES THE SURVIVING COPY   at loop 48's thresholds, unchanged:
                exact >= 0.90 OR corr >= 0.90. Same gate as the previous loop so the two are
                comparable and so it cannot be quietly softened for the harder fields.
    D3 THE ess RULE SURVIVES WITHOUT CIRCULARITY   ess is rebuilt from dep_frac RECOMPUTED FROM
                DepMap, never from the file's own dep_frac column. Thresholding the file's column
                and comparing to the file's other column tests nothing -- the file would be
                predicting itself, and it would score near 1.0 whether or not DepMap is the source.
    D4 SHUFFLED REBUILDS FAIL   THIS IS THE REAL GATE OF THIS LOOP. For every field, new and old,
                the rebuilt column is compared against a version of itself with the gene labels
                permuted. The real agreement must beat the 95th percentile of 200 shuffles. A field
                that does not clear its own shuffle is not evidence of provenance -- it is evidence
                that the column has a lopsided marginal distribution -- and it is WITHDRAWN from the
                verified set no matter how high its raw agreement looked.
    D5 VERIFIED COVERAGE PASSES HALF   after withdrawals, the share of module-reads backed by a
                fetched-and-checked source is at least 45%, up from 17%. This one is a consequence
                of the others and is gated anyway, because a coverage number that cannot fail is
                not worth printing.

-> outputs/loop_rebuild.json
"""
import collections
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
DEPMAP = ROOT / "outputs" / "orphan" / "depmap_vecs.npz"

GATE_EXACT = 0.90        # identical to loop 48, deliberately
GATE_CORR = 0.90
COVER_MIN = 0.80         # D1
N_SHUFFLE = 200          # D4
DEP_THRESHOLD = -0.5     # the conventional DepMap dependency cutoff
ESS_THRESHOLD = 0.5      # read off the file's own hard boundary, not fitted

log = []


def say(s=""):
    print(s)
    log.append(s)


# ---- sources -------------------------------------------------------------------------------------
def load_biogrid():
    """Distinct interaction partners per symbol, split by evidence type."""
    phys = collections.defaultdict(set)
    alle = collections.defaultdict(set)
    with gzip.open(SC / "biogrid_hs_edges.tsv.gz", "rt", errors="ignore") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            a, b, kind = p[0], p[1], p[2]
            if a == b or a == "-" or b == "-":
                continue
            alle[a].add(b)
            alle[b].add(a)
            if kind == "physical":
                phys[a].add(b)
                phys[b].add(a)
    return ({k: len(v) for k, v in phys.items()}, {k: len(v) for k, v in alle.items()})


def load_pubs():
    """Distinct PubMed ids per symbol, via the NCBI Entrez -> symbol map."""
    sym = {}
    with gzip.open(SC / "hs_gene_info.gz", "rt", errors="ignore") as f:
        f.readline()
        for ln in f:
            p = ln.split("\t")
            if len(p) > 2:
                sym[p[1]] = p[2]
    per = collections.defaultdict(set)
    with gzip.open(SC / "gene2pubmed_9606.tsv.gz", "rt", errors="ignore") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            s = sym.get(p[1])
            if s:
                per[s].add(p[2])
    return {k: len(v) for k, v in per.items()}


def load_depmap():
    """The gene-effect matrix, reconstructed from the stored z-scores."""
    z = np.load(DEPMAP, allow_pickle=True)
    eff = z["Z"] * z["sd"][:, None] + z["mu"][:, None]
    return {s: i for i, s in enumerate(z["syms"])}, eff


# ---- scoring -------------------------------------------------------------------------------------
def agree(orig, new, kind):
    """Exact agreement and, for numeric fields, correlation -- loop 48's two numbers."""
    pairs = [(o, n) for o, n in zip(orig, new) if o is not None and n is not None]
    if not pairs:
        return 0, float("nan"), float("nan")
    exact = float(np.mean([a == b for a, b in pairs]))
    corr = float("nan")
    if kind == "num":
        oa = np.array([p[0] for p in pairs], float)
        na = np.array([p[1] for p in pairs], float)
        m = np.isfinite(oa) & np.isfinite(na)
        if m.sum() > 10 and np.std(oa[m]) > 0 and np.std(na[m]) > 0:
            corr = float(np.corrcoef(oa[m], na[m])[0, 1])
    return len(pairs), exact, corr


def shuffle_null(orig, new, kind, rng):
    """What the same rebuild scores with the gene labels permuted.

    This is the control loop 48 did not run. A rebuilt column carries two things: the right VALUE
    for each gene, and the right SHAPE of the value distribution. Only the first is provenance. The
    permutation destroys the first and keeps the second exactly, so whatever the shuffle still
    scores is what the shape alone was worth.
    """
    idx = [i for i, (o, n) in enumerate(zip(orig, new)) if o is not None and n is not None]
    o = [orig[i] for i in idx]
    n = np.array([new[i] for i in idx], dtype=object)
    ex, co = [], []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(len(n))
        s = list(n[perm])
        _, e, c = agree(o, s, kind)
        ex.append(e)
        co.append(abs(c) if np.isfinite(c) else 0.0)
    return float(np.percentile(ex, 95)), float(np.percentile(co, 95))


def main():
    t0 = time.time()
    rng = np.random.default_rng(1904)
    say("=" * 100)
    say("  LOOP 49 -- the 30% that had a URL and no download")
    say("=" * 100)

    genes = json.load(open(CELL))["genes"]
    nm = [g["name"] for g in genes]
    say(f"\n  cell_complete.json: {len(genes):,} genes")

    # ---- D1 the sources arrive ------------------------------------------------------------------
    say(f"\n  D1 THE DECLARED SOURCES ARRIVE AND COVER THE GENES")
    ppi_phys, ppi_all = load_biogrid()
    pubs = load_pubs()
    dep_idx, dep_eff = load_depmap()
    dep_frac = {s: float(np.mean(dep_eff[i] < DEP_THRESHOLD)) for s, i in dep_idx.items()}
    cover = {"ppi": np.mean([s in ppi_phys for s in nm]),
             "pubs": np.mean([s in pubs for s in nm]),
             "dep_frac": np.mean([s in dep_frac for s in nm]),
             "ess": np.mean([s in dep_frac for s in nm])}
    for k, v in cover.items():
        say(f"     {k:<10}{v:>8.1%} of genes covered   "
            f"{'OK' if v >= COVER_MIN else 'TOO THIN TO CHECK'}")
    say(f"     BioGRID {sum(ppi_phys.values())//2:,} physical edges over {len(ppi_phys):,} symbols · "
        f"gene2pubmed {len(pubs):,} symbols · DepMap {len(dep_idx):,} x {dep_eff.shape[1]:,} lines")
    d1 = bool(all(v >= COVER_MIN for v in cover.values()))
    say(f"     D1 {'PASS' if d1 else 'FAIL'}  (each source covers >= {COVER_MIN:.0%} of the genes)")

    # ---- D2/D3 rebuild and check ----------------------------------------------------------------
    say(f"\n  D2 EACH REBUILT COLUMN MATCHES THE SURVIVING COPY")
    say(f"     {'field':<12}{'n':>8}{'exact':>9}{'corr':>9}   verdict")
    checks = {}
    built = {}

    def record(field, orig, new, kind):
        n, exact, corr = agree(orig, new, kind)
        ok = (exact >= GATE_EXACT) or (np.isfinite(corr) and corr >= GATE_CORR)
        v = ("MATCH" if exact >= GATE_EXACT else
             "SOURCE RIGHT, VERSION DRIFT" if np.isfinite(corr) and corr >= GATE_CORR else
             "MISMATCH")
        checks[field] = {"n": n, "exact": exact, "corr": corr, "pass": bool(ok), "kind": kind}
        built[field] = (orig, new, kind)
        say(f"     {field:<12}{n:>8,}{exact:>9.4f}"
            f"{(f'{corr:.4f}' if np.isfinite(corr) else '-'):>9}   {v}")

    record("ppi", [g.get("ppi") for g in genes], [ppi_phys.get(s) for s in nm], "num")
    record("pubs", [g.get("pubs") for g in genes], [pubs.get(s) for s in nm], "num")
    record("dep_frac", [g.get("dep_frac") for g in genes], [dep_frac.get(s) for s in nm], "num")
    # D3: ess from RECOMPUTED dep_frac, never from the file's column.
    ess_new = [None if s not in dep_frac else int(dep_frac[s] > ESS_THRESHOLD) for s in nm]
    record("ess", [g.get("ess") for g in genes], ess_new, "cat")
    d2 = bool(all(v["pass"] for v in checks.values()))
    say(f"     D2 {'PASS' if d2 else 'FAIL'}  (each field: exact >= {GATE_EXACT} OR corr >= "
        f"{GATE_CORR})")

    # the circularity the D3 construction avoids, measured so the size of the trap is on record
    ess_circ = [None if g.get("dep_frac") is None else int(g["dep_frac"] > ESS_THRESHOLD)
                for g in genes]
    _, circ_exact, _ = agree([g.get("ess") for g in genes], ess_circ, "cat")
    say(f"\n  D3 THE ess RULE SURVIVES WITHOUT CIRCULARITY")
    say(f"     from the file's own dep_frac column   {circ_exact:.4f}   <- proves nothing; the file "
        f"is predicting itself")
    say(f"     from dep_frac RECOMPUTED from DepMap  {checks['ess']['exact']:.4f}   <- this is the "
        f"one that can fail, and it is the gate")
    d3 = bool(checks["ess"]["pass"])
    say(f"     the gap between them, {circ_exact - checks['ess']['exact']:+.4f}, is the size of the "
        f"trap avoided")
    say(f"     D3 {'PASS' if d3 else 'FAIL'}")

    # ---- D4 the shuffle control, new fields AND loop 48's ---------------------------------------
    say(f"\n  D4 SHUFFLED REBUILDS FAIL")
    say(f"     the same rebuild with gene labels permuted, {N_SHUFFLE} times. What it still scores is")
    say(f"     what the shape of the distribution was worth, with the gene-to-value map destroyed.")
    prev = json.load(open(OUT / "build_cell_complete.json"))
    say(f"     {'field':<12}{'real':>9}{'shuffled':>10}{'margin':>9}   verdict")
    shuffles = {}
    for field, (orig, new, kind) in built.items():
        se, sc = shuffle_null(orig, new, kind, rng)
        c = checks[field]
        real = c["exact"] if c["exact"] >= GATE_EXACT else (abs(c["corr"])
                                                            if np.isfinite(c["corr"]) else c["exact"])
        null = se if c["exact"] >= GATE_EXACT else (sc if np.isfinite(c["corr"]) else se)
        beats = bool(real > null)
        shuffles[field] = {"real": real, "null95": null, "beats": beats}
        say(f"     {field:<12}{real:>9.4f}{null:>10.4f}{real - null:>+9.4f}   "
            f"{'clears its shuffle' if beats else 'DOES NOT CLEAR -- WITHDRAWN'}")

    # RETROACTIVE. Loop 48 stored its agreement numbers but not its rebuilt columns, so the null is
    # built by permuting the FILE's column instead of the rebuilt one. That is legitimate and if
    # anything generous to the null: the two have nearly identical marginals wherever they agree,
    # which is exactly the case being tested, and the permutation null depends on the marginal alone.
    say(f"\n     RETROACTIVE -- loop 48's fields, same question, because they were never asked it")
    for field, c in prev["checks"].items():
        col = [g.get(field) for g in genes]
        vals = [v for v in col if v is not None]
        if not vals:
            continue
        kind = "num" if field in ("gene_start", "gene_end", "ens_tss", "loeuf") else "cat"
        se, sc = shuffle_null(col, list(col), kind, rng)
        real = c["exact"] if c["exact"] >= GATE_EXACT else (abs(c["corr"])
                                                            if c["corr"] is not None
                                                            and np.isfinite(c["corr"]) else c["exact"])
        null = se if c["exact"] >= GATE_EXACT else (sc if c["corr"] is not None
                                                    and np.isfinite(c["corr"]) else se)
        beats = bool(real > null)
        shuffles[f"48:{field}"] = {"real": real, "null95": null, "beats": beats}
        say(f"     {field:<12}{real:>9.4f}{null:>10.4f}{real - null:>+9.4f}   "
            f"{'clears its shuffle' if beats else 'DOES NOT CLEAR -- WITHDRAWN'}")
    # D4 is asked of the fields that are CLAIMED. A field that already failed its match gate is not
    # a provenance claim, so requiring it to beat a shuffle too would fail this loop for a column
    # nobody is asserting -- and would make D4 impossible to interpret.
    claimed = {f for f, c in checks.items() if c["pass"]} | \
              {f"48:{f}" for f, c in prev["checks"].items() if c["pass"]}
    d4 = bool(all(v["beats"] for f, v in shuffles.items() if f in claimed))
    say(f"     D4 {'PASS' if d4 else 'FAIL'}  (every CLAIMED field, new and old, beats its own "
        f"shuffle; {len(claimed)} claimed of {len(shuffles)} tested)")

    # ---- D5 coverage after withdrawals -----------------------------------------------------------
    usage = prev["usage"]
    total_reads = prev["total_reads"]
    keep_new = {f for f, c in checks.items() if c["pass"] and shuffles[f]["beats"]}
    keep_old = {f for f, c in prev["checks"].items()
                if c["pass"] and shuffles.get(f"48:{f}", {}).get("beats", False)}
    withdrawn = ({f for f, c in checks.items() if c["pass"]} - keep_new) | \
                ({f for f, c in prev["checks"].items() if c["pass"]} - keep_old)
    r_new = sum(usage.get(f, 0) for f in keep_new)
    r_old = sum(usage.get(f, 0) for f in keep_old)
    frac = (r_new + r_old) / total_reads
    say(f"\n  D5 VERIFIED COVERAGE PASSES HALF")
    say(f"     loop 48 verified          {prev['reads_verified']:>6,}  "
        f"{prev['reads_verified']/total_reads:>6.1%}")
    say(f"     surviving from loop 48    {r_old:>6,}  {r_old/total_reads:>6.1%}   "
        f"({', '.join(sorted(keep_old))})")
    say(f"     added here                {r_new:>6,}  {r_new/total_reads:>6.1%}   "
        f"({', '.join(sorted(keep_new))})")
    say(f"     TOTAL VERIFIED            {r_new + r_old:>6,}  {frac:>6.1%}")
    if withdrawn:
        say(f"     WITHDRAWN for failing the shuffle: {', '.join(sorted(withdrawn))}")
    d5 = bool(frac >= 0.45)
    say(f"     D5 {'PASS' if d5 else 'FAIL'}  (>= 45%)")

    # diagnostic, not a gate: the evidence-type variant of ppi
    _, pe, pc = agree([g.get("ppi") for g in genes], [ppi_all.get(s) for s in nm], "num")
    say(f"\n     diagnostic  ppi over ALL evidence types (physical + genetic): exact {pe:.4f}, "
        f"corr {pc:.4f}")
    say(f"                 the declared claim is physical-only at exact "
        f"{checks['ppi']['exact']:.4f}, corr {checks['ppi']['corr']:.4f}; the variant is reported "
        f"and NOT adopted")

    say("\n" + "=" * 100)
    gates = {"D1 sources arrive and cover": d1, "D2 rebuilt columns match": d2,
             "D3 ess rule survives non-circularly": d3, "D4 shuffled rebuilds fail": d4,
             "D5 coverage passes half": d5}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    say(f"  THE 30% THAT HAD A URL NOW HAS A CHECK. Verified coverage of cell_complete.json's")
    say(f"  {total_reads:,} module-reads moves {prev['reads_verified']/total_reads:.0%} -> {frac:.0%}.")
    say(f"  Two of the four needed no download at all: DepMap has been in outputs/orphan the whole")
    say(f"  time as z-scores with mu and sd kept, so the gene-effect matrix reconstructs exactly.")
    say(f"  And loop 48's fields have now been asked the question they were never asked -- whether")
    say(f"  a wrong rebuild would have scored as well. {'Every field cleared its own shuffle.' if d4 else 'Not all of them did, and the ones that did not are withdrawn.'}")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "biogrid_hs_edges.tsv.gz"), str(SC / "gene2pubmed_9606.tsv.gz"),
                              str(SC / "hs_gene_info.gz"), str(DEPMAP), str(CELL)],
                      available=total_reads, used=r_new + r_old, selection="filtered", seed=1904,
                      controls=[f"{N_SHUFFLE} gene-label permutations per field, real must beat the "
                                f"95th percentile",
                                "loop 48's fields re-tested against the same control retroactively",
                                "ess rebuilt from recomputed dep_frac, never from the file's own "
                                "column, with the circular score reported for contrast",
                                "the ppi evidence-type variant reported and not adopted"],
                      note="the same agreement thresholds as loop 48, unchanged, so the two loops "
                           "are comparable")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_rebuild", "manifest": man, "gates": gates, "checks": checks,
               "coverage": cover, "shuffles": shuffles, "ess_circular_exact": circ_exact,
               "ppi_all_evidence": {"exact": pe, "corr": pc},
               "reads_verified": r_new + r_old, "total_reads": total_reads,
               "verified_fields": sorted(keep_new | keep_old), "withdrawn": sorted(withdrawn),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_rebuild.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_rebuild.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
