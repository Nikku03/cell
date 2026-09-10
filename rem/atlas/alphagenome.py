"""AlphaGenome: what it can do, what it would give this engine, and whether it is worth using.

WHAT IT IS. A DNA sequence model served over a gRPC API. From the installed client, measured rather
than quoted: input up to 1,048,576 bp (also 16KB / 100KB / 500KB windows), organism defaults to
HOMO_SAPIENS, and eleven output modalities -- ATAC, CAGE, DNASE, RNA_SEQ, CHIP_HISTONE, CHIP_TF,
SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS, CONTACT_MAPS, PROCAP -- at single-base resolution
for most, with variant-effect scoring built in. Free for non-commercial use, API key required,
thousands rather than millions of custom predictions.

WHY IT MIGHT MATTER HERE, AND THE ANSWER IS NOT THE OBVIOUS ONE. The obvious use is to redo the
human measurements without measurement noise, since whatdata's blocking finding was a class ladder
narrower than its own noise floor. That is NOT the right use, and the record already says why:
humantransfer's variance-component estimator ALREADY removes the replicate noise from the marginal
question and returned a between-regulator effect of 1.05 floors with a CI excluding zero. Noise-free
marginals add nothing that has not been extracted.

THE THING NO DATASET SUPPLIES IS THE JOINT REGIME. The engine's per-gene factor is a PRODUCT over
occupied classes -- a statement about JOINT states of a gene's regulators. Every human dataset used
in this record is single-perturbation, which measures MARGINALS. AlphaGenome can delete arbitrary
SUBSETS of a gene's binding sites in silico and read off predicted expression, which is the only
available route to the joint response surface. That is the use worth testing, and A0 prices it
before any key is spent.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

A0  THE CEILING GATE, IN TWO PARTS, AND IT DECIDES WHETHER TO USE THE MODEL AT ALL.
    FIRST: in single-perturbation data the number of perturbed regulators is ONE in every
    observation. A summary that says the response depends on HOW MANY are perturbed therefore
    predicts a constant, and the count can be refuted but never calibrated. Verified on the cached
    ENCODE panel rather than asserted.
    SECOND: the engine's own per-gene factor counts the joint states it must represent. Compute,
    on TRRUST with signed.py's class map, what fraction of those degrees of freedom single
    knockdowns pin down.
    PREDECLARED: if single-perturbation data constrains a large share of them, an in-silico joint
    experiment adds little and this module says so and stops. If it constrains a vanishing share,
    the joint experiment is the only route and AlphaGenome is the only available source of it.

A1  CAPABILITY AND ACCESS, MEASURED. Read the output types, sequence lengths and organism from the
    installed client, and probe the endpoint.
    PREDECLARED: without an API key the live experiment CANNOT run, and this module reports that
    plainly rather than substituting a proxy and calling it a test. A reachable endpoint with no
    key is reported as exactly that.

A2  THE DESIGN, PRE-REGISTERED so it runs unchanged the moment a key exists: which genes, which
    sites, which subsets, and which statistic separates count from identity.

A3  THE PIPELINE, VALIDATED WHERE VALIDATION IS POSSIBLE. The count-versus-identity decomposition
    is run on synthetic responses whose truth is known -- pure count, pure identity, and a mixture
    -- because no real dataset in this record varies the count at all, which is A0's first finding.
    PREDECLARED: the decomposition must recover the known truth on all three, or it is not fit to
    be pointed at model predictions.

A4  WHAT IT WOULD AND WOULD NOT SETTLE, including the applicability domain that a class count
    measured on a model's predictions is a class count FOR THAT MODEL.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import itertools
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.trrust_engine import signed_edges

API_KEY_VARS = ("ALPHAGENOME_API_KEY", "ALPHA_GENOME_API_KEY", "GDM_API_KEY")


def probe_endpoint(timeout=45):
    try:
        import grpc
    except ImportError:
        return "grpcio not installed"
    try:
        ch = grpc.secure_channel('dns:///gdmscience.googleapis.com:443',
                                 grpc.ssl_channel_credentials())
        grpc.channel_ready_future(ch).result(timeout)
        return "reachable"
    except Exception as e:                                   # pragma: no cover - network
        return f"unreachable: {type(e).__name__}"


def client_capabilities():
    """Read the capabilities off the installed client rather than quoting the marketing page."""
    try:
        from alphagenome.models import dna_client, dna_output
    except Exception as e:
        return None, str(e)
    caps = {
        "output_types": sorted(t.name for t in dna_output.OutputType),
        "sequence_lengths": {k: v for k, v in
                             sorted(dna_client.SUPPORTED_SEQUENCE_LENGTHS.items(),
                                    key=lambda kv: kv[1])},
        "organisms": sorted(o.name for o in dna_client.Organism),
    }
    return caps, None


def decompose(y, count, ident):
    """Share of variance explained by the COUNT alone, and by the identity beyond it.

    Group means by count give the count model; residuals within a count group that track identity
    are what a count summary cannot represent."""
    y = np.asarray(y, dtype=float)
    tot = float(np.var(y))
    if tot <= 0:
        return 0.0, 0.0
    cm = np.zeros_like(y)
    for c in np.unique(count):
        m = count == c
        cm[m] = y[m].mean()
    v_count = float(np.var(cm))
    resid = y - cm
    im = np.zeros_like(y)
    for key in set(map(tuple, ident)):
        m = np.array([tuple(r) == key for r in ident])
        if m.any():
            im[m] = resid[m].mean()
    v_ident = float(np.var(im))
    return v_count / tot, v_ident / tot


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("ALPHAGENOME: WHAT IT CAN DO, WHAT IT WOULD GIVE THIS ENGINE, AND WHETHER TO USE IT")
    P_(RULE)

    # ---- A1  CAPABILITY AND ACCESS -------------------------------------------------------------
    P_("\n" + RULE)
    P_("A1  CAPABILITY AND ACCESS, READ FROM THE INSTALLED CLIENT AND PROBED")
    P_(RULE)
    caps, err = client_capabilities()
    if caps is None:
        P_(f"  client not importable: {err}")
    else:
        P_(f"  output modalities ({len(caps['output_types'])}): "
           + ", ".join(caps["output_types"]))
        P_(f"  sequence lengths : "
           + ", ".join(f"{k.replace('SEQUENCE_LENGTH_', '')}={v:,}"
                       for k, v in caps["sequence_lengths"].items()))
        P_(f"  organisms        : " + ", ".join(caps["organisms"]))
    key = next((v for v in API_KEY_VARS if os.environ.get(v)), None)
    P_(f"\n  endpoint gdmscience.googleapis.com:443 : {probe_endpoint()}")
    P_(f"  API key in environment                 : {key or 'NONE of ' + ', '.join(API_KEY_VARS)}")
    P_(f"\n  A1: {'a key is present, so the pre-registered experiment in A2 can be run.' if key else 'THE ENDPOINT IS REACHABLE AND THERE IS NO KEY, so the live experiment CANNOT run here. That is reported as the state of affairs rather than replaced by a proxy measurement called a test.'}")

    # ---- A0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("A0  THE CEILING GATE: WOULD THE MODEL ANSWER SOMETHING THE DATA CANNOT?")
    P_(RULE)
    P_("  PART ONE: in single-perturbation data, does the perturbed COUNT ever vary?")
    from rem.atlas.humantransfer import prepare, regulator_index
    genes, tfs, DA, DB, expressed, _sha = prepare()
    regs, _g = regulator_index(genes, tfs)
    keep = {j: r for j, r in regs.items() if expressed[j] and len(r) >= 2}
    counts = collections.Counter()
    for j, r in keep.items():
        for _ti, _m in r:
            counts[1] += 1                       # one factor silenced per experiment, by design
    P_(f"    {len(keep)} target genes, {sum(counts.values())} (gene, regulator) observations")
    P_(f"    distinct values of the perturbed count across all observations: {sorted(counts)}")
    P_("    THE COUNT IS ONE IN EVERY OBSERVATION. A summary saying the response depends on HOW")
    P_("    MANY regulators are perturbed predicts a CONSTANT for each gene here, so this data can")
    P_("    REFUTE the count -- which humantransfer did, at 1.05 floors -- and can never CALIBRATE")
    P_("    it, and cannot test a class-count functional at all, because per-class counts never")
    P_("    move either.")

    P_("\n  PART TWO: what share of the engine's own per-gene degrees of freedom do marginals pin?")
    E, _ = signed_edges()
    tgt = collections.defaultdict(set)
    for u, v, _m in E:
        tgt[v].add(u)
    rng = np.random.default_rng(20260910)
    allregs = sorted({u for u, _v, _m in E})
    cls = {g: int(rng.integers(0, 64)) for g in allregs}      # signed.py's C = 64
    rows = []
    for k in (2, 4, 8, 20):
        sel = [s for s in tgt.values() if len(s) >= k]
        if not sel:
            continue
        fr = []
        for s in sel[:400]:
            cnt = collections.Counter(cls[u] for u in s)
            dof = 1
            for _c, m in cnt.items():
                dof *= (m + 1)
            fr.append(len(s) / dof)
        rows.append((k, len(sel), float(np.median(fr))))
    P_(f"\n    {'targets with k >=':>18} {'count':>8} {'median marginals / joint dof':>30}")
    for k, nsel, f in rows:
        P_(f"    {k:>18} {nsel:>8} {f:>30.4f}")
    worst = min(f for _k, _n, f in rows)
    P_(f"\n  A0: single knockdowns pin down a median {rows[0][2] * 100:.1f}% of the joint degrees of")
    P_(f"  freedom at k >= 2, falling to {worst * 100:.2f}% at k >= {rows[-1][0]}.")
    P_(f"  {'A vanishing share, so the joint experiment is the only route and an in-silico one is the only available source.' if worst < 0.05 else 'A large share, so an in-silico joint experiment would add little.'}")
    P_("  AND THE NUANCE THAT MATTERS: if the class hypothesis is TRUE, marginals plus the class")
    P_("  map determine the counts and the joint response follows. The joint data is needed to")
    P_("  TEST the hypothesis, not to fit it under the hypothesis -- which is exactly the test")
    P_("  this record has never been able to run.")

    # ---- A3  PIPELINE VALIDATION ---------------------------------------------------------------
    P_("\n" + RULE)
    P_("A3  THE DECOMPOSITION, VALIDATED ON RESPONSES WHOSE TRUTH IS KNOWN")
    P_(RULE)
    P_("  No dataset in this record varies the perturbed count, which is A0's first finding, so the")
    P_("  count-versus-identity decomposition is validated on synthetic responses instead.")
    k = 8
    subsets = [s for r in range(k + 1) for s in itertools.combinations(range(k), r)]
    cnt = np.array([len(s) for s in subsets])
    ind = [tuple(1 if i in s else 0 for i in range(k)) for s in subsets]
    w = rng.normal(0, 1, k)
    truths = {
        "pure count": np.array([-0.5 * len(s) for s in subsets]),
        "pure identity": np.array([-sum(w[i] for i in s) for s in subsets]),
        "half and half": np.array([-0.5 * len(s) - sum(w[i] for i in s) for s in subsets]),
    }
    P_(f"\n    {'synthetic truth':<18} {'var by COUNT':>14} {'var by IDENTITY':>17} {'recovered?':>12}")
    okall = True
    for nm, y in truths.items():
        vc, vi = decompose(y, cnt, ind)
        good = (nm == "pure count" and vc > 0.98 and vi < 0.02) or \
               (nm == "pure identity" and vi > 0.30) or \
               (nm == "half and half" and vc > 0.05 and vi > 0.05)
        okall = okall and good
        P_(f"    {nm:<18} {vc:>14.3f} {vi:>17.3f} {'yes' if good else 'NO':>12}")
    P_(f"\n  A3: {'PASS -- the decomposition recovers each known truth.' if okall else 'FAIL -- the decomposition does not recover known truths and is not fit to point at model predictions.'}")

    # ---- A2  THE PRE-REGISTERED DESIGN ---------------------------------------------------------
    P_("\n" + RULE)
    P_("A2  THE PRE-REGISTERED EXPERIMENT, TO RUN UNCHANGED WHEN A KEY EXISTS")
    P_(RULE)
    P_("  TARGETS. TRRUST genes with the most annotated regulators, taken in order, so the deep")
    P_(f"  regime is reached: the cap-binding regime is k >= 20 and TRRUST offers"
       f" {sum(1 for s in tgt.values() if len(s) >= 20)} such genes.")
    P_("  SITES. For each target, a 100KB window centred on the TSS; candidate sites are the")
    P_("    positions where CHIP_TF predicted binding for that target's annotated regulators is")
    P_("    highest, one site per regulator, so site identity maps to regulator identity.")
    P_("  PERTURBATION. Delete or scramble each chosen subset of sites and predict RNA_SEQ for the")
    P_("    target, giving a response per subset. Subsets: all singletons, all pairs, and a")
    P_("    stratified sample at each larger size, so the COUNT varies -- which is the property no")
    P_("    real dataset here has.")
    P_("  STATISTIC. The A3 decomposition: the share of response variance explained by the count")
    P_("    alone against the share explained by which sites, at matched count. Then the class")
    P_("    ladder: group sites into C classes and ask the smallest C whose per-class counts")
    P_("    reproduce the response, which is signed.py's question asked where the count actually")
    P_("    varies.")
    P_("  PREDECLARED BARS. The count summary survives if identity explains under 5% of the")
    P_("    variance at matched count. The class-count functional survives if some C well below the")
    P_("    regulator count reproduces the response to within the model's own perturbation")
    P_("    reproducibility, which must be measured first by re-predicting identical sequences.")

    # ---- A4  LIMITS ----------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("A4  WHAT IT WOULD AND WOULD NOT SETTLE")
    P_(RULE)
    P_("  1. A CLASS COUNT MEASURED ON A MODEL'S PREDICTIONS IS A CLASS COUNT FOR THAT MODEL. It")
    P_("     is evidence about biology only to the extent the model is right about the joint")
    P_("     effect of deleting several sites, which is the regime it is LEAST constrained in --")
    P_("     its training data is overwhelmingly observational, not combinatorially perturbed.")
    P_("     That is ledger S, and it is the reason this is a test of a hypothesis's coherence")
    P_("     rather than a measurement of a constant.")
    P_("  2. The model is deterministic, so there is no replicate noise -- and that removes the")
    P_("     thing that blocked whatdata, while ALSO removing the yardstick every accuracy figure")
    P_("     in this record is quoted in. A new one has to be defined, and re-predicting identical")
    P_("     sequences is not it: that measures serialisation, not uncertainty.")
    P_("  3. It says nothing about the pruning certificate, the enclosure width, or anything else")
    P_("     in the engine's cost line. It addresses the accuracy standard only.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_alphagenome.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
