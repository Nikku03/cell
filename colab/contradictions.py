"""THE CONTRADICTION REGISTRY -- negative evidence as data the model can read.

WHY THIS EXISTS. Every null and every retraction in this project currently lives as prose in UPDATES.md.
Prose is unreadable to the model that has to avoid repeating the mistake, so nothing stops a later module from
rediscovering that Hi-C adds nothing, or re-deriving a competition feature that was withdrawn for leakage, or
quietly reusing a claim that was overturned. Biological databases are already biased toward positive findings;
a model built on them without a matching record of what was TESTED AND ABSENT will drift steadily optimistic.

WHAT AN ENTRY IS. Not "this failed". Each one records what was measured, what killed it, and -- the field
that keeps this from becoming dogma -- what evidence would REOPEN it. A registry that can never be overturned
is worse than no registry, because it converts a measurement into a belief.

    claim        the thing that is NOT supported, stated positively so it is searchable
    status       retracted (was claimed here, then withdrawn) | null (tested, absent)
                 | bounded (true only inside a stated region)
    measured     the numbers, including the control that mattered
    killed_by    the specific control or comparison that did it
    reopens_if   what would change the verdict

STATUS IS NOT SEVERITY. `retracted` is not worse than `null`; it means this project asserted it and then
withdrew it, which is a different provenance from never having supported it. Both bind future work equally.

THE THREE RECURRING ERRORS are recorded separately at the bottom, because they are not about biology -- they
are about how this project reads its own numbers, and they have each recurred more than once.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SCHEMA = ("id", "claim", "status", "domain", "measured", "killed_by", "module", "reopens_if")

ENTRIES = [
    # ---------------------------------------------------------------- dynamics
    dict(id="dyn-pseudotime", status="null", domain="dynamics",
         claim="Regulatory state predicts the CHANGE in expression along a pseudotime ordering.",
         measured="held-out delta R2 0.0013 on the change; the corresponding LEVEL scored 0.887, which is "
                  "autoregression rather than dynamics",
         killed_by="split-half reliability of the change was +0.3991, so the null is not a power failure; "
                   "validated links scored -0.0034, so it is not a link-quality failure",
         module="colab/dynamic_stage1.py",
         reopens_if="a dataset where the same cells are followed longitudinally, rather than different cells "
                    "sampled at each point"),
    dict(id="dyn-clock", status="null", domain="dynamics",
         claim="Regulatory state predicts the change in expression across real clock time.",
         measured="21 lags tested; permuting the time axis scored equally well",
         killed_by="permuted-time control indistinguishable from real time",
         module="colab/lagcurve.py",
         reopens_if="a system with a driven transient rather than a steady state -- which dyn-driver then "
                    "supplied, and which also came back null"),
    dict(id="dyn-driver", status="null", domain="dynamics",
         claim="TF occupancy dynamics predict expression dynamics when a real driver and a real clock are "
               "present (A549 + dexamethasone, NR3C1 ChIP and RNA on one 16-point grid).",
         measured="NR3C1 occupancy +0.0068 held-out R2 over interval identity and gene history on the "
                  "gene+interval-demeaned change; a CONSTANT per-gene vector with no time resolution "
                  "recovered +0.0095, which is 139% of it. Occupancy level +0.0067, occupancy change "
                  "+0.0013. Every one of eight targets showed static >= time-resolved.",
         killed_by="the time-averaged (static) arm matching or beating the time-resolved arm; permuting the "
                   "time axis cost +0.0061 of +0.0068. Replicate reliability of the change was 0.601 raw and "
                   "0.560 demeaned, so the target was measurable and this is not a ceiling artefact",
         module="colab/gr_dynamics.py",
         reopens_if="occupancy measured as continuous signal rather than thresholded peaks, or a readout of "
                    "nascent transcription (PRO-seq / SLAM-seq) instead of steady-state mRNA, which would "
                    "remove the mRNA half-life lag between occupancy and the measured change",
         note="the usable inverse DOES hold: the GR site map identifies WHICH genes are volatile "
              "(+0.0095, GR-specific against CTCF +0.0020 and RAD21 +0.0020). It does not say when."),
    dict(id="dyn-magnitude", status="null", domain="dynamics",
         claim="The MAGNITUDE of a perturbation's effect is predictable from biological features.",
         measured="detection power contributed +0.0881, larger than any biological block",
         killed_by="magnitude is confounded with detection power; existence and sign were separable, "
                   "magnitude was not",
         module="colab/effectsize.py",
         reopens_if="an assay with uniform detection power across genes, so power cannot stand in for size"),

    # ---------------------------------------------------------------- 3D / contact
    dict(id="contact-hic", status="null", domain="3D chromatin",
         claim="Measured Hi-C contact adds predictive value to enhancer-gene assignment.",
         measured="-0.0031 incremental against a model already carrying 311 measured K562 TF ChIP tracks",
         killed_by="incremental test against a data-rich baseline",
         module="colab/hic_gate.py",
         reopens_if="a data-POOR cell type: stripping tracks moved measured Hi-C from -0.0053 to +0.0254 and "
                    "CTCF-count from +0.0523 to +0.0993, so the ledger is conditional on data richness and "
                    "the condition is load-bearing"),
    dict(id="contact-polymer", status="null", domain="3D chromatin",
         claim="Polymer simulation of chromatin contact improves enhancer-gene assignment.",
         measured="analytic Rouse +0.005; KMC loop extrusion with Langevin dynamics -0.0027",
         killed_by="both fell inside the noise of the baseline they were added to",
         module="colab/extrude.py, colab/fieldsim.py",
         reopens_if="same as contact-hic -- the data-poor regime, where contact features have room to "
                    "contribute"),
    dict(id="contact-4d-layer", status="retracted", domain="3D chromatin",
         claim="4D chromatin is a distinct predictive layer in the cell model.",
         measured="what carries it is 1D distance plus CTCF; adding contact on top adds nothing",
         killed_by="block ablation -- the '4D' contribution disappears once 1D and CTCF are present",
         module="colab/shortrange_mechanism.py",
         reopens_if="a task where 1D distance is held constant by construction, so contact is the only "
                    "variable left"),

    # ---------------------------------------------------------------- enhancer-gene
    dict(id="eg-competition", status="retracted", domain="enhancer-gene",
         claim="Competition between enhancers for a promoter predicts which enhancer wins (+0.0278 AUPRC).",
         measured="n_candidates ALONE scores AUROC 0.2733 -- the feature is group size, not biology. "
                  "Restricted to groups with >=5 candidates the block is worth -0.0068. Training on one "
                  "candidate definition and scoring another cost -0.1201 AUPRC",
         killed_by="group-size leakage plus candidate-set dependence: features defined relative to a "
                   "candidate set do not transfer between definitions",
         module="colab/competition_audit.py",
         reopens_if="a candidate set defined identically at training and inference, with group size "
                    "explicitly controlled"),
    dict(id="eg-distal", status="null", domain="enhancer-gene",
         claim="Distal (100-250 kb) enhancer-gene pairs can be called at usable precision from occupancy.",
         measured="0 edges emitted -- the stratum never reaches the 0.30 precision floor. v1 shipped 6,096 "
                  "distal edges and they were riding the leaky competition block",
         killed_by="per-distance-stratum calibration after the competition block was removed",
         module="colab/eg_layer.py",
         reopens_if="sequence-level features, or measured contact in a cell type where it is informative"),
    dict(id="eg-silencer", status="retracted", domain="enhancer-gene",
         claim="Elements whose perturbation raises a gene are silencers.",
         measured="CTCF p 0.84, H3K27me3 p 0.72 -- neither repressive mark is enriched. The "
                  "indirect-neighbour chain was also refuted: neighbours go DOWN, 29.5% vs 46.5%, p 9.4e-10",
         killed_by="the mechanism tests for both candidate explanations came back null; the polarity effect "
                   "itself shrank from AUROC 0.858 to 0.6497 (p 0.032) once matched",
         module="colab/polarity_stress.py, colab/distal_mechanism.py, colab/neighbour_repressor.py",
         reopens_if="nothing specific -- the positive-effect class is real but UNEXPLAINED, and that is the "
                    "honest state. Any proposed mechanism needs its own matched test"),

    # ---------------------------------------------------------------- network layers
    dict(id="net-admission", status="null", domain="network layers",
         claim="The model's network layers improve prediction of a perturbation's transcriptional "
               "consequence.",
         measured="on K562 CRISPRi -> dRNA, against a nuisance ladder reaching R2 0.0151: regulatory "
                  "(610,256 edges) -0.0002, signalling -0.0001, PPI -0.0000, co-dependency -0.0001, complex "
                  "co-member +0.0006 against its own control +0.0003. 0 of 5 admitted",
         killed_by="degree-matched (configuration-model) controls, and folds grouped by perturbed gene so "
                   "the row effect cannot be looked up",
         module="colab/chain_benchmark.py",
         reopens_if="ALREADY PARTLY REOPENED -- a direct high-power test showed the layers do carry "
                    "magnitude information (PPI 1.208x elevation in |lfc| over matched pairs, p ~0; "
                    "co-dependency 1.070x; regulatory 1.021x; signalling 1.028x at p 0.99). They carry "
                    "magnitude, not direction, and magnitude is what the ladder already supplies",
         note="the layers are not empty; they are redundant with the ladder on this task"),
    dict(id="net-curated-sign", status="bounded", domain="network layers",
         claim="The curated regulatory layer supplies edge direction.",
         measured="sign = 0 on 558,005 of 612,133 edges (91.2%); +1 on 46,448; -1 on 7,680",
         killed_by="direct inspection of the stored schema -- no evidence field exists at all, on any edge, "
                   "in any layer (PPI edges are bare 2-tuples with STRING confidence dropped)",
         module="colab/entity_registry.py",
         reopens_if="not applicable -- this is a property of the representation. reg_perturb_k562 supplies "
                    "sign from intervention instead"),
    dict(id="net-protein-arrow", status="retracted", domain="network layers",
         claim="Network arrows that work for mRNA do not work for protein.",
         measured="overturned at 20 proteins -- the original claim rested on too few",
         killed_by="expanding the protein set",
         module="colab/protein_orthogonal.py",
         reopens_if="not applicable -- the retraction is of the negative claim, not of a positive one"),
    dict(id="net-viper", status="null", domain="network layers",
         claim="VIPER/DoRothEA regulon-based TF activity inference is usable in this model.",
         measured="regulon overlap with the model's own gene universe was too sparse to score",
         killed_by="coverage, not method -- recorded as an honest negative at the time",
         module="colab/viper.py, colab/regulon.py",
         reopens_if="a regulon set built on this gene universe, e.g. from reg_perturb_k562 itself"),

    # ---------------------------------------------------------------- metabolism / protein
    dict(id="metab-dose", status="bounded", domain="metabolism",
         claim="A gene required for more reactions is more essential.",
         measured="TRUE only for 1-2 reactions (+0.1090 dep_frac matched, 100% of draws p<0.05). At 3+ it is "
                  "+0.0207 at only 35% of draws -- not detectable. The dose curve rises 0.068 -> 0.213 -> "
                  "0.244 then falls to 0.061 and 0.046",
         killed_by="the sub-class match against the same required-for-nothing group; high-count genes are "
                   "31% transport and 11% fatty-acid oxidation, replicated reaction families where the count "
                   "is model bookkeeping",
         module="colab/metabolic_layer.py",
         reopens_if="a reaction set that collapses replicated transport and acyl-chain families, so the "
                    "count reflects distinct catalytic jobs"),
    dict(id="prot-generic-abundance", status="bounded", domain="protein",
         claim="The generic `ppm` table is an adequate stand-in for K562 protein abundance.",
         measured="rank correlation 0.655 with the K562 measurement over 6,205 shared genes; 25% of genes "
                  "shift more than a quarter of the abundance range. Disagreement is structured: histones "
                  "over-stated, EP300 and TCF3 under-stated",
         killed_by="direct comparison against PaxDb 9606-iBAQ_K562_Geiger_2012",
         module="colab/k562_protein.py",
         reopens_if="not applicable -- use the cell-matched measurement where it exists and mark it unknown "
                    "where it does not"),
    dict(id="prot-mrna-ceiling", status="bounded", domain="protein",
         claim="mRNA change is a usable proxy for protein change.",
         measured="mRNA explains 15.9% of protein LEVEL variance in K562 (r 0.399, Spearman 0.501, 5,874 "
                  "genes both measured in this cell type)",
         killed_by="not killed -- bounded. This is an upper bound on the chain's dRNA -> dprotein arrow, and "
                   "a generous one, since a delta is harder to predict than a level",
         module="colab/k562_protein.py",
         reopens_if="K562 perturbation proteomics at scale, which does not currently exist anywhere"),

    # ---------------------------------------------------------------- transportability
    dict(id="transfer-penalty", status="bounded", domain="transportability",
         claim="Benchmark performance transfers to unseen cell types.",
         measured="POOLED minus LOCO = +0.0597 -- the pooled number is inflated by that much relative to "
                  "leave-one-cell-type-out",
         killed_by="leave-one-cell-type-out evaluation",
         module="colab/ranking_gate.py",
         reopens_if="not applicable -- the penalty is the finding. Report LOCO, not pooled"),
]

# Errors of reading rather than of biology. Each has recurred, which is why they are recorded as standing
# hazards rather than as individual incidents.
METHOD_ERRORS = [
    dict(id="err-net-of-shuffled", occurrences=4,
         error="Reading `R2 - shuffled` (or AUPRC - shuffled) as an effect size.",
         why_wrong="Subtracting a shuffled arm CREDITS a model for overfitting noise. The effect size is the "
                   "raw held-out value; the shuffled arm establishes significance only.",
         seen_in="effectsize.py, power_conditioned.py, and the verdict ladders of two further modules"),
    dict(id="err-single-draw", occurrences=3,
         error="Reading a single matched draw as the result.",
         why_wrong="One balanced draw is one sample of the matched population. In bound_causal.py the single "
                   "draw also left a residual confound (promoter busyness p 0.037) that 20 decile-matched "
                   "draws removed (p 0.698) -- and the effect then came out STRONGER, so the error is not "
                   "conservative in either direction.",
         seen_in="metabolic_layer.py, bound_causal.py, and one earlier polarity module"),
    dict(id="err-flat-is-opposite", occurrences=2,
         error="Labelling a non-significant result as REVERSED rather than NOT DETECTED.",
         why_wrong="'We cannot see it' is a strictly weaker claim than 'it runs the other way'. Both times "
                   "this came from thresholding on an absolute value without also reading significance.",
         seen_in="shortrange_mechanism.py, metabolic_layer.py"),
    dict(id="err-silent-join", occurrences=3,
         error="A join that matches nothing and returns empty instead of raising.",
         why_wrong="A control computed on an all-zero column passes, and a passing control is "
                   "indistinguishable from a clean result. `ppm`/`abund` are keyed by gene INDEX, so symbol "
                   "lookups returned None for every gene -- and abundance later proved the strongest "
                   "confound in that same analysis (p 6e-31).",
         seen_in="metabolic_layer.py, a STRING PPI join (0/11,636), a doubled-prefix URL 404",
         fixed_by="colab/entity_registry.py -- Registry.join() declares an expected rate and raises below it"),
    dict(id="err-provenance-on-the-axis", occurrences=2,
         error="Letting a processing difference lie along the axis being measured.",
         why_wrong="Two labs contributed to one time course with the second present at 60 minutes ONLY, and "
                   "every experiment carried both an RSEM V29 and a featureCounts V22 quantification under "
                   "the same output_type. Either would put a batch effect exactly where the signal was being "
                   "read.",
         seen_in="fetch_gr_timecourse.py",
         fixed_by="pin one lab, one pipeline version, one annotation -- and ASSERT it, since stating the "
                  "rule in a docstring is not checking it"),
]


def main():
    print("=" * 100)
    print("CONTRADICTION REGISTRY -- what has been tested and found absent")
    print("=" * 100)

    bad = []
    for e in ENTRIES:
        missing = [k for k in SCHEMA if k not in e]
        if missing:
            bad.append((e.get("id", "?"), f"missing fields {missing}"))
        for m in str(e.get("module", "")).split(","):
            m = m.strip()
            if m and not (ROOT / m).exists():
                bad.append((e["id"], f"module not found: {m}"))
    if bad:
        for i, why in bad:
            print(f"  INVALID {i}: {why}")
        raise SystemExit(f"{len(bad)} invalid entries -- a registry that rots silently is worse than none")

    from collections import Counter
    st = Counter(e["status"] for e in ENTRIES)
    dom = Counter(e["domain"] for e in ENTRIES)
    print(f"  {len(ENTRIES)} entries validated, every referenced module present")
    print(f"  status: {dict(st)}")
    print(f"  domain: {dict(dom)}")
    print(f"\n  {'id':24s} {'status':10s} {'domain':18s} claim")
    for e in sorted(ENTRIES, key=lambda x: (x["domain"], x["id"])):
        print(f"  {e['id']:24s} {e['status']:10s} {e['domain']:18s} {e['claim'][:62]}")

    print(f"\n  METHOD ERRORS (recurring, about reading numbers rather than about biology)")
    for m in METHOD_ERRORS:
        print(f"    {m['id']:24s} x{m['occurrences']}  {m['error']}")

    reopenable = [e for e in ENTRIES if "not applicable" not in e["reopens_if"]]
    print(f"\n  {len(reopenable)}/{len(ENTRIES)} entries name specific evidence that would REOPEN them.")
    print(f"  A registry that cannot be overturned converts a measurement into a belief, so the field is "
          f"required.")

    OUT.mkdir(parents=True, exist_ok=True)
    reg = {"purpose": "negative evidence as structured data; consult before asserting any claim in these "
                      "domains, and before building a layer whose value one of these entries already "
                      "measured",
           "generated_by": "colab/contradictions.py",
           "status_meaning": {
               "retracted": "asserted by this project, then withdrawn on further measurement",
               "null": "tested against controls and found absent",
               "bounded": "true only inside the stated region"},
           "n_entries": len(ENTRIES), "by_status": dict(st), "by_domain": dict(dom),
           "entries": ENTRIES, "method_errors": METHOD_ERRORS}
    json.dump(reg, open(OUT / "contradictions.json", "w"), indent=1)
    print(f"\n  -> {OUT/'contradictions.json'}")


if __name__ == "__main__":
    main()
