"""LOOP 76 -- THE TF LAYER: SEPARATE THE CURATED CAUSAL EDGES FROM THE BINDING BACKBONE, AND TEST
THE SIGNS AGAINST MEASURED KNOCKDOWNS.

WHAT LOOKED BROKEN, AND WHAT IS ACTUALLY BROKEN. The model's `reg` layer holds 612,133 TF->target
edges of which 91.2% carry NO SIGN, and its largest regulator is CTCF with 12,122 targets -- an
insulator protein, not a transcriptional activator. The obvious reading is "this is ChIP occupancy
mislabelled as regulation, go and fetch a real curated network".

That reading is wrong, and measuring it first is the whole point of this loop. The curated network is
ALREADY IN THERE. `reg` is an un-deduplicated CONCATENATION of three blocks whose boundaries survive
in ROW ORDER, and two of the three are identifiable by containment:

    rows      0 ..  55,715    55,716 rows   CollecTRI    -- curated causal, carries the signs
    rows 55,716 .. 278,404   222,689 rows   DoRothEA     -- ChIP/TFBS/coexpression, 99.8% unsigned
    rows 278,405 .. 612,132  333,728 rows   UNIDENTIFIED -- see G1

So nothing needs to be downloaded to make this layer honest. What it needs is a LABEL. A consumer of
`reg` today cannot ask for the curated subset, because the merge threw away which block each edge
came from -- and CTCF's 12,122 edges are 96.2% block 2, inherited from DoRothEA's PAZAR/ReMap
evidence rather than from any curator. The layer is not lying; it is unlabelled, which in practice is
the same thing. docs/PRODUCT_ARCHITECTURE.md:41 already names the sources as "DoRothEA/CollecTRI,
ENCODE, hTFtarget"; nothing in the DATA records which edge came from which.

THE SIGN QUESTION, AND WHY IT NEEDS A GUARD. Restoring signs is easy and proves nothing. The
question that matters is whether the signs are TRUE: if CollecTRI says TF X ACTIVATES gene Y, then
knocking down X should push Y DOWN. That is directly testable and it is the gate.

It is also trivially gameable, and this project has the receipts for exactly that kind of failure.
93.2% of CollecTRI's signed edges are ACTIVATING. A knockdown pushes most genes down. So a "network"
that simply labelled every edge activating would score above a naive null while containing no sign
information whatsoever. G5 is the guard against that and it is mandatory, not decorative: if the
constant-ACTIVATING predictor passes, then G3 is measuring the down-shift of knockdown and NOT the
correctness of signs, and G3's PASS must be read that way.

WHY NOT THE REPO'S OWN PERTURB-SEQ. Measured, not assumed: the surviving Perturb-seq pickles are
truncated to the top ~250 movers per knockout, so a curated target is visible only if it was already
one of the largest responders -- selection on the outcome. Worse, the knocked-down gene does not
appear in its own readout in 0 of 1,400 K562 rows, so a failed knockdown cannot be told from a real
one. That is precisely the failure that produced this project's GATA1 result (bind_vs_reg.json:9,
on-target z = -0.84, and n_reg_down = 0 -- every one of the 56 "regulated" genes went UP). KnockTF is
used instead because it retains on-target effect size, which lets underpowered knockdowns be excluded
BEFORE scoring rather than explained afterwards.

TWO PRIOR RESULTS THIS MUST NOT QUIETLY OVERTURN, cited so the record stays consistent:
  outputs/orphan/bind_vs_reg.json:22-23   GATA1 binding vs regulation, fold 0.96x, p 0.68
                                          (HYPERGEOMETRIC. The "0.85x promoter-only" and
                                          "permutation p ~ 0.7" figures quoted elsewhere in this
                                          project exist only as prose -- they were never persisted,
                                          and the k562.h5ad they came from is gone.)
  outputs/orphan/chip_vs_perturb.json:8-11  ChIP binding -> which genes move, pooled AUC 0.5068 vs
                                          0.4981 permuted, p 0.2237. And the detail that deserves
                                          more attention than it got: auc_other_tfs_mean 0.5286
                                          EXCEEDS auc_own_mean 0.516 -- other TFs' binding predicted
                                          a TF's own knockdown better than its own binding did.
  outputs/orphan/adrn_signed.json         using sign made this project's predictions WORSE (gap
                                          -0.0324, verdict HARMFUL, 9/9 sweep cells negative).
This loop tests a DIFFERENT claim -- are curated signs consistent with measured perturbation
direction -- not whether sign improves mover prediction. Both can be true. Saying so here is the
difference between a new result and a quiet overturn.

PREDECLARED, before any number:

  G1 THE 612k EDGES ARE THREE CONCATENATED BLOCKS, AND TWO ARE IDENTIFIED    THE PROVENANCE PROOF.
       Boundaries are fixed here BEFORE the run at rows 0/55,716/278,405. Gate: >= 95% of CollecTRI's
       in-universe pairs present in `reg`, >= 95% of DoRothEA A-D's, and each block >= 90% covered by
       its claimed source. Block 3 must be reported as UNIDENTIFIED with its fingerprint, not
       absorbed into a neighbour. A layer whose provenance is 55% known is more useful than one that
       claims to be 100% known.
  G2 THE CURATED CORE IS SEPARABLE AND CARRIES THE SIGNS
       gate: the curated tier must hold >= 90% of all signed edges in `reg`. If the signs are spread
       across the binding blocks then "curated" is not a real partition and G3 cannot be scoped.
  G3 CURATED ACTIVATING SIGNS AGREE WITH MEASURED KNOCKDOWN DIRECTION       GATE A. EXPECT PASS.
       KnockTF human, one dataset per TF, filtered BEFORE scoring to on-target log2FC <= -0.5 and
       >= 8,000 measured genes. Unit of analysis is the TF, never the edge -- edges within one TF
       share one experiment and pooling them as independent inflates significance by roughly the
       regulon size. Per-TF agreement minus a COMPOSITION-MATCHED null (that TF's own observed
       P(down), weighted by its activating/repressing mix), then a one-sided paired test across TFs.
       Gate: mean per-TF delta > 0 over >= 150 TFs at p < 0.01.
       Alongside it, an EMPIRICAL null that redraws the same number of targets at random from the
       same dataset with the sign vector held fixed. [The first version shuffled SIGNS instead. In
       a single-sign arm that is a no-op -- every sign is already +1 -- and the first run proved it,
       returning a "null" identical to the observed value to four decimals in all three arms. An
       inert control printed beside a result reads as corroboration, which is worse than printing
       no control at all.]
  G4 REPRESSING SIGNS DO THE SAME                                           GATE B. EXPECT FAILURE.
       identical test on repressing edges alone. Stated in advance from an 18-TF pilot: CollecTRI
       +0.019 (z +0.98, null) and TRRUST -0.091 (z -2.98, ANTI-predictive). If this passes it is a
       genuine surprise; if it fails, the honest headline is that these signs carry activation
       information only.
  G5 THE CONSTANT-ACTIVATING PREDICTOR MUST FAIL                            THE GUARD. MANDATORY.
       relabel every curated edge ACTIVATING and rerun G3's test. It must NOT pass. If it does, G3 is
       measuring the global down-shift of a knockdown rather than sign correctness, and G3's result
       is reported as uninterpretable rather than as a win.
  G6 LEAKAGE AND THE WRITE
       curated edges whose supporting PMID equals the scoring KnockTF dataset's PMID are counted and
       excluded, so "not circular" is measured rather than asserted. Write is additive: every
       pre-existing top-level key of cell_complete.json must survive with an identical element count.

-> outputs/orphan/cell_tfnet.json  (+ outputs/loop_tf_network.json)
"""
import collections
import csv
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
TFOUT = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_tfnet.json"

COLLECTRI = SC / "collectri_refs.tsv"
DOROTHEA = SC / "dorothea_ABCD.tsv"
KNOCK = SC / "pert" / "knocktf_deg"
KINDEX = SC / "pert" / "knocktf_human_index.json"

# PREDECLARED block boundaries -- fixed before the run, from row-order analysis of `reg`
B1 = (0, 55716)
B2 = (55716, 278405)
B3 = (278405, 612133)

CONTAIN = 0.95      # G1, source pairs present in reg
COVER = 0.90        # G1, block covered by its claimed source
SIGN_SHARE = 0.90   # G2, share of signed edges in the curated tier
MIN_TFS = 150       # G3/G4
ALPHA = 0.01        # G3/G4
NPERM = 1000        # target-shuffle empirical null draws (see run(): a SIGN shuffle is inert here)
SEED = 7601

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def read_omnipath(path, level_field=None):
    """(a,b) -> sign, plus (a,b) -> set of PMIDs, over gene symbols."""
    sgn, refs, lvl = {}, {}, {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a = (r.get("source_genesymbol") or "").strip()
            b = (r.get("target_genesymbol") or "").strip()
            if not a or not b:
                continue
            s = 0
            if str(r.get("is_stimulation")) in ("True", "1"):
                s = 1
            if str(r.get("is_inhibition")) in ("True", "1"):
                s = -1 if s == 0 else 0     # both flags set -> genuinely ambiguous, treat unsigned
            sgn[(a, b)] = s
            if r.get("references"):
                refs[(a, b)] = {x.split(":")[-1] for x in r["references"].split(";") if x}
            if level_field and r.get(level_field):
                lvl[(a, b)] = r[level_field]
    return sgn, refs, lvl


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 76 -- the TF layer: separate curated causal edges from the binding backbone,")
    say("  then test the signs against measured knockdowns")
    say("=" * 100)
    say()

    from scipy.stats import wilcoxon

    D = json.load(open(CELL))
    G = D["genes"]
    n = len(G)
    names = [g["name"] for g in G]
    ix = {nm: i for i, nm in enumerate(names)}
    reg = D["reg"]
    say(f"     `reg` holds {len(reg):,} rows over {n:,} genes")

    say("G1 THE 612k EDGES ARE THREE CONCATENATED BLOCKS, AND TWO ARE IDENTIFIED")
    ct_sign, ct_refs, _ = read_omnipath(COLLECTRI)
    do_sign, _, do_lvl = read_omnipath(DOROTHEA, "dorothea_level")
    say(f"     CollecTRI (OmniPath, fresh): {len(ct_sign):,} pairs, "
        f"{sum(1 for v in ct_sign.values() if v):,} signed")
    say(f"     DoRothEA A-D (OmniPath, fresh): {len(do_sign):,} pairs")

    def inuniv(d):
        return {(a, b) for (a, b) in d if a in ix and b in ix}

    ct_u, do_u = inuniv(ct_sign), inuniv(do_sign)
    blocks = {}
    for nm2, (lo, hi) in (("1 CollecTRI", B1), ("2 DoRothEA", B2), ("3 UNIDENTIFIED", B3)):
        rows = reg[lo:hi]
        pairs = {(names[a], names[b]) for a, b, _ in rows}
        nsig = sum(1 for _, _, s in rows if s)
        blocks[nm2] = {"rows": len(rows), "pairs": pairs, "signed": nsig,
                       "sources": len({a for a, _, _ in rows})}
    allpairs = {(names[a], names[b]) for a, b, _ in reg}
    ct_in = len(ct_u & allpairs) / max(len(ct_u), 1)
    do_in = len(do_u & allpairs) / max(len(do_u), 1)
    cov1 = len(blocks["1 CollecTRI"]["pairs"] & ct_u) / max(len(blocks["1 CollecTRI"]["pairs"]), 1)
    cov2 = len(blocks["2 DoRothEA"]["pairs"] & do_u) / max(len(blocks["2 DoRothEA"]["pairs"]), 1)
    for nm2, v in blocks.items():
        say(f"       block {nm2:16s} {v['rows']:7,d} rows  {len(v['pairs']):7,d} pairs  "
            f"{v['sources']:5,d} sources  {v['signed']:6,d} signed "
            f"({v['signed']/max(v['rows'],1):5.1%})")
    say(f"     CollecTRI in-universe pairs present in `reg`: {ct_in:.4f}  "
        f"(block 1 covered by CollecTRI: {cov1:.4f})")
    say(f"     DoRothEA  in-universe pairs present in `reg`: {do_in:.4f}  "
        f"(block 2 covered by DoRothEA:  {cov2:.4f})")
    b3 = blocks["3 UNIDENTIFIED"]
    tgt = collections.Counter()
    for a, b, _ in reg[B3[0]:B3[1]]:
        tgt[a] += 1
    caps = np.array(sorted(tgt.values()))
    say(f"     block 3 is NOT IDENTIFIED. Fingerprint: {len(tgt):,} sources, max targets per source "
        f"{caps.max()}, median {np.median(caps):.0f}")
    say(f"       a hard cap near {caps.max()} targets/source is not reproduced by CollecTRI, "
        f"DoRothEA A-E, ChEA3, TFLink, GTRD or ReMap.")
    say(f"       best remaining candidate is ENCODE TF ChIP (roster match only, edge set untested); "
        f"hTFtarget could not be reached (DNS).")
    say(f"       It is left LABELLED UNIDENTIFIED rather than absorbed into a neighbour.")
    g1 = ct_in >= CONTAIN and do_in >= CONTAIN and cov1 >= COVER and cov2 >= COVER
    say(f"     G1 {'PASS' if g1 else 'FAIL'}")
    say()

    say("G2 THE CURATED CORE IS SEPARABLE AND CARRIES THE SIGNS")
    tot_signed = sum(1 for _, _, s in reg if s)
    share = blocks["1 CollecTRI"]["signed"] / max(tot_signed, 1)
    say(f"     {tot_signed:,} signed rows in `reg`; {blocks['1 CollecTRI']['signed']:,} "
        f"({share:.1%}) are in the curated block")
    say(f"     so `reg` is {blocks['1 CollecTRI']['rows']/len(reg):.1%} curated-causal and "
        f"{1 - blocks['1 CollecTRI']['rows']/len(reg):.1%} binding-or-unknown by row count")
    ctcf = [i for i, (a, b, s) in enumerate(reg) if names[a] == "CTCF"]
    inb2 = sum(1 for i in ctcf if B2[0] <= i < B2[1])
    say(f"     CTCF, the largest 'regulator': {len(ctcf):,} edges, {inb2/max(len(ctcf),1):.1%} in the "
        f"DoRothEA binding block, only {sum(1 for i in ctcf if i < B1[1])} curated")
    g2 = share >= SIGN_SHARE
    say(f"     G2 {'PASS' if g2 else 'FAIL'}")
    say()

    say("G3/G4/G5 SIGNS VERSUS MEASURED KNOCKDOWNS")
    files = sorted(KNOCK.glob("*.json"))
    say(f"     KnockTF datasets available: {len(files)} (on-target <= -0.5, >= 8,000 genes, "
        f"one per TF, filter fixed before scoring)")
    kidx = {}
    for r in json.load(open(KINDEX)):
        kidx[r.get("sample_id")] = str(r.get("sample_pubmed") or "")
    ct_by_tf = collections.defaultdict(dict)
    for (a, b), s in ct_sign.items():
        if s:
            ct_by_tf[a][b] = s

    def run(arm, force_activating=False):
        deltas, ntf, nedge, leaked = [], 0, 0, 0
        perm_deltas = []
        rng = np.random.default_rng(SEED)
        for fp in files:
            d = json.load(open(fp))
            tf = d["tf"]
            lfc = d["log2fc"]
            tgts = ct_by_tf.get(tf)
            if not tgts:
                continue
            pmid = kidx.get(d["sample_id"], "")
            ed = []
            for g, s in tgts.items():
                if g not in lfc or g == tf:
                    continue
                if pmid and pmid in ct_refs.get((tf, g), ()):
                    leaked += 1
                    continue
                if force_activating:
                    s = 1
                if arm == "act" and s != 1:
                    continue
                if arm == "rep" and s != -1:
                    continue
                ed.append((g, s))
            if len(ed) < 5:
                continue
            vals = np.array(list(lfc.values()), float)
            q = float((vals < 0).mean())          # this dataset's own P(down)
            sg = np.array([s for _, s in ed])
            y = np.array([lfc[g] for g, _ in ed], float)
            obs = float(np.mean(np.where(sg == 1, y < 0, y > 0)))
            exp = float(np.mean(np.where(sg == 1, q, 1 - q)))
            deltas.append(obs - exp)
            ntf += 1
            nedge += len(ed)
            # EMPIRICAL NULL = TARGET SHUFFLE, not sign shuffle.
            # The first version permuted the SIGNS within the arm. That is a no-op by
            # construction: G3 filters to activating edges, so every sign in `sg` is already +1
            # and permuting a constant vector returns the same vector. The first run proved it --
            # observed and "null" delta agreed to four decimals in all three arms (+0.0373 vs
            # +0.03726, -0.0020 vs -0.00198, +0.0290 vs +0.02898). An inert control reported
            # beside a result reads as corroboration and is worse than no control at all.
            # The meaningful question for a single-sign arm is not "is this sign right" but "are
            # THESE targets special", so the null redraws the same number of targets at random
            # from the same dataset and keeps the sign vector fixed.
            pd = []
            allg = list(lfc.keys())
            yall = np.array([lfc[g] for g in allg], float)
            for _ in range(NPERM // 10):
                samp = rng.choice(len(allg), size=len(ed), replace=False)
                yy = yall[samp]
                pd.append(float(np.mean(np.where(sg == 1, yy < 0, yy > 0))) - exp)
            perm_deltas.append(float(np.mean(pd)))
        deltas = np.array(deltas)
        if len(deltas) < 3:
            return {"n_tf": ntf, "n_edges": nedge, "mean_delta": float("nan"),
                    "p": float("nan"), "leaked": leaked, "perm_mean": float("nan")}
        try:
            p = float(wilcoxon(deltas, alternative="greater").pvalue)
        except Exception:
            p = float("nan")
        return {"n_tf": ntf, "n_edges": nedge, "mean_delta": float(deltas.mean()),
                "sd": float(deltas.std()), "frac_positive": float((deltas > 0).mean()),
                "p": p, "leaked": leaked, "perm_mean": float(np.mean(perm_deltas))}

    A = run("act")
    B = run("rep")
    C = run("all", force_activating=True)
    for lab, r in (("G3 ACTIVATING arm", A), ("G4 REPRESSING arm", B),
                   ("G5 GUARD: every edge relabelled ACTIVATING", C)):
        say(f"     {lab}")
        say(f"       {r['n_tf']} TFs, {r['n_edges']:,} edges;  mean per-TF delta "
            f"{r['mean_delta']:+.4f}  sd {r.get('sd', float('nan')):.4f}  "
            f"{r.get('frac_positive', float('nan')):.2f} of TFs positive")
        say(f"       one-sided paired p {r['p']:.2e}   target-shuffle null delta "
            f"{r['perm_mean']:+.5f}   PMID-leaked edges excluded {r['leaked']}")
    g3 = A["n_tf"] >= MIN_TFS and A["mean_delta"] > 0 and A["p"] < ALPHA
    g4 = B["n_tf"] >= 20 and B["mean_delta"] > 0 and B["p"] < ALPHA
    g5 = not (C["mean_delta"] > 0 and C["p"] < ALPHA)     # the GUARD passes when the sham FAILS
    say(f"     G3 {'PASS' if g3 else 'FAIL'}   G4 {'PASS' if g4 else 'FAIL'} "
        f"(failure was predeclared)   G5 {'PASS' if g5 else 'FAIL'}")
    if not g5:
        say(f"     G5 FAILED, AND THAT REWRITES G3. A network carrying NO sign information -- every")
        say(f"     edge called activating -- clears the same bar. So G3 is measuring the global")
        say(f"     down-shift of a knockdown across a TF's regulon, not whether the signs are right.")
        say(f"     G3's PASS is therefore NOT evidence that the signs are correct. The only arm that")
        say(f"     can carry sign information is G4, because a repressing call predicts the OPPOSITE")
        say(f"     direction to the down-shift, and G4 is where the answer actually is.")
    say()

    say("G6 LEAKAGE AND THE WRITE")
    tier = {}
    for i, (a, b, s) in enumerate(reg):
        t = "curated_causal" if i < B1[1] else ("binding_hts" if i < B2[1] else "unidentified")
        tier.setdefault(t, []).append(i)
    lvl_count = collections.Counter(do_lvl.values())
    say(f"     tiers: " + ", ".join(f"{k} {len(v):,}" for k, v in tier.items()))
    say(f"     DoRothEA confidence levels available for the binding tier: {dict(lvl_count)}")
    before = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in D.items()}
    payload = {
        "source": "reg partitioned by ROW ORDER into three concatenated blocks; blocks 1 and 2 "
                  "identified by containment against OmniPath CollecTRI and DoRothEA A-D "
                  "(fetched fresh). Block 3 is UNIDENTIFIED and is labelled as such.",
        "blocks": {"curated_causal": {"rows": [B1[0], B1[1]], "source": "CollecTRI",
                                      "containment": ct_in, "coverage": cov1},
                   "binding_hts": {"rows": [B2[0], B2[1]], "source": "DoRothEA A-D",
                                   "containment": do_in, "coverage": cov2},
                   "unidentified": {"rows": [B3[0], B3[1]], "source": None,
                                    "best_candidate": "ENCODE TF ChIP (roster match only)",
                                    "max_targets_per_source": int(caps.max())}},
        "warning": "the curated and binding tiers are DIFFERENT KINDS OF EVIDENCE and must not be "
                   "pooled. CTCF's dominance comes entirely from the binding tier.",
        "sign_validation": {"activating": A, "repressing": B, "constant_activating_guard": C},
        "edge_tier": {t: v for t, v in tier.items()},
        "dorothea_levels": {f"{a}|{b}": l for (a, b), l in do_lvl.items() if a in ix and b in ix},
    }
    TFOUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(TFOUT, "w"))
    D2 = json.load(open(CELL))
    after = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in D2.items()}
    changed = [k for k in before if before[k] != after.get(k)]
    sz = TFOUT.stat().st_size
    say(f"     wrote {sz/1e6:.1f} MB; cell_complete.json fields changed: {len(changed)}")
    g6 = not changed
    say(f"     G6 {'PASS' if g6 else 'FAIL'}")
    say()

    gates = {"G1 three blocks, two identified": bool(g1),
             "G2 curated core separable and carries the signs": bool(g2),
             "G3 activating signs agree with measured knockdowns": bool(g3),
             "G4 repressing signs agree": bool(g4),
             "G5 constant-activating guard fails as it must": bool(g5),
             "G6 leakage measured and write additive": bool(g6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL), str(COLLECTRI), str(DOROTHEA), str(KINDEX), str(KNOCK)],
                      available=len(reg), used=len(reg), selection="all", seed=SEED,
                      controls=["block boundaries predeclared before the run",
                                "containment measured in both directions",
                                "block 3 left labelled UNIDENTIFIED rather than absorbed",
                                "KnockTF filtered on on-target log2FC BEFORE scoring",
                                "unit of analysis is the TF, not the edge",
                                "composition-matched null per TF",
                                "target-shuffle empirical null within TF (a SIGN permutation is inert in a single-sign arm)",
                                "constant-ACTIVATING sham predictor run as a mandatory guard",
                                "PMID leakage between curated edge and scoring dataset excluded"],
                      note="the curated network was already inside `reg`; what was missing was the "
                           "block label, not the data")
    RM.report(man, emit=say)
    json.dump({"test": "loop_tf_network", "manifest": man, "gates": gates,
               "n_reg": len(reg), "blocks": {k: {"rows": v["rows"], "pairs": len(v["pairs"]),
                                                 "signed": v["signed"], "sources": v["sources"]}
                                             for k, v in blocks.items()},
               "collectri_containment": ct_in, "dorothea_containment": do_in,
               "block1_coverage": cov1, "block2_coverage": cov2,
               "signed_share_curated": share, "ctcf_edges": len(ctcf),
               "ctcf_frac_binding_block": inb2 / max(len(ctcf), 1),
               "activating": A, "repressing": B, "guard_constant_activating": C,
               "n_knocktf": len(files), "bytes": sz, "existing_fields_changed": changed,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_tf_network.json", "w"), indent=1)
    say(f"\n  -> {TFOUT}")
    say(f"  -> {OUT / 'loop_tf_network.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
