"""activity_target -- honestly test the user's PIVOT: instead of predicting buffered mRNA, make the CHROMATIN ACTIVITY the target variable, because
the mechanistic signal lives there (perturb_atac: 67/227 TFs chromatin-active while mRNA-silent). The proposal is only a real asset if the chromatin
target is MORE PREDICTABLE FROM GENERALIZABLE INPUTS than mRNA is -- otherwise inverting the model just moves the same representation wall to a new
label. Tested on the K562 knockouts that have BOTH readouts (Perturb-seq mRNA + Spear-ATAC chromatin) and a DepMap co-dependency vector (the input).

Three measurements:
  A) TARGET-INPUT ALIGNMENT (the decisive one): build KO x KO similarity matrices in three spaces -- INPUT = DepMap co-dependency, and two candidate
     TARGETS = chromatin-activity (chromVAR motif deviations) and mRNA (tide-removed response). If corr(codep-sim, chromatin-sim) > corr(codep-sim,
     mRNA-sim), the chromatin target is better explained by a generalizable input -> the pivot helps. If equal, the wall moves with the target.
  B) NEIGHBOUR-TRANSFER (leave-one-out): predict each KO's target vector from its nearest co-dependency neighbour's target; correlation. Higher for
     chromatin than mRNA => chromatin transfers between related knockouts better => more learnable.
  C) CROSS-READOUT AGREEMENT: do chromatin-sim and mRNA-sim even agree on which knockouts are similar? (Are they measuring the same axes?)

HONEST BLOCKERS stated up front, independent of the result: (1) LABEL SCARCITY -- Spear-ATAC covers ~36 TF knockdowns, not the 1400 of Perturb-seq,
so a *general* knockout->activity model cannot be trained on this; (2) READOUT NARROWNESS -- chromVAR activity only moved for 4/25 sequence-specific
TFs (perturb_atac), and motifs are family-level, so 'chromatin activity' is a target only for accessibility-shaping TFs, not the metabolic/structural
majority. This module measures whether the pivot is even DIRECTIONALLY sound on the overlap; the blockers bound how far it can go. Deterministic."""
import os
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")) / "spatac"


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


def main():
    H = Harness("K562")
    d = np.load(SP / "large_motif_pb.npz", allow_pickle=True)
    groups = [str(x) for x in d["groups"]]; gi = {g: i for i, g in enumerate(groups)}
    mean = d["mean"]
    if "NT" not in gi:
        print("no NT control in ATAC; abort"); return
    NT = gi["NT"]
    tf_atac = [g for g in groups if g != "NT"]
    # chromatin-activity vector per TF = motif-deviation shift vs control
    chrom = {t: mean[gi[t]] - mean[NT] for t in tf_atac}

    # DepMap co-dependency input
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}

    # overlap: TF has chromatin + mRNA (in KO panel) + co-dep
    nti = H.nontide_idx
    ov = [t for t in tf_atac if t in H.ki and t in srow]
    print(f"K562 knockouts with BOTH chromatin (Spear-ATAC) and mRNA (Perturb-seq) and a co-dep input: {len(ov)}")
    print("  ", ", ".join(ov))
    if len(ov) < 8:
        print("too few overlapping knockouts for a powered test.");
    mrna = {t: H.M[H.ki[t]][nti] for t in ov}                     # tide-removed mRNA response
    codep = {t: Z[srow[t]] for t in ov}

    n = len(ov)
    # similarity matrices (upper triangle)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    codep_s = np.array([pear(codep[ov[i]], codep[ov[j]]) for i, j in pairs])
    chrom_s = np.array([pear(chrom[ov[i]], chrom[ov[j]]) for i, j in pairs])
    mrna_s = np.array([pear(mrna[ov[i]], mrna[ov[j]]) for i, j in pairs])

    # A) target-input alignment
    a_chrom = pear(codep_s, chrom_s); a_mrna = pear(codep_s, mrna_s); cross = pear(chrom_s, mrna_s)
    print(f"\nA) INPUT->TARGET alignment (does co-dependency similarity predict target similarity?):")
    print(f"     co-dep  ->  CHROMATIN target : r = {a_chrom:+.3f}")
    print(f"     co-dep  ->  mRNA target      : r = {a_mrna:+.3f}")
    print(f"     (chromatin vs mRNA agreement  : r = {cross:+.3f})")

    # B) leave-one-out nearest-codep-neighbour transfer
    def loo_transfer(target):
        cs = []
        for i in range(n):
            sims = [(pear(codep[ov[i]], codep[ov[j]]), j) for j in range(n) if j != i]
            j = max(sims)[1]
            cs.append(pear(target[ov[i]], target[ov[j]]))
        return float(np.mean(cs))
    t_chrom = loo_transfer(chrom); t_mrna = loo_transfer(mrna)
    print(f"\nB) NEIGHBOUR-TRANSFER (predict target from nearest co-dep neighbour, LOO correlation):")
    print(f"     CHROMATIN target : {t_chrom:+.3f}")
    print(f"     mRNA target      : {t_mrna:+.3f}")

    # shuffle null for A (permute one similarity vector)
    rng = np.random.RandomState(0)
    null_chrom = np.mean([abs(pear(codep_s, chrom_s[rng.permutation(len(chrom_s))])) for _ in range(500)])

    pivot_helps = (a_chrom > a_mrna + 0.05) and (t_chrom > t_mrna + 0.05)
    verdict = (
        f"ACTIVITY-TARGET PIVOT (activity_target.py): the user's proposal -- invert the model to predict CHROMATIN ACTIVITY (where the mechanistic "
        f"signal survives) instead of buffered mRNA. Tested on the {n} K562 knockouts with BOTH readouts + a co-dependency input. "
        f"DECISIVE TEST -- is the chromatin target better predicted by a generalizable input (co-dependency) than mRNA is? co-dep->chromatin "
        f"r={a_chrom:+.3f} vs co-dep->mRNA r={a_mrna:+.3f}; nearest-neighbour transfer chromatin {t_chrom:+.3f} vs mRNA {t_mrna:+.3f}. "
        + ("CHROMATIN IS MORE PREDICTABLE from the input than mRNA on both measures -- so the pivot is DIRECTIONALLY SOUND: making activity the "
           "target does move the model onto a more learnable signal, not just relabel the wall. The user is right that the target should be the "
           "activity layer. " if pivot_helps else
           "The chromatin target is NOT clearly more predictable from the input than mRNA is (both alignments are similar/low). MEANING: the pivot "
           "is intuitive but does not, on this overlap, escape the representation gap -- the chromatin response is about as PRIVATE (poorly "
           "predicted from generalizable inputs) as the mRNA response. Inverting the model to predict activity moves the target but not the wall: "
           "the specific signal is still only in the knockout's own measured profile, now a chromatin one. ")
        + f"(co-dep->chromatin alignment vs shuffle null {null_chrom:.3f}.) "
        + "HONEST BLOCKERS that bound the pivot regardless of direction: (1) LABEL SCARCITY -- Spear-ATAC has ~36 TF knockdowns vs Perturb-seq's "
        "1400, so a GENERAL knockout->activity model cannot be trained (this test's n is tiny and TF-biased); (2) READOUT NARROWNESS -- chromVAR "
        "activity moved for only 4/25 sequence-specific TFs and is family-level, so 'chromatin activity' is a target only for accessibility-shaping "
        "TFs, blind to the metabolic/structural majority of knockouts. CONCLUSION: the pivot is the right INSTINCT (measure/predict the layer that "
        "carries the signal), and directionally "
        + ("supported here, " if pivot_helps else "unsupported at this resolution, ")
        + "but the operational blocker is DATA: an activity-layer target at Perturb-seq SCALE and BREADTH does not exist yet. The lever is generating "
        "that measurement (genome-wide Perturb-ATAC / nascent / phospho), after which the target-pivot becomes trainable. Deterministic; small n, "
        "stated.")
    print("\nVERDICT: " + verdict)
    json.dump({"n_overlap": n, "overlap_kos": ov, "codep_to_chromatin_r": round(a_chrom, 4),
               "codep_to_mrna_r": round(a_mrna, 4), "chromatin_mrna_agreement_r": round(cross, 4),
               "transfer_chromatin": round(t_chrom, 4), "transfer_mrna": round(t_mrna, 4),
               "codep_chromatin_shuffle_null": round(float(null_chrom), 4), "pivot_directionally_helps": bool(pivot_helps),
               "verdict": verdict, "note": verdict}, open(OUT / "activity_target.json", "w"), indent=1)
    print("\n  -> outputs/orphan/activity_target.json")


if __name__ == "__main__":
    main()
