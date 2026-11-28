"""grn_reprogram — can we REPROGRAM the running cell with transcription factors? The Yamanaka experiment, in silico.

The essentiality null said: running the regulatory network doesn't tell you which genes MATTER. This asks a
different, more forgiving question that TFs are actually FOR: if you force a master TF ON in the resting cell and
let it run, does the cell flow to that TF's real lineage program — and SPECIFICALLY that one, not every lineage?

  force GATA1 on  -> should light up the ERYTHROID program (HBB, ALAS2, KLF1, GYPA, ...)   more than myeloid/muscle
  force MYOD1 on  -> should light up MUSCLE (MYOG, MYH1, CKM, ...)                          more than erythroid/...
  force POU5F1 on -> should light up PLURIPOTENCY (NANOG, SOX2, ...)                        etc.

Method (honest):
  * baseline = run to a resting attractor; then CLAMP the TF ON and re-run — Δ = what the TF induces from rest.
  * average Δ over several resting seeds (controls for the many-attractors problem).
  * programs are TEXTBOOK marker sets, defined independently of the edge list; we EXCLUDE the forced TF itself.
  * specificity = does a TF induce its OWN program more than the OTHER programs? (rank-1 hit, matched-vs-mismatched
    AUC). A TF that raises everything equally is NOT reprogramming.
  * circularity disclosed: we report how many program genes are DIRECT edge-targets of the TF (trivially wired) vs
    reached indirectly — a correct direct wiring is still a real positive, but the reader should see the split.

-> outputs/orphan/grn_reprogram.json + printed specificity matrix + verdict (null reported honestly if null).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from grn import GRN
from complete_cell import CompleteCell

# textbook master-TF -> lineage marker program (NOT derived from the edge list)
PROGRAMS = {
    "GATA1":  ["HBB", "HBA1", "HBA2", "ALAS2", "KLF1", "SLC4A1", "GYPA", "EPB42", "AHSP", "SPTA1"],   # erythroid
    "SPI1":   ["CSF1R", "ITGAM", "CD14", "LYZ", "MPO", "CEBPB", "FCGR1A", "CD68"],                    # myeloid
    "MYOD1":  ["MYOG", "MYH1", "ACTA1", "TNNT3", "DES", "CKM", "MYF6"],                               # skeletal muscle
    "POU5F1": ["NANOG", "SOX2", "ZFP42", "DPPA4", "LIN28A", "SALL4"],                                 # pluripotency
    "PAX5":   ["CD19", "CD79A", "CD79B", "MS4A1", "BLNK", "EBF1"],                                    # B-cell
    "HNF4A":  ["ALB", "APOB", "TTR", "APOA1", "SERPINA1", "APOC3"],                                   # hepatocyte
    "TP53":   ["CDKN1A", "BAX", "MDM2", "GADD45A", "BBC3", "SFN"],                                    # p53 stress/arrest
    "FOXP3":  ["IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],                                                # regulatory T
    "RUNX1":  ["TAL1", "GATA2", "ITGA2B", "PF4", "CD34"],                                             # HSC/megakaryo
}


def _auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(seeds=(0, 1, 2, 3, 4), save=True):
    C = CompleteCell()
    g = GRN(C=C)
    incore = {int(x): k for k, x in enumerate(g.nodes)}          # global gene idx -> local
    name2i = C.idx

    tfs = [tf for tf in PROGRAMS if name2i.get(tf) in incore]
    # program genes present in the core (local indices), excluding the driving TF handled per-row below
    prog_local = {tf: {name2i[p]: incore[name2i[p]] for p in genes
                       if name2i.get(p) in incore} for tf, genes in PROGRAMS.items()}

    # direct-target disclosure: how many program genes are direct edge-targets of the TF
    direct = {}
    for tf in tfs:
        ti = name2i[tf]
        tgts = {j for j, *_ in (C.causal_out.get(ti, []) or [])}
        direct[tf] = sum(1 for gi in prog_local[tf] if gi in tgts)

    # induction Δ per (forced TF), averaged over resting seeds
    dvecs = {}                                                    # tf -> Δ activation vector (len n)
    for tf in tfs:
        tl = incore[name2i[tf]]
        acc = np.zeros(g.n, dtype=np.float64)
        for s in seeds:
            base, _ = g.run(g._seed(seed=s), steps=400)
            forced, _ = g.run(base.copy(), steps=200, clamp={tl: 1.0})
            acc += (forced - base)
        dvecs[tf] = acc / len(seeds)

    # specificity matrix S[forced_tf][program] = mean induction of that program's genes (exclude the forced TF gene)
    S = {}
    for ftf in tfs:
        d = dvecs[ftf]
        row = {}
        for ptf in tfs:
            locs = [loc for gi, loc in prog_local[ptf].items() if gi != name2i[ftf]]
            row[ptf] = float(np.mean([d[l] for l in locs])) if locs else float("nan")
        S[ftf] = row

    # scoring: for each forced TF, rank programs by induction; is its OWN program #1? matched vs mismatched AUC
    hit1, ranks, matched, mismatched = 0, [], [], []
    per_tf = {}
    for ftf in tfs:
        order = sorted(tfs, key=lambda p: -S[ftf][p])
        r = order.index(ftf) + 1
        ranks.append(r); hit1 += int(r == 1)
        matched.append(S[ftf][ftf])
        mismatched += [S[ftf][p] for p in tfs if p != ftf]
        per_tf[ftf] = {"own_induction": S[ftf][ftf], "rank_of_own_program": r,
                       "top_induced": order[0], "direct_targets_in_program": direct[ftf],
                       "program_size_in_core": len(prog_local[ftf])}
    lab = [True] * len(matched) + [False] * len(mismatched)
    spec_auc = _auc(matched + mismatched, lab)

    # EMERGENCE test: repeat the matched-vs-mismatched AUC using ONLY program genes that are NOT direct edge-targets
    # of the forced TF (reached only through multi-step dynamics). If specificity survives here, it's not readback.
    tf_targets = {tf: {j for j, *_ in (C.causal_out.get(name2i[tf], []) or [])} for tf in tfs}
    im, imm, n_indirect = [], [], 0
    for ptf in tfs:
        for gi, loc in prog_local[ptf].items():
            if gi == name2i[ptf] or gi in tf_targets[ptf]:      # skip self + direct targets of its OWN master TF
                continue
            n_indirect += 1
            im.append(dvecs[ptf][loc])                          # induction under the matched TF
            imm += [dvecs[o][loc] for o in tfs if o != ptf]     # induction under mismatched TFs
    indirect_auc = _auc(im + imm, [True] * len(im) + [False] * len(imm)) if im else float("nan")

    # ---- print
    print(f"TF REPROGRAMMING on the running cell — force each master TF ON, avg over {len(seeds)} resting states")
    print(f"{len(tfs)} master TFs, textbook programs.  matrix = mean induction Δ (row forces TF, col = program)\n")
    hdr = "forced\\prog " + " ".join(f"{p[:6]:>7}" for p in tfs)
    print(hdr); print("-" * len(hdr))
    for ftf in tfs:
        cells = " ".join(f"{S[ftf][p]:>+7.2f}" for p in tfs)
        star = "  <- own #1" if per_tf[ftf]["rank_of_own_program"] == 1 else \
               f"  own rank {per_tf[ftf]['rank_of_own_program']}"
        print(f"{ftf:>10}  {cells}{star}")
    print()
    print(f"  reprogramming HIT@1 (own program most-induced): {hit1}/{len(tfs)}")
    print(f"  median rank of own program (of {len(tfs)}):       {int(np.median(ranks))}")
    print(f"  matched-vs-mismatched induction AUC (all program genes): {spec_auc:.3f}")
    disc = ", ".join(f"{t}:{direct[t]}/{len(prog_local[t])}" for t in tfs)
    print(f"  direct-target disclosure — program genes directly wired to the TF, per row: {disc}")
    print(f"  EMERGENCE (indirect-only, {n_indirect} program genes NOT directly wired): AUC {indirect_auc:.3f}"
          + ("  (too few to be decisive)" if n_indirect < 15 else ""))

    verdict = ("REAL: forcing a master TF ON drives its OWN lineage program specifically — the wiring encodes "
               f"correct reprogramming logic (hit@1 {hit1}/{len(tfs)}, AUC {spec_auc:.2f})." if hit1 >= 0.6 * len(tfs)
               and spec_auc > 0.65 else
               f"PARTIAL: some TFs reprogram correctly (hit@1 {hit1}/{len(tfs)}, AUC {spec_auc:.2f}) but it is not "
               "consistent across lineages." if hit1 >= 0.3 * len(tfs) or spec_auc > 0.6 else
               f"NULL: forcing a TF ON does not specifically induce its lineage program (hit@1 {hit1}/{len(tfs)}, "
               f"AUC {spec_auc:.2f}) — the wiring doesn't encode reprogramming.")
    print("\n" + "=" * 70 + f"\nVERDICT: {verdict}\n" + "=" * 70)

    out = {"n_tfs": len(tfs), "tfs": tfs, "hit_at_1": hit1, "median_rank_own": int(np.median(ranks)),
           "specificity_auc": spec_auc, "indirect_only_auc": indirect_auc, "n_indirect_program_genes": n_indirect,
           "matrix": S, "per_tf": per_tf, "direct_targets": direct, "seeds": list(seeds), "verdict": verdict}
    if save:
        json.dump(out, open(f"{os.path.dirname(__file__)}/../outputs/orphan/grn_reprogram.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
