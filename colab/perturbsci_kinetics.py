"""perturbsci_kinetics -- the definitive time/direction test at the RIGHT timescale: does NASCENT (newly-transcribed, 4sU-labelled) RNA isolate
the DIRECT transcriptional response that the whole/total endpoint buries in the indirect cascade?

Data: PerturbSci-Kinetics (GSE218566, HEK293 CRISPRi, 204 knockdowns incl. NO-TARGET control, ~98k cells). Each cell has a NASCENT and a
WHOLE gene x cell count matrix (nascent = made during the 4sU window; whole = total). If the user's thesis is right -- a fast/direct readout
recovers the direction a steady-state endpoint cannot -- then:

  TEST 1  SELF-KNOCKDOWN is sharper in nascent: the knocked-down gene's own transcription (nascent) should drop MORE than its total (whole)
          pool, because whole still contains pre-existing mRNA. (Clean, works for all KOs; validates nascent captures the DIRECT act.)
  TEST 2  DIRECT-TARGET (regulon) enrichment is higher in NASCENT than WHOLE, for KOs with curated regulons (TRRUST/SIGNOR): nascent movers
          should be more enriched for the KO's direct targets than whole movers.
  TEST 3  FOCUS: nascent response is sparser / more concentrated (direct only) vs whole (accumulated cascade).
If nascent > whole on these, the fast readout recovers the direct/directional signal the endpoint loses -- the one lever that moves the wall.
Deterministic. Streams the .mtx.gz (no full decompression)."""
import os, json, gzip, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
PSK = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/psk")
PRE = "GSM6752591_on_target_"
CTRL = "NO-TARGET"
THR = 0.25


def load_meta():
    """cell_name -> target_gene (KO)."""
    ko = {}
    with gzip.open(PSK / f"{PRE}cell_metadata.csv.gz", "rt") as f:
        hdr = f.readline().rstrip("\n").split(",")
        ci = hdr.index("cell_names"); ti = hdr.index("target_genes")
        for line in f:
            p = line.rstrip("\n").split(",")
            ko[p[ci]] = p[ti]
    return ko


def pseudobulk(kind, ko_of_cell):
    """stream the gene x cell mtx.gz -> per-KO summed counts (genes x KO) + per-KO cell counts. kind in {nascent,whole}."""
    genes = [l.split("\t")[0] for l in gzip.open(PSK / f"{PRE}{kind}_tx.Genes.tsv.gz", "rt").read().splitlines()]
    barc = gzip.open(PSK / f"{PRE}{kind}_tx.Barcodes.tsv.gz", "rt").read().splitlines()
    barc = [b.split("\t")[0] for b in barc]
    kos = sorted(set(ko_of_cell.values()))
    koidx = {k: i for i, k in enumerate(kos)}
    col_ko = np.array([koidx.get(ko_of_cell.get(b, None), -1) for b in barc], dtype=np.int64)
    ncell_ko = np.zeros(len(kos), np.int64)
    for c in col_ko:
        if c >= 0:
            ncell_ko[c] += 1
    S = np.zeros((len(genes), len(kos)), np.float32)
    with gzip.open(PSK / f"{PRE}{kind}_tx_count_matrix.mtx.gz", "rt") as f:
        for line in f:
            if line.startswith("%"):
                continue
            break  # first non-% line = dims header; skip it
        for line in f:
            a = line.split()
            if len(a) < 3:
                continue
            g = int(a[0]) - 1; c = int(a[1]) - 1; v = float(a[2])
            k = col_ko[c]
            if k >= 0:
                S[g, k] += v
    return genes, kos, S, ncell_ko


def cp10k_logfc(S, kos, ctrl_idx):
    """column-normalize to CP10k, log1p, response = KO - control."""
    col = S.sum(0); col[col == 0] = 1.0
    P = np.log1p(S / col[None, :] * 1e4)
    return P - P[:, ctrl_idx:ctrl_idx + 1]


def main():
    if not (PSK / "dl.done").exists():
        print("DATA NOT READY (download still running). Re-run when scratchpad/psk/dl.done exists."); return
    ko_of_cell = load_meta()
    gN, kos, Sn, ncN = pseudobulk("nascent", ko_of_cell)
    gW, kosW, Sw, ncW = pseudobulk("whole", ko_of_cell)
    assert kos == kosW
    ci = kos.index(CTRL)
    Rn = cp10k_logfc(Sn, kos, ci); Rw = cp10k_logfc(Sw, kos, ci)
    gnidx = {g: i for i, g in enumerate(gN)}; gwidx = {g: i for i, g in enumerate(gW)}
    perturbed = [k for k in kos if k != CTRL and ncN[kos.index(k)] >= 20]
    print(f"loaded {len(kos)} groups ({len(perturbed)} KOs with >=20 cells), {len(gN)} nascent genes / {len(gW)} whole genes\n")

    # TEST 1: self-knockdown, nascent vs whole
    sn, sw = [], []
    for k in perturbed:
        ki = kos.index(k)
        if k in gnidx:
            sn.append(float(Rn[gnidx[k], ki]))
        if k in gwidx:
            sw.append(float(Rw[gwidx[k], ki]))
    print(f"TEST 1 self-knockdown logFC (more negative = stronger direct repression):")
    print(f"    NASCENT self-drop {np.mean(sn):+.3f}    WHOLE self-drop {np.mean(sw):+.3f}   (n={len(sn)} KOs)")

    # TEST 2: direct-target (regulon) enrichment among movers, nascent vs whole
    Hk = Harness("K562"); reg, _, sources = build_causal(Hk)
    reg_t = {s: set(t for t, _ in ts) for s, ts in reg.items()}
    def enrich(R, genes, gidx, k):
        ki = kos.index(k); r = R[:, ki]
        movers = set(genes[i] for i in np.where(np.abs(r) > THR)[0])
        tgt_meas = reg_t.get(k, set()) & set(gidx)
        if not tgt_meas or not movers:
            return None
        hit = len(movers & tgt_meas) / len(movers); base = len(tgt_meas) / len(gidx)
        return hit / base if base > 0 else None
    en, ew = [], []
    reg_kos = [k for k in perturbed if k in reg_t and (reg_t[k] & set(gnidx))]
    for k in reg_kos:
        a = enrich(Rn, gN, gnidx, k); b = enrich(Rw, gW, gwidx, k)
        if a is not None and b is not None:
            en.append(a); ew.append(b)
    print(f"\nTEST 2 direct-target (regulon) enrichment among movers ({len(en)} KOs with regulons):")
    print(f"    NASCENT {np.mean(en):.2f}x    WHOLE {np.mean(ew):.2f}x" if en else "    (no KOs with curated regulons in this panel)")

    # TEST 3: focus (n movers), nascent vs whole
    nm_n = [int((np.abs(Rn[:, kos.index(k)]) > THR).sum()) for k in perturbed]
    nm_w = [int((np.abs(Rw[:, kos.index(k)]) > THR).sum()) for k in perturbed]
    print(f"\nTEST 3 movers per KO: NASCENT {np.mean(nm_n):.0f}   WHOLE {np.mean(nm_w):.0f}")

    self_sharper = np.mean(sn) < np.mean(sw) - 0.02
    reg_better = bool(en) and np.mean(en) > np.mean(ew) + 0.2
    verdict = (
        f"PERTURBSCI-KINETICS (perturbsci_kinetics.py): the fast/direct readout test at the right timescale (GSE218566, HEK293 CRISPRi, "
        f"{len(perturbed)} knockdowns, 4sU NASCENT vs WHOLE RNA). TEST 1 self-knockdown: nascent {np.mean(sn):+.2f} vs whole {np.mean(sw):+.2f} "
        + ("-- nascent captures the DIRECT repression more sharply (as it must: whole still holds pre-existing mRNA). " if self_sharper else
           "-- nascent is NOT sharper than whole on self-knockdown here. ")
        + (f"TEST 2 direct-target enrichment: nascent {np.mean(en):.2f}x vs whole {np.mean(ew):.2f}x " if en else "TEST 2: no curated regulons in this metabolic/chromatin panel. ")
        + (f"-- nascent isolates the direct regulon BETTER, so the fast readout recovers direction the endpoint buries. " if reg_better else
           ("-- nascent does not clearly beat whole on direct-target enrichment. " if en else ""))
        + f"TEST 3 focus: nascent {np.mean(nm_n):.0f} movers/KO vs whole {np.mean(nm_w):.0f}. "
        + ("VERDICT: the nascent/fast readout DOES sharpen the direct signal -- evidence the timescale is the lever, as the direction/time thesis "
           "predicted." if (self_sharper and (reg_better or not en)) else
           "VERDICT: even the nascent 4sU readout does not cleanly separate direct from indirect on this panel -- bounding the thesis further.")
        + " (Scope: HEK293 CRISPRi, 204 mostly-metabolic/chromatin targets; a single labelling window, not a dense hours-course.) Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_kos": len(perturbed), "self_kd_nascent": round(float(np.mean(sn)), 4), "self_kd_whole": round(float(np.mean(sw)), 4),
               "regulon_enrich_nascent": round(float(np.mean(en)), 3) if en else None,
               "regulon_enrich_whole": round(float(np.mean(ew)), 3) if en else None, "n_reg_kos": len(en),
               "movers_nascent": round(float(np.mean(nm_n)), 1), "movers_whole": round(float(np.mean(nm_w)), 1),
               "verdict": verdict, "note": verdict}, open(OUT / "perturbsci_kinetics.json", "w"), indent=1)
    print("\n  -> outputs/orphan/perturbsci_kinetics.json")


if __name__ == "__main__":
    main()
