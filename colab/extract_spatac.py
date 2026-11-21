"""extract_spatac -- one-time converter: Spear-ATAC (GSE168851, K562 CRISPRi + scATAC) chromVAR MOTIF-deviation SummarizedExperiment (.rds)
-> a compact npz. chromVAR 'z' = the bias-corrected TF-motif accessibility z-score per cell = a direct read-out of each TF's REGULATORY
ACTIVITY (how open its binding sites are), independent of that TF's mRNA level. We pseudobulk the per-cell z by sgRNA target group and keep
mean/std/n so the analysis module can test, per motif, whether a knockdown changed a TF's chromatin activity -- the 'activity' layer mRNA can't
see. Deterministic. Usage: python extract_spatac.py <pilot|large>."""
import sys, warnings, collections
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import os
import rdata
from scipy.sparse import csc_matrix

SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")) / "spatac"


def target_of(sg):
    """sgRNA label -> knocked-down TF, or 'NT' for non-targeting control. Handles 'sgGATA1-2', 'sgNT-1', and double-prefixed 'sgsgNT-2'."""
    import re
    s = str(sg)
    if s in ("None", "UNK", ""):
        return None
    core = s
    while core.startswith("sg"):                                  # strip repeated lowercase 'sg' guide prefix (leaves uppercase gene names)
        core = core[2:]
    core = re.sub(r"-\d+$", "", core)                             # strip trailing -<guide index>
    if core.upper().startswith("NT") or core.upper() in ("NEG", "CTRL", "NEGATIVE"):
        return "NT"
    return core


def gene_of_motif(m):
    """'GATA1_591#GATA:C2H2:242:0' -> 'GATA1'; 'MAX+MYC_1009#...' -> 'MAX+MYC' (the TF(s) the motif represents; '+' = dimer)."""
    import re
    pre = str(m).split("#")[0]                                    # 'GATA1_591' or 'MAX+MYC_1009'
    return re.sub(r"_\d+$", "", pre).upper()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    rds = SP / (f"{which}_motif.rds")
    conv = rdata.read_rds(str(rds))
    zobj = conv.assays.data.listData["z"]                          # dgCMatrix: motifs x cells (chromVAR z = TF activity)
    dim = np.asarray(zobj.Dim); nmot, ncell = int(dim[0]), int(dim[1])
    Z = csc_matrix((zobj.x, zobj.i, zobj.p), shape=(nmot, ncell)).toarray().astype(np.float32)  # fully populated
    motif_names = np.asarray(conv.NAMES)
    motif_genes = np.array([gene_of_motif(m) for m in motif_names])
    cd = conv.colData.listData
    sg = np.asarray(cd["sgAssign"]).astype(str)                    # per-cell best guide
    tgt = np.array([target_of(x) or "" for x in sg])
    groups = sorted(set(t for t in tgt if t) )
    print(f"[{which}] {nmot} motifs x {ncell} cells; groups: " + ", ".join(f"{g}({int((tgt==g).sum())})" for g in groups))

    mean = np.zeros((len(groups), nmot), np.float32); std = np.zeros_like(mean); n = np.zeros(len(groups), np.int32)
    for gi, g in enumerate(groups):
        cols = np.where(tgt == g)[0]
        n[gi] = len(cols)
        if len(cols):
            sub = Z[:, cols]; mean[gi] = sub.mean(1); std[gi] = sub.std(1)
    out = SP / f"{which}_motif_pb.npz"
    np.savez_compressed(out, motif_names=motif_names, motif_genes=motif_genes,
                        groups=np.array(groups), mean=mean, std=std, n=n)
    print(f"  -> {out}  (groups={len(groups)}, motifs={nmot})")
    # quick sanity: GATA1 motif activity, GATA1-KD vs NT
    gi = {g: i for i, g in enumerate(groups)}
    if "GATA1" in gi and "NT" in gi:
        mi = np.where(motif_genes == "GATA1")[0]
        for j in mi[:1]:
            d = mean[gi["GATA1"], j] - mean[gi["NT"], j]
            print(f"  sanity GATA1 motif ({motif_names[j]}): KD z={mean[gi['GATA1'],j]:.2f} vs NT z={mean[gi['NT'],j]:.2f}  delta={d:+.2f}")


if __name__ == "__main__":
    main()
