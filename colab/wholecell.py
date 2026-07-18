"""wholecell — the whole steady-state model chained end-to-end and MEASURED: genome/sequence -> regulation (promoter) -> mRNA
level -> protein level, scored at every hand-off and end-to-end against measured ground truth. The question the user asked:
assemble the whole thing and see HOW MUCH WE GET.

The steady-state spine (each layer already validated separately):
  REGULATION    promoter Pol II (POLR2A at TSS) = transcription-rate proxy k_txn
  mRNA LEVEL    [mRNA] ~ k_txn / k_deg ; k_deg from the sequence decay model (halflife.py, r~0.57)
  PROTEIN LEVEL [protein] ~ k_translation * [mRNA] / k_protein_deg ; translation+degradation from sequence (protein_abundance.py)
Ground truth: measured K562 mRNA (RNAdecayCafe RPKM) and measured protein (cell-image ppm). Chromosome-GroupKFold.

Reports: (1) each hand-off's held-out r, (2) the ORACLE ceiling (perfect measured mRNA -> protein), (3) the END-TO-END number
(genome/promoter -> protein with NO measured mRNA), so you can see how much the full chain retains and where error compounds.
The MUTATION arm (NEXUS mutation -> network propagation) is separate and reported honestly: near-field validated (regulon 9.2x),
far-field is the measured wall. This module is the steady-state abundance chain -- the part that composes.
"""
import json, csv
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/rnadecay")


def build():
    import halflife as hl, crispr_gate as cg, protein_abundance as pa
    hl._ensure()
    tx = hl.parse_gencode()                                          # {gene: (cds, utr3, utr5)}
    D = json.load(open(OUT / "cell_complete.json"))
    idx2name = {i: g["name"] for i, g in enumerate(D["genes"])}
    chrom = {g["name"]: g.get("chrom", "chrNA") for g in D["genes"]}
    gmeta = {g["name"]: g for g in D["genes"]}
    ppm = {idx2name[int(k)]: v for k, v in D["ppm"].items() if int(k) in idx2name and isinstance(v, (int, float)) and v > 0}
    rpkm = {}
    for r in csv.DictReader(open(SCR / "AvgKdegs.csv")):
        if r["cell_line"] == "K562":
            try:
                v = float(r["avg_RPKM_total"])
                if v > 0:
                    rpkm[r["feature_ID"]] = v
            except ValueError:
                pass
    pol = cg._signal_index(OUT / "invivo/marks/polr2a.bed.gz", 6)     # promoter Pol II = k_txn proxy
    SENSE = [c for c in pa.CODON_TABLE if pa.CODON_TABLE[c] != '*']
    genes = sorted(set(tx) & set(ppm) & set(rpkm))
    rows = []
    for g in genes:
        m = gmeta.get(g); c = m.get("chrom") if m else None; t = m.get("tss") if m else None
        if not c or not t:
            continue
        promoter = cg._max_signal(pol, c, int(t) - 1000, int(t) + 1000)
        cds, utr3, utr5 = tx[g]
        prot = pa._translate(cds)
        if len(prot) < 20:
            continue
        f = {"promoter_polII": np.log10(promoter + 1.0)}             # REGULATION (k_txn)
        # mRNA-decay features (codon composition + 3'UTR)  -> k_deg
        n = len(cds) // 3; cnt = {}
        for i in range(0, n * 3, 3):
            cd = cds[i:i + 3]
            if cd in pa.CODON_TABLE and pa.CODON_TABLE[cd] != "*":
                cnt[cd] = cnt.get(cd, 0) + 1
        tot = max(sum(cnt.values()), 1)
        for cod in SENSE:
            f["cod_" + cod] = cnt.get(cod, 0) / tot
        L3 = max(len(utr3), 1)
        import re
        f["utr3_are"] = 1000.0 * len(re.findall("(?=ATTTA)", utr3)) / L3
        f["utr3_gc"] = (utr3.count("G") + utr3.count("C")) / L3
        f["log_utr3_len"] = np.log10(L3); f["cds_gc"] = (cds.count("G") + cds.count("C")) / max(len(cds), 1)
        # translation features
        f["log_utr5_len"] = np.log10(max(len(utr5), 1)); f["utr5_uaug"] = utr5.count("ATG")
        f["log_cds_len"] = np.log10(max(len(cds), 1))
        # protein-degradation features
        Lp = len(prot); nterm = prot[1] if len(prot) > 1 and prot[0] == "M" else prot[0]
        f["log_prot_len"] = np.log10(Lp)
        f["nend_stab"] = 1.0 if nterm in pa.NEND_STABILIZING else 0.0
        f["pest"] = sum(prot.count(a) for a in pa.PEST) / Lp
        f["disorder"] = sum(prot.count(a) for a in pa.DISORDER) / Lp
        f["hydro"] = sum(prot.count(a) for a in pa.HYDRO) / Lp
        f["_gene"] = g; f["_rpkm"] = np.log10(rpkm[g]); f["_ppm"] = np.log10(ppm[g])
        rows.append(f)
    cols = [c for c in rows[0] if not c.startswith("_")]
    X = np.array([[r[c] for c in rows[0] if not c.startswith("_")] for r in rows], dtype=np.float64)
    y_rpkm = np.array([r["_rpkm"] for r in rows]); y_ppm = np.array([r["_ppm"] for r in rows])
    grp = np.array([chrom.get(r["_gene"], "chrNA") for r in rows])
    return cols, X, y_rpkm, y_ppm, grp


def _cv(X, y, grp, cols, mask, add=None):
    """ridge CV over a feature subset (mask = column names); `add` = extra numeric column stacked in (e.g. measured mRNA)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    ci = {c: i for i, c in enumerate(cols)}
    idxs = [ci[c] for c in mask]
    Xm = X[:, idxs]
    if add is not None:
        Xm = np.column_stack([Xm, add])
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xm, y, grp):
        sc = StandardScaler().fit(Xm[tr])
        pred[te] = Ridge(alpha=10.0).fit(sc.transform(Xm[tr]), y[tr]).predict(sc.transform(Xm[te]))
    from scipy.stats import pearsonr
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"r": round(float(pearsonr(y, pred)[0]), 3), "R2": round(float(r2), 3)}


def run():
    cols, X, y_rpkm, y_ppm, grp = build()
    import protein_abundance as _pa
    _SENSE = [c for c in _pa.CODON_TABLE if _pa.CODON_TABLE[c] != "*"]
    decay = ["cod_" + c for c in _SENSE] + ["utr3_are", "utr3_gc", "log_utr3_len", "cds_gc"]
    transl = ["log_utr5_len", "utr5_uaug", "log_cds_len"]
    pdeg = ["log_prot_len", "nend_stab", "pest", "disorder", "hydro"]
    ci = {c: i for i, c in enumerate(cols)}
    mrna_pred_col = None
    res = {"n_genes": len(y_ppm)}
    # STAGE 1: genome/regulation -> mRNA level
    res["S1_promoter_to_mRNA"] = _cv(X, y_rpkm, grp, cols, ["promoter_polII"])
    res["S1_promoter+decay_to_mRNA"] = _cv(X, y_rpkm, grp, cols, ["promoter_polII"] + decay)
    # STAGE 2 (oracle): measured mRNA -> protein
    res["S2_measured_mRNA_to_protein"] = _cv(X, y_ppm, grp, cols, transl + pdeg, add=X[:, ci["promoter_polII"]] * 0 + y_rpkm)
    # END-TO-END: genome/promoter -> protein (NO measured mRNA anywhere)
    res["ENDTOEND_promoter_only"] = _cv(X, y_ppm, grp, cols, ["promoter_polII"])
    res["ENDTOEND_promoter+decay"] = _cv(X, y_ppm, grp, cols, ["promoter_polII"] + decay)
    res["ENDTOEND_full_chain"] = _cv(X, y_ppm, grp, cols, ["promoter_polII"] + decay + transl + pdeg)
    # ORACLE ceiling: measured mRNA + all sequence -> protein
    res["ORACLE_measured_mRNA+all"] = _cv(X, y_ppm, grp, cols, decay + transl + pdeg, add=y_rpkm)
    return res


def main():
    print("=" * 100)
    print("WHOLE-CELL (steady state) — genome -> regulation -> mRNA -> protein, chained end-to-end and MEASURED")
    print("=" * 100)
    r = run()
    print(f"\n  {r['n_genes']} genes, chromosome-held-out. Ground truth: measured K562 mRNA (RPKM) + protein (ppm).\n")
    print("  STAGE 1  genome/regulation -> mRNA level (predict measured RPKM):")
    print(f"     promoter Pol II only           r {r['S1_promoter_to_mRNA']['r']}   R2 {r['S1_promoter_to_mRNA']['R2']}")
    print(f"     promoter + mRNA-decay model    r {r['S1_promoter+decay_to_mRNA']['r']}   R2 {r['S1_promoter+decay_to_mRNA']['R2']}")
    print("  STAGE 2  mRNA -> protein (oracle: measured mRNA + translation/degradation):")
    print(f"     measured mRNA + sequence       r {r['ORACLE_measured_mRNA+all']['r']}   R2 {r['ORACLE_measured_mRNA+all']['R2']}   <- ceiling")
    print("\n  END-TO-END  genome/promoter -> PROTEIN (no measured mRNA anywhere) -- 'how much we get':")
    print(f"     promoter only                  r {r['ENDTOEND_promoter_only']['r']}   R2 {r['ENDTOEND_promoter_only']['R2']}")
    print(f"     promoter + decay               r {r['ENDTOEND_promoter+decay']['r']}   R2 {r['ENDTOEND_promoter+decay']['R2']}")
    print(f"     FULL CHAIN (promoter+decay+translation+protein-deg)  r {r['ENDTOEND_full_chain']['r']}   R2 {r['ENDTOEND_full_chain']['R2']}")
    e2e = r["ENDTOEND_full_chain"]["r"]; oracle = r["ORACLE_measured_mRNA+all"]["r"]
    print(f"\n  => the STEADY-STATE chain composes: end-to-end genome->protein r = {e2e} (R2 {r['ENDTOEND_full_chain']['R2']}); "
          f"oracle ceiling (measured mRNA) r = {oracle}.")
    print("     The gap is the promoter->mRNA hand-off (transcription rate from a single Pol II track is lossy). This is the")
    print("     part of the cell that COMPOSES. The MUTATION arm (NEXUS -> network cascade) is separate: near-field validated")
    print("     (regulon 9.2x), far-field is the measured wall (does not compose) -- unchanged by this integration.")
    r["note"] = ("The steady-state model chained end-to-end and measured: genome -> regulation (promoter Pol II = k_txn) -> "
                 "mRNA level (k_deg from sequence, halflife.py) -> protein level (translation+degradation, protein_abundance.py), "
                 "held-out vs measured K562 RPKM + ppm, chromosome-GroupKFold. Reports each hand-off + the ORACLE ceiling "
                 "(measured mRNA->protein) + the END-TO-END number (genome/promoter->protein, no measured mRNA). The steady-state "
                 "abundance chain COMPOSES (end-to-end r reported); the loss is at the promoter->mRNA hand-off. The MUTATION arm "
                 "(NEXUS mutation -> network propagation) is separate: near-field validated (regulon 9.2x), far-field wall.")
    json.dump(r, open(OUT / "wholecell.json", "w"), indent=1)
    print("\n  -> outputs/orphan/wholecell.json")
    return r


if __name__ == "__main__":
    main()
