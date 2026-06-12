"""Per-gene SEQUENCE analysis of essentials vs non-essentials.

Naresh: "analyse the sequence of each essential gene itself."

Pure protein-sequence-level features computed FROM THE AMINO-ACID SEQUENCE
of each gene -- not from conservation, not from network, not from any
annotation. Just: what does the protein LOOK like?

Per-protein features (no external libs beyond pandas+numpy):
  length                    aa count
  gravy                     Kyte-Doolittle hydrophobicity (mean per residue)
  aromaticity               F+W+Y fraction
  pI_approx                 isoelectric point from titratable residue counts
  charge_pH7                net charge at neutral pH
  cys_frac                  cysteine fraction (disulfide/oxidation signal)
  trp_frac                  tryptophan fraction (expensive aa, rare in highly
                            expressed proteins -- counter-intuitive marker)
  met_frac                  methionine fraction
  low_complexity_frac       fraction in 10-aa windows with <5 distinct aa
  n_TM_helices_pred         crude prediction: count of 19+ consecutive
                            residues with avg hydrophobicity > 1.6
  n_signal_pep              N-terminal signal-peptide-like region present
                            (signal:  positively-charged N-term + hydrophobic
                            core in residues 5-25 + small residue at cleavage)

Then compare essential vs non-essential per organism AND pooled:
  - mean, Cohen's d, single-feature AUC

ORG COVERAGE:
  - 7 DEG benchmark orgs in repo labels: mgen, mpne, mtub, reut, spne19F,
    spneT4, syn3a (clean labels, ~14k genes)
  - sequences extracted from data/drive_import/genome_cache/<accession>/
    protein.faa where the manifest maps org -> accession

OUTPUT: outputs/essential_sequence_analysis.md
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "memory_bank" / "data" / "multiorg_essentiality" / "labels.csv"
SEQ_FASTA = REPO / "memory_bank" / "data" / "multiorg_essentiality" / "sequences.fasta"
MANIFEST = REPO / "data" / "drive_import" / "labels" / "genome_cache_manifest.csv"
CACHE = REPO / "data" / "drive_import" / "genome_cache"

# Kyte-Doolittle hydrophobicity scale
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
# approximate pKa for ionizable groups (Lehninger / EMBOSS-style)
PKA = {"C":8.5,"D":3.65,"E":4.25,"H":6.0,"K":10.5,"R":12.5,"Y":10.07,
       "N_term":9.0,"C_term":2.0}


def parse_faa(path: Path):
    """Yield (header_id, sequence). header_id = first whitespace-separated token."""
    cur_id, cur_seq = None, []
    with open(path) as f:
        for ln in f:
            if ln.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(cur_seq).upper().replace("*", "")
                cur_id = ln[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(ln.strip())
        if cur_id is not None:
            yield cur_id, "".join(cur_seq).upper().replace("*", "")


def features(seq: str) -> dict:
    n = len(seq)
    if n == 0: return {}
    counts = {a: seq.count(a) for a in "ACDEFGHIKLMNPQRSTVWY"}
    # GRAVY
    gravy = sum(KD.get(a, 0.0) * c for a, c in counts.items()) / n
    # aromaticity
    arom = (counts["F"] + counts["W"] + counts["Y"]) / n
    # charge at pH 7 (positive aa - negative aa, ignoring N/C termini for simplicity)
    pos = counts["K"] + counts["R"] + 0.5 * counts["H"]
    neg = counts["D"] + counts["E"]
    charge = pos - neg
    # approximate pI by binary search on net charge
    def net_q(pH):
        q = 0.0
        # N-term +1 below pKa
        q += 1.0 / (1.0 + 10 ** (pH - PKA["N_term"]))
        # C-term -1 above pKa
        q -= 1.0 / (1.0 + 10 ** (PKA["C_term"] - pH))
        for aa, n_aa in counts.items():
            if n_aa == 0: continue
            if aa in ("K", "R", "H"):
                q += n_aa / (1.0 + 10 ** (pH - PKA[aa]))
            elif aa in ("D", "E", "C", "Y"):
                q -= n_aa / (1.0 + 10 ** (PKA[aa] - pH))
        return q
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if net_q(mid) > 0: lo = mid
        else:              hi = mid
    pI = (lo + hi) / 2
    # low-complexity: fraction of 10-aa windows with <5 distinct aa
    if n >= 10:
        windows = sum(1 for i in range(n - 9) if len(set(seq[i:i+10])) < 5)
        lc = windows / (n - 9)
    else:
        lc = 0.0
    # crude TM-helix prediction: count maximal runs of 19+ residues
    # with mean KD > 1.6
    tm = 0; i = 0
    while i < n - 18:
        window = seq[i:i+19]
        kd = sum(KD.get(a, 0.0) for a in window) / 19.0
        if kd > 1.6:
            tm += 1; i += 19   # advance past this helix
        else:
            i += 1
    # signal peptide proxy: first 30 residues, positively-charged N-term
    # (charge of first 5 > 0) + hydrophobic 6-20 (mean KD > 1.5)
    sig = 0
    if n >= 25:
        n_term = seq[:5]
        q_nterm = sum(1 for a in n_term if a in "KR")
        core = seq[5:20]
        kd_core = sum(KD.get(a, 0.0) for a in core) / 15.0
        if q_nterm >= 1 and kd_core > 1.5:
            sig = 1
    return {
        "length": n, "gravy": gravy, "aromaticity": arom,
        "charge_pH7": charge, "pI_approx": pI,
        "cys_frac": counts["C"] / n, "trp_frac": counts["W"] / n,
        "met_frac": counts["M"] / n, "low_complexity_frac": lc,
        "n_TM_helices_pred": tm, "has_signal_pep": sig,
    }


def cohens_d(a, b):
    import numpy as np
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    pooled = (((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
              / (len(a)+len(b)-2)) ** 0.5
    return (a.mean() - b.mean()) / pooled if pooled else float("nan")


def auc(values, labels):
    import numpy as np
    import pandas as pd
    m = ~np.isnan(values)
    v, y = values[m], labels[m]
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos < 2 or n_neg < 2: return float("nan")
    r = pd.Series(v).rank(method="average").values
    U = r[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)


def main() -> int:
    import pandas as pd
    import numpy as np

    print("Loading labels + repo sequences.fasta ...")
    lab = pd.read_csv(LABELS)
    # parse the repo's pre-bridged FASTA: headers are 'org|locus_tag|gene_name'
    if not SEQ_FASTA.exists():
        print(f"ERROR: {SEQ_FASTA} not found", file=sys.stderr); return 1
    seqs_by_key = {}
    for hdr, seq in parse_faa(SEQ_FASTA):
        seqs_by_key[hdr] = seq
    print(f"  fasta entries: {len(seqs_by_key):,}")
    sample = list(seqs_by_key.keys())[:2]
    print(f"  header format examples: {sample}")
    # build (organism, locus_tag) -> seq map
    seq_lookup = {}
    for hdr, seq in seqs_by_key.items():
        parts = hdr.split("|")
        if len(parts) >= 2:
            seq_lookup[(parts[0], parts[1])] = seq

    print(f"\nExtracting per-protein features ...")
    all_feats = []
    for org in sorted(lab.organism.unique()):
        org_lab = lab[lab.organism == org]
        rows = []; matched = 0
        for _, r in org_lab.iterrows():
            seq = seq_lookup.get((org, str(r.locus_tag)))
            if seq is None: continue
            f = features(seq); f["organism"] = org
            f["locus_tag"] = str(r.locus_tag); f["essential"] = int(r.essential)
            f["gene_name"] = r.gene_name
            rows.append(f); matched += 1
        print(f"  {org:8s} labels {len(org_lab):>5d}  matched {matched:>5d}  "
              f"({matched/max(len(org_lab),1)*100:>3.0f}%)")
        all_feats.extend(rows)

    df = pd.DataFrame(all_feats)
    print(f"\nTotal: {len(df):,} proteins from {df.organism.nunique()} organisms"
          f"   essential={int(df.essential.sum()):,}  "
          f"non={int((df.essential==0).sum()):,}")
    if len(df) < 200:
        print("ERROR: too few matched. Check locus-tag conventions.",
              file=sys.stderr); return 1

    feats = ["length","gravy","aromaticity","charge_pH7","pI_approx",
             "cys_frac","trp_frac","met_frac","low_complexity_frac",
             "n_TM_helices_pred","has_signal_pep"]

    # ---- POOLED essential vs non-essential ----
    ess_df = df[df.essential == 1]; non_df = df[df.essential == 0]
    print(f"\n=== POOLED: essential vs non-essential per-protein features ===")
    hd = "d"
    print(f"  {'feature':<22s} {'ess mean':>10s} {'non mean':>10s} "
          f"{hd:>7s} {'AUC':>6s}")
    pooled_rows = []
    for f in feats:
        ea = pd.to_numeric(ess_df[f], errors="coerce").values.astype(float)
        na = pd.to_numeric(non_df[f], errors="coerce").values.astype(float)
        v  = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        d = cohens_d(ea, na)
        a = auc(v, df.essential.values.astype(int))
        em = float(np.nanmean(ea)); nm = float(np.nanmean(na))
        print(f"  {f:<22s} {em:>10.3f} {nm:>10.3f} {d:>+7.3f} {a:>6.3f}")
        pooled_rows.append((f, em, nm, d, a))

    # ---- per-organism (the within-organism control) ----
    print(f"\n=== WITHIN each organism (no cross-org confound) ===")
    print(f"  {'feature':<22s} " + "".join(f"{o[:7]:>10s}" for o in
                                            sorted(df.organism.unique())))
    per_org_rows = []
    for f in feats:
        cells = []
        row = [f]
        for org in sorted(df.organism.unique()):
            sub = df[df.organism == org]
            ea = pd.to_numeric(sub[sub.essential==1][f], errors="coerce").values.astype(float)
            na = pd.to_numeric(sub[sub.essential==0][f], errors="coerce").values.astype(float)
            v = pd.to_numeric(sub[f], errors="coerce").values.astype(float)
            y = sub.essential.values.astype(int)
            if (y==1).sum() >= 5 and (y==0).sum() >= 5:
                a = auc(v, y)
                cells.append(f"{a:>10.3f}"); row.append(a)
            else:
                cells.append(f"{'-':>10s}"); row.append(None)
        print(f"  {f:<22s} " + "".join(cells))
        per_org_rows.append(row)

    # ---- per-gene table written for further analysis ----
    out_csv = REPO / "outputs" / "essential_sequence_features.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df):,} per-gene rows)")

    # markdown report
    lines = [f"# Per-protein sequence analysis: essential vs non-essential\n",
             f"**{len(df):,} proteins** from "
             f"{df.organism.nunique()} DEG-benchmark organisms ("
             f"{', '.join(sorted(df.organism.unique()))}), "
             f"{int(df.essential.sum()):,} essential vs "
             f"{int((df.essential==0).sum()):,} non-essential.\n",
             "All features computed DIRECTLY FROM THE AMINO-ACID SEQUENCE, "
             "no annotation or conservation used.\n",
             "## Pooled comparison\n",
             "| feature | essential mean | non-essential mean | Cohen's d | AUC |",
             "|---|---:|---:|---:|---:|"]
    for f, em, nm, d, a in sorted(pooled_rows, key=lambda r: -abs(r[4]-0.5)):
        lines.append(f"| {f} | {em:.3f} | {nm:.3f} | {d:+.3f} | {a:.3f} |")
    lines.append("\nReading: |d| > 0.2 small, > 0.5 medium, > 0.8 large. "
                 "AUC = single-feature essentiality discrimination "
                 "(0.5 = useless, 1.0 = perfect).\n")
    out_md = REPO / "outputs" / "essential_sequence_analysis.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
