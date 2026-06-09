"""Infer the function of syn3.0's 'Unclear' essential genes -- entirely
in-sandbox, no network, using data already in the repo.

Three independent evidence lines per unknown gene:
  1. HOMOLOGY  -- k-mer prefilter + Smith-Waterman (BLOSUM62) against
                  the annotated M. genitalium proteome (KO + EC numbers
                  in its FASTA headers) and the annotated syn3a genes.
                  A hit transfers the function.
  2. CONTEXT   -- genomic neighbours (operon): in a minimal genome,
                  an unknown essential flanked by known genes likely
                  acts in their pathway (guilt by association).
  3. BIOPHYSICS-- precomputed TM-helix count, signal peptide, pI,
                  length, ribosomal/synthetase flags -> coarse role
                  (membrane transporter / secreted / cytoplasmic enzyme).

Output: outputs/syn3a_function_inference.md
"""
from __future__ import annotations
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------- compact BLOSUM62 ----------
_B62 = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_aa = _B62.split("\n")[1].split()
BL = {}
for line in _B62.strip().split("\n")[1:]:
    p = line.split(); r = p[0]
    for c, v in zip(_aa, p[1:]):
        BL[(r, c)] = int(v)
def score(a, b): return BL.get((a, b), -4)


def smith_waterman(s1, s2, gap=-6):
    """Local alignment max score + identity over aligned region. O(n*m)."""
    n, m = len(s1), len(s2)
    if n == 0 or m == 0: return 0, 0.0
    prev = [0]*(m+1); best = 0; best_ij = (0,0)
    H = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        a = s1[i-1]
        Hi = H[i]; Hp = H[i-1]
        for jj in range(1, m+1):
            d = Hp[jj-1] + score(a, s2[jj-1])
            v = max(0, d, Hp[jj]+gap, Hi[jj-1]+gap)
            Hi[jj] = v
            if v > best: best = v; best_ij = (i, jj)
    # traceback for identity
    i, j = best_ij; matches = aligned = 0
    while i > 0 and j > 0 and H[i][j] > 0:
        if H[i][j] == H[i-1][j-1] + score(s1[i-1], s2[j-1]):
            aligned += 1
            if s1[i-1] == s2[j-1]: matches += 1
            i -= 1; j -= 1
        elif H[i][j] == H[i-1][j] + gap: i -= 1
        else: j -= 1
    ident = matches/aligned if aligned else 0.0
    return best, ident


def parse_fasta(path):
    out = {}; cur = None; buf = []; hdr = {}
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None: out[cur] = ("".join(buf), hdr_line)
                hdr_line = line[1:].strip(); cur = hdr_line.split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cur is not None: out[cur] = ("".join(buf), hdr_line)
    return out


def kmers(seq, k=4):
    return set(seq[i:i+k] for i in range(len(seq)-k+1))


def main():
    import pandas as pd
    # ---- unclear essential genes ----
    s = pd.read_csv(REPO/"memory_bank/data/syn3a_essentiality_breuer2019.csv")
    gt = pd.read_csv(REPO/"memory_bank/data/syn3a_gene_table.csv")
    feat = pd.read_csv(REPO/"memory_bank/data/syn3a_gene_class_features.csv")
    ess = s[s.essentiality.str.lower()=="essential"]
    unclear = ess[ess.primary_function.str.contains("nclear|nknown",case=False,na=False)]

    # ---- syn3a protein sequences ----
    syn_fa = parse_fasta(REPO/"memory_bank/data/multiorg_essentiality/sequences.fasta")
    # keys look like syn3a|JCVISYN3A_0001|dnaA
    syn_seq = {}
    for k,(seq,hdr) in syn_fa.items():
        parts = k.split("|")
        if len(parts)>=2: syn_seq[parts[1]] = seq

    # ---- reference 1: M. genitalium proteome (KO + EC in headers) ----
    mgen = parse_fasta(REPO/"memory_bank/data/multiorg_essentiality/_xs_cache/protein_sequence.fasta")
    ref = []  # (id, seq, annotation)
    for k,(seq,hdr) in mgen.items():
        # >mge:MG_001 K02338 DNA polymerase III subunit beta [EC:2.7.7.7] | (GenBank) dnaN; ...
        m = re.match(r"(\S+)\s+(K\d{5}|no KO assigned)\s+(.*?)(\s*\|\s*.*)?$", hdr)
        if m:
            ko = m.group(2); func = m.group(3).strip()
        else:
            ko = ""; func = hdr
        if seq and "uncharacter" not in func.lower() and "hypothetical" not in func.lower():
            ref.append((k, seq, f"{func}" + (f"  [{ko}]" if ko.startswith('K') else "")))
    # ---- reference 2: annotated syn3a genes (known function) ----
    known = ess[~ess.primary_function.str.contains("nclear|nknown",case=False,na=False)]
    prod_map = dict(zip(s.locus_tag, s.gene_product))
    for lt in known.locus_tag:
        if lt in syn_seq:
            ref.append((f"syn3a:{lt}", syn_seq[lt], str(prod_map.get(lt,""))+" (syn3a)"))
    print(f"reference proteins (annotated): {len(ref)}")

    # k-mer index for prefilter
    ref_km = [(rid, rseq, rann, kmers(rseq)) for rid,rseq,rann in ref]

    # genomic neighbours
    gt_sorted = gt.sort_values("start_1based").reset_index(drop=True)
    pos = {r.locus_tag: i for i,r in gt_sorted.iterrows()}
    prod_all = dict(zip(gt.locus_tag, gt["product"]))
    strand_all = dict(zip(gt.locus_tag, gt.strand))
    feat_idx = feat.set_index("locus_tag")

    md = ["# Function inference for syn3.0's 29 'Unclear' essential genes\n\n",
          "Three independent evidence lines: homology (Smith-Waterman vs annotated "
          "M. genitalium + syn3a proteins), genomic context (operon neighbours), "
          "and biophysical signature.\n\n"]
    results = []
    for _, row in unclear.iterrows():
        lt = row.locus_tag
        seq = syn_seq.get(lt, "")
        hint = str(row.gene_product)
        # ---- homology ----
        best = None
        if seq:
            qk = kmers(seq)
            cand = sorted(((len(qk & rk), rid, rseq, rann)
                           for rid,rseq,rann,rk in ref_km),
                          reverse=True)[:8]
            scored = []
            for shared, rid, rseq, rann in cand:
                if shared < 3: continue
                sc, ident = smith_waterman(seq, rseq)
                scored.append((sc, ident, rid, rann, shared))
            scored.sort(reverse=True)
            if scored: best = scored[0]
        # ---- context ----
        ctx = []
        if lt in pos:
            i = pos[lt]
            for off in (-2,-1,1,2):
                jx = i+off
                if 0 <= jx < len(gt_sorted):
                    nb = gt_sorted.iloc[jx]
                    ctx.append((off, nb.locus_tag, str(nb["product"])[:45],
                                "same" if nb.strand==row_strand(strand_all,lt) else "opp"))
        # ---- biophysics ----
        bio = ""
        if lt in feat_idx.index:
            fr = feat_idx.loc[lt]
            tm = fr.get("tm_helix_count",0); sp = fr.get("has_signal_pep",0)
            paa = fr.get("protein_length_aa",0); pi = fr.get("pI",None)
            tags = []
            if tm and tm>=3: tags.append(f"{int(tm)} TM helices (membrane)")
            elif tm and tm>=1: tags.append(f"{int(tm)} TM helix")
            if sp: tags.append("signal peptide (secreted/lipoprotein)")
            if pi==pi and pi: tags.append(f"pI={pi:.1f}")
            if paa: tags.append(f"{int(paa)} aa")
            bio = ", ".join(tags)

        results.append((lt, hint, best, ctx, bio, seq))

    # print + markdown
    for lt, hint, best, ctx, bio, seq in results:
        print(f"\n=== {lt}  (hint: {hint[:50]}) ===")
        if best:
            sc, ident, rid, rann, shared = best
            conf = "HIGH" if (ident>0.30 and sc>80) else "MED" if (ident>0.22 and sc>50) else "LOW"
            print(f"  HOMOLOGY [{conf}]: {rid}  -> {rann}")
            print(f"    SW score={sc}  identity={ident*100:.0f}%  shared4mers={shared}")
        else:
            print(f"  HOMOLOGY: no significant hit")
        if bio: print(f"  BIOPHYSICS: {bio}")
        if ctx:
            print(f"  CONTEXT (operon neighbours):")
            for off,nl,np_,strand in ctx:
                print(f"    {off:+d}  {nl}  {np_}  [{strand} strand]")

        md.append(f"\n## {lt}\n\n")
        md.append(f"- **Prior hint:** {hint}\n")
        if best:
            sc, ident, rid, rann, shared = best
            conf = "HIGH" if (ident>0.30 and sc>80) else "MED" if (ident>0.22 and sc>50) else "LOW"
            md.append(f"- **Homology [{conf}]:** `{rid}` -> {rann}  "
                      f"(SW {sc}, {ident*100:.0f}% id, {shared} shared 4-mers)\n")
        else:
            md.append(f"- **Homology:** no significant hit in available reference\n")
        if bio: md.append(f"- **Biophysics:** {bio}\n")
        if ctx:
            md.append(f"- **Operon neighbours:** "
                      + "; ".join(f"{o:+d} {n} ({p.strip()})" for o,n,p,_ in ctx) + "\n")

    out = REPO/"outputs/syn3a_function_inference.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(md))
    print(f"\nwrote {out}")
    # summary
    n_hit = sum(1 for r in results if r[2])
    nhi = sum(1 for r in results if r[2] and r[2][1]>0.30 and r[2][0]>80)
    nmed= sum(1 for r in results if r[2] and 0.22<r[2][1]<=0.30)
    print(f"\nSUMMARY: {len(results)} unclear essentials | "
          f"homology hit: {n_hit} (HIGH conf: {nhi}, MED: {nmed})")
    return 0


def row_strand(strand_all, lt):
    return strand_all.get(lt, "+")


if __name__ == "__main__":
    sys.exit(main())
