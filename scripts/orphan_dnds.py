"""PATH-ORPHAN, STEP 3 (Phase A) -- dN/dS selection signature  [REAL in sandbox].

Hypothesis: a rogue essential (essential, conservation<0.1) that is genuinely
important is under PURIFYING selection (dN/dS << 1), even though it is rare
across organisms. Conservation (presence/absence) is blind to this; the codon
substitution pattern within the gene's few relatives is not.

We compute pairwise Nei-Gojobori (1986) dN/dS between each gene and its CLOSEST
ortholog (same OG, highest protein identity) drawn from other organisms that
have a nucleotide genome. Pure Python/NumPy -- NO codeml/HyPhy/Biopython needed.

EFFICIENCY: between closely related organisms (e.g. sister Ralstonia) many
orthologs are the SAME protein length, so the codon alignment is the identity
map and NG is exact with no aligner. We use same-length orthologs in the sandbox
(fast, artifact-free); the Colab path can add Biopython alignment for the rest.

DATA (sandbox & Colab): genome.fna + genomic.gff per assembly (CDS extracted by
coordinate, strand-aware) + orthology_features.csv (OG membership).

OUTPUT (idempotent):
  outputs/orphan/dnds_<org>.parquet  locus_tag, dnds, dn, ds, ortholog_org,
     ortholog_locus, pct_identity, prot_len, n_same_len_orthologs

--smoke : validate Nei-Gojobori on synthetic codon pairs (no external data)
--real  : compute for the org from cached genomes (runs in sandbox)
"""
from __future__ import annotations
import argparse, math, sys, itertools, csv, glob, os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "drive_import"
OUT = REPO / "outputs" / "orphan"

BASES = "TCAG"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {c: AA[i] for i, c in enumerate(CODONS)}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def tr(c):
    return CODON_TABLE.get(c.upper(), "X")


# ----------------------------- Nei-Gojobori -----------------------------

def codon_sites(codon):
    codon = codon.upper()
    aa = tr(codon)
    if aa in ("*", "X") or len(codon) != 3:
        return 0.0, 0.0
    n = 0.0
    for pos in range(3):
        ns = 0
        for b in BASES:
            if b == codon[pos]:
                continue
            m = tr(codon[:pos] + b + codon[pos + 1:])
            if m == "*" or m != aa:
                ns += 1
        n += ns / 3.0
    return 3.0 - n, n


def _diff_paths(c1, c2):
    diffs = [i for i in range(3) if c1[i] != c2[i]]
    if not diffs:
        return 0.0, 0.0
    Sd = Nd = 0.0; valid = 0
    for path in itertools.permutations(diffs):
        cur = c1; ok = True; s = n = 0
        for pos in path:
            nxt = cur[:pos] + c2[pos] + cur[pos + 1:]
            a1, a2 = tr(cur), tr(nxt)
            if a1 == "*" or a2 == "*":
                ok = False; break
            if a1 == a2:
                s += 1
            else:
                n += 1
            cur = nxt
        if ok:
            Sd += s; Nd += n; valid += 1
    if valid == 0:
        return 0.0, float(len(diffs))
    return Sd / valid, Nd / valid


def nei_gojobori(seq1, seq2):
    seq1, seq2 = seq1.upper(), seq2.upper()
    L = min(len(seq1), len(seq2)); L -= L % 3
    S = N = Sd = Nd = 0.0
    for i in range(0, L, 3):
        c1, c2 = seq1[i:i+3], seq2[i:i+3]
        if "-" in c1 or "-" in c2 or "N" in c1 or "N" in c2:
            continue
        s1, n1 = codon_sites(c1); s2, n2 = codon_sites(c2)
        S += (s1 + s2) / 2.0; N += (n1 + n2) / 2.0
        sd, nd = _diff_paths(c1, c2); Sd += sd; Nd += nd

    def jc(p):
        if p is None or p < 0 or p >= 0.75:
            return float("nan")
        return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)

    pS = Sd / S if S > 0 else float("nan")
    pN = Nd / N if N > 0 else float("nan")
    dS = jc(pS); dN = jc(pN)
    omega = (dN / dS) if (dS and dS > 0 and not math.isnan(dS)
                          and not math.isnan(dN)) else float("nan")
    return {"dN": dN, "dS": dS, "omega": omega, "pN": pN, "pS": pS,
            "Sd": Sd, "Nd": Nd}


# ----------------------------- CDS extraction -----------------------------

def _read_fna(path):
    seqs = {}; sid = None; buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if sid:
                    seqs[sid] = "".join(buf)
                sid = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if sid:
        seqs[sid] = "".join(buf)
    return seqs


def _gff_cds(path):
    """locus_tag -> (contig, strand, [(start,end),...])  for CDS features."""
    import re
    feats = {}
    with open(path) as f:
        for line in f:
            if "\tCDS\t" not in line:
                continue
            c = line.split("\t")
            lt = re.search(r'locus_tag=([^;\n]+)', c[8])
            if not lt:
                continue
            lt = lt.group(1)
            feats.setdefault(lt, [c[0], c[6], []])
            feats[lt][2].append((int(c[3]), int(c[4])))
    return feats


def extract_cds(acc):
    """protein-coding nt CDS per locus_tag for an assembly accession."""
    base = DATA / "genome_cache" / acc
    fna = base / "genome.fna"; gff = base / "genomic.gff"
    if not (fna.exists() and gff.exists()):
        return {}
    contigs = _read_fna(fna); feats = _gff_cds(gff)
    out = {}
    for lt, (ctg, strand, segs) in feats.items():
        s = contigs.get(ctg)
        if not s:
            continue
        segs = sorted(segs)
        seq = "".join(s[a-1:b] for a, b in segs)        # 1-based inclusive
        if strand == "-":
            seq = seq.translate(COMP)[::-1]
        out[lt] = seq.upper()
    return out


# ----------------------------- real pipeline -----------------------------

def _manifest_acc():
    acc = {}
    with open(DATA / "labels" / "genome_cache_manifest.csv") as f:
        for r in csv.DictReader(f):
            if r["accession"]:
                acc[r["organism"]] = r["accession"]
    return acc


def _identity(p1, p2):
    """ungapped % identity of two equal-length protein strings."""
    return sum(a == b for a, b in zip(p1, p2)) / len(p1) if p1 else 0.0


def run_real(args):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print(f"ERROR: {bridge} missing -- run orphan_bridge.py --real first")
        return 2
    out_path = OUT / f"dnds_{args.org}.parquet"
    if out_path.exists() and not args.force:
        print(f"STEP 3 dN/dS -- CACHED ({out_path}); --force to rebuild")
        _report(pd.read_parquet(out_path), pd.read_parquet(bridge))
        return 0

    df = pd.read_parquet(bridge)
    acc = _manifest_acc()
    # which orgs have a nucleotide genome (CDS extractable)?
    fna_orgs = [o for o, a in acc.items()
                if (DATA / "genome_cache" / a / "genome.fna").exists()]
    print("=" * 70)
    print(f"STEP 3 dN/dS (Nei-Gojobori) -- {args.org}")
    print(f"  ortholog source orgs with nucleotide genome: {len(fna_orgs)}")
    print("=" * 70)

    # CDS for the focal org + all source orgs, keyed by (org, locus_tag)
    cds = {}
    for o in set([args.org] + fna_orgs):
        if o in acc:
            for lt, seq in extract_cds(acc[o]).items():
                cds[(o, lt)] = seq
    print(f"  CDS extracted: {len(cds):,} across {len(set(k[0] for k in cds))} orgs")

    # OG -> members across fna source orgs (orthology table)
    og_members = {}
    src = set(fna_orgs) - {args.org}
    with open(DATA / "labels" / "orthology_features.csv") as f:
        for r in csv.DictReader(f):
            if r["organism"] in src:
                og_members.setdefault(r["og_id"], []).append(
                    (r["organism"], r["locus_tag"]))

    rows = []
    n_done = 0
    for rec in df.itertuples():
        lt = rec.locus_tag; og = rec.og_id
        c_self = cds.get((args.org, lt))
        res = {"locus_tag": lt, "dnds": np.nan, "dn": np.nan, "ds": np.nan,
               "ortholog_org": "", "ortholog_locus": "",
               "pct_identity": np.nan, "prot_len": np.nan,
               "n_same_len_orthologs": 0}
        if c_self and og in og_members and len(c_self) % 3 == 0:
            p_self = "".join(tr(c_self[i:i+3]) for i in range(0, len(c_self), 3))
            p_self = p_self.rstrip("*")
            best = None; n_same = 0
            for (oo, ol) in og_members[og]:
                c_o = cds.get((oo, ol))
                if not c_o or len(c_o) != len(c_self) or len(c_o) % 3:
                    continue
                p_o = "".join(tr(c_o[i:i+3]) for i in range(0, len(c_o), 3)).rstrip("*")
                if len(p_o) != len(p_self):
                    continue
                n_same += 1
                idn = _identity(p_self, p_o)
                if best is None or idn > best[0]:
                    best = (idn, oo, ol, c_o)
            res["n_same_len_orthologs"] = n_same
            if best is not None and best[0] < 1.0:    # need >=1 difference
                ng = nei_gojobori(c_self, best[3])
                res.update(dnds=ng["omega"], dn=ng["dN"], ds=ng["dS"],
                           ortholog_org=best[1], ortholog_locus=best[2],
                           pct_identity=round(best[0], 4),
                           prot_len=len(p_self))
                n_done += 1
        rows.append(res)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  computed dN/dS for {n_done:,}/{len(df):,} genes "
          f"(same-length ortholog found & >=1 substitution)")
    _report(out, df)
    print(f"\n  wrote {out_path}")
    return 0


def _report(dnds, bridge):
    """The decisive univariate test: do rogue ESSENTIALS have lower omega?"""
    import pandas as pd, numpy as np
    m = bridge.merge(dnds, on="locus_tag", how="left")
    have = m[m["dnds"].notna() & np.isfinite(m["dnds"])]
    print(f"\n  --- dN/dS distribution (finite omega): {len(have):,} genes ---")
    if len(have) == 0:
        print("  (none) "); return
    def q(s):
        return f"med {s.median():.3f}  mean {s.mean():.3f}"
    rogue = have[have["conservation"] < 0.10]
    print(f"  whole genome   essential   : {q(have[have.essential==1]['dnds'])}")
    print(f"  whole genome   non-essential: {q(have[have.essential==0]['dnds'])}")
    if len(rogue):
        re_e = rogue[rogue.essential == 1]["dnds"]
        re_n = rogue[rogue.essential == 0]["dnds"]
        print(f"  ROGUE (cons<0.1) essential   : {q(re_e)}  (n={len(re_e)})")
        print(f"  ROGUE (cons<0.1) non-essential: {q(re_n)}  (n={len(re_n)})")
        # the gate-relevant number: rank rogue genes by LOW omega -> R@P30
        from importlib import util
        s = -rogue["dnds"].to_numpy()          # low omega = essential-like
        y = rogue["essential"].to_numpy()
        order = np.argsort(-s); ys = y[order]
        tp = np.cumsum(ys); ks = np.arange(1, len(ys)+1); prec = tp/ks
        ok = np.where(prec >= 0.30)[0]
        if len(ok):
            k = int(ok.max()); r = tp[k]/y.sum()
            print(f"  ROGUE-ZONE dN/dS-only ranker: R@P30 = {r:.3f} "
                  f"(k={k+1}, vs conservation bar 0.000)")
        else:
            print(f"  ROGUE-ZONE dN/dS-only ranker: no P>=0.30 head "
                  f"(base {y.mean():.3f})")


def run_smoke():
    print("=" * 70)
    print("STEP 3 dN/dS SMOKE -- Nei-Gojobori on synthetic codon pairs")
    print("=" * 70)
    s = "ATGAAACGTGGT"
    assert nei_gojobori(s, s)["Nd"] == 0
    assert abs(nei_gojobori(s, "ATGAAGCGTGGT")["Sd"] - 1.0) < 1e-9   # synonymous
    rn = nei_gojobori(s, "ATGGAACGTGGT")                              # nonsyn
    assert abs(rn["Nd"] - 1.0) < 1e-9 and rn["Sd"] == 0
    assert codon_sites("GGT")[0] >= 1.0 and codon_sites("ATG")[0] == 0
    print("  syn/nonsyn classification, site counts: OK")
    # CDS extraction round-trip on a synthetic 2-contig genome (+ and - strand)
    import tempfile
    d = Path(tempfile.mkdtemp()); (d / "genome_cache" / "g").mkdir(parents=True)
    # plus-strand gene LTP=ATGAAATAA ; minus-strand gene LTM is RC of ATGAAATAA
    contig = "CCCATGAAATAAGGG" + "TTATTTCAT" + "AAA"
    (d / "genome_cache" / "g" / "genome.fna").write_text(">c1 test\n" + contig + "\n")
    (d / "genome_cache" / "g" / "genomic.gff").write_text(
        "c1\tx\tCDS\t4\t12\t.\t+\t0\tlocus_tag=LTP;product=p\n"
        "c1\tx\tCDS\t16\t24\t.\t-\t0\tlocus_tag=LTM;product=p\n")
    global DATA
    saved = DATA
    DATA = d
    try:
        cds = extract_cds("g")
    finally:
        DATA = saved
    print(f"  extracted CDS: LTP={cds.get('LTP')}  LTM={cds.get('LTM')}")
    assert cds["LTP"] == "ATGAAATAA", "plus-strand extraction"
    assert cds["LTM"] == "ATGAAATAA", "minus-strand reverse-complement"
    print("  CDS coordinate + strand extraction: OK")
    print("\n=== STEP 3 SMOKE PASS ===")
    print("  NG core + CDS extraction (both strands) validated.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
