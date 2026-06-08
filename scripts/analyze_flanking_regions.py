"""Flanking-region analysis: is there anything UNIQUE to essential genes
in the DNA before/after the ORF?

For each labelled gene, extract the upstream and downstream nucleotide
windows (default 300 bp each, configurable 200-500) relative to the
gene's 5'/3' (reverse-complemented for minus-strand genes so 'upstream'
is always the promoter/RBS side).

Computes, per gene:
  UPSTREAM (promoter + ribosome-binding-site region)
    sd_score        best Shine-Dalgarno (AGGAGG) match in -20..-5
    sd_spacing      distance from best SD to start codon (optimal 5-10)
    minus10_score   sigma-70 -10 box (TATAAT) best match in -40..-5
    minus35_score   sigma-70 -35 box (TTGACA) best match in -70..-30
    up_gc           GC fraction of upstream window
    up_at_run       longest A/T homopolymer run (UP element / AT-rich)
  DOWNSTREAM (terminator region)
    down_gc         GC fraction of downstream window
    has_hairpin     inverted-repeat presence (rho-indep terminator)
    polyT_run       longest T run (terminator tail)
  CONTEXT
    intergenic_prev/next, first_in_operon flag

DISCOVERY ENGINE (the 'find something unique' part):
  Scans every k-mer (k=5 and k=6) in the upstream -50..0 window and the
  downstream 0..+50 window. For each k-mer, compares its occurrence rate
  in essential vs non-essential flanks, ranks by enrichment + a
  binomial z-score. Surfaces any motif over-represented specifically in
  essential-gene flanks (a TF site, a specific RBS variant, etc.).

Inputs: genome.fna + genomic.gff + protein-less (labels.csv) per org
        from the cache dir (requires .fna -- push it from Drive).

Output:
  outputs/flanking/flanking_per_gene.csv     per-gene flank features
  outputs/flanking/flanking_numeric_stats.csv ess-vs-non per feature
  outputs/flanking/flanking_kmer_upstream.csv  enriched upstream k-mers
  outputs/flanking/flanking_kmer_downstream.csv
  outputs/flanking/flanking_summary.md
"""
from __future__ import annotations
import argparse, math, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REV = str.maketrans("ACGTN", "TGCAN")
SD = "AGGAGG"; M10 = "TATAAT"; M35 = "TTGACA"


def revcomp(s): return s.translate(REV)[::-1]


def parse_fasta(path):
    out = {}; cur = None; buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None: out[cur] = "".join(buf)
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip().upper())
    if cur is not None: out[cur] = "".join(buf)
    return out


def parse_gff_cds(path):
    gene_attrs = {}; cds = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9: continue
            d = {}
            for kv in p[8].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1); d[k] = v
            if p[2] == "gene" and "ID" in d:
                gene_attrs[d["ID"]] = {"locus_tag": d.get("locus_tag",""),
                                        "gene": d.get("gene","")}
            elif p[2] == "CDS":
                cds.append((p, d))
    feats = []
    for p, d in cds:
        lt = d.get("locus_tag",""); gene = d.get("gene","")
        old = d.get("old_locus_tag","")
        if not lt and d.get("Parent","") in gene_attrs:
            ga = gene_attrs[d["Parent"]]
            lt = ga["locus_tag"]; gene = gene or ga["gene"]
        feats.append({"seqid":p[0],"start":int(p[3]),"end":int(p[4]),
                       "strand":p[6],"locus_tag":lt,"old_locus_tag":old,
                       "gene":gene,"product":d.get("product","")})
    return feats


def motif_best(seq, motif):
    if len(seq) < len(motif): return 0.0
    L = len(motif); best = 0
    for i in range(len(seq)-L+1):
        m = sum(1 for a,b in zip(seq[i:i+L], motif) if a == b)
        if m > best: best = m
    return best / L


def motif_best_pos(seq, motif):
    """Return (best_score, position_of_best_match_from_left)."""
    if len(seq) < len(motif): return 0.0, -1
    L = len(motif); best = 0; pos = -1
    for i in range(len(seq)-L+1):
        m = sum(1 for a,b in zip(seq[i:i+L], motif) if a == b)
        if m > best: best = m; pos = i
    return best/L, pos


def gc(s):
    if not s: return float("nan")
    g = s.count("G")+s.count("C"); a = s.count("A")+s.count("T")
    return g/(g+a) if g+a else float("nan")


def longest_run(seq, chars):
    best = cur = 0
    for c in seq:
        if c in chars: cur += 1; best = max(best, cur)
        else: cur = 0
    return best


def has_hairpin(seq, min_arm=4, max_loop=8, max_arm=12):
    n = len(seq)
    for i in range(n - 2*min_arm):
        for arm in range(min_arm, min(max_arm, (n-i)//2)+1):
            for loop in range(3, max_loop+1):
                j = i + arm + loop
                if j + arm > n: break
                if seq[i:i+arm] == revcomp(seq[j:j+arm]):
                    return 1
    return 0


def mann_whitney_p(a, b):
    import numpy as np
    a = np.array([x for x in a if x == x]); b = np.array([x for x in b if x == x])
    if len(a) < 5 or len(b) < 5: return float("nan")
    c = np.concatenate([a,b]); r = c.argsort().argsort()+1
    u1 = r[:len(a)].sum() - len(a)*(len(a)+1)/2
    u = min(u1, len(a)*len(b)-u1); mu = len(a)*len(b)/2
    s = math.sqrt(len(a)*len(b)*(len(a)+len(b)+1)/12)
    return math.erfc(abs((u-mu)/s)/math.sqrt(2)) if s else 1.0


def cohens_d(a, b):
    import numpy as np
    a = np.array([x for x in a if x == x]); b = np.array([x for x in b if x == x])
    if len(a) < 2 or len(b) < 2: return float("nan")
    s = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/s if s else float("nan")


def auc_uni(scores, labels):
    import numpy as np
    s = np.array(scores, float); y = np.array(labels)
    m = ~np.isnan(s); s, y = s[m], y[m]
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    o = np.argsort(-s); y2 = y[o]
    npos = int(y2.sum()); nn = len(y2)-npos
    return 1 - (np.where(y2==1)[0].sum()+npos - npos*(npos+1)/2)/(npos*nn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT/"data/drive_import/genome_cache")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT/"data/drive_import/labels/genome_cache_manifest.csv")
    p.add_argument("--labels-csv", type=Path,
                   default=REPO_ROOT/"data/drive_import/labels/labels.csv")
    p.add_argument("--window", type=int, default=300,
                   help="flank window size in bp each side (200-500)")
    p.add_argument("--kmer-window", type=int, default=50,
                   help="window for k-mer discovery (closest to ORF)")
    p.add_argument("--min-join", type=float, default=0.3)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT/"outputs/flanking")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd, numpy as np
    labels = pd.read_csv(args.labels_csv)
    mf = pd.read_csv(args.manifest)
    elig = mf[(mf.kind=="ours") & mf.accession.notna() &
              mf.join_rate.notna() & (mf.join_rate >= args.min_join)]
    print(f"eligible orgs: {len(elig)}")

    W = args.window; KW = args.kmer_window
    rows = []
    n_missing_fna = 0
    for _, m in elig.iterrows():
        org = m.organism; acc = m.accession
        gd = args.cache_dir / acc
        fna_p = gd / "genome.fna"; gff_p = gd / "genomic.gff"
        if not fna_p.exists():
            print(f"  {org}: NO genome.fna (push it) -- skip"); n_missing_fna += 1; continue
        if not gff_p.exists():
            print(f"  {org}: no gff -- skip"); continue
        fna = parse_fasta(fna_p)
        feats = parse_gff_cds(gff_p)
        feats.sort(key=lambda r:(r["seqid"], r["start"]))
        ol = labels[labels.organism==org]
        lt2e = dict(zip(ol.locus_tag.astype(str), ol.essential.astype(int)))
        n_added = 0
        for i, f in enumerate(feats):
            e = lt2e.get(f["locus_tag"])
            if e is None and f["old_locus_tag"]: e = lt2e.get(f["old_locus_tag"])
            if e is None: continue
            seq = fna.get(f["seqid"])
            if not seq: continue
            s, en, strand = f["start"], f["end"], f["strand"]
            # upstream = W bp 5' of start codon; downstream = W bp 3' of stop
            if strand == "+":
                up = seq[max(0, s-1-W):s-1]
                dn = seq[en:en+W]
            else:
                up = revcomp(seq[en:en+W])
                dn = revcomp(seq[max(0, s-1-W):s-1])
            if len(up) < 30 or len(dn) < 30: continue
            # RBS region: last 20 bp before start (closest to ORF)
            rbs_region = up[-20:]
            sd_score, sd_pos = motif_best_pos(rbs_region, SD)
            sd_spacing = (len(rbs_region) - sd_pos - len(SD)) if sd_pos >= 0 else -1
            # promoter windows
            m10_region = up[-40:-5] if len(up) >= 40 else up
            m35_region = up[-70:-30] if len(up) >= 70 else up
            prev_f = feats[i-1] if i>0 and feats[i-1]["seqid"]==f["seqid"] else None
            next_f = feats[i+1] if i+1<len(feats) and feats[i+1]["seqid"]==f["seqid"] else None
            ig_prev = (s - prev_f["end"] - 1) if prev_f else None
            ss_prev = int(prev_f["strand"]==strand) if prev_f else None
            first_in_operon = int(ig_prev is None or ig_prev > 100 or ss_prev == 0)
            rows.append({
                "organism": org, "locus_tag": f["locus_tag"],
                "product": f["product"], "essential": e,
                "sd_score": sd_score, "sd_spacing": sd_spacing,
                "minus10_score": motif_best(m10_region, M10),
                "minus35_score": motif_best(m35_region, M35),
                "up_gc": gc(up), "up_at_run": longest_run(up, "AT"),
                "down_gc": gc(dn), "polyT_run": longest_run(dn[:50], "T"),
                "has_hairpin": has_hairpin(dn[:60]),
                "intergenic_prev": ig_prev,
                "first_in_operon": first_in_operon,
                "_up_kmer": up[-KW:],     # for k-mer scan
                "_dn_kmer": dn[:KW],
            })
            n_added += 1
        print(f"  {org} ({acc}): {n_added:,} genes")
    if n_missing_fna:
        print(f"\n  {n_missing_fna} orgs skipped for missing genome.fna -- "
              f"push the .fna files and re-run.")
    if not rows:
        print("ERROR: no rows (likely no .fna files present)", file=sys.stderr); return 1

    df = pd.DataFrame(rows)
    print(f"\n=== {len(df):,} genes, {int(df.essential.sum()):,} essential "
          f"({df.essential.mean()*100:.1f}%) ===")
    df.drop(columns=["_up_kmer","_dn_kmer"]).to_csv(
        args.out_dir/"flanking_per_gene.csv", index=False)

    # ---- numeric feature stats ----
    feats_num = ["sd_score","sd_spacing","minus10_score","minus35_score",
                 "up_gc","up_at_run","down_gc","polyT_run","has_hairpin",
                 "intergenic_prev","first_in_operon"]
    print(f"\n=== flank features: essential vs non-essential (sorted |d|) ===")
    print(f"  {'feature':<18s} {'ess_mean':>10s} {'non_mean':>10s} {'d':>6s} {'p_MW':>9s} {'AUC':>5s}")
    stats = []
    em = df.essential==1
    for fcol in feats_num:
        a = df.loc[em, fcol].dropna().tolist(); b = df.loc[~em, fcol].dropna().tolist()
        if len(a)<50 or len(b)<50: continue
        d = cohens_d(a,b); pv = mann_whitney_p(a,b)
        auc = auc_uni(df[fcol].tolist(), df.essential.tolist())
        stats.append({"feature":fcol,"ess_mean":sum(a)/len(a),"non_mean":sum(b)/len(b),
                       "cohen_d":d,"p_mannwhitney":pv,"auc":auc})
    stats.sort(key=lambda r:-abs(r["cohen_d"]) if r["cohen_d"]==r["cohen_d"] else 0)
    for r in stats:
        print(f"  {r['feature']:<18s} {r['ess_mean']:>10.4f} {r['non_mean']:>10.4f} "
              f"{r['cohen_d']:>+5.2f} {r['p_mannwhitney']:>9.2e} {r['auc']:>5.3f}")
    pd.DataFrame(stats).to_csv(args.out_dir/"flanking_numeric_stats.csv", index=False)

    # ---- k-mer discovery ----
    def kmer_scan(col, label):
        print(f"\n=== {label} k-mer discovery (k=5,6) ===")
        out_rows = []
        for k in (5, 6):
            ess_c, non_c = Counter(), Counter()
            n_ess = n_non = 0
            for _, r in df.iterrows():
                seqk = r[col]
                if len(seqk) < k: continue
                kmers = set(seqk[i:i+k] for i in range(len(seqk)-k+1) if "N" not in seqk[i:i+k])
                if r.essential == 1:
                    n_ess += 1
                    for km in kmers: ess_c[km] += 1
                else:
                    n_non += 1
                    for km in kmers: non_c[km] += 1
            for km in set(ess_c) | set(non_c):
                ne = ess_c.get(km,0); nn = non_c.get(km,0)
                if ne + nn < 50: continue
                pe = ne/max(n_ess,1); pn = nn/max(n_non,1)
                # binomial z for difference of proportions
                p_pool = (ne+nn)/(n_ess+n_non)
                se = math.sqrt(p_pool*(1-p_pool)*(1/n_ess + 1/n_non)) if p_pool>0 else 1
                z = (pe-pn)/se if se else 0
                out_rows.append({"k":k,"kmer":km,"pct_ess":pe*100,"pct_non":pn*100,
                                  "enrichment":(pe+1e-6)/(pn+1e-6),"z":z,
                                  "n_ess":ne,"n_non":nn})
        kdf = pd.DataFrame(out_rows).sort_values("z", ascending=False)
        print(f"  TOP enriched in essential flanks (by z-score):")
        print(f"  {'kmer':<8s} {'ess%':>6s} {'non%':>6s} {'enrich':>7s} {'z':>7s}")
        for _, r in kdf.head(20).iterrows():
            print(f"  {r.kmer:<8s} {r.pct_ess:>5.1f}% {r.pct_non:>5.1f}% "
                  f"{r.enrichment:>6.2f}x {r.z:>+6.1f}")
        return kdf

    up_k = kmer_scan("_up_kmer", "UPSTREAM (-%d..0)" % KW)
    up_k.to_csv(args.out_dir/"flanking_kmer_upstream.csv", index=False)
    dn_k = kmer_scan("_dn_kmer", "DOWNSTREAM (0..+%d)" % KW)
    dn_k.to_csv(args.out_dir/"flanking_kmer_downstream.csv", index=False)

    # ---- markdown ----
    md = ["# Flanking-region analysis: anything unique to essential genes?\n\n",
          f"**Genes:** {len(df):,} ({int(df.essential.sum()):,} essential) "
          f"across {df.organism.nunique()} organisms  \n",
          f"**Window:** {W} bp each side; k-mer scan on closest {KW} bp\n\n",
          "## Flank features (essential vs non-essential)\n\n",
          "| feature | ess_mean | non_mean | Cohen's d | AUC | p |\n|---|---|---|---|---|---|\n"]
    for r in stats:
        md.append(f"| {r['feature']} | {r['ess_mean']:.4f} | {r['non_mean']:.4f} | "
                  f"{r['cohen_d']:+.2f} | {r['auc']:.3f} | {r['p_mannwhitney']:.1e} |\n")
    md.append("\n## Top upstream k-mers enriched in essential flanks\n\n")
    md.append("| kmer | ess% | non% | enrichment | z |\n|---|---|---|---|---|\n")
    for _, r in up_k.head(20).iterrows():
        md.append(f"| {r.kmer} | {r.pct_ess:.1f}% | {r.pct_non:.1f}% | "
                  f"{r.enrichment:.2f}x | {r.z:+.1f} |\n")
    (args.out_dir/"flanking_summary.md").write_text("".join(md))
    print(f"\nwrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
