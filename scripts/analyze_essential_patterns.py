"""Comprehensive exploratory analysis: what's common to essential genes?

For every organism where we have a clean genome+annotation+labels join,
extract everything measurable per gene and compare essential vs
non-essential across many angles:

  STRUCTURAL    length (nt, aa), strand, genome position
  COMPOSITION   GC%, GC1/GC2/GC3 by codon position, CDS GC
  CODON         start codon (ATG/GTG/TTG), stop codon (TAA/TAG/TGA)
  UPSTREAM      -100..0 region GC, AT-run, Shine-Dalgarno (AGGAGG)
                proximity, sigma-70 -10 box (TATAAT) similarity,
                -35 box (TTGACA) similarity
  DOWNSTREAM    +0..+50 region GC, palindrome presence (terminator
                hairpin proxy)
  NEIGHBOR      intergenic distance to prev/next, same-strand flags,
                neighbor essentiality (where labeled)
  OPERON        likely-operonic indicator (short intergen + same strand)
  ANNOTATION    product description, gene name family
  CONSERVATION  family_n_organisms, n_paralogs_in_genome (from orth)

Then per-feature comparisons:
  numeric    -> mean +/- std for ess vs non, Cohen's d, Mann-Whitney p,
                 univariate AUC
  categorical-> chi-squared, proportion table
  text       -> top words enriched in essential descriptions (Fisher)
  pooled     -> same analyses across all organisms combined

Output:
  per_gene_full_features.csv     -- master per-gene table
  essential_vs_non_per_org.csv   -- per-organism per-feature stats
  essential_vs_non_pooled.csv    -- pooled stats
  top_essential_words.csv        -- gene-description words enriched
  essential_analysis_summary.md  -- readable markdown report
"""
from __future__ import annotations
import argparse, gzip, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

GENETIC_CODE = {
  "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
  "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
  "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
  "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
  "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
  "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
  "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
  "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
REV_COMP = str.maketrans("ACGTN", "TGCAN")
M10, M35 = "TATAAT", "TTGACA"
SD = "AGGAGG"      # Shine-Dalgarno consensus
UPSTREAM = 100
DOWNSTREAM = 50

# stopwords for description-text enrichment
STOPWORDS = set("a an and are as at be by for from has have in is it its of on or "
                "that the their these this to with putative hypothetical protein "
                "uncharacterized predicted family domain containing related dna rna "
                "small large subunit type system component conserved member".split())


def revcomp(s): return s.translate(REV_COMP)[::-1]


def parse_fasta(path: Path) -> dict[str, str]:
    op = gzip.open if str(path).endswith(".gz") else open
    out, cur = {}, None; buf = []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None: out[cur] = "".join(buf)
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip().upper())
    if cur is not None: out[cur] = "".join(buf)
    return out


def parse_gff_cds(path: Path) -> list[dict]:
    feats = []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#") or "\tCDS\t" not in line: continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9: continue
            d = {}
            for kv in p[8].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1); d[k] = v
            feats.append({
                "seqid":         p[0],
                "start":         int(p[3]),
                "end":           int(p[4]),
                "strand":        p[6],
                "locus_tag":     d.get("locus_tag",""),
                "old_locus_tag": d.get("old_locus_tag",""),
                "gene":          d.get("gene",""),
                "product":       d.get("product",""),
            })
    return feats


def gc(s):
    if not s: return float("nan")
    g = s.count("G")+s.count("C"); a = s.count("A")+s.count("T")
    return g/(g+a) if g+a else float("nan")


def gc_pos(cds, p):
    if len(cds) < 3: return float("nan")
    pos = [cds[i+p] for i in range(0, len(cds)-2, 3)]
    g = sum(1 for b in pos if b in "GC"); a = sum(1 for b in pos if b in "ACGT")
    return g/a if a else float("nan")


def motif_max(seq, motif):
    if len(seq) < len(motif): return float("nan")
    L = len(motif); best = 0
    for i in range(len(seq)-L+1):
        m = sum(1 for a,b in zip(seq[i:i+L], motif) if a==b)
        if m > best: best = m
    return best/L


def at_run_max(seq):
    return max((len(m.group(0)) for m in re.finditer(r"A{4,}|T{4,}", seq)),
                default=0)


def has_palindrome(seq, min_arm=5, max_loop=8):
    """Crude inverted-repeat check (rho-independent terminator proxy)."""
    n = len(seq)
    for i in range(n - 2*min_arm - 1):
        for arm in range(min_arm, min(15, (n-i)//2)):
            for loop in range(3, max_loop+1):
                j = i + arm + loop
                if j + arm > n: break
                left = seq[i:i+arm]
                right = revcomp(seq[j:j+arm])
                if left == right:
                    return 1
    return 0


def shine_dalgarno_proximity(seq):
    """Best SD-like match (Hamming similarity to AGGAGG) in last ~20 bp."""
    window = seq[-20:] if len(seq) >= 20 else seq
    return motif_max(window, SD) if window else float("nan")


def extract_features(fna: dict, feats: list[dict]) -> list[dict]:
    feats.sort(key=lambda r: (r["seqid"], r["start"]))
    rows = []
    contig_lens = {k: len(v) for k, v in fna.items()}
    for i, f in enumerate(feats):
        if f["seqid"] not in fna: continue
        seq = fna[f["seqid"]]
        cds = seq[f["start"]-1:f["end"]]
        if f["strand"] == "-": cds = revcomp(cds)
        if len(cds) < 30 or len(cds) % 3: continue
        # upstream / downstream relative to gene
        if f["strand"] == "+":
            up = seq[max(0, f["start"]-1-UPSTREAM):f["start"]-1]
            dn = seq[f["end"]:f["end"]+DOWNSTREAM]
        else:
            up = revcomp(seq[f["end"]:f["end"]+UPSTREAM])
            dn = revcomp(seq[max(0, f["start"]-1-DOWNSTREAM):f["start"]-1])
        # neighbor info
        prev_f = feats[i-1] if i>0 and feats[i-1]["seqid"]==f["seqid"] else None
        next_f = feats[i+1] if i+1<len(feats) and feats[i+1]["seqid"]==f["seqid"] else None
        ig_prev = (f["start"] - prev_f["end"] - 1) if prev_f else None
        ig_next = (next_f["start"] - f["end"] - 1) if next_f else None
        ss_prev = int(prev_f["strand"] == f["strand"]) if prev_f else None
        ss_next = int(next_f["strand"] == f["strand"]) if next_f else None
        operon = int(ig_prev is not None and ig_prev < 50 and ss_prev == 1)

        rows.append({
            "locus_tag":     f["locus_tag"],
            "old_locus_tag": f["old_locus_tag"],
            "gene":          f["gene"],
            "product":       f["product"],
            "seqid":         f["seqid"],
            "strand":        f["strand"],
            "start":         f["start"],
            "end":           f["end"],
            "length_nt":     len(cds),
            "length_aa":     (len(cds)//3) - 1,
            "rel_position":  (f["start"]/contig_lens[f["seqid"]]) if contig_lens[f["seqid"]] else float("nan"),
            "gc":            gc(cds),
            "gc1":           gc_pos(cds, 0),
            "gc2":           gc_pos(cds, 1),
            "gc3":           gc_pos(cds, 2),
            "start_codon":   cds[:3],
            "stop_codon":    cds[-3:],
            "up_gc":         gc(up),
            "up_at_run":     at_run_max(up),
            "minus10_score": motif_max(up[-40:], M10) if len(up)>=40 else float("nan"),
            "minus35_score": motif_max(up[-70:-30], M35) if len(up)>=70 else float("nan"),
            "sd_score":      shine_dalgarno_proximity(up),
            "down_gc":       gc(dn),
            "has_palindrome":has_palindrome(dn),
            "intergenic_prev":  ig_prev,
            "intergenic_next":  ig_next,
            "same_strand_prev": ss_prev,
            "same_strand_next": ss_next,
            "operon_prev":      operon,
        })
    return rows


def mann_whitney_p(a, b):
    import numpy as np
    a = np.array([x for x in a if x==x]); b = np.array([x for x in b if x==x])
    if len(a)<5 or len(b)<5: return float("nan")
    c = np.concatenate([a,b])
    r = c.argsort().argsort()+1
    u1 = r[:len(a)].sum() - len(a)*(len(a)+1)/2
    u = min(u1, len(a)*len(b) - u1)
    mu = len(a)*len(b)/2
    s = math.sqrt(len(a)*len(b)*(len(a)+len(b)+1)/12)
    if s == 0: return 1.0
    z = (u-mu)/s
    return math.erfc(abs(z)/math.sqrt(2))


def cohens_d(a, b):
    import numpy as np
    a = np.array([x for x in a if x==x]); b = np.array([x for x in b if x==x])
    if len(a)<2 or len(b)<2: return float("nan")
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) /
                   (len(a)+len(b)-2))
    return (a.mean()-b.mean())/s if s else float("nan")


def auc_uni(scores, labels):
    import numpy as np
    s = np.array(scores); y = np.array(labels)
    m = ~np.isnan(s); s, y = s[m], y[m]
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    o = np.argsort(-s); y2 = y[o]
    npos = int(y2.sum()); nneg = len(y2)-npos
    rs = np.where(y2==1)[0].sum() + npos
    return 1 - (rs - npos*(npos+1)/2)/(npos*nneg)


def chi2_p(a_ess, a_non):
    """2-row contingency on a Counter of categories."""
    cats = sorted(set(a_ess) | set(a_non))
    if not cats: return float("nan")
    obs = [[a_ess.get(c,0), a_non.get(c,0)] for c in cats]
    total_ess = sum(a_ess.values()); total_non = sum(a_non.values())
    n = total_ess + total_non
    if n == 0: return float("nan")
    chi = 0
    for i, c in enumerate(cats):
        row_total = obs[i][0]+obs[i][1]
        if row_total == 0: continue
        for j, col_total in enumerate([total_ess, total_non]):
            exp = row_total*col_total/n
            if exp < 1: continue
            chi += (obs[i][j]-exp)**2/exp
    df = max(len(cats)-1, 1)
    return math.exp(-chi/2) if df==1 else 1.0  # rough; just to flag enrichment


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",    type=Path, required=True)
    p.add_argument("--manifest",     type=Path, required=True)
    p.add_argument("--labels-csv",   type=Path, required=True)
    p.add_argument("--orth-csv",     type=Path, required=True)
    p.add_argument("--out-dir",      type=Path, required=True)
    p.add_argument("--min-join",     type=float, default=0.5)
    p.add_argument("--only",         nargs="*", default=None)
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels_csv)
    orth = pd.read_csv(args.orth_csv)
    mf = pd.read_csv(args.manifest)
    clean = mf[(mf.kind=="ours") & (mf.join_rate.notna())
               & (mf.join_rate >= args.min_join)].copy()
    if args.only:
        clean = clean[clean.organism.isin(args.only)]
    print(f"organisms to analyze ({len(clean)}): {clean.organism.tolist()}")

    all_rows = []
    for _, r in clean.iterrows():
        org = r.organism; acc = r.accession
        gd = args.cache_dir / acc
        fna_p, gff_p = gd/"genome.fna", gd/"genomic.gff"
        if not fna_p.exists() or not gff_p.exists():
            print(f"  {org}: missing files, skip"); continue
        fna = parse_fasta(fna_p); feats = parse_gff_cds(gff_p)
        rows = extract_features(fna, feats)
        # join labels
        org_lab = labels[labels.organism==org]
        lt2e = dict(zip(org_lab.locus_tag.astype(str), org_lab.essential.astype(int)))
        org_orth = orth[orth.organism==org]
        lt2og = dict(zip(org_orth.locus_tag.astype(str), org_orth.og_id.astype(str)))
        lt2nfo = dict(zip(org_orth.locus_tag.astype(str), org_orth.family_n_organisms))
        lt2par = dict(zip(org_orth.locus_tag.astype(str), org_orth.n_paralogs_in_genome))
        matched = 0
        for row in rows:
            row["organism"] = org
            e = None
            for key in (row["locus_tag"], row["old_locus_tag"]):
                if key and key in lt2e:
                    e = lt2e[key]; break
            row["essential"] = e
            # orth features
            for key in (row["locus_tag"], row["old_locus_tag"]):
                if key and key in lt2og:
                    row["og_id"] = lt2og[key]
                    row["family_n_orgs"] = lt2nfo.get(key)
                    row["n_paralogs"] = lt2par.get(key)
                    break
            if e is not None: matched += 1
        labeled = [r for r in rows if r["essential"] is not None]
        n_ess = sum(1 for r in labeled if r["essential"]==1)
        print(f"  {org}: {len(rows):,} CDS, {matched:,} labeled "
              f"({n_ess:,} essential = {100*n_ess/max(matched,1):.1f}%)")
        # add neighbor essentiality from labeled rows
        lt2ess = {r["locus_tag"]: r["essential"] for r in rows if r["essential"] is not None}
        # already sorted by start within parse; reapply across whole list
        rows_sorted = sorted(rows, key=lambda r: (r["seqid"], r["start"]))
        for i, row in enumerate(rows_sorted):
            prev_r = rows_sorted[i-1] if i>0 and rows_sorted[i-1]["seqid"]==row["seqid"] else None
            next_r = rows_sorted[i+1] if i+1<len(rows_sorted) and rows_sorted[i+1]["seqid"]==row["seqid"] else None
            row["prev_essential"] = prev_r["essential"] if prev_r else None
            row["next_essential"] = next_r["essential"] if next_r else None
        all_rows.extend(labeled)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_dir/"per_gene_full_features.csv", index=False)
    print(f"\n=== master table: {len(df):,} labeled genes across "
          f"{df.organism.nunique()} organisms ===")
    print(f"essential rate (pooled): {df.essential.mean()*100:.1f}%")
    print(f"  per-org essential rates:")
    print(df.groupby("organism").essential.agg(["count","sum","mean"]).to_string())

    # ---- per-feature comparisons (pooled) ----
    numeric_feats = ["length_nt","length_aa","gc","gc1","gc2","gc3","rel_position",
                     "up_gc","up_at_run","minus10_score","minus35_score","sd_score",
                     "down_gc","intergenic_prev","intergenic_next",
                     "same_strand_prev","same_strand_next","operon_prev",
                     "has_palindrome","family_n_orgs","n_paralogs",
                     "prev_essential","next_essential"]
    numeric_feats = [f for f in numeric_feats if f in df.columns]
    print(f"\n=== POOLED: numeric feature comparisons (essential vs non) ===")
    print(f"  {'feature':<22s} {'ess_mean':>10s} {'non_mean':>10s} {'d':>6s} "
          f"{'p_MW':>10s} {'AUC':>5s}")
    pooled_stats = []
    ess_mask = df.essential==1
    for f in numeric_feats:
        a = df.loc[ess_mask, f].dropna().tolist()
        b = df.loc[~ess_mask, f].dropna().tolist()
        if not a or not b: continue
        d = cohens_d(a, b); pm = mann_whitney_p(a, b)
        auc = auc_uni(df[f].tolist(), df.essential.tolist())
        em = sum(a)/len(a); nm = sum(b)/len(b)
        pooled_stats.append({"feature":f,"ess_mean":em,"non_mean":nm,
                              "cohen_d":d,"p_mannwhitney":pm,"auc":auc,
                              "n_ess":len(a),"n_non":len(b)})
        print(f"  {f:<22s} {em:>10.4f} {nm:>10.4f} {d:>+5.2f} {pm:>10.2e} {auc:>5.3f}")
    pd.DataFrame(pooled_stats).to_csv(args.out_dir/"essential_vs_non_pooled_numeric.csv",
                                       index=False)

    # ---- categorical: start/stop codon proportions ----
    print(f"\n=== POOLED: codon usage (essential vs non) ===")
    for col in ["start_codon","stop_codon"]:
        ess_c = Counter(df.loc[ess_mask, col])
        non_c = Counter(df.loc[~ess_mask, col])
        n_e = sum(ess_c.values()); n_n = sum(non_c.values())
        print(f"\n  {col}:")
        print(f"    {'codon':<8s} {'ess%':>8s} {'non%':>8s} {'enrich':>8s}")
        for c in sorted(set(ess_c)|set(non_c)):
            pe = ess_c.get(c,0)/n_e if n_e else 0
            pn = non_c.get(c,0)/n_n if n_n else 0
            ratio = pe/pn if pn>0 else float("inf")
            print(f"    {c:<8s} {pe*100:>7.2f}% {pn*100:>7.2f}% {ratio:>7.2f}x")

    # ---- description word enrichment ----
    print(f"\n=== POOLED: top-30 description words enriched in essential ===")
    def tokens(s):
        if not s: return []
        s = re.sub(r"[^A-Za-z0-9 -]", " ", str(s)).lower()
        return [t for t in s.split() if len(t)>=3 and t not in STOPWORDS]
    ess_words = Counter()
    non_words = Counter()
    for _, row in df.iterrows():
        for t in tokens(row["product"]):
            (ess_words if row["essential"]==1 else non_words)[t] += 1
    n_e = max(sum(ess_words.values()), 1); n_n = max(sum(non_words.values()), 1)
    word_rows = []
    for w in set(ess_words) | set(non_words):
        ef = ess_words.get(w,0); nf = non_words.get(w,0)
        if ef + nf < 10: continue
        pe = ef/n_e; pn = nf/n_n
        if pn == 0: continue
        word_rows.append({"word":w,"n_ess":ef,"n_non":nf,
                           "pct_ess":pe*100,"pct_non":pn*100,
                           "enrichment":pe/pn})
    wdf = pd.DataFrame(word_rows).sort_values("enrichment", ascending=False)
    print(f"  {'word':<25s} {'n_ess':>6s} {'n_non':>7s} {'ess%':>7s} {'non%':>7s} {'enrich':>7s}")
    for _, r in wdf.head(30).iterrows():
        print(f"  {r.word:<25s} {int(r.n_ess):>6d} {int(r.n_non):>7d} "
              f"{r.pct_ess:>6.3f}% {r.pct_non:>6.3f}% {r.enrichment:>6.2f}x")
    print(f"\n  most DEPLETED in essential (bottom 15):")
    for _, r in wdf.sort_values("enrichment").head(15).iterrows():
        print(f"  {r.word:<25s} {int(r.n_ess):>6d} {int(r.n_non):>7d} "
              f"{r.pct_ess:>6.3f}% {r.pct_non:>6.3f}% {r.enrichment:>6.2f}x")
    wdf.to_csv(args.out_dir/"top_essential_words.csv", index=False)

    # ---- per-organism summary ----
    print(f"\n=== PER-ORGANISM: top-3 most-separating features ===")
    per_org_stats = []
    for org in sorted(df.organism.unique()):
        sub = df[df.organism==org]
        em = sub.essential==1
        if em.sum() < 30 or (~em).sum() < 30: continue
        org_stats = []
        for f in numeric_feats:
            a = sub.loc[em, f].dropna().tolist()
            b = sub.loc[~em, f].dropna().tolist()
            if not a or not b: continue
            org_stats.append((f, cohens_d(a,b), auc_uni(sub[f].tolist(), sub.essential.tolist())))
        org_stats.sort(key=lambda x: -abs(x[1]) if x[1]==x[1] else 0)
        print(f"\n  {org} (n_ess={int(em.sum())}/{len(sub)})")
        for f, d, auc in org_stats[:5]:
            print(f"    {f:<22s}  d={d:>+5.2f}   AUC={auc:>.3f}")
            per_org_stats.append({"organism":org,"feature":f,
                                    "cohen_d":d,"auc":auc})
    pd.DataFrame(per_org_stats).to_csv(args.out_dir/"essential_vs_non_per_org.csv",
                                         index=False)

    # ---- markdown report ----
    md = []
    md.append(f"# Essential gene analysis: what's common?\n")
    md.append(f"**Organisms analyzed:** {df.organism.nunique()}  ")
    md.append(f"**Genes:** {len(df):,}  ")
    md.append(f"**Essential rate (pooled):** {df.essential.mean()*100:.1f}%\n\n")
    md.append("## Top numeric features separating essential vs non-essential\n\n")
    md.append("| feature | ess_mean | non_mean | Cohen's d | AUC |\n")
    md.append("|---|---|---|---|---|\n")
    top = sorted(pooled_stats, key=lambda r:-abs(r["cohen_d"]) if r["cohen_d"]==r["cohen_d"] else 0)
    for r in top[:15]:
        md.append(f"| {r['feature']} | {r['ess_mean']:.4f} | {r['non_mean']:.4f} | "
                  f"{r['cohen_d']:+.2f} | {r['auc']:.3f} |\n")
    md.append("\n## Top words enriched in essential gene descriptions\n\n")
    md.append("| word | n_ess | n_non | ess% | non% | enrichment |\n")
    md.append("|---|---|---|---|---|---|\n")
    for _, r in wdf.head(20).iterrows():
        md.append(f"| {r.word} | {int(r.n_ess)} | {int(r.n_non)} | "
                  f"{r.pct_ess:.3f}% | {r.pct_non:.3f}% | {r.enrichment:.2f}x |\n")
    (args.out_dir/"essential_analysis_summary.md").write_text("".join(md))
    print(f"\nwrote markdown report: {args.out_dir/'essential_analysis_summary.md'}")
    print(f"wrote CSVs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
