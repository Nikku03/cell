"""Full-scale essential-gene pattern analysis on the 91 cached
genomes + 175K labeled BERIL genes + orthology features.

For each cached organism with a clean labels.csv join:
  parse GFF      -> locus_tag, gene, product, start, end, strand
  parse protein  -> amino-acid sequence (per protein_id or locus_tag)
  match labels   -> essential 0/1
  match orth     -> og_id, family_n_organisms, n_paralogs_in_genome

Per gene compute:
  STRUCTURAL    length_nt, length_aa, strand, relative position
  NEIGHBOR      intergenic_prev/next, same_strand_prev/next, operon
  PROTEIN       GRAVY (hydrophobicity), pI, aromaticity, AA composition
  ORTHOLOGY     family_n_organisms, n_paralogs_in_genome
                family_frac_essential (averaged across folds)

Compare essential vs non-essential:
  numeric    -> mean +/- std, Cohen's d, Mann-Whitney p, AUC
  text       -> product-word enrichment
  per-org    -> top separating features
  pooled     -> across-organism summary
"""
from __future__ import annotations
import argparse, math, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"E":-3.5,"Q":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
PKA_NTERM, PKA_CTERM = 9.69, 2.34
PKA_SIDE = {"K":10.5,"R":12.4,"H":6.0,"D":3.65,"E":4.25,"C":8.33,"Y":10.07}
AROMATIC = set("FWY")
AAS = "ACDEFGHIKLMNPQRSTVWY"
STOPWORDS = set("a an and are as at be by for from has have in is it its of on or "
                "that the their these this to with putative hypothetical protein "
                "uncharacterized predicted family domain containing related dna rna "
                "small large subunit type system component conserved member "
                "n-terminal c-terminal binding region terminus".split())


def gravy(seq):
    s = [a for a in seq if a in KD]
    return sum(KD[a] for a in s)/len(s) if s else float("nan")


def aromaticity(seq):
    s = [a for a in seq if a in KD]
    return sum(1 for a in s if a in AROMATIC)/len(s) if s else float("nan")


def isoelectric(seq):
    n = Counter(a for a in seq if a in KD)
    lo, hi = 0.0, 14.0
    def q(pH):
        pos = 1/(1+10**(pH-PKA_NTERM))
        for aa, pka in (("K",PKA_SIDE["K"]),("R",PKA_SIDE["R"]),("H",PKA_SIDE["H"])):
            pos += n.get(aa,0)/(1+10**(pH-pka))
        neg = 1/(1+10**(PKA_CTERM-pH))
        for aa, pka in (("D",PKA_SIDE["D"]),("E",PKA_SIDE["E"]),
                          ("C",PKA_SIDE["C"]),("Y",PKA_SIDE["Y"])):
            neg += n.get(aa,0)/(1+10**(pka-pH))
        return pos - neg
    for _ in range(40):
        mid = (lo+hi)/2; c = q(mid)
        if abs(c) < 1e-3: return mid
        if c > 0: lo = mid
        else: hi = mid
    return (lo+hi)/2


def aa_comp(seq):
    s = [a for a in seq if a in KD]
    n = len(s); c = Counter(s)
    return {f"aa_{a}": c.get(a,0)/n if n else 0 for a in AAS}


def parse_faa(path):
    out = {}; cur = None; buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None: out[cur] = "".join(buf)
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cur is not None: out[cur] = "".join(buf)
    return out


def parse_gff_cds(path):
    """Returns CDS feats with locus_tag resolved through gene parent if
    necessary (older RefSeq GFFs put locus_tag on the 'gene' row)."""
    gene_attrs = {}; cds = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9: continue
            d = {}
            for kv in p[8].split(";"):
                if "=" in kv:
                    k,v = kv.split("=",1); d[k] = v
            if p[2] == "gene" and "ID" in d:
                gene_attrs[d["ID"]] = {"locus_tag": d.get("locus_tag",""),
                                        "gene":      d.get("gene","")}
            elif p[2] == "CDS":
                cds.append((p, d))
    feats = []
    for p, d in cds:
        lt = d.get("locus_tag",""); gene = d.get("gene","")
        old_lt = d.get("old_locus_tag","")
        if not lt and d.get("Parent","") in gene_attrs:
            ga = gene_attrs[d["Parent"]]
            lt = ga["locus_tag"]; gene = gene or ga["gene"]
        feats.append({
            "seqid":         p[0],
            "start":         int(p[3]),
            "end":           int(p[4]),
            "strand":        p[6],
            "locus_tag":     lt,
            "old_locus_tag": old_lt,
            "gene":          gene,
            "product":       d.get("product",""),
            "protein_id":    d.get("protein_id",""),
        })
    return feats


def mann_whitney_p(a, b):
    import numpy as np
    a = np.array([x for x in a if x == x])
    b = np.array([x for x in b if x == x])
    if len(a) < 5 or len(b) < 5: return float("nan")
    c = np.concatenate([a, b])
    r = c.argsort().argsort() + 1
    u1 = r[:len(a)].sum() - len(a)*(len(a)+1)/2
    u = min(u1, len(a)*len(b) - u1)
    mu = len(a)*len(b)/2
    s = math.sqrt(len(a)*len(b)*(len(a)+len(b)+1)/12)
    if s == 0: return 1.0
    return math.erfc(abs((u-mu)/s)/math.sqrt(2))


def cohens_d(a, b):
    import numpy as np
    a = np.array([x for x in a if x == x])
    b = np.array([x for x in b if x == x])
    if len(a) < 2 or len(b) < 2: return float("nan")
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                   / (len(a)+len(b)-2))
    return (a.mean()-b.mean())/s if s else float("nan")


def auc_uni(scores, labels):
    import numpy as np
    s = np.array(scores, dtype=float); y = np.array(labels)
    m = ~np.isnan(s); s, y = s[m], y[m]
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    o = np.argsort(-s); y2 = y[o]
    np_pos = int(y2.sum()); nn = len(y2) - np_pos
    rs = np.where(y2 == 1)[0].sum() + np_pos
    return 1 - (rs - np_pos*(np_pos+1)/2)/(np_pos*nn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",   type=Path,
                   default=REPO_ROOT/"data/drive_import/genome_cache")
    p.add_argument("--manifest",    type=Path,
                   default=REPO_ROOT/"data/drive_import/labels/genome_cache_manifest.csv")
    p.add_argument("--labels-csv",  type=Path,
                   default=REPO_ROOT/"data/drive_import/labels/labels.csv")
    p.add_argument("--orth-csv",    type=Path,
                   default=REPO_ROOT/"data/drive_import/labels/orthology_features.csv")
    p.add_argument("--out-dir",     type=Path,
                   default=REPO_ROOT/"outputs/essential_full")
    p.add_argument("--min-join",    type=float, default=0.3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd, numpy as np

    print("Loading labels + orthology + manifest ...")
    labels = pd.read_csv(args.labels_csv)
    orth = pd.read_csv(args.orth_csv)
    mf = pd.read_csv(args.manifest)
    print(f"  labels: {len(labels):,} rows  ({labels.organism.nunique()} organisms)")
    print(f"  orth:   {len(orth):,} rows")
    print(f"  manifest: {len(mf)} entries")

    # which orgs have a usable cache?
    eligible = mf[(mf.kind=="ours") & (mf.accession.notna()) &
                  (mf.join_rate.notna()) & (mf.join_rate >= args.min_join)]
    print(f"  eligible orgs (join >= {args.min_join}): {len(eligible)}")

    # also include DEG orgs (kind=='deg', no join_rate needed; they only have ESS list)
    # Skipping DEG-only orgs here -- this analysis needs the 0/1 labels, which we
    # only get from labels.csv (our organisms).

    fold_cols = [c for c in orth.columns if c.startswith("family_frac_essential_fold")]
    orth["ff_mean"] = orth[fold_cols].mean(axis=1) if fold_cols else float("nan")
    orth_idx = orth.set_index(["organism","locus_tag"])

    all_rows = []
    for _, mfr in eligible.iterrows():
        org = mfr.organism; acc = mfr.accession
        gd = args.cache_dir / acc
        gff_p = gd / "genomic.gff"; faa_p = gd / "protein.faa"
        if not gff_p.exists() or not faa_p.exists():
            print(f"  {org}: missing files in {gd}, skip"); continue

        feats = parse_gff_cds(gff_p)
        feats.sort(key=lambda r: (r["seqid"], r["start"]))
        seqs = parse_faa(faa_p)
        # also index proteins by locus_tag (some assemblies key by it directly)
        contig_lens = defaultdict(int)
        for f in feats:
            if f["end"] > contig_lens[f["seqid"]]: contig_lens[f["seqid"]] = f["end"]

        # labels for this org
        ol = labels[labels.organism == org]
        lt2e = dict(zip(ol.locus_tag.astype(str), ol.essential.astype(int)))
        matched_ess, matched_old = 0, 0
        n_added = 0
        for i, f in enumerate(feats):
            beril = lt2e.get(f["locus_tag"])
            if beril is None and f["old_locus_tag"]:
                beril = lt2e.get(f["old_locus_tag"])
                if beril is not None: matched_old += 1
            else:
                if beril is not None: matched_ess += 1
            if beril is None: continue

            prev_f = feats[i-1] if i>0 and feats[i-1]["seqid"]==f["seqid"] else None
            next_f = feats[i+1] if i+1<len(feats) and feats[i+1]["seqid"]==f["seqid"] else None
            ig_prev = (f["start"] - prev_f["end"] - 1) if prev_f else None
            ig_next = (next_f["start"] - f["end"] - 1) if next_f else None
            ss_prev = int(prev_f["strand"] == f["strand"]) if prev_f else None
            ss_next = int(next_f["strand"] == f["strand"]) if next_f else None
            operon = int(ig_prev is not None and ig_prev < 50 and ss_prev == 1)

            # protein sequence
            seq = ""
            for key in (f["protein_id"], f["locus_tag"], f["gene"]):
                if key and key in seqs:
                    seq = seqs[key]; break

            length_nt = f["end"] - f["start"] + 1
            rec = {
                "organism": org, "locus_tag": f["locus_tag"], "gene": f["gene"],
                "product": f["product"],
                "length_nt": length_nt, "length_aa": int(length_nt/3 - 1),
                "strand": 1 if f["strand"] == "+" else -1,
                "rel_position": f["start"]/contig_lens[f["seqid"]],
                "intergenic_prev": ig_prev, "intergenic_next": ig_next,
                "same_strand_prev": ss_prev, "same_strand_next": ss_next,
                "operon_prev": operon,
                "essential": beril,
            }
            if seq:
                rec["gravy"] = gravy(seq)
                rec["pI"]    = isoelectric(seq)
                rec["aromaticity"] = aromaticity(seq)
                rec.update(aa_comp(seq))
            # orth features (per (org, locus_tag))
            key = (org, f["locus_tag"])
            if key in orth_idx.index:
                row = orth_idx.loc[key]
                # in case of duplicates, take first
                if hasattr(row, "iloc"): row = row.iloc[0] if hasattr(row, "shape") and len(row.shape)==2 else row
                for c in ["family_n_organisms","n_paralogs_in_genome","ff_mean"]:
                    if c in row.index if hasattr(row,"index") else False:
                        v = row[c]
                        if pd.notna(v): rec[c] = float(v)
            all_rows.append(rec); n_added += 1
        print(f"  {org} ({acc}): {n_added:,} genes added "
              f"(via_lt={matched_ess}, via_old_lt={matched_old})")

    if not all_rows:
        print("ERROR: no rows", file=sys.stderr); return 1
    df = pd.DataFrame(all_rows)
    print(f"\n=== Master table: {len(df):,} labelled genes across "
          f"{df.organism.nunique()} organisms ===")
    print(f"  pooled essential rate: {df.essential.mean()*100:.1f}%")
    print(f"  per-org:")
    print(df.groupby("organism").agg(n=("essential","count"),
                                       n_ess=("essential","sum"),
                                       pct=("essential","mean")).to_string())
    df.to_csv(args.out_dir / "per_gene_full.csv", index=False)

    # ---- numeric feature comparisons ----
    base = ["length_nt","length_aa","strand","rel_position",
            "intergenic_prev","intergenic_next",
            "same_strand_prev","same_strand_next","operon_prev",
            "gravy","pI","aromaticity",
            "family_n_organisms","n_paralogs_in_genome","ff_mean"]
    aa_cols = [f"aa_{a}" for a in AAS]
    numeric = [c for c in base + aa_cols if c in df.columns]

    print(f"\n=== POOLED: essential vs non-essential (sorted by |d|) ===")
    print(f"  {'feature':<24s} {'ess_mean':>10s} {'non_mean':>10s} {'d':>6s} "
          f"{'p_MW':>9s} {'AUC':>5s}")
    stats = []
    em = df.essential == 1
    for f in numeric:
        a = df.loc[em, f].dropna().tolist()
        b = df.loc[~em, f].dropna().tolist()
        if len(a) < 50 or len(b) < 50: continue
        d = cohens_d(a, b); p = mann_whitney_p(a, b)
        auc = auc_uni(df[f].tolist(), df.essential.tolist())
        stats.append({"feature":f,"ess_mean":sum(a)/len(a),
                       "non_mean":sum(b)/len(b),
                       "cohen_d":d,"p_mannwhitney":p,"auc":auc,
                       "n_ess":len(a),"n_non":len(b)})
    stats.sort(key=lambda r: -abs(r["cohen_d"]) if r["cohen_d"]==r["cohen_d"] else 0)
    for r in stats[:30]:
        print(f"  {r['feature']:<24s} {r['ess_mean']:>10.4f} {r['non_mean']:>10.4f} "
              f"{r['cohen_d']:>+5.2f} {r['p_mannwhitney']:>9.2e} {r['auc']:>5.3f}")
    pd.DataFrame(stats).to_csv(args.out_dir / "pooled_numeric.csv", index=False)

    # ---- per-organism dominant features ----
    print(f"\n=== PER-ORGANISM: top 3 separating features per organism ===")
    per_org = []
    for org in sorted(df.organism.unique()):
        sub = df[df.organism == org]
        em_o = sub.essential == 1
        if em_o.sum() < 30 or (~em_o).sum() < 30: continue
        rows = []
        for f in numeric:
            a = sub.loc[em_o, f].dropna().tolist()
            b = sub.loc[~em_o, f].dropna().tolist()
            if len(a) < 10 or len(b) < 10: continue
            d = cohens_d(a, b)
            auc = auc_uni(sub[f].tolist(), sub.essential.tolist())
            rows.append((f, d, auc))
        rows.sort(key=lambda x: -abs(x[1]) if x[1]==x[1] else 0)
        top = rows[:3]
        for f, d, auc in top:
            per_org.append({"organism":org,"feature":f,"cohen_d":d,"auc":auc})
        if top:
            print(f"  {org:<32s} (n={len(sub)}, n_ess={int(em_o.sum())})  "
                  f"{top[0][0]} d={top[0][1]:+.2f}, "
                  f"{top[1][0]} d={top[1][1]:+.2f}, "
                  f"{top[2][0]} d={top[2][1]:+.2f}")
    pd.DataFrame(per_org).to_csv(args.out_dir / "per_org_top.csv", index=False)

    # ---- word enrichment ----
    print(f"\n=== POOLED: words enriched in essential gene products ===")
    def tokens(s):
        if not s: return []
        s = re.sub(r"[^A-Za-z0-9 -]"," ", str(s)).lower()
        return [t for t in s.split() if len(t) >= 3 and t not in STOPWORDS]
    ess_g, non_g = Counter(), Counter()
    n_ess_genes = int((df.essential==1).sum())
    n_non_genes = int((df.essential==0).sum())
    for _, r in df.iterrows():
        ws = set(tokens(r["product"]))
        for t in ws:
            (ess_g if r.essential==1 else non_g)[t] += 1
    rows = []
    for w in set(ess_g) | set(non_g):
        ne = ess_g.get(w,0); nn = non_g.get(w,0)
        if ne+nn < 30: continue
        pe = ne / max(n_ess_genes,1); pn = nn / max(n_non_genes,1)
        if pn == 0: continue
        rows.append({"word":w,"n_ess":ne,"n_non":nn,
                      "pct_ess":pe*100,"pct_non":pn*100,
                      "enrichment":pe/pn,
                      "log_odds":math.log((pe+1e-6)/(1-pe+1e-6)) -
                                  math.log((pn+1e-6)/(1-pn+1e-6))})
    wdf = pd.DataFrame(rows).sort_values("log_odds", ascending=False)
    print(f"  TOP 25 enriched in essential:")
    print(f"  {'word':<25s} {'n_ess':>7s} {'n_non':>8s} {'ess%':>6s} {'non%':>6s} {'enrich':>7s}")
    for _, r in wdf.head(25).iterrows():
        print(f"  {r.word:<25s} {int(r.n_ess):>7d} {int(r.n_non):>8d} "
              f"{r.pct_ess:>5.2f}% {r.pct_non:>5.2f}% {r.enrichment:>6.2f}x")
    print(f"\n  TOP 15 depleted in essential (enriched in non-essential):")
    for _, r in wdf.tail(15).iterrows():
        print(f"  {r.word:<25s} {int(r.n_ess):>7d} {int(r.n_non):>8d} "
              f"{r.pct_ess:>5.2f}% {r.pct_non:>5.2f}% {r.enrichment:>6.2f}x")
    wdf.to_csv(args.out_dir / "word_enrichment.csv", index=False)

    # ---- markdown report ----
    md = []
    md.append("# Essential gene patterns: full-scale analysis\n\n")
    md.append(f"**Organisms analyzed:** {df.organism.nunique()}  \n")
    md.append(f"**Total labelled genes:** {len(df):,}  \n")
    md.append(f"**Pooled essential rate:** {df.essential.mean()*100:.1f}%\n\n")
    md.append("## Top features distinguishing essential vs non-essential\n\n")
    md.append("Sorted by |Cohen's d| (effect size). AUC = univariate classifier.\n\n")
    md.append("| feature | ess_mean | non_mean | Cohen's d | AUC | p (MW) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in stats[:25]:
        md.append(f"| {r['feature']} | {r['ess_mean']:.4f} | {r['non_mean']:.4f} | "
                  f"{r['cohen_d']:+.2f} | {r['auc']:.3f} | {r['p_mannwhitney']:.1e} |\n")
    md.append("\n## Words enriched in essential gene products\n\n")
    md.append("| word | n_ess | n_non | ess% | non% | enrichment |\n")
    md.append("|---|---|---|---|---|---|\n")
    for _, r in wdf.head(25).iterrows():
        md.append(f"| {r.word} | {int(r.n_ess)} | {int(r.n_non)} | "
                  f"{r.pct_ess:.2f}% | {r.pct_non:.2f}% | {r.enrichment:.2f}x |\n")
    md.append("\n## Words depleted in essential (enriched in non-essential)\n\n")
    md.append("| word | n_ess | n_non | ess% | non% | enrichment |\n")
    md.append("|---|---|---|---|---|---|\n")
    for _, r in wdf.tail(15).iterrows():
        md.append(f"| {r.word} | {int(r.n_ess)} | {int(r.n_non)} | "
                  f"{r.pct_ess:.2f}% | {r.pct_non:.2f}% | {r.enrichment:.2f}x |\n")
    (args.out_dir / "summary.md").write_text("".join(md))
    print(f"\nwrote markdown -> {args.out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
