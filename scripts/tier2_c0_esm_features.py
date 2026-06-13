"""TIER 2 C0 -- does a frozen ESM-2 encoder beat conservation (family_frac)?

The decisive follow-up to Stage C's null result. Stage C showed: at 48-org
scale, a coupling head learned FROM SCRATCH adds nothing over the marginal
(family_frac) -- but it cleared Gate 1 (it wasn't fitting phylogeny). The open
question that decides what comes next:

    Is the bottleneck the ARCHITECTURE or the DATA SCALE?

C0 isolates it. Instead of learning per-OG embeddings from 48 organisms, we
take FROZEN ESM-2 protein-language-model embeddings (trained on ~250M proteins)
as the per-gene representation, and ask a simple linear probe:

    does ESM-2 beat family_frac at predicting essentiality --
    especially on the HARD SLICE (weak-conservation OGs) where family_frac
    only reaches ~0.20 AUPRC and there is real headroom?

Three comparisons per held-out clade (leak-free):
    family_frac        : the conservation bar (scalar, no training)
    ESM-2 linear probe : logistic regression on frozen embeddings
    ESM-2 + family_frac: do they ADD, or is ESM just re-encoding conservation?

VERDICT LOGIC:
    If ESM (or ESM+ff) beats family_frac by >=10% relative on the hard slice
    -> richer per-gene features unblock the signal; the pretrained-encoder
       route (and ultimately the pangenome bet) is justified.
    If ESM does NOT beat family_frac on the hard slice
    -> the ceiling is the DATA, not the features; accept the conservation
       ceiling for Tier 1, and the pangenome is the only remaining lever.

MAPPING (all local):
    labels.csv (org, locus_tag, essential)
      x orthology_features (og_id, family_frac_essential_leakfree)
      x GFF (locus_tag -> protein_id)
      x protein.faa (protein_id -> sequence)

--smoke : sandbox, no torch/esm. Validates the locus_tag->sequence join
          coverage and the family_frac bar on real cached genomes. This is the
          part that can silently fail; we test it before paying for GPU.
--real  : Colab+GPU. Embeds with ESM-2 (cached to disk), runs the probes,
          prints per-clade comparison + verdict.
"""
from __future__ import annotations
import argparse, gzip, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
LAB = REPO / "data" / "drive_import" / "labels"
CACHE = REPO / "data" / "drive_import" / "genome_cache"
OUT = REPO / "outputs" / "tier2"
ESM_CACHE = OUT / "esm_cache"
sys.path.insert(0, str(SCRIPTS))

from tier2_stage_b_coupling_model import metrics            # noqa: E402

FF_COL = "family_frac_essential_leakfree"
HARD_SLICE_MAX_SUPPORT = 10
PERF_GATE_REL = 0.10
DEFAULT_MODEL = "esm2_t12_35M_UR50D"   # small + fast; bump to t33_650M to scale
MAX_AA = 1022


# ---- sequence mapping ----------------------------------------------------
def gff_locus_to_protein(gff_path: Path):
    """locus_tag -> protein_id from CDS rows."""
    out = {}
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#") or "\tCDS\t" not in line:
                continue
            attr = line.rstrip("\n").split("\t")[8]
            kv = dict(p.split("=", 1) for p in attr.split(";") if "=" in p)
            lt, pid = kv.get("locus_tag"), kv.get("protein_id")
            if lt and pid:
                out.setdefault(lt, pid)
    return out


def faa_sequences(faa_path: Path):
    """protein_id -> amino-acid sequence."""
    seqs, pid, buf = {}, None, []
    opener = gzip.open if str(faa_path).endswith(".gz") else open
    with opener(faa_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if pid:
                    seqs[pid] = "".join(buf)
                pid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if pid:
            seqs[pid] = "".join(buf)
    return seqs


def build_gene_table(orgs, acc_map, with_seq=True):
    """(org, locus_tag, og_id, essential, family_frac[, seq]) for given orgs."""
    import pandas as pd
    lab = pd.read_csv(LAB / "labels.csv",
                      usecols=["organism", "locus_tag", "essential"])
    orth = pd.read_csv(LAB / "orthology_features.csv",
                       usecols=["organism", "locus_tag", "og_id", FF_COL])
    df = lab.merge(orth, on=["organism", "locus_tag"], how="inner")
    df = df[df.organism.isin(orgs)].copy()
    if not with_seq:
        return df
    seqs = []
    for org in orgs:
        acc = acc_map.get(org)
        gff = CACHE / str(acc) / "genomic.gff"
        faa = CACHE / str(acc) / "protein.faa"
        if not (acc and gff.exists() and faa.exists()):
            continue
        l2p = gff_locus_to_protein(gff)
        pseq = faa_sequences(faa)
        sub = df[df.organism == org]
        for lt in sub.locus_tag:
            pid = l2p.get(lt)
            s = pseq.get(pid) if pid else None
            if s:
                seqs.append((org, lt, s[:MAX_AA]))
    sdf = pd.DataFrame(seqs, columns=["organism", "locus_tag", "seq"])
    return df.merge(sdf, on=["organism", "locus_tag"], how="left")


def hard_mask(df, train_support):
    import numpy as np
    sup = df.og_id.map(train_support).fillna(0).to_numpy()
    return sup < HARD_SLICE_MAX_SUPPORT


# ---- ESM embedding (real mode only) -------------------------------------
def embed_with_esm(df, model_name, batch=16):
    """Mean-pooled frozen ESM-2 embedding per gene, cached per org to disk."""
    import torch, esm, numpy as np, pandas as pd
    ESM_CACHE.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = getattr(esm.pretrained, model_name)()
    bc = alphabet.get_batch_converter()
    model.eval().to(dev)
    n_layers = model.num_layers
    embs = {}
    for org, sub in df.dropna(subset=["seq"]).groupby("organism"):
        cf = ESM_CACHE / f"{model_name}_{org}.parquet"
        if cf.exists():
            embs[org] = pd.read_parquet(cf); continue
        rows, vecs = sub[["locus_tag", "seq"]].values.tolist(), []
        for i in range(0, len(rows), batch):
            chunk = [(lt, s) for lt, s in rows[i:i + batch]]
            _, _, toks = bc(chunk)
            toks = toks.to(dev)
            with torch.no_grad():
                rep = model(toks, repr_layers=[n_layers])["representations"][n_layers]
            for j, (lt, s) in enumerate(chunk):
                L = min(len(s), MAX_AA)
                vecs.append([lt] + rep[j, 1:L + 1].mean(0).cpu().numpy().tolist())
        cols = ["locus_tag"] + [f"e{k}" for k in range(len(vecs[0]) - 1)]
        edf = pd.DataFrame(vecs, columns=cols)
        edf.to_parquet(cf, index=False)
        embs[org] = edf
        print(f"    embedded {org}: {len(edf)} genes -> {cf.name}")
    return pd.concat([e.assign(organism=o) for o, e in embs.items()],
                     ignore_index=True)


def probe(Xtr, ytr, Xte):
    """Standardized logistic-regression probe; returns test scores."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


# ---- real ----------------------------------------------------------------
def run_real(args):
    import pandas as pd, numpy as np
    clade_map = json.loads((OUT / "clade_holdout.json").read_text())
    mf = pd.read_csv(LAB / "genome_cache_manifest.csv")
    acc_map = dict(zip(mf.organism, mf.accession))

    # all orgs we touch (train uses everything covered & not held)
    all_orgs = sorted({o for orgs in clade_map.values() for o in orgs})
    print("building gene table + sequences ...")
    df = build_gene_table(all_orgs, acc_map, with_seq=True)
    cov = df.seq.notna().mean()
    print(f"  genes: {len(df):,}  with sequence: {df.seq.notna().sum():,} "
          f"({cov:.1%})  -- uncovered orgs lack a cached genome (NaN accession)")

    print("embedding with ESM-2 (cached) ...")
    emb = embed_with_esm(df, args.model, batch=args.batch)
    ecols = [c for c in emb.columns if c.startswith("e")]
    df = df.merge(emb, on=["organism", "locus_tag"], how="inner")
    df = df.dropna(subset=[FF_COL])
    covered_orgs = sorted(df.organism.unique())
    print(f"  genes with ESM + family_frac: {len(df):,} "
          f"over {len(covered_orgs)} genome-backed orgs")

    # coverage-driven held-clade selection: only clades whose covered test
    # positives clear the floor (Ralstonia survives; uncovered clades skip)
    if args.clades:
        held_clades = args.clades.split(",")
    else:
        cand = []
        for clade, orgs in clade_map.items():
            sub = df[df.organism.isin(orgs)]
            npos = int(sub.essential.sum())
            if npos >= args.min_pos and len(set(orgs) & set(covered_orgs)) >= 1:
                cand.append((clade, npos))
        cand.sort(key=lambda r: r[1], reverse=True)
        held_clades = [c for c, _ in cand[:args.n_clades]]
    print(f"  held-out clades (genome-backed, >= {args.min_pos} pos): "
          f"{held_clades}")
    if not held_clades:
        print("  NO genome-backed clade has enough held-out positives. "
              "ESM C0 is blocked by the disjoint-data wall on the labeled set.",
              file=sys.stderr)
        return 3

    results = {}
    for clade in held_clades:
        held = set(clade_map.get(clade, []))
        tr = df[~df.organism.isin(held)]
        te = df[df.organism.isin(held)]
        if len(te) < 50 or te.essential.sum() < 10:
            print(f"  {clade}: too few held-out positives, skip"); continue
        sup = tr.groupby("og_id").size().to_dict()
        hmask = hard_mask(te, sup)
        yte = te.essential.values.astype(int)

        # 1. family_frac (no training)
        ff = metrics(yte, te[FF_COL].values)
        ff_h = metrics(yte[hmask], te[FF_COL].values[hmask]) \
            if hmask.sum() > 20 else {"auprc": None}
        # 2. ESM probe
        s_esm = probe(tr[ecols].values, tr.essential.values.astype(int),
                      te[ecols].values)
        em = metrics(yte, s_esm)
        em_h = metrics(yte[hmask], s_esm[hmask]) if hmask.sum() > 20 else {"auprc": None}
        # 3. ESM + family_frac
        Xtr = np.hstack([tr[ecols].values, tr[[FF_COL]].values])
        Xte = np.hstack([te[ecols].values, te[[FF_COL]].values])
        s_cmb = probe(Xtr, tr.essential.values.astype(int), Xte)
        cm = metrics(yte, s_cmb)
        cm_h = metrics(yte[hmask], s_cmb[hmask]) if hmask.sum() > 20 else {"auprc": None}

        print(f"\n  === {clade} (held {te.organism.nunique()} orgs, "
              f"trained on {tr.organism.nunique()} orgs, test n={len(te):,}, "
              f"pos={int(yte.sum())}, hard n={int(hmask.sum())}) ===")
        print(f"    family_frac : full AUPRC={fmt(ff['auprc'])}  "
              f"hard={fmt(ff_h['auprc'])}")
        print(f"    ESM-2 probe : full AUPRC={fmt(em['auprc'])}  "
              f"hard={fmt(em_h['auprc'])}")
        print(f"    ESM + ff    : full AUPRC={fmt(cm['auprc'])}  "
              f"hard={fmt(cm_h['auprc'])}")
        results[clade] = {"ff": ff, "ff_hard": ff_h, "esm": em,
                          "esm_hard": em_h, "comb": cm, "comb_hard": cm_h,
                          "n_test": len(te), "n_hard": int(hmask.sum())}

    verdict = decide_c0(results)
    (OUT / "c0_esm_results.json").write_text(
        json.dumps({"model": args.model, "results": results,
                    "verdict": verdict}, indent=2, default=str))
    print_verdict_c0(verdict)
    return 0


def decide_c0(results):
    import numpy as np
    esm_rel, comb_rel = [], []
    for c, r in results.items():
        ff = r["ff_hard"]["auprc"] or r["ff"]["auprc"]
        if not ff:
            continue
        e = r["esm_hard"]["auprc"] or r["esm"]["auprc"]
        cm = r["comb_hard"]["auprc"] or r["comb"]["auprc"]
        if e:
            esm_rel.append((e - ff) / ff)
        if cm:
            comb_rel.append((cm - ff) / ff)
    em = float(np.mean(esm_rel)) if esm_rel else None
    cmb = float(np.mean(comb_rel)) if comb_rel else None
    best = max([x for x in (em, cmb) if x is not None], default=None)
    return {"esm_mean_rel_hard": _r(em), "comb_mean_rel_hard": _r(cmb),
            "best_rel": _r(best), "threshold": PERF_GATE_REL,
            "pass": bool(best is not None and best >= PERF_GATE_REL)}


def _r(x):
    return round(x, 4) if x is not None else None


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def print_verdict_c0(v):
    print(f"\n{'='*60}\nC0 VERDICT -- ESM-2 vs conservation on the hard slice\n{'='*60}")
    print(f"  ESM-2 alone   rel-lift (hard): {fmt(v['esm_mean_rel_hard'])}")
    print(f"  ESM-2 + ff    rel-lift (hard): {fmt(v['comb_mean_rel_hard'])}")
    print(f"  threshold: +{v['threshold']:.0%} relative")
    if v["pass"]:
        print("\n  >>> ESM-2 BEATS conservation on the hard slice. The bottleneck")
        print("      was FEATURES, not just data scale. Pretrained-encoder route")
        print("      justified -> wire ESM into the coupling model; pangenome bet live.")
    else:
        print("\n  >>> ESM-2 does NOT beat conservation on the hard slice. The")
        print("      ceiling is the DATA, not the features. Accept the conservation")
        print("      ceiling for Tier 1; the pangenome is the only remaining lever.")


# ---- smoke ---------------------------------------------------------------
def run_smoke():
    import pandas as pd, numpy as np
    print("=" * 60)
    print("TIER 2 C0 -- smoke (sequence-join coverage + family_frac bar)")
    print("=" * 60)
    clade_map = json.loads((OUT / "clade_holdout.json").read_text())
    mf = pd.read_csv(LAB / "genome_cache_manifest.csv")
    acc_map = dict(zip(mf.organism, mf.accession))

    # pick a couple held clades and check sequence-join coverage (the part
    # that silently fails on Colab if the GFF/faa mapping is wrong)
    test_clades = ["pseudomonas", "ralstonia", "burkholderia"]
    any_orgs = []
    for c in test_clades:
        any_orgs += clade_map.get(c, [])
    any_orgs = sorted(set(any_orgs))
    print(f"  held-clade orgs to map: {any_orgs}")

    df = build_gene_table(any_orgs, acc_map, with_seq=True)
    print(f"  genes in those orgs: {len(df):,}")
    have = df.dropna(subset=["seq"])
    print(f"  T1 sequence-join coverage: {len(have):,}/{len(df):,} "
          f"({len(have)/max(len(df),1):.1%})")
    per_org = df.assign(has=df.seq.notna()).groupby("organism").has.mean()
    for o, frac in per_org.items():
        flag = "ok" if frac > 0.5 else "LOW"
        print(f"      {o:<28s} {frac:.1%}  [{flag}]")
    assert len(have) > 100, "sequence join produced too few genes"

    # family_frac bar present + hard slice computable
    df2 = df.dropna(subset=[FF_COL])
    sup = df2.groupby("og_id").size().to_dict()
    hmask = hard_mask(df2, sup)
    print(f"  T2 family_frac present: {len(df2):,} genes; "
          f"hard-slice genes: {int(hmask.sum()):,}")
    m = metrics(df2.essential.values.astype(int), df2[FF_COL].values)
    print(f"  T3 family_frac AUPRC (in-sample sanity): {fmt(m['auprc'])}")
    assert m["auprc"] is not None

    # mean seq length (ESM cost proxy)
    ln = have.seq.str.len()
    print(f"  T4 seq length: median={ln.median():.0f} aa, "
          f">{MAX_AA}aa (truncated): {(ln>=MAX_AA).sum()}")

    print(f"\n=== C0 SMOKE PASS ===")
    print(f"  locus_tag->protein->sequence join works on real cached genomes;")
    print(f"  family_frac bar + hard-slice machinery ready. Run --real on")
    print(f"  Colab (pip install fair-esm) to get the ESM-vs-conservation number.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clades", default="")
    ap.add_argument("--n_clades", type=int, default=3)
    ap.add_argument("--min_pos", type=int, default=50)
    args = ap.parse_args()
    if args.real:
        return run_real(args)
    return run_smoke()


if __name__ == "__main__":
    sys.exit(main())
