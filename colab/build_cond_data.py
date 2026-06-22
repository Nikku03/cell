"""Stage 2a -- build the conditional-essentiality dataset from feba.db.

Produces a (gene, condition) training set joined to the existing gene
representation (af_msa_cache), with three things the naive extraction lacked:

  1. COMPOSITIONAL condition features (multi-hot over shared components:
     media class, carbon source, nitrogen source, stress type, electron
     acceptor, supplement, aerobicity). This is what lets a model GENERALISE
     to a held-out condition -- an unseen 'L-Valine' still lights up the
     'amino_acid' + 'defined_media' dims seen via other amino acids.
  2. CONTINUOUS target: per-(gene, condition) mean z-scored fitness (fit_z),
     not only the binary essential call -- preserves the full signal.
  3. PER-ORG NOISE FLOOR: median replicate-pair Spearman per organism = the
     ceiling any predictor must be judged against.

Join: feba locusId must match our cache locus_tag (true for native-ID beril
orgs; GFF-recalled orgs won't join and are reported/dropped).

Outputs (under data/drive_import/cond/):
  cond_pairs.npz       gene_idx, cond_feat[P,D], y_fit[P], y_ess[P], org_idx[P],
                       clade[P], cluster_id[P]
  cond_meta.json       component names, per-org noise floor, dense org list,
                       n_pairs, join report
"""
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

ESS_FIT_T, ESS_T_T = -2.0, 4.0

# reuse the org map from the extractor
from extract_feba_conditions import MANUAL_MAP, map_feba_to_our_orgs, pick_table

# ---- compositional condition parser ---------------------------------------
# Each condition_1 / media string is scanned for keywords; every hit sets the
# corresponding component dimension. A condition is the UNION of its hits, so
# held-out conditions still share dims with training conditions.
COMPONENTS = [
    # (dim_name, [keywords])
    ("media_rich",        ["lb", "rich", "broth", "bhi", "todd", "mueller", "tye", "pye", "blood", "columbia", "2xyt", "nutrient"]),
    ("media_minimal",     ["minimal", "m9", "mops", "defined", "m63", "gmm"]),
    ("carbon_glucose",    ["glucose", "dextrose"]),
    ("carbon_pyruvate",   ["pyruvate"]),
    ("carbon_lactate",    ["lactate"]),
    ("carbon_acetate",    ["acetate"]),
    ("carbon_formate",    ["formate"]),
    ("carbon_succinate",  ["succinate", "fumarate", "malate", "citrate"]),
    ("carbon_sugar_other",["fructose", "galactose", "mannose", "xylose", "arabinose", "sucrose", "maltose", "ribose", "sorbitol", "glycerol"]),
    ("nitrogen_ammonium", ["ammonium", "ammonia", "nh4"]),
    ("nitrogen_nitrate",  ["nitrate"]),
    ("aa_supplement",     ["alanine", "serine", "valine", "methionine", "leucine", "isoleucine", "glutam", "aspart", "lysine", "arginine", "histidine", "threonine", "proline", "glycine", "cysteine", "tryptophan", "tyrosine", "phenylalanine", "amino acid", "casamino"]),
    ("stress_oxidative",  ["peroxide", "h2o2", "paraquat", "oxidative", "menadione"]),
    ("stress_antibiotic", ["cin", "mycin", "cillin", "tetracyc", "chloramphenicol", "rifampic", "nalidixic", "polymyxin", "tobramycin", "kanamycin", "spectinomycin", "fluoroquinolone", "antibiotic", "carbenicillin", "gentamicin"]),
    ("stress_metal",      ["copper", "zinc", "cadmium", "cobalt", "nickel", "iron", "manganese", "chromium", "arsen", "metal", "fe(", "cu(", "zn"]),
    ("stress_ph",         ["ph ", "ph3", "ph4", "ph5", "ph6", "ph8", "ph9", "acidic", "alkaline"]),
    ("stress_osmotic",    ["nacl", "sodium chloride", "sorbitol", "osmotic", "salt"]),
    ("stress_temp",       ["temperature", "heat", "cold", "37c", "42c", "degrees"]),
    ("electron_nitrite",  ["nitrite"]),
    ("electron_sulfate",  ["sulfate", "sulfite", "thiosulfate", "dmso", "tmao"]),
    ("host_invivo",       ["murine", "in vivo", "host", "infection", "sputum", "cystic", "macrophage", "mouse", "pneumonia"]),
    ("anaerobic",         ["anaerobic"]),
    ("aerobic",           ["aerobic"]),   # NOTE: handled specially (substring of anaerobic)
    ("stress_bile",       ["bile", "deoxycholate"]),
    ("carbon_generic",    ["carbon source"]),
]
COMP_NAMES = [c[0] for c in COMPONENTS]


def parse_condition(media, aerobic, condition_1):
    s = f"{media} {aerobic} {condition_1}".lower()
    vec = np.zeros(len(COMPONENTS), np.float32)
    for i, (name, kws) in enumerate(COMPONENTS):
        if name == "aerobic":
            # 'aerobic' is a substring of 'anaerobic' -> only set when truly aerobic
            vec[i] = 1.0 if ("aerobic" in s and "anaerobic" not in s) else 0.0
            continue
        if any(k in s for k in kws):
            vec[i] = 1.0
    return vec


def org_noise_floor(sub, clusters):
    """median replicate-pair Spearman of per-gene fit within clusters."""
    from scipy.stats import spearmanr
    rhos = []
    for cl, g in clusters:
        exps = list(dict.fromkeys(g["expName"].tolist()))
        if len(exps) < 2:
            continue
        for i in range(len(exps)):
            for j in range(i + 1, min(i + 3, len(exps))):
                a = sub[sub.expName == exps[i]].set_index("locusId").fit
                b = sub[sub.expName == exps[j]].set_index("locusId").fit
                common = a.index.intersection(b.index)
                if len(common) >= 100:
                    r, _ = spearmanr(a.loc[common], b.loc[common])
                    if r == r:
                        rhos.append(r)
    return float(np.median(rhos)) if rhos else float("nan")


def main(db_path, cache_path, labels_dir, out_dir, min_conditions):
    import pandas as pd
    import sqlite3
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path)
    if not db_path.exists():
        from kappa_floor import fetch_feba
        print(f"downloading feba.db -> {db_path}")
        fetch_feba(db_path)

    # ---- cache gene index: (org_name, locus_tag) -> gene_idx ----
    cache = np.load(cache_path, allow_pickle=True)
    orgs = list(cache["orgs"]); meta_org = cache["meta_org"]; lt = cache["lt"]
    clade = cache["clade"]
    gene_index = {}
    for gi in range(len(lt)):
        gene_index[(str(orgs[int(meta_org[gi])]), str(lt[gi]))] = gi
    print(f"cache: {len(lt):,} genes across {len(orgs)} orgs")

    con = sqlite3.connect(str(db_path))
    exp_t = pick_table(con, ["Experiment", "Experiments", "Exp"], "experiment")
    fit_t = pick_table(con, ["GeneFitness", "Fitness", "GeneFit"], "fitness")
    exp = pd.read_sql(f"SELECT orgId, expName, media, aerobic, condition_1 FROM {exp_t}", con)
    for c in ("media", "aerobic", "condition_1"):
        exp[c] = exp[c].fillna("-").astype(str).str.strip()
    exp["cluster"] = exp.media + " | " + exp.aerobic + " | " + exp.condition_1

    dense = (exp.groupby("orgId").cluster.nunique().sort_values(ascending=False))
    dense_orgs = dense[dense >= min_conditions].index.tolist()
    print(f"dense orgs (>= {min_conditions} conditions): {len(dense_orgs)}")

    our_orgs = set(orgs)
    org_map = map_feba_to_our_orgs(dense_orgs, our_orgs)
    print(f"of those, mapped to our cache: {len(org_map)}")

    P_gene = []; P_feat = []; P_fit = []; P_ess = []; P_org = []; P_clade = []; P_clu = []
    noise = {}; join_report = {}; cluster_vocab = {}; cluster_next = 0

    for fb in dense_orgs:
        if fb not in org_map:
            continue
        our = org_map[fb]
        oexp = exp[exp.orgId == fb]
        usable = {c for c, g in oexp.groupby("cluster") if g.expName.nunique() >= 2}
        if not usable:
            continue
        sub = pd.read_sql(f"SELECT locusId, expName, fit, t FROM {fit_t} WHERE orgId=?",
                          con, params=(fb,))
        if sub.empty:
            continue
        sub.locusId = sub.locusId.astype(str)
        # join check
        cache_hits = sum(1 for lid in sub.locusId.unique()
                         if (our, lid) in gene_index)
        match_pct = 100 * cache_hits / max(1, sub.locusId.nunique())
        join_report[fb] = dict(our_org=our, feba_genes=int(sub.locusId.nunique()),
                               cache_matched=int(cache_hits), match_pct=round(match_pct, 1))
        if match_pct < 30:
            print(f"  [skip] {fb}->{our}: only {match_pct:.0f}% gene match")
            continue
        # per-experiment z-score of fit
        sub["fit_z"] = sub.groupby("expName").fit.transform(
            lambda x: (x - x.median()) / (x.std() + 1e-6))
        sub["is_ess"] = ((sub.fit < ESS_FIT_T) & (sub.t.abs() >= ESS_T_T)).astype(int)
        clusters = [(c, g) for c, g in oexp.groupby("cluster") if c in usable]
        noise[our] = org_noise_floor(sub, clusters)
        m = sub.merge(oexp[["expName", "cluster"]], on="expName", how="inner")
        m = m[m.cluster.isin(usable)]
        agg = (m.groupby(["locusId", "cluster"])
                 .agg(fit_z=("fit_z", "mean"), is_ess=("is_ess", "mean"),
                      n=("expName", "nunique")).reset_index())
        n_pairs_org = 0
        for row in agg.itertuples():
            gi = gene_index.get((our, row.locusId))
            if gi is None:
                continue
            if row.cluster not in cluster_vocab:
                cluster_vocab[row.cluster] = cluster_next; cluster_next += 1
            mediastr, aer, cond1 = (row.cluster.split(" | ") + ["-", "-", "-"])[:3]
            P_gene.append(gi)
            P_feat.append(parse_condition(mediastr, aer, cond1))
            P_fit.append(float(row.fit_z))
            P_ess.append(1 if row.is_ess >= 0.5 else 0)
            P_org.append(our); P_clade.append(str(clade[gi]))
            P_clu.append(cluster_vocab[row.cluster]); n_pairs_org += 1
        print(f"  {fb:<12}->{our:<24} match {match_pct:>5.1f}%  noise rho {noise[our]:.3f}  "
              f"pairs {n_pairs_org:,}")

    if not P_gene:
        sys.exit("no pairs produced -- check gene-id join")
    cond_feat = np.array(P_feat, np.float32)
    np.savez_compressed(out_dir / "cond_pairs.npz",
                        gene_idx=np.array(P_gene, np.int32),
                        cond_feat=cond_feat,
                        y_fit=np.array(P_fit, np.float32),
                        y_ess=np.array(P_ess, np.int8),
                        org=np.array(P_org, dtype=object),
                        clade=np.array(P_clade, dtype=object),
                        cluster_id=np.array(P_clu, np.int32),
                        comp_names=np.array(COMP_NAMES, dtype=object))
    meta = dict(n_pairs=len(P_gene), n_genes=int(len(set(P_gene))),
                n_clusters=len(cluster_vocab), n_orgs=int(len(set(P_org))),
                ess_rate=round(float(np.mean(P_ess)), 4),
                components=COMP_NAMES,
                noise_floor=noise,
                noise_floor_median=round(float(np.nanmedian(list(noise.values()))), 4),
                join_report=join_report)
    json.dump(meta, open(out_dir / "cond_meta.json", "w"), indent=2)
    print(f"\n=== {len(P_gene):,} (gene,condition) pairs | {len(set(P_org))} orgs | "
          f"{len(cluster_vocab)} clusters | ess {np.mean(P_ess)*100:.1f}% ===")
    print(f"median per-org noise floor (ceiling): rho {meta['noise_floor_median']}")
    print(f"condition feature dims: {cond_feat.shape[1]} (mean active/cond: {cond_feat.sum(1).mean():.1f})")
    print(f"wrote {out_dir/'cond_pairs.npz'} + cond_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feba_db", default="/tmp/feba.db")
    ap.add_argument("--cache", default="outputs/orphan/af_msa_cache.npz")
    ap.add_argument("--labels_dir", default="data/drive_import/labels_aug")
    ap.add_argument("--out_dir", default="data/drive_import/cond")
    ap.add_argument("--min_conditions", type=int, default=50)
    a = ap.parse_args()
    main(a.feba_db, a.cache, a.labels_dir, a.out_dir, a.min_conditions)
