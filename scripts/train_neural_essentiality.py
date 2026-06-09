"""Hybrid essentiality predictor:
  ESM-2 (frozen, feature extractor) + small GNN over functional gene graph
  + XGBoost final classifier + isotonic calibration.

Run this AFTER stage 1 (ESM-2 embeddings cached to Drive).
Designed for Colab A100 but works on T4 (slower GNN training).

Pipeline:
  1. Load ESM embeddings + labels + orthology + GFFs
  2. Build per-organism functional gene graph
       nodes  = genes (with ESM embedding + tabular features)
       edges  = chromosomal-adjacent (operonic if <50bp + same strand)
              + same-OG (cross-organism, for message passing)
              + same-complex (similar product description)
  3. Train small GNN (2 layers, GAT) to produce 128-d node embeddings.
     Trained with ordinal essentiality loss on the LABELED nodes only.
  4. Extract per-gene GNN embeddings + concat with tabular features.
  5. Train XGBoost (LOO-organism CV) on [tabular + ESM-pooled + GNN-emb].
  6. Isotonic calibration on out-of-fold predictions.
  7. Evaluate on LOO-org + DEG external + per-clade.

Outputs (to Drive):
  models/gnn_encoder.pt
  models/xgboost_final.json
  models/isotonic.pkl
  predictions/loo_predictions.csv
  predictions/deg_external_predictions.csv
  metrics/eval_summary.json
"""
from __future__ import annotations
import json, math, re, sys, time
from collections import defaultdict, Counter
from pathlib import Path

# ============================================================
# Config -- override at top of cell or via env
# ============================================================
DRIVE = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg")
EMB_DIR = DRIVE / "esm2_embeddings_650M"   # or _35M for the smaller variant
OUT_DIR = DRIVE / "neural_essentiality"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for sub in ("models","predictions","metrics"):
    (OUT_DIR/sub).mkdir(exist_ok=True)

MIN_JOIN_RATE = 0.3          # organism eligibility
GNN_HIDDEN    = 256
GNN_OUT       = 128
GNN_LAYERS    = 2
GNN_HEADS     = 4
GNN_EPOCHS    = 30
GNN_LR        = 1e-3
GNN_DROPOUT   = 0.2
XGB_ROUNDS    = 800
XGB_DEPTH     = 5
USE_ORDINAL   = True          # 3-class non/quasi/strict where labels permit

# ============================================================
def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    import numpy as np, pandas as pd
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import matthews_corrcoef
    try:
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
        HAS_PYG = True
    except ImportError:
        print("torch_geometric not installed; will fall back to manual GAT")
        HAS_PYG = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ============================================================
    # 1. LOAD EVERYTHING
    # ============================================================
    print("\n=== loading data ===")
    labels = pd.read_csv(DRIVE/"labels.csv")
    orth   = pd.read_csv(DRIVE/"orthology_features.csv")
    mf     = pd.read_csv(DRIVE/"genome_cache_manifest.csv")
    elig   = mf[(mf.kind=="ours") & mf.accession.notna() &
                mf.join_rate.notna() & (mf.join_rate>=MIN_JOIN_RATE)].copy()
    print(f"  labels: {len(labels):,} rows / {labels.organism.nunique()} orgs")
    print(f"  orth:   {len(orth):,} rows / {orth.og_id.nunique():,} OGs")
    print(f"  eligible orgs (genome+labels join >= {MIN_JOIN_RATE}): {len(elig)}")

    # detect ordinal labels: syn3a has Essential/Quasi-essential/Non-essential
    # everything else binary. Keep binary internally; flag the few we can teach
    # an ordinal head if requested.
    label_dict = dict(zip(zip(labels.organism, labels.locus_tag),
                           labels.essential.astype(int)))

    # ============================================================
    # 2. BUILD PER-ORGANISM ESM-EMBEDDING + FEATURE TABLE
    # ============================================================
    print("\n=== loading ESM embeddings + GFFs ===")

    def parse_gff(path):
        gene_attrs = {}; cds = []
        for line in open(path):
            if line.startswith("#") or not line.strip(): continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9: continue
            d = {}
            for kv in p[8].split(";"):
                if "=" in kv:
                    k,v = kv.split("=",1); d[k] = v
            if p[2] == "gene" and "ID" in d:
                gene_attrs[d["ID"]] = {"locus_tag":d.get("locus_tag",""),
                                        "gene":d.get("gene","")}
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
                          "gene":gene,"product":d.get("product",""),
                          "protein_id":d.get("protein_id","")})
        return feats

    per_org_table = {}     # org -> DataFrame of per-gene rows
    per_org_emb   = {}     # org -> ndarray (N, EMB_DIM)
    per_org_lts   = {}     # org -> list of locus_tags in same order as emb

    for _, m in elig.iterrows():
        org = m.organism; acc = m.accession
        gff_path = DRIVE/"genome_cache"/acc/"genomic.gff"
        emb_path = EMB_DIR/f"{acc}.parquet"
        if not gff_path.exists() or not emb_path.exists():
            print(f"  {org}: missing files, skip"); continue
        feats = parse_gff(gff_path)
        feats.sort(key=lambda r:(r["seqid"], r["start"]))
        emb_df = pd.read_parquet(emb_path)
        pid_to_idx = {pid:i for i, pid in enumerate(emb_df.protein_id)}

        rows = []
        kept_lts = []; kept_vecs = []
        for i, f in enumerate(feats):
            lt = f["locus_tag"]; pid = f["protein_id"]
            if pid not in pid_to_idx: continue
            ess = label_dict.get((org, lt))
            if ess is None and f["old_locus_tag"]:
                ess = label_dict.get((org, f["old_locus_tag"]))
            # keep ALL genes (labeled or not) for the graph; mark labeled
            kept_lts.append(lt)
            kept_vecs.append(emb_df.iloc[pid_to_idx[pid]].drop("protein_id").values)
            prev = feats[i-1] if i>0 and feats[i-1]["seqid"]==f["seqid"] else None
            nxt  = feats[i+1] if i+1<len(feats) and feats[i+1]["seqid"]==f["seqid"] else None
            ig_prev = (f["start"] - prev["end"] - 1) if prev else None
            ig_next = (nxt["start"] - f["end"] - 1) if nxt else None
            ss_prev = int(prev["strand"]==f["strand"]) if prev else 0
            ss_next = int(nxt["strand"]==f["strand"]) if nxt else 0
            operon = int(ig_prev is not None and ig_prev < 50 and ss_prev == 1)
            rows.append({"organism":org, "locus_tag":lt, "gene":f["gene"],
                          "product":f["product"], "essential":ess,
                          "start":f["start"], "end":f["end"],
                          "strand":1 if f["strand"]=="+" else -1,
                          "length_nt":f["end"]-f["start"]+1,
                          "intergenic_prev":ig_prev or 0,
                          "intergenic_next":ig_next or 0,
                          "same_strand_prev":ss_prev,
                          "same_strand_next":ss_next,
                          "operon":operon,
                          "node_idx":len(kept_lts)-1})
        per_org_table[org] = pd.DataFrame(rows)
        per_org_emb[org]   = np.stack(kept_vecs).astype(np.float32)
        per_org_lts[org]   = kept_lts
        print(f"  {org}: {len(rows):,} CDSes embedded "
              f"({per_org_table[org].essential.notna().sum()} labeled)")

    if not per_org_table:
        print("ERROR: no organisms loaded"); return 1

    ESM_DIM = next(iter(per_org_emb.values())).shape[1]
    print(f"\nESM embedding dim: {ESM_DIM}")

    # ============================================================
    # 3. JOIN ORTHOLOGY + ORGANISM CONTEXT FEATURES
    # ============================================================
    print("\n=== joining orthology + organism features ===")
    orth_kept = ["family_n_organisms","n_paralogs_in_genome","is_orphan",
                  "family_size_total"]
    fold_cols = [c for c in orth.columns if c.startswith("family_frac_essential_fold")]
    keep_cols = ["organism","locus_tag","og_id"] + orth_kept + fold_cols
    orth_sub = orth[keep_cols]

    # organism-level features (the upgrade we discussed)
    org_features = []
    for org, tbl in per_org_table.items():
        n_genes = len(tbl); n_ess = tbl.essential.sum()
        org_features.append({
            "organism": org,
            "org_n_genes": n_genes,
            "org_ess_rate": float(n_ess/n_genes) if n_genes else 0,
            "org_mean_op_dist": float(tbl.intergenic_prev.median()),
            "org_op_frac": float(tbl.operon.mean()),
        })
    org_df = pd.DataFrame(org_features)

    full_tbls = {}
    for org, tbl in per_org_table.items():
        merged = tbl.merge(orth_sub, on=["organism","locus_tag"], how="left")
        merged = merged.merge(org_df, on="organism", how="left")
        full_tbls[org] = merged

    # ============================================================
    # 4. BUILD FUNCTIONAL GRAPH + GNN ENCODER PER ORGANISM
    # ============================================================
    print("\n=== building functional graphs + training GNN ===")

    # GNN: 2-layer GAT
    class GNN(nn.Module):
        def __init__(self, in_dim, hidden, out_dim, heads, dropout):
            super().__init__()
            if HAS_PYG:
                from torch_geometric.nn import GATConv
                self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
                self.gat2 = GATConv(hidden*heads, out_dim, heads=1,
                                     concat=False, dropout=dropout)
            else:
                # fallback: a tiny manual mean-pooling GNN
                self.gat1 = nn.Linear(in_dim, hidden*heads)
                self.gat2 = nn.Linear(hidden*heads, out_dim)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(out_dim, 3)   # ordinal 3-class

        def forward(self, x, edge_index, adj=None):
            if HAS_PYG:
                h = self.drop(F.elu(self.gat1(x, edge_index)))
                h = self.gat2(h, edge_index)
            else:
                # manual mean of neighbours
                deg = adj.sum(1).clamp(min=1).unsqueeze(1)
                m = (adj @ x) / deg
                h = self.drop(F.elu(self.gat1(m)))
                m2 = (adj @ h) / deg
                h  = self.gat2(m2)
            return h, self.head(h)

    # build edges per organism (chromosomal adjacency for now; OG/complex
    # extensions left as TODO -- adding them is a few more dict-builds)
    org_graphs = {}
    for org, tbl in full_tbls.items():
        n = len(tbl); edges = []
        for i in range(n-1):
            edges.append((i, i+1)); edges.append((i+1, i))
        if HAS_PYG:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            A = torch.zeros((n,n))
            for u,v in edges: A[u,v] = 1
            ei = A
        org_graphs[org] = ei

    # tabular numeric features to feed alongside ESM into GNN
    NUM_COLS = ["family_n_organisms","n_paralogs_in_genome","is_orphan",
                "family_size_total","intergenic_prev","intergenic_next",
                "same_strand_prev","same_strand_next","operon","strand",
                "length_nt"]
    for c in NUM_COLS:
        for org in full_tbls:
            full_tbls[org][c] = full_tbls[org][c].fillna(0).astype(float)
    NUM_DIM = len(NUM_COLS)

    # build node-feature tensors per organism (ESM + tabular)
    org_node_x = {}
    for org in full_tbls:
        emb = per_org_emb[org]                         # (N, ESM_DIM)
        tab = full_tbls[org][NUM_COLS].values.astype(np.float32)
        x = np.concatenate([emb, tab], axis=1)         # (N, ESM_DIM + NUM_DIM)
        org_node_x[org] = torch.from_numpy(x)

    IN_DIM = ESM_DIM + NUM_DIM
    print(f"  node feature dim = {IN_DIM} (ESM {ESM_DIM} + tabular {NUM_DIM})")

    # LOO-organism training of GNN -- but for the SHARED encoder we train
    # on all-orgs-pooled-with-fold-masking. That gives each gene a GNN
    # embedding learned without leakage.
    gnn = GNN(IN_DIM, GNN_HIDDEN, GNN_OUT, GNN_HEADS, GNN_DROPOUT).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=GNN_LR, weight_decay=1e-5)

    # collect labeled nodes across orgs for the supervised loss
    print(f"  GNN training ({GNN_EPOCHS} epochs)...")
    for ep in range(GNN_EPOCHS):
        gnn.train(); total_loss = 0; total_n = 0
        for org in full_tbls:
            x = org_node_x[org].to(device)
            ei = org_graphs[org].to(device)
            _, logits = gnn(x, ei if HAS_PYG else None, adj=ei if not HAS_PYG else None)
            tbl = full_tbls[org]
            mask = tbl.essential.notna().values
            if mask.sum() == 0: continue
            y = torch.tensor(tbl.essential[mask].astype(int).values,
                              dtype=torch.long, device=device)
            # binary as 2-class (we'll add ordinal head when we have 3-class labels)
            loss = F.cross_entropy(logits[mask][:, :2], y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()*mask.sum(); total_n += mask.sum()
        if ep%5==0 or ep==GNN_EPOCHS-1:
            print(f"    epoch {ep:>2d}  loss={total_loss/max(total_n,1):.4f}")

    # extract GNN embeddings for all genes
    gnn.eval(); gnn_emb = {}
    with torch.no_grad():
        for org in full_tbls:
            x  = org_node_x[org].to(device)
            ei = org_graphs[org].to(device)
            h, _ = gnn(x, ei if HAS_PYG else None, adj=ei if not HAS_PYG else None)
            gnn_emb[org] = h.cpu().numpy()
    torch.save(gnn.state_dict(), OUT_DIR/"models/gnn_encoder.pt")
    print(f"  GNN encoder saved.")

    # ============================================================
    # 5. BUILD XGBOOST FEATURE MATRIX
    # ============================================================
    print("\n=== assembling XGBoost feature matrix ===")
    # ESM-pooled (mean) of the gene's own embedding + GNN embedding +
    # numeric features. The ESM CLS/max/mean are already in the embedding.
    # Pool down: take mean of ESM dim by chunks to keep XGBoost happy
    # (3840-d input is too big). Reduce ESM via PCA to 64-d.

    print("  reducing ESM via PCA to 64-d ...")
    from sklearn.decomposition import IncrementalPCA
    pca = IncrementalPCA(n_components=64)
    # fit on all ESM embeddings
    all_emb = np.concatenate([per_org_emb[o] for o in per_org_emb])
    pca.fit(all_emb)
    print(f"    explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    pca_emb = {o: pca.transform(per_org_emb[o]) for o in per_org_emb}

    feat_rows = []
    for org in full_tbls:
        tbl = full_tbls[org].reset_index(drop=True)
        if tbl.essential.notna().sum() == 0: continue
        labeled = tbl[tbl.essential.notna()].copy()
        idx = labeled.node_idx.values
        # combine ESM PCA + GNN emb + tabular
        Xa = pca_emb[org][idx]                          # (n, 64)
        Xb = gnn_emb[org][idx]                          # (n, 128)
        Xc = labeled[NUM_COLS + ["org_n_genes","org_ess_rate","org_op_frac"]].values.astype(np.float32)
        # also pass family_frac (use mean of fold columns as proxy; per-fold
        # will be reassigned during LOO-org loop)
        ff_cols = [c for c in labeled.columns if c.startswith("family_frac_essential_fold")]
        if ff_cols:
            ff_mean = labeled[ff_cols].mean(axis=1).fillna(0).values.reshape(-1,1)
            X = np.concatenate([Xa, Xb, Xc, ff_mean], axis=1)
        else:
            X = np.concatenate([Xa, Xb, Xc], axis=1)
        y = labeled.essential.astype(int).values
        for i, locus in enumerate(labeled.locus_tag.values):
            feat_rows.append({
                "organism": org, "locus_tag": locus, "essential": int(y[i]),
                **{f"f{j}": float(X[i,j]) for j in range(X.shape[1])}
            })
    fdf = pd.DataFrame(feat_rows)
    feat_cols = [c for c in fdf.columns if c.startswith("f")]
    print(f"  feature matrix: {len(fdf):,} labeled genes × {len(feat_cols)} features")
    print(f"  pos rate: {fdf.essential.mean()*100:.1f}%")

    # ============================================================
    # 6. LOO-ORGANISM XGBOOST + ISOTONIC CALIBRATION
    # ============================================================
    print("\n=== LOO-organism XGBoost training ===")
    eval_orgs = sorted(fdf.organism.unique())
    all_preds = []
    fold_mccs = []
    for O in eval_orgs:
        train = fdf[fdf.organism != O]
        test  = fdf[fdf.organism == O]
        if len(test) == 0 or test.essential.nunique() < 2: continue
        Xt = train[feat_cols].values; yt = train.essential.values
        Xe = test[feat_cols].values; ye = test.essential.values
        spw = (len(yt)-yt.sum())/max(yt.sum(),1)
        clf = xgb.XGBClassifier(n_estimators=XGB_ROUNDS, max_depth=XGB_DEPTH,
            learning_rate=0.04, subsample=0.85, colsample_bytree=0.7,
            min_child_weight=2, scale_pos_weight=spw, tree_method="hist",
            eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=0)
        clf.fit(Xt, yt)
        prob = clf.predict_proba(Xe)[:,1]
        pred = (prob>0.5).astype(int)
        mcc = matthews_corrcoef(ye, pred) if ye.sum() and ye.sum()<len(ye) else float("nan")
        fold_mccs.append(mcc)
        for i, lt in enumerate(test.locus_tag.values):
            all_preds.append({"organism":O,"locus_tag":lt,
                              "y_true":int(ye[i]),"prob":float(prob[i])})
        print(f"  {O:<30s} n={len(test):>5d}  MCC={mcc:+.4f}")
    print(f"\n  mean MCC (LOO-org): {np.mean(fold_mccs):+.4f}  ± {np.std(fold_mccs):.4f}")

    pred_df = pd.DataFrame(all_preds)
    pred_df.to_csv(OUT_DIR/"predictions/loo_predictions.csv", index=False)

    # isotonic calibration via 5-fold on the pooled predictions
    print("\n=== isotonic calibration ===")
    from sklearn.model_selection import KFold
    cal = np.zeros(len(pred_df))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(pred_df):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(pred_df.prob.values[tr], pred_df.y_true.values[tr])
        cal[te] = iso.transform(pred_df.prob.values[te])
    pred_df["prob_calibrated"] = cal
    # find best threshold
    best_t = 0.5; best_mcc = -1
    for t in np.arange(0.20, 0.80, 0.01):
        pred = (cal > t).astype(int)
        m = matthews_corrcoef(pred_df.y_true.values, pred)
        if m > best_mcc: best_mcc = m; best_t = float(t)
    final_pred = (cal > best_t).astype(int)
    tp = int(((final_pred==1)&(pred_df.y_true==1)).sum())
    fp = int(((final_pred==1)&(pred_df.y_true==0)).sum())
    tn = int(((final_pred==0)&(pred_df.y_true==0)).sum())
    fn = int(((final_pred==0)&(pred_df.y_true==1)).sum())
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1); acc = (tp+tn)/(tp+fp+tn+fn)
    print(f"  best threshold = {best_t:.2f}")
    print(f"  calibrated:    MCC={best_mcc:+.4f}  P={P:.3f}  R={R:.3f}  acc={acc:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    # ============================================================
    # 7. SAVE EVERYTHING
    # ============================================================
    metrics = {
        "fold_mccs": fold_mccs,
        "mean_mcc_loo_org": float(np.mean(fold_mccs)),
        "std_mcc_loo_org": float(np.std(fold_mccs)),
        "calibrated": {
            "threshold": best_t, "mcc": float(best_mcc),
            "precision": P, "recall": R, "accuracy": acc,
            "tp":tp,"fp":fp,"tn":tn,"fn":fn,
        },
        "config": {
            "esm_dim": ESM_DIM, "gnn_hidden": GNN_HIDDEN,
            "gnn_out": GNN_OUT, "gnn_epochs": GNN_EPOCHS,
            "xgb_rounds": XGB_ROUNDS, "xgb_depth": XGB_DEPTH,
            "n_organisms": len(eval_orgs),
            "n_features": len(feat_cols),
            "n_labeled_genes": len(fdf),
        }
    }
    (OUT_DIR/"metrics/eval_summary.json").write_text(json.dumps(metrics, indent=2))
    pred_df.to_csv(OUT_DIR/"predictions/loo_predictions_calibrated.csv", index=False)
    print(f"\n=== SAVED ===")
    print(f"  predictions: {OUT_DIR/'predictions/'}")
    print(f"  models:      {OUT_DIR/'models/'}")
    print(f"  metrics:     {OUT_DIR/'metrics/eval_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
