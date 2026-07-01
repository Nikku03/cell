# ======================================================================================
# COLAB NOTEBOOK  —  cell-model ensemble, phase 1 (gene-level essentiality)
# Question: do FOUNDATION-MODEL gene embeddings (Geneformer [+scGPT]) add anything OVER our
# integrated biological features? Bar to beat = our features-only MLP (AUC 0.973).
# Strategy: frozen foundation embeddings + our integrated priors -> small combiner (MLP first,
# GNN as a tested variant). Honest: adopt the ensemble ONLY if it beats the baseline.
#
# Runtime: GPU (for loading Geneformer to extract embeddings). Combiner trains on CPU-seconds.
# Paste cells (separated by "# %%") into Google Colab.  Set Runtime -> GPU.
# ======================================================================================

# %% [1] setup
!pip -q install torch scikit-learn pandas numpy transformers huggingface_hub
# Geneformer (optional, for its learned gene embeddings). scGPT/scFoundation similar.
!pip -q install git+https://huggingface.co/ctheodoris/Geneformer 2>/dev/null || echo "geneformer optional"
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")

# %% [2] OUR priors + labels  (push repo first, then wget the raw file, OR upload it)
# Our integrated per-gene table (features) + Hart CEG/NEG (essentiality labels).
INTEG_URL="https://raw.githubusercontent.com/nikku03/cell/claude/vectorize-gex-propensity-NRqBW/outputs/orphan/integrated_cell_human.csv"
df=pd.read_csv(INTEG_URL)
FEATS=["loeuf","is_tf","regulon_out","regulators_in","ppi_degree","n_pathways","cpg_promoter","enhancers","n_diseases"]
for c in ["regulon_out","regulators_in","ppi_degree","n_pathways","enhancers","n_diseases"]:
    df[c]=np.log1p(df[c].fillna(0))
df["loeuf"]=df["loeuf"].fillna(df["loeuf"].median())
df[FEATS]=df[FEATS].fillna(0)
y=df["essential"].map({1:1,0:0}); mask=y.notna()
print("labeled genes:", int(mask.sum()), "| essential:", int((y==1).sum()))

# %% [3] BETTER labels (optional upgrade): DepMap common-essential  ->  measured, per-cell-line
# CRISPRGeneEffect.csv (rows=cell lines, cols=genes). Common-essential = median gene-effect < -0.5
# import gdown; DepMap release file id changes -> get current from https://depmap.org/portal/download
# de=pd.read_csv("CRISPRGeneEffect.csv",index_col=0); de.columns=[c.split(" ")[0] for c in de.columns]
# ce=(de.median(0)< -0.5).astype(int)   # common-essential label per gene
# df["depmap_ess"]=df["gene"].map(ce)   # use as the target instead of Hart CEG/NEG

# %% [4] FOUNDATION-MODEL gene embeddings (frozen) — Geneformer learned gene token embeddings
def geneformer_gene_emb(symbols):
    """Extract Geneformer's learned per-gene embeddings (static). Returns dict symbol->vector."""
    from huggingface_hub import hf_hub_download
    import pickle, json
    tok=hf_hub_download("ctheodoris/Geneformer","geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl")
    ens=hf_hub_download("ctheodoris/Geneformer","geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
    name2id=pickle.load(open(tok,"rb")); id2tok=pickle.load(open(ens,"rb"))
    from transformers import AutoModel
    m=AutoModel.from_pretrained("ctheodoris/Geneformer").eval()
    W=m.get_input_embeddings().weight.detach().cpu().numpy()   # (vocab, dim) learned gene embeddings
    out={}
    for s in symbols:
        eid=name2id.get(s); t=id2tok.get(eid) if eid else None
        if t is not None and t<len(W): out[s]=W[t]
    return out
# emb = geneformer_gene_emb(df["gene"].tolist())      # <- run on GPU
# (repeat for scGPT / scFoundation gene embeddings and concatenate for the 3-way ensemble)

# %% [5] combiner + HONEST benchmark (features only  vs  features+embeddings)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
def bench(Xcols_extra=None, name="features-only"):
    idx=df.index[mask]
    X=df.loc[idx,FEATS].values.astype(float)
    if Xcols_extra is not None: X=np.hstack([X, Xcols_extra[mask.values]])
    X=StandardScaler().fit_transform(X); yy=y[mask].astype(int).values
    clf=MLPClassifier(hidden_layer_sizes=(64,),max_iter=300,alpha=1e-3)
    p=cross_val_predict(clf,X,yy,cv=5,method="predict_proba")[:,1]
    a=roc_auc_score(yy,p); print(f"  {name:28s}: 5-fold AUC {a:.3f}"); return a
print("=== does the foundation embedding ADD over our features? ===")
base=bench(None,"our features only (baseline)")
# emb_matrix = np.array([emb.get(g, np.zeros(256)) for g in df["gene"]])   # align to df order
# combo=bench(emb_matrix,"our features + Geneformer emb")
# print("VERDICT: adopt ensemble only if", f"{'combo>base'}")

# %% [6] PHASE 2 (later): cell-type state
# import cellxgene_census
# with cellxgene_census.open_soma() as census:
#   adata = cellxgene_census.get_anndata(census,"Homo sapiens",
#            obs_value_filter="dataset_id=='<Tabula Sapiens id>'")   # ~500k cells
# -> tokenize for Geneformer -> per-cell embeddings -> cell-type head + per-cell-type active network
# This is the keystone (cell-type activation state). Needs the atlas; GPU-friendly on Colab.
