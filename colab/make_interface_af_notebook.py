"""make_interface_af_notebook — emit colab/interface_af_multimer.ipynb: a careful full-MSA AlphaFold-Multimer run
(via ColabFold) over the honest discovery pair set, reading ipTM + pDockQ. This is the ONE variant the fetched
Predictomes screen could not rule out (it used high-throughput AF + raw contact counts, which gave chance on genuine
discovery). Templates are DISABLED so AF cannot retrieve a solved co-structure — this tests prediction, not recall.

Run order in Colab (GPU runtime): install -> load pairs -> fetch sequences -> fold -> score.
Honest expectation: likely near chance. AF-Multimer's PPI signal comes from paired-MSA co-evolution + PDB precedent,
both thin for novel human pairs; if even careful ipTM stays ~0.5-0.6 on these in_pdb=0 hard pairs, structure does
NOT beat domains (0.61) for discovery and the door is closed honestly.
"""
import json, os

OUT = os.path.join(os.path.dirname(__file__), "interface_af_multimer.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


CELLS = [
    md("""# AlphaFold-Multimer interface vs the discovery ceiling

**Question:** does a *careful* full-MSA AF-Multimer run (ipTM / pDockQ) beat domain compatibility (**0.61**) at
predicting **novel** human PPIs — pairs with no solved co-structure and no network shortcut?

**Why this notebook exists.** We already fetched the Predictomes AF-Multimer screen (1.6M pairs, no GPU). Its raw
`num_unique_contacts` scored **0.82 on pairs whose complex is in the PDB (memorised) but 0.50 on genuine discovery
(in_pdb=0, 0 shared neighbours)** — chance. The only untested variant is a careful run at full MSA depth reading the
proper interface confidences. That needs a GPU, so: run it here.

**Honesty controls baked in:**
- Pairs = `interface_pairs.json`: 80 real PPIs with **in_pdb=0** (0-shared) + 80 **degree-matched zero-evidence**
  non-edges. Balanced, leak-clean.
- **Templates OFF** — AF cannot pull a co-structure from the PDB; this measures prediction, not recall.
- Report **ipTM** and **pDockQ** AUC vs the labels, next to the domain (0.61) and topology (0.50) baselines.

**Expectation:** modest. If ipTM ≈ 0.5–0.6, structure does not beat domains for discovery, and we close this branch
honestly instead of shipping a me-too."""),

    md("""## 1 · Install ColabFold (GPU runtime required: Runtime → Change runtime type → GPU)
JAX on Colab is version-fragile: installing ColabFold then letting pip upgrade JAX leaves `jaxlib` mismatched with
the CUDA plugin (`register_custom_type_id_handler` error). Fix = install ONE consistent CUDA-matched set,
`jax[cuda12]==0.5.3` (localcolabfold's tested pin). On a fresh runtime this needs no restart. If you already ran a
broken install this session, run this cell, then **Runtime → Restart session**, then continue at cell 2."""),
    code("""import os
if not os.path.isfile("COLABFOLD_READY"):
    print("Installing ColabFold — a few minutes…")
    os.system("pip -q install --no-warn-conflicts "
              "'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
    # one consistent CUDA-matched JAX (avoids the jaxlib/plugin version collision)
    os.system("pip -q uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt")
    os.system("pip -q install --no-warn-conflicts 'jax[cuda12]==0.5.3'")
    open("COLABFOLD_READY", "w").close()
    print("done.")
import jax
gpu = any(d.platform == "gpu" for d in jax.devices())
print("JAX", jax.__version__, "| devices:", jax.devices(), "| GPU OK" if gpu else "| NO GPU: set Runtime -> GPU")"""),

    md("""## 2 · Load the honest discovery pair set
Upload `interface_pairs.json` (from `outputs/orphan/`), or mount Drive and point `PAIRS` at it."""),
    code("""import json
try:
    pairs_data = json.load(open("interface_pairs.json"))
except FileNotFoundError:
    from google.colab import files
    up = files.upload()
    pairs_data = json.load(open(list(up)[0]))
PAIRS = pairs_data["pairs"]
print(len(PAIRS), "pairs |", sum(p["label"] for p in PAIRS), "positives")
print("baselines -> domain", pairs_data["baseline_domain_auc"], "| topology", pairs_data["baseline_topology_auc"])"""),

    md("## 3 · Fetch UniProt sequences by accession"),
    code("""import urllib.request, time
def uniprot_seq(acc):
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30) as r:
                lines = r.read().decode().splitlines()
            return "".join(l.strip() for l in lines if not l.startswith(">"))
        except Exception:
            time.sleep(2)
    return None

seqs = {}
for p in PAIRS:
    for acc in (p["acc_a"], p["acc_b"]):
        if acc not in seqs:
            seqs[acc] = uniprot_seq(acc)
missing = [a for a, s in seqs.items() if not s]
print("fetched", sum(1 for s in seqs.values() if s), "sequences |", len(missing), "missing")"""),

    md("""## 4 · Build ColabFold input & fold (AF-Multimer, templates OFF, full MSA)
Each complex is one FASTA with the two chains joined by `:`. Long pairs are capped to keep folds ~1–2 min. This is
the slow cell — ~160 folds; use an A100/L4 runtime. Lower `MAX_LEN` or the pair count if you hit the session limit.

The fold runs in the background and this cell **polls for per-fold progress**: `[i/N]`, time for that fold, total
elapsed, and ETA. The ETA uses the steady-state rate (fold #1 carries a one-time model-compile + MSA warmup, so it
is excluded from the estimate). The batch log is captured to `af_run.log` — `!tail af_run.log` if something stalls."""),
    code("""import os, glob, time, subprocess
os.makedirs("af_in", exist_ok=True); os.makedirs("af_out", exist_ok=True)
MAX_LEN = 1200  # cap total complex length (residues) so a fold stays ~1-2 min
folded = []
for k, p in enumerate(PAIRS):
    sa, sb = seqs.get(p["acc_a"]), seqs.get(p["acc_b"])
    if not sa or not sb or len(sa) + len(sb) > MAX_LEN:
        continue
    name = f"{k:03d}_{p['gene_a']}_{p['gene_b']}_L{p['label']}"
    open(f"af_in/{name}.fasta", "w").write(f">{name}\\n{sa}:{sb}\\n")
    folded.append((name, p["label"]))
N = len(folded)
print(f"folding {N} pairs (templates OFF, 3 recycles) — first fold is slow (model compile + MSA)…\\n")

def fmt(s):
    s = int(max(s, 0)); return f"{s//3600}h{(s%3600)//60:02d}m{s%60:02d}s"

# --templates is NOT passed -> no template retrieval (genuine prediction, not PDB recall)
log = open("af_run.log", "w")
proc = subprocess.Popen(["colabfold_batch", "--num-recycle", "3", "--num-models", "1", "af_in", "af_out"],
                        stdout=log, stderr=subprocess.STDOUT)
done_pat = "af_out/*_scores_rank_001_*.json"
t0 = time.time(); last = t0; t1 = None; done = set()
while proc.poll() is None or len(done) < N:
    cur = {os.path.basename(f).split("_scores_rank")[0] for f in glob.glob(done_pat)}
    for nm in sorted(cur - done):
        now = time.time(); dt = now - last; last = now
        done.add(nm); i = len(done)
        if i == 1:
            t1 = now; eta = "—  (warmup fold)"
        else:
            rate = (now - t1) / (i - 1); eta = fmt(rate * (N - i))
        print(f"[{i:3d}/{N}]  {nm[:44]:44s}  fold {fmt(dt):>9s}  |  elapsed {fmt(now-t0):>9s}  |  ETA {eta}")
    if proc.poll() is not None and (cur - done) == set():
        break                                      # batch ended; stragglers (too-long/errored) won't appear
    time.sleep(5)
n_out = len(glob.glob(done_pat))
print(f"\\ndone. {n_out}/{N} folded in {fmt(time.time()-t0)}"
      + ("" if n_out else "  — 0 outputs: check  !tail -40 af_run.log"))"""),

    md("## 5 · Score: ipTM (and pDockQ) AUC vs labels, against the baselines"),
    code("""import glob, json, numpy as np
from sklearn.metrics import roc_auc_score

def iptm_for(name):
    fs = glob.glob(f"af_out/{name}*_scores_rank_001_*.json")
    if not fs:
        return None
    d = json.load(open(fs[0]))
    return float(d.get("iptm", d.get("ptm", 0)))

def pdockq_for(name):
    # pDockQ = 0.724 / (1+exp(-0.052*(x-152.611))) + 0.018 ; x = n_interface_contacts * mean interface pLDDT
    import numpy as np
    pdb = sorted(glob.glob(f"af_out/{name}*_unrelaxed_rank_001_*.pdb"))
    if not pdb:
        return None
    ca = {}
    for line in open(pdb[0]):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ch = line[21]; xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            b = float(line[60:66]); ca.setdefault(ch, []).append((xyz, b))
    chs = list(ca)
    if len(chs) < 2:
        return 0.0
    A, B = np.array([x for x, _ in ca[chs[0]]]), np.array([x for x, _ in ca[chs[1]]])
    bA = np.array([b for _, b in ca[chs[0]]]), ; bB = np.array([b for _, b in ca[chs[1]]])
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    ia, ib = np.where(D < 8.0)
    if len(ia) == 0:
        return 0.0
    n = len(ia)
    mean_plddt = (np.concatenate([bA[0][np.unique(ia)], bB[0][np.unique(ib)]])).mean()
    x = n * mean_plddt
    return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

names = [(n, l) for n, l in folded if iptm_for(n) is not None]
y = np.array([l for _, l in names])
iptm = np.array([iptm_for(n) for n, _ in names])
print(f"scored {len(names)} folds ({y.sum()} pos)")
print(f"ipTM   discovery AUC = {roc_auc_score(y, iptm):.3f}")
try:
    pdq = np.array([pdockq_for(n) or 0 for n, _ in names])
    print(f"pDockQ discovery AUC = {roc_auc_score(y, pdq):.3f}")
except Exception as e:
    print("pDockQ skipped:", str(e)[:80])
print("baselines -> domain 0.61 | topology 0.50 | Predictomes throughput contacts 0.50")
print("\\nVERDICT: ipTM > ~0.66 => careful AF beats domains for discovery (worth pursuing).")
print("         ipTM ~0.5-0.6 => structure does NOT beat domains on novel pairs; close the branch honestly.")"""),

    md("""## 6 · Confound audit — only believe a high ipTM if it SURVIVES controls
A >0.66 raw ipTM AUC can be an artifact. Two confounds fake a win: **MSA depth** (AF's signal is co-evolution, so
if positives just have deeper paired MSAs than the zero-evidence negatives, ipTM separates them by data, not by
predicting a novel interface) and **length** (the MAX_LEN filter can skew class lengths). This cell checks whether
the ipTM separation holds when those are controlled. Trust it only if ipTM stays > domains (0.61) within length
strata AND the ipTM logistic coefficient dominates length/MSA-depth."""),
    code("""import glob, json, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def iptm_for(name):
    fs = glob.glob(f"af_out/{name}*_scores_rank_001_*.json")
    return float(json.load(open(fs[0])).get("iptm", 0)) if fs else None
def msa_depth(name):
    a = glob.glob(f"af_out/{name}.a3m") + glob.glob(f"af_out/{name}/*.a3m") + glob.glob(f"af_out/{name}_*.a3m")
    return sum(1 for l in open(a[0]) if l.startswith(">")) if a else 0

rows = []
for nm, lab in folded:
    it = iptm_for(nm)
    if it is None:
        continue
    seq = "".join(x.strip() for x in open(f"af_in/{nm}.fasta") if not x.startswith(">"))
    rows.append((lab, it, len(seq.replace(":", "")), msa_depth(nm)))
y = np.array([r[0] for r in rows]); iptm = np.array([r[1] for r in rows])
L = np.array([r[2] for r in rows]); md = np.array([r[3] for r in rows])
print(f"n={len(rows)}  pos={int(y.sum())}  neg={int((1-y).sum())}")
print(f"raw ipTM AUC = {roc_auc_score(y, iptm):.3f}   (domain baseline 0.61)")
print(f"  is LENGTH a confound?   pos-vs-neg length AUC   = {roc_auc_score(y, L):.3f}   (~0.5 = no)")
if md.any():
    print(f"  is MSA-DEPTH a confound? pos-vs-neg depth AUC    = {roc_auc_score(y, md):.3f}   (~0.5 = no)")
feats = [iptm, L] + ([md] if md.any() else [])
names = ["ipTM", "length"] + (["MSAdepth"] if md.any() else [])
lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(np.column_stack(feats)), y)
print("  standardized logistic coefs:", {n: round(float(c), 2) for n, c in zip(names, lr.coef_[0])},
      "(ipTM should dominate)")
q = np.quantile(L, [1/3, 2/3])
for lo, hi, tag in [(-1, q[0], "short"), (q[0], q[1], "mid"), (q[1], 1e12, "long")]:
    m = (L > lo) & (L <= hi)
    if y[m].sum() >= 5 and (1 - y[m]).sum() >= 5:
        print(f"  length {tag:5s} n={int(m.sum()):3d}  ipTM AUC = {roc_auc_score(y[m], iptm[m]):.3f}")
print("\\nTRUST only if ipTM stays > 0.61 within length strata AND its coef dominates length/MSAdepth;")
print("if it collapses when controlled, the headline AUC was a confound, not interface discovery.")"""),
]


def build():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "accelerator": "GPU", "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(OUT, "w"), indent=1)
    print("wrote", OUT, "(", len(CELLS), "cells )")


if __name__ == "__main__":
    build()
