"""Builds colab/cell_reasoning_llm_a100.ipynb and a flat .py copy for a CPU smoke test."""
import json, sys

CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))

md(r"""
# Cell reasoning engine + 30B LLM context, on an A100

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nikku03/cell/blob/claude/vectorize-gex-propensity-zp09w8/colab/cell_reasoning_llm_a100.ipynb)

This notebook asks one question: **can a reasoning engine predict what happens to a human cell when one gene is switched off, better than a plain learner, if a 30-billion-parameter language model gives it context about each gene?**

What it does, step by step:

1. **Data.** Downloads every input it needs:
   - the Human Protein Atlas (gene descriptions, protein classes, functions, subcellular locations, K-562 RNA levels, protein–protein interactions);
   - two Perturb-seq experiments in K562 cells (Replogle et al. 2022);
   - DepMap gene-dependency data and Reactome pathways.
2. **GPU data processing.** Loads the knockdown experiments onto the A100. It removes the response every knockdown shares (the "tide") and splits genes into TRAIN and TEST by a fixed hash.
3. **LLM context.** `Qwen3-30B-A3B-Instruct-2507` (30.5B parameters, Apache 2.0) reads a Protein Atlas profile of each knocked-down gene and is asked what knocking it down would do. Its internal representation of that answer becomes the gene's *context vector*. It also writes plain-text context for a few genes.
4. **Reasoning engine** (PyTorch, on the GPU).
   - Each "neuron" is a bump node, wired as a dendritic tree.
   - The tree runs for 3 steps.
   - At every step it **snaps** to the k most similar *real measured experiments* in its memory and reads their responses.
   - The memory holds the TRAIN knockdowns, never the gene being predicted.
5. **Fair test.**
   - **Baselines:** the engine, a plain ridge learner and a plain neural net all get the same inputs.
   - **Is it the LLM?** Versions without the LLM context, and with the LLM context shuffled across genes, show whether the LLM is what helps.
   - **Upper bound:** the same protein measured in the second, independent K562 experiment.
6. **LLM action.** For a few random TEST genes, the LLM reads the engine's predicted pathway changes and writes a mechanism and an experiment to test it. Its output is labelled as a hypothesis.
7. **Fine-tuning the LLM on documents → results (QLoRA).** The LLM is taught, on TRAIN genes only, to read a gene's Atlas document and predict what was actually measured when that gene was knocked down. It then predicts TEST genes it never saw. Only small adapter layers (LoRA) and a prediction head are trained.
9. **The aim (section 14).** The selfish-gene view, given to the models as an objective: a cell like K562 is organised around growing and dividing; losing a gene shifts it between GROWTH and PROTECTION. The engine and the LLM both learn to predict that aim first, and the LLM also reads it as a guiding principle.
10. **A bigger engine (section 15).** The same engine at four sizes (wider, deeper trees, more reasoning steps, more memories read per step), each beside a plain neural net with the same number of parameters.
8. **The nameless-documents test.** The LLM was pretrained on public text that may include these screens' papers. If it does well only because it recognises the gene's *name* and recalls published results, that is memory, not reasoning. Section 13 fine-tunes again, from scratch and with identical settings, on documents with the gene's name, aliases and full name removed. A second version also replaces the interaction partners' names with a count.

**Runtime.** About 45–90 min on an A100 for sections 1–11, plus about 30 min per fine-tuning run: one in section 12, two in section 13 (set `RUN_STRICT = False` to skip the second), and one in section 14. Most of that is downloading the model weights (≈61 GB) and the data, plus the LLM pass over ~5,000 genes. On a 40 GB A100 the LLM loads in 4-bit; on 80 GB, in bf16.

**Runtime → Change runtime type → A100 GPU** before running. Then *Runtime → Run all*.
""")

md(r"""
## Predeclared tests (fixed before any result exists)

Every prediction is scored per TEST knockdown as the Pearson r between the predicted and measured *specific* response, over ~8,200 genes (the knocked-down gene itself is left out). Comparisons are paired over TEST knockdowns with a two-sided sign test.

**BETTER** needs both of:
- p < 0.05;
- a median gain of at least **+0.01**.

**WORSE** is the mirror; anything else is **NO DIFFERENCE**.

| Test | Comparison | Question |
|---|---|---|
| **H0** | — | Leak check: ridge trained on shuffled answers must score median r < 0.02. |
| **E1** | ENGINE vs RIDGE | Does the reasoning engine beat the plain learner? |
| **E2** | ENGINE vs ENGINE-noLLM | Does the LLM context help the engine? |
| **E3** | RIDGE vs RIDGE-noLLM | Does the LLM context help the plain learner? |
| **E4** | ENGINE vs ENGINE-shuffledLLM | Does the LLM's *content* matter? (Must be BETTER for E2 to count.) |
| **E5** | ENGINE vs MLP | Do memory + snapped steps beat a plain network on the same inputs? |

**Fine-tuning tests (section 12), same rule.**

| Test | Comparison | Question |
|---|---|---|
| **F1** | LLM-FT vs RIDGE | Does the fine-tuned LLM beat the plain learner? |
| **F2** | LLM-FT vs RIDGE on frozen-LLM vectors | Does fine-tuning add anything over the same LLM left as it was? |
| **F3** | LLM-FT vs LLM-FT reading another gene's document | Does it really read the document? (Must be BETTER for F1/F2 to count.) |
| **F4** | ENSEMBLE (mean of LLM-FT and RIDGE, each put on the same scale first) vs RIDGE | Does it add something the plain learner lacks? |

Fine-tuning sees TRAIN genes only. 10% of TRAIN is held out to pick the best epoch. TEST genes are touched once, at the end.

**Nameless-documents tests (section 13), same rule.**

| Test | Comparison | Question |
|---|---|---|
| **N1** | LLM-FT-nameless vs LLM-FT (named) | Did names carry part of the gain? WORSE = yes. |
| **N2** | LLM-FT-nameless vs RIDGE | Same question as F1, without names. |
| **N3** | ENSEMBLE-nameless vs RIDGE | Same question as F4, without names. **Primary.** |
| **N4** | LLM-FT-nameless vs reading another gene's nameless document | Must be BETTER for N2/N3 to count. |
| **S1–S3** | As N1–N3, with partner names also removed | Is it still there with no gene names at all? |

**Verdicts, fixed now.**
- **GAIN SURVIVES WITHOUT NAMES** if N3 and N4 are both BETTER.
- **GAIN SURVIVES WITHOUT ANY GENE NAMES** if S3 is also BETTER.
- **GAIN NEEDS NAMES** if N3 is not BETTER and N1 is WORSE.

**The aim tests (section 14), same rule.** The aim is three numbers per gene, used as training answers for TRAIN genes only:
1. **Threat to growth.** K562's own DepMap gene effect.
2. **Grow vs protect.** The response's score on the top pattern of TRAIN responses, oriented so that ribosomal-protein genes going up counts as growth.
3. **How much the cell is shaken.** The size of the response.

| Test | Comparison | Question |
|---|---|---|
| **G1** | AIM-ENGINE vs ENGINE | Does giving the engine the aim help? |
| **G2** | AIM-ENGINE vs RIDGE | Does the aim engine beat the plain learner? |
| **G3** | RIDGE + predicted aim vs RIDGE | Does the aim help even a plain learner? Separates the principle from the extra K562 data. |
| **G4** | LLM-FT-AIM vs LLM-FT (section 12) | Does the aim help the LLM? It reads the principle and learns to predict the aim. |
| **G5** | ENSEMBLE-AIM (LLM-FT-AIM + RIDGE) vs RIDGE | The F4 question, with the aim. |

Reported, not tested: how well each model predicts the aim itself on TEST genes (threat-to-growth correlation and AUROC for essential genes, grow-vs-protect correlation).

**Verdicts, fixed now.**
- **THE AIM HELPS THE ENGINE** if G1 is BETTER.
- **THE AIM HELPS THE LLM** if G4 is BETTER.
- If G3 is also BETTER, the gain is from the extra K562 survival data, not the principle.
- **THE AIM HELPS NOTHING** if G1, G3 and G4 are all not BETTER.

**The size tests (section 15), same rule.**

| Size | Units | Tree depth | Steps | Memories per step |
|---|---|---|---|---|
| S (as in section 9) | 96 | 2 | 3 | 16 |
| M | 192 | 2 | 4 | 24 |
| L | 256 | 3 | 6 | 32 |
| XL | 384 | 3 | 8 | 48 |

The best size is chosen by held-out TRAIN r only; TEST decides nothing.

| Test | Comparison | Question |
|---|---|---|
| **Z1** | CHOSEN vs S | Does bigger help? **Primary.** |
| **Z2** | CHOSEN vs a plain net of the same parameter count | Does size help the engine's design, or any network? |
| **Z3** | CHOSEN vs RIDGE | For reference. |

**Verdicts, fixed now.**
- **BIGGER HELPS** if Z1 is BETTER.
- **…AND THE ENGINE'S DESIGN MATTERS** if Z2 is also BETTER.
- **SIZE DOES NOT HELP** if S is chosen or Z1 is not BETTER.

Expectation, written before the run: with ~2,300 training genes and an upper bound of 0.315, gains will be small, and the largest sizes may memorise.

**Caveats, stated now.**
- The LLM was trained on public text that may include papers describing these very knockdowns. A gain from the LLM could partly be remembered results, not reasoning.
- One cell line (K562), steady state.
- The upper bound is the same reagent on a different day.

**Licences.**
- Replogle 2022 Perturb-seq: CC BY 4.0.
- Human Protein Atlas v25: CC BY 4.0. Its interaction edges come from IntAct, BioGRID, BioPlex and OpenCell, each under its own terms.
- DepMap 24Q4: CC BY 4.0.
- Reactome: CC0.
- Qwen3: Apache 2.0.
""")

code(r"""
# ---- 0. setup ---------------------------------------------------------------------------------------
import os, sys, json, math, time, hashlib, gzip, zipfile, io, csv, collections, subprocess, urllib.request
SMOKE = os.environ.get("CELL_SMOKE") == "1"           # CPU smoke test of the non-LLM parts (used before release)
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)   # unused here; old versions break peft
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58", "accelerate>=1.0",
                    "bitsandbytes>=0.45", "peft>=0.13", "h5py", "scipy"], check=True)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import binomtest

DATA = os.environ.get("CELL_DATA", "/content/data")
OUTD = os.environ.get("CELL_OUT", "/content/outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"         # 30.5B parameters (MoE, 3.3B active), Apache 2.0
FALLBACK_MODEL_ID = "Qwen/Qwen3-32B"                  # dense 32.8B, used only if the MoE cannot load on this GPU
EMB_LAYER_FRAC = 0.66                                  # which layer gives the context vector (fixed in advance)
N_EXPLAIN, MIN_GAIN = 6, 0.01
SEEDS = (0,) if SMOKE else (0, 1, 2)
EPOCHS = 3 if SMOKE else 150
DEV = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEV == "cuda" else 0.0
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", f"{VRAM_GB:.0f} GB")
if DEV != "cuda" and not SMOKE:
    print("WARNING: no GPU. Runtime > Change runtime type > A100 GPU, then run again.")
torch.manual_seed(0); np.random.seed(0)
""")

code(r"""
# ---- 1. download the data ---------------------------------------------------------------------------
FILES = {
    "gwps.h5ad": "https://ndownloader.figshare.com/files/35773217",          # K562 genome-wide Perturb-seq, CC BY 4.0
    "ess.h5ad": "https://ndownloader.figshare.com/files/35780870",           # K562 essential (independent experiment)
    "proteinatlas.tsv.zip": "https://www.proteinatlas.org/download/proteinatlas.tsv.zip",
    "rna_celline.tsv.zip": "https://www.proteinatlas.org/download/tsv/rna_celline.tsv.zip",
    "subcellular_location.tsv.zip": "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip",
    "interaction_consensus.tsv.zip": "https://www.proteinatlas.org/download/tsv/interaction_consensus.tsv.zip",
    "CRISPRGeneEffect.csv": "https://ndownloader.figshare.com/files/51064667",  # DepMap 24Q4, CC BY 4.0
    "ReactomePathways.gmt.zip": "https://reactome.org/download/current/ReactomePathways.gmt.zip",
}

def fetch(name, url, tries=5):
    path = os.path.join(DATA, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cell-notebook"})
            with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as fh:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
            os.replace(path + ".part", path)
            return path
        except Exception as ex:
            print(f"  {name}: attempt {k + 1} failed ({ex}); retrying")
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f"could not download {name}")

t0 = time.time()
P = {n: fetch(n, u) for n, u in FILES.items()}
for n, p in P.items():
    print(f"  {n:32s} {os.path.getsize(p) / 1e6:8.1f} MB")
print(f"downloads ready in {time.time() - t0:.0f}s")
""")

code(r"""
# ---- 2. Perturb-seq onto the GPU ------------------------------------------------------------------
import h5py

def load_screen(path):
    # One row per knocked-down gene (its strongest-knockdown row), plus non-targeting control rows.
    f = h5py.File(path, "r")
    X = f["X"][()]
    gt = [s.decode() for s in f["obs/gene_transcript"][()]]
    fold, core, ep = f["obs/fold_expr"][()], f["obs/core_control"][()], f["obs/energy_test_p_value"][()]
    cats = [s.decode() for s in f["var/__categories/gene_name"][()]]
    genes = [cats[c] for c in f["var/gene_name"][()]]
    best, ens = {}, {}
    for i, g in enumerate(gt):
        p = g.split("_"); sym = p[1] if len(p) > 1 else g
        if "non-targeting" in g.lower() or core[i]:
            continue
        fv = fold[i] if np.isfinite(fold[i]) else 1.0
        if sym not in best or fv < best[sym][1]:
            best[sym] = (i, fv)
        if p[-1].startswith("ENSG"):
            ens[sym] = p[-1]
    kos = sorted(best); rows = [best[k][0] for k in kos]
    Xt = torch.tensor(X[rows], dtype=torch.float32, device=DEV)
    Xt[~torch.isfinite(Xt)] = float("nan")
    return dict(X=Xt, kos=kos, ens=[ens.get(k) for k in kos], genes=genes, ep=np.array([ep[r] for r in rows]))

split = lambda k: "TRAIN" if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 2 == 0 else "TEST"   # same split as the repo

A = load_screen(P["gwps.h5ad"])
gidx = {}
for j, g in enumerate(A["genes"]):
    gidx.setdefault(g, j)
phen = [i for i, e in enumerate(A["ep"]) if np.isfinite(e) and e < 0.05]
TR = [i for i in phen if split(A["kos"][i]) == "TRAIN"]; TE = [i for i in phen if split(A["kos"][i]) == "TEST"]
ACT = A["X"].clone()
for i in phen:
    if A["kos"][i] in gidx:
        ACT[i, gidx[A["kos"][i]]] = float("nan")              # never score a knockdown on its own gene
tide = torch.nanmean(ACT[TR], 0)
Y_tr = torch.nan_to_num(ACT[TR] - tide)                      # specific responses the model learns from
A_te = ACT[TE] - tide                                        # TEST answers (NaN kept, never trained on)
print(f"knockdowns with a phenotype {len(phen):,}: TRAIN {len(TR):,}, TEST {len(TE):,}; genes {A['X'].shape[1]:,}")

def pearson_rows(Pm, Am):
    m = torch.isfinite(Am) & torch.isfinite(Pm)
    n = m.sum(1, keepdim=True).clamp(min=1)
    Pz, Az = torch.where(m, Pm, 0.0), torch.where(m, Am, 0.0)
    dp = torch.where(m, Pm - Pz.sum(1, keepdim=True) / n, 0.0)
    da = torch.where(m, Am - Az.sum(1, keepdim=True) / n, 0.0)
    return (dp * da).sum(1) / (dp.norm(dim=1) * da.norm(dim=1)).clamp(min=1e-12)
""")

code(r"""
# ---- 3. Human Protein Atlas: per-gene profile, locations, K-562 RNA, interactions ----------------------
def tsv_rows(zpath):
    z = zipfile.ZipFile(zpath)
    with z.open(z.namelist()[0]) as fh:
        yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")

HPA = {}
for r in tsv_rows(P["proteinatlas.tsv.zip"]):
    HPA[r["Ensembl"]] = dict(sym=r["Gene"], desc=r.get("Gene description", ""), syn=r.get("Gene synonym", ""),
                             cls=", ".join(x.strip() for x in (r.get("Protein class") or "").split(",") if x.strip() and "essential" not in x.lower()),
                             bp=r.get("Biological process", ""), mf=r.get("Molecular function", ""),
                             loc=r.get("Subcellular location", ""), cl_spec=r.get("RNA cell line specificity", ""))
LOCW = {"Enhanced": 1.0, "Supported": 0.75, "Approved": 0.5, "Uncertain": 0.25}
LOC = collections.defaultdict(dict)
for r in tsv_rows(P["subcellular_location.tsv.zip"]):
    for lev, w in LOCW.items():
        for x in (r.get(lev) or "").split(";"):
            if x.strip():
                LOC[r["Gene"]][x.strip()] = max(w, LOC[r["Gene"]].get(x.strip(), 0.0))
K562_TPM = {}
z = zipfile.ZipFile(P["rna_celline.tsv.zip"])
with z.open(z.namelist()[0]) as fh:
    next(fh)
    for line in fh:
        p = line.decode().rstrip("\n").split("\t")
        if p[2] == "K-562":
            K562_TPM[p[0]] = float(p[5])
PPI = collections.defaultdict(set)
for r in tsv_rows(P["interaction_consensus.tsv.zip"]):
    a, b = r["ensembl_gene_id_1"], r["ensembl_gene_id_2"]
    if a != b:
        PPI[a].add(b); PPI[b].add(a)
print(f"HPA genes {len(HPA):,}; with locations {len(LOC):,}; K-562 RNA {len(K562_TPM):,}; with interactions {len(PPI):,}")

KN = [A["kos"][i] for i in phen]; KE = [A["ens"][i] for i in phen]
NPH = len(phen); ROW = {i: j for j, i in enumerate(phen)}
rtr = torch.tensor([ROW[i] for i in TR]); rte = torch.tensor([ROW[i] for i in TE])
cov = dict(profile=sum(e in HPA for e in KE), location=sum(e in LOC for e in KE),
           k562_rna=sum(e in K562_TPM for e in KE), interactions=sum(e in PPI for e in KE))
print("HPA coverage of the knocked-down genes:", cov, "of", NPH)
""")

code(r"""
# ---- 4. per-gene feature blocks (no LLM yet) ---------------------------------------------------------
def multihot(dicts):
    vocab = sorted({k for d in dicts for k in d}); ix = {k: j for j, k in enumerate(vocab)}
    M = torch.zeros(len(dicts), len(vocab) + 1)
    for i, d in enumerate(dicts):
        if d:
            for k, w in d.items():
                M[i, ix[k]] = w
        else:
            M[i, -1] = 1.0
    return M

B = {}
B["hloc"] = multihot([LOC.get(e, {}) for e in KE])
B["hcls"] = multihot([{c.strip(): 1.0 for c in HPA[e]["cls"].split(",") if c.strip()} if e in HPA else {} for e in KE])
B["hexp"] = torch.tensor([[math.log1p(K562_TPM[e]), 1.0] if e in K562_TPM else [0.0, 0.0] for e in KE])
# interactions: low-rank embedding of the HPA consensus network (computed on the GPU)
allg = sorted(set(PPI) | {e for e in KE if e}); pix = {g: j for j, g in enumerate(allg)}
ii, jj = zip(*[(pix[a], pix[b]) for a in PPI for b in PPI[a]])
Adj = torch.sparse_coo_tensor(torch.tensor([ii, jj]), torch.ones(len(ii)), (len(allg), len(allg))).to(DEV)
U, S, _ = torch.svd_lowrank(Adj, q=128, niter=4)
emb = (U * S).cpu()
B["ppi"] = torch.stack([emb[pix[e]] if e in PPI else torch.zeros(emb.shape[1]) for e in KE])
# DepMap gene effects across 1,177 lines, K562 itself removed
import pandas as pd
dep = pd.read_csv(P["CRISPRGeneEffect.csv"], index_col=0)
dep = dep.drop(index=[r for r in dep.index if r == "ACH-000551"])
dep.columns = [c.split(" (")[0] for c in dep.columns]
dep = dep.loc[:, ~dep.columns.duplicated()]
D = torch.zeros(NPH, dep.shape[0] + 1)
for i, (s, e) in enumerate(zip(KN, KE)):
    name = s if s in dep.columns else (HPA[e]["sym"] if e in HPA and HPA[e]["sym"] in dep.columns else None)
    if name:
        v = torch.tensor(dep[name].values, dtype=torch.float32)
        D[i, :-1] = torch.nan_to_num(v, nan=float(np.nanmean(dep[name].values)) if np.isfinite(dep[name].values).any() else 0.0)
        D[i, -1] = 1.0
mu, sd = D[rtr].mean(0), D[rtr].std(0) + 1e-6
B["dep"] = (D - mu) / sd
print({k: tuple(v.shape) for k, v in B.items()}, "| DepMap matched", int(D[:, -1].sum()), "of", NPH)
""")

code(r"""
# ---- 5. the 30B LLM: context vectors for every knocked-down gene ---------------------------------------
def prompt_for(sym, e):
    h = HPA.get(e, {})
    lines = [f"Gene: {sym}" + (f" ({h['desc']})" if h.get("desc") else "")]
    for key, label in (("cls", "Protein class"), ("bp", "Biological process"), ("mf", "Molecular function"),
                       ("loc", "Subcellular location"), ("cl_spec", "RNA cell line specificity")):
        if h.get(key):
            lines.append(f"{label}: {h[key]}")
    if e in K562_TPM:
        lines.append(f"RNA level in K-562 cells: {K562_TPM[e]:.1f} nTPM")
    partners = sorted(HPA[p]["sym"] for p in PPI.get(e, ()) if p in HPA)[:20]
    if partners:
        lines.append("Known interaction partners: " + ", ".join(partners))
    lines.append(f"Question: in K562 erythroleukemia cells, what happens to the cell when {sym} is knocked down "
                 f"by CRISPRi? Which cellular processes and pathways go up, and which go down?")
    return "\n".join(lines)

PROMPTS = [prompt_for(s, e) for s, e in zip(KN, KE)]
print(PROMPTS[0][:800])
tok = llm = None; USED_MODEL = None

def load_llm(mid, four_bit):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    t = AutoTokenizer.from_pretrained(mid)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    kw = dict(device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(mid, **kw).eval()
    off = [d for d in getattr(m, "hf_device_map", {}).values() if str(d) in ("cpu", "disk")]
    if off:
        raise RuntimeError(f"{mid} did not fit on the GPU ({len(off)} modules offloaded)")
    return t, m

def get_llm():
    # load the LLM once: bf16 on an 80 GB GPU, 4-bit on 40 GB; fall back to the dense 32B model if needed
    global tok, llm, USED_MODEL
    if llm is not None:
        return
    four = VRAM_GB < 75
    try:
        tok, llm = load_llm(MODEL_ID, four); USED_MODEL = MODEL_ID
    except Exception as ex:
        print("primary model failed:", ex, "-> falling back to", FALLBACK_MODEL_ID, "(4-bit)")
        llm = None; torch.cuda.empty_cache()
        tok, llm = load_llm(FALLBACK_MODEL_ID, True); USED_MODEL = FALLBACK_MODEL_ID
    print("LLM:", USED_MODEL, "| 4-bit" if four else "| bf16", f"| GPU memory used {torch.cuda.memory_allocated() / 1e9:.1f} GB")

emb_path = os.path.join(OUTD, "llm_context_vectors.npz")
if SMOKE:
    E_llm = torch.randn(NPH, 64)                               # stand-in; the real run uses the LLM below
    print("SMOKE: random stand-in context vectors")
elif os.path.exists(emb_path) and list(np.load(emb_path, allow_pickle=True)["genes"]) == KN:
    z_ = np.load(emb_path, allow_pickle=True); E_llm = torch.tensor(z_["E"]); USED_MODEL = str(z_["model"])
    print("loaded cached context vectors from", USED_MODEL)
else:
    get_llm()
    layers = llm.model.layers; L_ix = int(round(EMB_LAYER_FRAC * (len(layers) - 1)))
    grab = {}
    hook = layers[L_ix].register_forward_hook(lambda m, i, o: grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    texts = [chat(p) for p in PROMPTS]
    order = np.argsort([len(t) for t in texts]); E_llm = torch.zeros(NPH, llm.config.hidden_size)
    bs, t1 = 16, time.time()
    with torch.no_grad():
        for b0 in range(0, NPH, bs):
            idx = order[b0:b0 + bs]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(llm.device)
            llm(**enc, use_cache=False)
            h = grab["h"].float(); m = enc["attention_mask"].unsqueeze(-1).float()
            E_llm[torch.tensor(idx)] = ((h * m).sum(1) / m.sum(1)).cpu()
            if b0 % (bs * 50) == 0:
                print(f"  {b0:,}/{NPH:,} genes, {time.time() - t1:.0f}s")
    hook.remove()
    np.savez(emb_path, E=E_llm.numpy(), genes=np.array(KN, dtype=object), model=USED_MODEL, layer=L_ix)
    print(f"context vectors {tuple(E_llm.shape)} from layer {L_ix} in {time.time() - t1:.0f}s")

# compress to 256 dimensions with a PCA fitted on TRAIN genes only
E0 = E_llm - E_llm[rtr].mean(0)
_, _, V = torch.pca_lowrank(E0[rtr], q=min(256, len(rtr) - 1, E0.shape[1]), center=False)
B["llm"] = E0 @ V
g = torch.Generator().manual_seed(7)
B["llm_shuffled"] = B["llm"][torch.randperm(NPH, generator=g)]
""")

code(r"""
# ---- 6. what the LLM says about a few genes (plain-text context) --------------------------------------
def generate(prompts, max_new_tokens=220):
    tok.padding_side = "left"
    chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(chat, return_tensors="pt", padding=True).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    tok.padding_side = "right"
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

if llm is not None:
    for s, txt in zip(KN[:3], generate(PROMPTS[:3])):
        print(f"=== {s} ===\n{txt}\n")
else:
    print("(LLM not loaded in this run: cached vectors or smoke test)")
""")

code(r"""
# ---- 7. design matrices: every model gets the same inputs ----------------------------------------------
def design(names):
    out = []
    for n in names:
        M = B[n].float()
        c = float((M[rtr] ** 2).sum(1).mean()) + 1e-12          # each block scaled to unit mean squared row norm
        out.append(M / math.sqrt(c))
    return torch.cat(out, 1).to(DEV)

BASE = ["dep", "ppi", "hloc", "hcls", "hexp"]
X_full, X_nollm, X_shuf = design(BASE + ["llm"]), design(BASE), design(BASE + ["llm_shuffled"])
print("features:", X_full.shape[1], "with LLM,", X_nollm.shape[1], "without")
# targets in a compact basis: top 128 principal components of TRAIN responses
_, _, Vy = torch.pca_lowrank(Y_tr, q=128, center=False)
Ztr = Y_tr @ Vy                                              # TRAIN responses as 128 numbers each
print("128 components keep", f"{float((Ztr ** 2).sum() / (Y_tr ** 2).sum()):.1%}", "of TRAIN response energy")
""")

code(r"""
# ---- 8. plain learner: ridge with lambda by 5-fold CV inside TRAIN -------------------------------------
LAMS = [10.0 ** k for k in range(-2, 5)]

def ridge_fit_predict(Xtr, Ytr, Xte, lams):
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    U_, S_, Vt_ = torch.linalg.svd(Xtr - xm, full_matrices=False)
    UtY = U_.T @ (Ytr - ym); FV = (Xte - xm) @ Vt_.T
    return {l: FV @ ((S_ / (S_ ** 2 + l))[:, None] * UtY) + ym for l in lams}

def ridge(X, Ytr_, shuffle=False):
    Xtr, Xte = X[rtr], X[rte]
    Yt = Ytr_[torch.randperm(len(Ytr_), generator=torch.Generator().manual_seed(11))] if shuffle else Ytr_
    folds = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(2)).chunk(5)
    acc = collections.defaultdict(list)
    for f in folds:
        keep = torch.ones(len(Xtr), dtype=torch.bool); keep[f] = False
        for l, Pm in ridge_fit_predict(Xtr[keep.to(DEV)], Yt[keep.to(DEV)], Xtr[f.to(DEV)], LAMS).items():
            acc[l].append(float(pearson_rows(Pm, Yt[f.to(DEV)]).median()))
    lam = max(LAMS, key=lambda l: np.mean(acc[l]))
    return ridge_fit_predict(Xtr, Yt, Xte, [lam])[lam], lam

PRED, INFO = {}, {}
for name, X in (("RIDGE", X_full), ("RIDGE-noLLM", X_nollm)):
    PRED[name], INFO[name] = ridge(X, Y_tr)
PRED["RIDGE shuffled answers"], _ = ridge(X_full, Y_tr, shuffle=True)
print("ridge lambdas:", INFO)
""")

code(r"""
# ---- 9. the reasoning engine: bump-node dendritic tree, 3 steps, snap to real experiments ---------------
class BumpTree(nn.Module):
    # units x (branch ** levels) linear leaves; each level sums `branch` children, then a learnt bump:
    # out = v + a * exp(-(v - mu)^2 / (s^2 + 0.01))   (the node that beat the ion-channel node 10/10)
    def __init__(self, d_in, units=96, branch=4, levels=2):
        super().__init__()
        self.u, self.b, self.L = units, branch, levels
        self.leaf = nn.Linear(d_in, units * branch ** levels)
        self.w, self.a, self.mu, self.s = (nn.ParameterList() for _ in range(4))
        nodes = branch ** levels
        for _ in range(levels):
            nodes //= branch
            self.w.append(nn.Parameter(torch.randn(units, nodes, branch) / math.sqrt(branch)))
            self.a.append(nn.Parameter(torch.full((units, nodes), 0.5)))
            self.mu.append(nn.Parameter(torch.zeros(units, nodes)))
            self.s.append(nn.Parameter(torch.ones(units, nodes)))

    def forward(self, z):
        h = self.leaf(z).view(z.shape[0], self.u, -1)
        for l in range(self.L):
            v = (h.view(h.shape[0], self.u, -1, self.b) * self.w[l]).sum(-1)
            h = v + self.a[l] * torch.exp(-(v - self.mu[l]) ** 2 / (self.s[l] ** 2 + 0.01))
        return h.squeeze(-1)


class Engine(nn.Module):
    # state h; each step: query the memory of measured experiments, SNAP to the k most similar real ones,
    # read their responses, update h with the bump tree. Output mixes the snapped real responses with a decoder.
    def __init__(self, d_x, d_y, units=96, steps=3, k=16, d_key=64):
        super().__init__()
        self.steps, self.k = steps, k
        self.enc = nn.Sequential(nn.Linear(d_x, units), nn.Tanh())
        self.q, self.key, self.val = nn.Linear(units, d_key), nn.Linear(d_x, d_key), nn.Linear(d_y, units)
        self.tree = BumpTree(d_x + 2 * units, units)
        self.out, self.mix = nn.Linear(units, d_y), nn.Linear(units, 1)

    def forward(self, x, mx, my, self_idx=None):
        h = self.enc(x); K = self.key(mx); V = self.val(my)
        for _ in range(self.steps):
            S = self.q(h) @ K.T / math.sqrt(K.shape[1])
            if self_idx is not None:                         # a TRAIN item may not look itself up
                S = S.masked_fill(F.one_hot(self_idx, S.shape[1]).bool(), float("-inf"))
            top = S.topk(self.k, dim=1)
            w = torch.softmax(top.values, 1).unsqueeze(-1)
            r_y = (w * my[top.indices]).sum(1); r_h = (w * V[top.indices]).sum(1)
            h = torch.tanh(self.tree(torch.cat([x, r_h, h], 1)))
        g = torch.sigmoid(self.mix(h))
        return g * r_y + (1 - g) * self.out(h)


class MLP(nn.Module):
    def __init__(self, d_x, d_y, hidden=384):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_x, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_y))

    def forward(self, x, mx=None, my=None, self_idx=None):
        return self.f(x)


def mlp_hidden(d_x, d_y):
    # hidden width that gives the plain net about as many parameters as the engine
    P_eng = sum(p.numel() for p in Engine(d_x, d_y).parameters())
    b = d_x + d_y + 3
    return int((-b + math.sqrt(b * b + 4 * P_eng)) / 2)


def train_eval(kind, X, seed):
    # 90/10 split of TRAIN for early stopping; loss = 1 - Pearson r in gene space; memory = TRAIN items only
    torch.manual_seed(seed)
    n = len(rtr); perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed)).to(DEV)
    va, fi = perm[: n // 10], perm[n // 10:]
    Xtr, Xte = X[rtr.to(DEV)], X[rte.to(DEV)]
    m = (Engine(X.shape[1], Ztr.shape[1]) if kind == "engine" else MLP(X.shape[1], Ztr.shape[1], mlp_hidden(X.shape[1], Ztr.shape[1]))).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    mx, my = Xtr[fi], Ztr[fi]                                    # the memory during training
    slot = torch.full((n,), -1, dtype=torch.long, device=DEV); slot[fi] = torch.arange(len(fi), device=DEV)
    best, state = -2.0, None
    for ep in range(EPOCHS):
        m.train()
        for b in fi[torch.randperm(len(fi), device=DEV)].split(256):
            pz = m(Xtr[b], mx, my, self_idx=slot[b] if kind == "engine" else None)
            loss = 1 - pearson_rows(pz @ Vy.T, Y_tr[b]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            m.eval()
            with torch.no_grad():
                r = float(pearson_rows(m(Xtr[va], mx, my) @ Vy.T, Y_tr[va]).median())
            if r > best:
                best, state = r, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad():                                        # at test time the memory is all of TRAIN
        return (m(Xte, Xtr, Ztr) @ Vy.T), best, sum(p.numel() for p in m.parameters())

for name, kind, X in (("ENGINE", "engine", X_full), ("ENGINE-noLLM", "engine", X_nollm),
                      ("ENGINE-shuffledLLM", "engine", X_shuf), ("MLP", "mlp", X_full)):
    t1 = time.time(); outs = [train_eval(kind, X, s) for s in SEEDS]
    PRED[name] = torch.stack([o[0] for o in outs]).mean(0)
    INFO[name] = dict(val_r=[round(o[1], 3) for o in outs], params=outs[0][2])
    print(f"{name:20s} seeds {len(SEEDS)}  validation r {INFO[name]['val_r']}  params {outs[0][2]:,}  {time.time() - t1:.0f}s")
""")

code(r"""
# ---- 10. scores on TEST, the upper bound, and the predeclared verdicts -------------------------------
R = {n: torch.nan_to_num(pearson_rows(Pm, A_te)).cpu().numpy() for n, Pm in PRED.items()}
Bs = load_screen(P["ess.h5ad"]); bk = {p: i for i, p in enumerate(Bs["kos"])}
bg = {}
for j, g in enumerate(Bs["genes"]):
    bg.setdefault(g, j)
cols = [j for j, g in enumerate(A["genes"]) if g in bg]; bcols = [bg[A["genes"][j]] for j in cols]
btr = [bk[A["kos"][i]] for i in TR if A["kos"][i] in bk]
btide = torch.nanmean(Bs["X"][btr][:, bcols], 0)
te_in_b = [t for t, i in enumerate(TE) if A["kos"][i] in bk]
Bp = torch.nan_to_num(Bs["X"][[bk[A["kos"][TE[t]]] for t in te_in_b]][:, bcols] - btide)
ceil = torch.nan_to_num(pearson_rows(Bp, A_te[te_in_b][:, cols])).cpu().numpy()

print("TEST knockdowns: median / mean r of the predicted specific response")
for n in ("RIDGE", "RIDGE-noLLM", "MLP", "ENGINE", "ENGINE-noLLM", "ENGINE-shuffledLLM", "RIDGE shuffled answers"):
    print(f"  {n:24s} {np.median(R[n]):+.3f} / {np.mean(R[n]):+.3f}")
print(f"  {'UPPER BOUND: screen B':24s} {np.median(ceil):+.3f} / {np.mean(ceil):+.3f}   ({len(ceil)} TEST genes also in B)")

def verdict(a, b):
    d = R[a] - R[b]; d = d[d != 0]
    w, l = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0; md = float(np.median(R[a] - R[b]))
    v = "BETTER" if p < 0.05 and md >= MIN_GAIN else "WORSE" if p < 0.05 and md <= -MIN_GAIN else "NO DIFFERENCE"
    return dict(better=w, worse=l, p=float(p), median_diff=md, verdict=v)

h0 = float(np.median(R["RIDGE shuffled answers"])) < 0.02
print(f"\nH0 leak check: shuffled-answer median r {np.median(R['RIDGE shuffled answers']):+.3f} -> {'PASS' if h0 else 'FAIL'}")
GATES = {}
for tag, a, b in (("E1 engine vs plain learner", "ENGINE", "RIDGE"), ("E2 LLM context helps the engine", "ENGINE", "ENGINE-noLLM"),
                  ("E3 LLM context helps the plain learner", "RIDGE", "RIDGE-noLLM"),
                  ("E4 LLM content matters (vs shuffled)", "ENGINE", "ENGINE-shuffledLLM"), ("E5 memory + steps vs plain net", "ENGINE", "MLP")):
    GATES[tag] = verdict(a, b)
    g_ = GATES[tag]
    print(f"{tag:42s} better {g_['better']:,} / worse {g_['worse']:,}  median diff {g_['median_diff']:+.4f}  p {g_['p']:.2g} -> {g_['verdict']}")
if GATES["E2 LLM context helps the engine"]["verdict"] == "BETTER" and GATES["E4 LLM content matters (vs shuffled)"]["verdict"] != "BETTER":
    print("NOTE: E2 does not count -- shuffled LLM vectors did as well, so the gain is not from the LLM's content.")
""")

code(r"""
# ---- 11. LLM action: explain the engine's predictions for a few random TEST genes ------------------------
GMT = {}
z = zipfile.ZipFile(P["ReactomePathways.gmt.zip"])
for line in z.read(z.namelist()[0]).decode().splitlines():
    p = line.split("\t"); mem = [gidx[g] for g in p[2:] if g in gidx]
    if 15 <= len(mem) <= 300:
        GMT[p[0]] = mem
names = list(GMT); Mp = torch.zeros(A["X"].shape[1], len(names), device=DEV)
for c, n in enumerate(names):
    Mp[GMT[n], c] = 1.0
pw = lambda v: (torch.nan_to_num(v) @ Mp) / Mp.sum(0)
rng = np.random.default_rng(3); pick = rng.choice(len(TE), size=min(N_EXPLAIN, len(TE)), replace=False)   # random, not cherry-picked
REPORTS = []
for t in pick:
    s, e = A["kos"][TE[t]], A["ens"][TE[t]]
    pp, pa = pw(PRED["ENGINE"][t]), pw(A_te[t])
    up = [names[i] for i in pp.argsort(descending=True)[:5].tolist()]; dn = [names[i] for i in pp.argsort()[:5].tolist()]
    mup = [names[i] for i in pa.argsort(descending=True)[:5].tolist()]; mdn = [names[i] for i in pa.argsort()[:5].tolist()]
    rep = dict(gene=s, r=float(R["ENGINE"][t]), predicted_up=up, predicted_down=dn, measured_up=mup, measured_down=mdn)
    if llm is not None:
        q = (prompt_for(s, e).rsplit("Question:", 1)[0] + f"\nA reasoning engine predicts that knocking down {s} in K562 cells "
             f"RAISES: {'; '.join(up)}\nand LOWERS: {'; '.join(dn)}.\nIn 3-4 sentences, give the most likely mechanism, "
             f"say which parts you doubt, and propose one experiment that would test it. This is a hypothesis.")
        rep["llm_hypothesis"] = generate([q], max_new_tokens=260)[0]
    REPORTS.append(rep)
    print(f"=== {s}  (engine r on this gene {rep['r']:+.2f}) ===")
    print("  predicted up:  ", "; ".join(up)); print("  measured up:   ", "; ".join(mup))
    print("  predicted down:", "; ".join(dn)); print("  measured down: ", "; ".join(mdn))
    if "llm_hypothesis" in rep:
        print("  LLM hypothesis:", rep["llm_hypothesis"])
    print()
""")

code(r"""
# ---- 12. fine-tune the LLM: gene document in, measured knockdown response out (QLoRA) -------------------
# Only small LoRA adapters on the attention layers and a linear prediction head are trained, on TRAIN genes.
# The head reads the mean of the LLM's last hidden layer and predicts the 128 response components; the loss is
# the same as the engine's (1 - Pearson r in gene space). 10% of TRAIN picks the best epoch; TEST is used once.
FT = {}

def tiny_llm_for_smoke_test():
    # used only by the CPU smoke test before release: a 2-layer random Qwen3-MoE with the real tokenizer
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    t = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = Qwen3MoeConfig(vocab_size=len(t), hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=8, num_experts_per_tok=2, max_position_embeddings=4096)
    return t, Qwen3MoeForCausalLM(cfg)

if SMOKE and llm is None and os.environ.get("CELL_TINY_LLM") == "1":
    tok, llm = tiny_llm_for_smoke_test(); USED_MODEL = "tiny smoke-test model"
if llm is None and not SMOKE:
    print("Loading the LLM for fine-tuning (the context vectors above came from the cache).")
    get_llm()
DO_FT = llm is not None
if not DO_FT:
    print("Fine-tuning skipped: no LLM in this smoke test.")
else:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    FT_EPOCHS = 1 if SMOKE else (2 if USED_MODEL == MODEL_ID else 1)
    FT_BATCH, FT_ACCUM, FT_MAXLEN = 8, 2, 512
    tok.padding_side = "right"
    llm.config.use_cache = False
    gc_kw = dict(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(llm, "is_loaded_in_4bit", False):
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True, **gc_kw)
    else:
        llm.gradient_checkpointing_enable(**gc_kw)
    pm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    dec = pm.get_base_model().model                     # transformer body with LoRA inside; no vocabulary head
    dev0 = next(dec.parameters()).device
    head = nn.Sequential(nn.LayerNorm(llm.config.hidden_size), nn.Linear(llm.config.hidden_size, Ztr.shape[1])).to(DEV)
    CHAT = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in PROMPTS]

    def pooled(ix):
        enc = tok([CHAT[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=FT_MAXLEN)
        enc = {k: v.to(dev0) for k, v in enc.items()}
        h = dec(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1)).to(DEV)

    def ft_predict(phen_rows, bs=16):
        pm.eval(); head.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(phen_rows), bs):
                out.append(head(pooled(phen_rows[k:k + bs])) @ Vy.T)
        return torch.cat(out)

    n_tr = len(rtr); perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(100))
    va, fi = perm[: n_tr // 10], perm[n_tr // 10:]
    if SMOKE:
        va, fi = va[:32], fi[:64]
    rtr_l = rtr.tolist()
    lora = [q for q in pm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}])
    print(f"fine-tuning {USED_MODEL}: {sum(q.numel() for q in lora):,} LoRA + {sum(q.numel() for q in head.parameters()):,} head "
          f"parameters; {len(fi):,} TRAIN genes, {len(va):,} held out; {FT_EPOCHS} epoch(s)")
    best, best_state, t1, step = -2.0, None, time.time(), 0
    for ep in range(FT_EPOCHS):
        pm.train(); head.train(); opt.zero_grad()
        for k, b in enumerate(fi[torch.randperm(len(fi))].split(FT_BATCH)):
            z = head(pooled([rtr_l[j] for j in b.tolist()]))
            loss = (1 - pearson_rows(z @ Vy.T, Y_tr[b.to(DEV)]).mean()) / FT_ACCUM
            loss.backward()
            if (k + 1) % FT_ACCUM == 0:
                nn.utils.clip_grad_norm_(lora + list(head.parameters()), 1.0); opt.step(); opt.zero_grad()
            step += 1
            if step == 10:
                per = (time.time() - t1) / 10
                print(f"  {per:.1f}s per batch -> about {per * FT_EPOCHS * math.ceil(len(fi) / FT_BATCH) / 60:.0f} min of training")
        r_va = float(pearson_rows(ft_predict([rtr_l[j] for j in va.tolist()]), Y_tr[va.to(DEV)]).median())
        print(f"  epoch {ep + 1}: held-out TRAIN median r {r_va:+.3f}  ({time.time() - t1:.0f}s)")
        if r_va > best:
            best = r_va
            best_state = ({nm: q.detach().clone() for nm, q in pm.named_parameters() if q.requires_grad},
                          {kk: v.detach().clone() for kk, v in head.state_dict().items()})
    with torch.no_grad():
        for nm, q in pm.named_parameters():
            if nm in best_state[0]:
                q.copy_(best_state[0][nm])
    head.load_state_dict(best_state[1])
    te_rows = rte.tolist()
    PRED["LLM-FT"] = ft_predict(te_rows)
    sh = np.random.default_rng(5).permutation(len(te_rows))
    while (sh == np.arange(len(te_rows))).any():
        sh = np.random.default_rng(int(sh[0]) + 6).permutation(len(te_rows))
    PRED["LLM-FT reading another gene's document"] = ft_predict([te_rows[i] for i in sh])
    PRED["RIDGE on frozen-LLM vectors"], _ = ridge(design(["llm"]), Y_tr)
    unitrow = lambda M: (M - M.mean(1, keepdim=True)) / (M - M.mean(1, keepdim=True)).norm(dim=1, keepdim=True).clamp(min=1e-12)
    PRED["ENSEMBLE (LLM-FT + RIDGE)"] = (unitrow(PRED["LLM-FT"]) + unitrow(PRED["RIDGE"])) / 2   # same scale first: the FT loss ignores scale
    FT = dict(model=USED_MODEL, epochs=FT_EPOCHS, best_heldout_r=best, minutes=(time.time() - t1) / 60)
    for nm in ("LLM-FT", "LLM-FT reading another gene's document", "RIDGE on frozen-LLM vectors", "ENSEMBLE (LLM-FT + RIDGE)"):
        R[nm] = torch.nan_to_num(pearson_rows(PRED[nm], A_te)).cpu().numpy()
        print(f"  {nm:40s} {np.median(R[nm]):+.3f} / {np.mean(R[nm]):+.3f}")
    print(f"  {'RIDGE (for comparison)':40s} {np.median(R['RIDGE']):+.3f} / {np.mean(R['RIDGE']):+.3f}")
    for tag, a, b in (("F1 fine-tuned LLM vs plain learner", "LLM-FT", "RIDGE"),
                      ("F2 fine-tuning vs frozen LLM", "LLM-FT", "RIDGE on frozen-LLM vectors"),
                      ("F3 reads the document (vs another gene's)", "LLM-FT", "LLM-FT reading another gene's document"),
                      ("F4 ensemble vs plain learner", "ENSEMBLE (LLM-FT + RIDGE)", "RIDGE")):
        GATES[tag] = verdict(a, b); g_ = GATES[tag]
        print(f"{tag:44s} better {g_['better']:,} / worse {g_['worse']:,}  median diff {g_['median_diff']:+.4f}  p {g_['p']:.2g} -> {g_['verdict']}")
    if GATES["F3 reads the document (vs another gene's)"]["verdict"] != "BETTER":
        print("NOTE: F1/F2 do not count -- predictions were as good with another gene's document.")
""")

code(r"""
# ---- 13. the nameless-documents test: biology from the text, or recall of a named gene? -------------------
# Fine-tunes again from scratch (fresh LoRA + head, identical settings, same TRAIN / held-out / TEST genes) on
# documents with the gene's own names removed. A second run also replaces partner names with a count.
import re
NAMELESS = {}
RUN_STRICT = True

def prompt_nameless(sym, e, partners=True):
    h = HPA.get(e, {})
    lines = ["Gene: [name withheld]"]
    for key, label in (("cls", "Protein class"), ("bp", "Biological process"), ("mf", "Molecular function"),
                       ("loc", "Subcellular location"), ("cl_spec", "RNA cell line specificity")):
        if h.get(key):
            lines.append(f"{label}: {h[key]}")
    if e in K562_TPM:
        lines.append(f"RNA level in K-562 cells: {K562_TPM[e]:.1f} nTPM")
    ps = sorted(HPA[q]["sym"] for q in PPI.get(e, ()) if q in HPA)
    if ps:
        lines.append("Known interaction partners: " + ", ".join(ps[:20]) if partners else f"Number of known interaction partners: {len(ps)}")
    lines.append("Question: in K562 erythroleukemia cells, what happens to the cell when this gene is knocked down "
                 "by CRISPRi? Which cellular processes and pathways go up, and which go down?")
    text = "\n".join(lines)
    names = {sym, h.get("sym", "")} | {x.strip() for x in h.get("syn", "").split(",") if x.strip()}
    hits = 0
    for nm in sorted((x for x in names if len(x) >= 2), key=len, reverse=True):   # any leftover own-name mention
        text, k = re.subn(r"(?<![A-Za-z0-9-])" + re.escape(nm) + r"(?![A-Za-z0-9-])", "[name withheld]", text)
        hits += k
    return text, hits

def finetune_on(texts, label):
    # identical to section 12, but starting again from fresh adapters and a fresh head
    torch.manual_seed(0)
    for mod in pm.modules():
        if hasattr(mod, "reset_lora_parameters") and "default" in getattr(mod, "lora_A", {}):
            mod.reset_lora_parameters("default", True)
    hd = nn.Sequential(nn.LayerNorm(llm.config.hidden_size), nn.Linear(llm.config.hidden_size, Ztr.shape[1])).to(DEV)
    chat = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in texts]

    def pool(ix):
        enc = tok([chat[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=FT_MAXLEN)
        enc = {k: v.to(dev0) for k, v in enc.items()}
        h = dec(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1)).to(DEV)

    def predict(rows, bs=16):
        pm.eval(); hd.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(rows), bs):
                out.append(hd(pool(rows[k:k + bs])) @ Vy.T)
        return torch.cat(out)

    lora = [q for q in pm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": hd.parameters(), "lr": 1e-3}])
    best, state, t1 = -2.0, None, time.time()
    for ep in range(FT_EPOCHS):
        pm.train(); hd.train(); opt.zero_grad()
        for k, b in enumerate(fi[torch.randperm(len(fi))].split(FT_BATCH)):
            z = hd(pool([rtr_l[j] for j in b.tolist()]))
            loss = (1 - pearson_rows(z @ Vy.T, Y_tr[b.to(DEV)]).mean()) / FT_ACCUM
            loss.backward()
            if (k + 1) % FT_ACCUM == 0:
                nn.utils.clip_grad_norm_(lora + list(hd.parameters()), 1.0); opt.step(); opt.zero_grad()
        r_va = float(pearson_rows(predict([rtr_l[j] for j in va.tolist()]), Y_tr[va.to(DEV)]).median())
        print(f"  [{label}] epoch {ep + 1}: held-out TRAIN median r {r_va:+.3f}  ({time.time() - t1:.0f}s)")
        if r_va > best:
            best = r_va
            state = ({nm: q.detach().clone() for nm, q in pm.named_parameters() if q.requires_grad},
                     {kk: v.detach().clone() for kk, v in hd.state_dict().items()})
    with torch.no_grad():
        for nm, q in pm.named_parameters():
            if nm in state[0]:
                q.copy_(state[0][nm])
    hd.load_state_dict(state[1])
    te_rows = rte.tolist()
    return predict(te_rows), predict([te_rows[i] for i in sh]), best

if not DO_FT:
    print("Nameless test skipped: no fine-tuning in this run.")
else:
    runs = [("nameless", True)] + ([("nameless, no partner names", False)] if RUN_STRICT else [])
    unit_ = lambda M: (M - M.mean(1, keepdim=True)) / (M - M.mean(1, keepdim=True)).norm(dim=1, keepdim=True).clamp(min=1e-12)
    for tag, partners in runs:
        made = [prompt_nameless(sname, e, partners) for sname, e in zip(KN, KE)]
        texts = [t for t, _ in made]
        left = sum(1 for t, sname in zip(texts, KN) if re.search(r"(?<![A-Za-z0-9-])" + re.escape(sname) + r"(?![A-Za-z0-9-])", t))
        print(f"{tag}: own-name mentions removed {sum(h for _, h in made):,}; documents still naming their gene: {left}")
        print("  example:", texts[0][:400].replace("\n", " | "))
        p_ft, p_other, best = finetune_on(texts, tag)
        PRED[f"LLM-FT {tag}"] = p_ft
        PRED[f"LLM-FT {tag}, another gene's document"] = p_other
        PRED[f"ENSEMBLE {tag}"] = (unit_(p_ft) + unit_(PRED["RIDGE"])) / 2
        NAMELESS[tag] = dict(best_heldout_r=best)
        for nm in (f"LLM-FT {tag}", f"LLM-FT {tag}, another gene's document", f"ENSEMBLE {tag}"):
            R[nm] = torch.nan_to_num(pearson_rows(PRED[nm], A_te)).cpu().numpy()
            print(f"  {nm:58s} {np.median(R[nm]):+.3f} / {np.mean(R[nm]):+.3f}")
    print(f"  {'LLM-FT (named, section 12)':58s} {np.median(R['LLM-FT']):+.3f} / {np.mean(R['LLM-FT']):+.3f}")
    print(f"  {'RIDGE':58s} {np.median(R['RIDGE']):+.3f} / {np.mean(R['RIDGE']):+.3f}")
    tests = [("N1 names carried part of the gain? (nameless vs named)", "LLM-FT nameless", "LLM-FT"),
             ("N2 nameless fine-tuned LLM vs plain learner", "LLM-FT nameless", "RIDGE"),
             ("N3 nameless ensemble vs plain learner (PRIMARY)", "ENSEMBLE nameless", "RIDGE"),
             ("N4 nameless: reads the document (vs another gene's)", "LLM-FT nameless", "LLM-FT nameless, another gene's document")]
    if RUN_STRICT:
        t2 = "nameless, no partner names"
        tests += [("S1 no names at all vs named", f"LLM-FT {t2}", "LLM-FT"),
                  ("S2 no names at all vs plain learner", f"LLM-FT {t2}", "RIDGE"),
                  ("S3 no names at all, ensemble vs plain learner", f"ENSEMBLE {t2}", "RIDGE")]
    for tag, a, b in tests:
        GATES[tag] = verdict(a, b); g_ = GATES[tag]
        print(f"{tag:54s} better {g_['better']:,} / worse {g_['worse']:,}  median diff {g_['median_diff']:+.4f}  p {g_['p']:.2g} -> {g_['verdict']}")
    v = lambda t: GATES[t]["verdict"]
    n3, n4, n1 = v(tests[2][0]), v(tests[3][0]), v(tests[0][0])
    if n3 == "BETTER" and n4 == "BETTER":
        final = "GAIN SURVIVES WITHOUT NAMES"
        if RUN_STRICT and v(tests[6][0]) == "BETTER":
            final = "GAIN SURVIVES WITHOUT ANY GENE NAMES"
    elif n1 == "WORSE":
        final = "GAIN NEEDS NAMES"
    else:
        final = "UNRESOLVED (see the lines above)"
    NAMELESS["verdict"] = final
    print("\nVERDICT:", final)
""")

code(r"""
# ---- 14. the aim: the selfish-gene view, given to the engine and to the LLM -------------------------------
AIM_TEXT = ("Guiding principle (the selfish-gene view): a cell's machinery exists because it helped copies of its genes "
            "get made. A proliferating cancer cell such as K562 is organised around one aim: grow and divide. When a gene is "
            "lost, the cell rebalances between GROWTH (making ribosomes and proteins, copying DNA, dividing) and PROTECTION "
            "(stress responses, repair, recycling, quality control). Judge first how much losing this gene threatens growth, "
            "then which way the cell rebalances, then which specific processes change.")
import re
from scipy.stats import spearmanr, rankdata
AIMRES = {}
# --- the aim, measured. TRAIN values are training answers; TEST values are used only to score, never to train ---
k562_eff = {}
with open(P["CRISPRGeneEffect.csv"]) as fh:
    hdr = next(fh).rstrip("\n").split(",")
    for line in fh:
        if line.startswith("ACH-000551,"):                      # K562's own row (excluded from every input)
            for c, v in zip(hdr[1:], line.rstrip("\n").split(",")[1:]):
                if v not in ("", "NA"):
                    k562_eff[c.split(" (")[0]] = float(v)
            break

def k562_fitness(sym, e):
    if sym in k562_eff:
        return k562_eff[sym]
    alt = HPA.get(e, {}).get("sym")
    return k562_eff.get(alt, float("nan"))

fit_all = torch.tensor([k562_fitness(s_, e_) for s_, e_ in zip(KN, KE)], dtype=torch.float32)
v1 = Vy[:, 0].clone()                                           # top pattern of TRAIN responses (gene space)
ribo = [j for j, g in enumerate(A["genes"]) if re.match(r"RP[LS]\d", g)]
if float(v1[ribo].mean()) < 0:
    v1 = -v1                                                    # orient: ribosomal-protein genes up = GROWTH
if "Mp" in globals():
    lo = (v1 @ Mp) / Mp.sum(0)
    print("grow-vs-protect pattern, growth side:", "; ".join(names[i] for i in lo.argsort(descending=True)[:4].tolist()))
    print("                         protection side:", "; ".join(names[i] for i in lo.argsort()[:4].tolist()))
A_te0 = torch.nan_to_num(A_te)
aim_tr_raw = torch.stack([fit_all[rtr].to(DEV), Y_tr @ v1, Y_tr.norm(dim=1)], 1)
aim_te_raw = torch.stack([fit_all[rte].to(DEV), A_te0 @ v1, A_te0.norm(dim=1)], 1)
am_mu = torch.stack([torch.nanmean(aim_tr_raw[:, c]) for c in range(3)])
am_sd = torch.stack([aim_tr_raw[:, c][torch.isfinite(aim_tr_raw[:, c])].std() for c in range(3)]) + 1e-6
AIM_tr, AIM_te = (aim_tr_raw - am_mu) / am_sd, (aim_te_raw - am_mu) / am_sd
print(f"aim measured for TRAIN genes: threat-to-growth {int(torch.isfinite(AIM_tr[:, 0]).sum()):,} of {len(rtr):,} (K562 DepMap); "
      f"grow-vs-protect and shake-size for all")

def aim_loss(pred, tgt):
    ok = torch.isfinite(tgt)
    return ((pred - torch.nan_to_num(tgt)) ** 2 * ok).sum() / ok.sum().clamp(min=1)

def aim_scores(pred_te):
    # how well the aim itself is predicted on TEST genes (descriptive)
    pt, tt = pred_te.detach().cpu().numpy(), AIM_te.cpu().numpy()
    ok = np.isfinite(tt[:, 0])
    rho_fit = spearmanr(pt[ok, 0], tt[ok, 0])[0]
    ess = tt[ok, 0] < (-0.5 - am_mu[0].item()) / am_sd[0].item()     # DepMap effect < -0.5 = essential
    r_ = rankdata(-pt[ok, 0]); npos = int(ess.sum()); nneg = int((~ess).sum())
    auc = (r_[ess].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
    return dict(threat_rho=float(rho_fit), essential_auroc=float(auc),
                grow_protect_r=float(np.corrcoef(pt[:, 1], tt[:, 1])[0, 1]), shake_rho=float(spearmanr(pt[:, 2], tt[:, 2])[0]))

# --- the engine with the aim: it predicts the aim first, then reasons with it ------------------------------
class AimEngine(nn.Module):
    def __init__(self, d_x, d_y, n_aim=3, units=96):
        super().__init__()
        self.aim = nn.Sequential(nn.Linear(d_x, units), nn.Tanh(), nn.Linear(units, n_aim))
        self.core = Engine(d_x + n_aim, d_y, units)
        self.last_aim = None

    def forward(self, x, mx, my, self_idx=None):
        a, am = self.aim(x), self.aim(mx)
        self.last_aim = a
        return self.core(torch.cat([x, a], 1), torch.cat([mx, am], 1), my, self_idx)

def train_eval_aim(X, seed, lam=0.5):
    # identical to train_eval (section 9) plus the aim loss; the epoch is still chosen on response r only
    torch.manual_seed(seed)
    n = len(rtr); perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed)).to(DEV)
    va, fi_ = perm[: n // 10], perm[n // 10:]
    Xtr, Xte = X[rtr.to(DEV)], X[rte.to(DEV)]
    m = AimEngine(X.shape[1], Ztr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    mx, my = Xtr[fi_], Ztr[fi_]
    slot = torch.full((n,), -1, dtype=torch.long, device=DEV); slot[fi_] = torch.arange(len(fi_), device=DEV)
    best, state = -2.0, None
    for ep in range(EPOCHS):
        m.train()
        for b in fi_[torch.randperm(len(fi_), device=DEV)].split(256):
            pz = m(Xtr[b], mx, my, self_idx=slot[b])
            loss = 1 - pearson_rows(pz @ Vy.T, Y_tr[b]).mean() + lam * aim_loss(m.last_aim, AIM_tr[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            m.eval()
            with torch.no_grad():
                r = float(pearson_rows(m(Xtr[va], mx, my) @ Vy.T, Y_tr[va]).median())
            if r > best:
                best, state = r, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        out = m(Xte, Xtr, Ztr) @ Vy.T
        return out, m.last_aim.clone(), best

t1 = time.time(); outs = [train_eval_aim(X_full, s_) for s_ in SEEDS]
PRED["AIM-ENGINE"] = torch.stack([o[0] for o in outs]).mean(0)
AIMRES["AIM-ENGINE aim"] = aim_scores(torch.stack([o[1] for o in outs]).mean(0))
print(f"AIM-ENGINE seeds {len(SEEDS)}  validation r {[round(o[2], 3) for o in outs]}  {time.time() - t1:.0f}s")

# --- the plain learner given the same aim: stage 1 predicts the aim (out-of-fold for TRAIN), stage 2 uses it ---
Xtr_, Xte_ = X_full[rtr.to(DEV)], X_full[rte.to(DEV)]
folds = torch.randperm(len(rtr), generator=torch.Generator().manual_seed(4)).to(DEV).chunk(5)
oof = torch.zeros(len(rtr), 3, device=DEV); te_aim = torch.zeros(len(rte), 3, device=DEV)
for c in range(3):
    y = AIM_tr[:, c]; ok = torch.isfinite(y); err = collections.defaultdict(float)
    for f in folds:
        keep = torch.ones(len(rtr), dtype=torch.bool, device=DEV); keep[f] = False
        for l, Pm in ridge_fit_predict(Xtr_[keep & ok], y[keep & ok][:, None], Xtr_[f], LAMS).items():
            mm = ok[f]; err[l] += float(((Pm[:, 0][mm] - y[f][mm]) ** 2).sum())
    lam_c = min(err, key=err.get)
    for f in folds:
        keep = torch.ones(len(rtr), dtype=torch.bool, device=DEV); keep[f] = False
        oof[f, c] = ridge_fit_predict(Xtr_[keep & ok], y[keep & ok][:, None], Xtr_[f], [lam_c])[lam_c][:, 0]
    te_aim[:, c] = ridge_fit_predict(Xtr_[ok], y[ok][:, None], Xte_, [lam_c])[lam_c][:, 0]
AIMRES["RIDGE stage-1 aim"] = aim_scores(te_aim)
ah = torch.zeros(NPH, 3); ah[rtr] = oof.cpu(); ah[rte] = te_aim.cpu(); B["aim_hat"] = ah
PRED["RIDGE + predicted aim"], _ = ridge(design(BASE + ["llm", "aim_hat"]), Y_tr)

# --- the LLM with the aim: reads the principle, learns to predict the aim and the response ----------------------
def finetune_general(chat, label, n_aim=0, lam=0.5):
    torch.manual_seed(0)
    for mod in pm.modules():
        if hasattr(mod, "reset_lora_parameters") and "default" in getattr(mod, "lora_A", {}):
            mod.reset_lora_parameters("default", True)
    hd = nn.Sequential(nn.LayerNorm(llm.config.hidden_size), nn.Linear(llm.config.hidden_size, Ztr.shape[1] + n_aim)).to(DEV)

    def pool(ix):
        enc = tok([chat[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=FT_MAXLEN)
        enc = {k: v.to(dev0) for k, v in enc.items()}
        h = dec(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1)).to(DEV)

    def predict(rows, bs=16):
        pm.eval(); hd.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(rows), bs):
                out.append(hd(pool(rows[k:k + bs])))
        o = torch.cat(out)
        return o[:, :Ztr.shape[1]] @ Vy.T, o[:, Ztr.shape[1]:]

    lora = [q for q in pm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": hd.parameters(), "lr": 1e-3}])
    best, state, t1 = -2.0, None, time.time()
    for ep in range(FT_EPOCHS):
        pm.train(); hd.train(); opt.zero_grad()
        for k, b in enumerate(fi[torch.randperm(len(fi))].split(FT_BATCH)):
            o = hd(pool([rtr_l[j] for j in b.tolist()]))
            loss = 1 - pearson_rows(o[:, :Ztr.shape[1]] @ Vy.T, Y_tr[b.to(DEV)]).mean()
            if n_aim:
                loss = loss + lam * aim_loss(o[:, Ztr.shape[1]:], AIM_tr[b.to(DEV)])
            (loss / FT_ACCUM).backward()
            if (k + 1) % FT_ACCUM == 0:
                nn.utils.clip_grad_norm_(lora + list(hd.parameters()), 1.0); opt.step(); opt.zero_grad()
        r_va = float(pearson_rows(predict([rtr_l[j] for j in va.tolist()])[0], Y_tr[va.to(DEV)]).median())
        print(f"  [{label}] epoch {ep + 1}: held-out TRAIN median r {r_va:+.3f}  ({time.time() - t1:.0f}s)")
        if r_va > best:
            best = r_va
            state = ({nm: q.detach().clone() for nm, q in pm.named_parameters() if q.requires_grad},
                     {kk: v.detach().clone() for kk, v in hd.state_dict().items()})
    with torch.no_grad():
        for nm, q in pm.named_parameters():
            if nm in state[0]:
                q.copy_(state[0][nm])
    hd.load_state_dict(state[1])
    return predict(rte.tolist())

if DO_FT:
    chat_aim = [tok.apply_chat_template([{"role": "system", "content": AIM_TEXT}, {"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True) for q in PROMPTS]
    p_resp, p_aim = finetune_general(chat_aim, "aim", n_aim=3)
    PRED["LLM-FT-AIM"] = p_resp
    unit_ = lambda M: (M - M.mean(1, keepdim=True)) / (M - M.mean(1, keepdim=True)).norm(dim=1, keepdim=True).clamp(min=1e-12)
    PRED["ENSEMBLE-AIM (LLM-FT-AIM + RIDGE)"] = (unit_(p_resp) + unit_(PRED["RIDGE"])) / 2
    AIMRES["LLM-FT-AIM aim"] = aim_scores(p_aim)

for nm in [n for n in ("AIM-ENGINE", "RIDGE + predicted aim", "LLM-FT-AIM", "ENSEMBLE-AIM (LLM-FT-AIM + RIDGE)") if n in PRED]:
    R[nm] = torch.nan_to_num(pearson_rows(PRED[nm], A_te)).cpu().numpy()
print("\nTEST knockdowns, median / mean r:")
for nm in ("ENGINE", "AIM-ENGINE", "RIDGE", "RIDGE + predicted aim", "LLM-FT", "LLM-FT-AIM", "ENSEMBLE (LLM-FT + RIDGE)", "ENSEMBLE-AIM (LLM-FT-AIM + RIDGE)"):
    if nm in R:
        print(f"  {nm:40s} {np.median(R[nm]):+.3f} / {np.mean(R[nm]):+.3f}")
print("\nHow well the aim itself is predicted on TEST genes (descriptive):")
for nm, d_ in AIMRES.items():
    print(f"  {nm:22s} threat-to-growth rho {d_['threat_rho']:+.2f}, essential AUROC {d_['essential_auroc']:.2f}, "
          f"grow-vs-protect r {d_['grow_protect_r']:+.2f}, shake-size rho {d_['shake_rho']:+.2f}")
tests = [("G1 aim helps the engine", "AIM-ENGINE", "ENGINE"), ("G2 aim engine vs plain learner", "AIM-ENGINE", "RIDGE"),
         ("G3 aim helps the plain learner", "RIDGE + predicted aim", "RIDGE")]
if "LLM-FT-AIM" in R and "LLM-FT" in R:
    tests += [("G4 aim helps the LLM", "LLM-FT-AIM", "LLM-FT"),
              ("G5 aim ensemble vs plain learner", "ENSEMBLE-AIM (LLM-FT-AIM + RIDGE)", "RIDGE")]
print()
for tag, a, b in tests:
    GATES[tag] = verdict(a, b); g_ = GATES[tag]
    print(f"{tag:36s} better {g_['better']:,} / worse {g_['worse']:,}  median diff {g_['median_diff']:+.4f}  p {g_['p']:.2g} -> {g_['verdict']}")
v = lambda t: GATES.get(t, {}).get("verdict")
final = []
if v("G1 aim helps the engine") == "BETTER":
    final.append("THE AIM HELPS THE ENGINE")
if v("G4 aim helps the LLM") == "BETTER":
    final.append("THE AIM HELPS THE LLM")
if final and v("G3 aim helps the plain learner") == "BETTER":
    final.append("(the plain learner gains too: the gain is from the extra K562 survival data, not the principle)")
if not final and v("G3 aim helps the plain learner") != "BETTER":
    final = ["THE AIM HELPS NOTHING"]
AIMRES["verdict"] = " ".join(final) if final else "MIXED (see the lines above)"
print("\nVERDICT:", AIMRES["verdict"])

# --- what the LLM says through the selfish-gene lens, for 4 random TEST genes --------------------------------
if DO_FT and llm is not None:
    pick2 = np.random.default_rng(9).choice(len(rte), size=min(4, len(rte)), replace=False)
    qs = [prompt_for(KN[rte[t]], KE[rte[t]]).rsplit("Question:", 1)[0] +
          "Using the guiding principle, explain in 4 sentences how losing this gene shifts the cell between growth and "
          "protection, and which processes change most. This is a hypothesis." for t in pick2]
    chat_q = [tok.apply_chat_template([{"role": "system", "content": AIM_TEXT}, {"role": "user", "content": q}],
                                      tokenize=False, add_generation_prompt=True) for q in qs]
    tok.padding_side = "left"
    enc = tok(chat_q, return_tensors="pt", padding=True).to(dev0)
    with torch.no_grad(), pm.disable_adapter():
        gen = pm.generate(**enc, max_new_tokens=300, do_sample=False, use_cache=True)
    tok.padding_side = "right"
    pa_ = p_aim.cpu().numpy(); ta_ = AIM_te.cpu().numpy()
    for t, o in zip(pick2, gen):
        print(f"\n=== {KN[rte[t]]}: predicted threat {pa_[t, 0]:+.1f} (measured {ta_[t, 0]:+.1f}); "
              f"grow-vs-protect predicted {pa_[t, 1]:+.1f} (measured {ta_[t, 1]:+.1f})  [standardised units] ===")
        print(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
""")

code(r"""
# ---- 15. a bigger engine: four sizes, each beside a plain net with the same number of parameters -----------
SIZES = {"S": dict(units=96, levels=2, steps=3, k=16), "M": dict(units=192, levels=2, steps=4, k=24),
         "L": dict(units=256, levels=3, steps=6, k=32), "XL": dict(units=384, levels=3, steps=8, k=48)}
if SMOKE:
    SIZES = {k: SIZES[k] for k in ("S", "M")}
SIZERES = {}

class EngineN(Engine):
    def __init__(self, d_x, d_y, units, levels, steps, k, d_key=64):
        super().__init__(d_x, d_y, units=units, steps=steps, k=k, d_key=d_key)
        self.tree = BumpTree(d_x + 2 * units, units, levels=levels)

def n_params(m):
    return sum(q.numel() for q in m.parameters())

def mlp_for(d_x, d_y, P_target):
    b = d_x + d_y + 3
    return MLP(d_x, d_y, int((-b + math.sqrt(b * b + 4 * P_target)) / 2))

def train_eval_model(make, X, seed, is_engine):
    # the recipe of section 9 (90/10 split of TRAIN, Pearson loss, AdamW, early stopping), any model
    torch.manual_seed(seed)
    n = len(rtr); perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed)).to(DEV)
    va_, fi_ = perm[: n // 10], perm[n // 10:]
    Xtr, Xte = X[rtr.to(DEV)], X[rte.to(DEV)]
    m = make().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    mx, my = Xtr[fi_], Ztr[fi_]
    slot = torch.full((n,), -1, dtype=torch.long, device=DEV); slot[fi_] = torch.arange(len(fi_), device=DEV)
    best, state = -2.0, None
    for ep in range(EPOCHS):
        m.train()
        for b in fi_[torch.randperm(len(fi_), device=DEV)].split(256):
            pz = m(Xtr[b], mx, my, self_idx=slot[b] if is_engine else None)
            loss = 1 - pearson_rows(pz @ Vy.T, Y_tr[b]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            m.eval()
            with torch.no_grad():
                r = float(pearson_rows(m(Xtr[va_], mx, my) @ Vy.T, Y_tr[va_]).median())
            if r > best:
                best, state = r, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(Xte, Xtr, Ztr) @ Vy.T, best

d_x, d_y = X_full.shape[1], Ztr.shape[1]
print(f"{'size':5s} {'engine params':>14s} {'held-out r':>11s} {'TEST median':>12s}   {'plain net params':>16s} {'held-out r':>11s} {'TEST median':>12s}")
for nm, cfg in SIZES.items():
    t1 = time.time()
    P_eng = n_params(EngineN(d_x, d_y, **cfg))
    eo = [train_eval_model(lambda: EngineN(d_x, d_y, **cfg), X_full, s_, True) for s_ in SEEDS]
    mo = [train_eval_model(lambda: mlp_for(d_x, d_y, P_eng), X_full, s_, False) for s_ in SEEDS]
    PRED[f"ENGINE-{nm}"] = torch.stack([o[0] for o in eo]).mean(0)
    PRED[f"MLP-{nm}"] = torch.stack([o[0] for o in mo]).mean(0)
    for k_ in (f"ENGINE-{nm}", f"MLP-{nm}"):
        R[k_] = torch.nan_to_num(pearson_rows(PRED[k_], A_te)).cpu().numpy()
    SIZERES[nm] = dict(params=P_eng, engine_val=float(np.mean([o[1] for o in eo])), mlp_val=float(np.mean([o[1] for o in mo])),
                       engine_test=float(np.median(R[f"ENGINE-{nm}"])), mlp_test=float(np.median(R[f"MLP-{nm}"])))
    d_ = SIZERES[nm]
    print(f"{nm:5s} {P_eng:>14,} {d_['engine_val']:>+11.3f} {d_['engine_test']:>+12.3f}   {n_params(mlp_for(d_x, d_y, P_eng)):>16,} "
          f"{d_['mlp_val']:>+11.3f} {d_['mlp_test']:>+12.3f}   ({time.time() - t1:.0f}s)")

chosen = max(SIZES, key=lambda nm: SIZERES[nm]["engine_val"])            # chosen on held-out TRAIN only
print(f"\nchosen by held-out TRAIN r: {chosen}")
tests = [("Z1 bigger helps (chosen vs S)", f"ENGINE-{chosen}", "ENGINE-S"),
         ("Z2 engine design vs same-size plain net", f"ENGINE-{chosen}", f"MLP-{chosen}"),
         ("Z3 chosen engine vs plain learner", f"ENGINE-{chosen}", "RIDGE")]
for tag, a, b in tests:
    if a == b:
        print(f"{tag:42s} -> NOT TESTED (S was chosen)"); continue
    GATES[tag] = verdict(a, b); g_ = GATES[tag]
    print(f"{tag:42s} better {g_['better']:,} / worse {g_['worse']:,}  median diff {g_['median_diff']:+.4f}  p {g_['p']:.2g} -> {g_['verdict']}")
v = lambda t: GATES.get(t, {}).get("verdict")
if chosen != "S" and v(tests[0][0]) == "BETTER":
    final = "BIGGER HELPS" + (" AND THE ENGINE'S DESIGN MATTERS" if v(tests[1][0]) == "BETTER" else "")
else:
    final = "SIZE DOES NOT HELP"
SIZERES["chosen"], SIZERES["verdict"] = chosen, final
print("\nVERDICT:", final)
""")

code(r"""
# ---- 16. save everything ----------------------------------------------------------------------------
res = dict(model=USED_MODEL, smoke=SMOKE, device=DEV, n_train=len(TR), n_test=len(TE), hpa_coverage=cov,
           scores={n: dict(median=float(np.median(v)), mean=float(np.mean(v))) for n, v in R.items()},
           upper_bound=dict(median=float(np.median(ceil)), mean=float(np.mean(ceil)), n=len(ceil)),
           H0=bool(h0), gates=GATES, info=INFO, explanations=REPORTS, fine_tuning=FT, nameless=NAMELESS, aim=AIMRES, sizes=SIZERES)
with open(os.path.join(OUTD, "results.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("saved", os.path.join(OUTD, "results.json"))
# To keep results after the session ends:  from google.colab import drive; drive.mount('/content/drive')
# then copy /content/outputs to /content/drive/MyDrive/.
""")

nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(
    accelerator="GPU", colab=dict(gpuType="A100", machine_shape="hm", provenance=[]),
    kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python")),
    cells=[])
for k, (t, s) in enumerate(CELLS):
    c = dict(cell_type=t, metadata={}, source=s.splitlines(keepends=True), id=f"c{k:02d}")
    if t == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
open(sys.argv[2], "w").write("\n\n".join(s for t, s in CELLS if t == "code") + "\n")
print("cells:", len(CELLS))
