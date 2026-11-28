"""TRAIT2CELL -- USE 4 of the AlphaGenome 2 data: disease variants -> cell types -> genes -> drugs.

From the memory bank (rem/atlas/_cache/ag2bank): Open Targets 26.03 GWAS credible sets (fine-mapped variants, each with
a posterior probability, PIP), ENCODE Registry v4 element activity per biosample (DNase-based annotations only; the
mark-only ones say which histone marks are high, not whether an element is open), GTEx v11 SuSiE eQTL credible sets,
the use-3 switch-to-gene model (e2gmap `final`, corrected co-activity, fitted on the K562 CRISPRi DEV set only), and
the drug layer (memory_bank/cell_v6: Open Targets drugs and mechanisms per target gene).

V1  PRIMARY, KNOWN ANSWERS: which cell type does a trait's genetics point to?
    8 traits with a textbook answer (TRAITS). For each, ONE GWAS study: among the Open Targets GWAS studies whose
    reported trait matches the trait's pattern (and not 'polygenic|medication|family history'), the one with the most
    credible sets (ties: more samples, then study id). t_e = summed PIP of the study's credible-set variants that lie
    in registry element e (point overlap; a variant in two credible sets adds both).
    For each DNase-based annotation b (1,325 human), ordinary least squares over all 2.35M registry elements
        t_e ~ 1 + breadth_e + active_be
    breadth_e = share of the DNase-based annotations in which e is open (classes PLS..CA-only); active_be = 1 if e is
    open in b. Score = the t statistic of active_be: PIP in b's open elements beyond what their breadth predicts.
    The trait's answer = the biosample of the top-scoring annotation.
    Expected categories, from ENCODE's own BiosampleType slims (fetched once to outputs/trait2cell/biosample_slims.json)
    plus the explicit lists below:
      LDL cholesterol -> liver; lymphocyte count -> lymphoid; red blood cell count -> erythroid; inflammatory bowel
      disease -> immune; atrial fibrillation -> heart; schizophrenia -> brain/neural; monocyte count -> myeloid;
      estimated glomerular filtration rate -> kidney.
    PASS if >= 6 of 8 top biosamples are in the expected category AND the chance probability of at least that many
    hits is < 0.05 (Poisson-binomial; per trait, the share of the DNase-based annotations in its category).
    Scored once (lock). Descriptive, no gate: Alzheimer's disease (myeloid), type 2 diabetes (pancreas), platelet
    count (erythroid/megakaryocyte), height (none declared); top 5 annotations per trait; rank of the first expected.
V2  VARIANT -> GENE: does the use-3 model name a causal variant's gene better than "the nearest gene"?
    GTEx v11 SuSiE eQTLs in 39 tissues matched to ENCODE biosamples (TISSUES). Items: SNVs with PIP >= 0.9 lying in a
    registry element open (activity > 0) in the matched biosample(s); truth = every gene for which that variant has
    PIP >= 0.9 in that tissue. Candidates: GENCODE 46 genes with ENCODE RNA-seq whose TSS is within 1 Mb of the
    variant; items whose true genes are all outside the candidates are dropped (counted). At most 2,000 items per
    tissue (sha256 order); a variant already used in an earlier tissue (TISSUES order) is not used again.
    Model: e2gmap `final` with features computed as in its map build (element = the registry element holding the
    variant; activity of the matched biosample); predicted gene = the highest score. Baseline: the candidate whose
    TSS is nearest the variant. Exact McNemar test on paired right/wrong. BETTER if model accuracy > nearest and
    p < 0.05; WORSE if the reverse with p < 0.05; else NO DIFFERENCE. Descriptive: per tissue; variants > 10 kb from
    every candidate TSS; top-3 accuracy. Scored once (lock).
CHAIN (descriptive, needs V1): per trait, its top biosample; variants with PIP >= 0.1 in elements open there; the
    model scores their candidate genes; gene weight = sum over variants of PIP x score; top 15 genes joined to the drug
    layer. Known-answer look for LDL cholesterol: ranks of HMGCR, PCSK9, NPC1L1, APOB, LDLR, ANGPTL3, ACLY, CETP.
NOT: one study per trait; ENCODE's biosamples are uneven (many T-cell states, few liver or erythroid samples); point
    overlap only, no LD beyond what the credible sets carry; eQTL genes stand in for causal genes and favour the
    nearest-gene baseline (many eQTLs sit at promoters); the drug layer lists mechanism-of-action targets only.

Run: python3 -P rem/atlas/trait2cell.py prep | v1 | genes      (genes = V2, then the chain once V1 is scored)
"""

from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tarfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("e2gmap", os.path.join(HERE, "e2gmap.py"))
e2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e2)
ag, AG, CC = e2.ag, e2.AG, e2.CC
OT = os.path.join(AG, "genetics", "opentargets_26.03")
GTX = os.path.join(AG, "genetics", "gtex_v11", "GTEx_v11_eQTL_SuSiE.tar")
CACHE = os.path.join(HERE, "_cache", "trait2cell")
OUTD = R("outputs", "trait2cell")
DRUGL = R("memory_bank", "cell_v6")
RES = os.path.join(HERE, "RESULTS_trait2cell.txt")
OPEN_MAX = 6                                   # classes 0..6 (PLS, pELS, dELS, CA-H3K4me3, CA-CTCF, CA-TF, CA-only)

TRAITS = [  # key, label, pattern on the reported trait, expected category
    ("ldl", "LDL cholesterol", r"^LDL|low density lipoprotein cholesterol", "liver"),
    ("lymph", "lymphocyte count", r"^lymphocyte count", "lymphoid"),
    ("rbc", "red blood cell count", r"^red blood cell count|^erythrocyte count", "erythroid"),
    ("ibd", "inflammatory bowel disease", r"^inflammatory bowel disease$", "immune"),
    ("af", "atrial fibrillation", r"^atrial fibrillation", "heart"),
    ("scz", "schizophrenia", r"^schizophrenia$", "brain"),
    ("mono", "monocyte count", r"^monocyte count", "myeloid"),
    ("egfr", "eGFR", r"glomerular filtration rate", "kidney"),
]
DESC_TRAITS = [
    ("alz", "Alzheimer's disease", r"^alzheimer['’]?s disease$", "myeloid"),
    ("t2d", "type 2 diabetes", r"^type 2 diabetes", "pancreas"),
    ("plt", "platelet count", r"^platelet count", "erythroid"),
    ("height", "height", r"^(standing )?height$", None),
]
EXCLUDE = r"polygenic|medication|family history"
LYMPH_CELLS = {"T cell", "B cell", "NK cell", "lymphoblast", "CD4+ T cell", "CD8+ T cell"}
MYELOID_CELLS = {"myeloid cell", "monocyte", "macrophage", "dendritic cell", "neutrophil", "granulocyte"}
MYELOID_EXTRA = {"HL-60", "NB4", "KBM-7"}                    # myeloid leukaemia lines without a myeloid slim
ERYTHROID = {"K562", "CMK", "common myeloid progenitor, CD34-positive", "hematopoietic multipotent progenitor cell"}
IMMUNE_ORGANS = {"spleen", "thymus", "lymph node", "lymphoid tissue", "immune organ"}
LDL_KNOWN = ["HMGCR", "PCSK9", "NPC1L1", "APOB", "LDLR", "ANGPTL3", "ACLY", "CETP"]
TISSUES = [  # GTEx v11 tissue -> ENCODE biosample term(s) (most open class over them)
    ("Liver", ["liver", "right lobe of liver", "left lobe of liver"]), ("Cells_EBV-transformed_lymphocytes", ["GM12878"]),
    ("Lung", ["upper lobe of left lung", "lung"]), ("Heart_Left_Ventricle", ["heart left ventricle"]),
    ("Heart_Atrial_Appendage", ["right atrium auricular region"]), ("Muscle_Skeletal", ["gastrocnemius medialis"]),
    ("Thyroid", ["thyroid gland"]), ("Adrenal_Gland", ["adrenal gland"]), ("Stomach", ["stomach"]),
    ("Pancreas", ["body of pancreas"]), ("Spleen", ["spleen"]), ("Colon_Transverse", ["transverse colon"]),
    ("Colon_Sigmoid", ["sigmoid colon"]), ("Esophagus_Muscularis", ["esophagus muscularis mucosa"]),
    ("Esophagus_Mucosa", ["esophagus squamous epithelium"]), ("Esophagus_Gastroesophageal_Junction", ["gastroesophageal sphincter"]),
    ("Artery_Tibial", ["tibial artery"]), ("Artery_Aorta", ["thoracic aorta"]), ("Artery_Coronary", ["coronary artery"]),
    ("Nerve_Tibial", ["tibial nerve"]), ("Skin_Sun_Exposed_Lower_leg", ["lower leg skin"]),
    ("Skin_Not_Sun_Exposed_Suprapubic", ["suprapubic skin"]), ("Breast_Mammary_Tissue", ["breast epithelium"]),
    ("Prostate", ["prostate gland"]), ("Testis", ["testis"]), ("Uterus", ["uterus"]), ("Vagina", ["vagina"]),
    ("Ovary", ["ovary"]), ("Adipose_Subcutaneous", ["subcutaneous adipose tissue"]), ("Adipose_Visceral_Omentum", ["omental fat pad"]),
    ("Small_Intestine_Terminal_Ileum", ["Peyer's patch"]), ("Cells_Cultured_fibroblasts", ["fibroblast of dermis"]),
    ("Brain_Caudate_basal_ganglia", ["caudate nucleus"]), ("Brain_Putamen_basal_ganglia", ["putamen"]),
    ("Brain_Cerebellum", ["cerebellum"]), ("Brain_Frontal_Cortex_BA9", ["dorsolateral prefrontal cortex"]),
    ("Brain_Substantia_nigra", ["substantia nigra"]), ("Brain_Hippocampus", ["layer of hippocampus"]), ("Bladder", ["urinary bladder"]),
]
STAGE = {"APPROVAL": 4, "PREAPPROVAL": 3.5, "PHASE_3": 3, "PHASE_2_3": 2.5, "PHASE_2": 2, "PHASE_1_2": 1.5, "PHASE_1": 1,
         "EARLY_PHASE_1": 0.5, "IND": 0.2, "PRECLINICAL": 0.1}


# ------------------------------------------------------------------------------------------------ inputs
def slims(meta):
    p = os.path.join(OUTD, "biosample_slims.json")
    if os.path.exists(p):
        return json.load(open(p))["terms"]
    url = ("https://www.encodeproject.org/search/?type=BiosampleType&limit=all&format=json&field=term_name"
           "&field=organ_slims&field=cell_slims&field=system_slims&field=classification")
    d = json.loads(ag._get(url))["@graph"]; terms = {b["biosample"] for b in meta}; out = {}
    for r in d:
        t = r.get("term_name")
        if t in terms:
            o = out.setdefault(t, {"organ_slims": [], "cell_slims": [], "system_slims": []})
            for k in o:
                o[k] = sorted(set(o[k]) | set(r.get(k) or []))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(dict(source=url, fetched=time.strftime("%Y-%m-%d"), terms=out), open(p, "w"), indent=0)
    return out


def categories(term, sl):
    if term not in sl:                                   # no BiosampleType record under this exact name
        lym = bool(re.search(r"T cell|B cell|T-cell|T-helper|natural killer", term))
        return {"lymphoid", "immune"} if lym else set()
    o, c = set(sl[term]["organ_slims"]), set(sl[term]["cell_slims"]); k = set()
    if "liver" in o: k.add("liver")
    if c & LYMPH_CELLS or "thymus" in o: k.add("lymphoid")
    if c & MYELOID_CELLS or term in MYELOID_EXTRA: k.add("myeloid")
    if term in ERYTHROID: k.add("erythroid")
    if "leukocyte" in c or o & IMMUNE_ORGANS or k & {"lymphoid", "myeloid"}: k.add("immune")
    if "heart" in o: k.add("heart")
    if o & {"brain", "spinal cord"} or "neural cell" in c: k.add("brain")
    if "kidney" in o: k.add("kidney")
    if "pancreas" in o: k.add("pancreas")
    return k


def gwas_studies():
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    os.makedirs(CACHE, exist_ok=True); p = os.path.join(CACHE, "gwas_counts.json")
    if not os.path.exists(p):
        n = collections.Counter()
        for f in sorted(os.listdir(OT)):
            if f.startswith("credible_set_"):
                t = pq.read_table(os.path.join(OT, f), columns=["studyId", "studyType"])
                t = t.filter(pc.equal(t["studyType"], "gwas")); n.update(t["studyId"].to_pylist())
        json.dump(n, open(p, "w"))
    n = json.load(open(p))
    s = pq.read_table(os.path.join(OT, "study_0000.parquet"), columns=["studyId", "studyType", "traitFromSource", "nSamples", "projectId"])
    s = s.filter(pc.equal(s["studyType"], "gwas")).to_pylist()
    return [dict(r, n_cs=n.get(r["studyId"], 0)) for r in s if n.get(r["studyId"], 0)]


def choose(stud, pat):
    c = [r for r in stud if r["traitFromSource"] and re.search(pat, r["traitFromSource"], re.I) and not re.search(EXCLUDE, r["traitFromSource"], re.I)]
    c.sort(key=lambda r: (-r["n_cs"], -(r["nSamples"] or 0), r["studyId"]))
    return c[0] if c else None


def trait_variants(chosen):
    """{trait key: (chrom array, pos array, pip array)} from the chosen studies' credible sets (cached)."""
    import pyarrow.parquet as pq
    p = os.path.join(CACHE, "trait_variants.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False); out = {}
        for k in chosen:
            out[k] = (z[k + "_chrom"], z[k + "_pos"], z[k + "_pip"])
        return out
    sid = {v["studyId"]: k for k, v in chosen.items()}; acc = {k: collections.defaultdict(float) for k in chosen}
    for f in sorted(os.listdir(OT)):
        if f.startswith("credible_set_"):
            t = pq.read_table(os.path.join(OT, f), columns=["studyId", "locus"], filters=[("studyId", "in", list(sid))])
            for s_, loc in zip(t["studyId"].to_pylist(), t["locus"].to_pylist()):
                for v in loc or []:
                    acc[sid[s_]][v["variantId"]] += float(v["posteriorProbability"] or 0)
    out, save = {}, {}
    for k, d in acc.items():
        ch, ps, pp = [], [], []
        for vid, w in d.items():
            f = vid.split("_")
            if len(f) >= 2 and f[1].isdigit():
                ch.append("chr" + f[0]); ps.append(int(f[1])); pp.append(w)
        out[k] = (np.array(ch), np.array(ps, np.int64), np.array(pp, np.float64))
        save[k + "_chrom"], save[k + "_pos"], save[k + "_pip"] = out[k]
    np.savez_compressed(p, **save)
    return out


def elem_of(reg, chroms, pos0):
    """registry element holding each 0-based position, -1 if none."""
    out = np.full(len(pos0), -1, np.int64)
    for c in np.unique(chroms):
        if c not in reg.by:
            continue
        ix = np.nonzero(chroms == c)[0]; o, st, en, _ = reg.by[c]
        i = np.searchsorted(st, pos0[ix], side="right") - 1; ok = i >= 0
        ok[ok] = en[i[ok]] > pos0[ix][ok]
        out[ix[ok]] = o[i[ok]]
    return out


def dnase_annotations(reg):
    return [b for b in reg.meta if b.get("kind") == "dnase"]


def load_open(b):
    s = np.load(os.path.join(CC, "shards", b["annotation"] + ".npz"))
    return s["idx"][s["cls"] <= OPEN_MAX].astype(np.int64)


def prep():
    reg = e2.Registry(); sl = slims(reg.meta); stud = gwas_studies()
    chosen = {}
    for k, lab, pat, cat in TRAITS + DESC_TRAITS:
        r = choose(stud, pat)
        chosen[k] = dict(label=lab, expected=cat, studyId=r["studyId"], trait=r["traitFromSource"], credible_sets=r["n_cs"],
                         n=r["nSamples"], project=r["projectId"], gated=k in {t[0] for t in TRAITS})
        print(k, chosen[k], flush=True)
    tv = trait_variants(chosen)
    for k in chosen:
        chosen[k]["variants"] = int(len(tv[k][0])); chosen[k]["pip_total"] = round(float(tv[k][2].sum()), 1)
    ann = dnase_annotations(reg); cat = collections.Counter(c for b in ann for c in categories(b["biosample"], sl))
    json.dump(dict(chosen=chosen, dnase_annotations=len(ann), category_annotations=dict(cat)),
              open(os.path.join(OUTD, "studies.json"), "w"), indent=1)
    print("annotations per category:", dict(cat), flush=True)


# ------------------------------------------------------------------------------------------------ V1
def poisson_binomial_tail(ps, k):
    d = np.zeros(len(ps) + 1); d[0] = 1.0
    for p in ps:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= (1 - p)
    return float(d[k:].sum())


def v1():
    lock = os.path.join(OUTD, "v1_lock.json")
    if os.path.exists(lock):
        print("V1 already scored; refusing."); return
    t0 = time.time()
    st = json.load(open(os.path.join(OUTD, "studies.json"))); chosen = st["chosen"]; keys = list(chosen)
    reg = e2.Registry(); sl = slims(reg.meta); ann = dnase_annotations(reg); N = len(reg.chrom)
    tv = trait_variants(chosen); T = np.zeros((len(keys), N))
    for i, k in enumerate(keys):
        ch, ps, pp = tv[k]; e = elem_of(reg, ch, ps - 1); ok = e >= 0
        T[i] = np.bincount(e[ok], weights=pp[ok], minlength=N)
    cnt = np.zeros(N, np.int32)
    for b in ann:
        cnt[load_open(b)] += 1
    br = cnt / len(ann); print(f"breadth over {len(ann)} annotations ({time.time() - t0:.0f}s)", flush=True)
    Sbr, Sbr2, St, Sbrt, Stt = br.sum(), (br * br).sum(), T.sum(1), T @ br, (T * T).sum(1)
    Z = np.zeros((len(keys), len(ann)))
    for j, b in enumerate(ann):
        a = load_open(b); n = len(a); sb = br[a].sum(); stb = T[:, a].sum(1)
        inv = np.linalg.inv(np.array([[N, Sbr, n], [Sbr, Sbr2, sb], [n, sb, n]], np.float64))
        for i in range(len(keys)):
            xty = np.array([St[i], Sbrt[i], stb[i]]); beta = inv @ xty
            s2 = max(Stt[i] - beta @ xty, 1e-300) / (N - 3); Z[i, j] = beta[2] / math.sqrt(s2 * inv[2, 2])
    print(f"scores done ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(CACHE, "v1_scores.npz"), Z=Z, keys=np.array(keys), ann=np.array([b["annotation"] for b in ann]))
    cats = [categories(b["biosample"], sl) for b in ann]
    share = {c: float(np.mean([c in x for x in cats])) for c in {t[3] for t in TRAITS + DESC_TRAITS if t[3]}}
    res, hits, ps = {}, 0, []
    for i, k in enumerate(keys):
        o = np.argsort(-Z[i]); exp = chosen[k]["expected"]
        top = [dict(annotation=ann[j]["annotation"], biosample=ann[j]["biosample"], description=ann[j].get("description", ""),
                    t=round(float(Z[i, j]), 2), categories=sorted(cats[j])) for j in o[:5]]
        hit = bool(exp and exp in cats[o[0]])
        first = next((r + 1 for r, j in enumerate(o) if exp and exp in cats[j]), None)
        res[k] = dict(label=chosen[k]["label"], study=chosen[k]["studyId"], expected=exp, hit=hit, first_expected_rank=first,
                      top=top, gated=chosen[k]["gated"], variants_in_elements=int((T[i] > 0).sum()),
                      pip_in_elements=round(float(T[i].sum()), 1), pip_total=chosen[k]["pip_total"])
        if chosen[k]["gated"]:
            hits += hit; ps.append(share[exp])
    p = poisson_binomial_tail(ps, hits)
    verdict = "PASS" if hits >= 6 and p < 0.05 else "FAIL"
    out = dict(hits=hits, of=len(ps), chance_p=p, chance_expected=float(sum(ps)), verdict=verdict, category_share=share,
               traits=res, annotations=len(ann), scored=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(out, open(lock, "w"), indent=1)
    L = ["=" * 97, "TRAIT2CELL -- use 4: disease variants -> cell types (Open Targets credible sets x ENCODE open chromatin)", "=" * 97,
         f"  V1 PRIMARY (known answers): {hits} of {len(ps)} traits' top biosample in the expected category "
         f"(chance expectation {sum(ps):.2f}, p = {p:.2g}) -> {verdict}"]
    for k in keys:
        r = res[k]; t = r["top"][0]
        L.append(f"  {'*' if r['gated'] else ' '} {r['label']:28s} -> {t['biosample'][:44]:44s} t {t['t']:6.1f}  expected {str(r['expected']):9s}"
                 f" {'HIT ' if r['hit'] else 'miss'} (first expected at rank {r['first_expected_rank']})")
        L.append(f"      next: " + "; ".join(x["biosample"][:30] for x in r["top"][1:5]))
    L.append("  (* gated; the others are descriptive. Score = t statistic of 'open in this biosample' for the trait's PIP, "
             "controlling for how many biosamples the element is open in.)")
    open(RES, "w").write("\n".join(L) + "\n"); print("\n".join(L))


# ------------------------------------------------------------------------------------------------ V2 + chain
def gtex(tissue):
    import pyarrow.parquet as pq
    d = os.path.join(CACHE, "gtex"); os.makedirs(d, exist_ok=True); p = os.path.join(d, tissue + ".parquet")
    if not os.path.exists(p):
        name = f"GTEx_Analysis_v11_eQTL/{tissue}.v11.eQTLs.SuSiE_summary.parquet"
        with tarfile.open(GTX) as tf:
            src = tf.extractfile(tf.getmember(name)); open(p + ".part", "wb").write(src.read())
        os.replace(p + ".part", p)
    t = pq.read_table(p, columns=["phenotype_id", "variant_id", "pip"]).to_pandas()
    return t[t["pip"] >= 0.9]


class Genes:
    def __init__(self, gcol):
        tss = json.load(open(os.path.join(AG, "raw", "gencode46_tss.json")))
        self.by = {}
        for gid, name, gt, ch, pos in tss:
            if gid in gcol:
                self.by.setdefault(ch, []).append((pos, gid, name))
        for ch in self.by:
            v = sorted(self.by[ch]); self.by[ch] = (np.array([x[0] for x in v], np.int64), [x[1] for x in v], [x[2] for x in v])
        self.sym = {g: n for ch in self.by for g, n in zip(self.by[ch][1], self.by[ch][2])}

    def near(self, ch, pos, w=1_000_000):
        if ch not in self.by:
            return np.array([], np.int64), []
        t, g, _ = self.by[ch]; lo, hi = np.searchsorted(t, pos - w), np.searchsorted(t, pos + w, side="right")
        return t[lo:hi], g[lo:hi]


def max_activity(reg, terms):
    a = np.zeros(len(reg.chrom), np.float32)
    for t in terms:
        a = np.maximum(a, reg.activity(t).astype(np.float32))
    return a


class Model:
    """e2gmap `final` (corrected co-activity, DEV fit) scoring (element, candidate genes) in one biosample."""

    def __init__(self, reg, elements):
        dd, Fd = e2._prep(e2.DEVF, "dev_c"); self.m = e2._fit(Fd, dd["Regulated"].astype(bool).to_numpy())
        terms, E, gcol, A = e2.coactivity_inputs(reg)
        self.el = {int(k): i for i, k in enumerate(sorted(set(int(x) for x in elements)))}
        S = A[:, sorted(self.el)].astype(np.float64); del A
        S -= S.mean(0, keepdims=True); S /= np.linalg.norm(S, axis=0, keepdims=True) + 1e-9
        self.An, self.Ez, self.gcol, self.reg, self.terms = S, e2._z(E), gcol, reg, terms

    def score(self, k, tss, gids, act, denom):
        reg = self.reg; ch = reg.chrom[k]; o, st, en, md = reg.by[ch]; a = float(act[k])
        dc = e2.dist_contact(reg.mid[k] - tss); D = np.empty(len(tss))
        for j, (t, g) in enumerate(zip(tss, gids)):
            if g not in denom:
                lo, hi = np.searchsorted(md, t - 5_000_000), np.searchsorted(md, t + 5_000_000); k5 = o[lo:hi]
                denom[g] = float((act[k5].astype(np.float64) * e2.dist_contact(reg.mid[k5] - t)).sum()) + 1e-12
            D[j] = denom[g]
        abc = a * dc / (D + a * dc); ca = self.An[:, self.el[int(k)]] @ self.Ez[:, [self.gcol[g] for g in gids]]
        X = np.c_[np.log(abc + 1e-6), np.log(dc), np.full(len(tss), a), ca, ca * a]
        return self.m.predict_proba(X)[:, 1]


def _h(s):
    return hashlib.sha256(("t2c-v2:" + s).encode()).hexdigest()


def genes():
    lock = os.path.join(OUTD, "v2_lock.json")
    if os.path.exists(lock):
        print("V2 already scored; refusing."); return
    from scipy.stats import binomtest
    t0 = time.time(); reg = e2.Registry()
    rna = np.load(os.path.join(AG, "bulk", "encode_rna_human.npz")); gcol = {g: i for i, g in enumerate(rna["cols"].tolist())}; del rna
    G = Genes(gcol)
    # ---- V2 items
    used, items, dropped, per_t = set(), [], collections.Counter(), {}
    for tissue, terms in TISSUES:
        act = max_activity(reg, terms); df = gtex(tissue)
        truth = collections.defaultdict(set)
        for ph, v in zip(df["phenotype_id"], df["variant_id"]):
            truth[v].add(ph.split(".")[0])
        vids = [v for v in truth if v not in used]
        f = [v.split("_") for v in vids]
        keep = [i for i, x in enumerate(f) if len(x) >= 4 and len(x[2]) == 1 and len(x[3]) == 1 and x[1].isdigit()]
        vids = [vids[i] for i in keep]; ch = np.array([f[i][0] for i in keep]); pos = np.array([int(f[i][1]) for i in keep], np.int64)
        el = elem_of(reg, ch, pos - 1); cand = []
        for v, c, p, k in zip(vids, ch, pos, el):
            if k < 0 or act[k] <= 0:
                dropped["not in an open element"] += 1; continue
            tss, gids = G.near(c, p)
            if not set(gids) & truth[v]:
                dropped["true gene not a candidate"] += 1; continue
            cand.append((_h(v), v, c, int(p), int(k), tss, gids, truth[v] & set(gids)))
        cand.sort(key=lambda x: x[0]); cand = cand[:2000]
        for x in cand:
            used.add(x[1]); items.append((tissue,) + x[1:])
        per_t[tissue] = len(cand); print(f"V2 items {tissue}: {len(cand)} ({time.time() - t0:.0f}s)", flush=True)
    # ---- chain items (needs V1)
    v1l = os.path.join(OUTD, "v1_lock.json"); chain_in = {}
    if os.path.exists(v1l):
        V1 = json.load(open(v1l)); chosen = json.load(open(os.path.join(OUTD, "studies.json")))["chosen"]; tv = trait_variants(chosen)
        for k, r in V1["traits"].items():
            term = r["top"][0]["biosample"]; act = max_activity(reg, [term]); ch, ps, pp = tv[k]
            m = pp >= 0.1; ch, ps, pp = ch[m], ps[m], pp[m]; el = elem_of(reg, ch, ps - 1)
            ok = (el >= 0) & (act[np.maximum(el, 0)] > 0)
            chain_in[k] = (term, ch[ok], ps[ok], pp[ok], el[ok])
    elements = [x[4] for x in items] + [int(e) for v in chain_in.values() for e in v[4]]
    M = Model(reg, elements); print(f"model ready, {len(M.el)} elements, {len(M.terms)} co-activity terms ({time.time() - t0:.0f}s)", flush=True)
    # ---- V2 scoring
    rows, cur, act, denom = [], None, None, {}
    for tissue, v, c, p, k, tss, gids, tru in items:
        if tissue != cur:
            cur = tissue; act = max_activity(reg, dict(TISSUES)[tissue]); denom = {}
        s = M.score(k, tss, gids, act, denom); o = np.argsort(-s)
        nearest = gids[int(np.argmin(np.abs(tss - p)))]
        rows.append(dict(tissue=tissue, variant=v, model=gids[o[0]] in tru, nearest=nearest in tru,
                         top3=any(gids[j] in tru for j in o[:3]), distal=bool(np.min(np.abs(tss - p)) > 10_000), n_cand=len(gids)))
    mr = np.array([r["model"] for r in rows]); nr = np.array([r["nearest"] for r in rows])
    b, c_ = int((mr & ~nr).sum()), int((~mr & nr).sum()); pval = float(binomtest(b, b + c_, 0.5).pvalue) if b + c_ else 1.0
    verdict = "BETTER" if mr.mean() > nr.mean() and pval < 0.05 else ("WORSE" if mr.mean() < nr.mean() and pval < 0.05 else "NO DIFFERENCE")
    dist = np.array([r["distal"] for r in rows]); t3 = np.array([r["top3"] for r in rows])
    by_t = {t: dict(n=int(sum(r["tissue"] == t for r in rows)), model=float(np.mean([r["model"] for r in rows if r["tissue"] == t])),
                    nearest=float(np.mean([r["nearest"] for r in rows if r["tissue"] == t]))) for t in per_t if per_t[t]}
    V2 = dict(items=len(rows), model_acc=float(mr.mean()), nearest_acc=float(nr.mean()), model_only=b, nearest_only=c_, mcnemar_p=pval,
              verdict=verdict, top3_acc=float(t3.mean()), distal=dict(n=int(dist.sum()), model=float(mr[dist].mean()), nearest=float(nr[dist].mean())),
              mean_candidates=float(np.mean([r["n_cand"] for r in rows])), dropped=dict(dropped), by_tissue=by_t, scored=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(V2, open(lock, "w"), indent=1)
    L = ["", f"  V2 VARIANT -> GENE (GTEx v11 fine-mapped eQTLs, PIP >= 0.9, in open elements; {len(rows):,} variants, {len(by_t)} tissues, "
         f"{V2['mean_candidates']:.0f} candidate genes each)",
         f"     model {mr.mean():.1%} right vs nearest gene {nr.mean():.1%}; model-only {b}, nearest-only {c_}, McNemar p = {pval:.2g} -> {verdict}",
         f"     top-3 {t3.mean():.1%}; variants > 10 kb from every TSS ({dist.sum():,}): model {mr[dist].mean():.1%} vs nearest {nr[dist].mean():.1%}"]
    # ---- chain
    if chain_in:
        tg = json.load(open(os.path.join(DRUGL, "targets.json"))); dr = json.load(open(os.path.join(DRUGL, "drugs.json")))
        CH = {}
        L.append(""); L.append("  CHAIN (descriptive): trait -> top cell type -> genes (sum of PIP x model score) -> drugs on those genes")
        for k, (term, ch, ps, pp, el) in chain_in.items():
            act = max_activity(reg, [term]); denom = {}; W = collections.defaultdict(float)
            for c, p, w, e in zip(ch, ps, pp, el):
                tss, gids = G.near(c, int(p))
                if not gids:
                    continue
                s = M.score(int(e), tss, gids, act, denom)
                for g, x in zip(gids, s):
                    W[g] += w * x
            rank = sorted(W, key=lambda g: -W[g]); top = []
            for g in rank[:15]:
                ds = sorted((tg.get(g) or {}).get("drugs", []), key=lambda d: -STAGE.get(d["stage"], 0))
                names = []
                for d in ds:
                    nm = (dr.get(d["chembl"]) or {}).get("name") or d["chembl"]
                    if nm not in names:
                        names.append(nm)
                top.append(dict(gene=G.sym.get(g, g), ensembl=g, weight=round(W[g], 3), drugs=len(ds),
                                best_stage=ds[0]["stage"] if ds else None, examples=names[:3]))
            CH[k] = dict(cell_type=term, variants=int(len(el)), genes=top)
            if k == "ldl":
                pos_ = {G.sym.get(g, g): i + 1 for i, g in enumerate(rank)}
                CH[k]["known_target_ranks"] = {s_: pos_.get(s_) for s_ in LDL_KNOWN}
            L.append(f"   {V1['traits'][k]['label']} -> {term}: " + ", ".join(
                f"{x['gene']}" + (f" [{x['examples'][0].lower()}, {x['best_stage'].lower()}]" if x["drugs"] else "") for x in top[:10]))
        if "ldl" in CH:
            L.append("   LDL known drug targets, rank among scored genes: " + ", ".join(f"{s_} {r_}" for s_, r_ in CH["ldl"]["known_target_ranks"].items()))
        json.dump(CH, open(os.path.join(OUTD, "chain.json"), "w"), indent=1)
    open(RES, "a").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"prep": prep, "v1": v1, "genes": genes}[sys.argv[1]]()
