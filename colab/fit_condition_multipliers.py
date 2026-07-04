"""FIT CONDITION MULTIPLIERS (#1) — replace the crude flat direction x3 with DATA-DRIVEN per-gene
fold-changes measured from ARCHS4 (the human RNA-seq compendium, ~700k GEO samples).

The E. coli method:  beta(gene, condition) = intrinsic beta(promoter) x SUM master-program activities.
The "master-program activity" magnitude for a human condition is exactly the fold-change of a gene's
expression between condition (case) and baseline (control) samples. We measure it directly:

  for each condition C (same names as compute_conditions.CONDITIONS):
    case    = ARCHS4 samples whose metadata matches C's keywords
    control = samples matching untreated/normal/control (and NOT matching C)
    mult(gene, C) = median_CPM(case) / median_CPM(control)      (clipped to [0.1, 10])

-> condition_multipliers.json  {condition: {gene: multiplier}}   (consumed by compute_concentration.py #1)

Heavy (ARCHS4 h5 ~ tens of GB) -> RUN ON COLAB. Reads only the case/control sample columns it needs,
never the full matrix. Skips gracefully (writes nothing) if the ARCHS4 file is absent.
"""
import json, os, re
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
CLIP_LO,CLIP_HI=0.1,10.0
MIN_CASE=6; MIN_CTRL=6           # need at least this many samples on each side to trust a condition
MAX_SAMPLES=1500                 # cap case/control sample reads per condition (memory over Drive FUSE)

# condition -> positive metadata keywords (must match the CONDITIONS keys in compute_conditions.py)
COND_KW = {
 "heat / high temperature":      ["heat shock","heat stress","hyperthermia","heat-shock","42c","42 c","45c","thermal stress","elevated temperature"],
 "cold / low temperature":       ["cold shock","cold stress","hypothermia","cold exposure","4c ","low temperature","32c"],
 "acidosis / low pH":            ["acidosis","low ph","acidic ph","lactic acid","acid stress","ph 6","ph6","extracellular acidification"],
 "high pressure / mechanical":   ["mechanical stress","cyclic stretch","shear stress","substrate stiffness","matrix stiffness","hydrostatic pressure","mechanical stretch","compression"],
 "osmotic stress":               ["osmotic stress","hyperosmotic","hypertonic","sorbitol","mannitol","nacl stress","osmolarity","hyperosmolar"],
 "hypoxia / low O2":             ["hypoxia","hypoxic","low oxygen","1% o2","0.5% o2","anoxia","cocl2","dfx","ischemi","oxygen deprivation"],
 "oxidative stress":             ["oxidative stress","hydrogen peroxide","h2o2","paraquat","tbhp","menadione","reactive oxygen","ros stress","diamide"],
 "nutrient starvation":          ["starvation","serum starvation","glucose deprivation","amino acid deprivation","nutrient deprivation","fasting","caloric restriction","serum-free","glucose starvation"],
 "ER stress / unfolded protein": ["er stress","thapsigargin","tunicamycin","unfolded protein","brefeldin","dtt treatment","endoplasmic reticulum stress"],
 "DNA damage":                   ["dna damage","irradiation","ionizing radiation","uv exposure","uv irradiation","doxorubicin","etoposide","cisplatin","genotoxic","gamma radiation","bleomycin"],
 "inflammation / immune":        ["lps","lipopolysaccharide","tnf","tnf-alpha","interferon","ifn-gamma","il-1","il1b","inflammation","inflammatory","poly(i:c)","cytokine stimulation"],
 "heavy metal / xenobiotic":     ["cadmium","zinc treatment","arsenic","heavy metal","dioxin","tcdd","benzo(a)pyrene","xenobiotic","copper stress","mercury"],
}
CTRL_KW=["control","untreated","normal","wild type","wild-type","wildtype","healthy","dmso","vehicle","mock","unstimulated","baseline","non-treated","untreated control"]
NEG_STRESS=[kw for kws in COND_KW.values() for kw in kws]   # a control must match none of these

def find_h5():
    # IDENTICAL resolution to compute_coexpr._find_h5 so #1 reuses the exact same in-place 59 GB ARCHS4
    # file the co-expression lens already reads (any divergence = #1 silently skips while coexpr succeeds).
    for env in ("ARCHS4_H5","COEXPR_H5"):
        v=os.environ.get(env)
        if v and Path(v).exists(): return Path(v)
    names=["archs4_human_gene.h5","archs4_gene_human.h5","human_gene_v2.h5","archs4_human.h5","human_gene_v2.latest.h5"]
    dirs=[H, Path("/content/drive/MyDrive/virtual_cell_data/expression_geo"),
          Path("/content/drive/MyDrive/virtual_cell_data/human_raw")]
    for d in dirs:
        for n in names:
            p=d/n
            if p.exists() and p.stat().st_size>1e7: return p
        if d.exists():
            hits=sorted(d.glob("archs4*.h5"))+sorted(d.glob("*human*gene*.h5"))
            hits=[p for p in hits if p.stat().st_size>1e7]
            if hits: return hits[0]
    return None

def load_meta(f):
    """returns (symbols[np.str], sample_text[list[str] lowercased], expr_dataset, genes_axis 0|1)."""
    import h5py
    h=h5py.File(f,"r")
    def dec(a): return [x.decode() if isinstance(x,bytes) else str(x) for x in a]
    sym=None
    for k in ("meta/genes/symbol","meta/genes/genes","meta/genes/gene_symbol"):
        if k in h: sym=np.array(dec(h[k][:])); break
    if sym is None: raise RuntimeError("no gene symbols in h5")
    # concatenate the informative sample metadata fields
    fields=[]
    for k in ("meta/samples/title","meta/samples/characteristics_ch1","meta/samples/source_name_ch1",
              "meta/samples/series_id","meta/samples/geo_accession"):
        if k in h: fields.append(dec(h[k][:]))
    n=len(fields[0]); txt=[" ".join(fields[j][i] for j in range(len(fields))).lower() for i in range(n)]
    expr=h["data/expression"] if "data/expression" in h else h["expression"]
    genes_axis=0 if expr.shape[0]==len(sym) else 1
    return sym, txt, expr, genes_axis, h

def match_idx(txt, kws, exclude=None):
    ex=exclude or []
    hit=[]
    for i,t in enumerate(txt):
        if any(k in t for k in kws) and not any(e in t for e in ex): hit.append(i)
    return hit

def cpm_median(expr, genes_axis, sample_idx):
    """median CPM per gene across the given samples. Reads only those sample columns."""
    sample_idx=sorted(set(sample_idx))
    if len(sample_idx)>MAX_SAMPLES:
        step=len(sample_idx)/MAX_SAMPLES
        sample_idx=[sample_idx[int(i*step)] for i in range(MAX_SAMPLES)]
    if genes_axis==0:  M=expr[:, sample_idx].astype(np.float64)          # genes x samples
    else:              M=expr[sample_idx, :].astype(np.float64).T        # -> genes x samples
    colsum=M.sum(axis=0); colsum[colsum==0]=1.0
    M=M/colsum*1e6
    return np.median(M, axis=1)

def main():
    # wrapper: any failure (OOM, h5 read error over Drive FUSE, ...) is surfaced to STDOUT and a diagnostic
    # condition_multipliers.json is always written, so the run never leaves this step invisible.
    try:
        _fit()
    except BaseException as e:
        import traceback
        tb=traceback.format_exc()
        print("fit_condition_multipliers FAILED:", repr(e)[:200], flush=True)
        print(tb[-1500:], flush=True)
        try: json.dump({"_report":{"error":repr(e)[:300]}, "_meta":{"status":"failed"}},
                       open(OUT/"condition_multipliers.json","w"))
        except Exception: pass

def _fit():
    f=find_h5()
    if f is None:
        print("no ARCHS4 h5 found (set ARCHS4_H5) -> condition_multipliers.json NOT written; flat COND_FOLD used", flush=True)
        json.dump({"_report":{"status":"no ARCHS4 h5 found"}}, open(OUT/"condition_multipliers.json","w")); return
    import h5py  # noqa
    D=json.load(open(OUT/"cell_complete.json")); model_genes=set(g["name"] for g in D["genes"])
    print(f"fitting condition multipliers from {f} ({f.stat().st_size/1e9:.1f} GB) ...", flush=True)
    sym, txt, expr, ga, h = load_meta(f)
    sym_up=np.array([s.upper() for s in sym])
    keep=np.array([s in model_genes for s in sym_up])                    # only emit model genes
    print(f"  {len(sym)} genes, {len(txt)} samples; {keep.sum()} genes overlap the model")
    ctrl=match_idx(txt, CTRL_KW, exclude=NEG_STRESS)
    print(f"  baseline/control pool: {len(ctrl)} samples (reading control CPM ...)", flush=True)
    out={}; report={}
    ctrl_med=cpm_median(expr, ga, ctrl) if len(ctrl)>=MIN_CTRL else None
    print(f"  control CPM read OK; fitting {len(COND_KW)} conditions ...", flush=True)
    for cond,kws in COND_KW.items():
        case=match_idx(txt, kws)
        if len(case)<MIN_CASE or ctrl_med is None or len(ctrl)<MIN_CTRL:
            report[cond]=dict(n_case=len(case), n_control=len(ctrl), status="insufficient -> flat fallback"); continue
        case_med=cpm_median(expr, ga, case)
        fc=(case_med+1.0)/(ctrl_med+1.0)
        fc=np.clip(fc, CLIP_LO, CLIP_HI)
        cm={}
        for gi in np.where(keep)[0]:
            g=sym_up[gi]; m=float(fc[gi])
            if g in cm: continue                                         # first (h5 symbols can repeat)
            if abs(np.log2(m))>=0.585: cm[g]=round(m,3)                  # keep >=1.5x shifts only (signal, not noise)
        out[cond]=cm
        report[cond]=dict(n_case=len(case), n_control=len(ctrl), n_genes_shifted=len(cm),
                          median_up=round(float(np.median([v for v in cm.values() if v>1] or [0])),2),
                          median_down=round(float(np.median([v for v in cm.values() if v<1] or [0])),2))
        print(f"  {cond:32} case={len(case):5d} ctrl={len(ctrl):5d} -> {len(cm)} genes shifted >=1.5x")
    h.close()
    # flat top level: {condition: {gene: mult}} so compute_concentration's mult.get(cond) works directly;
    # _report/_meta are underscore-prefixed and never collide with a condition name.
    payload=dict(out)
    payload["_report"]=report
    payload["_meta"]=dict(source=f.name, clip=[CLIP_LO,CLIP_HI], min_shift="1.5x",
                          method="median CPM case/control from ARCHS4 metadata keyword match")
    json.dump(payload, open(OUT/"condition_multipliers.json","w"))
    fit=[c for c in out if out[c]]
    print(f"fit {len(fit)}/{len(COND_KW)} conditions from data -> condition_multipliers.json")
    print("  (compute_concentration.py will now use these per-gene fold-changes instead of flat COND_FOLD)")

if __name__=="__main__":
    main()
