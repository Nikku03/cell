"""Test the original integral vs the factored version on real E. coli data.

ORIGINAL (one integral, W and C inside):
  ΔG_bind = -∫ (Φ_TF(x) · Ψ_DNA(x)) · W(x) · C(x) dx
  rank by score s_A = -ΔG (larger = better target)

FACTORED (three layers; my version):
  ΔG_aff(x) = -∫ (Φ_TF(x) · Ψ_DNA(x)) dx        # sequence only
  occupancy = σ(β([TF] - ΔG_aff)) · C(x)        # abundance + accessibility upstairs
  score s_B = max_x occupancy(x) over the upstream window
  (W enters NOT as a spatial integral term but as a scalar [TF] abundance)

Test: for each of N TFs with known RegulonDB targets, extract 250 bp upstream of
each TARGET vs an equal number of RANDOM non-target genes. Score every upstream
window with both equations. Report AUC(target vs non-target) per TF and overall.

Components (all from data we have):
  Ψ_DNA(x) -- continuous biochemistry track, 3 channels:
    minor groove width (DNAshape, Rohs '09 pentamer averages, simplified)
    electrostatic potential (proxy: AT-richness drives more negative potential)
    purine/pyrimidine asymmetry (proxy for major-groove H-bond pattern)
  Φ_TF(x) -- family-specific consensus rendered as a continuous field, anchored
    to TF family detected from product annotation. Per-family templates from the
    motif table we already built (CRP TGTGA-N6-TCACA, LysR T-N11-A palindrome,
    TetR short palindrome, OmpR direct repeat, etc.)
  W(x) -- exp(-|x - x_TF|/lambda); we sweep lambda. For testing on the WHOLE
    upstream region of all targets, W is computed from chromosomal positions.
  C(x) -- 1 (no accessibility data; both equations get C=1, so it factors out.
    This is the honest move: C is a placeholder in both formulations).

We will see whether putting W INSIDE the integral (original) helps or hurts
relative to the factored version (W as scalar abundance term outside).
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")

# ---- genome ----
seq=[]
for l in open("data/external_data/ecoli_genome/ecoli_K12.fna"):
    if not l.startswith(">"): seq.append(l.strip())
GENOME="".join(seq).upper()
GLEN=len(GENOME)
print(f"genome: {GLEN} bp")

def revcomp(s):
    t={"A":"T","T":"A","G":"C","C":"G","N":"N"}
    return "".join(t.get(c,"N") for c in reversed(s))

# ---- gene coords ----
genes=[]   # bnum, name, start, end, strand
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes.append((p[2], p[1], int(p[3]), int(p[4]), p[5]))
name2g={g[1].lower():g for g in genes}
print(f"genes: {len(genes)}")

# RegulonDB truth
tf_targets=defaultdict(set); tf_outdeg=defaultdict(int)
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    tf,tgt=p[0].lower(), p[1].lower()
    tf_targets[tf].add(tgt); tf_outdeg[tf]+=1
print(f"TFs with truth: {len(tf_targets)}")

# ---- (1) Ψ_DNA(x): continuous biochemistry track per base ----
# Minor-groove width: A-tracts are narrow; use a simplified pentamer-style
# dinucleotide table from Rohs '09 (very simplified — full DNAshape lookup
# uses pentamers, we collapse to dinucleotide averages):
MGW_DI = {  # angstroms, approximate
 "AA":4.5,"AT":4.6,"TA":5.2,"TT":4.5,
 "AC":5.0,"AG":5.0,"CA":5.2,"CT":5.0,
 "GA":5.0,"GC":5.5,"CG":5.7,"GG":5.0,
 "TG":5.2,"TC":5.0,"CC":5.0,"GT":5.0,
}
# Electrostatic potential: A/T tracts are more negative (narrow minor groove)
ESP_BASE = {"A":-1.0,"T":-1.0,"G":-0.3,"C":-0.3,"N":0.0}
# Purine asymmetry
PUR_BASE = {"A":+1,"G":+1,"T":-1,"C":-1,"N":0}

def psi_field(s):
    """Return Ψ(x), shape (L,3): channels = MGW, ESP, PUR."""
    L=len(s); F=np.zeros((L,3))
    for i in range(L):
        F[i,1]=ESP_BASE.get(s[i],0)
        F[i,2]=PUR_BASE.get(s[i],0)
        if i<L-1:
            di=s[i:i+2]
            F[i,0]=MGW_DI.get(di, 5.0)
    F[L-1,0]=F[L-2,0] if L>1 else 5.0
    # center channel 0 (MGW) around its mean -> deviation field
    F[:,0] = F[:,0] - 5.0
    return F

# ---- (2) Φ_TF: per-family consensus rendered as a field ----
# A consensus is encoded as a TARGET (Φ) field; alignment with Ψ_DNA via inner
# product = high when DNA at that position matches what the TF wants.
# Channels: (MGW_pref, ESP_pref, PUR_pref) — symmetric to Ψ.
# Per-family templates:
FAMILIES = {
 # CRP/FNR: TGTGA-N6-TCACA inverted repeat over ~22 bp
 "crp": dict(length=22, sym="inv", core_left="TGTGA", core_right="TCACA"),
 "fnr": dict(length=22, sym="inv", core_left="TTGAT", core_right="ATCAA"),
 # LysR-type: T-N11-A palindrome ~15 bp
 "lysr": dict(length=15, sym="inv", core_left="TATAC", core_right="GTATA"),
 # TetR ~15 bp palindrome
 "tetr": dict(length=15, sym="inv", core_left="TCCCT", core_right="AGGGA"),
 # MarR AT-rich palindrome
 "marr": dict(length=18, sym="inv", core_left="TTGTC", core_right="GACAA"),
 # OmpR/PhoB direct repeat
 "ompr": dict(length=20, sym="dir", core_left="CTGTC", core_right="CTGTC"),
 # GntR family palindrome
 "gntr": dict(length=15, sym="inv", core_left="GTGTA", core_right="TACAC"),
 # LacI palindrome
 "laci": dict(length=21, sym="inv", core_left="AATTG", core_right="CAATT"),
 # generic palindrome fallback
 "any":  dict(length=18, sym="inv", core_left="TTGAC", core_right="GTCAA"),
}

def consensus_to_field(motif, length):
    """Render a sequence as a Φ field by treating it as the IDEAL Ψ."""
    return psi_field(motif)

def phi_template(family):
    f=FAMILIES.get(family, FAMILIES["any"])
    L=f["length"]; left=f["core_left"]; right=f["core_right"]
    # build a synthetic ideal DNA: left core - spacer - right core
    spacer="N"*(L - len(left) - len(right))
    motif=left + spacer + right
    return psi_field(motif[:L]), L

# pre-build templates
TEMPLATES={fam: phi_template(fam) for fam in FAMILIES}

# ---- (3) infer family for each TF from its annotation ----
desc={}
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=7: desc[p[1].lower()]=p[6].lower()

def tf_family_of(tf):
    d=desc.get(tf,"")
    for fam_key in ("crp","fnr","lysr","tetr","marr","ompr","gntr","laci"):
        kw={"crp":"crp","fnr":"fnr","lysr":"lysr","tetr":"tetr","marr":"marr",
            "ompr":"ompr","gntr":"gntr","laci":"laci"}[fam_key]
        if kw in d.replace("-",""): return fam_key
    # known global regs
    if tf in ("crp","fnr","fis","ihf","hns","arca","lrp"): return "crp"
    return "any"

# ---- (4) score a window with both equations ----
def upstream_window(g, pad=250):
    bnum,name,start,end,strand=g
    if strand=="forward":
        a=max(0, start-pad-1); b=start-1
        s=GENOME[a:b]
    else:
        a=end; b=min(GLEN, end+pad)
        s=revcomp(GENOME[a:b])
    return s, (a,b)

def affinity_field(psi, phi, mode="conv"):
    """Sliding inner product (cross-correlation) of psi with phi.
    Returns array of scores for each window position."""
    L = phi.shape[0]
    if psi.shape[0] < L: return np.array([0.0])
    # normalize by channel scale
    out = np.zeros(psi.shape[0]-L+1)
    for i in range(out.shape[0]):
        out[i] = float(np.sum(psi[i:i+L] * phi))
    return out

def wave_W(positions, tf_pos, lam):
    """W(x) = exp(-|x - x_TF|/lambda); positions are bp on chromosome."""
    return np.exp(-np.abs(np.array(positions) - tf_pos) / lam)

def score_original(window_seq, window_chrom_start, phi_tpl, tf_pos, lam, C=1.0):
    """Original eq: integrate (Φ·Ψ) · W · C dx, return max."""
    psi=psi_field(window_seq)
    raw=affinity_field(psi, phi_tpl[0])
    if len(raw)==0: return 0.0
    # W per position (within the window)
    L=phi_tpl[1]
    pos_x = np.arange(len(raw)) + window_chrom_start
    W = wave_W(pos_x, tf_pos, lam)
    integrand = raw * W * C
    return float(np.max(integrand))

def score_factored(window_seq, phi_tpl, tf_abundance, beta=0.5, C=1.0):
    """Factored: affinity (sequence only), then occupancy with TF abundance."""
    psi=psi_field(window_seq)
    raw=affinity_field(psi, phi_tpl[0])
    if len(raw)==0: return 0.0
    dG_aff = -raw   # affinity score: higher inner-product = lower ΔG
    # occupancy with scalar abundance
    occ = 1.0/(1.0+np.exp(-beta*(tf_abundance - dG_aff))) * C
    return float(np.max(occ))

# ---- (5) build the test set ----
# Use TFs with: coords + >=10 RegulonDB targets that have coords
testable=[]
for tf, tgts in tf_targets.items():
    if tf not in name2g: continue
    coord_tgts=[t for t in tgts if t in name2g]
    if len(coord_tgts) >= 10:
        testable.append((tf, coord_tgts))
print(f"\ntestable TFs (>=10 coord-mapped targets): {len(testable)}")

# TF abundance proxy: known global hubs get high abundance
ABUNDANCE = {"crp":7.0,"fnr":6.0,"fis":7.5,"ihf":7.0,"hns":8.0,"arca":6.0,
             "lrp":6.0,"fur":6.0,"narl":4.5,"narp":4.5}
def abundance(tf):
    return ABUNDANCE.get(tf, 2.0)  # local TFs default to low ~10 copies

def auc(s_pos, s_neg):
    """AUC for ranking positives above negatives."""
    if not s_pos or not s_neg: return 0.5
    s=np.concatenate([s_pos, s_neg])
    y=np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s))
    pos=y==1
    return float((r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum()))

# ---- (6) RUN the test ----
np.random.seed(7)
results=[]
all_gene_names=[g[1].lower() for g in genes]
for tf, coord_tgts in testable[:60]:
    fam = tf_family_of(tf)
    phi_tpl = TEMPLATES[fam]
    tf_g = name2g[tf]
    tf_pos = (tf_g[2]+tf_g[3])//2
    abund = abundance(tf)
    # positives: real targets
    pos_seqs=[]
    pos_chrom=[]
    for t in coord_tgts:
        g=name2g[t]
        s,(a,b) = upstream_window(g)
        pos_seqs.append((s, a))
    # negatives: random genes not in target set
    non_targets = [n for n in all_gene_names if n not in tf_targets[tf] and n != tf]
    neg_picks = list(np.random.choice(non_targets, size=min(len(pos_seqs)*3, len(non_targets)), replace=False))
    neg_seqs=[]
    for t in neg_picks:
        g=name2g[t]
        s,(a,b)=upstream_window(g)
        neg_seqs.append((s,a))

    # try multiple λ for the wave-field; pick BEST for original (charitable test)
    best_A=0.5; best_lam=None
    for lam in (5e3, 5e4, 5e5, 5e6):
        sA_pos=[score_original(s, a, phi_tpl, tf_pos, lam) for s,a in pos_seqs]
        sA_neg=[score_original(s, a, phi_tpl, tf_pos, lam) for s,a in neg_seqs]
        a_ = auc(sA_pos, sA_neg)
        if a_>best_A: best_A=a_; best_lam=lam
    # factored
    sB_pos=[score_factored(s, phi_tpl, abund) for s,_ in pos_seqs]
    sB_neg=[score_factored(s, phi_tpl, abund) for s,_ in neg_seqs]
    auc_B = auc(sB_pos, sB_neg)

    # also a "sequence-only" affinity baseline (no abundance, no W) -- pure Φ·Ψ
    sR_pos=[float(np.max(affinity_field(psi_field(s), phi_tpl[0]))) for s,_ in pos_seqs]
    sR_neg=[float(np.max(affinity_field(psi_field(s), phi_tpl[0]))) for s,_ in neg_seqs]
    auc_R = auc(sR_pos, sR_neg)

    results.append(dict(tf=tf, family=fam, n_targets=len(pos_seqs),
                        auc_original_best=round(best_A,3), best_lambda=best_lam,
                        auc_factored=round(auc_B,3),
                        auc_seq_only=round(auc_R,3),
                        is_global=tf in ("crp","fnr","fis","ihf","hns","arca","lrp","fur","narl","cpxr","cra")))

results.sort(key=lambda r:-r["n_targets"])

# aggregate
import statistics
def mean(xs): return round(statistics.mean(xs),3) if xs else float("nan")
all_A=[r["auc_original_best"] for r in results]
all_B=[r["auc_factored"] for r in results]
all_R=[r["auc_seq_only"] for r in results]
glob=[r for r in results if r["is_global"]]
local=[r for r in results if not r["is_global"]]

print(f"\n=== AUC (target vs random non-target), N={len(results)} TFs ===")
print(f"{'TF':<10}{'fam':<8}{'n':>4}{'global?':>8}{'orig*':>8}{'fact':>8}{'seq':>8}{'best_λ':>10}")
for r in results[:25]:
    print(f"  {r['tf']:<8}{r['family']:<8}{r['n_targets']:>4}{('Y' if r['is_global'] else 'n'):>8}"
          f"{r['auc_original_best']:>8.3f}{r['auc_factored']:>8.3f}{r['auc_seq_only']:>8.3f}"
          f"{(str(r['best_lambda']) if r['best_lambda'] else '—'):>10}")
print(f"\nMEAN AUC: original(best λ)={mean(all_A)}  factored={mean(all_B)}  seq-only={mean(all_R)}")
print(f"   GLOBAL TFs (n={len(glob)}): orig={mean([r['auc_original_best'] for r in glob])} "
      f"fact={mean([r['auc_factored'] for r in glob])} seq={mean([r['auc_seq_only'] for r in glob])}")
print(f"   LOCAL TFs  (n={len(local)}): orig={mean([r['auc_original_best'] for r in local])} "
      f"fact={mean([r['auc_factored'] for r in local])} seq={mean([r['auc_seq_only'] for r in local])}")

json.dump(dict(n=len(results),
   mean_auc_original=mean(all_A), mean_auc_factored=mean(all_B), mean_auc_seq_only=mean(all_R),
   global_mean=dict(orig=mean([r['auc_original_best'] for r in glob]),
                    fact=mean([r['auc_factored'] for r in glob]),
                    seq=mean([r['auc_seq_only'] for r in glob])),
   local_mean=dict(orig=mean([r['auc_original_best'] for r in local]),
                   fact=mean([r['auc_factored'] for r in local]),
                   seq=mean([r['auc_seq_only'] for r in local])),
   per_tf=results), open(OUT/"binding_field_test.json","w"), indent=2)
print(f"\nwrote outputs/orphan/binding_field_test.json")
