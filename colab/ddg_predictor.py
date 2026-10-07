"""ΔΔG stability predictor — keystone node of the mutation→phenotype chain.

Predicts the folding-stability change ΔΔG (kcal/mol) of a single missense mutation. CPU-only, no torch /
no MSA / no protein-LM: 19 biophysical features (amino-acid property deltas + BLOSUM62 + Gly/Pro/hydrophobic
flags + pH/T) plus a structural BURIAL feature (atomic contact number within 10 Å of the mutation site — the
dominant stability determinant), fed to a gradient-boosted regressor trained on S2648 with reverse-mutation
augmentation for thermodynamic consistency.

Validated blind on S669 (≤30% identity to training): see colab/validate_ddg.py -> ddg_validation.json.
Two roles in the cell model:
  1. mutation→phenotype chain: Mutation → structure → ΔΔG → (Δ active fraction [E]) and, in parallel,
     mutant sequence → CatPred → Δ(kcat/Km) → ec-flux → pathway → phenotype.
  2. active-enzyme estimate: a destabilizing ΔΔG lowers the folded/active fraction, i.e. usable [E].

Convention: ΔΔG > 0 = destabilizing (matches ddG_experimental in the S669/S2648 benchmark).
"""
import os, urllib.request, pickle, numpy as np

REPO = "https://raw.githubusercontent.com/PeptoneLtd/proteinmpnn_ddg/main/paper/datasets"
AA = "ACDEFGHIKLMNPQRSTVWY"

KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,
          K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
VOL = dict(A=88.6,R=173.4,N=114.1,D=111.1,C=108.5,Q=143.8,E=138.4,G=60.1,H=153.2,I=166.7,
           L=166.7,K=168.6,M=162.9,F=189.9,P=112.7,S=89.0,T=116.1,W=227.8,Y=193.6,V=140.0)
CHG = dict(A=0,R=1,N=0,D=-1,C=0,Q=0,E=-1,G=0,H=.1,I=0,L=0,K=1,M=0,F=0,P=0,S=0,T=0,W=0,Y=0,V=0)
POL = dict(A=0,R=52,N=3.38,D=49.7,C=1.48,Q=3.53,E=49.9,G=0,H=51.6,I=0.13,L=0.13,K=49.5,
           M=1.43,F=0.35,P=1.58,S=1.67,T=1.66,W=2.1,Y=1.61,V=0.13)
FLEX = dict(A=0.36,R=0.53,N=0.46,D=0.51,C=0.35,Q=0.49,E=0.5,G=0.54,H=0.32,I=0.46,L=0.37,
            K=0.47,M=0.3,F=0.31,P=0.51,S=0.51,T=0.44,W=0.31,Y=0.42,V=0.39)
MW = dict(A=89,R=174,N=132,D=133,C=121,Q=146,E=147,G=75,H=155,I=131,L=131,K=146,M=149,
          F=165,P=115,S=105,T=119,W=204,Y=181,V=117)
try:
    from Bio.Align import substitution_matrices
    _B62 = substitution_matrices.load("BLOSUM62")
    def blosum(a, b): return float(_B62[a, b]) if (a in _B62.alphabet and b in _B62.alphabet) else 0.0
except Exception:
    def blosum(a, b): return 0.0


def biophysical(pre, post, pH=7.0, T=298.0):
    """19 sequence/biophysical features for a WT->mut substitution."""
    return [KD[post]-KD[pre], VOL[post]-VOL[pre], CHG[post]-CHG[pre], POL[post]-POL[pre],
            FLEX[post]-FLEX[pre], MW[post]-MW[pre], blosum(pre, post),
            KD[pre], VOL[pre], CHG[pre], KD[post], VOL[post], CHG[post],
            1.0 if pre in 'GP' else 0.0, 1.0 if post in 'GP' else 0.0,
            1.0 if pre in 'AILMFWVC' else 0.0, 1.0 if post in 'AILMFWVC' else 0.0,
            (pH-7.0), (T-298.0)/50.0]


# ---------- structure: atomic contact number (burial), numpy-only ----------
def parse_pdb(path):
    cb, ca, heavy = {}, {}, []
    for ln in open(path):
        if not ln.startswith('ATOM'): continue
        if (ln[76:78].strip() or ln[13]) == 'H': continue
        atom, ch, res = ln[12:16].strip(), ln[21], ln[22:26].strip()
        try: xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError: continue
        heavy.append(xyz)
        if atom == 'CB': cb[(ch, res)] = xyz
        if atom == 'CA': ca[(ch, res)] = xyz
    for k, v in ca.items(): cb.setdefault(k, v)          # Gly has no CB -> use CA
    return cb, (np.array(heavy) if heavy else np.zeros((0, 3)))


def contact_number(cb, heavy, chain, pos, cutoff=10.0):
    """Heavy atoms within `cutoff` Å of the mutation-site CB (excl. self). ~inverse of solvent accessibility."""
    site = cb.get((chain, str(pos)))
    if site is None or len(heavy) == 0: return None
    d = np.linalg.norm(heavy - np.array(site), axis=1)
    return float(((d < cutoff) & (d > 0.1)).sum())


def features(pre, post, cn, pH=7.0, T=298.0, pmpnn=0.0):
    """biophysical + burial + ProteinMPNN structure log-odds. pmpnn = ProteinMPNN-ddG (structure-based, the
    dominant signal: 0.405 -> 0.472 on S669). 0.0 falls back to biophysical+burial (no structure model)."""
    cn = 0.0 if cn is None else cn
    pmpnn = 0.0 if (pmpnn is None or not np.isfinite(pmpnn)) else pmpnn
    return biophysical(pre, post, pH, T) + [cn, cn*(KD[post]-KD[pre]), cn*abs(KD[post]-KD[pre]), pmpnn]


# ---------- predictor ----------
class DDGPredictor:
    """Load a trained model; predict ΔΔG for a mutation given a structure (PDB)."""
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def predict_from_structure(self, pdb_path, chain, pos, wt, mut, pH=7.0, T=298.0):
        cb, heavy = parse_pdb(pdb_path)
        cn = contact_number(cb, heavy, chain, pos)
        pm = proteinmpnn_logodds(pdb_path, chain, pos, wt, mut) or 0.0   # structure log-odds (0 if unavailable)
        x = np.nan_to_num(np.array([features(wt, mut, cn, pH, T, pm)]))
        return float(self.model.predict(x)[0]), (cn is not None)

    @staticmethod
    def alphafold_pdb(uniprot, cache='pdbs'):
        """Fetch an AlphaFold structure by UniProt accession (resolves the current model version via the API)."""
        os.makedirs(cache, exist_ok=True)
        dst = f'{cache}/AF-{uniprot}.pdb'
        if not os.path.exists(dst):
            import json
            meta = json.load(urllib.request.urlopen(
                f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}", timeout=30))
            urllib.request.urlretrieve(meta[0]["pdbUrl"], dst)
        return dst


# ---------- ProteinMPNN structure log-odds (the dominant ΔΔG feature: 0.405 -> 0.472) ----------
MPNN_AA = 'ACDEFGHIKLMNPQRSTVWYX'
_THREE1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
           'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def proteinmpnn_logodds(pdb_path, chain, pos, wt, mut, mpnn_dir=None):
    """log P(mut) - log P(wt) at the site from ProteinMPNN (structure-conditioned). Requires a ProteinMPNN
    checkout (env PROTEINMPNN_DIR or `mpnn_dir`); returns None if unavailable -> features falls back to
    biophysical+burial. Structure > sequence: this beats an ESM-2 sequence feature (which gave ~0)."""
    import subprocess, tempfile, glob
    mpnn_dir = mpnn_dir or os.environ.get("PROTEINMPNN_DIR")
    if not mpnn_dir or not os.path.isdir(mpnn_dir):
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            import shutil
            os.makedirs(f"{td}/pdb"); shutil.copy(pdb_path, f"{td}/pdb/q.pdb")
            subprocess.run([__import__("sys").executable, f"{mpnn_dir}/helper_scripts/parse_multiple_chains.py",
                            f"--input_path={td}/pdb", f"--output_path={td}/x.jsonl"], check=True, capture_output=True)
            subprocess.run([__import__("sys").executable, f"{mpnn_dir}/protein_mpnn_run.py",
                            "--jsonl_path", f"{td}/x.jsonl", "--out_folder", td,
                            "--unconditional_probs_only", "1", "--model_name", "v_48_020", "--batch_size", "1"],
                           check=True, capture_output=True)
            f = glob.glob(f"{td}/unconditional_probs_only/*.npz")
            if not f: return None
            lp = np.load(f[0])['log_p'][0]                      # (L, 21)
            rns = [(int(l[22:26]), _THREE1.get(l[17:20].strip()))
                   for l in open(pdb_path) if l.startswith('ATOM') and l[12:16].strip() == 'CA' and l[21] == chain]
            idx = {rn: k for k, (rn, _) in enumerate(rns)}
            k = idx.get(int(pos))
            if k is None or k >= lp.shape[0] or rns[k][1] != wt: return None
            return float(lp[k, MPNN_AA.index(mut)] - lp[k, MPNN_AA.index(wt)])
    except Exception:
        return None
