"""First-principles functional features for orphan proteins — when homology (Foldseek/darkfn) fails, read the
signal straight off the AlphaFold structure + sequence. No homolog required.

Computes, per protein:
  - disorder        : fraction of residues with low AlphaFold pLDDT (B-factor col). High disorder => not an
                      enzyme; likely a scaffold / regulatory / condensate protein (and WHY Foldseek abstains).
  - transmembrane   : count of hydrophobic membrane-spanning windows (Kyte-Doolittle) => membrane protein.
  - signal_peptide  : N-terminal hydrophobic secretion signal => secreted / cell-surface.
  - nls             : basic (K/R) clusters => nuclear import.
  - pocket          : a geometry proxy for a concave buried cavity => possible active/binding site (enzyme/binder).
  - localization    : the call these features imply (membrane / secreted / nuclear / disordered-cytoplasmic / globular).

Honest scope: localization signals + disorder + TM are well-established, sequence/structure-computable calls.
'Pocket' here is a lightweight burial-geometry proxy (a real run would use fpocket); it flags candidacy, not a
proven catalytic site. Interaction MODE (competition/masking/activation) is NOT decided here — see notes.
"""
import io, urllib.request, json
import numpy as np

KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2,
      'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
      'Y': -1.3, 'V': 4.2}
THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
             'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
             'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"


def alphafold_pdb_text(uniprot, timeout=30):
    meta = json.loads(urllib.request.urlopen(AF_API.format(uniprot), timeout=timeout).read())
    return urllib.request.urlopen(meta[0]["pdbUrl"], timeout=timeout).read().decode()


def parse_ca(pdb_text):
    """CA coordinates, per-residue sequence, and pLDDT (B-factor) from an AlphaFold PDB."""
    xyz, seq, plddt = [], [], []
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            res = ln[17:20].strip()
            seq.append(THREE2ONE.get(res, "X"))
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            plddt.append(float(ln[60:66]))
    return np.array(xyz), "".join(seq), np.array(plddt)


def _tm_and_signal(seq):
    """Kyte-Doolittle sliding window -> transmembrane spans; N-terminal hydrophobic stretch -> signal peptide."""
    h = np.array([KD.get(a, 0.0) for a in seq])
    if len(h) < 19:
        return 0, False
    win = np.convolve(h, np.ones(19) / 19, mode="valid")
    # TM segments: window mean hydropathy > 1.6, merge adjacent
    tm, i = 0, 0
    above = win > 1.6
    while i < len(above):
        if above[i]:
            tm += 1
            while i < len(above) and above[i]:
                i += 1
        else:
            i += 1
    # signal peptide: a strong hydrophobic window within the first ~30 residues (but not a full TM protein N-term)
    sig = bool(len(win) > 5 and np.max(win[:min(25, len(win))]) > 2.0 and seq[0] == "M")
    return tm, sig


def _nls(seq):
    """crude monopartite NLS: a run of >=4 basic (K/R) residues in a 6-window."""
    b = np.array([1 if a in "KR" else 0 for a in seq])
    if len(b) < 6:
        return False
    return bool(np.max(np.convolve(b, np.ones(6), mode="valid")) >= 4)


def _pocket_proxy(xyz, plddt):
    """lightweight concave-cavity proxy: among well-folded (pLDDT>70) residues, fraction that are buried
    (many CA neighbours within 10A) yet border a low-density region => candidate pocket lining. Returns a
    0-1 score; high on globular domains with an internal cavity, low on extended/all-surface folds."""
    ok = plddt > 70
    if ok.sum() < 30:
        return 0.0
    P = xyz[ok]
    d = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    contacts = (d < 10).sum(1) - 1
    buried = contacts > np.percentile(contacts, 60)
    # a pocket needs buried residues that still have some short-range void: use spread of contact counts
    return float(np.clip((np.std(contacts) / (np.mean(contacts) + 1e-9)) * buried.mean(), 0, 1))


def features(uniprot, seq_only=None):
    """compute the first-principles feature set for a UniProt accession (fetches AlphaFold structure)."""
    try:
        pdb = alphafold_pdb_text(uniprot)
    except Exception as e:
        return dict(uniprot=uniprot, ok=False, reason=f"no AlphaFold structure: {e}")
    xyz, seq, plddt = parse_ca(pdb)
    if len(seq) < 10:
        return dict(uniprot=uniprot, ok=False, reason="structure too short")
    disorder = float((plddt < 50).mean())
    tm, sig = _tm_and_signal(seq)
    nls = _nls(seq)
    pocket = _pocket_proxy(xyz, plddt)
    # localization call from the signals (priority order)
    if sig and tm <= 1:
        loc = "secreted / cell-surface (signal peptide)"
    elif tm >= 2:
        loc = f"membrane protein ({tm} TM segments)"
    elif disorder > 0.5:
        loc = "intrinsically disordered (scaffold/regulatory/condensate — no rigid fold)"
    elif nls:
        loc = "possibly nuclear (basic NLS-like cluster)"
    else:
        loc = "globular cytoplasmic"
    return dict(uniprot=uniprot, ok=True, length=len(seq),
                mean_plddt=round(float(plddt.mean()), 1), disorder_frac=round(disorder, 2),
                transmembrane_segments=tm, signal_peptide=sig, nls_like=nls,
                pocket_score=round(pocket, 2), localization=loc,
                enzyme_candidate=bool(pocket > 0.35 and disorder < 0.3),
                note="disorder from AlphaFold pLDDT; TM/signal/NLS from sequence rules; pocket is a burial-geometry proxy")


if __name__ == "__main__":
    import sys
    print(json.dumps(features(sys.argv[1] if len(sys.argv) > 1 else "Q9H7C9"), indent=2))
