"""End-to-end pipeline: fetch AlphaFold/ESMFold structures for Syn3A's
455 proteins, run atom_engine stability MD on each, output a per-gene
feature CSV that can be joined into the cross-org training data.

Stages
------
  extract    : pull (locus_tag, sequence) from syn3A.gb (offline, ~1 s)
  fetch      : retrieve a PDB structure per protein (REQUIRES INTERNET)
                  Tries ESMFold API first (free, ~5 s/protein, max 400 aa)
                  Falls back to AlphaFold DB by NCBI protein_id mapping
                  (limited; many Syn3A IDs lack UniProt links)
  stability  : run atom_stability MD on each fetched PDB (offline, ~5 s each)
  aggregate  : write outputs/syn3a_stability_features.csv (offline)
  all        : run the full pipeline end-to-end

Design notes
------------
  - Fetch is idempotent: if outputs/syn3a_pdbs/<locus>.pdb already exists,
    skip. This lets you resume after a network blip.
  - Stability is also idempotent: results cached in
    outputs/syn3a_pdbs/<locus>_stab.json
  - Long proteins (>400 aa) are skipped at fetch stage with reason
    "esmfold_too_long"; you can manually drop AlphaFold PDBs into the
    pdbs/ dir to fill them in.

Output schema
-------------
  outputs/syn3a_stability_features.csv columns:
    locus_tag, gene_name, length_aa, has_structure, fail_reason,
    rmsd_post_eq_nm, bonds_preserved_frac, temp_stability_score,
    overall_stability_score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

GENBANK = ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb'
PDB_DIR = ROOT / 'outputs/syn3a_pdbs'
OUT_CSV = ROOT / 'outputs/syn3a_stability_features.csv'

ESMFOLD_URL = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
ESMFOLD_MAX_LEN = 400         # API ceiling


# ───────────────────────── stage: extract ─────────────────────────
def extract_proteins() -> list[dict]:
    """Parse syn3A.gb and return a list of dicts with locus_tag, gene_name,
    protein_id, sequence."""
    from Bio import SeqIO
    rec = next(SeqIO.parse(str(GENBANK), 'genbank'))
    out = []
    for f in rec.features:
        if f.type != 'CDS': continue
        q = f.qualifiers
        seq = q.get('translation', [''])[0]
        if not seq: continue
        out.append({
            'locus_tag': q.get('locus_tag', [''])[0],
            'gene_name': q.get('gene', [''])[0] or q.get('product', [''])[0][:30],
            'protein_id': q.get('protein_id', [''])[0],
            'length_aa': len(seq),
            'sequence': seq.rstrip('*'),
        })
    return out


# ───────────────────────── stage: fetch ───────────────────────────
def fetch_esmfold(seq: str, *, timeout: float = 120.0) -> str | None:
    """Submit sequence to ESMFold public API, return PDB text or None."""
    if len(seq) > ESMFOLD_MAX_LEN:
        return None
    req = urllib.request.Request(
        ESMFOLD_URL, data=seq.encode('ascii'), method='POST',
        headers={'Content-Type': 'text/plain'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('ascii')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def fetch_alphafold_by_uniprot(uniprot_id: str, *, timeout: float = 30.0
                                  ) -> str | None:
    """Fetch from AlphaFold DB given a UniProt accession."""
    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode('ascii')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def fetch_all_structures(proteins: list[dict], *, max_proteins: int = 0,
                            sleep_s: float = 1.0) -> list[dict]:
    """For each protein, fetch + cache PDB. Returns proteins with
    `has_structure` and `fail_reason` filled in."""
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    if max_proteins > 0:
        proteins = proteins[:max_proteins]
    n = len(proteins)
    n_ok = n_skip = n_fail = 0
    for i, p in enumerate(proteins):
        pdb_path = PDB_DIR / f'{p["locus_tag"]}.pdb'
        if pdb_path.exists():
            p['has_structure'] = True
            p['fail_reason'] = ''
            n_skip += 1
            continue
        if p['length_aa'] > ESMFOLD_MAX_LEN:
            p['has_structure'] = False
            p['fail_reason'] = f'esmfold_too_long({p["length_aa"]}aa)'
            n_fail += 1
            continue
        # Try ESMFold
        pdb_text = fetch_esmfold(p['sequence'])
        if pdb_text and 'ATOM' in pdb_text:
            pdb_path.write_text(pdb_text)
            p['has_structure'] = True
            p['fail_reason'] = ''
            n_ok += 1
            print(f'  [{i+1}/{n}] {p["locus_tag"]} ({p["length_aa"]}aa) '
                  f'OK -> {pdb_path.name}')
        else:
            p['has_structure'] = False
            p['fail_reason'] = 'esmfold_api_fail'
            n_fail += 1
            print(f'  [{i+1}/{n}] {p["locus_tag"]} ({p["length_aa"]}aa) '
                  f'FAIL')
        if sleep_s and (n_ok + n_fail) % 1 == 0:
            time.sleep(sleep_s)
    print(f'\nfetch summary: ok={n_ok}, skipped(cached)={n_skip}, '
          f'fail={n_fail}, total={n}')
    return proteins


# ───────────────────────── stage: stability ───────────────────────
def run_stability_one(pdb_path: Path, *, n_steps: int = 4000,
                          dt_ps: float = 0.001, temperature_K: float = 300.0,
                          ) -> dict:
    """Run atom_stability MD on a single PDB; return metrics dict."""
    from cell_sim.atom_engine.integrator import (
        IntegratorConfig, SimState, run, current_temperature_K,
        build_shake_constraints,
    )
    from cell_sim.atom_engine.force_field import ForceFieldConfig
    from cell_sim.atom_engine.pdb_importer import load_pdb
    import numpy as np
    structure = load_pdb(str(pdb_path), temperature_K=temperature_K)
    state = SimState(atoms=structure.atoms, bonds=list(structure.bonds),
                       angles=list(structure.angles),
                       dihedrals=list(structure.dihedrals))
    state.shake_pairs, state.shake_r0_sq = build_shake_constraints(
        state.atoms, state.bonds)
    ff = ForceFieldConfig(lj_cutoff_nm=1.0, use_confinement=False,
                              use_neighbor_list=True, use_coulomb=True)
    int_cfg = IntegratorConfig(
        dt_ps=dt_ps, target_temperature_K=temperature_K,
        thermostat='langevin', langevin_gamma_inv_ps=2.0,
        shake=True, dynamic_bonding=False,
    )
    # Backbone indices (N, CA, C, O atoms; fall back to all heavy atoms)
    from cell_sim.atom_engine.element import Element
    bb = []
    for i, a in enumerate(state.atoms):
        nm = (getattr(a, 'name', None) or '').strip().upper()
        if nm in ('N', 'CA', 'C', 'O'): bb.append(i)
    if not bb:
        bb = [i for i, a in enumerate(state.atoms) if a.element != Element.H]
    bb = np.array(bb, dtype=np.int64)
    pos0 = np.array([state.atoms[i].position for i in bb])
    n_initial_bonds = len(state.bonds)
    rmsd_traj, temp_traj = [], []
    chunk = max(50, n_steps // 40)
    n_done = 0
    while n_done < n_steps:
        n_run = min(chunk, n_steps - n_done)
        run(state, n_run, ff, int_cfg)
        n_done += n_run
        pos_now = np.array([state.atoms[i].position for i in bb])
        pn = pos_now - pos_now.mean(axis=0)
        p0 = pos0 - pos0.mean(axis=0)
        rmsd = float(np.sqrt(np.mean(np.sum((pn - p0) ** 2, axis=1))))
        rmsd_traj.append(rmsd)
        temp_traj.append(current_temperature_K(state.atoms))
    eq_start = max(1, len(rmsd_traj) // 2)
    rmsd_eq = rmsd_traj[eq_start:]; temp_eq = temp_traj[eq_start:]
    rmsd_avg = float(np.mean(rmsd_eq)) if rmsd_eq else float('nan')
    if temp_eq:
        t_mean = float(np.mean(temp_eq)); t_std = float(np.std(temp_eq))
        t_drift = t_std / max(1.0, t_mean)
    else:
        t_mean = t_std = float('nan'); t_drift = 1.0
    bonds_pres = float(len(state.bonds) / max(1, n_initial_bonds))
    t_stab = float(max(0, 1 - t_drift))
    rmsd_score = float(1.0 / (1.0 + rmsd_avg * 10.0)) if not np.isnan(rmsd_avg) else 0.0
    overall = float((rmsd_score + bonds_pres + t_stab) / 3.0)
    return {
        'n_atoms': len(state.atoms),
        'n_initial_bonds': n_initial_bonds,
        'rmsd_post_eq_nm': rmsd_avg,
        'bonds_preserved_frac': bonds_pres,
        'temp_stability_score': t_stab,
        'overall_stability_score': overall,
    }


def run_stability_batch(proteins: list[dict], *, n_steps: int = 4000,
                          ) -> list[dict]:
    """Run stability MD on every protein with a cached PDB. Cache results
    per-locus in <pdb>_stab.json so reruns are cheap."""
    n = sum(1 for p in proteins if p.get('has_structure'))
    print(f'\n[stability] running MD on {n} proteins ({n_steps} steps each)')
    n_ok = n_skip = 0
    t0 = time.time()
    for i, p in enumerate(proteins):
        if not p.get('has_structure'):
            continue
        pdb_path = PDB_DIR / f'{p["locus_tag"]}.pdb'
        cache = PDB_DIR / f'{p["locus_tag"]}_stab.json'
        if cache.exists():
            p.update(json.loads(cache.read_text()))
            n_skip += 1
            continue
        try:
            metrics = run_stability_one(pdb_path, n_steps=n_steps)
            p.update(metrics)
            cache.write_text(json.dumps(metrics))
            n_ok += 1
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(proteins)}] {p["locus_tag"]} '
                  f'rmsd_eq={metrics["rmsd_post_eq_nm"]:.3f} '
                  f'bonds={metrics["bonds_preserved_frac"]:.2f} '
                  f'overall={metrics["overall_stability_score"]:.3f} '
                  f'({elapsed:.0f}s elapsed)')
        except Exception as e:
            p['fail_reason'] = (p.get('fail_reason') or
                                 f'stability_err:{type(e).__name__}')
            print(f'  [{i+1}/{len(proteins)}] {p["locus_tag"]} FAIL: '
                  f'{type(e).__name__}: {str(e)[:80]}')
    print(f'\nstability summary: ok={n_ok}, cached={n_skip}, total_with_pdb={n}')
    return proteins


# ───────────────────────── stage: aggregate ───────────────────────
def aggregate(proteins: list[dict]) -> pd.DataFrame:
    rows = []
    for p in proteins:
        rows.append({
            'organism': 'syn3a',
            'locus_tag': p['locus_tag'],
            'gene_name': p['gene_name'],
            'length_aa': p['length_aa'],
            'has_structure': bool(p.get('has_structure')),
            'fail_reason': p.get('fail_reason', ''),
            'rmsd_post_eq_nm': p.get('rmsd_post_eq_nm'),
            'bonds_preserved_frac': p.get('bonds_preserved_frac'),
            'temp_stability_score': p.get('temp_stability_score'),
            'overall_stability_score': p.get('overall_stability_score'),
        })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}  ({len(df)} rows)')
    n_ok = int(df.has_structure.sum())
    n_with_metrics = int(df.overall_stability_score.notna().sum())
    print(f'  proteins with structure: {n_ok}/{len(df)}')
    print(f'  proteins with stability metrics: {n_with_metrics}/{len(df)}')
    return df


# ─────────────────────────── main ─────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['extract', 'fetch', 'stability',
                                              'aggregate', 'all'],
                     required=True)
    ap.add_argument('--max-proteins', type=int, default=0,
                     help='limit number of proteins (0 = all 455)')
    ap.add_argument('--md-steps', type=int, default=4000,
                     help='atom_stability MD steps per protein (default 4000 = 4 ps)')
    ap.add_argument('--fetch-sleep', type=float, default=1.0,
                     help='seconds to sleep between API calls (rate limit)')
    args = ap.parse_args()

    print(f'[1] extract: parsing {GENBANK}')
    proteins = extract_proteins()
    print(f'  {len(proteins)} CDS records with translations')
    print(f'  length distribution: '
          f'min={min(p["length_aa"] for p in proteins)}  '
          f'median={sorted(p["length_aa"] for p in proteins)[len(proteins)//2]}  '
          f'max={max(p["length_aa"] for p in proteins)}')

    if args.stage == 'extract':
        # Just dump sequences as FASTA
        out = ROOT / 'outputs/syn3a_proteins.fasta'
        with open(out, 'w') as f:
            for p in proteins:
                f.write(f'>{p["locus_tag"]} {p["gene_name"]} '
                         f'protein_id={p["protein_id"]}\n')
                f.write(p['sequence'] + '\n')
        print(f'wrote {out}')
        return

    # Mark cached PDBs as has_structure=True and load any cached stability
    # metrics so subsequent stages can pick up where prior runs left off.
    for p in proteins:
        if (PDB_DIR / f'{p["locus_tag"]}.pdb').exists():
            p['has_structure'] = True
        cache = PDB_DIR / f'{p["locus_tag"]}_stab.json'
        if cache.exists():
            try:
                p.update(json.loads(cache.read_text()))
            except Exception:
                pass

    if args.stage in ('fetch', 'all'):
        print(f'\n[2] fetch: ESMFold API for proteins ≤{ESMFOLD_MAX_LEN}aa')
        proteins = fetch_all_structures(
            proteins, max_proteins=args.max_proteins,
            sleep_s=args.fetch_sleep)

    if args.stage in ('stability', 'all'):
        print(f'\n[3] stability: atom_engine MD per protein')
        proteins = run_stability_batch(proteins, n_steps=args.md_steps)

    if args.stage in ('aggregate', 'all'):
        print(f'\n[4] aggregate: write {OUT_CSV}')
        aggregate(proteins)


if __name__ == '__main__':
    main()
