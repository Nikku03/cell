"""The DNA-binding domain as a measured shape, not as a sequence proxy.

WHAT THIS REPLACES AND WHY. `tf_domains.py` gave each factor a charge density, an arginine
fraction and a mean residue volume, all computed from the DOMAIN'S SEQUENCE. Those were stand-ins
for the quantities that actually decide whether a protein can grip a particular groove: how far the
domain reaches, how the charge is distributed over its surface, and whether its arginines point
outward into a groove or inward into the fold. A sequence cannot distinguish an arginine buried in
the core from one presented on the recognition helix, and the complementarity block built on those
proxies has failed every gate it was put through (loop 173 E7/E7b, loop 174 F6). This module
computes the same quantities from AlphaFold structures instead, so the block gets one honest
attempt with measured geometry before it is written off.

WHAT IS COMPUTED, per factor, over the DNA-binding domain's own residue range (the range
`tf_domains.py` already resolved, by DNA_BIND annotation, zinc-finger span, named fold, or the
labelled basic-window fallback):

    rg              radius of gyration of the domain's CA atoms -- how compact the fold is
    max_dim         the largest CA-to-CA distance -- how far along the duplex it can reach, which
                    is the structural form of "does the protein extend past the 6 bp core"
    dipole          magnitude of the charge dipole: the vector sum of charged side-chain positions
                    weighted by charge, divided by the radius of gyration. A domain that grips a
                    phosphate backbone has its positive charge on one face, and this is the number
                    that says so. Sequence net charge cannot.
    surf_charge     net charge of residues whose relative solvent accessibility exceeds 0.25,
                    computed with Shrake-Rupley. This is the charge a groove actually sees.
    arg_out         mean distance of arginine guanidinium carbons from the domain centroid divided
                    by the radius of gyration -- above one means the arginines point outward, which
                    is the geometry minor-groove readout requires (Rohs et al., Nature 2009).
    plddt           mean AlphaFold confidence over the domain. Carried as a CONTROL, not a feature:
                    a domain predicted at low confidence has geometry that may be invented, and any
                    result that depends on those factors should be visible as such.

DISK. AlphaFold entries are fetched one at a time, measured, and DELETED. Only the descriptors are
kept. Holding 700 structures would cost a third of the free space on this machine for numbers that
occupy a few kilobytes.

Output: colab/data/tf_structures.json
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from enh import tf_domains as TD             # noqa: E402

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "tf_structures.json"
TMP = SP / "af_tmp"
LOCAL = [SP / "af", SP / "af_cache"]
# AlphaFold DB moved to v6; the v4 path this module first used now 404s for every
# accession, which the first run discovered by failing 725 times in silence. The direct v6 URL is
# tried first and the prediction API is the fallback, so the next version bump costs one extra
# request per accession instead of the whole build.
AF = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
SASA_CUT = 0.25

POS = {"ARG": 1.0, "LYS": 1.0, "HIS": 0.1}
NEG = {"ASP": -1.0, "GLU": -1.0}
# Tien et al. (2013) theoretical maximum accessible surface area, square Angstrom
MAXASA = dict(ALA=129, ARG=274, ASN=195, ASP=193, CYS=167, GLN=225, GLU=223, GLY=104, HIS=224,
              ILE=197, LEU=201, LYS=236, MET=224, PHE=240, PRO=159, SER=155, THR=172, TRP=285,
              TYR=263, VAL=174)


def local_path(acc):
    for d in LOCAL:
        for name in (f"{acc}.pdb", f"AF-{acc}.pdb", f"AF-{acc}-F1-model_v4.pdb"):
            p = d / name
            if p.exists():
                return p
    return None


def fetch(acc, report=print):
    p = local_path(acc)
    if p:
        return p, False
    TMP.mkdir(parents=True, exist_ok=True)
    q = TMP / f"{acc}.pdb"
    urls = [AF.format(acc=acc)]
    for i, u in enumerate(urls):
        try:
            with urllib.request.urlopen(u, timeout=120) as r:
                q.write_bytes(r.read())
            return q, True
        except Exception:
            pass
    try:
        with urllib.request.urlopen(AF_API.format(acc=acc), timeout=60) as r:
            d = json.load(r)
        u = d[0].get("pdbUrl")
        if u:
            with urllib.request.urlopen(u, timeout=120) as r:
                q.write_bytes(r.read())
            return q, True
    except Exception:
        pass
    return None, False


def measure(path, lo, hi):
    """Descriptors over residues [lo, hi] (1-based inclusive) of an AlphaFold model."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    st = PDBParser(QUIET=True).get_structure("m", str(path))
    ch = next(st[0].get_chains())
    res = [r for r in ch if r.id[0] == " " and lo <= r.id[1] <= hi]
    if len(res) < 8:
        return None
    try:
        ShrakeRupley().compute(st[0], level="R")
        have_sasa = True
    except Exception:
        have_sasa = False
    ca = np.array([r["CA"].coord for r in res if "CA" in r], dtype=np.float64)
    if len(ca) < 8:
        return None
    cen = ca.mean(0)
    rg = float(np.sqrt(((ca - cen) ** 2).sum(1).mean()))
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=2)
    out = dict(n_res=len(res), rg=rg, max_dim=float(d.max()))
    dip = np.zeros(3)
    surf = 0.0
    args = []
    pl = []
    for r in res:
        nm = r.get_resname()
        q = POS.get(nm, 0.0) + NEG.get(nm, 0.0)
        if "CA" in r:
            pl.append(float(r["CA"].get_bfactor()))
        if q:
            atoms = [a.coord for a in r]
            dip += q * (np.mean(atoms, axis=0) - cen)
        if have_sasa and nm in MAXASA:
            rel = getattr(r, "sasa", 0.0) / MAXASA[nm]
            if rel > SASA_CUT:
                surf += q
        if nm == "ARG" and "CZ" in r:
            args.append(float(np.linalg.norm(r["CZ"].coord - cen)))
    out["dipole"] = float(np.linalg.norm(dip) / max(rg, 1e-6))
    out["surf_charge"] = float(surf) if have_sasa else float("nan")
    out["arg_out"] = float(np.mean(args) / max(rg, 1e-6)) if args else 0.0
    out["plddt"] = float(np.mean(pl)) if pl else float("nan")
    return out


def build(report=print):
    """The domain RANGE is not in tf_domains.json -- only its length was persisted -- so the
    UniProt features are re-fetched and `tf_domains.pick_domain` is re-run here. Re-running the
    same function rather than re-deriving the range from the stored length guarantees the geometry
    is measured over exactly the residues the sequence proxies were computed over, which is the
    whole point of the comparison this module exists to enable."""
    dom = TD.load()
    accs = {}
    for mid, r in dom.items():
        if r.get("uniprot") and r.get("route"):
            accs.setdefault(r["uniprot"], []).append(mid)
    report(f"    {len(accs)} distinct accessions behind {len(dom)} matrices")
    up = TD.uniprot_batch(list(accs), report)
    report(f"    UniProt features re-fetched for {len(up)} accessions")

    rows, n_local, n_fetch, n_fail, n_norange = {}, 0, 0, 0, 0
    routes = {}
    t0 = time.time()
    for k, (acc, mids) in enumerate(sorted(accs.items()), 1):
        rec = up.get(acc)
        if rec is None:
            n_norange += 1
            continue
        lo_, hi_, route = TD.pick_domain(rec)
        p, tmp = fetch(acc, report)
        if p is None:
            n_fail += 1
            continue
        n_fetch += int(tmp)
        n_local += int(not tmp)
        try:
            m = measure(p, lo_, hi_)
        except Exception:
            m = None
        if tmp:
            try:
                p.unlink()
            except Exception:
                pass
        if m is None:
            n_fail += 1
            continue
        m["route"] = route
        m["domain_start"] = int(lo_)
        m["domain_end"] = int(hi_)
        routes[route] = routes.get(route, 0) + 1
        for mid in mids:
            rows[mid] = m
        if k % 100 == 0:
            el = time.time() - t0
            report(f"      {k}/{len(accs)}  [{el:.0f}s, eta {el/k*(len(accs)-k):.0f}s]")
    report(f"    measured {len(rows)}/{len(dom)} matrices "
           f"({n_local} from the local cache, {n_fetch} fetched, {n_fail} failed, "
           f"{n_norange} without a UniProt record)")
    for r_, c_ in sorted(routes.items(), key=lambda x: -x[1]):
        report(f"      route {r_:14} {c_:4d}")
    if rows:
        for k in ("n_res", "rg", "max_dim", "dipole", "surf_charge", "arg_out", "plddt"):
            v = np.array([r[k] for r in rows.values()], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                report(f"      {k:12} median {np.median(v):8.2f}  "
                       f"range [{v.min():8.2f}, {v.max():8.2f}]")
    DATA.mkdir(parents=True, exist_ok=True)
    json.dump({"matrices": rows, "n": len(rows),
               "local": n_local, "fetched": n_fetch, "failed": n_fail,
               "no_uniprot_record": n_norange, "routes": routes,
               "source": "AlphaFold DB v4; Shrake-Rupley SASA; Tien et al. 2013 max ASA"},
              open(OUT, "w"), indent=1)
    report(f"  -> {OUT}")


def load():
    if not OUT.exists():
        raise SystemExit(f"{OUT} missing -- run `python colab/enh/tf_structures.py` first")
    return json.load(open(OUT))["matrices"]


if __name__ == "__main__":
    print("=" * 100)
    print("TF DNA-BINDING DOMAINS AS MEASURED GEOMETRY, from AlphaFold")
    print("=" * 100)
    build()
