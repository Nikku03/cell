"""Compute I-RMSD and difficulty class for every DB5.5 case. Derived, not transcribed."""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from rem.docking.data import (list_complexes, load_case, interface_rmsd, classify,
                              residue_name_agreement)

ids = list_complexes()
rows, bad = [], []
t0 = time.time()
for i, cid in enumerate(ids):
    try:
        c = load_case(cid)
        r = interface_rmsd(c)
        ar, nr = residue_name_agreement(c["r_u"], c["r_b"])
        al, nl = residue_name_agreement(c["l_u"], c["l_b"])
        # Residues are now paired by SEQUENCE ALIGNMENT, which only emits a mapping when the
        # chain alignment reaches 70% identity -- so a bad alignment yields no mapping rather
        # than a plausible wrong one. The usability test is therefore on the alignment's own
        # output: enough residues mapped, and enough interface backbone atoms to superimpose.
        ok = (r.get("receptor_n_mapped", 0) >= 10 and r.get("ligand_n_mapped", 0) >= 10
              and r["receptor_n_atoms"] >= 12 and r["ligand_n_atoms"] >= 12
              and np.isfinite(r["combined"]))
        rec = {"id": cid, "irmsd": float(r["combined"]),
               "irmsd_receptor": float(r["receptor"]), "irmsd_ligand": float(r["ligand"]),
               "class": classify(r["combined"]) if ok else "unusable",
               "rec_name_agree": ar, "lig_name_agree": al,
               "n_rec": int(nr), "n_lig": int(nl),
               "n_mapped_r": int(r.get("receptor_n_mapped", 0)),
               "n_mapped_l": int(r.get("ligand_n_mapped", 0)),
               "n_iface_atoms_r": int(r["receptor_n_atoms"]),
               "n_iface_atoms_l": int(r["ligand_n_atoms"]),
               "n_atoms_r": int(len(c["r_u"])), "n_atoms_l": int(len(c["l_u"]))}
        (rows if ok else bad).append(rec)
    except Exception as e:
        bad.append({"id": cid, "class": "error", "err": f"{type(e).__name__}: {e}"})
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(ids)}  [{time.time()-t0:.0f}s]", flush=True)

print()
print(f"usable {len(rows)} / {len(ids)}   unusable/error {len(bad)}")
from collections import Counter
cnt = Counter(r["class"] for r in rows)
tot = sum(cnt.values())
print()
print(f"  {'class':>10s} {'n':>5s} {'measured':>9s}   published DB5")
pub = {"rigid": "~65%", "medium": "~20%", "difficult": "~15%"}
for k in ("rigid", "medium", "difficult"):
    print(f"  {k:>10s} {cnt[k]:>5d} {cnt[k]/tot:>8.1%}   {pub[k]}")
q = np.array([r["irmsd"] for r in rows])
print(f"\n  I-RMSD  min {q.min():.2f}  median {np.median(q):.2f}  max {q.max():.2f} A")
print(f"  hardest 6: " + ", ".join(f"{r['id']}({r['irmsd']:.2f})"
                                   for r in sorted(rows, key=lambda x: -x['irmsd'])[:6]))
if bad:
    print(f"\n  excluded ({len(bad)}): " + ", ".join(b['id'] for b in bad[:14])
          + (" ..." if len(bad) > 14 else ""))
json.dump({"usable": rows, "excluded": bad}, open("benchmarks/db5_classification.json", "w"),
          indent=1)
print("\n  written benchmarks/db5_classification.json")
