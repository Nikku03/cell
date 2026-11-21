"""FIRE: run the whole-cell consistency + completion ML across every relation on the complete (ghost-patched)
cell, and save the full audit. -> outputs/orphan/whole_cell_audit.json"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from self_consistency import SelfConsistency
from patch_ghosts import apply_patch

sc = SelfConsistency()
apply_patch(sc.D)                                    # complete the cell before auditing it
rep = sc.whole_cell_audit(relations=("ppi", "reg", "sig"), top_per=50)
# keep the report compact: top flags per kind + all verified fixes
top = {}
for f in rep["flags"]:
    k = f.get("evidence", f["tier"])
    top.setdefault(k, [])
    if len(top[k]) < 25:
        top[k].append(f)
out = dict(summary=dict(n_flags=rep["n_flags"], by_tier=rep["by_tier"], by_kind=rep["by_kind"],
                        relations=rep["relations_scanned"], n_verified_fixes=len(rep["verified_fixes"])),
           top_by_kind=top, verified_fixes=rep["verified_fixes"][:40], hierarchy=rep["hierarchy"])
json.dump(out, open("outputs/orphan/whole_cell_audit.json", "w"), indent=2)
print("=" * 76)
print("WHOLE-CELL AUDIT — complete cell, all relations")
print("=" * 76)
print(f"  total flags: {rep['n_flags']}")
print(f"  by tier:  {rep['by_tier']}")
print(f"  by kind:  {rep['by_kind']}")
print(f"  verified fixes: {len(rep['verified_fixes'])}")
print("\n  top missing-link candidates (completions), by relation:")
for f in rep["flags"]:
    if f.get("evidence") == "completion":
        d = f["detail"]
        print(f"    [{d.get('relation')}] {f['entity']:24} {d.get('shared_partners')} shared partners (conf {f['confidence']})")
