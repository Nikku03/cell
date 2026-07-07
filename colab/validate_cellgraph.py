"""VALIDATE the CellGraph learned model — gate its measured capabilities (all leakage-controlled).

Reads outputs/orphan/cellgraph_metrics.json (produced by colab/cellgraph.py) and checks each capability
clears an honest bar, well above its random baseline:
  - link prediction (what binds what): PPI AUC (hard negatives, test edges removed)
  - perturbation -> downstream DIRECTION: sign-accuracy on held-out signed edges
  - drug polypharmacology: held-out drug-target recovery AUC
  - structure -> function: essentiality/TF AUC from a pure-topology embedding
-> cellgraph_validation.json
"""
import json
from pathlib import Path

OUT = Path("outputs/orphan")


def validate():
    p = OUT / "cellgraph_metrics.json"
    if not p.exists():
        print("no cellgraph_metrics.json (run colab/cellgraph.py)."); return None
    m = json.load(open(p))
    lp = m["link_prediction"]["ppi"]["auc"]
    pdir = m["perturbation"]["direction"]["sign_accuracy"]
    drug = m["drug_polypharmacology"]["auc"]
    fn = m["node_function"]
    struct_fn = max(fn.get("ess", {}).get("auc", 0), fn.get("tf", {}).get("auc", 0))
    checks = dict(
        link_prediction_ppi=dict(value=lp, bar=0.70, ok=lp >= 0.70),
        perturbation_direction=dict(value=pdir, bar=0.70, ok=pdir >= 0.70),
        drug_polypharmacology=dict(value=drug, bar=0.70, ok=drug >= 0.70),
        structure_predicts_function=dict(value=struct_fn, bar=0.65, ok=struct_fn >= 0.65),
    )
    all_ok = all(c["ok"] for c in checks.values())
    return dict(all_pass=all_ok, checks=checks,
                summary=dict(link_ppi_auc=lp, perturb_direction_acc=pdir,
                             drug_auc=drug, structure_function_auc=round(struct_fn, 3)))


def main():
    res = validate()
    if not res:
        return
    json.dump(res, open(OUT / "cellgraph_validation.json", "w"), indent=2)
    print("=" * 74)
    print("CELLGRAPH — learned whole-cell model capabilities (leakage-controlled)")
    print("=" * 74)
    for k, c in res["checks"].items():
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {k:28} {c['value']}  (bar {c['bar']})")
    print(f"\n  all pass: {res['all_pass']}")


if __name__ == "__main__":
    main()
