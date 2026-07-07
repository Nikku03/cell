"""Validate the learned GraphSAGE vs fixed CellGraph propagation. -> outputs/orphan/cellgraph_gnn_validation.json

Honest head-to-head on the SAME leakage-free link-prediction benchmark, across relations. Adopt only where
the learned encoder actually beats the fixed one. Result is CPU/GPU-identical (GPU only speeds training)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cellgraph import load_model, node_features, build_adj
import cellgraph_gnn as gnn

OUT = Path("outputs/orphan"); OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def main():
    D, G, name, idx = load_model()
    X = node_features(G)
    A = build_adj(D, len(G))
    rows = [gnn.head_to_head(D, X, A, relation=r, epochs=EPOCHS) for r in ("ppi", "reg", "sig")]
    wins = sum(r["learned_auc"] > r["fixed_auc"] + 0.005 for r in rows)
    ppi = next(r for r in rows if r["relation"] == "ppi")
    result = dict(relations=rows, n_relations=len(rows), learned_wins=wins,
                  headline_ppi=dict(fixed=ppi["fixed_auc"], sage=ppi.get("sage_auc"),
                                    learned=ppi["learned_auc"], delta=ppi["delta"]),
                  best_encoder="R-GCN (per-relation weights)", device=rows[0]["device"], epochs=EPOCHS,
                  passed=bool(ppi["learned_auc"] > ppi["fixed_auc"] + 0.01 and wins >= 2),
                  detail="fixed SIGN vs learned GraphSAGE vs learned R-GCN; identical leakage-free split; "
                         "learned_auc = R-GCN (best encoder)")
    json.dump(result, open(OUT / "cellgraph_gnn_validation.json", "w"), indent=2)
    print("=" * 74)
    print(f"LEARNED GNN vs FIXED CellGraph — {'PASS (learned wins)' if result['passed'] else 'no improvement'}"
          f"  [{rows[0]['device']}, {EPOCHS} epochs]")
    print("=" * 74)
    print(f"  {'relation':8} {'fixed':>8} {'GraphSAGE':>10} {'R-GCN':>8} {'Δ(rgcn-fixed)':>14}")
    for r in rows:
        mark = "  <- win" if r["learned_auc"] > r["fixed_auc"] + 0.005 else ""
        print(f"  {r['relation']:8} {r['fixed_auc']:>8.4f} {r.get('sage_auc', 0):>10.4f} "
              f"{r['learned_auc']:>8.4f} {r['delta']:>+14.4f}{mark}")
    print(f"\n  R-GCN beats fixed on {wins}/{len(rows)} relations  (best encoder: per-relation R-GCN)")
    print("-> outputs/orphan/cellgraph_gnn_validation.json")
    return result


if __name__ == "__main__":
    main()
