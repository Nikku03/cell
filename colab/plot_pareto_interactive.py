"""Interactive (Plotly) v1 BCE deployment dashboard.

Panel A -- Coverage vs Precision Pareto:
  * Transformer alone (sweep brightness threshold), hover = tau/coverage/precision/n
  * + GBM rescue (sweep stacker tau), hover = tau/coverage/precision
  * 0.90 precision floor; the locked deployment operating point (star)
Panel B -- Per-clade detail:
  * scatter of neCov@P90 vs AUC, bubble size = #genes, hover = clade/n/AUC/AUPRC/neCov
  * colour-coded so the weak clades (burkholderia, archaea) stand out

Run in a Colab cell for inline interactivity:
    import sys; sys.path.insert(0, "colab")
    import plot_pareto_interactive as P
    fig = P.build(); fig.show()
or:  python colab/plot_pareto_interactive.py   # writes the self-contained HTML
"""
import json, argparse
from pathlib import Path
import numpy as np

from af_common import OUT


def transformer_sweep(preds_path, n_steps=100):
    Z = np.load(preds_path, allow_pickle=True)
    p, br, y = Z["p"], Z["brightness"], Z["y"]
    ev = ~np.isnan(p); p, br, y = p[ev], br[ev], y[ev]
    N = len(y); call = (p > 0.5).astype(int)
    out = []
    for tau in np.linspace(0.0, 1.0, n_steps):
        m = br >= tau; n = int(m.sum())
        if n < 20:
            continue
        out.append((float(tau), n / N, float((call[m] == y[m]).mean()), n))
    return out, N


def build(preds_path=None, cooccur_json=None, results_json=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    preds_path = preds_path or (OUT / "af_torch_preds_aug.npz")
    cooccur_json = cooccur_json or (OUT / "smart_cooccur_results_aug.json")
    results_json = results_json or (OUT / "af_torch_results_aug.json")

    transf, N = transformer_sweep(preds_path)
    cj = json.load(open(cooccur_json))["with_dnds"]
    comb = cj["tradeoff_curve"]; op = cj["smart_combined"]
    per = json.load(open(results_json)).get("per_clade", {})

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.58, 0.42],
        subplot_titles=(f"Coverage vs Precision ({N:,} genes, 28 clades)",
                        "Per-clade: neCov@P90 vs AUC (bubble = #genes)"))

    # --- Panel A: transformer sweep ---
    tt = [f"brightness≥{t:.2f}<br>coverage {c:.3f}<br>precision {p:.3f}<br>n={n:,}"
          for t, c, p, n in transf]
    fig.add_trace(go.Scatter(
        x=[c for _, c, _, _ in transf], y=[p for _, _, p, _ in transf],
        mode="lines+markers", name="Transformer alone",
        line=dict(color="#2980B9", width=3), marker=dict(size=5),
        text=tt, hoverinfo="text"), row=1, col=1)

    # --- Panel A: combined + GBM ---
    tt2 = [f"stacker τ={c['tau']:.3f}<br>coverage {c['combined_coverage']:.3f}"
           f"<br>precision {c['combined_precision']:.3f}<br>rescued {c['rescued']:,}"
           for c in comb]
    fig.add_trace(go.Scatter(
        x=[c["combined_coverage"] for c in comb],
        y=[c["combined_precision"] for c in comb],
        mode="lines+markers", name="+ GBM rescue",
        line=dict(color="#27AE60", width=3), marker=dict(size=6, symbol="square"),
        text=tt2, hoverinfo="text"), row=1, col=1)

    # floor + operating point
    fig.add_hline(y=0.90, line=dict(color="red", dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[op["coverage"]], y=[op["precision"]], mode="markers+text",
        name="deployment", marker=dict(color="black", size=20, symbol="star"),
        text=[f"  {op['coverage']:.3f} @ {op['precision']:.3f}"],
        textposition="middle right", hoverinfo="text"), row=1, col=1)

    # --- Panel B: per-clade ---
    cl = list(per)
    aucs = [per[c]["auc"] for c in cl]
    necov = [per[c]["ne_cov_p90"] for c in cl]
    ns = [per[c]["n"] for c in cl]
    auprc = [per[c].get("auprc", float("nan")) for c in cl]
    txt = [f"<b>{c}</b><br>AUC {per[c]['auc']:.3f}<br>AUPRC {per[c].get('auprc','?')}"
           f"<br>neCov@P90 {per[c]['ne_cov_p90']:.2f}<br>n={per[c]['n']:,}" for c in cl]
    sizes = 8 + 30 * (np.array(ns) / max(ns)) ** 0.5
    fig.add_trace(go.Scatter(
        x=aucs, y=necov, mode="markers", name="clades", showlegend=False,
        marker=dict(size=sizes, color=necov, colorscale="RdYlGn", cmin=0, cmax=1,
                    line=dict(width=1, color="grey"),
                    colorbar=dict(title="neCov@P90", x=1.0, len=0.8)),
        text=txt, hoverinfo="text"), row=1, col=2)

    fig.update_xaxes(title_text="combined coverage", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="precision", range=[0.5, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="clade AUC", row=1, col=2)
    fig.update_yaxes(title_text="neCov@P90", range=[-0.05, 1.05], row=1, col=2)
    fig.update_layout(title="v1 BCE essentiality model — deployment dashboard",
                      hovermode="closest", template="plotly_white",
                      legend=dict(x=0.01, y=0.02), height=560, width=1150)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--cooccur", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=str(OUT / "v1_bce_pareto.html"))
    a = ap.parse_args()
    fig = build(a.preds, a.cooccur, a.results)
    fig.write_html(a.out, include_plotlyjs="cdn")
    print(f"wrote {a.out}")
