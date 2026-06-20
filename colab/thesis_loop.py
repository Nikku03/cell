"""Thesis-production loop: gather every result JSON + figure produced by the
A100 run, render a compact results section, and splice it into the capstone
paper between auto-gen markers (without touching the hand-written prose).

If CAPSTONE_PAPER.md contains the markers
    <!--AUTOGEN_START--> ... <!--AUTOGEN_END-->
the block between them is replaced. Otherwise a section is appended. A
standalone outputs/orphan/RESULTS_AUTOGEN.md is always written.
"""
import json
from datetime import date
from pathlib import Path

from af_common import OUT, REPO

START, END = "<!--AUTOGEN_START-->", "<!--AUTOGEN_END-->"


def _load(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None


def render():
    af = _load("af_torch_results.json")
    sc = _load("smart_cooccur_results.json")
    al = _load("active_learning_results.json")
    L = [f"## Auto-generated results ({date.today().isoformat()})", ""]

    if af:
        pl = af["pooled"]
        L += ["### 1. Transformer (PyTorch, leave-one-clade-out)", "",
              f"- Pooled AUC **{pl['auc']}**, AUPRC {pl['auprc']}",
              f"- Deployment operating point: brightness>=0.85 covers "
              f"**{pl.get('brightness85_coverage','?')}** of genes at "
              f"**{pl.get('brightness85_accuracy','?')}** accuracy",
              f"- Config: {af['config'].get('heads')}x{af['config'].get('head_dim')} heads, "
              f"{af['config'].get('blocks')} block(s), {af['config'].get('epochs')} epochs "
              f"({af.get('seconds','?')}s)", ""]
        L += ["| clade | n | AUC | nonEss cov@P90 | Ess cov@P70 |",
              "|---|---|---|---|---|"]
        for cl, d in af["per_clade"].items():
            L.append(f"| {cl} | {d['n']} | {d['auc']} | {d['ne_cov_p90']} | {d['ess_cov_p70']} |")
        L.append("")

    if sc:
        L += ["### 2. Smart cooccurrence rescue (calibrated stacker @ P>=0.90)", ""]
        for tag in ("with_dnds", "no_dnds"):
            if tag in sc:
                t = sc[tag]["transformer"]; s = sc[tag]["smart_combined"]; d = sc[tag]["delta"]
                L.append(f"- **{tag}**: transformer {t['coverage']:.3f}@{t['precision']:.3f} -> "
                         f"combined **{s['coverage']:.3f}@{s['precision']:.3f}** "
                         f"({d['coverage_pp']:+.2f}pp coverage, {d['precision_pp']:+.2f}pp precision; "
                         f"rescued {s['rescued']} @ tau={s['tau']:.3f})")
        if "dnds_marginal_coverage_pp" in sc:
            L.append(f"- dN/dS marginal coverage: **{sc['dnds_marginal_coverage_pp']:+.2f}pp**")
        L.append("")

    if al:
        L += ["### 3. Active-learning curves", ""]
        for cl, strat_d in al["results"].items():
            for strat, cv in strat_d.items():
                if cv:
                    a0, a1 = cv[0]["auc"], cv[-1]["auc"]
                    L.append(f"- {cl}/{strat}: AUC {a0} -> {a1} over "
                             f"{cv[0]['budget']}->{cv[-1]['budget']} revealed labels")
        L += ["", "![active learning](outputs/orphan/active_learning.png)", ""]

    return "\n".join(L)


def main():
    block = render()
    (OUT / "RESULTS_AUTOGEN.md").write_text(block)
    print("wrote RESULTS_AUTOGEN.md")
    paper = REPO / "CAPSTONE_PAPER.md"
    if paper.exists():
        txt = paper.read_text()
        wrapped = f"{START}\n{block}\n{END}"
        if START in txt and END in txt:
            pre = txt.split(START)[0]; post = txt.split(END)[1]
            txt = pre + wrapped + post
        else:
            txt = txt.rstrip() + "\n\n" + wrapped + "\n"
        paper.write_text(txt)
        print("spliced auto-gen results into CAPSTONE_PAPER.md")
    else:
        print("CAPSTONE_PAPER.md not found; standalone RESULTS_AUTOGEN.md only")


if __name__ == "__main__":
    main()
