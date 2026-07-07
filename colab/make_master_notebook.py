"""Build colab/master.ipynb — the whole current cell model in one Colab run.

Clones the branch and runs, in order: the 24-axis recovery scorecard, the CellQA question-answering layer, the
whole-cell self-consistency audit, and the reasoned variant predictor (incl. the sickle-cell blind-spot demo).
Core cells are dependency-light (stdlib + numpy) and read committed data; heavier cells (AlphaMissense/ESM/
Human-GEM) are optional and clearly marked. CPU-only.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"
REPO = "https://github.com/Nikku03/cell.git"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}


cells = [
    md("# The cell model — one Colab run",
       "",
       "Clones the branch and runs the whole current system:",
       "1. **Recovery scorecard** — 24 capability axes, each gated vs known biology (the trust layer).",
       "2. **CellQA** — ask the cell: what binds X · knock out X · mutation effect · drug off-targets · "
       "disease→target · kcat · dark-gene function · cell-type-conditioned answers.",
       "3. **Whole-cell audit** — the model checks ITSELF: physics violations, localization conflicts, "
       "missing/wrong links, gaps — ranked, verified, nothing auto-applied.",
       "4. **Reasoned variant predictor** — AlphaMissense + mechanism + the sickle-cell blind-spot guard.",
       "",
       "Core cells need only numpy and the committed data. Heavier cells (network / Human-GEM / ESM) are marked "
       "**optional**."),

    md("## 1. Setup — clone the branch + light deps"),
    code(f"!git clone --depth 1 -b {BR} {REPO} 2>/dev/null || (cd cell && git pull)",
         "%cd cell",
         "!pip -q install numpy scipy scikit-learn 2>/dev/null",
         "import sys, os; sys.path.insert(0, 'colab')",
         "print('ready')"),

    md("## 1b. Restore the core data from Drive  **(required for CellQA / audit)**",
       "The 36 MB `cell_complete.json` is git-ignored on purpose, so it is **not** in the clone. Put it (or a "
       "`.json.gz`) in your Google Drive under **`MyDrive/cell_model/`** — the same convention the other "
       "notebooks use — and this cell restores it. (The scorecard cell below works without it.)"),
    code("os.makedirs('outputs/orphan', exist_ok=True)",
         "dst = 'outputs/orphan/cell_complete.json'",
         "if not os.path.exists(dst):",
         "    from google.colab import drive; drive.mount('/content/drive')",
         "    import glob, gzip, shutil",
         "    c = sorted(glob.glob('/content/drive/MyDrive/cell_model/**/cell_complete*.json*', recursive=True),",
         "               key=lambda p: os.path.getsize(p), reverse=True)",
         "    if not c:",
         "        raise FileNotFoundError('Upload cell_complete.json (or .json.gz) to MyDrive/cell_model/ — '",
         "                                'it is the 36MB core data, git-ignored on purpose.')",
         "    src = c[0]; print('restoring from', src)",
         "    (shutil.copyfileobj(gzip.open(src,'rb'), open(dst,'wb')) if src.endswith('.gz') else shutil.copy(src, dst))",
         "import json; D = json.load(open(dst))",
         "print('model loaded:', len(D['genes']), 'genes |', len(D['ctnames']), 'cell types |', len(D['emask']), 'in emask')"),

    md("## 2. Recovery scorecard — 24/24 (reads committed results, no network)"),
    code("!python colab/recovery_scorecard.py"),

    md("## 3. CellQA — ask the cell (committed data; fast)"),
    code("from cell_qa import CellQA, coverage",
         "coverage()"),
    code("qa = CellQA()   # loads the complete (ghost-patched) model",
         "for label, r in [",
         "    ('what does TP53 bind?', qa.what_binds('TP53', 5)),",
         "    ('remove SREBF2 -> downstream?', qa.knockout('SREBF2', 6)),",
         "    ('SREBF2 regulates?', qa.regulates('SREBF2', 6)),",
         "    ('kcat of HK1?', qa.kcat('HK1')),",
         "    ('psoriasis (IL23) -> target?', qa.disease_target(['IL23A','IL12B'],",
         "        ['IL17A','IL17F','IL22','CCL20','IL21'],",
         "        pathway=['IL23A','IL12B','IL23R','JAK2','STAT3','RORC','IL17A','STAT4'], k=5)),",
         "    ('GATA1 binds in erythroid?', qa.what_binds('GATA1', 5, cell_type='erythroid progenitor')),",
         "    ('GATA1 binds in T cell? (gated off)', qa.what_binds('GATA1', 5, cell_type='regulatory T cell')),",
         "]:",
         "    print('\\nQ:', label)",
         "    if r.get('abstain'): print('   ABSTAIN —', r['reason']); continue",
         "    for k in ('measured','predicted'):",
         "        if r.get(k): print(f'   {k}: ' + ', '.join(f\"{a['entity']}({a['confidence']})\" for a in r[k][:5]))",
         "    for k in ('kcat_per_s','tier','note'):",
         "        if k in r: print(f'   {k}: {r[k]}')"),

    md("## 4. Whole-cell self-consistency audit (numpy + committed data)",
       "Physics violations · localization conflicts · missing links (triadic closure) · wrong links · gaps — "
       "ranked, under the anti-trap hierarchy (hard-constraint > measured > predicted; nothing auto-applied)."),
    code("from self_consistency import SelfConsistency",
         "from patch_ghosts import apply_patch",
         "sc = SelfConsistency(); apply_patch(sc.D)",
         "rep = sc.whole_cell_audit(relations=('ppi','reg','sig'), top_per=40)",
         "print('flags:', rep['n_flags'], '| by tier:', rep['by_tier'], '| by kind:', rep['by_kind'])",
         "print('\\ntop missing-link candidates (PPI, triadic closure):')",
         "for f in [x for x in rep['flags'] if x.get('evidence')=='completion'][:10]:",
         "    print('  ', f['entity'], '—', f['detail'].get('shared_partners'), 'shared partners')",
         "print('\\nphysics violations (kcat past the diffusion limit):')",
         "for f in [x for x in rep['flags'] if x.get('evidence')=='provable'][:5]:",
         "    print('  ', f['entity'], '::', f['odd'][:60])"),

    md("## 5. Reasoned variant predictor — the sickle-cell lesson  *(optional: network)*",
       "AlphaMissense drives the call; ΔΔG/active-site give the reason; a gain-of-function guard overrides a "
       "false-benign call. Even the SOTA scores sickle cell benign — the predictor refuses that false certainty."),
    code("from reasoned_variant import ReasonedVariant",
         "rv = ReasonedVariant()",
         "for g,up,pos,wt,mut,lab in [('HBB','P68871',7,'E','V','sickle cell (gain-of-function)'),",
         "                            ('PAH','P00439',408,'R','W','PKU R408W (classic pathogenic)')]:",
         "    r = rv.predict(g, up, pos, wt, mut)",
         "    print(f\"\\n{lab}: {g} {wt}{pos}{mut}\")",
         "    print('  AlphaMissense', r.get('score'), '-> call:', r['call'], '| blind_spot:', r.get('ml_blind_spot'))",
         "    print('  why:', r['reasoning']['why'][:160])"),

    md("## 6. Whole-cell kcat consistency — test the PREDICTED kcats (not just central carbon)",
       "The flux-based check only reaches the ~334 enzymes carrying flux in one condition (central carbon, mostly "
       "measured). This tests **all 2,549** kcats — especially the ~2,100 predicted ones — against provable, "
       "flux-free bounds: the physics ceiling (kcat/Km < diffusion limit) and the independent in-vivo floor "
       "(an enzyme can't run faster than its own kcat). Measured kcats stay facts; only predicted outliers are "
       "flagged, each with a proposed value."),
    code("!python colab/whole_cell_kcat.py"),

    md("## 7. Optional deep runs *(heavier: Human-GEM / ESM / network)*",
       "Uncomment to run the flux-based kinetics⇄flux check, the cross-validation harness, or measured capacity."),
    code("# !pip -q install cobra mygene fair-esm torch --quiet",
         "# !python colab/kinetics_flux_consistency.py     # flux-based capacity check (needs Human-GEM + mygene)",
         "# !python colab/crossval_measured.py             # cross-validate predictions vs cell-line codep",
         "# !python colab/validate_ecflux_ppm.py           # measured per-enzyme capacity",
         "print('see docs/ for each capability; DATASET_TRAINING_PLAN.md for adding DepMap/Tahoe/GEO')"),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path("colab/master.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
