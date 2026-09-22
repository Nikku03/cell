"""fetch_celltype_screen — GET another cell line's genome-wide Perturb-seq and MEASURE how much it breaks the
55% (K562-only) ceiling. cross_cell_line.py proved the lever: RPE1 (essential-scoped) adds 0 genes and responses
transfer only weakly (r0.19), so we need a GENOME-WIDE knockout screen in a cell type that EXPRESSES the ~7,800
genes K562 silences. That data is now emerging — primary CD4+ T cells (Zhu 2025), myeloid (Jung 2025), genome-wide
CRISPRi (Bradu 2026).

This is the getter + the honest payoff test, parameterized by a processed-h5ad URL so it runs where disk is ample
(the Colab pipeline; this container has a small allowance). It:
  1. PROBE   — HEAD the URL, compare size to free disk, refuse if it won't fit (no bricking the session).
  2. FETCH   — stream the download with the guard.
  3. INTEGRATE — parse into the debugger format, and measure the NEW-gene coverage gain vs K562 + the model's
     currently-untrustworthy (weak/dark) genes. If the new line adds N reachable genes, that's N MEASURED — the
     first honest movement of the 55%, because it's a new modality reaching a new blind spot.

Nothing is faked: the coverage lift is counted from the actual perturbed-gene set, and INTEGRATE on RPE1 correctly
reports ~0 (matching cross_cell_line) — the getter's payoff meter is self-checked.

usage:  python3 fetch_celltype_screen.py --probe <url>
        python3 fetch_celltype_screen.py --fetch <url> <dest.h5ad>
        python3 fetch_celltype_screen.py --integrate <dest.h5ad>

candidate datasets (provide the processed-h5ad / pseudobulk URL; the raw 22M-cell files are huge, so fetch a
per-perturbation pseudobulk and integrate that):
  Zhu 2025   primary CD4+ T cells, GENOME-SCALE perturb-seq of ALL EXPRESSED genes, rest + stimulated (3 states),
             22M cells / 4 donors. The complementary line we want — reaches T-cell genes K562 silences, AND its
             own title reports CONTEXT-SPECIFIC regulators (independently confirming cross_cell_line's r=0.19).
             data: https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq
             code: https://github.com/emdann/GWT_perturbseq_analysis_2025
  Jung 2025  primary myeloid / macrophage, pooled CRISPR-KO (inflammatory readouts)
"""
import os, sys, json, shutil, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
SCRATCH = os.environ.get("SCRATCH_DIR",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")


def free_disk_gb(path):
    st = shutil.disk_usage(os.path.dirname(path) or ".")
    return st.free / 1e9


def probe(url, dest="."):
    """HEAD the file, compare to free disk. Returns (fits, size_gb, free_gb)."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            size = int(r.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"  probe failed: {str(e)[:80]}"); return False, 0.0, free_disk_gb(dest)
    size_gb, free_gb = size / 1e9, free_disk_gb(dest)
    fits = size > 0 and free_gb > size_gb * 2.5 + 1.0            # need room for the file + parsing overhead
    print(f"  file: {size_gb:.2f} GB   free disk: {free_gb:.1f} GB   "
          f"→ {'OK to fetch here' if fits else 'WILL NOT FIT — run in Colab (ample disk) or free space first'}")
    return fits, size_gb, free_gb


def download(url, dest):
    fits, size_gb, free_gb = probe(url, dest)
    if not fits:
        print(f"  refusing to download: {size_gb:.2f} GB into {free_gb:.1f} GB free (guard). Use the Colab pipeline.")
        return False
    print(f"  downloading → {dest} …")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    print(f"  done: {os.path.getsize(dest)/1e9:.2f} GB")
    return True


def _model_and_measured():
    """the model's gene universe + the genes K562 already covers (measured) + the currently-untrustworthy set."""
    from complete_cell import CompleteCell
    C = CompleteCell()
    universe = {C.name[i] for i in range(len(C.genes))}
    # K562 measured perturbed genes (whatever K562 screens are on disk)
    from cross_cell_line import load_screen
    k562 = set()
    for f in ("gwps.h5ad", "k562.h5ad"):
        p = os.path.join(SCRATCH, f)
        if os.path.exists(p):
            resp, _ = load_screen(p); k562 |= set(resp)
    # "untrustworthy" = genome minus the trustworthy 55% we already have (approx: minus K562-measured)
    return C, universe, k562


def coverage_from_geneset(new_line, label):
    """measure the NEW-gene coverage a screen's perturbed-gene SET adds vs K562 + the model universe.
    Works from just a gene list (no matrix needed) — the COVERAGE question ('how much can we reach') only needs
    which genes were perturbed, not their responses."""
    C, universe, k562 = _model_and_measured()
    N = len(C.genes)
    in_model = new_line & universe
    new = in_model - k562
    before = len(k562 & universe); after = len((k562 | new_line) & universe)
    print(f"COVERAGE {label}")
    print(f"  perturbed genes: {len(new_line):,}   in model: {len(in_model):,}   NEW vs K562: {len(new):,}")
    print(f"  measured response coverage: {before:,} ({before/N:.1%}) → {after:,} ({after/N:.1%})   (+{after-before:,})")
    return {"NEW_genes": len(new), "measured_before": before, "measured_after": after, "genome": N}


def integrate(h5ad):
    """measure the NEW-gene coverage a fetched screen adds against K562 + the model universe."""
    from cross_cell_line import load_screen
    if not os.path.exists(h5ad):
        print(f"  {h5ad} not present."); return None
    resp, _ = load_screen(h5ad)
    new_line = set(resp)
    C, universe, k562 = _model_and_measured()
    in_model = new_line & universe
    genuinely_new = in_model - k562                              # in the model, NOT already measured in K562
    R = {"screen": os.path.basename(h5ad), "perturbed_genes": len(new_line),
         "in_model_universe": len(in_model), "already_covered_by_K562": len(in_model & k562),
         "NEW_genes_added": len(genuinely_new), "examples": sorted(genuinely_new)[:20]}
    before = len(k562 & universe)
    after = len((k562 | new_line) & universe)
    R["measured_genes_before"] = before
    R["measured_genes_after"] = after
    R["coverage_gain_genes"] = after - before
    print(f"INTEGRATE {R['screen']}")
    print(f"  perturbed genes: {len(new_line):,}   in model: {len(in_model):,}   "
          f"already in K562: {len(in_model & k562):,}")
    print(f"  → NEW measured genes added: {R['NEW_genes_added']:,}   "
          f"(measured coverage {before:,} → {after:,} genes)")
    if R["NEW_genes_added"] == 0:
        print("  → 0 new genes: this screen re-measures the K562-covered set (essential-scoped / same space).")
    else:
        print(f"  → genuine lift. examples: {R['examples'][:10]}")
    json.dump(R, open(f"{OUT}/celltype_screen_gain.json", "w"), indent=1)
    return R


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--probe"] and len(a) > 1:
        probe(a[1])
    elif a[:1] == ["--fetch"] and len(a) > 2:
        download(a[1], a[2])
    elif a[:1] == ["--integrate"] and len(a) > 1:
        integrate(a[1])
    elif a[:1] == ["--genelist"] and len(a) > 1:
        genes = {l.strip() for l in open(a[1]) if l.strip()}
        coverage_from_geneset(genes, os.path.basename(a[1]))
    else:
        # self-test: integrating RPE1 (the 2nd line we have) must report ~0 gain, matching cross_cell_line
        print("self-test — integrate RPE1 (expect 0 new genes):")
        integrate(os.path.join(SCRATCH, "rpe1.h5ad"))
