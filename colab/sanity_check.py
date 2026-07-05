"""SANITY CHECK — verify the pipeline is wired and runnable BEFORE the long assemble.

Checks, fast and offline:
  1. every colab/*.py parses (no syntax errors);
  2. cross-module imports resolve (compute_metabolites/dfba/space/invivo import compute_flux symbols);
  3. the quantitative dependency ORDER is correct (each producer runs before its consumer);
  4. required Python packages import;
  5. the key data anchors are present (delegates to preflight for the full list).
Prints PASS/FAIL per group and a one-line verdict. Never mutates anything.
"""
import ast, os, sys, importlib.util
from pathlib import Path
COLAB=Path("colab") if Path("colab").is_dir() else Path(".")

# producer -> output -> consumer chains that MUST be ordered correctly in the assemble
CHAINS=[
 ("fetch_flux_data.py","flux_medium.tsv","compute_flux.py"),
 ("compute_kinetics.py","kinetics.json","compute_flux.py"),
 ("compute_concentration.py","concentration.json","compute_flux.py"),
 ("compute_flux.py","flux.json","compute_metabolites.py"),
 ("compute_metabolites.py","metabolites.json","compute_metabolite_conc.py"),
 ("compute_flux.py","flux.json","compute_invivo_kcat.py"),
 ("compute_flux.py","flux.json","compute_catpred_kinetics.py"),
 ("fetch_kinetics_extra.py","kinetics_measured_extra.tsv","compute_kinetics.py"),
 ("fetch_metabolite_conc.py","metabolite_conc_measured.tsv","compute_metabolite_conc.py"),
 ("fetch_kinetics_extra.py","ki_measured.tsv","refine_kinetics.py"),
 ("compute_concentration.py","concentration.json","compute_davidi_kcat.py"),
 ("compute_metabolic_graph.py","metabolic_graph.json","refine_kinetics.py"),
 ("compute_catpred_kinetics.py","catpred_kinetics.json","refine_kinetics.py"),
 ("compute_davidi_kcat.py","davidi_kcat.json","refine_kinetics.py"),
 ("compute_flux.py","flux.json","refine_kinetics.py"),
 ("compute_conditions.py","conditions.json","compute_attractors.py"),
 ("compute_flux.py","flux.json","compute_variants.py"),
 ("compute_flux.py","flux.json","compute_dfba.py"),
]
# cross-module imports that must resolve
IMPORTS=[("compute_metabolites.py",["parse_gem","base_met"]),
         ("compute_dfba.py",["parse_gem","build_S","solve_fba","met_to_exchange","load_medium","mx_get"]),
         ("compute_davidi_kcat.py",["parse_gem","build_S","solve_fba","pfba","met_to_exchange","mx_get","vmax_bounds","find_biomass"]),
         ("compute_space.py",["parse_gem"]),
         ("compute_invivo_kcat.py",[])]
PKGS=["numpy","scipy","pandas","h5py"]

def notebook_order():
    """map script -> first position it is invoked in the generated notebook (for order checks)."""
    import json, re
    nb=Path("colab/build_complete_cell.ipynb")
    if not nb.exists(): return None
    src="\n".join("".join(c["source"]) for c in json.load(open(nb))["cells"] if c["cell_type"]=="code")
    pos={}
    for m in re.finditer(r"colab/([a-z_]+\.py)", src):
        pos.setdefault(m.group(1), m.start())
    return pos

def main():
    ok=True
    pos=notebook_order()
    # 1. parse — only the scripts actually WIRED into the notebook (ignore old/experimental colab/*.py)
    wired=sorted(set(pos)|{c for _,_,c in CHAINS}|{p for p,_,_ in CHAINS}|{f for f,_ in IMPORTS}) if pos else \
          sorted(p.name for p in COLAB.glob("compute_*.py"))
    bad=[]
    for name in wired:
        p=COLAB/name
        if not p.exists(): bad.append(f"{name}: NOT FOUND (wired but missing)"); continue
        try: ast.parse(open(p).read())
        except SyntaxError as e: bad.append(f"{name}: {e}")
    print(f"[1] syntax: {'PASS' if not bad else 'FAIL'} ({len(wired)} wired scripts)")
    for b in bad: print("     ", b); ok=False
    # 2. cross-imports resolve
    def symbols(fname):
        try: return {n.name for node in ast.walk(ast.parse(open(COLAB/fname).read()))
                     if isinstance(node,(ast.FunctionDef,ast.Assign)) for n in ([node] if isinstance(node,ast.FunctionDef) else node.targets) if hasattr(n,'name') or isinstance(n,ast.Name)}
        except Exception: return set()
    flux_syms={n.name for node in ast.walk(ast.parse(open(COLAB/"compute_flux.py").read())) if isinstance(node,ast.FunctionDef) for n in [node]}
    imp_ok=True
    for fname,needs in IMPORTS:
        miss=[s for s in needs if s not in flux_syms]
        if miss: print(f"[2] import: {fname} needs {miss} from compute_flux -> MISSING"); imp_ok=False; ok=False
    print(f"[2] cross-imports: {'PASS' if imp_ok else 'FAIL'}")
    # 3. order
    ord_ok=True
    if pos:
        for prod,out,cons in CHAINS:
            if prod in pos and cons in pos and pos[prod]>=pos[cons]:
                print(f"[3] ORDER: {prod} runs after {cons} but must produce {out} first -> WRONG"); ord_ok=False; ok=False
        print(f"[3] dependency order: {'PASS' if ord_ok else 'FAIL'} ({len(CHAINS)} chains checked)")
    else:
        print("[3] dependency order: SKIP (notebook not found)")
    # 4. packages
    miss=[p for p in PKGS if importlib.util.find_spec(p) is None]
    print(f"[4] packages: {'PASS' if not miss else 'MISSING '+str(miss)}")
    if miss: ok=False
    # 5. data anchors (delegate)
    print("[5] data sources -> running preflight_check ...")
    try:
        import subprocess; subprocess.run([sys.executable, str(COLAB/"preflight_check.py")])
    except Exception as e: print("     preflight failed:", repr(e)[:80])
    print("="*60); print("SANITY:", "ALL CONNECTED — safe to run" if ok else "PROBLEMS ABOVE — fix before the long run"); print("="*60)

if __name__=="__main__":
    main()
