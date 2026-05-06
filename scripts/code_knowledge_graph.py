"""Build a code-knowledge graph of this repository (GitNexus-inspired).

Walks all .py files under the project root, parses them with Python's `ast`
module, and builds a multi-relation graph:

  Nodes  : modules (one per .py), functions, classes
  Edges  : imports, calls, inherits, contains (file -> defs)

Outputs:
  outputs/code_graph.json
      Full graph as nodes + edges JSON for downstream use
  outputs/code_graph_modules.png
      Static matplotlib rendering of the MODULE-level subgraph
      (easier to read than the full call graph)
  outputs/code_graph_layers.csv
      Per-layer summary: # files, # functions, # imports in/out
  outputs/code_graph_interactive.html
      Self-contained HTML + vis.js (CDN-loaded) for browser exploration

Run:  python3 scripts/code_knowledge_graph.py [--root /path/to/repo]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

# ────────────────────────────────────────────────────────────────────
# Layer color scheme — matches the cell-sim layered architecture so the
# module graph visually highlights the layering. Falls back gracefully
# for paths that don't match a known layer.
LAYER_COLORS = {
    'layer1_sbml': '#1f77b4',
    'layer2_field': '#ff7f0e',
    'layer3_reactions': '#2ca02c',
    'layer4_regulation': '#d62728',
    'layer5_bioelectric': '#9467bd',
    'layer6_essentiality': '#8c564b',
    'layer_ml': '#e377c2',
    'atom_engine': '#7f7f7f',
    'features': '#bcbd22',
    'scripts': '#17becf',
    'notebooks': '#aec7e8',
    'tests': '#ffbb78',
    'data': '#98df8a',
    'experiments': '#f7b6d2',
    'priority1_removal_patch': '#ff9896',
    'other': '#c7c7c7',
}


def detect_layer(rel_path: str) -> str:
    parts = Path(rel_path).parts
    for p in parts:
        if p in LAYER_COLORS:
            return p
    return 'other'


# ────────────────────────────────────────────────────────────────────
class FileVisitor(ast.NodeVisitor):
    """One pass over a .py file. Captures imports, top-level defs, calls."""

    def __init__(self, module_name: str):
        self.module = module_name
        self.imports: list[str] = []         # imported module names
        self.functions: list[str] = []        # top-level + nested fn names (with qualifiers)
        self.classes: list[tuple[str, list[str]]] = []   # (class, [parent_names])
        self.calls: set[tuple[str, str]] = set()  # (caller_qualname, callee_name)
        self._scope: list[str] = [module_name]

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base = node.module
            if node.level:
                base = '.' * node.level + base
            self.imports.append(base)
            for alias in node.names:
                self.imports.append(f'{base}.{alias.name}')

    def visit_FunctionDef(self, node: ast.FunctionDef):
        qual = '.'.join(self._scope) + '.' + node.name
        self.functions.append(qual)
        self._scope.append(node.name)
        # Walk inside to capture calls and nested defs
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = self._call_name(child.func)
                if callee:
                    self.calls.add((qual, callee))
        self._scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        qual = '.'.join(self._scope) + '.' + node.name
        parents = [self._call_name(b) or '' for b in node.bases]
        parents = [p for p in parents if p]
        self.classes.append((qual, parents))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = FileVisitor._call_name(node.value)
            return f'{base}.{node.attr}' if base else node.attr
        return None


# ────────────────────────────────────────────────────────────────────
def parse_file(py_path: Path, root: Path) -> dict:
    rel = str(py_path.relative_to(root))
    module_name = rel.replace('/', '.').removesuffix('.py')
    try:
        src = py_path.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return {'module': module_name, 'rel_path': rel,
                'imports': [], 'functions': [], 'classes': [], 'calls': set(),
                'lines': 0, 'parse_error': True}
    v = FileVisitor(module_name)
    v.visit(tree)
    return {
        'module': module_name, 'rel_path': rel,
        'imports': v.imports, 'functions': v.functions,
        'classes': v.classes, 'calls': v.calls,
        'lines': src.count('\n') + 1, 'parse_error': False,
    }


# ────────────────────────────────────────────────────────────────────
def build_graph(root: Path, files: list[dict]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    # Map module_name -> file rel_path for resolving imports
    module_to_file = {f['module']: f for f in files}

    # 1. Add module nodes
    for f in files:
        G.add_node(f['module'], kind='module', layer=detect_layer(f['rel_path']),
                    rel_path=f['rel_path'], lines=f['lines'],
                    n_functions=len(f['functions']),
                    n_classes=len(f['classes']))

    # 2. Add function + class nodes (qualified names)
    for f in files:
        for fn in f['functions']:
            G.add_node(fn, kind='function', layer=detect_layer(f['rel_path']),
                        rel_path=f['rel_path'])
            G.add_edge(f['module'], fn, relation='contains')
        for cls, parents in f['classes']:
            G.add_node(cls, kind='class', layer=detect_layer(f['rel_path']),
                        rel_path=f['rel_path'], parents=parents)
            G.add_edge(f['module'], cls, relation='contains')
            for parent in parents:
                G.add_edge(cls, parent, relation='inherits')

    # 3. Add import edges (module -> module). Resolve only if the imported
    # module belongs to this repo.
    for f in files:
        for imp in f['imports']:
            # Truncate trailing symbol after last dot (e.g. .ClassName)
            base = imp
            for cand in [imp, imp.rsplit('.', 1)[0]]:
                if cand in module_to_file:
                    G.add_edge(f['module'], cand, relation='imports')
                    break

    # 4. Add call edges where we can resolve callee name to a known function
    name_to_qualname = {}
    for n, d in G.nodes(data=True):
        if d.get('kind') in ('function', 'class'):
            short = n.rsplit('.', 1)[-1]
            name_to_qualname.setdefault(short, []).append(n)
    for f in files:
        for caller, callee in f['calls']:
            short = callee.rsplit('.', 1)[-1]
            candidates = name_to_qualname.get(short, [])
            for cand in candidates[:3]:   # cap to avoid spam
                G.add_edge(caller, cand, relation='calls')
    return G


# ────────────────────────────────────────────────────────────────────
def render_module_png(G: nx.MultiDiGraph, out_path: Path):
    """Module-level subgraph only — no functions/classes — colored by layer."""
    M = nx.DiGraph()
    for n, d in G.nodes(data=True):
        if d.get('kind') == 'module':
            M.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        if d.get('relation') == 'imports' and u in M and v in M:
            M.add_edge(u, v)

    print(f'  module-level graph: {M.number_of_nodes()} nodes, '
          f'{M.number_of_edges()} import edges')

    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(M, k=2.0, iterations=80, seed=42)

    layers_present = sorted({M.nodes[n]['layer'] for n in M.nodes()})
    for layer in layers_present:
        layer_nodes = [n for n in M.nodes() if M.nodes[n]['layer'] == layer]
        sizes = [50 + 6 * M.nodes[n].get('lines', 0) ** 0.5 for n in layer_nodes]
        nx.draw_networkx_nodes(M, pos, nodelist=layer_nodes,
                                  node_size=sizes,
                                  node_color=LAYER_COLORS.get(layer, '#888'),
                                  edgecolors='black', linewidths=0.4,
                                  alpha=0.85, ax=ax)

    nx.draw_networkx_edges(M, pos, edge_color='#444', alpha=0.3,
                              arrows=True, arrowstyle='->', arrowsize=8,
                              width=0.6, ax=ax,
                              connectionstyle='arc3,rad=0.05')

    # Label only the most-imported modules
    in_degree = dict(M.in_degree())
    top_imported = sorted(in_degree, key=in_degree.get, reverse=True)[:25]
    labels = {n: n.split('.')[-1] for n in top_imported}
    nx.draw_networkx_labels(M, pos, labels=labels, font_size=9,
                              font_weight='bold', ax=ax)

    legend = [mpatches.Patch(color=c, label=l)
              for l, c in LAYER_COLORS.items() if l in layers_present]
    ax.legend(handles=legend, loc='upper left', fontsize=10, framealpha=0.95)
    ax.set_title(f'Code Knowledge Graph — module-level\n'
                  f'{M.number_of_nodes()} modules, {M.number_of_edges()} imports\n'
                  f'(node size = file lines | colored by layer | '
                  f'top 25 most-imported labeled)',
                  fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


# ────────────────────────────────────────────────────────────────────
def render_interactive_html(G: nx.MultiDiGraph, out_path: Path,
                              max_nodes: int = 800):
    """Self-contained interactive HTML using vis.js from CDN. Filters to
    the most-connected nodes if the graph exceeds max_nodes for usability."""
    nodes_sorted = sorted(G.nodes(),
                            key=lambda n: -(G.in_degree(n) + G.out_degree(n)))
    keep = set(nodes_sorted[:max_nodes])
    sub = G.subgraph(keep).copy()

    edge_color = {
        'imports':  '#3a7bd5',
        'calls':    '#d34a3a',
        'inherits': '#8c564b',
        'contains': '#bbbbbb',
    }
    nodes_json = []
    for n, d in sub.nodes(data=True):
        kind = d.get('kind', 'other')
        layer = d.get('layer', 'other')
        size = 8 + (d.get('lines', 0) ** 0.4 if kind == 'module' else 4)
        nodes_json.append({
            'id': n,
            'label': n.split('.')[-1],
            'group': layer,
            'shape': {'module': 'dot', 'class': 'square',
                       'function': 'triangle'}.get(kind, 'dot'),
            'size': size,
            'color': LAYER_COLORS.get(layer, '#888'),
            'title': f'{n}<br>kind={kind}<br>layer={layer}<br>'
                      f'rel_path={d.get("rel_path", "")}',
        })
    edges_json = []
    seen = set()
    for u, v, d in sub.edges(data=True):
        rel = d.get('relation', 'other')
        # Collapse duplicate edges of same relation
        key = (u, v, rel)
        if key in seen: continue
        seen.add(key)
        edges_json.append({
            'from': u, 'to': v,
            'arrows': 'to' if rel != 'contains' else '',
            'color': {'color': edge_color.get(rel, '#999'), 'opacity': 0.55},
            'width': 2 if rel in ('inherits', 'imports') else 1,
            'dashes': rel == 'contains',
            'title': rel,
        })

    html = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>cell-sim code graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; }
  #header { padding: 10px 20px; background: #2a2a2a; color: white; }
  #header h1 { margin: 0; font-size: 16px; }
  #header p { margin: 4px 0 0; font-size: 12px; opacity: 0.85; }
  #network { width: 100vw; height: calc(100vh - 60px); border: none; }
  .legend { position: absolute; right: 12px; top: 70px; background: white;
            padding: 10px; border: 1px solid #ccc; font-size: 11px;
            border-radius: 4px; }
  .legend div { margin: 2px 0; }
  .legend span.dot { display: inline-block; width: 11px; height: 11px;
                       border-radius: 50%; margin-right: 5px;
                       vertical-align: middle; }
</style></head>
<body>
<div id="header">
  <h1>cell-sim code knowledge graph</h1>
  <p>__N_NODES__ nodes, __N_EDGES__ edges. Drag to pan. Scroll to zoom. Click a node to explore.</p>
</div>
<div id="network"></div>
<div class="legend"><strong>Layers</strong>__LEGEND__</div>
<script>
const nodes = new vis.DataSet(__NODES__);
const edges = new vis.DataSet(__EDGES__);
const network = new vis.Network(document.getElementById('network'),
  { nodes, edges },
  {
    physics: { stabilization: { iterations: 200 }, barnesHut: { gravitationalConstant: -3000, springConstant: 0.01 } },
    interaction: { hover: true, multiselect: true, tooltipDelay: 100 },
    nodes: { font: { size: 11 } },
    edges: { smooth: { enabled: true, type: 'dynamic' } },
    groups: __GROUPS__,
  });
network.on('click', (params) => {
  if (params.nodes.length) {
    const id = params.nodes[0];
    console.log('node:', id, nodes.get(id));
  }
});
</script></body></html>'''

    legend_html = ''.join(
        f'<div><span class="dot" style="background:{c}"></span>{l}</div>'
        for l, c in LAYER_COLORS.items())
    groups = {l: {'color': c} for l, c in LAYER_COLORS.items()}
    html = (html
             .replace('__N_NODES__', str(len(nodes_json)))
             .replace('__N_EDGES__', str(len(edges_json)))
             .replace('__NODES__', json.dumps(nodes_json))
             .replace('__EDGES__', json.dumps(edges_json))
             .replace('__GROUPS__', json.dumps(groups))
             .replace('__LEGEND__', legend_html))
    out_path.write_text(html)
    print(f'  wrote {out_path}  (showing top {len(nodes_json)} nodes)')


# ────────────────────────────────────────────────────────────────────
def layer_summary(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    by_layer = defaultdict(lambda: {'modules': 0, 'functions': 0,
                                       'classes': 0, 'lines': 0})
    for n, d in G.nodes(data=True):
        layer = d.get('layer', 'other')
        kind = d.get('kind', '')
        if kind == 'module':
            by_layer[layer]['modules'] += 1
            by_layer[layer]['lines'] += d.get('lines', 0)
        elif kind == 'function':
            by_layer[layer]['functions'] += 1
        elif kind == 'class':
            by_layer[layer]['classes'] += 1
    # Cross-layer imports
    cross = defaultdict(lambda: defaultdict(int))
    for u, v, d in G.edges(data=True):
        if d.get('relation') != 'imports': continue
        lu = G.nodes[u].get('layer', 'other'); lv = G.nodes[v].get('layer', 'other')
        if lu != lv:
            cross[lu][lv] += 1
    for layer, stats in by_layer.items():
        out_imports = sum(cross[layer].values())
        in_imports = sum(d[layer] for d in cross.values())
        rows.append({
            'layer': layer,
            'modules': stats['modules'],
            'functions': stats['functions'],
            'classes': stats['classes'],
            'total_lines': stats['lines'],
            'imports_out': out_imports,
            'imports_in': in_imports,
        })
    return pd.DataFrame(rows).sort_values('total_lines', ascending=False)


# ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path,
                     default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--out', type=Path, default=None,
                     help='output directory (default: <root>/outputs)')
    args = ap.parse_args()
    root = args.root.resolve()
    out_dir = (args.out or (root / 'outputs')).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv',
                  'target', '.cache', 'dist', 'build', '.ipynb_checkpoints'}

    print(f'[1] scanning {root}...')
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith('.py'):
                py_files.append(Path(dirpath) / f)
    print(f'  found {len(py_files)} .py files')

    print('\n[2] parsing...')
    parsed = []
    errors = 0
    for p in py_files:
        f = parse_file(p, root)
        if f.get('parse_error'):
            errors += 1
        parsed.append(f)
    total_funcs = sum(len(f['functions']) for f in parsed)
    total_cls = sum(len(f['classes']) for f in parsed)
    total_calls = sum(len(f['calls']) for f in parsed)
    total_imports = sum(len(f['imports']) for f in parsed)
    print(f'  files parsed:      {len(parsed)}  ({errors} errors)')
    print(f'  functions found:   {total_funcs:,}')
    print(f'  classes found:     {total_cls:,}')
    print(f'  calls captured:    {total_calls:,}')
    print(f'  imports captured:  {total_imports:,}')

    print('\n[3] building graph...')
    G = build_graph(root, parsed)
    print(f'  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    print('\n[4] layer summary:')
    layer_df = layer_summary(G)
    print(layer_df.to_string(index=False))
    layer_df.to_csv(out_dir / 'code_graph_layers.csv', index=False)

    print('\n[5] rendering module-level PNG...')
    render_module_png(G, out_dir / 'code_graph_modules.png')

    print('\n[6] rendering interactive HTML...')
    render_interactive_html(G, out_dir / 'code_graph_interactive.html',
                              max_nodes=600)

    print('\n[7] dumping JSON...')
    json_data = {
        'nodes': [{'id': n, **{k: v for k, v in d.items()
                                  if isinstance(v, (str, int, float, list, bool))}}
                   for n, d in G.nodes(data=True)],
        'edges': [{'source': u, 'target': v, 'relation': d.get('relation')}
                   for u, v, d in G.edges(data=True)],
        'meta': {'n_files': len(parsed), 'n_parse_errors': errors,
                  'total_functions': total_funcs, 'total_classes': total_cls,
                  'total_calls': total_calls, 'total_imports': total_imports},
    }
    (out_dir / 'code_graph.json').write_text(json.dumps(json_data, default=str))
    print(f'  wrote {out_dir / "code_graph.json"}')

    print(f'\nDONE. Outputs in {out_dir}/')


if __name__ == '__main__':
    main()
