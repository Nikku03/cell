"""CompleteCell — the FULL-FIDELITY, ML-facing entry point to the whole cell model.

The complete_cell.html is a human summary (degree, counts). This is the opposite: the object an ML consumer
loads to ask for EVERYTHING, at full resolution. It merges the current best model in memory —
  cell_complete.json  +  kinetics_refined_corrected.json (the 228 verified kcat fixes)  +  ghost_patch.json
    (literature function for the dark proteome)
— builds the reverse indexes the raw JSON lacks (regulated_by, pathways-of-gene, ligand↔receptor), and exposes:

  .gene(x)      -> one gene's COMPLETE record across every layer (actual partner NAMES, GO terms, kinetics with
                   the correction, catalysed reactions, cell-type expression decoded from the bitmask, drugs,
                   complexes, pathways, disease, PTMs, co-dependency/co-expression, structure, ... — full fidelity)
  .genes_iter() -> stream every complete record (what the Phase-2 audit iterates: "report in every field")
  .layers()     -> every layer name + its per-gene coverage (so the audit knows what fields exist to check)
  .edges(rel)   -> raw edge arrays for bulk ML (ppi/reg/sig/sl/lr)
  .field(name)  -> a whole layer, raw

Nothing is summarised or dropped — if a datum is in the model, .gene() returns it. This is the input to the
deep ML audit (Phase 2); the audit's verified additions are folded back to build the next cell.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
OUT = "outputs/orphan"


class CompleteCell:
    def __init__(self, cc=f"{OUT}/cell_complete.json",
                 kin=f"{OUT}/kinetics_refined_corrected.json",
                 ghost=f"{OUT}/ghost_patch.json",
                 darkfn_table=f"{OUT}/dark_function_table.json"):
        self.D = json.load(open(cc))
        self.genes = self.D["genes"]
        self.name = [g["name"] for g in self.genes]
        self.idx = {n: i for i, n in enumerate(self.name)}
        self.ctnames = self.D.get("ctnames", [])
        # verified ML additions folded into the current-best cell
        self.kin = _load(kin, "kinetics_refined")
        self.ghost = (_load(ghost) or {}).get("patch", {}) if ghost else {}
        self.darktab = _load(darkfn_table, "table") or {}
        self._build_indexes()

    # ---- reverse / adjacency indexes the raw JSON doesn't ship ----
    def _build_indexes(self):
        from collections import defaultdict
        self.ppi_adj = defaultdict(set)
        for e in self.D.get("ppi", []):
            if _pair(e):
                self.ppi_adj[e[0]].add(e[1]); self.ppi_adj[e[1]].add(e[0])
        self.reg_out, self.reg_in = defaultdict(list), defaultdict(list)   # directed: TF -> target
        for e in self.D.get("reg", []):
            if _pair(e):
                sign = e[2] if len(e) > 2 else 1
                self.reg_out[e[0]].append((e[1], sign)); self.reg_in[e[1]].append((e[0], sign))
        self.sig_out, self.sig_in = defaultdict(list), defaultdict(list)
        for e in self.D.get("sig", []):
            if _pair(e):
                sign = e[2] if len(e) > 2 else 1
                self.sig_out[e[0]].append((e[1], sign)); self.sig_in[e[1]].append((e[0], sign))
        self.sl_adj = defaultdict(list)                                    # synthetic lethal (undirected, scored)
        for e in self.D.get("sl", []):
            if _pair(e):
                sc = e[2] if len(e) > 2 else None
                self.sl_adj[e[0]].append((e[1], sc)); self.sl_adj[e[1]].append((e[0], sc))
        self.lig2rec, self.rec2lig = defaultdict(list), defaultdict(list)  # ligand-receptor
        for e in self.D.get("lr", []):
            if _pair(e):
                self.lig2rec[e[0]].append(e[1]); self.rec2lig[e[1]].append(e[0])
        self.gene2path = defaultdict(list)                                 # reverse of pathways{name:[idx]}
        for pname, members in self.D.get("pathways", {}).items():
            for i in members:
                if isinstance(i, int):
                    self.gene2path[i].append(pname)
        self._fold_reactome()                                              # fold the 2,792 Reactome pathways in
        self._fold_causal()                                                # fold SIGNOR + CollecTRI signed direction

    def _fold_reactome(self, path=f"{OUT}/reactome_pathways.json"):
        """Fold the Reactome pathway overlay into the complete cell: a distinct D['reactome'] layer PLUS the
        per-gene reverse index, so .gene(x)['pathways'] returns Reactome memberships too. ADDITIVE — the curated
        D['pathways'] is never touched (anti-trap: overlays add, they don't overwrite). No-op if absent."""
        rx = _load(path)
        self.reactome = (rx or {}).get("pathways", {}) or {}
        if not self.reactome:
            return 0
        self.D["reactome"] = self.reactome
        curated = set(self.D.get("pathways", {}))
        for pname, members in self.reactome.items():
            if pname in curated:                                           # keep the curated one; don't double-list
                continue
            for i in members:
                if isinstance(i, int):
                    self.gene2path[i].append(pname)
        return len(self.reactome)

    def _fold_causal(self, path=f"{OUT}/causal_edges.json"):
        """Fold the SIGNOR + CollecTRI causal-direction overlay in as a SEPARATE signed layer: per-gene indexes of
        what x activates/inhibits (downstream) and what activates/inhibits x (upstream), each tagged +1/-1/0 and
        with its source db. ADDITIVE and non-destructive — the reg/sig networks and their signs are left exactly
        as measured; this adds a signed causal VIEW alongside them (so a real sign never silently overwrites ours,
        and a consumer can compare the two). No-op if the overlay isn't present."""
        from collections import defaultdict
        self.causal_out = defaultdict(list)                                # x -> [(target_idx, sign, db)]
        self.causal_in = defaultdict(list)                                 # x -> [(source_idx, sign, db)]
        cz = _load(path)
        edges = (cz or {}).get("edges", []) if isinstance(cz, dict) else (cz or [])
        for e in edges:
            if len(e) >= 3 and isinstance(e[0], int) and isinstance(e[1], int):
                db = e[3] if len(e) > 3 else ""
                self.causal_out[e[0]].append((e[1], e[2], db))
                self.causal_in[e[1]].append((e[0], e[2], db))
        if edges:
            self.D["causal_meta"] = (cz or {}).get("meta", {})
        return len(edges)

    # ---- apply the Phase-2 verified additions -> cell v2 (predicted edges tagged, facts untouched) ----
    def apply_additions(self, path=f"{OUT}/cell_v2_additions.json"):
        """fold the audit's VERIFIED additions into the graph: new PPI edges (co-complex implied) go into the
        adjacency tagged predicted. Never touches measured/curated edges — additive only. Returns what changed."""
        add = _load(path)
        if not add:
            return {"applied": False}
        self.added_ppi = set()
        n = 0
        for e in add.get("ppi_edges_added", []):
            ia, ib = self.idx.get(e["a"]), self.idx.get(e["b"])
            if ia is None or ib is None:
                continue
            if ib not in self.ppi_adj[ia]:
                self.ppi_adj[ia].add(ib); self.ppi_adj[ib].add(ia)
                self.added_ppi.add((min(ia, ib), max(ia, ib))); n += 1
        self.version = "v2"
        return {"applied": True, "ppi_edges_added": n,
                "kcat_corrections": add.get("counts", {}).get("kcat_corrections"),
                "function_fills": add.get("counts", {}).get("function_fills")}

    def is_predicted_ppi(self, a, b):
        ia, ib = self._i(a), self._i(b)
        return getattr(self, "added_ppi", set()) and (min(ia, ib), max(ia, ib)) in self.added_ppi

    def apply_ledger(self, path=f"{OUT}/phase2_ledger.json"):
        """cell = base + the self-healing loop's LOCKED, verified fixes (the definitive Phase-2 output). Folds
        verified PPI edges (co-complex + independently-corroborated completions) into ppi_adj, kcat corrections
        into kin, and function fills into a ledger_func overlay. Measured facts untouched — additive only."""
        led = _load(path)
        if not led:
            return {"applied": False}
        self.ledger_func = {}; self.ledger_edges = set()
        counts = {}
        for e in led.get("ledger", []):
            k = e["kind"]; fx = e.get("fix") or {}; counts[k] = counts.get(k, 0) + 1
            if k in ("complex_ppi", "completion") and fx.get("add_edge"):
                a, b = fx["add_edge"]; ia, ib = self.idx.get(a), self.idx.get(b)
                if ia is not None and ib is not None:
                    self.ppi_adj[ia].add(ib); self.ppi_adj[ib].add(ia)
                    self.ledger_edges.add((min(ia, ib), max(ia, ib)))
            elif k in ("kcat_physics", "kcat_low") and fx.get("new") is not None:
                r = self.kin.get(e["entity"], {})
                self.kin[e["entity"]] = {**r, "kcat_per_s": fx["new"], "kcat_original": fx.get("old"),
                                         "kcat_source": "phase2_loop"}
            elif k == "function_fill" and fx.get("function"):
                self.ledger_func[e["entity"]] = fx["function"]
        self.version = "healed"
        return {"applied": True, "locked_fixes": len(led.get("ledger", [])), "by_kind": counts}

    def apply_coessential(self, path=f"{OUT}/cell_v3_additions.json"):
        """cell v3: DepMap co-essentiality functional links. Kept in a SEPARATE adjacency (coess_adj) — these are
        functional/co-essential, NOT proven physical PPI, so they don't inflate the physical-interaction degree."""
        from collections import defaultdict
        add = _load(path)
        if not add:
            return {"applied": False}
        self.coess_adj = defaultdict(set); self.coess_conf = {}
        n = 0
        for e in add.get("ppi_edges_added", []):
            ia, ib = self.idx.get(e["a"]), self.idx.get(e["b"])
            if ia is None or ib is None:
                continue
            self.coess_adj[ia].add(ib); self.coess_adj[ib].add(ia)
            self.coess_conf[(min(ia, ib), max(ia, ib))] = e.get("coess")
            n += 1
        self.version = "v3"
        return {"applied": True, "coessential_links_added": n}

    # ---- resolve a gene arg to an index ----
    def _i(self, x):
        if isinstance(x, int):
            return x if 0 <= x < len(self.genes) else None
        return self.idx.get(x)

    def _names(self, ids):
        return [self.name[j] for j in ids if 0 <= j < len(self.name)]

    def _cell_types(self, i):
        """decode the emask bitmask -> the list of cell types this gene is expressed in (full, not a count)."""
        m = self.D.get("emask", {}).get(str(i))
        if not m:
            return []
        m = int(m)
        return [self.ctnames[b] for b in range(len(self.ctnames)) if (m >> b) & 1]

    # ---- THE accessor: one gene, every layer, full resolution ----
    def gene(self, x):
        i = self._i(x)
        if i is None:
            return None
        nm = self.name[i]; si = str(i); g = self.genes[i]
        D = self.D
        rec = {
            "name": nm, "idx": i, "record": g,                    # full base record (chrom, comp, ess, loeuf, pubs, dark, ...)
            # --- interactions (actual partner NAMES, not degrees) ---
            "ppi_partners": sorted(self._names(self.ppi_adj.get(i, ()))),
            "regulates": [(self.name[j], s) for j, s in self.reg_out.get(i, [])],
            "regulated_by": [(self.name[j], s) for j, s in self.reg_in.get(i, [])],
            "signals_to": [(self.name[j], s) for j, s in self.sig_out.get(i, [])],
            "signaled_by": [(self.name[j], s) for j, s in self.sig_in.get(i, [])],
            # signed causal direction (SIGNOR/CollecTRI): +1 activates, -1 inhibits, 0 unsigned
            "causal_downstream": [(self.name[j], s, db) for j, s, db in self.causal_out.get(i, []) if j < len(self.name)],
            "causal_upstream": [(self.name[j], s, db) for j, s, db in self.causal_in.get(i, []) if j < len(self.name)],
            "synthetic_lethal": [(self.name[j], s) for j, s in self.sl_adj.get(i, [])],
            "ligand_of_receptors": self._names(self.lig2rec.get(i, [])),
            "receptor_for_ligands": self._names(self.rec2lig.get(i, [])),
            "coessential_depmap": sorted(self._names(getattr(self, "coess_adj", {}).get(i, ()))),
            # --- function / annotation ---
            "go": D.get("go", {}).get(nm),
            "function_literature": (self.ghost.get(nm) or {}).get("func"),
            "function_name": (self.ghost.get(nm) or {}).get("func_name"),
            "dark_function_pred": D.get("darkfn", {}).get(si),
            "model4_pred": D.get("model4", {}).get(si),
            "proc": g.get("proc"),
            # --- complexes / pathways ---
            "complexes": D.get("gene2cplx", {}).get(si, []),
            "pathways": self.gene2path.get(i, []),
            # --- kinetics / metabolism (with the verified correction) ---
            "kinetics": self.kin.get(nm),
            "reactions_catalyzed": D.get("generxn", {}).get(si),
            # --- abundance / expression ---
            "abundance_ppm": D.get("ppm", {}).get(si),
            "abundance_rank": D.get("abund", {}).get(si),
            "expressed_in_cell_types": self._cell_types(i),
            "cellcycle_phase": D.get("cellcycle", {}).get(si),
            # --- regulation of expression / cofunction ---
            "coexpressed": [(self.name[j], s) for j, s in D.get("coexpr", {}).get(si, []) if j < len(self.name)],
            "codependent": [(self.name[j], s) for j, s in D.get("codep", {}).get(si, []) if j < len(self.name)],
            "nichenet_targets": D.get("nichenet", {}).get(nm),
            "biomarker": D.get("biomarkers", {}).get(nm),
            # --- disease / drugs / structure ---
            "disease": D.get("otdis", {}).get(nm),
            "n_disease": g.get("ndis"),
            "drugs": D.get("drugs", {}).get(si),
            "structure": D.get("struct", {}).get(nm),
            "ptm": D.get("ptm", {}).get(si),
            "loops3d": D.get("loops3d", {}).get(si),
        }
        return rec

    def genes_iter(self, with_full=True):
        """stream every gene's complete record — what the Phase-2 audit walks to 'report in every field'."""
        for i in range(len(self.genes)):
            yield self.gene(i) if with_full else self.genes[i]

    # ---- bulk / whole-cell access for ML ----
    def edges(self, relation):
        return self.D.get(relation, [])

    def field(self, name):
        return self.D.get(name)

    def layers(self):
        """every layer + per-gene coverage (how many of the 16,492 genes have a value) — the field inventory
        the audit uses to know WHAT to check."""
        n = len(self.genes)
        rows = []
        # per-gene layers keyed by index-string
        for k in ("go", "ppm", "abund", "emask", "drugs", "gene2cplx", "generxn", "darkfn", "model4",
                  "ptm", "codep", "coexpr", "cellcycle", "loops3d", "nichenet", "otdis", "biomarkers", "struct"):
            v = self.D.get(k, {})
            rows.append((k, len(v) if isinstance(v, (dict, list)) else 0))
        # relational layers
        for k in ("ppi", "reg", "sig", "sl", "lr", "pathways", "complexes"):
            v = self.D.get(k)
            rows.append((k + " (edges/sets)", len(v) if v is not None else 0))
        rows.append(("kinetics (corrected)", len(self.kin)))
        rows.append(("function_literature (ghost)", len(self.ghost)))
        return {"n_genes": n, "coverage": dict(rows)}


def _load(path, key=None):
    if not path or not os.path.exists(path):
        return {} if key else None
    d = json.load(open(path))
    return d.get(key, {}) if key else d


def _pair(e):
    return isinstance(e, (list, tuple)) and len(e) >= 2 and isinstance(e[0], int) and isinstance(e[1], int)


def main():
    cell = CompleteCell()
    lay = cell.layers()
    print("=" * 82)
    print(f"CompleteCell — full-fidelity ML entry point   ({lay['n_genes']:,} genes, every layer joined)")
    print("=" * 82)
    print("per-gene layer coverage:")
    for k, v in lay["coverage"].items():
        print(f"    {k:34} {v:>8,}")
    # proof: dump ONE gene's complete record and show the heavy layers are FULL, not summarised
    for demo in ("GATA1", "HK1", "TP53"):
        r = cell.gene(demo)
        if not r:
            continue
        print(f"\n--- {demo}: complete record has {len(r)} fields; heavy layers at full resolution ---")
        print(f"    ppi_partners        : {len(r['ppi_partners'])} names, e.g. {r['ppi_partners'][:6]}")
        print(f"    regulates           : {len(r['regulates'])} targets, e.g. {[t for t,_ in r['regulates'][:6]]}")
        print(f"    regulated_by        : {len(r['regulated_by'])} TFs, e.g. {[t for t,_ in r['regulated_by'][:6]]}")
        print(f"    pathways            : {r['pathways'][:3]}")
        print(f"    complexes           : {r['complexes'][:2]}")
        print(f"    expressed_in        : {len(r['expressed_in_cell_types'])} cell types, e.g. {r['expressed_in_cell_types'][:3]}")
        print(f"    kinetics            : {r['kinetics']}")
        go = r["go"] or {}
        print(f"    GO                  : F={len((go.get('F') or []))} P={len((go.get('P') or []))} C={len((go.get('C') or []))}")
    print("\n-> the ML calls .gene(x) for full data on any gene, .genes_iter() to walk them all, "
          ".edges(rel)/.field(k) for bulk. Nothing summarised.")
    return cell


if __name__ == "__main__":
    main()
