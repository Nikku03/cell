"""Phase 2, restructured — the self-healing audit LOOP: 3 processes x 3 runs, until convergence.

The design (the user's, with the agreed refinements):

  A PROCESS = three runs on the CURRENT cell state
    Run 1  ANALYSE : load the full-fidelity cell with every locked fix from prior processes applied.
    Run 2  DETECT  : scan every field, list all gaps / errors / wrong figures — MINUS anything already locked
                     as fixed, accepted as not-an-error, or parked as needs-data (so we never re-flag them).
    Run 3  FIX     : take each new flag one by one -> propose a fix -> VERIFY it (simulate where there is a
                     mechanism, corroborate where there isn't) -> if it holds, LOCK it; if it fails, redo — and
                     each redo is FAILURE-GUIDED (it uses why the last attempt failed, so it never repeats a dead
                     attempt) up to 5 redos. Every flag ends in exactly one of THREE outcomes:
                        fixed        (verified + locked into the cell)
                        needs_data   (real, but unfixable here without new data / human review)
                        accepted     (not actually an error — a known blind spot / a measured fact)

  Then START OVER (next process): because fixing exposes new problems (added edges change triadic closure;
  un-blocking one metabolite reveals the next). Repeat until a process finds nothing new to fix (CONVERGED),
  capped at 3 processes.

  LOCKED LEDGER + REGRESSION CHECK: every locked fix is recorded; at the end of each process we RE-VERIFY every
  locked fix still holds under the new state, so a later process can never silently undo an earlier verified fix.

  Anti-trap throughout: physics > measured > predicted. A measured fact is never overwritten — at most escalated.

-> outputs/orphan/phase2_loop_report.json   (per-process: new flags, fixed / needs_data / accepted, regressions)
-> outputs/orphan/phase2_ledger.json        (the locked, verified fixes — the cell's ML-added layer)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

DIFFUSION_LIMIT = 1e9
KCAT_MAX = 1e7
ROUND_PLACEHOLDERS = {1e6, 1e5, 1e4, 1000.0, 100.0, 10.0, 1.0}
MEASURED = {"measured", "EC-measured"}
_LOC = {"nucleus": "nuclear", "nuclear": "nuclear",
        "cytoplasm": "cytosolic", "cytosol": "cytosolic", "cytosolic": "cytosolic",
        "mitochond": "mitochondrial",
        "secreted": "secretory", "extracellular": "secretory", "plasma membrane": "secretory",
        "endoplasmic": "secretory", "golgi": "secretory", "lysosom": "secretory",
        "endosom": "secretory", "cell surface": "secretory"}
_INCOMPAT = {frozenset(("nuclear", "secretory")), frozenset(("nuclear", "mitochondrial")),
             frozenset(("mitochondrial", "secretory"))}
# per-field detection bounds (LOGGED). Finite, all-verifiable fields (co-complex PPI, kcat, localization,
# function, coverage) are UNCAPPED so a process resolves them fully and the next process only sees what the
# fixes newly EXPOSED — that's what makes convergence meaningful. Only triadic completions (a large pool) are
# capped, by strongest evidence first.
CAP_COMPLETION = 300

# "meaningfully present" thresholds for the combiner's INDEPENDENT features (structural ones excluded), used by
# the completion independence guard. Keyed by name so adding features never breaks the indices.
_INDEP_THRESH = {"same_complex": 0.5, "coexpression": 1e-6, "codependency": 1e-6, "coessentiality": 0.2,
                 "string_score": 0.3, "embedding_cos": 0.5, "expr_corr": 0.3, "tahoe_corr": 0.3}


def _groups(t):
    t = (t or "").lower()
    return {g for k, g in _LOC.items() if k in t}


class Phase2Loop:
    def __init__(self, use_depmap=True):
        self.C = CompleteCell()
        self.C.apply_additions()                # start from the already-verified v2 co-complex additions? No —
        # we want the loop to REDERIVE from the base cell so the ledger is self-contained. Reset ppi to base:
        self._reset_base()
        self.ledger = []                        # locked, verified fixes (the cell's ML layer)
        self.locked_ids = set()
        self.accepted_ids = set()               # not-an-error (won't re-flag)
        self.needs_data_ids = set()             # real but unfixable here (won't re-flag; reported)
        self.dm = None
        if use_depmap:
            try:
                from phase3_depmap import DepMapEdges
                self.dm = DepMapEdges()
            except Exception as e:
                print("  (DepMap co-essentiality lens unavailable:", str(e)[:60], "- continuing without it)")
        # trained signal-combiner (calibrated P(edge) from all signals) — optional; used as a calibrated lens.
        # feature computation goes through a SignalCombiner sharing this cell + DepMap, so the loop AUTOMATICALLY
        # uses whatever features the model was trained with (6 core here; 7-8 with STRING/embeddings in Colab).
        import signal_combiner
        self.combiner = signal_combiner.load()
        self.sc = None
        if self.combiner:
            self.sc = signal_combiner.SignalCombiner(C=self.C, dm=self.dm)
            kept = self.combiner.get("features", [])            # the KEPT features the model was trained on
            allf = self.combiner.get("features_all", kept)
            if self.sc.features_list != allf:
                print(f"  (combiner off: available features {self.sc.features_list} != model's {allf})")
                self.combiner = None
            else:
                self.kept_idx = [self.sc.features_list.index(f) for f in kept]
                self.indep_names = self.combiner.get("informative_independent", [])
                print(f"  signal-combiner on (kept {kept}, threshold {round(self.combiner['threshold'],3)}) "
                      f"— calibrated lens; gates only on {self.indep_names}")

    def _combiner_prob(self, u, v):
        import numpy as _np
        f = self.combiner
        full = self.sc.features(u, v)
        x = _np.array([[full[i] for i in self.kept_idx]])       # subset to the kept features
        if f.get("uses_scaler"):
            x = f["scaler"].transform(x)
        return float(f["model"].predict_proba(x)[0, 1])

    def _independent_present(self, u, v):
        """an INFORMATIVE independent feature (one that survived the AUC cut) is meaningfully present — so a
        dropped/dead feature (e.g. tahoe_corr at 0.50) can no longer gate a completion."""
        full = self.sc.features(u, v); F = self.sc.features_list
        return any(full[F.index(nm)] >= _INDEP_THRESH.get(nm, 0.5)
                   for nm in self.indep_names if nm in _INDEP_THRESH)

    def _reset_base(self):
        """rebuild ppi adjacency from the base cell (no additions) so the loop's ledger is the whole ML layer."""
        from collections import defaultdict
        self.C.ppi_adj = defaultdict(set)
        for e in self.C.D.get("ppi", []):
            if isinstance(e[0], int) and isinstance(e[1], int):
                self.C.ppi_adj[e[0]].add(e[1]); self.C.ppi_adj[e[1]].add(e[0])
        self.base_ppi_pairs = sum(len(v) for v in self.C.ppi_adj.values()) // 2

    # ================= Run 2: DETECT (on the live state) =================
    def detect(self):
        C = self.C; flags = []
        seen = self.locked_ids | self.accepted_ids | self.needs_data_ids

        def add(kind, entity, field, detail, fixability, severity=0.5):
            fid = f"{kind}::{entity}"
            if fid in seen:
                return
            flags.append(dict(id=fid, kind=kind, entity=entity, field=field, detail=detail,
                              fixability=fixability, severity=severity))

        # --- kinetics: physics + too-low vs in-vivo floor + round placeholders ---
        for g, r in list(C.kin.items())[:100000]:
            kc, km = r.get("kcat_per_s"), r.get("km_uM")
            measured = r.get("tier") in MEASURED
            if kc is None:
                continue
            if kc > KCAT_MAX or kc < 0:
                add("kcat_physics", g, "kinetics", dict(kcat=kc, km_uM=km, measured=measured),
                    "accepted" if measured else "simulate", 0.95)
            elif km and km > 0 and kc / (km * 1e-6) > DIFFUSION_LIMIT:
                add("kcat_physics", g, "kinetics", dict(kcat=kc, km_uM=km, measured=measured),
                    "accepted" if measured else "simulate", 0.9)
            floor = r.get("davidi_kcat_max_per_s") or r.get("invivo_apparent_kcat_per_s")
            if floor and not measured and floor > kc * 2 and r.get("kcat_original") is None:
                add("kcat_low", g, "kinetics", dict(kcat=kc, km_uM=km, floor=float(floor)), "simulate", 0.7)

        # --- co-complex members with no PPI edge (UNCAPPED: finite + all verifiable by membership) ---
        for (u, v, cname) in self._complex_ppi_missing():
            add("complex_ppi", f"{C.name[u]}—{C.name[v]}", "ppi",
                dict(u=u, v=v, complex=cname), "corroborate", 0.85)

        # --- missing edges by triadic closure (CAPPED, logged: large pool, strongest evidence first) ---
        for (u, v, sh) in self._triadic(CAP_COMPLETION):
            add("completion", f"{C.name[u]}—{C.name[v]}", "ppi",
                dict(u=u, v=v, shared=sh), "corroborate", min(0.9, 0.4 + 0.04 * sh))

        # --- localization: model comp vs GO curated location (mutually exclusive) — UNCAPPED (finite) ---
        for nm, m, l, gm, gl in self._loc_conflicts():
            add("localization", nm, "compartment", dict(model=gm, go=gl, model_comp=m),
                "needs_review", 0.6)

        # --- dark genes with no function but a prediction available — UNCAPPED (finite) ---
        for nm, i in self._fillable_dark():
            add("function_fill", nm, "function", dict(idx=i), "corroborate", 0.5)

        # --- systemic coverage gaps (reported once, then accepted as needs-data) ---
        cov = self._coverage_gaps()
        for key, d in cov.items():
            add(f"coverage_{key}", key, "coverage", d, "needs_data", 0.4)

        flags.sort(key=lambda f: -f["severity"])
        return flags

    def _complex_ppi_missing(self, max_complex=25):
        C = self.C; adj = C.ppi_adj; out = []; seen = set()
        for cname, members in C.D.get("complexes", {}).items():
            mem = [m for m in members if isinstance(m, int) and 0 <= m < len(C.name)]
            if len(mem) < 2 or len(mem) > max_complex:
                continue
            for a in range(len(mem)):
                for b in range(a + 1, len(mem)):
                    u, v = mem[a], mem[b]
                    key = (min(u, v), max(u, v))
                    if key in seen or v in adj.get(u, ()):
                        continue
                    seen.add(key); out.append((key[0], key[1], cname))
        return out

    def _triadic(self, n_flags, n_hubs=500, cap=600, min_shared=4):
        from collections import Counter
        C = self.C; adj = C.ppi_adj
        hubs = sorted(adj, key=lambda u: -len(adj[u]))[:n_hubs]
        cand = {}
        for a in hubs:
            pa = adj[a]
            if len(pa) > cap:
                continue
            two = Counter()
            for p in pa:
                for b in adj.get(p, ()):
                    if b != a and b not in pa:
                        two[b] += 1
            for b, sh in two.items():
                if sh >= min_shared and a < b and not _same_family(C.name[a], C.name[b]):
                    cand[(a, b)] = max(cand.get((a, b), 0), sh)
        return [(a, b, sh) for (a, b), sh in sorted(cand.items(), key=lambda kv: -kv[1])[:n_flags]]

    def _loc_conflicts(self):
        C = self.C; out = []
        for nm, gg in C.D.get("go", {}).items():
            i = C.idx.get(nm)
            if i is None:
                continue
            gm = _groups(C.genes[i].get("comp")); gl = _groups(" ".join(gg.get("C", []) or []))
            if not gm or not gl or (gm & gl):
                continue
            if all(frozenset((m, l)) in _INCOMPAT for m in gm for l in gl):
                out.append((nm, C.genes[i].get("comp"), gg.get("C"), sorted(gm), sorted(gl)))
        return out

    def _fillable_dark(self):
        C = self.C; out = []
        for i, g in enumerate(C.genes):
            if not g.get("dark"):
                continue
            nm = g["name"]
            if C.D.get("go", {}).get(nm) or C.ghost.get(nm):
                continue
            if str(i) in C.D.get("darkfn", {}) or str(i) in C.D.get("model4", {}):
                out.append((nm, i))
        return out

    def _coverage_gaps(self):
        C = self.C; n = len(C.genes)
        cov = set()
        for members in C.D.get("pathways", {}).values():
            cov.update(m for m in members if isinstance(m, int))
        return {
            "pathway_membership": dict(covered=len(cov), total=n, pct=round(100 * len(cov) / n, 1),
                                       reason="only curated Reactome top-level sets loaded"),
            "single_cell_expression": dict(covered=len(C.D.get("emask", {})), total=n,
                                           pct=round(100 * len(C.D.get("emask", {})) / n, 1),
                                           reason="atlas coverage; needs more single-cell datasets"),
        }

    # ================= Run 3: FIX + VERIFY (failure-guided, 3 outcomes) =================
    def fix_and_verify(self, flag, max_redos=5):
        kind = flag["kind"]
        HANDLERS = {"kcat_physics": self._fx_kcat, "kcat_low": self._fx_kcat,
                    "complex_ppi": self._fx_complex, "completion": self._fx_completion,
                    "localization": self._fx_localization, "function_fill": self._fx_function}
        handler = self._fx_coverage if kind.startswith("coverage") else HANDLERS.get(kind)
        if handler is None:
            return self._outcome(flag, "needs_data", "no fix mechanism for this field", attempts=0)
        failures = []
        for attempt in range(1, max_redos + 1):
            res = handler(flag, failures, attempt)
            if res["status"] in ("fixed", "accepted", "needs_data"):
                res["attempts"] = attempt
                return res
            failures.append(res.get("why", "unspecified failure"))     # FAILURE-GUIDED: feed forward
            if not res.get("retry"):
                return self._outcome(flag, res.get("terminal", "needs_data"),
                                     res.get("why", "unfixable"), attempts=attempt)
        return self._outcome(flag, "needs_data",
                             f"unresolved after {max_redos} failure-guided redos: {failures[-1]}", attempts=max_redos)

    # -- kcat physics: measured -> accepted (fact); predicted -> cap at diffusion limit, verify physics --
    def _fx_kcat(self, flag, failures, attempt):
        if flag["kind"] == "kcat_low":
            return self._fx_kcat_low(flag, failures, attempt)
        d = flag["detail"]
        if d.get("measured"):
            return self._outcome(flag, "accepted",
                                 "measured kcat violates physics -> label noise; a fact is not overwritten (escalate to human)")
        kc, km = d["kcat"], d.get("km_uM")
        corrected = min(DIFFUSION_LIMIT * km * 1e-6, KCAT_MAX) if km else KCAT_MAX
        if corrected >= kc * 0.999:
            # the "correction" wouldn't actually lower kcat -> this enzyme sits AT the diffusion limit within
            # float tolerance (physically legal; some enzymes really are diffusion-limited). Not a fix.
            return self._outcome(flag, "accepted",
                                 "kcat/Km is at the diffusion limit within tolerance — physically legal, no change")
        ok = (not km) or corrected / (km * 1e-6) <= DIFFUSION_LIMIT * 1.0001
        if ok:
            self.C.kin[flag["entity"]] = {**self.C.kin.get(flag["entity"], {}),
                                          "kcat_per_s": round(corrected, 3), "kcat_original": kc,
                                          "kcat_source": "physics_cap"}
            return self._outcome(flag, "fixed", f"kcat {kc:.2g}->{corrected:.2g}/s; kcat/Km now <= diffusion limit",
                                 fix=dict(old=kc, new=round(corrected, 3)))
        return dict(status="retry", why="cap still violates physics", retry=False, terminal="needs_data")

    def _fx_kcat_low(self, flag, failures, attempt):
        d = flag["detail"]; kc, km, floor = d["kcat"], d.get("km_uM"), d["floor"]
        corrected = floor
        if km and km > 0:
            corrected = min(corrected, DIFFUSION_LIMIT * km * 1e-6)      # cap so raising it can't break physics
        if corrected <= kc:
            return self._outcome(flag, "accepted", "diffusion cap pulls the floor below current value; leave as is")
        if km and corrected / (km * 1e-6) > DIFFUSION_LIMIT * 1.0001:
            return dict(status="retry", why="raising to floor would break the diffusion limit", retry=False,
                        terminal="needs_data")
        self.C.kin[flag["entity"]] = {**self.C.kin.get(flag["entity"], {}),
                                      "kcat_per_s": round(corrected, 3), "kcat_original": kc,
                                      "kcat_source": "invivo_floor"}
        return self._outcome(flag, "fixed", f"predicted kcat {kc:.2g}->{corrected:.2g}/s (in-vivo floor)",
                             fix=dict(old=kc, new=round(corrected, 3)))

    # -- co-complex PPI: verified by curated complex membership -> add edge, lock --
    def _fx_complex(self, flag, failures, attempt):
        d = flag["detail"]; u, v = d["u"], d["v"]
        self.C.ppi_adj[u].add(v); self.C.ppi_adj[v].add(u)
        return self._outcome(flag, "fixed", f"co-members of curated complex '{d['complex'][:40]}' -> PPI edge added",
                             fix=dict(add_edge=[self.C.name[u], self.C.name[v]], relation="ppi"))

    # -- completion: FAILURE-GUIDED multi-lens corroboration. Each redo tries a different INDEPENDENT evidence
    #    source (independent of the PPI graph itself). Triadic closure is what SURFACED the candidate — it is NOT
    #    counted as its own corroboration (that would be circular: the graph confirming the graph). A candidate
    #    with no independent support is parked as needs_data, not locked. --
    def _fx_completion(self, flag, failures, attempt):
        C = self.C; d = flag["detail"]; u, v = d["u"], d["v"]; a, b = C.name[u], C.name[v]
        # calibrated path: if the trained combiner is loaded, lock when P(edge) clears the 0.9-precision threshold
        # AND an INDEPENDENT signal is actually present (guard: don't let the structural features close cliques
        # on their own — that would be the graph confirming itself). One-shot, so it runs on attempt 1 only.
        if self.combiner and attempt == 1:
            # independence guard: at least one INFORMATIVE independent signal (survived the AUC cut) must be
            # present, so neither the structural features nor a dead feature can lock an edge on their own.
            indep_present = self._independent_present(u, v)
            p = self._combiner_prob(u, v)
            if p >= self.combiner["threshold"] and indep_present:
                self.C.ppi_adj[u].add(v); self.C.ppi_adj[v].add(u)
                return self._outcome(flag, "fixed",
                                     f"signal-combiner P(edge)={p:.2f} (>= {self.combiner['threshold']}, ~0.9 precision) "
                                     f"WITH independent support; predicted PPI edge added",
                                     fix=dict(add_edge=[a, b], relation="ppi", lens="combiner", prob=round(p, 3)))
            # combiner didn't clear the bar -> fall through to the single-lens attempts below (still may corroborate)
        lenses = ["shared_complex", "coessential", "coexpression", "codependency"]
        if attempt > len(lenses):
            return self._outcome(flag, "needs_data",
                                 f"triadic candidate ({d.get('shared')} shared partners) with NO independent "
                                 "corroboration (complex/co-essentiality/co-expression/co-dependency) — kept as a "
                                 "predicted candidate, not locked")
        lens = lenses[attempt - 1]
        ok, why = self._lens(lens, u, v, a, b, d.get("shared", 0))
        if ok:
            self.C.ppi_adj[u].add(v); self.C.ppi_adj[v].add(u)
            return self._outcome(flag, "fixed", f"corroborated by {lens}: {why}; predicted PPI edge added",
                                 fix=dict(add_edge=[a, b], relation="ppi", lens=lens))
        return dict(status="retry", why=f"{lens} did not corroborate ({why})", retry=True)

    def _lens(self, lens, u, v, a, b, shared):
        C = self.C
        if lens == "shared_complex":
            ca = set(C.D.get("gene2cplx", {}).get(str(u), []) or [])
            cb = set(C.D.get("gene2cplx", {}).get(str(v), []) or [])
            s = ca & cb
            return (bool(s), f"share {sorted(s)[:1]}" if s else "no shared complex")
        if lens == "coessential":
            if not self.dm:
                return (False, "DepMap not loaded")
            ce = self.dm.coess(a, b)
            return (ce is not None and ce >= 0.3, f"coess={round(ce,3) if ce is not None else None}")
        if lens == "coexpression":
            part = {j for j, _ in C.D.get("coexpr", {}).get(str(u), [])}
            return (v in part, "co-expressed" if v in part else "not co-expressed")
        if lens == "codependency":
            part = {j for j, _ in C.D.get("codep", {}).get(str(u), [])}
            return (v in part, "co-dependent" if v in part else "not co-dependent")
        return (False, "unknown lens")

    # -- localization: GO curated location outranks a predicted comp, but comp may be measured -> needs review --
    def _fx_localization(self, flag, failures, attempt):
        d = flag["detail"]
        return self._outcome(flag, "needs_data",
                             f"model comp {d['model']} vs GO {d['go']}: GO curated location outranks, but the "
                             f"model compartment may itself be measured -> flag for human/source review, not auto-changed",
                             fix=dict(proposed_location=d["go"]))

    # -- function fill: corroborate the dark-gene prediction by evidence strength --
    def _fx_function(self, flag, failures, attempt):
        C = self.C; i = flag["detail"]["idx"]
        pred = C.D.get("darkfn", {}).get(str(i)) or C.D.get("model4", {}).get(str(i)) or {}
        n_ev = pred.get("n") or len(pred.get("ev", []) or [])
        conf = pred.get("conf")
        if conf == "high" and n_ev >= 8:
            return self._outcome(flag, "fixed",
                                 f"adopt predicted function '{(pred.get('pred') or '')[:40]}' "
                                 f"(conf {conf}, {n_ev} evidence genes)",
                                 fix=dict(function=pred.get("pred"), source=pred.get("src")))
        return self._outcome(flag, "needs_data",
                             f"prediction too weak to lock (conf {conf}, {n_ev} evidence) -> keep as candidate")

    def _fx_coverage(self, flag, failures, attempt):
        d = flag["detail"]
        return self._outcome(flag, "needs_data",
                             f"{d['pct']}% coverage ({d['covered']}/{d['total']}): {d['reason']} — needs new data, "
                             "not fixable by self-audit")

    def _outcome(self, flag, status, why, fix=None, attempts=1):
        return dict(id=flag["id"], kind=flag["kind"], entity=flag["entity"], field=flag["field"],
                    status=status, why=why, fix=fix, attempts=attempts)

    # ================= regression check =================
    def regression_check(self):
        """re-verify every locked fix still holds under the current state. A later process must not silently
        undo an earlier verified fix."""
        regressions = []
        for e in self.ledger:
            ok = True; why = ""
            if e["kind"] in ("complex_ppi", "completion"):
                a, b = e["entity"].split("—")
                ia, ib = self.C.idx.get(a), self.C.idx.get(b)
                ok = ib in self.C.ppi_adj.get(ia, set())
                why = "edge missing from graph" if not ok else ""
            elif e["kind"] in ("kcat_physics", "kcat_low"):
                r = self.C.kin.get(e["entity"], {}); kc, km = r.get("kcat_per_s"), r.get("km_uM")
                ok = not (km and km > 0 and kc and kc / (km * 1e-6) > DIFFUSION_LIMIT * 1.001)
                why = "kcat/Km back over the diffusion limit" if not ok else ""
            if not ok:
                regressions.append(dict(id=e["id"], why=why))
        return regressions

    # ================= the loop =================
    def loop(self, max_processes=3):
        processes = []
        for p in range(1, max_processes + 1):
            # Run 1: analyse — state already carries all locked fixes (they were applied in place)
            n_edges = sum(len(v) for v in self.C.ppi_adj.values()) // 2
            # Run 2: detect
            flags = self.detect()
            # Run 3: fix + verify (one by one, failure-guided)
            fixed = needs = accepted = 0; details = []
            for f in flags:
                out = self.fix_and_verify(f)
                if out["status"] == "fixed":
                    fixed += 1; self.locked_ids.add(out["id"])
                    self.ledger.append({**out, "process": p})
                elif out["status"] == "accepted":
                    accepted += 1; self.accepted_ids.add(out["id"])
                else:
                    needs += 1; self.needs_data_ids.add(out["id"])
                details.append(out)
            regressions = self.regression_check()
            processes.append(dict(process=p, ppi_edges_at_start=n_edges, new_flags=len(flags),
                                  fixed=fixed, needs_data=needs, accepted=accepted,
                                  regressions=regressions, sample=details[:8]))
            print(f"  process {p}: {len(flags):4d} new flags -> {fixed} fixed, {needs} needs-data, "
                  f"{accepted} accepted | regressions: {len(regressions)}")
            if fixed == 0:                      # CONVERGED: nothing new got fixed this process
                print(f"  converged at process {p} (no new fixes)")
                break
        return processes

    @staticmethod
    def _kind_of(fid):
        k = fid.split("::")[0]
        return "coverage" if k.startswith("coverage") else k

    def save(self, processes):
        by_kind = {}
        for e in self.ledger:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        needs_by_kind, acc_by_kind = {}, {}
        for fid in self.needs_data_ids:
            k = self._kind_of(fid); needs_by_kind[k] = needs_by_kind.get(k, 0) + 1
        for fid in self.accepted_ids:
            k = self._kind_of(fid); acc_by_kind[k] = acc_by_kind.get(k, 0) + 1
        report = dict(
            design="3 processes x {analyse, detect, fix-verify}; failure-guided redo (5); 3 outcomes; "
                   "locked ledger + regression check; stop on convergence",
            n_processes_run=len(processes),
            converged=processes[-1]["fixed"] == 0 if processes else False,
            total_locked_fixes=len(self.ledger), locked_by_kind=by_kind,
            total_needs_data=len(self.needs_data_ids), needs_data_by_kind=needs_by_kind,
            total_accepted=len(self.accepted_ids), accepted_by_kind=acc_by_kind,
            total_regressions=sum(len(p["regressions"]) for p in processes),
            base_ppi_pairs=self.base_ppi_pairs,
            final_ppi_pairs=sum(len(v) for v in self.C.ppi_adj.values()) // 2,
            processes=processes,
            hierarchy="physics > measured > predicted; measured facts never overwritten")
        json.dump(report, open("outputs/orphan/phase2_loop_report.json", "w"), indent=2, default=_np)
        json.dump(dict(meta=dict(rule="locked, verified fixes from the self-healing loop; facts untouched"),
                       ledger=self.ledger), open("outputs/orphan/phase2_ledger.json", "w"), default=_np)
        return report


def _same_family(a, b):
    import re
    pa = re.match(r"[A-Z]+[0-9]*", a or "X").group()[:4]
    pb = re.match(r"[A-Z]+[0-9]*", b or "Y").group()[:4]
    return len(pa) >= 3 and pa == pb


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def main():
    print("=" * 88)
    print("PHASE 2 — self-healing audit LOOP (3 processes x 3 runs, until convergence)")
    print("=" * 88)
    L = Phase2Loop()
    procs = L.loop()
    rep = L.save(procs)
    print("-" * 88)
    print(f"FIXED (locked, verified): {rep['total_locked_fixes']}  {rep['locked_by_kind']}")
    print(f"NEEDS-DATA (real, unfixable here): {rep['total_needs_data']}  {rep['needs_data_by_kind']}")
    print(f"ACCEPTED (not an error): {rep['total_accepted']}  {rep['accepted_by_kind']}")
    print(f"REGRESSIONS (locked fix later broke): {rep['total_regressions']}")
    print(f"PPI pairs: {rep['base_ppi_pairs']:,} -> {rep['final_ppi_pairs']:,}  "
          f"(+{rep['final_ppi_pairs']-rep['base_ppi_pairs']:,} verified edges)")
    print(f"converged: {rep['converged']} after {rep['n_processes_run']} process(es)  "
          f"(finite fields resolved in process 1; completions locked only on INDEPENDENT corroboration)")
    print("-> outputs/orphan/phase2_loop_report.json  +  phase2_ledger.json")
    return rep


if __name__ == "__main__":
    main()
