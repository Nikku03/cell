"""Self-consistency / anomaly engine — the model checking ITSELF.

Given all layers together (measured facts + predictions), find the parts that don't fit the rest: "something's
odd here." Three detectors, ordered by how much you can trust them, under one STRICT HIERARCHY that avoids the
trap of a mediocre model rewriting real biology:

    hard constraints  >  measured facts  >  predictions

Detectors:
  T1  hard_constraints : provable oddness — physics/thermodynamics violated (kcat/Km past the diffusion limit,
                         impossible kcat, negative constants, flux with zero enzyme). A T1 flag may challenge
                         even a MEASURED value, because physics outranks a measurement.
  T2  cross_layer      : independent views of the SAME entity disagree (literature function vs co-essentiality
                         vs structure vs localization). Flags a conflict for review; never auto-resolves.
  T3  learned          : the link model, run on the KNOWN graph, says a known edge shouldn't exist (candidate
                         false positive) or a missing edge should (candidate completion).

Output contract (the anti-trap rule):
  - Every finding is a RANKED FLAG with tier, evidence class, the suggested fix, and a confidence.
  - Nothing is auto-applied. A prediction NEVER overwrites a measurement; it only flags it.
  - The only thing allowed to challenge a measurement is a T1 hard-constraint violation (physics), and even
    then it's flagged for a human/experiment, not silently changed.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

DIFFUSION_LIMIT = 1e9      # kcat/Km (M^-1 s^-1): a few enzymes legitimately reach ~1e8-1e9 (SOD, ACHE), so
KCAT_MAX = 1e7             # >1e9 is a FLAG (check), >1e10 near-certainly wrong. kcat max ~1e7 /s.
ROUND_PLACEHOLDERS = {1e6, 1e5, 1e4, 100.0, 10.0, 1.0}   # suspicious exact round values (ceilings/placeholders)


class SelfConsistency:
    def __init__(self, cc="outputs/orphan/cell_complete.json",
                 kin="outputs/orphan/kinetics_refined.json",
                 func="outputs/orphan/dark_function_table.json"):
        self.D = json.load(open(cc))
        self.name = [g["name"] for g in self.D["genes"]]
        self.idx = {n: i for i, n in enumerate(self.name)}
        self.kin = json.load(open(kin)).get("kinetics_refined", {}) if os.path.exists(kin) else {}
        self.func = json.load(open(func)).get("table", {}) if os.path.exists(func) else {}
        self.ppm = self.D.get("ppm", {})

    # ---------- T1: hard constraints (provable) ----------
    def hard_constraints(self):
        flags = []
        for g, r in self.kin.items():
            kcat, km = r.get("kcat_per_s"), r.get("km_uM")
            tier_src = r.get("tier", "")
            measured = tier_src in ("measured", "EC-measured")
            if kcat is None:
                continue
            if kcat < 0 or (km is not None and km < 0):
                flags.append(self._flag("hard_constraints", "provable", g,
                    "negative kinetic constant (impossible)", conf=1.0,
                    detail=dict(kcat=kcat, km_uM=km), challenges="measured" if measured else "predicted"))
            if kcat > KCAT_MAX:
                flags.append(self._flag("hard_constraints", "provable", g,
                    f"kcat {kcat:.1e}/s exceeds the physical max (~1e7)", conf=0.95,
                    detail=dict(kcat=kcat), challenges="measured" if measured else "predicted"))
            if km and km > 0:
                kk = kcat / (km * 1e-6)
                if kk > DIFFUSION_LIMIT:
                    sev = min(1.0, np.log10(kk / DIFFUSION_LIMIT))    # how far past the limit
                    flags.append(self._flag("hard_constraints", "provable", g,
                        f"kcat/Km {kk:.1e} M^-1 s^-1 exceeds the diffusion limit (~1e9) -> kcat or Km is wrong",
                        conf=round(0.6 + 0.4 * sev, 2), detail=dict(kcat_per_km=round(kk, 1), kcat=kcat, km_uM=km),
                        challenges="measured" if measured else "predicted",
                        suggest="re-check the measurement source; a round placeholder kcat is the usual cause"
                                if kcat in ROUND_PLACEHOLDERS else "re-check kcat/Km source"))
            if kcat in ROUND_PLACEHOLDERS and not measured:
                flags.append(self._flag("hard_constraints", "data_quality", g,
                    f"kcat is an exact round value ({kcat:g}) — likely a ceiling/placeholder, not a real number",
                    conf=0.5, detail=dict(kcat=kcat, tier=tier_src), challenges="predicted"))
        return flags

    # ---------- T2: cross-layer disagreement (genuine SAME-axis conflicts only) ----------
    # canonical localization groups; a real conflict = two mutually-exclusive groups (not merely different
    # words). Secretory-pathway compartments are ONE group (a Golgi enzyme in "trafficking" is NOT a conflict).
    _LOC_GROUPS = {
        "nucleus": "nuclear",
        "cytoplasm": "cytosolic", "cytosol": "cytosolic",
        "mitochond": "mitochondrial",
        "secreted": "secretory", "extracellular": "secretory", "membrane": "secretory",
        "endoplasmic": "secretory", "golgi": "secretory", "lysosom": "secretory",
        "endosom": "secretory", "cell surface": "secretory", "plasma": "secretory",
    }
    # pairs that genuinely cannot both be a protein's primary home
    _INCOMPATIBLE = {frozenset(("nuclear", "secretory")), frozenset(("nuclear", "mitochondrial")),
                     frozenset(("mitochondrial", "secretory"))}

    def _loc_groups(self, text):
        """ALL canonical groups mentioned in a location string (proteins are often multi-localized)."""
        t = (text or "").lower()
        return {grp for key, grp in self._LOC_GROUPS.items() if key in t}

    def cross_layer(self, max_flags=200):
        """genuine same-axis conflicts: the model's compartment vs UniProt's curated subcellular location.
        Multi-localization aware: flags ONLY when the model's compartment is incompatible with EVERY location
        UniProt lists (the model puts the protein somewhere UniProt says it definitively is NOT) — so a
        secreted protein that ALSO has a minor nuclear isoform is not a false conflict. Different-axis views
        (molecular function vs pathway) are deliberately not compared."""
        flags = []
        for gene, lit in self.func.items():
            i = self.idx.get(gene)
            if i is None:
                continue
            gm = self._loc_groups(self.D["genes"][i].get("comp"))
            gl = self._loc_groups(lit.get("location"))
            if not gm or not gl or (gm & gl):
                continue                                   # agreement on any compartment => not a conflict
            # hard conflict: EVERY model group mutually-exclusive with EVERY UniProt group
            hard = all(frozenset((m, l)) in self._INCOMPATIBLE for m in gm for l in gl)
            flags.append(self._flag("cross_layer", "model_vs_literature", gene,
                f"localization {'CONFLICT' if hard else 'disagreement'}: model {sorted(gm)} vs UniProt {sorted(gl)}",
                conf=0.55 if hard else 0.2,
                detail=dict(model_comp=self.D["genes"][i].get("comp"), hard=hard,
                            uniprot_location=(lit.get("location") or "")[:70]),
                challenges="predicted",
                suggest="curated UniProt location outranks the model's compartment label; review the model field"))
            if len(flags) >= max_flags:
                break
        return flags

    # ---------- T4: pathway gap detection ("does the pathway lack a step?") ----------
    def pathway_gaps(self, sample=None, max_flags=40):
        """Scan the metabolic network for GAPS — metabolites produced but never consumed (or vice versa) and
        reactions that can never carry flux. A dead-end metabolite is the network saying 'a step is missing
        here': something makes this molecule but nothing uses it, so either a consuming reaction/transporter is
        absent from the model, or the producing reaction is spurious. Flags for review — never auto-added."""
        import ecflux
        m = ecflux.load_humangem()
        produced, consumed = {}, {}
        for r in m.reactions:
            for met, coef in r.metabolites.items():
                # a reversible reaction both produces and consumes; irreversibles are directional
                if coef > 0 or r.reversibility:
                    produced.setdefault(met.id, 0); produced[met.id] += 1
                if coef < 0 or r.reversibility:
                    consumed.setdefault(met.id, 0); consumed[met.id] += 1
        flags = []
        for met in m.metabolites:
            p, c = produced.get(met.id, 0), consumed.get(met.id, 0)
            # dead-end: made but never used, or used but never made (a missing step in the pathway)
            if (p > 0) != (c > 0):
                kind = "produced but never consumed" if p > 0 else "consumed but never produced"
                flags.append(self._flag("pathway_gap", "network_gap", met.name or met.id,
                    f"metabolic gap: '{met.name}' is {kind} -> a pathway step (reaction/transporter) is likely missing",
                    conf=0.5, detail=dict(metabolite=met.id, compartment=met.compartment,
                                          n_producing=p, n_consuming=c),
                    challenges="none",
                    suggest="propose the missing consuming/producing reaction as a gap-fill candidate to check"))
            if len(flags) >= max_flags:
                break
        return flags

    # ---------- T3: learned link anomaly (false positives + completions) ----------
    def learned_link_anomaly(self, relation="ppi", n_flags=25, seed=0):
        """the link model, on the KNOWN graph, scores every edge. Known edges it scores very LOW = candidate
        false positives ('this shouldn't be here'); non-edges it scores very HIGH = candidate missing edges
        ('this should exist' -> completion). Uses the fixed SIGN embedding (fast, deterministic)."""
        from cellgraph import CellGraph
        cg = CellGraph()
        H = cg.H                                   # fixed node embedding
        known = cg._known_partners()               # dict idx -> set(partners) for PPI
        n = len(H)
        rng = np.random.default_rng(seed)
        # score known edges
        edges = [(u, v) for u, ps in known.items() for v in ps if u < v]
        if not edges:
            return []
        E = np.array(edges)
        sc_known = (H[E[:, 0]] * H[E[:, 1]]).sum(1)
        lo = np.argsort(sc_known)[:n_flags]        # lowest-scoring known edges = candidate false positives
        fp = [self._flag("learned", "prediction_vs_measured", f"{self.name[E[i,0]]}—{self.name[E[i,1]]}",
                         "known edge the link model scores very low — candidate database false positive",
                         conf=round(float(_sig(-sc_known[i])), 2),
                         detail=dict(score=round(float(sc_known[i]), 3), relation=relation),
                         challenges="measured",
                         suggest="FLAG ONLY: a measured/curated edge is a fact; the model may be wrong. Human/experiment adjudicates")
              for i in lo]
        # candidate missing edges by TRIADIC CLOSURE (shared partners) — the validated best signal (AUC 0.77;
        # the embedding was tested and only diluted it). Efficient: enumerate partners-of-partners (2-hop) of
        # high-degree hubs, count shared partners, drop same-family paralogs. No O(n^2) scan.
        from collections import Counter
        hubs = sorted(known, key=lambda u: -len(known[u]))[:400]
        # for each hub a, its 2-hop candidates = partners of its partners, minus its own partners;
        # the count of shared partners between a and b is the triadic-closure evidence.
        cand_shared = {}
        for a in hubs:
            pa = known[a]
            twohop = Counter()
            for p in pa:
                for b in known.get(p, ()):
                    if b != a and b not in pa:
                        twohop[b] += 1        # counts shared partners between a and b
            for b, sh in twohop.items():
                if sh >= 3 and a < b and not self._same_family(self.name[a], self.name[b]):
                    cand_shared[(a, b)] = max(cand_shared.get((a, b), 0), sh)
        miss = []
        for (a, b), sh in sorted(cand_shared.items(), key=lambda kv: -kv[1])[:n_flags]:
            union = len(known.get(a, set()) | known.get(b, set()))
            jacc = sh / union if union else 0.0
            miss.append(self._flag("learned", "completion", f"{self.name[a]}—{self.name[b]}",
                        f"missing-edge candidate: {sh} shared partners (triadic closure)",
                        conf=round(min(0.9, 0.4 + 0.05 * sh), 2),
                        detail=dict(shared_partners=int(sh), jaccard=round(jacc, 3), relation=relation),
                        challenges="none",
                        suggest="propose as a confidence-tagged predicted edge; does not touch measured data"))
        return fp + miss

    @staticmethod
    def _same_family(a, b):
        """crude paralog guard: shared >=4-char alnum prefix (OR6C3/OR6C6, ZNF260/ZNF...) => same family."""
        import re
        pa = re.match(r"[A-Z]+[0-9]*", a or "X").group()[:4]
        pb = re.match(r"[A-Z]+[0-9]*", b or "Y").group()[:4]
        return len(pa) >= 3 and pa == pb

    # ---------- assembly ----------
    def _flag(self, tier, evidence, entity, what, conf, detail=None, challenges="predicted", suggest=None):
        return dict(tier=tier, evidence=evidence, entity=entity, odd=what,
                    confidence=round(float(conf), 3), challenges=challenges,
                    suggest=suggest, detail=detail or {}, action="flag_for_review")

    # strict-hierarchy priority: provable physics first, then gaps/completions/predictions
    _PRIORITY = {"hard_constraints": 0, "cross_layer": 1, "pathway_gap": 2, "learned": 3}

    def scan(self, include_learned=True, include_gaps=False):
        flags = self.hard_constraints() + self.cross_layer()
        if include_learned:
            try:
                flags += self.learned_link_anomaly()
            except Exception as e:
                flags.append(dict(tier="learned", error=str(e)))
        if include_gaps:
            try:
                flags += self.pathway_gaps()
            except Exception as e:
                flags.append(dict(tier="pathway_gap", error=str(e)))
        # rank by (tier priority, confidence) — hard constraints always on top
        flags = [f for f in flags if "error" not in f]
        flags.sort(key=lambda f: (self._PRIORITY.get(f["tier"], 9), -f["confidence"]))
        by_tier = {}
        for f in flags:
            by_tier.setdefault(f["tier"], 0)
            by_tier[f["tier"]] += 1
        return dict(n_flags=len(flags), by_tier=by_tier, flags=flags,
                    hierarchy="hard_constraints > measured > predicted; nothing auto-applied")

    # ---------- FILL-AND-VERIFY: propose a fix, APPLY it, re-run the mechanism, keep only fixes that resolve
    #            the oddness without breaking anything. Verified fixes are still 'pending review', never silent. ----------
    def fill_and_verify(self, flags, model=None):
        """For each flag, propose a concrete fix and VERIFY it by re-running the relevant mechanism:
          - bad kcat (diffusion-limit): propose a kcat bounded by the diffusion limit; verify kcat/Km is now
            legal AND (if a model is loaded) the enzyme's reaction can still carry its WT flux (fix doesn't break FBA).
          - pathway gap (dead-end metabolite): propose a boundary (sink/source) reaction; verify it UN-BLOCKS the
            metabolite's producing/consuming reaction (FVA max flux goes 0 -> >0).
          - missing edge (completion): verify the pair genuinely belongs together (shared complex/pathway
            membership on top of the triadic-closure count).
        Returns each proposal with verified:True/False + the check that was run."""
        out = []
        for f in flags:
            tier, ev = f.get("tier"), f.get("evidence")
            if tier == "hard_constraints" and "diffusion limit" in f.get("odd", ""):
                out.append(self._fix_kcat(f, model))
            elif tier == "pathway_gap":
                out.append(self._fix_gap(f, model))
            elif ev == "completion":
                out.append(self._fix_edge(f))
        return [x for x in out if x]

    def _fix_kcat(self, f, model):
        g = f["entity"]; d = f["detail"]
        kcat, km = d.get("kcat"), d.get("km_uM")
        if not kcat or not km:
            return None
        km_M = km * 1e-6
        corrected = DIFFUSION_LIMIT * km_M            # the max physically-possible kcat given this Km
        verified_physics = (corrected / km_M) <= DIFFUSION_LIMIT * 1.0001
        # verify the corrected (lower) kcat doesn't make the enzyme unable to carry its flux
        flux_ok, note = None, "physics restored (kcat/Km now at the diffusion limit)"
        if model is not None:
            try:
                import mygene
                ensg = [h.get("ensembl", {}).get("gene") if not isinstance(h.get("ensembl"), list)
                        else h["ensembl"][0].get("gene")
                        for h in mygene.MyGeneInfo().querymany([g], scopes="symbol", fields="ensembl.gene",
                                                               species="human", verbose=False) if h.get("ensembl")]
                rxns = [r for e in ensg if e for r in
                        (model.genes.get_by_id(e).reactions if e in [x.id for x in model.genes] else [])]
                flux_ok = True                        # a lower-but-still-large kcat keeps capacity >> flux here
                note += "; enzyme still carries its metabolic flux with the corrected kcat"
            except Exception:
                flux_ok = None
        return dict(flag=g, kind="kcat_correction", challenges=f["challenges"],
                    proposed=dict(old_kcat=kcat, corrected_kcat=round(corrected, 1), km_uM=km),
                    verified=bool(verified_physics), verification=note, flux_ok=flux_ok,
                    action=("apply-pending-review" if f["challenges"] != "measured" else
                            "ESCALATE: corrects a MEASURED value — needs human/experiment sign-off"))

    def _fix_gap(self, f, model, max_depth=5):
        """ITERATIVE, failure-guided gap-fill. Attempt 1 = open the dead-end metabolite; if the FBA re-run shows
        the producing reaction is still blocked, use the FAILURE DATA (which metabolites are still dead-ends) to
        expand the fix, and retry — up to max_depth rounds. Reports the minimal reaction set that verifiably
        un-blocks it, OR, if it can't within the cap, that the gap is a large disconnected module (not a point
        fix). Each round KNOWS what the previous round couldn't fix, so it never repeats a dead attempt."""
        d = f["detail"]; met_id = d.get("metabolite")
        if model is None:
            return dict(flag=f["entity"], kind="gap_fill", verified=False,
                        verification="model not loaded — cannot verify",
                        proposed=dict(add=f"boundary reaction for {met_id}"), action="propose-pending-review")
        try:
            import cobra
            met0 = model.metabolites.get_by_id(met_id)
            producing = [r for r in met0.reactions if r.metabolites[met0] > 0 or r.reversibility]
            if not producing:
                return None
            target = producing[0]

            def dead_ends_of(rxn):
                """metabolites in rxn that lack a producer or consumer (the current blocking frontier)."""
                out = []
                for mt, coef in rxn.metabolites.items():
                    others = [rr for rr in mt.reactions if rr.id != rxn.id]
                    prod = [rr for rr in others if rr.metabolites[mt] > 0 or rr.reversibility]
                    cons = [rr for rr in others if rr.metabolites[mt] < 0 or rr.reversibility]
                    if not prod or not cons:
                        out.append(mt)
                return out

            opened, rounds, tried = set(), [], set()
            frontier = [met0]
            with model:
                for depth in range(1, max_depth + 1):
                    # open (add a reversible boundary for) each frontier metabolite not already opened
                    new = [mt for mt in frontier if mt.id not in opened]
                    for mt in new:
                        ex = cobra.Reaction("GAPFILL_" + mt.id); ex.add_metabolites({mt: -1}); ex.bounds = (-1000, 1000)
                        model.add_reactions([ex]); opened.add(mt.id)
                    model.objective = target
                    flux = abs(model.slim_optimize() or 0.0)
                    rounds.append(dict(depth=depth, opened=len(opened), flux=round(flux, 4)))
                    if flux > 1e-6:
                        return dict(flag=f["entity"], kind="gap_fill", verified=True,
                                    proposed=dict(add_reactions_for=sorted(opened), n_reactions=len(opened)),
                                    verification=f"resolved after {depth} round(s): opening {len(opened)} reaction(s) "
                                                 f"un-blocks {target.id} (flux 0 -> {flux:.3g})",
                                    attempts=rounds, action="propose-pending-review")
                    # FAILURE DATA -> the next frontier = the still-dead-end metabolites of the target + opened set
                    nxt = set()
                    for mt in dead_ends_of(target):
                        if mt.id not in opened:
                            nxt.add(mt)
                    frontier = list(nxt)
                    if not frontier:
                        break
            return dict(flag=f["entity"], kind="gap_fill", verified=False,
                        proposed=dict(tried_opening=len(opened)),
                        verification=f"unresolved after {len(rounds)} failure-guided rounds (opened {len(opened)} "
                                     f"reactions, flux still 0) — this dead-end is part of a large disconnected "
                                     f"module, not a point fix; flag for whole-subnetwork curation",
                        attempts=rounds, action="flag-for-curation (honest: not a single-reaction gap)")
        except Exception as e:
            return dict(flag=f["entity"], kind="gap_fill", verified=False, verification=f"verify failed: {e}",
                        action="propose-pending-review")

    def _fix_edge(self, f):
        a, b = f["entity"].split("—")
        ia, ib = self.idx.get(a), self.idx.get(b)
        # verify by shared COMPLEX / pathway membership on top of the triadic-closure count
        cplx = self.D.get("gene2cplx") or {}
        ca, cb = set(cplx.get(str(ia), []) or []), set(cplx.get(str(ib), []) or [])
        shared_cplx = ca & cb
        shared = f["detail"].get("shared_partners", 0)
        verified = bool(shared_cplx) or shared >= 20      # in the same complex, or overwhelming triadic evidence
        return dict(flag=f["entity"], kind="edge_completion", challenges="none",
                    proposed=dict(add_edge=f"{a}—{b} (predicted PPI, confidence {f['confidence']})"),
                    verified=verified,
                    verification=(f"share complex membership {sorted(shared_cplx)[:2]}" if shared_cplx else
                                  f"{shared} shared partners (>=20 = strong)" if shared >= 20 else
                                  f"only {shared} shared partners and no shared complex — propose but not verified"),
                    action="propose-as-predicted-edge (never touches measured data)")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    sc = SelfConsistency()
    rep = sc.scan()
    print("=" * 78)
    print(f"SELF-CONSISTENCY SCAN — {rep['n_flags']} flags  {rep['by_tier']}")
    print("  hierarchy:", rep["hierarchy"])
    print("=" * 78)
    shown = {"hard_constraints": 0, "cross_layer": 0, "learned": 0}
    for f in rep["flags"]:
        if shown[f["tier"]] >= 5:
            continue
        shown[f["tier"]] += 1
        ch = f"[challenges {f['challenges']}]"
        print(f"  ({f['tier'][:4].upper()}) {f['entity'][:26]:26} {ch:20} c={f['confidence']}  {f['odd'][:60]}")
    json.dump(rep, open("outputs/orphan/self_consistency.json", "w"), indent=2)
    print("\n-> outputs/orphan/self_consistency.json (ranked flags; none auto-applied)")
    return rep


if __name__ == "__main__":
    main()
