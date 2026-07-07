"""CellQA — the whole-cell question-answering layer.

The goal ("map the whole cell, miss nothing, answer anything") made concrete: one interface that answers the
question types the goal names, and — critically — tags every answer as **fact vs prediction** with a
**confidence** and **provenance**, and **abstains** when it genuinely can't. Measured data is returned as fact
(confidence 1.0); unknowns are filled by the validated predictors with their calibrated confidence.

Question types (from the stated goal):
  1. what does protein X bind?              -> known PPI (fact) + predicted novel bindings (CellGraph)
  2. remove protein X -> downstream effect  -> perturbation direction + confidence
  3. what happens when a mutation occurs    -> ΔΔG stability -> destabilization call (DDGun-tier)
  4. drug -> what else can it interact with -> known targets (fact) + predicted off-targets
  5. what regulates X / X regulates what    -> known regulatory edges (fact) + predicted
  6. how fast is enzyme X (kcat)            -> measured (fact) or CatPred (prediction), tiered

Each answer: {entity, tier: measured|predicted|inferred, confidence 0-1, provenance, ...}.
"""
import os, sys, json, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cellgraph import CellGraph, perturb_downstream

MIN_CONF = 0.15   # below this -> abstain


class CellQA:
    def __init__(self):
        self.cg = CellGraph()
        self.D, self.idx, self.name, self.H = self.cg.D, self.cg.idx, self.cg.name, self.cg.H
        self._ddg = None
        self._kin = None
        self._ctc = None
        # regulatory edges as directed sets (fact layer)
        self._reg_out = {}; self._reg_in = {}
        for e in self.D.get("reg", []):
            self._reg_out.setdefault(e[0], set()).add(e[1]); self._reg_in.setdefault(e[1], set()).add(e[0])

    # ---- helpers ----
    def _a(self, name, tier, conf, **x):
        return dict(entity=name, tier=tier, confidence=round(float(max(0.0, min(1.0, conf))), 3), **x)

    def _abstain(self, q, why):
        return dict(question=q, abstain=True, reason=why)

    def _conditioner(self):
        """lazy cell-type conditioner (emask gate). Novel dimension: same question, cell-type-specific answer.
        The gate is validated cell-type-specific (markers land in their own lineage ~9x background); it gates
        answers by expression, it is not a predictive lift over the global model (see docs/CELLTYPE_CONDITIONING)."""
        if self._ctc is None:
            try:
                from celltype_conditioning import CellTypeConditioner
                self._ctc = CellTypeConditioner()
            except Exception:
                self._ctc = False
        return self._ctc or None

    def _ct_gate(self, i, cell_type):
        """(conditioner, t, expressed_set) if cell_type resolves and gene i is expressed there; else a reason."""
        c = self._conditioner()
        if not c:
            return None, None, "cell-type conditioner unavailable"
        t = c.resolve_celltype(cell_type)
        if t is None:
            return c, None, f"cell type '{cell_type}' not found"
        if not c.is_expressed(i, t):
            return c, t, f"not expressed in {c.ct[t]}"
        return c, t, None

    # ---- 1. binding ----
    def what_binds(self, protein, k=8, cell_type=None):
        q = f"what does {protein} bind?" + (f" (in {cell_type})" if cell_type else "")
        if protein not in self.idx:
            return self._abstain(q, f"{protein} not in model")
        i = self.idx[protein]; known = self.cg._known_partners()[i]
        # cell-type conditioning: gate partners to those co-expressed in the cell type (interaction can occur there)
        gate = None
        if cell_type is not None:
            c, t, why = self._ct_gate(i, cell_type)
            if t is None or why:
                return self._abstain(q, f"cell-type conditioning: {why}")
            gate = lambda j: c.is_expressed(j, t)
        facts = [self._a(self.name[j], "measured", 1.0, source="PPI database")
                 for j in sorted(known) if (gate is None or gate(j))][:k]
        sim = self.H @ self.H[i]; preds = []
        for j in np.argsort(-sim):
            if j == i or j in known or (gate is not None and not gate(j)):
                continue
            c2 = float(sim[j])
            if c2 < MIN_CONF:
                break
            preds.append(self._a(self.name[j], "predicted", c2, source="CellGraph link"))
            if len(preds) >= k:
                break
        note = (f"{len(known)} known partners (fact) + top predicted novel bindings"
                if cell_type is None else
                f"conditioned on {cell_type}: only partners co-expressed there are active (emask gate)")
        return dict(question=q, measured=facts, predicted=preds, cell_type=cell_type, note=note)

    # ---- 2. knockout -> downstream ----
    def knockout(self, protein, k=10, cell_type=None):
        q = f"remove {protein} -> downstream effect?" + (f" (in {cell_type})" if cell_type else "")
        eff = self.cg.knockout_effect(protein, k if cell_type is None else max(k, 40))
        if not eff:
            return self._abstain(q, "no downstream signal (not in model or no out-edges)")
        gate = None
        if cell_type is not None:
            i = self.idx.get(protein)
            c, t, why = self._ct_gate(i, cell_type) if i is not None else (None, None, "not in model")
            if t is None or why:
                return self._abstain(q, f"cell-type conditioning: {why}")
            gate = lambda nm: (nm in self.idx) and c.is_expressed(self.idx[nm], t)
        out = []
        for nm, d, v in eff:
            if gate is not None and not gate(nm):
                continue
            conf = min(1.0, abs(v)) if v is not None else 0.3
            out.append(self._a(nm, "predicted", conf, direction=d, source="CellGraph perturbation"))
            if len(out) >= k:
                break
        note = ("direction (up/down) is calibrated (~0.81); magnitude is relative, not absolute"
                if cell_type is None else
                f"conditioned on {cell_type}: only downstream genes expressed there are shown (emask gate)")
        return dict(question=q, predicted=out, cell_type=cell_type, note=note)

    # ---- 3. mutation -> effect ----
    def mutation_effect(self, gene, uniprot, pos, wt, mut):
        q = f"what happens when {gene} {wt}{pos}{mut} occurs?"
        if self._ddg is None:
            from ddg_predictor import DDGPredictor
            mp = "outputs/orphan/ddg_model.pkl"
            self._ddg = DDGPredictor(mp) if os.path.exists(mp) else False
        if not self._ddg:
            return self._abstain(q, "ΔΔG model not available")
        try:
            pdb = self._ddg.alphafold_pdb(uniprot)
            ddg, used = self._ddg.predict_from_structure(pdb, "A", pos, wt, mut)
        except Exception as e:
            return self._abstain(q, f"structure/prediction failed: {e}")
        # DDGun-tier reliability: only large |ΔΔG| is trustworthy per-call
        conf = min(0.6, abs(ddg) / 8.0)      # capped — honest about per-call noise
        call = "destabilizing" if ddg > 1.0 else ("stabilizing" if ddg < -1.0 else "near-neutral")
        return dict(question=q, ddg_kcal_mol=round(float(ddg), 2), call=call, tier="predicted",
                    confidence=round(conf, 3), provenance="ΔΔG predictor (DDGun-tier)",
                    caveat="per-call noisy at r~0.4; trust the ranking of strong destabilizers, not single small calls")

    # ---- 4. drug -> interactions ----
    def drug_interactions(self, drug, k=8):
        q = f"{drug} -> what can it interact with?"
        known = {int(gi) for gi, lst in self.D.get("drugs", {}).items()
                 if any(str(drug).lower() == d.get("d", "").lower() for d in lst)}
        if not known:
            return self._abstain(q, f"{drug} not in the drug-target table")
        facts = [self._a(self.name[j], "measured", 1.0, source="drug-target database")
                 for j in sorted(known) if j < len(self.name)][:k]
        centroid = self.H[sorted(known)].mean(0); sim = self.H @ centroid; preds = []
        for j in np.argsort(-sim):
            if j in known:
                continue
            c = float(sim[j])
            if c < MIN_CONF:
                break
            preds.append(self._a(self.name[j], "predicted", c, source="CellGraph polypharmacology"))
            if len(preds) >= k:
                break
        return dict(question=q, measured=facts, predicted=preds,
                    note="known targets (fact) + predicted off-targets/polypharmacology")

    # ---- 4b. disease -> druggable target (blind selection by simulation) ----
    def disease_target(self, apex, readout, pathway=None, k=8):
        """Given a disease's driver (apex) + pathogenic readout (+ optional candidate pathway), SELECT the
        druggable bottleneck by perturbation-to-wild-type — the goal's 'given phenotype, identify the
        intervention'. Validated on 6 OOD diseases: recovers a known drug target in 5 (2.1x over random-label).
        Each returned target is tagged with its rescue score (confidence), direction, and druggability (fact)."""
        q = f"disease driven by {apex} (readout {readout}) -> druggable target?"
        try:
            from disease_target_pipeline import cascade_subgraph, propagate
        except Exception as e:
            return self._abstain(q, f"pipeline unavailable: {e}")
        ap = [self.idx[g] for g in apex if g in self.idx]
        ro = [self.idx[g] for g in readout if g in self.idx]
        if len(ap) < 1 or len(ro) < 2:
            return self._abstain(q, "need >=1 driver and >=2 readout genes in the model")
        keep, out = cascade_subgraph(self.D, ap, ro, Lf=4, Lb=3)
        v0 = propagate(out, ap)
        ro_active = [r for r in ro if v0.get(r, 0) > 0]
        if len(ro_active) < 2:
            return self._abstain(q, "driver does not activate the readout in the network (sim uninformative)")
        cand = [self.idx[g] for g in pathway if g in self.idx] if pathway else [n for n in keep if n not in ap]
        preds = []
        for n in cand:
            if n not in keep:
                continue
            ro_eff = [r for r in ro_active if r != n]
            vdis = propagate(out, [a for a in ap if a != n], removed={n})
            rescue = sum(1 for r in ro_eff if vdis.get(r, 0) <= 0 or vdis.get(r, 0) < 0.4 * v0.get(r, 0)) / max(1, len(ro_eff))
            if rescue <= 0:
                continue
            druggable = str(n) in self.D.get("drugs", {})
            preds.append(self._a(self.name[n], "predicted", rescue, direction="disable",
                                 druggable=druggable, source="disease->target simulation"))
        preds.sort(key=lambda a: -a["confidence"])
        return dict(question=q, predicted=preds[:k],
                    note=("ranked druggable rescuers (blind selection by simulation); rescue=confidence. "
                          "Limit: targets that are cytokine receptors absent from the directed reg/sig graph "
                          "(bind via PPI) can be missed."))

    # ---- 5. regulation ----
    def regulates(self, protein, k=12):
        q = f"what does {protein} regulate?"
        if protein not in self.idx:
            return self._abstain(q, f"{protein} not in model")
        i = self.idx[protein]
        targets = [self._a(self.name[j], "measured", 1.0, source="regulatory network")
                   for j in sorted(self._reg_out.get(i, set())) if j < len(self.name)][:k]
        return dict(question=q, measured=targets,
                    note=f"{len(self._reg_out.get(i, set()))} known regulatory targets (fact)")

    # ---- 6. kinetics (measured fact vs predicted) ----
    def kcat(self, enzyme, kinetics_path=None):
        q = f"how fast is {enzyme} (kcat)?"
        if self._kin is None:
            self._kin = self._load_kinetics(kinetics_path)
        rec = (self._kin or {}).get(enzyme)
        if not rec:
            return self._abstain(q, f"no kinetics for {enzyme}")
        tier = rec.get("tier", "")
        measured = tier in ("measured", "EC-measured")
        return dict(question=q, kcat_per_s=rec.get("kcat_per_s"), km_uM=rec.get("km_uM"),
                    tier=("measured" if measured else "predicted"), source_tier=tier,
                    confidence=(1.0 if measured else (0.5 if "catpred" in tier else 0.25)),
                    provenance=("literature measurement" if measured else
                                "CatPred prediction (~3.3x, at noise floor)" if "catpred" in tier else "EC/family prior"))

    # ---- 7. function of an unknown/dark protein (literature -> structure -> co-essentiality) ----
    def function(self, gene, uniprot=None):
        """What does this (possibly dark) protein DO? Up to three tiers, strongest first:
          - literature: UniProt curated FUNCTION (distilled from papers), tagged experimental/by-similarity;
          - structure : AlphaFold + Foldseek structural homology (molecular activity of the fold);
          - co-essentiality: the model's darkfn (Perturb-seq guilt-by-association → the pathway it acts in).
        Most 'dark' genes were only blank in OUR model — the literature tier fills that from curated papers;
        structure/co-essentiality carry the genuinely uncharacterized tail. Agreement across tiers = confidence."""
        q = f"what does {gene} do?"
        ans = dict(question=q)
        # (a) LITERATURE tier — UniProt curated function (from papers), the authoritative fill for model gaps
        lit = self._dark_function_table().get(gene)
        if lit and lit.get("function"):
            ans["literature"] = dict(function=lit["function"], name=lit.get("name"),
                                     evidence=lit.get("evidence"), location=lit.get("location"),
                                     keywords=lit.get("keywords"),
                                     provenance=f"UniProt curated function ({lit.get('evidence')})")
        # (b) model's co-essentiality dark-function call (from measured Perturb-seq)
        i = self.idx.get(gene)
        df = (self.D.get("darkfn") or {}).get(str(i)) if i is not None else None
        if df:
            ans["coessentiality"] = dict(prediction=df.get("pred"), confidence_tier=df.get("conf"),
                                         source=df.get("src"), evidence_genes=(df.get("ev") or [])[:5],
                                         provenance="Perturb-seq co-essentiality (pathway-level)")
        # (b) structure-based transfer (molecular activity of the fold)
        try:
            from foldseek_function import predict_function
            r = predict_function(gene, self_uniprot=uniprot)
            if not r.get("abstain"):
                ans["structure"] = dict(prediction=r["prediction"], confidence=r["confidence"],
                                        homolog=r.get("homolog"), homolog_seqId=r.get("homolog_seqId"),
                                        twilight_homologs=(r.get("twilight_homologs") or [])[:3],
                                        provenance=r["provenance"])
            else:
                ans["structure"] = dict(abstain=True, reason=r["reason"])
        except Exception as e:
            ans["structure"] = dict(abstain=True, reason=f"structure pipeline unavailable: {e}")
        if ("literature" not in ans and "coessentiality" not in ans
                and ans.get("structure", {}).get("abstain")):
            return self._abstain(q, "no curated function, no co-essentiality call, no confident structural homolog")
        ans["note"] = "literature (curated from papers) > structure (fold activity) > co-essentiality (pathway)"
        return ans

    def _dark_function_table(self):
        """lazy load of the UniProt curated-function table for dark genes (function extracted from papers)."""
        if not hasattr(self, "_dft"):
            self._dft = {}
            for p in ("outputs/orphan/dark_function_table.json",
                      os.path.join(os.path.dirname(__file__), "..", "outputs/orphan/dark_function_table.json")):
                if os.path.exists(p):
                    self._dft = json.load(open(p)).get("table", {}); break
        return self._dft

    def _load_kinetics(self, path):
        # committed/Drive path preferred; falls back to the session's Drive-read copy
        for p in ([path] if path else []) + ["outputs/orphan/kinetics_refined.json"]:
            if p and os.path.exists(p):
                d = json.load(open(p)); return d.get("kinetics_refined", d)
        TR = ("/root/.claude/projects/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/tool-results/"
              "mcp-Google_Drive-read_file_content-1783416612863.txt")
        if os.path.exists(TR):
            raw = re.sub(r'\\(?=[^"\\/bfnrtu])', '', json.load(open(TR))["fileContent"])
            return json.loads(raw)["kinetics_refined"]
        return {}


# validated accuracy per question type (from the recovery scorecard) — the "trust certificate" for each answer
COVERAGE = {
    "what_binds":         dict(engine="CellGraph link (R-GCN/hybrid)", validated="PPI link AUC 0.89", tier="fact+prediction"),
    "knockout":           dict(engine="CellGraph perturbation",        validated="direction acc 0.81",  tier="prediction"),
    "mutation_effect":    dict(engine="ΔΔG (ProteinMPNN+biophysical)",  validated="S669 r=0.47 (top-benchmark)", tier="prediction"),
    "drug_interactions":  dict(engine="CellGraph polypharmacology",    validated="drug AUC 0.80",       tier="fact+prediction"),
    "disease_target":     dict(engine="disease->target simulation",    validated="5/6 OOD diseases, 2.1x", tier="prediction"),
    "regulates":          dict(engine="regulatory network",           validated="curated edges",       tier="fact"),
    "kcat":               dict(engine="tiered kinetics (CatPred)",     validated="3.3x, at noise floor", tier="fact or prediction"),
    "function":           dict(engine="AlphaFold+Foldseek + co-essentiality", validated="5/5, twilight-zone", tier="prediction"),
    "cell_type_gate":     dict(engine="emask conditioning (200 types)", validated="marker specificity 9x bg", tier="conditioning"),
}


def coverage():
    """The map: which goal question-types are answerable, by what engine, at what validated accuracy."""
    print("=" * 78)
    print("CellQA COVERAGE — 'map the whole cell, answer anything' (fact vs prediction + confidence)")
    print("=" * 78)
    for q, c in COVERAGE.items():
        print(f"  {q:18} {c['tier']:26} {c['validated']:22} <- {c['engine']}")
    print("\n  every answer tagged: tier(measured=fact / predicted) + confidence + provenance; abstains when unsure")


def demo():
    qa = CellQA()
    for label, r in [
        ("what does TP53 bind?", qa.what_binds("TP53", 5)),
        ("remove SREBF2 -> downstream?", qa.knockout("SREBF2", 6)),
        ("SREBF2 regulates?", qa.regulates("SREBF2", 6)),
        ("Imatinib interactions?", qa.drug_interactions("Imatinib", 5)),
        ("psoriasis (IL23 driver) -> target?",
         qa.disease_target(["IL23A", "IL12B"], ["IL17A", "IL17F", "IL22", "CCL20", "IL21"],
                           pathway=["IL23A", "IL12B", "IL23R", "JAK2", "STAT3", "RORC", "IL17A", "STAT4"], k=5)),
        ("kcat of HK1?", qa.kcat("HK1")),
        ("SOD1 A4V mutation effect?", qa.mutation_effect("SOD1", "P00441", 5, "A", "V")),
        ("GATA1 binds — in erythroid?", qa.what_binds("GATA1", 5, cell_type="erythroid progenitor")),
        ("GATA1 binds — in T cell? (gated off)", qa.what_binds("GATA1", 5, cell_type="regulatory T cell")),
        ("unknown gene?", qa.what_binds("NOTAGENE")),
    ]:
        print(f"\nQ: {label}")
        if r.get("abstain"):
            print(f"   ABSTAIN — {r['reason']}"); continue
        for key in ("measured", "predicted"):
            if r.get(key):
                print(f"   {key:9}: " + ", ".join(f"{a['entity']}({a['confidence']})" for a in r[key][:5]))
        for k in ("ddg_kcal_mol", "call", "kcat_per_s", "tier", "confidence"):
            if k in r:
                print(f"   {k}: {r[k]}")


if __name__ == "__main__":
    coverage(); print(); demo()
