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
        # regulatory edges as directed sets (fact layer)
        self._reg_out = {}; self._reg_in = {}
        for e in self.D.get("reg", []):
            self._reg_out.setdefault(e[0], set()).add(e[1]); self._reg_in.setdefault(e[1], set()).add(e[0])

    # ---- helpers ----
    def _a(self, name, tier, conf, **x):
        return dict(entity=name, tier=tier, confidence=round(float(max(0.0, min(1.0, conf))), 3), **x)

    def _abstain(self, q, why):
        return dict(question=q, abstain=True, reason=why)

    # ---- 1. binding ----
    def what_binds(self, protein, k=8):
        q = f"what does {protein} bind?"
        if protein not in self.idx:
            return self._abstain(q, f"{protein} not in model")
        i = self.idx[protein]; known = self.cg._known_partners()[i]
        facts = [self._a(self.name[j], "measured", 1.0, source="PPI database") for j in sorted(known)][:k]
        sim = self.H @ self.H[i]; preds = []
        for j in np.argsort(-sim):
            if j == i or j in known:
                continue
            c = float(sim[j])
            if c < MIN_CONF:
                break
            preds.append(self._a(self.name[j], "predicted", c, source="CellGraph link"))
            if len(preds) >= k:
                break
        return dict(question=q, measured=facts, predicted=preds,
                    note=f"{len(known)} known partners (fact) + top predicted novel bindings")

    # ---- 2. knockout -> downstream ----
    def knockout(self, protein, k=10):
        q = f"remove {protein} -> downstream effect?"
        eff = self.cg.knockout_effect(protein, k)
        if not eff:
            return self._abstain(q, "no downstream signal (not in model or no out-edges)")
        out = []
        for nm, d, v in eff:
            conf = min(1.0, abs(v)) if v is not None else 0.3
            out.append(self._a(nm, "predicted", conf, direction=d, source="CellGraph perturbation"))
        return dict(question=q, predicted=out,
                    note="direction (up/down) is calibrated (~0.81); magnitude is relative, not absolute")

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
    "mutation_effect":    dict(engine="ΔΔG predictor",                 validated="S669 r=0.41 (DDGun-tier)", tier="prediction (low per-call)"),
    "drug_interactions":  dict(engine="CellGraph polypharmacology",    validated="drug AUC 0.80",       tier="fact+prediction"),
    "regulates":          dict(engine="regulatory network",           validated="curated edges",       tier="fact"),
    "kcat":               dict(engine="tiered kinetics (CatPred)",     validated="3.3x, at noise floor", tier="fact or prediction"),
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
        ("kcat of HK1?", qa.kcat("HK1")),
        ("SOD1 A4V mutation effect?", qa.mutation_effect("SOD1", "P00441", 5, "A", "V")),
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
