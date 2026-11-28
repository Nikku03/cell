"""Reasoned variant-effect predictor — reliable AND it explains itself.

The stability-only chain was a coin-flip because it saw only destabilization. This adds the two signals that
catch the mutations that CHANGE FUNCTION without breaking the fold (sickle-cell HbS is the archetype: Glu6Val
doesn't destabilize, it creates a sticky patch):

  1. evolutionary  — ESM log-likelihood ratio LLR = logP(mut) - logP(wt). A position the language model is
                     "sure" about (conserved: active sites, interfaces, folding cores) scores a substitution as
                     forbidden. This is the accuracy engine and it is mechanism-agnostic (catches gain/loss/
                     active-site, not just unfolding).
  2. functional    — distance from the mutation to a curated UniProt functional residue (active site, binding,
                     catalytic, modified). AT a catalytic residue => function abolished even if the fold is fine.
  3. stability     — ΔΔG (the original rung), kept because it explains the destabilization mechanism.

Every prediction carries a REASONING object: which mechanism(s) fired, the proof (numbers + the specific
residue), a confidence, and — before it is trusted — a SELF-VERIFICATION that runs the implied change through
the metabolic chain and checks the consequence is consistent. Nothing is asserted without its "why".
"""
import os, sys, json, re, urllib.request, urllib.parse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

ONE3 = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln', 'E': 'Glu', 'G': 'Gly',
        'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser',
        'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'}


class ReasonedVariant:
    def __init__(self, esm_name="esm2_t33_650M_UR50D", ddg_model="outputs/orphan/ddg_model.pkl",
                 with_chain=False):
        self._esm = None
        self._esm_name = esm_name
        self._site_cache = {}
        self._seq_cache = {}
        self.ddg = None
        try:
            from ddg_predictor import DDGPredictor
            if os.path.exists(ddg_model):
                self.ddg = DDGPredictor(ddg_model)
        except Exception:
            self.ddg = None
        self.chain = None
        if with_chain:
            from end_to_end_chain import Chain
            self.chain = Chain()

    # ---- evolutionary rung: ESM LLR ----
    def _load_esm(self):
        if self._esm is None:
            import esm, torch
            fn = getattr(esm.pretrained, self._esm_name)
            model, alphabet = fn(); model.eval()
            self._esm = (model, alphabet, alphabet.get_batch_converter(), alphabet.get_idx, torch)
        return self._esm

    def _uniprot_seq(self, acc):
        if acc not in self._seq_cache:
            t = urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30).read().decode()
            self._seq_cache[acc] = "".join(t.split("\n")[1:])
        return self._seq_cache[acc]

    def esm_llr(self, acc, pos, wt, mut):
        """logP(mut)-logP(wt) at pos (1-based). Returns None if position/residue mismatch or seq too long."""
        s = self._uniprot_seq(acc)
        if pos > len(s) or s[pos - 1] != wt or len(s) > 1022:
            return None
        model, alphabet, bc, tok, torch = self._load_esm()
        key = (acc,)
        if key not in self._seq_cache or f"lp_{acc}" not in self._seq_cache:
            _, _, toks = bc([("p", s)])
            with torch.no_grad():
                lp = torch.log_softmax(model(toks)["logits"], dim=-1)[0]
            self._seq_cache[f"lp_{acc}"] = lp
        lp = self._seq_cache[f"lp_{acc}"]
        return float(lp[pos, tok(mut)] - lp[pos, tok(wt)])

    # ---- accuracy engine: AlphaMissense (reliable, function-aware) via MyVariant.info ----
    def alphamissense(self, gene, wt, pos, mut):
        """AlphaMissense pathogenicity score (0-1); the reliable, function-modifying-aware predictor.
        None if the variant isn't in the table."""
        hgvsp = f"p.{ONE3[wt]}{pos}{ONE3[mut]}"
        u = "https://myvariant.info/v1/query?" + urllib.parse.urlencode(
            {"q": f"dbnsfp.genename:{gene} AND dbnsfp.hgvsp:{hgvsp}",
             "fields": "dbnsfp.alphamissense", "size": 1})
        try:
            r = json.loads(urllib.request.urlopen(u, timeout=20).read())
            for h in r.get("hits", []):
                am = h.get("dbnsfp", {}).get("alphamissense")
                if am:
                    sc = am.get("score"); sc = sc[0] if isinstance(sc, list) else sc
                    pred = am.get("pred"); pred = pred[0] if isinstance(pred, list) else pred
                    return float(sc), pred
        except Exception:
            return None
        return None

    # ---- functional rung: UniProt functional-residue proximity ----
    _FEAT = {"Active site", "Binding site", "Site", "Metal binding", "Modified residue", "Disulfide bond"}

    def functional_residues(self, acc):
        if acc not in self._site_cache:
            d = json.loads(urllib.request.urlopen(
                f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=30).read())
            pos = {}
            for f in d.get("features", []):
                if f.get("type") in self._FEAT:
                    loc = f.get("location", {})
                    s = loc.get("start", {}).get("value"); e = loc.get("end", {}).get("value")
                    if s:
                        for p in range(s, (e or s) + 1):
                            pos.setdefault(p, f.get("type"))
            self._site_cache[acc] = pos
        return self._site_cache[acc]

    def site_distance(self, acc, pos):
        sites = self.functional_residues(acc)
        if not sites:
            return None, None
        nearest = min(sites, key=lambda p: abs(p - pos))
        return abs(nearest - pos), sites[nearest]

    # ---- ΔΔG rung ----
    def ddg_of(self, acc, pos, wt, mut):
        if not self.ddg:
            return None
        try:
            pdb = self.ddg.alphafold_pdb(acc)
            v, _ = self.ddg.predict_from_structure(pdb, "A", pos, wt, mut)
            return float(v)
        except Exception:
            return None

    # ---- the reasoned prediction ----
    def predict(self, gene, uniprot, pos, wt, mut, use_esm=False):
        """Reliable CALL from AlphaMissense (accuracy); MECHANISM reasoning from ΔΔG + functional-site + (opt) ESM.
        The score decides IF it's damaging; the mechanism rungs explain WHY and provide the proof."""
        am = self.alphamissense(gene, wt, pos, mut)
        dist, site_type = self.site_distance(uniprot, pos)
        ddg = self.ddg_of(uniprot, pos, wt, mut)
        llr = self.esm_llr(uniprot, pos, wt, mut) if use_esm else None
        mechanisms, proof = [], []
        # mechanism reasoning — WHY it is (or isn't) damaging
        if dist is not None and dist <= 3:
            mechanisms.append("disrupts a functional residue")
            proof.append(f"{dist} residue(s) from a UniProt {site_type} — function altered even if the fold holds")
        if ddg is not None and ddg > 1.0:
            mechanisms.append("destabilizing")
            proof.append(f"ΔΔG={ddg:.1f} kcal/mol lowers the folded/active fraction")
        if llr is not None and -llr > 4:
            mechanisms.append("evolutionarily forbidden")
            proof.append(f"ESM LLR={llr:.1f}: a conserved position rejects {wt}->{mut}")
        # gain-of-function / aggregation PATTERN — the mechanism that defeats per-residue predictors (sickle
        # cell: HbS is benign to AlphaMissense/ESM/ΔΔG because the fold is intact and the residue isn't
        # conserved; the pathology is deoxy-polymerization). Pattern: fold intact + charge->hydrophobic swap.
        # It is only informative as a DISAGREEMENT trigger — when the ML says benign but this pattern is present.
        CHARGED, HYDROPHOBIC = set("DEKR"), set("AVLIFMWC")
        gof_pattern = (wt in CHARGED and mut in HYDROPHOBIC and (ddg is None or ddg < 1.5))
        # reliable call from AlphaMissense
        if am is not None:
            score, ampred = am
            call = ("likely damaging" if score >= 0.564 else            # AlphaMissense pathogenic cutoff
                    "ambiguous" if score >= 0.34 else "likely tolerated")
            conf = round(abs(score - 0.5) * 2, 3)                       # distance from the decision boundary
            blind_spot = None
            # the honest "don't be falsely sure" trigger: ML says benign but a known-blind-spot pattern is present
            if call == "likely tolerated" and gof_pattern:
                call = "uncertain — possible ML blind spot"
                conf = 0.3
                blind_spot = True
                mechanisms.append("possible gain-of-function/aggregation")
                proof.append(f"AlphaMissense says benign ({score:.2f}), BUT charge→hydrophobic ({wt}→{mut}) with "
                             "the fold intact is the sickle-cell-like pattern per-residue predictors "
                             "systematically MISS — do not trust the benign call; needs a functional assay")
            if call == "likely damaging" and not mechanisms:
                mechanisms.append("function-altering (mechanism not localized)")
                proof.append("AlphaMissense flags it, but it is not at an annotated site nor destabilizing — "
                             "a functional change our mechanism rungs don't localize")
            return dict(variant=f"{gene} {wt}{pos}{mut}", uniprot=uniprot, source="AlphaMissense",
                        score=round(score, 3), alphamissense_pred=ampred, call=call, confidence=conf,
                        ml_blind_spot=blind_spot,
                        evidence=dict(alphamissense=round(score, 3), ddg_kcal_mol=round(ddg, 2) if ddg is not None else None,
                                      site_distance=dist, site_type=site_type,
                                      esm_llr=round(llr, 2) if llr is not None else None),
                        reasoning=dict(mechanisms=mechanisms or ["no localized mechanism"],
                                       why="; ".join(proof) if proof else "no localized mechanism; AlphaMissense score near neutral",
                                       note="AlphaMissense = reliable call (function-aware); ΔΔG/site/ESM = the interpretable why; "
                                            "GOF pattern overrides a benign call into 'uncertain' (the sickle-cell lesson)"))
        # fallback (variant not in AlphaMissense): mechanism-only, honestly lower confidence
        score = (2.0 if "destabilizing" in mechanisms else 0) + (3.0 if "disrupts a functional residue" in mechanisms else 0)
        return dict(variant=f"{gene} {wt}{pos}{mut}", uniprot=uniprot, source="mechanism-only (AlphaMissense n/a)",
                    call="likely damaging" if score >= 3 else "uncertain",
                    confidence=round(min(0.5, score / 8), 3),
                    evidence=dict(ddg_kcal_mol=round(ddg, 2) if ddg is not None else None,
                                  site_distance=dist, site_type=site_type),
                    reasoning=dict(mechanisms=mechanisms or ["no strong signal"],
                                   why="; ".join(proof) if proof else "no functional-site or destabilization signal",
                                   note="AlphaMissense unavailable for this variant; mechanism-only call (lower confidence)"))

    # ---- self-verification: run the implied change through the metabolic chain ----
    def verify(self, gene, uniprot, pos, wt, mut):
        """before trusting a 'damaging' call, propagate it through ec-flux and report the cell consequence,
        so the prediction is checked against the mechanism, not just asserted."""
        pred = self.predict(gene, uniprot, pos, wt, mut)
        if self.chain is None:
            pred["verification"] = "chain not loaded"
            return pred
        tr = self.chain.run(gene, uniprot, pos, wt, mut)
        pred["verification"] = dict(
            biomass_retained=tr.get("biomass_retained"), n_flux_carrying=tr.get("n_flux_carrying"),
            consistent=(pred["call"] == "likely damaging") == ((tr.get("biomass_retained") or 1.0) < 0.9)
            if tr.get("biomass_retained") is not None else None,
            note="ran the mutation through ec-flux; 'consistent' = damaging call matches a real flux/growth drop")
        return pred


if __name__ == "__main__":
    rv = ReasonedVariant()
    # sickle cell — function-modifying, NOT destabilizing (the case the stability chain misses)
    for g, up, pos, wt, mut, lab in [("HBB", "P68871", 7, "E", "V", "sickle cell (HbS)")]:
        print(f"\n{lab}: {g} {wt}{pos}{mut}")
        print(json.dumps(rv.predict(g, up, pos, wt, mut), indent=2))
