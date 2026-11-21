"""investigate — build a PREDICTED investigative dossier for genes we do NOT have measured surveillance for, the way
you'd profile a person from partial records. For each gene, assemble:

  1. RECORD      — the textbook/database record first (proc = job type, path = pathway, comp = compartment, litmine
                   PubMed papers). What is already on file.
  2. FAMILY      — the relatives: co-dependent + physically-interacting neighbours. If the family does X, the gene
                   probably does too (guilt-by-association). For a DARK gene this IS the function prediction.
  3. DESTINATION — where it goes (subcellular compartment).
  4. PATH        — the pathway it travels in (curated, or decoded from co-dependency).
  5. TIMING      — WHEN it acts, relative to a reference: its pathway tier (upstream→downstream) / metabolic step.
  6. INTERACTORS — the people it meets (STRING physical partners).
  7. SURVEILLANCE— measured removal-effect if in the debugger; else PREDICTED from measured neighbours (honest r).
  8. REQUIRED SUPPORT — reason the jobs its role IMPLIES must exist around it (needs fuel→a fuel stop, service→a
                   mechanic, carries cargo→a warehouse): translocation, folding, transport, energy, disposal.
  9. UNDERWORLD  — required roles that NO gene on file provides near it → candidate dark-gene assignments (matched by
                   capability: compartment + job + domains). The functions the city needs that only the unknown do.

Closest relatives to the measured set FIRST (that is where prediction is reliable); accuracy is VALIDATED per line on
held-out annotated genes and honestly degrades with distance. -> outputs/orphan/investigate.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


# ---- reasoned REQUIRED-SUPPORT rules: what a job in a place MUST have around it (textbook cell biology) ----
def required_support(comp, proc, tf, is_enzyme, has_complex):
    """given destination + job, the supporting functions the cell MUST provide — reasoned, not looked up."""
    R = []
    c = (comp or "").lower(); p = (proc or "").lower()
    if any(x in c for x in ("extracellular", "plasma membrane", "membrane", "secret", "er", "golgi")):
        R += [("ER translocation", "SEC61 translocon — a secreted/membrane protein must be threaded into the ER"),
              ("signal-peptide cleavage", "signal peptidase (SPCS) removes the targeting sequence"),
              ("N-glycosylation", "OST transfers glycans in the ER lumen"),
              ("vesicle trafficking", "COPII/COPI + SNAREs carry it ER→Golgi→surface")]
    if "mitochond" in c:
        R += [("mitochondrial import", "TOMM/TIMM translocases pull the protein across both membranes"),
              ("presequence cleavage", "MPP (PMPCA/PMPCB) clips the mito-targeting sequence")]
    if "nucleus" in c or tf:
        R += [("nuclear import", "importins (KPNA/KPNB) carry it through the nuclear pore")]
    if tf or "transcription" in p:
        R += [("chromatin access", "remodelers (SWI/SNF) + the Pol II machinery to actually transcribe")]
    if is_enzyme or "metabol" in p:
        R += [("substrate supply", "an upstream step must deliver the substrate"),
              ("cofactor / energy", "ATP / NAD(P) / metal cofactor to run the reaction"),
              ("product disposal", "a downstream step must consume the product or it inhibits")]
    if "translation" in p:
        R += [("ribosome + tRNA", "the translation apparatus and charged tRNAs")]
    if "degrad" in p:
        R += [("substrate tagging", "E3 ubiquitin ligase / adaptor marks the target first")]
    if has_complex:
        R += [("co-subunit assembly", "its partner subunits must be present and stoichiometric"),
              ("chaperone", "HSP70/90 or a dedicated assembly chaperone to fold/assemble it")]
    if not R:
        R += [("housekeeping folding", "chaperones + quality control (the minimum any protein needs)")]
    return R


# ---- library of named cellular MACHINES + their canonical anchor genes. A dark gene's SPECIFIC role = the machine
#      whose anchors it most co-depends with (co-essentiality recovers complexes strongly). Anchors are well-known
#      members; the dark/query gene is always excluded when scoring, so this is leave-one-out by construction. ----
MACHINES = {
    "EMC (ER membrane-protein insertase)": ["EMC1", "EMC2", "EMC3", "EMC4", "EMC6", "EMC8", "MMGT1"],
    "OST (N-glycosylation)": ["STT3A", "STT3B", "DDOST", "RPN1", "RPN2", "DAD1", "OSTC", "MAGT1"],
    "SEC61 translocon (ER import)": ["SEC61A1", "SEC61B", "SEC61G", "SEC62", "SEC63", "SSR1", "SSR2"],
    "signal peptidase (SPCS)": ["SPCS1", "SPCS2", "SPCS3", "SEC11A", "SEC11C"],
    "COPII (ER→Golgi vesicles)": ["SEC23A", "SEC23B", "SEC24A", "SEC24C", "SEC13", "SEC31A", "SAR1A", "PREB"],
    "COPI (Golgi→ER vesicles)": ["COPA", "COPB1", "COPB2", "COPG1", "COPE", "ARCN1", "COPZ1"],
    "mito ribosome (large)": ["MRPL2", "MRPL3", "MRPL4", "MRPL9", "MRPL11", "MRPL13", "MRPL22", "MRPL44"],
    "mito ribosome (small)": ["MRPS2", "MRPS5", "MRPS7", "MRPS9", "MRPS12", "MRPS15", "MRPS22", "MRPS27"],
    "TOM/TIM (mito protein import)": ["TOMM20", "TOMM22", "TOMM40", "TOMM70", "TIMM23", "TIMM17A", "TIMM44", "TIMM50"],
    "cytosolic ribosome (large)": ["RPL3", "RPL4", "RPL5", "RPL7", "RPL10", "RPL11", "RPL13", "RPL23"],
    "cytosolic ribosome (small)": ["RPS2", "RPS3", "RPS5", "RPS6", "RPS8", "RPS9", "RPS11", "RPS15"],
    "proteasome 20S (core protease)": ["PSMA1", "PSMA3", "PSMA5", "PSMB1", "PSMB2", "PSMB5", "PSMB6"],
    "proteasome 19S (regulatory)": ["PSMC1", "PSMC2", "PSMC3", "PSMD1", "PSMD2", "PSMD4", "PSMD7"],
    "spliceosome": ["SF3B1", "SF3B2", "SF3B3", "SF3A1", "PRPF8", "SNRNP200", "EFTUD2", "U2AF1"],
    "RNA exosome": ["EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC7", "EXOSC9", "DIS3"],
    "CCT/TRiC chaperonin": ["TCP1", "CCT2", "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8"],
    "V-ATPase (lysosomal pump)": ["ATP6V0C", "ATP6V1A", "ATP6V1B2", "ATP6V1C1", "ATP6V1D", "ATP6V1E1", "ATP6V0D1"],
    "respiratory complex I": ["NDUFA9", "NDUFB8", "NDUFS1", "NDUFS2", "NDUFS3", "NDUFV1", "NDUFB10"],
    "respiratory complex III": ["UQCRC1", "UQCRC2", "UQCRB", "UQCRQ", "UQCRFS1", "CYC1"],
    "respiratory complex IV (COX)": ["COX4I1", "COX5A", "COX6B1", "COX7A2", "COX8A", "COX15", "SCO1"],
    "ATP synthase": ["ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5MC1", "ATP5PO", "ATP5PB"],
    "nuclear pore complex": ["NUP93", "NUP107", "NUP133", "NUP155", "NUP160", "NUP205", "NUP62"],
    "MCM replicative helicase": ["MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7"],
    "origin recognition (ORC)": ["ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6", "CDC6"],
    "cohesin": ["SMC1A", "SMC3", "RAD21", "STAG1", "STAG2", "NIPBL"],
    "condensin": ["SMC2", "SMC4", "NCAPD2", "NCAPG", "NCAPH"],
    "APC/C (anaphase-promoting)": ["ANAPC1", "ANAPC2", "ANAPC4", "ANAPC5", "CDC16", "CDC23", "CDC27"],
    "TFIID": ["TAF1", "TAF2", "TAF4", "TAF5", "TAF6", "TAF9", "TBP"],
    "Mediator": ["MED1", "MED12", "MED14", "MED17", "MED23", "MED24", "MED6"],
    "Integrator": ["INTS1", "INTS2", "INTS3", "INTS4", "INTS6", "INTS7", "INTS9"],
    "BAF (SWI/SNF remodeler)": ["SMARCA4", "SMARCB1", "SMARCC1", "SMARCC2", "ARID1A", "PBRM1"],
    "MICOS (mito cristae)": ["IMMT", "MICOS10", "MICOS13", "APOOL", "APOO", "CHCHD3"],
    "NDC80 (kinetochore)": ["NDC80", "NUF2", "SPC24", "SPC25"],
    "TREX/mRNA export": ["THOC1", "THOC2", "THOC3", "ALYREF", "DDX39B"],
    "GPI-anchor biosynthesis": ["PIGA", "PIGB", "PIGC", "PIGH", "PIGK", "PIGS", "PIGT"],
    "SRP (co-translational targeting)": ["SRP9", "SRP14", "SRP54", "SRP68", "SRP72", "SRPRA", "SRPRB"],
    "GET/TRC (tail-anchored insertion)": ["ASNA1", "GET4", "BAG6", "UBL4A", "WRB", "CAMLG", "GET1"],
    "ALG (N-glycan precursor)": ["ALG1", "ALG2", "ALG3", "ALG5", "ALG8", "ALG11", "ALG13", "ALG14"],
    "ERAD (ER-associated degradation)": ["SEL1L", "HRD1", "SYVN1", "DERL1", "DERL2", "OS9", "UBE2J1"],
    "mismatch repair (MMR)": ["MSH2", "MSH6", "MLH1", "PMS2", "MSH3"],
    "Fanconi anemia core": ["FANCA", "FANCB", "FANCC", "FANCG", "FANCL", "FANCF", "FANCM"],
    "MTOR/GATOR (nutrient sensing)": ["MTOR", "RPTOR", "RICTOR", "DEPDC5", "NPRL2", "NPRL3", "WDR24"],
    "TRAPP (Golgi tethering)": ["TRAPPC1", "TRAPPC2", "TRAPPC3", "TRAPPC8", "TRAPPC10", "TRAPPC11"],
    "retromer / WASH (endosomal recycling)": ["VPS35", "VPS26A", "VPS29", "WASHC2C", "WASHC4", "WASHC5"],
}


class Investigator:
    def __init__(self, kernel=None):
        import cellos
        self.k = kernel if kernel is not None else cellos.CellKernel(quiet=True)
        self.C = self.k.C
        self.name2i = self.C.idx
        self.dbg = self.k._load_debugger()
        self.measured = set(self.dbg["pgenes"]) if self.dbg else set()
        # co-dependency space
        z = np.load(f"{OUT}/depmap_vecs.npz", allow_pickle=True)
        self.dsyms = [str(s) for s in z["syms"]]
        Z = np.nan_to_num(z["Z"].astype("float32"), nan=0.0)
        nz = np.linalg.norm(Z, axis=1, keepdims=True); nz[nz == 0] = 1.0
        self.Zn = Z / nz
        self.didx = {s: i for i, s in enumerate(self.dsyms)}
        # annotated pool (genes WITH a job label) that also have a co-dep vector — the "people on file"
        self.annot = [g for g in self.dsyms if g in self.name2i and self._proc(g)]
        self.aidx = np.array([self.didx[g] for g in self.annot])
        # layers
        self.loc = json.load(open(f"{OUT}/localization.json")).get("labels", {}) if os.path.exists(f"{OUT}/localization.json") else {}
        self.lev = json.load(open(f"{OUT}/cell_levels.json")).get("labels", {}) if os.path.exists(f"{OUT}/cell_levels.json") else {}
        self.str = json.load(open(f"{OUT}/string_degree.json")).get("layer", {}) if os.path.exists(f"{OUT}/string_degree.json") else {}
        self.lit = json.load(open(f"{OUT}/litmine.json")) if os.path.exists(f"{OUT}/litmine.json") else {}
        # machine anchor indices (only anchors present in DepMap)
        self.machines = {m: np.array([self.didx[a] for a in anch if a in self.didx])
                         for m, anch in MACHINES.items()}
        self.machines = {m: idx for m, idx in self.machines.items() if len(idx) >= 3}

    def _g(self, gene):
        i = self.name2i.get(gene)
        return self.C.genes[i] if i is not None else {}

    def _proc(self, gene):
        v = self._g(gene).get("proc")
        return v if v and v != "other" else None            # "other" is not an informative job label

    def _field(self, gene, f):
        return self._g(gene).get(f)

    # ---- family: co-dependency + STRING neighbours, and their consensus job/place/path ----
    def neighbours(self, gene, k=25):
        out = []
        i = self.didx.get(gene)
        if i is not None:
            sims = self.Zn[self.aidx] @ self.Zn[i]
            for j in np.argsort(-sims)[:k]:
                g = self.annot[j]
                if g != gene and sims[j] > 0.1:
                    out.append((g, float(sims[j]), "co-dep"))
        for p in self.str.get(gene, {}).get("partners", [])[:12]:
            out.append((p["gene"], p["score"] / 1000.0, "STRING"))
        return out

    def consensus(self, neighbours, field, getter=None):
        getter = getter or (lambda g: self._field(g, field))
        votes = {}
        for g, w, _ in neighbours:
            v = getter(g)
            if v:
                votes[v] = votes.get(v, 0.0) + w
        if not votes:
            return None, 0.0
        top = max(votes, key=votes.get)
        return top, round(votes[top] / (sum(votes.values()) + 1e-9), 2)

    # ---- CONNECTIONS: the real, MEASURED edges a gene has (associations, not cause-and-effect) ----
    def connections(self, gene, k=10):
        """the gene's observed links to others: co-dependency (co-essential across 1,150 lines — a measured functional
        association), physical PPI (STRING), and pathway/complex co-membership. All REAL/measured, undirected — the
        'connections it has with others', not a predicted causal effect. Includes dark partners (the map needs them)."""
        out = {"co_dependency": [], "physical_ppi": [], "pathway": None, "complex_comembers": []}
        i = self.didx.get(gene)
        if i is not None:
            sims = self.Zn @ self.Zn[i]; sims[i] = -1.0
            for j in np.argsort(-sims)[:k]:
                if sims[j] > 0.15:
                    out["co_dependency"].append({"gene": self.dsyms[j], "coessentiality": round(float(sims[j]), 2),
                                                 "dark": bool(self._g(self.dsyms[j]).get("dark"))})
        for p in self.str.get(gene, {}).get("partners", [])[:k]:
            out["physical_ppi"].append({"gene": p["gene"], "string_score": p["score"]})
        gd = self._g(gene)
        out["pathway"] = gd.get("path") or self.consensus(self.neighbours(gene), "path")[0]
        # complex co-members = machine anchors it co-depends with (a curated-complex edge)
        mm, ms, _ = self.machine_match(gene)
        if mm and ms >= 0.25:
            out["complex_comembers"] = [{"machine": mm, "members": [a for a in MACHINES[mm] if a in self.didx][:6]}]
        return out

    def _blast(self, vec, topn=8):
        order = np.argsort(vec)
        down = [(self.dbg["syms"][j], round(float(vec[j]), 2)) for j in order[:topn] if vec[j] < 0]
        up = [(self.dbg["syms"][j], round(float(vec[j]), 2)) for j in order[::-1][:topn] if vec[j] > 0]
        return down, up

    def predict_surveillance(self, gene, exclude_self=False):
        """BUILD the surveillance data: if the gene is measured, its own removal-effect; else construct the predicted
        removal-effect (the blast radius) via the VALIDATED cellformer predictor — context-weighted average of its
        neighbours' Perturb-seq responses (same-complex 3×, co-expression 2×, PPI 1×), which recovers r~0.23 for genes
        in a complex, ~0.05 for singletons. This is the 'build the surveillance for it too' step, done honestly."""
        if not self.dbg:
            return {"status": "no-debugger"}
        M, pidx = self.dbg["M"], self.dbg["pidx"]
        if gene in pidx and not exclude_self:
            down, up = self._blast(M[pidx[gene]])
            return {"status": "MEASURED", "down": down, "up": up, "note": "removal effect measured on file (Perturb-seq)"}
        pred = self.k._predict_vec(gene)                        # cellformer context-weighted prediction (leave-one-out)
        if pred is None:
            return {"status": "UNPREDICTABLE", "down": [], "up": [],
                    "note": "no complex partner / <3 network neighbours in the screen (a singleton) — unbuildable"}
        ctx = self.k._last_ctx
        down, up = self._blast(pred)
        in_cx = bool(self.C.genes[self.name2i[gene]].get("npath")) if gene in self.name2i else False
        exp_r = "~0.23 (has a complex/context)" if len(ctx) >= 6 else "~0.05 (weak context)"
        return {"status": "PREDICTED", "from_context": ctx[:6], "n_context": len(ctx),
                "expected_fidelity_r": exp_r, "down": down, "up": up,
                "note": f"removal effect BUILT via cellformer from {len(ctx)} context genes; expected Pearson {exp_r} "
                        f"— a directional lead (complex members share responses), not a precise vector"}

    def validate_surveillance(self, sample=250, seed=0):
        """honest fidelity of the BUILT surveillance via the cellformer predictor (Pearson, matching cellformer),
        stratified by whether the gene has a curated complex (where the method works). Self held out (the context
        never includes the gene itself). Reports mean r vs a random-gene baseline."""
        import cellformer as cf
        M, pidx, pg = self.dbg["M"], self.dbg["pidx"], self.dbg["pgenes"]
        rng = np.random.default_rng(seed)
        genes = [g for g in self.measured if g in self.name2i]
        genes = list(np.array(genes, dtype=object)[rng.choice(len(genes), min(sample, len(genes)), replace=False)])
        real, base, cx, nocx = [], [], [], []
        for g in genes:
            pred = self.k._predict_vec(g)                       # cellformer context prediction (excludes g itself)
            if pred is None:
                continue
            c = cf._pearson(pred, M[pidx[g]])
            if np.isfinite(c):
                real.append(c)
                (cx if bool(self.C.genes[self.name2i[g]].get("npath")) else nocx).append(c)
            rg = pg[rng.integers(len(pg))]
            cb = cf._pearson(M[pidx[rg]], M[pidx[g]])
            if np.isfinite(cb):
                base.append(cb)
        return {"mean_pearson": round(float(np.mean(real)), 3) if real else None, "n": len(real),
                "with_complex": round(float(np.mean(cx)), 3) if cx else None, "n_complex": len(cx),
                "without_complex": round(float(np.mean(nocx)), 3) if nocx else None, "n_singleton": len(nocx),
                "random_baseline": round(float(np.mean(base)), 3) if base else None,
                "note": "built removal-effect (cellformer, complex 3x/coexpr 2x/PPI 1x) vs real response, self held "
                        "out — modest for complex genes, near-null for singletons; a directional lead, not a vector"}

    def profile(self, gene, predict_mode=False):
        """full dossier. predict_mode=True hides the gene's OWN record (for held-out validation)."""
        gd = self._g(gene)
        nb = self.neighbours(gene)
        rec_proc = None if predict_mode else self._proc(gene)
        rec_comp = None if predict_mode else gd.get("comp")
        rec_path = None if predict_mode else (gd.get("path") or None)
        job, jc = (rec_proc, 1.0) if rec_proc else self.consensus(nb, "proc", getter=self._proc)
        place, pc = (rec_comp, 1.0) if rec_comp else self.consensus(nb, "comp")
        path, ptc = (rec_path, 1.0) if rec_path else self.consensus(nb, "path")
        tf = bool(gd.get("tf")); has_cx = bool(gd.get("npath")) or bool(self.str.get(gene, {}).get("partners"))
        is_enz = (job in ("metabolism",)) or ("metabol" in (job or ""))
        machine, ms, mm = self.machine_match(gene)              # specific named machine, if any
        interactors = [p["gene"] for p in self.str.get(gene, {}).get("partners", [])[:8]]
        conn = self.connections(gene)                          # the REAL measured edges (surveillance, not cause-effect)
        # WHAT it is / HOW it acts — synthesized from the observed lines (no cause-and-effect claim)
        kind = "transcription factor" if tf else ("enzyme" if is_enz else "membrane protein" if place in ("membrane", "plasma membrane", "ER") else "protein")
        product = f"{kind} product; localizes to {place or 'an unresolved compartment'}"
        how = (f"operates as part of {machine}" if machine else f"operates within its {place or ''} module") + \
              (f", physically via {', '.join(interactors[:3])}" if interactors else "")
        why = f"serves {path or job or 'its module'}"
        dossier = {
            "gene": gene, "dark": bool(gd.get("dark")),
            "product": product, "mechanism_how": how, "purpose_why": why,
            "machine": {"name": machine, "score": ms, "margin": mm} if machine else None,
            "record": {"job_proc": rec_proc, "compartment": rec_comp, "pathway": rec_path,
                       "pubs": gd.get("pubs", 0), "litmine_papers": self.lit.get(gene, {}).get("n_papers", 0)},
            "job": {"predicted": job, "confidence": jc, "source": "record" if rec_proc else "family-consensus"},
            "destination": {"predicted": place, "confidence": pc, "source": "record" if rec_comp else "family/loc"},
            "path": {"predicted": path, "confidence": ptc, "source": "record" if rec_path else "co-dependency-decode"},
            "timing": self.lev.get(gene, {"status": "no-orderable-context"}),
            "interactors": interactors,
            "connections": conn,                               # the measured surveillance: who it is linked to
            "surveillance_raw": "measured Perturb-seq on file (use strace)" if gene in self.measured else "not measured",
            "required_support": [{"needs": n, "why": w} for n, w in required_support(place, job, tf, is_enz, has_cx)],
        }
        return dossier

    # ---- ranking: closest relatives to the measured set first ----
    def rank_targets(self, need_profile):
        """rank candidate genes by proximity (max co-dep cosine) to any MEASURED gene — closest first."""
        midx = np.array([self.didx[g] for g in self.measured if g in self.didx])
        Zm = self.Zn[midx]
        scored = []
        for g in need_profile:
            i = self.didx.get(g)
            if i is None:
                continue
            prox = float((Zm @ self.Zn[i]).max())
            scored.append((g, prox))
        scored.sort(key=lambda x: -x[1])
        return scored

    # ---- SPECIFIC role: which named MACHINE does the gene belong to? (co-dependency with each machine's anchors) ----
    def machine_match(self, gene, min_score=0.15):
        """assign the gene to the named machine whose anchor genes it most co-depends with. Always excludes the gene
        itself from the anchor set (leave-one-out). Returns (machine, score, margin_over_2nd) or (None, ...)."""
        i = self.didx.get(gene)
        if i is None:
            return None, 0.0, 0.0
        v = self.Zn[i]
        scored = []
        for m, idx in self.machines.items():
            keep = idx[idx != i]                                # leave-one-out
            if len(keep) < 3:
                continue
            sims = self.Zn[keep] @ v
            scored.append((m, float(np.mean(np.sort(sims)[-5:]))))   # mean of its top-5 anchor co-deps
        if not scored:
            return None, 0.0, 0.0
        scored.sort(key=lambda x: -x[1])
        best, s = scored[0]
        margin = s - (scored[1][1] if len(scored) > 1 else 0.0)
        return (best, round(s, 3), round(margin, 3)) if s >= min_score else (None, round(s, 3), round(margin, 3))

    @staticmethod
    def _tier(score):
        """confidence word for a machine match (the validated global top-1 is ~59%; score calibrates within that)."""
        return "confident" if score >= 0.40 else "probable" if score >= 0.25 else "tentative"

    def validate_machines(self):
        """honest check: for the anchor genes themselves (held out one at a time), does machine_match put them back in
        their OWN machine? Accuracy vs the number of machines (chance)."""
        hit = tot = 0
        gene2machine = {}
        for m, anch in MACHINES.items():
            for a in anch:
                gene2machine.setdefault(a, m)
        for g, true_m in gene2machine.items():
            if g not in self.didx or true_m not in self.machines:
                continue
            pred, s, _ = self.machine_match(g)
            tot += 1; hit += int(pred == true_m)
        return {"machine_top1": round(hit / tot, 3) if tot else None, "n": tot,
                "n_machines": len(self.machines), "chance": round(1 / max(len(self.machines), 1), 3)}

    # ---- the UNDERWORLD: dark genes embedded in a characterized co-dependency module = hidden crew ----
    def underworld(self, k=20, min_crew=4, min_sim=0.3):
        """find DARK genes whose strongest co-dependency partners form a COHERENT characterized module — the hidden
        operators. Predict their role (job/place) from the crew, and the required-support function they likely fill.
        This is the reasoned 'someone must be doing this job and it's nobody on file' step."""
        out = []
        for g in self.dsyms:
            if g not in self.name2i or not self._g(g).get("dark"):
                continue
            i = self.didx.get(g)
            if i is None:
                continue
            sims = self.Zn[self.aidx] @ self.Zn[i]
            order = np.argsort(-sims)[:12]
            crew = [(self.annot[j], float(sims[j])) for j in order if sims[j] > min_sim]
            if len(crew) < min_crew:
                continue
            nb = [(cg, cs, "co-dep") for cg, cs in crew]
            job, jc = self.consensus(nb, "proc", getter=self._proc)
            place, pc = self.consensus(nb, "comp")
            if not job:
                continue
            machine, ms, mm = self.machine_match(g)             # SPECIFIC named machine (sharpened role)
            crew_names = [c for c, _ in crew[:5]]
            # fallback role is SPECIFIC to this gene's actual module (its crew), not a generic reasoned service
            fallback = f"uncharacterized {place} module (co-crew: {', '.join(crew_names[:3])})"
            out.append({"dark_gene": g, "coherence": round(float(np.mean([s for _, s in crew])), 2),
                        "predicted_job": job, "job_conf": jc, "predicted_place": place, "place_conf": pc,
                        "crew": crew_names,
                        "machine": machine, "machine_score": ms, "machine_margin": mm,
                        "likely_role": machine or fallback,     # specific machine if matched, else this gene's own module
                        "measured": g in self.measured})
        out.sort(key=lambda x: -(x["coherence"] * x["job_conf"]))
        return out[:k]

    # ---- honest validation: hold out a gene's record, predict from family, measure per-line recovery ----
    def validate(self, genes, fields=("proc", "comp")):
        res = {f: {"hit": 0, "n": 0} for f in fields}
        from collections import Counter
        base = {}
        for f in fields:
            vals = [self._field(g, f) if f != "proc" else self._proc(g) for g in self.annot]
            vals = [v for v in vals if v]
            c = Counter(vals); base[f] = c.most_common(1)[0][1] / max(len(vals), 1)
        for g in genes:
            nb = self.neighbours(g)
            for f in fields:
                true = self._field(g, f) if f != "proc" else self._proc(g)
                if not true:
                    continue
                getter = self._proc if f == "proc" else (lambda x: self._field(x, f))
                pred, _ = self.consensus(nb, f, getter=getter)
                res[f]["n"] += 1; res[f]["hit"] += int(pred == true)
        return {f: {"top1": round(res[f]["hit"] / res[f]["n"], 3) if res[f]["n"] else None,
                    "n": res[f]["n"], "majority_baseline": round(base[f], 3)} for f in fields}


def render(d):
    """pretty-print one dossier the way a case file reads."""
    tim = d["timing"].get("status", "—") if isinstance(d["timing"], dict) else str(d["timing"])
    mach = d.get("machine")
    machln = (f"{mach['name']}  [{Investigator._tier(mach['score'])}: co-dep score {mach['score']}, "
              f"margin {mach['margin']} over the next machine]"
              if mach else "no specific machine matched — see the module named below")
    conn = d["connections"]
    codep = ", ".join(f"{c['gene']}({c['coessentiality']}{'*' if c['dark'] else ''})" for c in conn["co_dependency"][:8]) or "—"
    ppi = ", ".join(f"{c['gene']}({c['string_score']})" for c in conn["physical_ppi"][:8]) or "—"
    cxc = (conn["complex_comembers"][0]["machine"] + ": " + ", ".join(conn["complex_comembers"][0]["members"])) if conn["complex_comembers"] else "—"
    L = [f"  ┌─ DOSSIER: {d['gene']}" + ("  [DARK — no record on file]" if d["dark"] else "  [has record]"),
         f"  │ RECORD    : job={d['record']['job_proc'] or '—'}, compartment={d['record']['compartment'] or '—'}, "
         f"pathway={d['record']['pathway'] or '—'}, pubs={d['record']['pubs']}, litmine={d['record']['litmine_papers']}",
         f"  │ PRODUCT   : {d['product']}",
         f"  │ MACHINE   : {machln}",
         f"  │ JOB (fn)  : {d['job']['predicted']}  (conf {d['job']['confidence']}, {d['job']['source']})",
         f"  │ WHERE     : {d['destination']['predicted']}  (conf {d['destination']['confidence']}, {d['destination']['source']})",
         f"  │ WHEN      : {tim}",
         f"  │ HOW       : {d['mechanism_how']}",
         f"  │ WHY       : {d['purpose_why']}",
         f"  │ ── CONNECTIONS (measured associations — its edges on the map) ──",
         f"  │   co-dependency : {codep}   (*=dark partner)",
         f"  │   physical PPI  : {ppi}",
         f"  │   complex       : {cxc}",
         f"  │   pathway       : {conn['pathway'] or '—'}",
         f"  │ SURVEILLANCE raw: {d['surveillance_raw']}",
         f"  │ REQUIRED SUPPORT (reasoned — the services its job implies):"]
    for r in d["required_support"][:5]:
        L.append(f"  │    • {r['needs']}: {r['why']}")
    L.append("  └─")
    return "\n".join(L)


def main():
    print("=" * 96)
    print("INVESTIGATE — predicted investigative dossiers; closest relatives to the measured set first")
    print("=" * 96)
    inv = Investigator()
    allnames = [inv.C.genes[i].get("name") for i in range(len(inv.C.genes))]
    # targets: genes we have NO straight surveillance dataset about (not in the measured debugger), closest first
    need = [g for g in allnames if g and g not in inv.measured and g in inv.didx]
    ranked = inv.rank_targets(need)
    print(f"\n  {len(ranked):,} genes have NO measured surveillance; ranked by proximity to the measured set (closest first).")

    out = {"n_targets": len(ranked), "bands": {}}
    for N in (100, 200, 500):
        band = [g for g, _ in ranked[:N]]
        prox = [p for _, p in ranked[:N]]
        val = inv.validate(band)                                 # held-out per-line recovery for THIS band
        dossiers = [inv.profile(g) for g in band]               # textbook where present, predict the attribute gaps
        # CONNECTION-MAP coverage: how many real edges each gene gets (co-dep + PPI) — the surveillance we DO have
        n_codep = [len(d["connections"]["co_dependency"]) for d in dossiers]
        n_ppi = [len(d["connections"]["physical_ppi"]) for d in dossiers]
        with_edges = sum(1 for d in dossiers if d["connections"]["co_dependency"] or d["connections"]["physical_ppi"])
        out["bands"][N] = {"proximity_min": round(min(prox), 3), "proximity_median": round(float(np.median(prox)), 3),
                           "validation": val, "median_codep_edges": int(np.median(n_codep)),
                           "median_ppi_edges": int(np.median(n_ppi)), "genes_with_edges": with_edges,
                           "dossiers": dossiers if N == 100 else None}   # keep the full 100 as the deliverable
        print(f"\n  ── closest {N} (proximity {min(prox):.2f}–{max(prox):.2f}) ──")
        for f, r in val.items():
            if r["top1"] is not None:
                lift = r["top1"] / (r["majority_baseline"] + 1e-9)
                print(f"     {f:5} recovery: {r['top1']:.0%}  (majority baseline {r['majority_baseline']:.0%}, "
                      f"{lift:.1f}× ; n={r['n']})")
        print(f"     CONNECTIONS mapped for {with_edges}/{N} genes  (median {int(np.median(n_codep))} co-dependency "
              f"+ {int(np.median(n_ppi))} physical-PPI edges each) — real measured associations")

    # two FULL dossiers so the case file is legible (a dark one + a well-connected one)
    print("\n  ── full example dossiers (observed profile + connections) ──")
    band100 = [g for g, _ in ranked[:100]]
    show = [g for g in band100 if inv._g(g).get("dark")][:1] + [g for g in band100 if not inv._g(g).get("dark")][:1]
    for g in show:
        print(render(inv.profile(g)))

    # SHARPENED role: validate machine-matching (does it put known members back in their own machine?)
    mv = inv.validate_machines()
    out["machine_validation"] = mv
    print(f"\n  ── MACHINE-MATCH validation (held-out known members → own machine): "
          f"{mv['machine_top1']:.0%} top-1 across {mv['n_machines']} machines (chance {mv['chance']:.0%}, n={mv['n']}) ──")

    # THE UNDERWORLD — dark genes embedded in characterized modules, now named to the SPECIFIC machine
    uw = inv.underworld(k=25)
    out["underworld"] = uw
    named = [u for u in uw if u.get("machine")]
    print(f"\n  ── THE UNDERWORLD: {len(uw)} dark genes in a coherent module; {len(named)} matched to a SPECIFIC machine ──")
    for u in uw[:14]:
        tag = "measured" if u["measured"] else "unmeasured"
        role = (f"→ {u['machine']} ({inv._tier(u['machine_score'])} {u['machine_score']})" if u.get("machine")
                else f"→ {u['likely_role']}")
        print(f"     {u['dark_gene']:12} {role:60} coh {u['coherence']}, {tag}")

    json.dump(out, open(f"{OUT}/investigate.json", "w"), indent=1)
    print("\n" + "=" * 96)
    return out


if __name__ == "__main__":
    main()
