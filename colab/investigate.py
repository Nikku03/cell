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

    def predicted_surveillance(self, gene):
        """measured if in the debugger; else predict the removal effect from measured neighbours (the `predict`
        approach). Honest expected fidelity from how many measured neighbours vote."""
        if gene in self.measured:
            return {"status": "MEASURED", "note": "removal effect on file (Perturb-seq) — use strace"}
        nb = [(g, w) for g, w, _ in self.neighbours(gene) if g in self.measured]
        if not nb:
            return {"status": "UNPREDICTABLE", "note": "no measured neighbour — a singleton; honest failure mode"}
        exp_r = 0.43 if len(nb) >= 4 else 0.30 if len(nb) >= 2 else 0.19
        return {"status": "PREDICTED", "from_n_measured_neighbours": len(nb),
                "expected_fidelity_r": exp_r, "top_neighbours": [g for g, _ in nb[:5]],
                "note": f"removal effect predicted from {len(nb)} measured neighbours (expected r≈{exp_r})"}

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
        dossier = {
            "gene": gene, "dark": bool(gd.get("dark")),
            "record": {"job_proc": rec_proc, "compartment": rec_comp, "pathway": rec_path,
                       "pubs": gd.get("pubs", 0), "litmine_papers": self.lit.get(gene, {}).get("n_papers", 0)},
            "job": {"predicted": job, "confidence": jc, "source": "record" if rec_proc else "family-consensus"},
            "destination": {"predicted": place, "confidence": pc, "source": "record" if rec_comp else "family/loc"},
            "path": {"predicted": path, "confidence": ptc, "source": "record" if rec_path else "co-dependency-decode"},
            "timing": self.lev.get(gene, {"status": "no-orderable-context"}),
            "interactors": [p["gene"] for p in self.str.get(gene, {}).get("partners", [])[:8]],
            "family": [{"gene": g, "w": round(w, 2), "via": s} for g, w, s in nb[:6]],
            "surveillance": self.predicted_surveillance(gene),
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
            reqs = required_support(place, job, False, job == "metabolism", True)
            out.append({"dark_gene": g, "coherence": round(float(np.mean([s for _, s in crew])), 2),
                        "predicted_job": job, "job_conf": jc, "predicted_place": place, "place_conf": pc,
                        "crew": [c for c, _ in crew[:5]], "likely_role": reqs[0][0],
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
    fam = ", ".join("{}({})".format(f["gene"], f["via"]) for f in d["family"][:5]) or "—"
    interact = ", ".join(d["interactors"][:6]) or "—"
    tim = d["timing"].get("status", "—") if isinstance(d["timing"], dict) else str(d["timing"])
    L = [f"  ┌─ DOSSIER: {d['gene']}" + ("  [DARK — no record on file]" if d["dark"] else "  [has record]"),
         f"  │ RECORD    : job={d['record']['job_proc'] or '—'}, compartment={d['record']['compartment'] or '—'}, "
         f"pathway={d['record']['pathway'] or '—'}, pubs={d['record']['pubs']}, litmine={d['record']['litmine_papers']}",
         f"  │ JOB       : {d['job']['predicted']}  (conf {d['job']['confidence']}, {d['job']['source']})",
         f"  │ DESTINATION: {d['destination']['predicted']}  (conf {d['destination']['confidence']}, {d['destination']['source']})",
         f"  │ PATH      : {d['path']['predicted'] or '—'}  ({d['path']['source']})",
         f"  │ TIMING    : {tim}",
         f"  │ INTERACTORS: {interact}",
         f"  │ FAMILY    : {fam}",
         f"  │ SURVEILLANCE: {d['surveillance']['status']} — {d['surveillance']['note']}",
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
    # targets: genes lacking a clear job (dark or proc unknown) — the ones needing prediction
    need = [g for g in allnames if g and (inv._g(g).get("dark") or not inv._proc(g))]
    ranked = inv.rank_targets(need)
    print(f"\n  {len(ranked):,} genes need a predicted profile; ranked by proximity to the measured surveillance set.")

    out = {"n_targets": len(ranked), "bands": {}}
    for N in (100, 200, 500):
        band = [g for g, _ in ranked[:N]]
        prox = [p for _, p in ranked[:N]]
        val = inv.validate(band)                                 # held-out per-line recovery for THIS band
        # example dossiers (first 3 of the band)
        examples = [inv.profile(g, predict_mode=inv._g(g).get("dark") or not inv._proc(g)) for g in band[:3]]
        out["bands"][N] = {"proximity_min": round(min(prox), 3), "proximity_median": round(float(np.median(prox)), 3),
                           "validation": val, "examples": examples}
        print(f"\n  ── closest {N} (proximity {min(prox):.2f}–{max(prox):.2f}) ──")
        for f, r in val.items():
            if r["top1"] is not None:
                lift = r["top1"] / (r["majority_baseline"] + 1e-9)
                print(f"     {f:5} recovery: {r['top1']:.0%}  (majority baseline {r['majority_baseline']:.0%}, "
                      f"{lift:.1f}× ; n={r['n']})")
        ex = examples[0]
        print(f"     e.g. {ex['gene']}: job={ex['job']['predicted']}({ex['job']['confidence']}), "
              f"dest={ex['destination']['predicted']}({ex['destination']['confidence']}), "
              f"surveillance={ex['surveillance']['status']}")

    # two FULL dossiers so the case file is legible (a dark one + a well-connected one)
    print("\n  ── full example dossiers ──")
    band100 = [g for g, _ in ranked[:100]]
    show = [g for g in band100 if inv._g(g).get("dark")][:1] + [g for g in band100 if not inv._g(g).get("dark")][:1]
    for g in show:
        print(render(inv.profile(g, predict_mode=inv._g(g).get("dark") or not inv._proc(g))))

    # THE UNDERWORLD — dark genes embedded in characterized modules (hidden crew)
    uw = inv.underworld(k=20)
    out["underworld"] = uw
    print(f"\n  ── THE UNDERWORLD: {len(uw)} dark genes embedded in a coherent characterized module (hidden crew) ──")
    for u in uw[:10]:
        tag = "measured" if u["measured"] else "unmeasured"
        print(f"     {u['dark_gene']:12} → {u['predicted_job']}/{u['predicted_place']} "
              f"(coherence {u['coherence']}, {tag}) — crew: {', '.join(u['crew'][:4])}; likely role: {u['likely_role']}")

    json.dump(out, open(f"{OUT}/investigate.json", "w"), indent=1)
    print("\n" + "=" * 96)
    return out


if __name__ == "__main__":
    main()
