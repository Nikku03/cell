"""influence — use the perturbation data for ONE thing: a directed network of what gene's removal AFFECTS what other
gene-products (the measured END influence of each knockout — we know the ends, not the mechanism). Then reconstruct
the ROUTE for each effect from the structure we DO know, without ever predicting an unmeasured effect:

  affects(X)   : the measured end-influence of removing X — the gene-products that go up/down (the "ends").
  explain(X,P) : reconstruct the route KO(X) -> P:
                   DIRECT     — X physically/causally contacts P (PPI / curated causal / same complex): stage 1.
                   MEDIATED   — a specific, non-hub, MEASURED stepping-stone M: X strongly moves M AND M strongly &
                                specifically moves P, sign-consistent (X->M->P). Both hops are real knockout data —
                                this is where the 'keep earlier knockout data in context' comes in: the intermediate
                                is another gene we actually knocked out.
                   UNRESOLVED — neither; distal/regulatory/diffuse (the far-field that does not compose).
  cascade(X)   : the staged reconstruction of X's whole blast radius — entry points (direct) then mediated steps.

HONEST, measured on this screen: of all strong measured effects, ~1.4% are DIRECT contacts (7x chance), ~13% are
MEDIATED by a specific non-hub measured stepping-stone (~5x a placebo), and ~86% stay UNRESOLVED. So the influence
network itself is 100% real data; the route is reconstructable for a minority and honestly flagged for the rest. The
confound that had to be controlled: pan-hub mediators (whose knockout moves everything) are excluded, and structure
(PPI/pathway) is used only to name the DIRECT entry point — as a mediator filter it just re-selects hubs and doesn't
help, so it is not used there.
-> outputs/orphan/influence.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


class InfluenceNetwork:
    def __init__(self, kernel=None, thr=0.5, hub_pct=0.85, specificity=4.0):
        import cellos
        self.k = kernel if kernel is not None else cellos.CellKernel(quiet=True)
        self.C = C = self.k.C
        self.thr = thr; self.hub_pct = hub_pct; self.spec = specificity
        dbg = self.k._load_debugger()
        if not dbg:
            raise RuntimeError("no perturbation data mounted")
        self.M = np.asarray(dbg["M"], dtype="float32")
        self.pgenes = list(dbg["pgenes"]); self.syms = list(dbg["syms"])
        self.pidx = {g: i for i, g in enumerate(self.pgenes)}
        self.sidx = {g: i for i, g in enumerate(self.syms)}
        self.dual = [g for g in self.pgenes if g in self.sidx]
        self.hi_prec = dbg.get("hi_prec", set())
        # confound bookkeeping: each knockout's blast radius (percentile) and typical effect magnitude
        blast = (np.abs(self.M) > thr).sum(1)
        self.blast_pct = {self.pgenes[i]: float((blast < blast[i]).mean()) for i in range(len(self.pgenes))}
        self.med_abs = {g: float(np.median(np.abs(self.M[self.pidx[g]]))) for g in self.dual}

    # --------------------------------------------------------------- structure accessors (symbols)
    def _ppi(self, g):
        i = self.C.idx.get(g)
        return {self.C.name[j] for j in self.C.ppi_adj.get(i, ())} if i is not None else set()

    def _causal(self, g):
        i = self.C.idx.get(g)
        return {self.C.name[j] for j, *_ in (self.C.causal_out.get(i, []) or [])} if i is not None else set()

    def _mates(self, g):
        i = self.C.idx.get(g)
        if i is None:
            return set()
        out = set()
        for cn in (self.C.D.get("gene2cplx", {}) or {}).get(str(i), []):
            out |= {self.C.name[m] for m in self.C.D.get("complexes", {}).get(cn, []) if isinstance(m, int)}
        return out - {g}

    def _direct_set(self, g):
        return self._ppi(g) | self._causal(g) | self._mates(g)

    def _contact_kind(self, X, P):
        if P in self._ppi(X):
            return "physical PPI"
        if P in self._causal(X):
            return "curated causal edge"
        if P in self._mates(X):
            return "same complex"
        return None

    # --------------------------------------------------------------- the measured influence (the ends)
    def _row(self, X):
        return self.M[self.pidx[X]] if X in self.pidx else None

    def affects(self, X, topn=12):
        """the measured END influence of removing X — the gene-products that go up/down (what we actually know)."""
        row = self._row(X)
        if row is None:
            return {"status": "not-knocked-out", "gene": X}
        order = np.argsort(row)
        down = [(self.syms[j], round(float(row[j]), 2)) for j in order[:topn] if row[j] < -self.thr]
        up = [(self.syms[j], round(float(row[j]), 2)) for j in order[::-1][:topn] if row[j] > self.thr]
        n = int((np.abs(row) > self.thr).sum())
        return {"status": "ok", "gene": X, "n_affected_strong": n, "down": down, "up": up,
                "hi_precision": X in self.hi_prec}

    def _mediator(self, X, P, row=None):
        """a specific, non-hub, MEASURED stepping stone M with X->M and M->P strong and sign(X->M)*sign(M->P)=sign(X->P).
        Candidates are X's own strong measured targets (the earlier-knockout data used in context)."""
        row = self._row(X) if row is None else row
        if row is None or P not in self.sidx:
            return None
        xps = np.sign(row[self.sidx[P]])
        # candidate mediators = genes X strongly moves, that are dual & non-hub
        cand = [self.syms[j] for j in np.where(np.abs(row) > self.thr)[0]]
        best = None
        for Mg in cand:
            if Mg in (X, P) or Mg not in self.pidx or self.blast_pct[Mg] > self.hub_pct:
                continue
            mp = self.M[self.pidx[Mg]][self.sidx[P]]
            if abs(mp) <= self.thr or abs(mp) < self.spec * self.med_abs[Mg]:   # M->P strong AND specific to M
                continue
            if np.sign(row[self.sidx[Mg]]) * np.sign(mp) == xps:
                return {"mediator": Mg, "x_to_m": round(float(row[self.sidx[Mg]]), 2),
                        "m_to_p": round(float(mp), 2), "m_blast_pct": round(self.blast_pct[Mg], 2)}
            best = best or {"mediator": Mg, "x_to_m": round(float(row[self.sidx[Mg]]), 2),
                            "m_to_p": round(float(mp), 2), "m_blast_pct": round(self.blast_pct[Mg], 2),
                            "sign": "inconsistent"}
        return best if (best and "sign" in best) else None

    def explain(self, X, P):
        """reconstruct the route KO(X) -> P: DIRECT contact, MEASURED-MEDIATED stepping stone, or UNRESOLVED."""
        row = self._row(X)
        if row is None:
            return {"status": "not-knocked-out"}
        if P not in self.sidx:
            return {"status": "not-measured", "note": f"{P} was not measured"}
        val = float(row[self.sidx[P]])
        if abs(val) <= self.thr:
            return {"status": "ok", "route": "no-effect", "value": round(val, 2)}
        kind = self._contact_kind(X, P)
        if kind:
            return {"status": "ok", "route": "direct", "via": kind, "value": round(val, 2), "stage": 1}
        med = self._mediator(X, P, row)
        if med and "sign" not in med:
            return {"status": "ok", "route": "mediated", "value": round(val, 2), "stage": 2,
                    "path": f"{X} -> {med['mediator']} -> {P}", "stepping_stone": med}
        return {"status": "ok", "route": "unresolved", "value": round(val, 2),
                "note": "distal / regulatory / diffuse — no direct contact or specific measured stepping stone"}

    def cascade(self, X, cap=400):
        """the staged reconstruction of X's whole measured blast radius: entry points (direct), then mediated steps,
        then the unresolved remainder — each affected gene-product routed as far as the real data allows."""
        row = self._row(X)
        if row is None:
            return {"status": "not-knocked-out", "gene": X}
        strong = [self.syms[j] for j in np.argsort(-np.abs(row)) if abs(row[j]) > self.thr][:cap]
        direct, mediated, unresolved = [], [], 0
        dset = self._direct_set(X)
        for P in strong:
            if P == X:
                continue
            if P in dset:
                direct.append({"gene": P, "via": self._contact_kind(X, P), "value": round(float(row[self.sidx[P]]), 2)})
            else:
                med = self._mediator(X, P, row)
                if med and "sign" not in med:
                    mediated.append({"gene": P, "via": med["mediator"], "value": round(float(row[self.sidx[P]]), 2)})
                else:
                    unresolved += 1
        tot = len(direct) + len(mediated) + unresolved
        return {"status": "ok", "gene": X, "n_affected": tot,
                "stage1_direct": direct[:15], "n_direct": len(direct),
                "stage2_mediated": mediated[:15], "n_mediated": len(mediated),
                "n_unresolved": unresolved,
                "explained_frac": round((len(direct) + len(mediated)) / tot, 3) if tot else 0.0}

    # --------------------------------------------------------------- honest validation
    def validate(self, n_kos=120, seed=1):
        rng = np.random.default_rng(seed)
        Xs = list(rng.choice(self.dual, min(n_kos, len(self.dual)), replace=False))
        cat = {"direct": 0, "mediated": 0, "unresolved": 0}; total = 0
        placebo_hits = 0; placebo_n = 0
        for X in Xs:
            row = self._row(X); dset = self._direct_set(X)
            strong = [self.syms[j] for j in np.where(np.abs(row) > self.thr)[0] if self.syms[j] != X]
            for P in strong:
                total += 1
                if P in dset:
                    cat["direct"] += 1
                    continue
                m = self._mediator(X, P, row)
                if m and "sign" not in m:
                    cat["mediated"] += 1
                else:
                    cat["unresolved"] += 1
            # placebo: genes X does NOT move — how often does a "mediator" spuriously appear?
            unaff = [self.syms[j] for j in np.where(np.abs(row) < 0.05)[0]]
            for P in rng.choice(unaff, min(20, len(unaff)), replace=False):
                placebo_n += 1
                m = self._mediator(X, P, row)
                placebo_hits += 1 if (m and "sign" not in m) else 0
        # direct enrichment baseline
        base = float(np.mean([len(self._direct_set(X)) / len(self.C.name) for X in Xs[:40]]))
        return {"n_effects": total,
                "direct_frac": round(cat["direct"] / total, 4), "direct_x_chance": round((cat["direct"]/total)/base, 1),
                "mediated_frac": round(cat["mediated"] / total, 4),
                "unresolved_frac": round(cat["unresolved"] / total, 4),
                "explained_frac": round((cat["direct"] + cat["mediated"]) / total, 4),
                "placebo_mediated_frac": round(placebo_hits / placebo_n, 4) if placebo_n else None}


def main():
    print("=" * 96)
    print("INFLUENCE — the perturbation data as a network of what a knockout AFFECTS, with the route reconstructed")
    print("=" * 96)
    G = InfluenceNetwork()
    print(f"\n  {len(G.pgenes):,} knockouts x {len(G.syms):,} measured gene-products; "
          f"{len(G.dual):,} genes are both (usable as stepping stones)\n")

    print("  AFFECTS — the measured end-influence of a knockout (the ends we actually know):")
    a = G.affects("SF3B1", topn=6)
    print(f"    KO {a['gene']} moves {a['n_affected_strong']} genes; down: "
          f"{', '.join(f'{s}({v})' for s,v in a['down'][:5])}; up: {', '.join(f'{s}({v})' for s,v in a['up'][:4])}")

    print("\n  EXPLAIN — reconstruct the route for individual effects:")
    for X, P in (("PSMB5", "PSMA4"), ("SF3B1", "RBM22"), ("MYC", "NPM1"), ("GATA1", "HBB")):
        e = G.explain(X, P)
        if e["status"] == "ok" and e["route"] != "no-effect":
            extra = e.get("via") or e.get("path") or e.get("note", "")
            print(f"    {X:6} -> {P:7} [{e['route']:10}] {extra}  (Δ={e['value']})")

    print("\n  CASCADE — the staged reconstruction of one knockout's blast radius (GATA1):")
    c = G.cascade("GATA1")
    print(f"    {c['n_affected']} affected: {c['n_direct']} direct entry, {c['n_mediated']} mediated, "
          f"{c['n_unresolved']} unresolved  ({c['explained_frac']:.0%} explained)")
    for d in c["stage1_direct"][:4]:
        print(f"      stage1 direct : {d['gene']:8} via {d['via']}  (Δ={d['value']})")
    for m in c["stage2_mediated"][:4]:
        print(f"      stage2 mediated: {m['gene']:8} via {m['via']}  (Δ={m['value']})")

    print("\n" + "-" * 96 + "\n  HONEST VALIDATION — how much of the cascade is reconstructable?\n" + "-" * 96)
    v = G.validate()
    print(f"    effects categorized: {v['n_effects']:,}")
    print(f"    DIRECT contact       {v['direct_frac']:.1%}  ({v['direct_x_chance']}x chance)")
    print(f"    MEASURED-MEDIATED    {v['mediated_frac']:.1%}  (placebo {v['placebo_mediated_frac']:.1%} — "
          f"{v['mediated_frac']/max(v['placebo_mediated_frac'],1e-9):.0f}x)")
    print(f"    UNRESOLVED           {v['unresolved_frac']:.1%}")
    print(f"    => EXPLAINED         {v['explained_frac']:.1%}")

    verdict = (
        f"The perturbation data is now ONE directed network: KO(X) -> the gene-products it measurably moves (the ends). "
        f"That network is 100% real data. Reconstructing the ROUTE of each effect works for a validated MINORITY and is "
        f"honestly flagged for the rest: {v['direct_frac']:.0%} of strong effects are a DIRECT contact (PPI / curated "
        f"causal / same complex; {v['direct_x_chance']}x chance), {v['mediated_frac']:.0%} are MEDIATED by a specific "
        f"non-hub MEASURED stepping stone (another knockout in the data — {v['mediated_frac']/max(v['placebo_mediated_frac'],1e-9):.0f}x "
        f"a placebo), and {v['unresolved_frac']:.0%} stay UNRESOLVED — distal/regulatory/diffuse, the far-field that "
        f"does not compose. The confound that had to be controlled: pan-hub mediators (whose knockout moves everything) "
        f"are excluded, and PPI/pathway structure is used only to name the direct entry point — as a mediator filter it "
        f"just re-selects hubs and does not help. So the honest product is a real measured influence network with a "
        f"per-effect route: direct, measured-mediated, or explicitly unresolved.")
    print("\n" + "=" * 96 + f"\nVERDICT: {verdict}\n" + "=" * 96)
    out = {"n_knockouts": len(G.pgenes), "n_measured": len(G.syms), "n_dual": len(G.dual),
           "validation": v, "verdict": verdict,
           "example_cascade_GATA1": G.cascade("GATA1")}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/influence.json", "w"), indent=1)
    print(f"  -> {OUT}/influence.json")
    return out


if __name__ == "__main__":
    main()
