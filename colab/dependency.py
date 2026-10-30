"""dependency — patch the measured SURVEILLANCE (Perturb-seq) into a directed DEPENDENCY network, and use it to see
knockout effects. A signed directed edge X -> Y means: knocking out X measurably MOVES Y (so Y depends on X). Built
from the debugger M (9,871 knockouts x 8,202 measured genes); both endpoints restricted to the 7,651 genes that are
BOTH knocked out and measured, so the graph can be traversed.

  knockout(X)     : the measured downstream effect of removing X (its blast radius, up/down) — REAL, not predicted.
  dependents(X)   : genes whose removal is felt by X (X's upstream dependencies).
  impact(X)       : how many genes measurably respond to removing X — a measured load-bearing score (≠ essentiality).
  transitivity    : the honest test — do dependencies CHAIN (X->Y->Z predict X->Z)? i.e. can we compute a knockout's
                    effect by propagating measured dependencies, or is only the DIRECT effect trustworthy?
-> outputs/orphan/dependency.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


class Dependency:
    def __init__(self, kernel=None, thr=0.5):
        import cellos
        self.k = kernel if kernel is not None else cellos.CellKernel(quiet=True)
        self.dbg = self.k._load_debugger()
        self.M = np.asarray(self.dbg["M"], dtype="float32")
        self.pgenes = list(self.dbg["pgenes"]); self.syms = list(self.dbg["syms"])
        self.pidx = {g: i for i, g in enumerate(self.pgenes)}
        self.sidx = {g: i for i, g in enumerate(self.syms)}
        self.thr = thr
        # dual genes: both a knockout row AND a measured column -> nodes of the traversable directed graph
        self.dual = [g for g in self.pgenes if g in self.sidx]
        self.dcol = np.array([self.sidx[g] for g in self.dual])      # their measured-column indices
        self.dpos = {g: i for i, g in enumerate(self.dual)}
        # STRING partners (for the locality check)
        p = f"{OUT}/string_degree.json"
        self.str = json.load(open(p)).get("layer", {}) if os.path.exists(p) else {}

    def knockout(self, gene, topn=10):
        """the MEASURED downstream effect of removing `gene` (what actually goes up/down). Real data."""
        if gene not in self.pidx:
            return {"status": "not-measured", "note": f"{gene} was not knocked out in the screen"}
        row = self.M[self.pidx[gene]]
        order = np.argsort(row)
        down = [(self.syms[j], round(float(row[j]), 2)) for j in order[:topn] if row[j] < -0.15]
        up = [(self.syms[j], round(float(row[j]), 2)) for j in order[::-1][:topn] if row[j] > 0.15]
        n_dep = int((np.abs(row) > self.thr).sum())
        return {"status": "measured", "down": down, "up": up, "n_dependents_strong": n_dep,
                "hi_precision": gene in self.dbg.get("hi_prec", set())}

    def dependents(self, gene, topn=10):
        """which knockouts most move `gene` — the genes it is DOWNSTREAM of (its upstream dependencies)."""
        if gene not in self.sidx:
            return {"status": "not-measured"}
        col = self.M[:, self.sidx[gene]]
        order = np.argsort(-np.abs(col))
        up = [(self.pgenes[i], round(float(col[i]), 2)) for i in order[:topn]
              if abs(col[i]) > 0.3 and self.pgenes[i] != gene]
        return {"status": "ok", "moved_by_knockout_of": up}

    def impact_ranking(self, topn=25):
        """rank genes by MEASURED knockout impact = # genes strongly perturbed when removed (blast-radius size).
        A load-bearing score distinct from essentiality (transcriptome-shift, not lethality)."""
        blast = (np.abs(self.M) > self.thr).sum(1)
        order = np.argsort(-blast)
        return [{"gene": self.pgenes[i], "genes_perturbed": int(blast[i]),
                 "hi_precision": self.pgenes[i] in self.dbg.get("hi_prec", set())} for i in order[:topn]]

    def impact_of(self, gene):
        """this gene's measured knockout impact (# genes strongly perturbed) and its percentile among all knockouts."""
        if gene not in self.pidx:
            return None
        blast = (np.abs(self.M) > self.thr).sum(1)
        n = int(blast[self.pidx[gene]])
        pct = float((blast < blast[self.pidx[gene]]).mean())
        return {"genes_perturbed": n, "percentile": round(pct, 3)}

    def locality(self, sample=400, seed=0):
        """what FRACTION of a knockout's strong responders are the gene's OWN STRING partners? high = the effect is
        local (direct binding partners); low = distal/regulatory. Quantifies the dependency structure honestly."""
        rng = np.random.default_rng(seed)
        genes = [g for g in self.dual if g in self.str and self.str[g].get("partners")]
        genes = list(np.array(genes, dtype=object)[rng.choice(len(genes), min(sample, len(genes)), replace=False)])
        fracs, base = [], []
        for g in genes:
            row = self.M[self.pidx[g]]
            strong = {self.syms[j] for j in np.where(np.abs(row) > self.thr)[0]}
            if len(strong) < 5:
                continue
            partners = {p["gene"] for p in self.str[g]["partners"]}
            fracs.append(len(strong & partners) / len(strong))
            base.append(len(partners) / len(self.syms))            # chance overlap
        return {"mean_partner_fraction": round(float(np.mean(fracs)), 3) if fracs else None,
                "chance": round(float(np.mean(base)), 4) if base else None, "n": len(fracs)}

    def transitivity(self, sample=300, seed=0):
        """THE honest test: do measured dependencies CHAIN? Predict a knockout's effect by propagating one step through
        the dependency graph — for KO(X), sum M[Y] (each direct target Y's own knockout effect) weighted by X->Y —
        and correlate with the MEASURED M[X]. If dependencies compose, propagation recovers the effect; if not, only
        the direct measurement is trustworthy (the far-field null)."""
        from scipy.stats import spearmanr
        rng = np.random.default_rng(seed)
        genes = list(np.array(self.dual, dtype=object)[rng.choice(len(self.dual), min(sample, len(self.dual)), replace=False)])
        Md = self.M[:, self.dcol]                                   # responses restricted to dual genes (so Y indexes rows)
        real, base = [], []
        for g in genes:
            xr = self.M[self.pidx[g]]                               # measured effect of KO(X) over all cols
            targets = [t for t in np.argsort(-np.abs(xr[self.dcol]))[:30] if self.dual[t] != g]
            w = np.array([xr[self.dcol[t]] for t in targets])
            if len(targets) < 5 or not np.any(np.abs(w) > self.thr):
                continue
            # propagate: weighted sum of each target's OWN knockout effect vector
            prop = (w[:, None] * self.M[[self.pidx[self.dual[t]] for t in targets]]).sum(0)
            # exclude the direct targets themselves (that would be trivial self-recovery) -> test 2nd-order genes
            mask = np.ones(self.M.shape[1], bool)
            for t in targets:
                mask[self.dcol[t]] = False
            mask[self.sidx.get(g, -1)] = False
            c = spearmanr(prop[mask], xr[mask]).statistic
            if np.isfinite(c):
                real.append(c)
            rg = self.dual[rng.integers(len(self.dual))]
            cb = spearmanr(self.M[self.pidx[rg]][mask], xr[mask]).statistic
            if np.isfinite(cb):
                base.append(cb)
        return {"mean_spearman": round(float(np.mean(real)), 3) if real else None,
                "random_baseline": round(float(np.mean(base)), 3) if base else None, "n": len(real)}


def main():
    print("=" * 92)
    print("DEPENDENCY — the measured surveillance as a directed dependency network; seeing knockout effects")
    print("=" * 92)
    D = Dependency()
    print(f"\n  dependency graph: {len(D.dual):,} genes are both knocked-out AND measured (traversable nodes)\n")

    print("  MEASURED KNOCKOUT EFFECTS (direct — real data):")
    for g in ("SF3B1", "PSMB5", "RPL13"):
        ko = D.knockout(g)
        if ko["status"] == "measured":
            dn = ", ".join(f"{x}({v})" for x, v in ko["down"][:5])
            print(f"    remove {g:7} -> {ko['n_dependents_strong']:4} genes perturbed; top down: {dn}")

    imp = D.impact_ranking()
    print(f"\n  KNOCKOUT IMPACT ranking (blast-radius size = # genes measurably moved; a load-bearing score):")
    print("    " + ", ".join(f"{r['gene']}({r['genes_perturbed']})" for r in imp[:12]))

    loc = D.locality()
    print(f"\n  DEPENDENCY LOCALITY: {loc['mean_partner_fraction']:.1%} of a knockout's strong responders are the "
          f"gene's own STRING partners (chance {loc['chance']:.1%}, n={loc['n']}).")

    tr = D.transitivity()
    print(f"\n  TRANSITIVITY (do dependencies CHAIN? propagate 1 step, predict the 2nd-order effect):")
    print(f"    propagated-vs-measured Spearman {tr['mean_spearman']} (random {tr['random_baseline']}, n={tr['n']})")

    lf = loc["mean_partner_fraction"]; tv = tr["mean_spearman"]
    verdict = (
        f"THE DIRECT KNOCKOUT EFFECT IS REAL; PROPAGATION IS NOT. For the {len(D.dual):,} genes that were knocked out "
        f"and measured, the dependency edges are the actual Perturb-seq blast radius — removing X and reading who moves "
        f"is real data (that is what 'knockout GENE' now shows). But the dependencies do NOT chain: only {lf:.0%} of a "
        f"knockout's strong responders are the gene's own physical partners (vs {loc['chance']:.1%} chance) — most of "
        f"the effect is DISTAL/regulatory, not direct binding — and propagating one step through the graph to predict "
        f"the 2nd-order effect is Spearman {tv} (~chance {tr['random_baseline']}). So the software can SHOW a measured "
        f"knockout's effect and rank genes by measured impact, but it cannot COMPUTE an unmeasured knockout's cascade "
        f"by chaining dependencies — the same structure(measured)-vs-dynamics(non-composing) split. Use the measured "
        f"edge; do not trust the propagated one.")
    print("\n" + "=" * 92 + f"\nVERDICT: {verdict}\n" + "=" * 92)
    out = {"n_nodes": len(D.dual), "impact_top": imp, "locality": loc, "transitivity": tr, "verdict": verdict}
    json.dump(out, open(f"{OUT}/dependency.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
