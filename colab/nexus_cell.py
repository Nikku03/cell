"""nexus_cell — NEXUS wired INTO the running cell. The nexus.py node was validated in isolation (dual-sensor
orthogonality, regsign direction); this makes it a live CellOS query that reasons a real mutation through the whole
stack and hands the result to the cell's own essentiality layer, so the cell can answer a question it could not before:

    given a mutation, is it LOSS or GAIN of function, does it break FOLDING or BINDING, and does the cell DEPEND on it?

Four layers, each already validated on its own; this composes them and is honest about which are solid:
  1. DIRECTION  — regsign: does the gene carry a breakable inhibitory BRAKE? (UniProt activity-regulation text) ->
                  GOF-capable vs LOF-only.  [precision 86% when it fires; recall-limited]
  2. INTRINSIC  — ddg_predictor folding ΔΔG on a monomer structure -> does it still FOLD?  [near-field structure, solid]
  3. EXTRINSIC  — binding ΔΔG at an interface (measured SKEMPI when we have it) -> does it still DOCK?  [orthogonal to 2]
  4. ACTIVITY   — nexus.directed_activity(fold, bind, inhibitory=sign): soft-AND for LOF, brake-release for GOF.
  + CELL        — the kernel's own reason() essentiality: does the cell depend on this protein at all?

HONEST BOUNDARY (unchanged): the sensors live in near-field STRUCTURE (the recoverable regime). The last hop
activity -> whole-cell phenotype is far-field and does NOT compose (buffered), so the cell DEPENDENCY is reported as
CONTEXT, not a validated phenotype prediction. Where a structure/partner is absent, the structural sensors abstain and
the query still returns the (structure-free) regulatory direction + cell dependency.
"""
import os, sys, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "orphan")


class NexusCell:
    def __init__(self, kernel=None, pdbdir=None):
        self.k = kernel                                          # the CellKernel, for the essentiality layer
        self.pdbdir = pdbdir or os.path.join(OUT, "pdb_cache")
        self._ddg = None
        self._reg = {}                                          # gene/acc -> (sign, brake_score, snippet) cache
        self._muts = None

    # ---- layer 1: regulatory DIRECTION (structure-free) --------------------------------------------------------
    def _sign(self, gene, uniprot=None):
        import regsign
        key = uniprot or gene
        if key in self._reg:
            return self._reg[key]
        q = f"accession:{uniprot}" if uniprot else f"(gene:{gene}) AND organism_id:9606 AND reviewed:true"
        u = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": q, "fields": "accession,gene_primary,cc_activity_regulation", "format": "tsv", "size": "1"})
        reg = None
        try:
            txt = urllib.request.urlopen(u, timeout=45).read().decode()
            rows = txt.strip().split("\n")
            if len(rows) > 1:
                c = rows[1].split("\t"); reg = c[2] if len(c) > 2 else ""
        except Exception:
            reg = None
        if reg is None:
            res = {"sign": None, "brake": None, "snippet": None}            # fetch failed / unknown
        else:
            res = {"sign": regsign.regulatory_sign(reg), "brake": regsign.brake_score(reg),
                   "snippet": (reg.replace("ACTIVITY REGULATION:", "").strip()[:110] or None)}
        self._reg[key] = res
        return res

    # ---- layer 2: intrinsic FOLDING sensor ---------------------------------------------------------------------
    def _fetch_pdb(self, pdb):
        os.makedirs(self.pdbdir, exist_ok=True)
        dst = os.path.join(self.pdbdir, f"{pdb}.pdb")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return dst
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", dst)
        except Exception:
            return None
        return dst if os.path.exists(dst) and os.path.getsize(dst) > 1000 else None

    def _intrinsic(self, pdb, chain, pos, wt, mut):
        path = self._fetch_pdb(pdb)
        if path is None:
            return None
        try:
            import ddg_predictor as dp
            if self._ddg is None:
                self._ddg = dp.DDGPredictor(os.path.join(OUT, "ddg_model.pkl"))
            v, ok = self._ddg.predict_from_structure(path, chain, pos, wt, mut)
            return float(v) if ok else None
        except Exception:
            return None

    # ---- layer 3: extrinsic BINDING sensor (measured when we have it) ------------------------------------------
    def _extrinsic_measured(self, pdb, chain, pos, mut):
        import flex_physics as fp
        if self._muts is None:
            try:
                self._muts = fp.load_pdb_muts()
            except Exception:
                self._muts = []
        for r in self._muts:
            if r["pdb"] == pdb and r["chain"] == chain and r["pos"] == pos and r["mut"] == mut:
                return float(r["ddg"])
        return None

    # ---- cell layer: essentiality (the dependency context) -----------------------------------------------------
    def _cell(self, gene):
        if self.k is None:
            return None
        try:
            txt = self.k.reason(gene)
        except Exception:
            return None
        concl = conf = p = None
        for ln in txt.splitlines():
            if "CONCLUSION:" in ln:
                seg = ln.split("CONCLUSION:", 1)[1].strip()
                concl = seg.split("confidence")[0].strip()
                if "confidence" in seg:
                    conf = seg.split("confidence", 1)[1].split("(")[0].strip()
                if "P≈" in seg:
                    try:
                        p = float(seg.split("P≈", 1)[1].split("%")[0]) / 100.0
                    except Exception:
                        p = None
        return {"conclusion": concl, "confidence": conf, "p_essential": p} if concl else None

    # ---- compose -----------------------------------------------------------------------------------------------
    def query(self, gene, uniprot=None, pos=None, wt=None, mut=None, pdb=None, chain=None):
        import nexus
        sg = self._sign(gene, uniprot)
        ddg_fold = ddg_bind = None; bind_src = None
        have_mut = all(x is not None for x in (pos, wt, mut))
        if pdb and chain and have_mut:
            ddg_fold = self._intrinsic(pdb, chain, int(pos), wt, mut)
            ddg_bind = self._extrinsic_measured(pdb, chain, int(pos), mut)
            bind_src = "measured (SKEMPI)" if ddg_bind is not None else None
        # activity (only meaningful once at least one structural sensor fired)
        act_lof = act_gof = kd_ratio = None
        if ddg_fold is not None or ddg_bind is not None:
            import ecflux
            f = ddg_fold if ddg_fold is not None else 0.0
            b = ddg_bind if ddg_bind is not None else 0.0
            act_lof = nexus.nexus_activity(f, b)
            act_gof = nexus.directed_activity(f, b, inhibitory=bool(sg["sign"]))
            if ddg_bind is not None and ddg_bind > 0:
                import math
                kd_ratio = float(math.exp(ddg_bind / ecflux.RT_310))    # fold-change in Kd (affinity loss), the informative readout
        # direction call
        if sg["sign"] is None:
            direction = "unknown (no UniProt activity-regulation text retrieved)"
        elif sg["sign"] == 1:
            direction = "GOF-capable — carries a breakable inhibitory brake (disrupting it can RAISE activity)"
        else:
            direction = "LOF-only — no annotated brake (a damaging mutation LOWERS activity)"
        cell = self._cell(gene)
        return {"gene": gene, "uniprot": uniprot,
                "mutation": (f"{chain}{pos}{wt}->{mut}" if have_mut and chain else
                             (f"{pos}{wt}->{mut}" if have_mut else None)),
                "pdb": pdb, "chain": chain,
                "sign": sg, "direction": direction,
                "ddg_fold": ddg_fold, "ddg_bind": ddg_bind, "bind_source": bind_src, "kd_ratio": kd_ratio,
                "activity_lof": act_lof, "activity_gof_if_brake": act_gof,
                "cell": cell}

    def render(self, r):
        L = [f"nexus {r['gene']}" + (f"  {r['mutation']}" if r["mutation"] else "") +
             "  —  reasoning a mutation through the dual-sensor node + the cell"]
        # 1) direction
        sn = f"  (\"{r['sign']['snippet']}...\")" if r["sign"]["snippet"] else ""
        L.append(f"  DIRECTION (regsign): {r['direction']}{sn}")
        # 2/3) structural sensors — lead with the informative ΔΔG decomposition + affinity fold-change
        if r["ddg_fold"] is not None or r["ddg_bind"] is not None:
            fold_ok = r["ddg_fold"] is not None and r["ddg_fold"] < 2.0
            f = "n/a" if r["ddg_fold"] is None else f"{r['ddg_fold']:+.2f}" + (" (still folds)" if fold_ok else " (destabilises)")
            b = "n/a" if r["ddg_bind"] is None else f"{r['ddg_bind']:+.2f}"
            kd = f" → ~{r['kd_ratio']:.0f}× weaker binding" if r.get("kd_ratio") else ""
            src = f" [{r['bind_source']}]" if r["bind_source"] else ""
            L.append(f"  INTRINSIC (fold ΔΔG): {f} kcal/mol   ·   EXTRINSIC (bind ΔΔG): {b}{src}{kd}")
            bind_break = r["ddg_bind"] is not None and r["ddg_bind"] > 2.0
            fold_break = r["ddg_fold"] is not None and r["ddg_fold"] > 2.0
            which = "binding" if bind_break else ("folding" if fold_break else None)
            if r["sign"]["sign"] == 1 and which:
                call = f"GOF-capable — breaking a brake-bearing interface can RAISE activity (gated by folding: {'ok' if fold_ok else 'AT RISK'})"
            elif which == "binding":
                call = ("LOF via the BINDING axis — the fold predictor calls it stable, yet binding is broken; the two "
                        "failure modes are ORTHOGONAL, which is exactly why the node carries both sensors")
            elif which == "folding":
                call = "LOF — destabilises the FOLD"
            else:
                call = "near-neutral / mild by both sensors"
            L.append(f"  READOUT: {call}")
            if r["activity_lof"] is not None:
                L.append(f"  (soft-AND occupancy {r['activity_lof']:.2f}× — two-state SATURATION at high WT affinity buffers "
                         f"the occupancy number; the ΔΔG above is the informative structural readout.)")
        else:
            L.append("  INTRINSIC/EXTRINSIC: abstained — no complex structure+interface mutation given "
                     "(pass  PDB CHAIN  with a measured/known interface residue to fire the structural sensors)")
        # cell layer
        if r["cell"]:
            c = r["cell"]; pp = f"  (calibrated P≈{c['p_essential']:.0%})" if c.get("p_essential") is not None else ""
            L.append(f"  CELL DEPENDENCY: {c['conclusion']}  [{c.get('confidence','?')}]{pp} — does the cell depend on this protein?")
        # honest regime
        L.append("  ── the structural sensors are near-field (solid); the cell-dependency is CONTEXT, not a validated "
                 "phenotype (activity→phenotype is far-field/buffered).")
        return "\n".join(L)


def main():
    """standalone demo: reason a few real mutations through the integrated node (needs net for UniProt/PDB)."""
    sys.path.insert(0, HERE)
    try:
        from cellos import CellKernel
        k = CellKernel()
    except Exception as e:
        print(f"(cell layer unavailable: {type(e).__name__} — running structure/direction layers only)")
        k = None
    nc = NexusCell(kernel=k)
    cases = [
        ("GHR",  "P10912", 304, "W", "A", "1A22", "B"),            # FULL PIPELINE: measured interface breaker on a human complex the cell knows
        ("BRAF", "P15056", None, None, None, None, None),          # brake -> GOF-capable (V600E class)
        ("ABL1", "P00519", None, None, None, None, None),          # brake -> GOF-capable (BCR-ABL / imatinib class)
        ("TP53", "P04637", 175, "R", "H", None, None),             # no brake -> LOF-only (tumor suppressor; correct)
        ("KRAS", "P01116", 12, "G", "D", None, None),              # GOF; regsign fires on the 'inactive form bound to GDP' cue -> correct direction
    ]
    out = []
    for g, up, pos, wt, mut, pdb, ch in cases:
        r = nc.query(g, uniprot=up, pos=pos, wt=wt, mut=mut, pdb=pdb, chain=ch)
        print("\n" + nc.render(r)); out.append(r)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"cases": out}, open(f"{OUT}/nexus_cell.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT}/nexus_cell.json")
    return out


if __name__ == "__main__":
    main()
