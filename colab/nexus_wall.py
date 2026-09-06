"""nexus_wall -- the ONE honest bridge from NEXUS's physical/obligate layer to the WALL (mRNA far field).

NEXUS-KO (nexus_ko.py) predicts a PROTEIN-level consequence (which obligate complex partners co-fail), validated on
co-dependency -- explicitly NOT the mRNA far field, NOT the wall. wall_test.py showed the wall is lowered not solved: a one-hop
network MODEL recalls ~0.38 of a knockout's SPECIFIC (tide-removed) mRNA movers vs an ORACLE ceiling ~0.62 -- a modelling/representation
gap, not data-impossibility.

Can NEXUS advance to help the WALL? There is exactly one physically principled path -- a TWO-HOP composition the one-hop MODEL cannot
reach by construction:
      KO(A)  --physical/obligate-->  protein B destabilised  --B is a TF-->  B's regulatory targets move in mRNA
              (the NEXUS layer)                                (the reg layer)          (the wall readout)
If A physically stabilises a transcription factor B (obligate destabilisation), then KO(A) lowers B's PROTEIN, dysregulating B's
regulatory TARGETS at the mRNA level. Those targets sit TWO hops from A (physical then regulatory), so a one-hop neighbour model misses
them structurally. This is the only way the physical/co-dependency layer produces an mRNA far-field signal.

Test, on wall_test's EXACT held-out protocol (K562, leave-one-KO-out, TAU=1.0, tide removed, recall@K over the non-tide universe):
  RANDOM / TIDE-null / MODEL(1-hop nbr) / ORACLE* -- reproduced from wall_test as the flanking baselines, unchanged.
  NEXUS-2HOP  -- PURE-TOPOLOGY forward prediction: score gene T by sum over physical partners B of A of [obligate weight(A,B)] * [B
                 regulates T]. No peeking at measured profiles -- a mechanistic prior, the NEXUS spirit.
  NEXUS-2HOP (count) -- unweighted variant (count of physical-partner-TFs that regulate T), to see if the obligate weighting matters.
  COMBO       -- late fusion (standardised MODEL + standardised NEXUS-2HOP): does the two-hop path ADD specific-mover recall BEYOND the
                 one-hop model? This is THE decisive question. If COMBO <= MODEL, NEXUS stays on its own layer (honest negative).

Reported two ways: UNCONDITIONAL (all scored KOs -- the deployment number, dragged down where the mechanism does not fire) and
CONDITIONAL on the two-hop path being non-empty (where the mechanism can even apply -- the fair test of the mechanism itself).
"""
import os
import json, pickle, collections
from pathlib import Path
import numpy as np
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU = 1.0
TIDE_FRAC = 0.05
K = 50
MIN_SPEC = 5
N_NEI = 10
KS = [20, 50, 100]


def load_onehop_nb(idx_set, D):
    """the wall_test MODEL neighbourhood: PPI + complex/pathway co-membership + trrust regulon, pooled at one hop."""
    info = {g["name"]: g for g in D["genes"]}
    nb = collections.defaultdict(set)
    for e in D.get("ppi", []):
        if e[0] in idx_set and e[1] in idx_set:
            nb[e[0]].add(e[1]); nb[e[1]].add(e[0])
    cplx = collections.defaultdict(set)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        for c in (cs if isinstance(cs, (list, set, tuple)) else [cs]) if cs else []:
            cplx[("c", c)].add(g)
    for g in idx_set:
        p = info.get(g, {}).get("path")
        if p:
            cplx[("p", p)].add(g)
    for grp, mem in cplx.items():
        if 2 <= len(mem) <= 200:
            ml = list(mem)
            for a in ml:
                nb[a].update(x for x in ml if x != a)
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    for tf, ts in trr.items():
        for t in ts:
            g = t[0] if isinstance(t, (list, tuple)) else t
            if tf in idx_set and g in idx_set:
                nb[tf].add(g); nb[g].add(tf)
    return nb


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; idx = {n: i for i, n in enumerate(names)}; Nn = len(names)
    codep = {int(k): {int(j): float(s) for j, s in lst} for k, lst in D["codep"].items()}
    def codep_val(i, j):
        return max(codep.get(i, {}).get(j, 0.0), codep.get(j, {}).get(i, 0.0))

    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO)
    genes = sorted({g for v in KO.values() for g in v})
    gi = {g: i for i, g in enumerate(genes)}
    ki = {k: i for i, k in enumerate(kos)}
    nK, nG = len(kos), len(genes)
    print(f"K562: {nK} knockouts x {nG} moved-gene universe (top-250/KO)")

    M = np.zeros((nK, nG), dtype=np.float32)
    for k, v in KO.items():
        r = ki[k]
        for g, z in v.items():
            M[r, gi[g]] = z
    A = np.abs(M)
    mover = A >= TAU
    mover_freq = mover.mean(axis=0)
    tide_mask = mover_freq >= TIDE_FRAC
    nontide = ~tide_mask
    nontide_idx = np.where(nontide)[0]
    Az = A.copy()
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    print(f"tide genes: {int(tide_mask.sum())}  |  non-tide universe: {int(nontide.sum())}")

    # one-hop neighbour graph (same as wall_test MODEL)
    nb = load_onehop_nb(set(genes), D)

    # ---- NEXUS two-hop machinery: physical partners (hop 1) and regulatory targets (hop 2), name-space ----
    phys = collections.defaultdict(set)
    for a, b in D.get("ppi", []):
        if isinstance(a, int) and isinstance(b, int) and a < Nn and b < Nn:
            phys[names[a]].add(names[b]); phys[names[b]].add(names[a])
    csize_name = {}
    for members in D["complexes"].values():
        m = [x for x in members if isinstance(x, int) and x < Nn]
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                na, nbn = names[m[a]], names[m[b]]
                phys[na].add(nbn); phys[nbn].add(na)
                key = (min(na, nbn), max(na, nbn))
                csize_name[key] = min(csize_name.get(key, 99), len(m))
    # regulatory targets: trrust tf_targets UNION signed reg edges, TF-name -> {target-names}
    reg = collections.defaultdict(set)
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    for tf, ts in trr.items():
        for t in ts:
            g = t[0] if isinstance(t, (list, tuple)) else t
            reg[tf].add(g)
    for e in D.get("reg", []):
        if len(e) >= 2 and isinstance(e[0], int) and isinstance(e[1], int) and e[0] < Nn and e[1] < Nn:
            reg[names[e[0]]].add(names[e[1]])

    def obl_w(a_name, b_name):
        """how strongly KO of A is expected to destabilise B: measured co-dependency + a small-complex (obligate) structural bonus,
        + a small base so any physical partner contributes something."""
        ia, ib = idx.get(a_name), idx.get(b_name)
        w = 0.1
        if ia is not None and ib is not None:
            w += codep_val(ia, ib)
            if csize_name.get((min(a_name, b_name), max(a_name, b_name)), 99) <= 4:
                w += 0.25
        return w

    def nexus2hop(x, weighted=True):
        s = np.zeros(nG); fired = 0
        for b in sorted(phys.get(x, ())):                # sorted iteration => bit-reproducible summation (no set-hash float wobble)
            tgts = reg.get(b)
            if not tgts:
                continue
            w = obl_w(x, b) if weighted else 1.0
            for t in sorted(tgts):
                ti = gi.get(t)
                if ti is not None:
                    s[ti] += w; fired = 1
        return s, fired

    def obligate_tf_rows(x):
        """A's physical partners that are TFs (have regulatory targets) AND were themselves knocked out -> (ki-rows, obligate weights)."""
        rows, ws = [], []
        for b in sorted(phys.get(x, ())):               # sorted => deterministic order
            if not reg.get(b):          # B must be a TF (has regulatory targets) for the two-hop mRNA path
                continue
            if b not in ki or b == x:   # B must itself have a measured KO profile to transfer
                continue
            rows.append(ki[b]); ws.append(obl_w(x, b))
        return rows, ws

    def prof_transfer(rows, ws):
        if not rows:
            return np.zeros(nG), False
        s = np.zeros(nG); wsum = 0.0
        for r, w in zip(rows, ws):
            s += w * Az[r]; wsum += w
        return (s / wsum, True) if wsum > 0 else (s, False)

    rngc = np.random.RandomState(1)   # independent stream for control-subset selection (keeps s_rand reproducible)
    def ctrl_rows(x, n, pool):
        """n random KO'd neighbours of x drawn from `pool` ('nbr'=all one-hop neighbours, 'phys'=physical partners) -- the control for
        'is it the obligate-TF SELECTION that helps, or does upweighting ANY equal-size neighbour subset help?'"""
        if pool == "nbr":
            cand = sorted(ki[g] for g in nb.get(x, ()) if g in ki and g != x)      # sort => deterministic regardless of set-hash order
        else:
            cand = sorted(ki[g] for g in phys.get(x, ()) if g in ki and g != x)
        if n <= 0 or not cand:
            return [], []
        n = min(n, len(cand))
        sel = rngc.choice(len(cand), n, replace=False)
        return [cand[i] for i in sel], [1.0] * n

    def zstd(v):
        sd = v.std()
        return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0

    rng = np.random.RandomState(0)

    def recall_at(scores, truth_idx, restrict, k):
        order = restrict[np.argsort(-scores[restrict])][:k]
        return len(set(order.tolist()) & truth_idx) / max(len(truth_idx), 1)

    res = collections.defaultdict(list)
    res_fire = collections.defaultdict(list)     # conditional on 2-hop TOPOLOGY firing
    res_firep = collections.defaultdict(list)    # conditional on 2-hop PROFILE firing (B is an obligate TF partner AND was KO'd)
    scored = 0; n_fire = 0; n_firep = 0
    for x in kos:
        r = ki[x]
        mv = np.where(mover[r])[0]
        if len(mv) < MIN_SPEC:
            continue
        spec = set(i for i in mv.tolist() if nontide[i])
        if len(spec) < MIN_SPEC:
            continue
        scored += 1
        s_rand = rng.rand(nG)
        s_tide = mover_freq.copy()
        neigh = [ki[g] for g in nb.get(x, ()) if g in ki and g != x]
        s_model = Az[neigh].mean(axis=0) if neigh else np.zeros(nG)
        sim = Mn @ Mn[r]; sim[r] = -1
        top = np.argsort(-sim)[:N_NEI]
        s_oracle = Az[top].mean(axis=0)
        s_2hop, fired = nexus2hop(x, weighted=True)
        s_2hopc, _ = nexus2hop(x, weighted=False)
        otrows, otws = obligate_tf_rows(x)
        s_2hopp, firedp = prof_transfer(otrows, otws)
        s_combo = zstd(s_model) + zstd(s_2hop)
        s_combop = zstd(s_model) + zstd(s_2hopp)
        # CONTROLS: fuse the model with a RANDOM equal-size neighbour subset (from all one-hop nbrs, and from physical partners).
        cr_rows, cr_ws = ctrl_rows(x, len(otrows), "nbr")
        cp_rows, cp_ws = ctrl_rows(x, len(otrows), "phys")
        s_ctrln, _ = prof_transfer(cr_rows, cr_ws)
        s_ctrlp, _ = prof_transfer(cp_rows, cp_ws)
        s_combo_ctrln = zstd(s_model) + zstd(s_ctrln)
        s_combo_ctrlp = zstd(s_model) + zstd(s_ctrlp)
        preds = [("rand", s_rand), ("tide", s_tide), ("model", s_model),
                 ("nexus2hop", s_2hop), ("nexus2hopc", s_2hopc), ("nexus2hopp", s_2hopp),
                 ("combo", s_combo), ("combop", s_combop),
                 ("ctrl_combon", s_combo_ctrln), ("ctrl_combop", s_combo_ctrlp), ("oracle", s_oracle)]
        for k in KS:
            for name, s in preds:
                v = recall_at(s, spec, nontide_idx, k)
                res[f"spec_{name}_{k}"].append(v)
                if fired:
                    res_fire[f"spec_{name}_{k}"].append(v)
                if firedp:
                    res_firep[f"spec_{name}_{k}"].append(v)
        if fired:
            n_fire += 1
        if firedp:
            n_firep += 1

    mean = {k: float(np.mean(v)) for k, v in res.items()}
    meanf = {k: float(np.mean(v)) for k, v in res_fire.items()} if n_fire else {}
    meanfp = {k: float(np.mean(v)) for k, v in res_firep.items()} if n_firep else {}
    order = [("rand", "RANDOM"), ("tide", "TIDE null"), ("model", "MODEL(1hop)"),
             ("nexus2hop", "NEXUS-2HOP(topo)"), ("nexus2hopc", "NEXUS-2HOP(cnt)"),
             ("nexus2hopp", "NEXUS-2HOP(prof)"), ("combo", "COMBO topo"), ("combop", "COMBO prof"),
             ("ctrl_combon", "CTRL rand-nbr"), ("ctrl_combop", "CTRL rand-phys"), ("oracle", "ORACLE*")]
    print(f"\nscored {scored} held-out knockouts; two-hop TOPOLOGY path fires for {n_fire} ({100*n_fire/max(scored,1):.0f}%), "
          f"two-hop PROFILE path (obligate TF partner also KO'd) fires for {n_firep} ({100*n_firep/max(scored,1):.0f}%)\n")
    print("SPECIFIC-mover recall (tide removed) -- ALL scored KOs (deployment number):")
    print(f"   {'predictor':17s} " + "".join(f"| @{k:<6d}" for k in KS))
    for name, lab in order:
        print(f"   {lab:17s} " + "".join(f"|  {mean[f'spec_{name}_{k}']:.3f}" for k in KS))
    if n_fire:
        print(f"\nSPECIFIC-mover recall -- ONLY the {n_fire} KOs where the two-hop TOPOLOGY path fires:")
        print(f"   {'predictor':17s} " + "".join(f"| @{k:<6d}" for k in KS))
        for name, lab in order:
            print(f"   {lab:17s} " + "".join(f"|  {meanf[f'spec_{name}_{k}']:.3f}" for k in KS))
    if n_firep:
        print(f"\nSPECIFIC-mover recall -- ONLY the {n_firep} KOs where the two-hop PROFILE path fires (fairest test: obligate TF partner "
              f"with a measured KO profile to transfer):")
        print(f"   {'predictor':17s} " + "".join(f"| @{k:<6d}" for k in KS))
        for name, lab in order:
            print(f"   {lab:17s} " + "".join(f"|  {meanfp[f'spec_{name}_{k}']:.3f}" for k in KS))

    m2 = mean[f"spec_nexus2hop_{K}"]; mm = mean[f"spec_model_{K}"]; mc = mean[f"spec_combo_{K}"]
    mt = mean[f"spec_tide_{K}"]; mo = mean[f"spec_oracle_{K}"]
    # decisive: does the two-hop path (a) point at real specific movers at all, and (b) ADD over the one-hop model?
    twohop_real = (meanf.get(f"spec_nexus2hop_{K}", 0.0) > 1.3 * meanf.get(f"spec_tide_{K}", 1e-9)) if n_fire else False
    combo_adds = mc > mm + 0.01
    combo_adds_fire = (meanf.get(f"spec_combo_{K}", 0.0) > meanf.get(f"spec_model_{K}", 0.0) + 0.01) if n_fire else False
    m2f = meanf.get(f"spec_nexus2hop_{K}", float("nan")); mmf = meanf.get(f"spec_model_{K}", float("nan"))
    mcf = meanf.get(f"spec_combo_{K}", float("nan")); mtf = meanf.get(f"spec_tide_{K}", float("nan"))
    # the FAIREST test WITH THE DECISIVE CONTROL: on the subset where an obligate TF partner has a measured profile, does the
    # obligate-REWEIGHTED profile transfer (combop) beat the one-hop model AND beat fusing the model with a RANDOM equal-size neighbour
    # subset? Only if it beats the random-subset controls is the gain attributable to the NEXUS obligate-TF SELECTION rather than to
    # generic 'upweight some neighbours' fusion mechanics.
    mmp = meanfp.get(f"spec_model_{K}", float("nan")); mcpp = meanfp.get(f"spec_combop_{K}", float("nan"))
    m2pp = meanfp.get(f"spec_nexus2hopp_{K}", float("nan"))
    mctrln = meanfp.get(f"spec_ctrl_combon_{K}", float("nan")); mctrlp = meanfp.get(f"spec_ctrl_combop_{K}", float("nan"))
    combop_beats_model = bool(n_firep and mcpp > mmp + 0.01)
    combop_beats_ctrl = bool(n_firep and mcpp > max(mctrln, mctrlp) + 0.01)
    # the honest attribution: the NEXUS mechanism gets credit ONLY if it beats BOTH the model and the random-subset controls
    combop_adds_fire = combop_beats_model and combop_beats_ctrl
    any_add = bool(combo_adds_fire or combop_adds_fire)

    verdict = (
        "CAN NEXUS ADVANCE TO HELP THE WALL? Tested the ONLY physically principled bridge -- the two-hop composition KO(A) -> "
        "obligate-destabilise protein B -> (if B is a TF) B's regulatory targets move in mRNA -- on wall_test's EXACT held-out protocol. "
        f"FIRST FACT: the two-hop path is SPARSE -- it fires (a physical partner of the knockout is a TF with targets in the universe) for "
        f"{n_fire}/{scored} knockouts ({100*n_fire/max(scored,1):.0f}%). Where it does not fire it can add nothing by construction. "
        + (f"WHERE IT FIRES (fair test): the PURE-TOPOLOGY NEXUS-2HOP recalls {m2f:.3f} of specific movers vs the tide-null {mtf:.3f} -- "
           + ("ABOVE the null, so the mechanistic path DOES point at real specific movers. " if twohop_real else
              f"essentially at the RANDOM floor ({mean[f'spec_rand_{K}']:.3f}), so the mechanistic target-set is NOT the specific movers. ")
           + f"GIVING THE IDEA ITS STRONGEST FORM (measured obligate-TF-reweighted profile transfer, on the {n_firep} KOs where an "
           f"obligate TF partner was itself knocked out): COMBO-prof {mcpp:.3f} vs the one-hop MODEL {mmp:.3f} -- a raw +{mcpp-mmp:.3f}. "
           f"BUT THE DECISIVE CONTROL: fusing the model with a RANDOM equal-size neighbour subset scores {mctrln:.3f} (random one-hop "
           f"nbr) / {mctrlp:.3f} (random physical partner) on the same KOs. "
           + ("COMBO-prof beats BOTH controls, so the gain is attributable to the NEXUS obligate-TF SELECTION, not to generic "
              "upweight-some-neighbours fusion mechanics -- a REAL, controlled contribution of the physical layer to the mRNA wall."
              if combop_adds_fire else
              "COMBO-prof does NOT clearly beat the random-subset controls -- so the apparent +{:.3f} over the model is the generic effect "
              "of re-emphasising ANY neighbour subset via z-score fusion, NOT the NEXUS obligate-TF mechanism. Attributing it to NEXUS "
              "would be the exact overstatement this project guards against.".format(mcpp - mmp))
           + (f" ONE HONEST INCIDENTAL (not a NEXUS result): the random-PHYSICAL-partner fusion itself ({mctrlp:.3f}) edges the uniform "
              f"one-hop MODEL ({mmp:.3f}), hinting wall_test's MODEL could be modestly improved by upweighting PHYSICAL-partner "
              "neighbours over pathway/regulon co-members -- a small neighbour-weighting tweak, still far under the oracle."
              if mctrlp > mmp + 0.01 else "")
           + " "
           if n_fire else "The two-hop path never fires on the scorable knockouts. ")
        + f"UNCONDITIONAL (all {scored} KOs, the deployment number): COMBO {mc:.3f} vs MODEL {mm:.3f} vs ORACLE ceiling {mo:.3f}. "
        + ("NET: NEXUS CAN nudge the wall through this one bridge, but only on the sparse subset where a knockout's obligate partner is a "
           "transcription factor -- it does not move the deployment-average wall, and the ORACLE head-room stays open. The honest reading "
           "is unchanged from wall_test: the wall is a similarity-representation gap; the NEXUS physical layer is a correct but "
           "SMALL-COVERAGE contributor to it, not the key that closes it."
           if any_add else
           "NET: NO -- NEXUS does NOT advance to solve the wall. The one physically principled bridge (physical->regulatory two-hop) is "
           "both sparse and, where it fires, does not beat the existing one-hop model. NEXUS's value stays on its OWN layer (protein "
           "co-dependency / which complexes break), where it is validated; the mRNA wall is carried by the regulatory layer and remains a "
           "similarity-representation gap (wall_test: model 0.38 -> oracle 0.62). This is a clean honest negative, not a failure to try: "
           "the layers are genuinely distinct.")
        + " HONEST CAVEATS: NEXUS-2HOP is pure topology (no measured profiles), so it is a mechanistic prior not a fitted model; the "
        "obligate weight is the nexus_ko proxy (co-dependency + small-complex), not per-pair interface energy; and the readout is "
        "steady-state mRNA in one cell type.")
    print(f"\nVERDICT: {verdict}")

    out = {"cell_type": "K562", "n_scored": scored, "n_twohop_topology_fires": n_fire, "n_twohop_profile_fires": n_firep,
           "tau": TAU, "K": K,
           "recall_specific_all": {name: mean[f"spec_{name}_{K}"] for name, _ in order},
           "recall_specific_byK_all": {f"@{k}": {name: mean[f"spec_{name}_{k}"] for name, _ in order} for k in KS},
           "recall_specific_where_topology_fires": {name: meanf.get(f"spec_{name}_{K}") for name, _ in order} if n_fire else None,
           "recall_specific_where_profile_fires": {name: meanfp.get(f"spec_{name}_{K}") for name, _ in order} if n_firep else None,
           "twohop_beats_null_where_fires": bool(twohop_real),
           "combo_adds_over_model_all": bool(combo_adds), "combo_topology_adds_where_fires": bool(combo_adds_fire),
           "combo_profile_adds_where_fires": bool(combop_adds_fire), "nexus_advances_wall": any_add,
           "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "nexus_wall.json", "w"), indent=1)
    print("\n  -> outputs/orphan/nexus_wall.json")
    return out


if __name__ == "__main__":
    main()
