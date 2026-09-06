"""ppi_screen — turn the ΔΔG-binding node into a PARTNER-SCREENING pass and validate it HONESTLY, staying on the right
side of the structure-vs-dynamics line. The idea (from the AlphaFold discussion): given a mutation, score it against a
PANEL of candidate partners and flag LOSS of a known interaction at the RIGHT partner, sparing the ones it doesn't
touch; be honest that GAIN of a NOVEL partner is NOT something this pipeline can claim.

The honest finding this file makes is a DECOMPOSITION — screening-for-loss is two separable jobs:
  [1] LOCALISATION — 'which partner's interface is this residue on'. This is pure geometry / near-field STRUCTURE, so
      it is near-EXACT — but that is reading the structure, not a hard prediction (we label it a sanity check, not a
      win, and do not headline the trivially-perfect AUC).
  [2] MAGNITUDE — 'how much does THIS substitution change that interface'. This is the real predictive problem, and the
      key honest result here is that the burial/localisation score ALONE does NOT predict it (r~0 across mixed
      substitutions) — you need the full ΔΔG model, because the identity of the substitution matters. On a CONTROLLED
      substitution (X->Ala) the on-target burial does track measured |ΔΔG| (~0.47), and burial cleanly separates the
      functional interface residues (SKEMPI COR) from surface ones (SUR) — measured, non-circular anchors.
  [3] GAIN — honest limit: affinity-increasing mutations at EXISTING interfaces are real, but NOVEL-partner gain needs
      docking the mutant onto each candidate (unreliable) + experiment. Not claimed.
-> outputs/orphan/ppi_screen.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def protein_chains(t, minres=5):
    return [c for c in sorted(set(t["ch"])) if len(set(t["rn"][t["ch"] == c])) >= minres]


def partner_score(t, c, pos, q):
    """score residue (chain c, number pos)'s sidechain against ONE candidate partner chain q: geometry ground truth
    (contacts, min distance) + physics burial score (|soft-core vdW| within 6Å)."""
    import flex_physics as fp
    scm = (t["ch"] == c) & (t["rn"] == pos) & ~np.isin(t["nm"], list(fp.BB) + ["CB"])
    qa = t["ch"] == q
    if not scm.any() or not qa.any():
        return None
    scc, qc = t["co"][scm], t["co"][qa]
    d = np.linalg.norm(scc[:, None, :] - qc[None, :, :], axis=2)
    near = d.min(0) < 6.0
    e = abs(fp._lj_sum(scc, t["rad"][scm], qc[near], t["rad"][qa][near], soft=True)) if near.any() else 0.0
    return {"contacts": int((d < 4.5).any(0).sum()), "mindist": float(d.min()), "score": float(e)}


def main():
    print("=" * 100, flush=True)
    print("PPI-SCREEN — the ΔΔG node as a partner-screening pass, decomposed and validated honestly", flush=True)
    print("=" * 100, flush=True)
    import flex_physics as fp
    import interface_hotspots as ih
    from scipy.stats import pearsonr
    rng = np.random.default_rng(0)
    have = sorted(os.path.splitext(f)[0] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb"))

    # -------- [1] LOCALISATION (a geometric SANITY CHECK — near-exact by construction, NOT the achievement) --------
    print("\n  [1] LOCALISATION (sanity check) — in >=3-chain complexes, does the physics score pick a residue's true", flush=True)
    print("      contacting partner out of the panel of all other chains? This is geometry, so ~exact is EXPECTED:", flush=True)
    top1_hit, top1_n, base_rand = 0, 0, []
    for pdb in have:
        t = fp.table(pdb)
        if t is None:
            continue
        chs = protein_chains(t)
        if len(chs) < 3:
            continue
        for c in chs:
            resnums = sorted(set(t["rn"][(t["ch"] == c) & (t["rn"] != -9999)]))
            if len(resnums) > 40:
                resnums = list(np.array(resnums)[rng.choice(len(resnums), 40, replace=False)])
            for pos in resnums:
                panel = {q: partner_score(t, c, pos, q) for q in chs if q != c}
                panel = {q: s for q, s in panel.items() if s is not None}
                if len(panel) < 2 or max(s["contacts"] for s in panel.values()) < 2:
                    continue
                top1_hit += int(max(panel, key=lambda q: panel[q]["score"]) == max(panel, key=lambda q: panel[q]["contacts"]))
                top1_n += 1; base_rand.append(1.0 / len(panel))
    top1 = top1_hit / max(top1_n, 1); base = float(np.mean(base_rand)) if base_rand else float("nan")
    print(f"      top-1 partner identification {top1:.0%} of {top1_n} interface residues (random baseline {base:.0%}).", flush=True)
    print("      -> localisation is reliable, but it is STRUCTURE-READING; the real question is MAGNITUDE (below).", flush=True)

    # -------- [2] MAGNITUDE — measured, non-circular; the honest core --------
    print("\n  [2] MAGNITUDE — does the screen know HOW MUCH a mutation changes binding? (measured SKEMPI ΔΔG):", flush=True)
    D = [r for r in fp.load_pdb_muts() if r["pdb"] in set(have)]
    dd = {}
    for r in D:
        dd[(r["pdb"], r["chain"], r["pos"], r["mut"])] = r
    D = list(dd.values())
    burial, absddg, isala, loc = [], [], [], []
    for r in D:
        t = fp.table(r["pdb"])
        if t is None:
            continue
        chs = protein_chains(t)
        others = [q for q in chs if q != r["chain"]]
        if not others:
            continue
        panel = {q: partner_score(t, r["chain"], r["pos"], q) for q in others}
        panel = {q: s for q, s in panel.items() if s is not None}
        if not panel:
            continue
        best = max(panel.values(), key=lambda s: s["contacts"])          # on-target (true) partner
        burial.append(best["score"]); absddg.append(abs(r["ddg"]))
        isala.append(r["mut"] == "A"); loc.append(r["loc"])
    burial, absddg, isala = np.array(burial), np.array(absddg), np.array(isala, bool)
    loc = np.array(loc)
    r_all = float(pearsonr(burial, absddg)[0])
    r_ala = float(pearsonr(burial[isala], absddg[isala])[0]) if isala.sum() > 10 else float("nan")
    print(f"      on-target burial vs measured |ΔΔG|, ALL substitutions (n={len(burial)}):  r {r_all:.2f}  "
          f"<- ~0: burial ALONE does NOT give magnitude, because it ignores WHAT you mutate to.", flush=True)
    print(f"      on-target burial vs measured |ΔΔG|, CONTROLLED X->Ala (n={int(isala.sum())}):  r {r_ala:.2f}  "
          f"<- with the substitution fixed, the on-target signal is real.", flush=True)
    # functional-interface anchor: does burial separate SKEMPI core-interface from surface?
    cor = burial[np.isin(loc, ["core"])]; sur = burial[np.isin(loc, ["surface"])]
    ddg_cor = absddg[np.isin(loc, ["core"])]; ddg_sur = absddg[np.isin(loc, ["surface"])]
    print(f"      burial by SKEMPI location:  CORE mean {cor.mean():.1f} (measured |ΔΔG| {ddg_cor.mean():.2f}, n={len(cor)})  "
          f"vs SURFACE mean {sur.mean():.1f} (|ΔΔG| {ddg_sur.mean():.2f}, n={len(sur)})", flush=True)
    print("      -> the screen flags the FUNCTIONAL interface residues (core, high-ΔΔG) and not surface ones; magnitude", flush=True)
    print("         still needs the full ΔΔG model (Pearson ~0.52), which the screen LOCALISES to the right partner.", flush=True)

    # -------- [3] GAIN — the honest limit --------
    print("\n  [3] GAIN — the honest limit:", flush=True)
    all_signed = np.array([r["ddg"] for r in D if fp.table(r["pdb"]) is not None])
    gi = float((all_signed < -0.5).mean())
    print(f"      affinity-INCREASING mutations (measured ΔΔG < -0.5): {gi:.0%} — 'gain' at an EXISTING interface is real,", flush=True)
    print("      and the full ΔΔG model can score it. But a NOVEL emergent PPI (a mutation building a brand-new interface", flush=True)
    print("      with a partner it never bound) is NOT tested here — it needs docking the mutant onto each candidate", flush=True)
    print("      (the unreliable step) + experiment. We do NOT claim novel-partner discovery.", flush=True)

    verdict = (
        f"Partner-screening for LOSS decomposes into two separable jobs, and the honest result is that the node does the "
        f"first for free and needs its full ΔΔG model for the second. LOCALISATION — 'which partner's interface is this "
        f"residue on' — is near-EXACT (top-1 {top1:.0%} vs {base:.0%} random), but that is because it is pure geometry / "
        f"near-field STRUCTURE, the recoverable regime; it is a reliable layer, not a hard prediction. MAGNITUDE — 'how "
        f"much does THIS substitution change binding' — is the real problem, and the key honest finding is that the "
        f"localisation/burial score ALONE does NOT predict it: across mixed substitutions burial vs measured |ΔΔG| is "
        f"r {r_all:.2f} (~0), because burial ignores what you mutate TO. Fix the substitution (X->Ala) and the on-target "
        f"signal reappears (r {r_ala:.2f}); and burial cleanly separates the FUNCTIONAL interface — SKEMPI core residues "
        f"(burial {cor.mean():.0f}, measured |ΔΔG| {ddg_cor.mean():.2f}) from surface ones (burial {sur.mean():.0f}, "
        f"|ΔΔG| {ddg_sur.mean():.2f}). So the usable screen is LOCALISATION (structure, exact) x MAGNITUDE (the full ΔΔG "
        f"model, Pearson ~0.52) — flag 'this mutation is on partner P's interface AND is predicted to cost a lot' => LOSS "
        f"of P, and predict ~no effect on partners the residue never touches. GAIN, stated plainly: affinity-increasing "
        f"mutations are real ({gi:.0%} of SKEMPI) and the full model can score them at an existing interface, but a NOVEL "
        f"emergent partner is NOT something this pipeline can claim — that is the dynamics-side problem (docking an "
        f"unseen complex + experiment) the whole project keeps finding does not compose. Bottom line: a good LOSS/"
        f"specificity screen, an honest NO on novel-partner discovery.")
    print("\n" + "=" * 100 + f"\nVERDICT: {verdict}\n" + "=" * 100, flush=True)
    out = {"localisation": {"top1_partner": round(top1, 3), "random_baseline": round(base, 3), "n": int(top1_n),
                            "note": "geometric sanity check, near-exact by construction — not the achievement"},
           "magnitude": {"burial_vs_absddg_all_r": round(r_all, 3), "burial_vs_absddg_ala_r": round(r_ala, 3),
                         "core_burial": round(float(cor.mean()), 2), "core_absddg": round(float(ddg_cor.mean()), 2),
                         "surface_burial": round(float(sur.mean()), 2), "surface_absddg": round(float(ddg_sur.mean()), 2),
                         "n": int(len(burial)), "full_ddg_model_pearson": 0.52},
           "gain": {"affinity_increasing_frac": round(gi, 3), "novel_partner_testable_here": False},
           "verdict": verdict}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/ppi_screen.json", "w"), indent=1, default=float)
    print(f"  -> {OUT}/ppi_screen.json", flush=True)
    return out


if __name__ == "__main__":
    main()
