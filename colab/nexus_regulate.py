"""nexus_regulate — wire NEXUS (protein-mutation health) into the regulation stack. The two layers meet at the TF: a TF is
a protein, so NEXUS converts a mutation in it into a graded ACTIVITY, and the regulation layer says what that TF controls.

  NEXUS (near-field, validated structure):  mutation (ΔΔG_fold, ΔΔG_bind) -> activity a in [0,1] (LOF) or >1 (GOF)
                                             a = folded_fraction(ΔΔG_fold) * bound_fraction(ΔΔG_bind)   (soft-AND)
  Regulation layer (this project):           the TF's DIRECT targets + sign (TRRUST curated regulon; the well-powered map)
  Chain:                                     each direct target's predicted first-order change = (a - 1) x sign
                                             activated target (TF activates it):  LOF -> target DOWN ;  GOF -> UP
                                             repressed target (TF represses it):  LOF -> target UP   ;  GOF -> DOWN

HONEST SCOPE: this composes two things this project validated SEPARATELY -- NEXUS's structural activity sensor
(~0.5 Pearson / 0.77 hotspot AUC, near-field) and the TF's curated direct regulon (binding->regulation 2.3x, GATA1). It gives
the FIRST-ORDER regulatory consequence of a TF mutation (which direct targets move, and which way). It does NOT predict the
genome-wide downstream cascade -- that is the dynamics wall measured all session (graph propagation ~chance). And NEXUS's
extrinsic sensor needs a KNOWN protein interface; folding ΔΔG is general. So: a mechanistic mutation->direct-regulation chain,
honestly bounded at the direct regulon.
"""
import json
import os
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def tf_activity(ddg_fold, ddg_bind, inhibitory=False):
    """NEXUS graded activity of the TF after a mutation (reuses the validated dual-sensor node)."""
    import nexus
    return nexus.directed_activity(ddg_fold, ddg_bind, inhibitory=inhibitory)


def _regulon(tf):
    """TF's curated direct targets with sign (TRRUST v2; +1 activation, -1 repression, 0 unknown)."""
    tr = json.load(open(OUT / "trrust_regulon.json"))
    out = []
    for t in tr.get("tf_targets", {}).get(tf, []):
        sign = int(t[1]) if len(t) > 1 else 0        # TRRUST: 1=Activation, -1=Repression, 0=Unknown
        out.append((t[0], sign))
    return out


def mutate_tf(tf, ddg_fold=0.0, ddg_bind=0.0, inhibitory=False, limit=25):
    """the cross-layer call: a mutation in TF -> NEXUS activity -> predicted first-order change of its direct regulon."""
    a = tf_activity(ddg_fold, ddg_bind, inhibitory)
    reg = _regulon(tf)
    delta = a - 1.0                                   # <0 = LOF (lost activity), >0 = GOF
    up, down, unknown = [], [], []
    for g, sign in reg:
        if sign == 0:
            unknown.append(g); continue
        # target change = sign * (a-1): activated target FOLLOWS activity (LOF->down, GOF->up); repressed is anti-correlated
        eff = sign * delta                             # >0 target rises, <0 target falls
        if abs(eff) < 0.05:                             # near-neutral activity -> target does not move
            continue
        (up if eff > 0 else down).append((g, round(eff, 2)))
    up.sort(key=lambda x: -x[1]); down.sort(key=lambda x: x[1])
    call = ("LOF (activity lost)" if a < 0.85 else "GOF (activity raised)" if a > 1.15 else "near-neutral")
    return {"tf": tf, "ddg_fold": ddg_fold, "ddg_bind": ddg_bind, "inhibitory_interface": inhibitory,
            "nexus_activity": round(a, 3), "call": call,
            "n_direct_targets": len(reg), "n_signed": len(up) + len(down),
            "targets_up": up[:limit], "targets_down": down[:limit], "targets_unknown_sign": unknown[:10]}


def main():
    print("=" * 96)
    print("NEXUS -> REGULATION — a TF mutation's protein health (NEXUS) scaled onto its direct regulon")
    print("=" * 96)
    print("\n  the two layers meet at the TF: NEXUS gives its activity after a mutation; the regulon gives what it controls.\n")
    # NOTE: folding is conservative (proteins are marginally stable) -- mild ΔΔG (1-3) barely dents activity; only a severe
    # destabiliser (~>=6 kcal/mol) or an interface-breaking binding ΔΔG drives LOF. That is biologically correct.
    for tf, ddgf, ddgb, inh, label in [("GATA1", 7.5, 0.0, False, "severe destabilising fold mutation"),
                                       ("GATA1", 0.0, 9.0, False, "interface-breaking mutation (extrinsic)"),
                                       ("GATA1", 2.0, 0.0, False, "mild fold mutation")]:
        r = mutate_tf(tf, ddgf, ddgb, inh)
        print(f"  {tf}  {label}: ΔΔG_fold={ddgf} ΔΔG_bind={ddgb} -> NEXUS activity {r['nexus_activity']:.2f}  [{r['call']}]")
        n_move = len(r["targets_down"]) + len(r["targets_up"])
        print(f"    {r['n_direct_targets']} curated direct targets ({r['n_signed']} signed) -> {n_move} predicted to move:")
        if r["targets_down"]:
            print(f"      DOWN: {', '.join(g for g,_ in r['targets_down'][:8])}")
        if r["targets_up"]:
            print(f"      UP:   {', '.join(g for g,_ in r['targets_up'][:8])}")
        if n_move == 0:
            print("      (activity near wild-type -> no substantial first-order change)")
    print("\n  [honest] NEXUS activity = validated near-field structure (~0.5 Pearson / 0.77 hotspot AUC); direct regulon =")
    print("  curated (binding->regulation 2.3x). This is the FIRST-ORDER consequence -- which direct targets move & which way.")
    print("  It does NOT predict the genome-wide downstream cascade (the dynamics wall). Extrinsic sensor needs a known interface.")
    out = {"demo_GATA1_LOF": mutate_tf("GATA1", 3.0, 0.0),
           "note": "Cross-layer integration: NEXUS (protein-mutation dual-sensor activity, soft-AND of folding x binding "
                   "fractions, validated near-field ~0.5 Pearson/0.77 hotspot AUC) wired into the regulation layer via the TF. "
                   "A TF mutation -> NEXUS activity a in [0,1] (LOF) or >1 (GOF) -> each direct target (TRRUST signed regulon) "
                   "predicted to move by sign*(a-1). Gives the FIRST-ORDER regulatory consequence of a TF mutation; does NOT "
                   "predict the genome-wide cascade (dynamics wall). Extrinsic sensor needs a known protein interface; folding "
                   "is general. Composes two separately-validated layers -- honest, bounded at the direct regulon."}
    json.dump(out, open(OUT / "nexus_regulate.json", "w"), indent=1)
    print("\n  -> outputs/orphan/nexus_regulate.json")
    return out


if __name__ == "__main__":
    main()
