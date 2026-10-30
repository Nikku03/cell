"""tfbs_score — score how a DNA variant changes a TF's binding, using the TF's measured motif. This is the HONEST,
tractable version of "make NEXUS work on protein-DNA": NOT a retrofit of NEXUS's amino-acid physics (protein-DNA is
different chemistry — 4 bases, phosphate-backbone electrostatics, groove/base-stacking, and a 3D SHAPE readout), but the
mature, measured-data approach — a position weight matrix (PWM) from JASPAR (743 vertebrate TFs, tf_motifs.json, derived
from HT-SELEX/ChIP), scored as a log-odds over a sliding window on both strands.

Neighbouring nucleotides ARE accounted for: the PWM spans the whole motif window, so a variant's effect depends on where it
sits (core vs flank) and on the surrounding bases. The one thing a PWM does NOT model is DNA SHAPE ("rotation" — minor
groove width, roll, helical twist, propeller) and dinucleotide dependencies; those are the Rohs DNA-shape / higher-order
add-on (a modest, validated improvement over PWM alone), left as a documented extension since it needs the pentamer shape
table.

HONEST SCOPE: PWM log-odds predicts INTRINSIC, in-vitro sequence PREFERENCE well (literature AUC ~0.85-0.9, validated by
construction). It does NOT predict in-vivo binding or regulation — that depends on chromatin, cofactors and TF
concentration (the same context wall this project keeps hitting). So this is the right tool for "does a non-coding /
regulatory variant strengthen or weaken a TF's site" — variant interpretation — not for "the TF binds here and turns this
gene on in this cell".
"""
import os
import json, math
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
COMP = str.maketrans("ACGT", "TGCA")


def _load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    raw = json.load(open(OUT / "tf_motifs.json"))["motifs"]
    pwm = {}                                                  # gene name -> log-odds matrix [L][4] + id
    for gi, rec in raw.items():
        gi = int(gi)
        if gi >= len(names):
            continue
        M = rec["pwm"]                                        # 4 x L counts (rows ACGT)
        L = len(M[0])
        lo = []
        for p in range(L):
            col = [M[b][p] for b in range(4)]
            tot = sum(col) + 4 * 0.25                         # pseudocount
            lo.append([math.log(((col[b] + 0.25) / tot) / 0.25) for b in range(4)])   # log-odds vs uniform
        pwm.setdefault(names[gi], {"id": rec["id"], "lo": lo})
    return pwm


NT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _score_strand(lo, seq):
    L = len(lo)
    best = -1e9
    for i in range(len(seq) - L + 1):
        s = 0.0
        ok = True
        for p in range(L):
            b = NT.get(seq[i + p])
            if b is None:
                ok = False; break
            s += lo[p][b]
        if ok and s > best:
            best = s
    return best


def score(tf, seq, pwm=None):
    """best log-odds of the TF's motif over a DNA window, both strands (higher = stronger predicted binding)."""
    if pwm is None:
        pwm = _load()
    if tf not in pwm:
        return None
    lo = pwm[tf]["lo"]
    seq = seq.upper()
    rc = seq.translate(COMP)[::-1]
    return max(_score_strand(lo, seq), _score_strand(lo, rc))


def variant_effect(tf, seq, pos, alt, pwm=None):
    """how a single-base variant at 0-based `pos` (ref -> alt) changes the TF's best binding score. Negative = weakens."""
    if pwm is None:
        pwm = _load()
    seq = seq.upper()
    ref_s = score(tf, seq, pwm)
    if ref_s is None:
        return None
    mut = seq[:pos] + alt.upper() + seq[pos + 1:]
    alt_s = score(tf, mut, pwm)
    return {"tf": tf, "motif": pwm[tf]["id"], "ref_score": round(ref_s, 2), "alt_score": round(alt_s, 2),
            "delta": round(alt_s - ref_s, 2),
            "call": "abolishes/weakens site" if alt_s - ref_s < -1.5 else
                    ("creates/strengthens site" if alt_s - ref_s > 1.5 else "little effect")}


def main():
    print("=" * 90)
    print("TFBS-SCORE — how a DNA variant changes TF binding (JASPAR PWM log-odds; neighbour-aware window)")
    print("=" * 90)
    pwm = _load()
    print(f"\n  motifs loaded: {len(pwm)} TFs (JASPAR2024)")
    # demo on ARNT (E-box CACGTG) in a 21-bp context
    tf = "ARNT"
    ctx = "TTGACACGTGCTAGGCTAAGT"          # contains the E-box CACGTG at pos 5-10
    core = ctx.index("CACGTG")
    print(f"\n  TF {tf} (motif {pwm[tf]['id']}, consensus CACGTG) in context: {ctx}")
    print(f"    wild-type best score: {score(tf, ctx, pwm):.2f}")
    print("    variant effects (showing neighbouring-nucleotide / position dependence):")
    for pos, alt, label in [(core + 1, "T", "CORE base (A->T inside CACGTG)"),
                            (core + 3, "A", "CORE base (G->A inside CACGTG)"),
                            (core - 3, "C", "FLANK base (3 bp upstream of core)"),
                            (0, "G", "FAR flank (edge of window)")]:
        v = variant_effect(tf, ctx, pos, alt, pwm)
        print(f"      pos {pos:2d} {ctx[pos]}->{alt}  {label:36s} Δscore {v['delta']:+.2f}  [{v['call']}]")
    out = {"n_tf": len(pwm), "source": "JASPAR2024 CORE vertebrates (tf_motifs.json)",
           "note": "PWM log-odds variant-effect scorer for TF binding sites. Neighbour-aware (motif window). Predicts "
                   "INTRINSIC in-vitro preference (validated by construction; literature AUC ~0.85-0.9), NOT in-vivo "
                   "binding/regulation (chromatin/cofactor/concentration = the context wall). DNA-shape ('rotation') and "
                   "dinucleotide dependencies are the documented Rohs add-on. This is the honest protein-DNA analogue of "
                   "NEXUS's binding node — a NEW motif/shape engine, not a retrofit of the amino-acid physics."}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "tfbs_score.json", "w"), indent=1)
    print(f"\n  -> {OUT/'tfbs_score.json'}")
    return out


if __name__ == "__main__":
    main()
