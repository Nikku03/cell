"""cell_conditions — infer the ACTIVATION CONDITION of a pathway by tracing its genes back to the condition-responsive
TFs that control them. Implements the subtractive strategy: pathways whose genes show NO enrichment for any stress/signal
TF are CONSTITUTIVE (always-on — the housekeeping machinery); pathways enriched for a condition-TF's targets are
CONDITIONAL, and the top TF names the trigger (HIF1A → hypoxia, SREBF → low sterol, ATF4/XBP1 → ER stress, TP53 → DNA
damage, NF-kB → inflammation, STAT → cytokine/IFN, HSF1 → heat, …).

The backward trace (pathway gene ← controlling TF) is a STRUCTURAL query on the curated directed regulatory graph
(cell_complete["reg"], 612k edges) scored by a HYPERGEOMETRIC enrichment — NOT the dynamical propagation that this project
measured fails at chance. Enrichment of a curated TF's targets in a gene set is an annotation-level inference (like
connect's corroboration), and it VALIDATES: on 111 pathways whose condition is stated in their name, the traced condition-TF
is top-1 60% / top-3 77% (chance 7% / 20%) — ~8x / 4x chance. HONEST LIMITS: condition-CATEGORY resolution (it can confuse
related stresses, e.g. hypoxia vs ER stress); only ~20 curated condition-TFs (major stress/signal axes, not all); the reg
edges are annotation (ChIP + curated), so this is "regulated-by" not "measured-functional-in-K562". A hypothesis generator
for what turns a pathway on, not a proven trigger.
"""
import json, collections, math
from pathlib import Path
from scipy.stats import hypergeom
OUT = Path("outputs/orphan")

# condition-responsive TF -> the human-readable stimulus it reports (curated; expandable). Deliberately the transient
# STRESS/SIGNAL axes — NOT broad proliferation TFs (MYC/E2F1), whose targets are the constitutive growth machinery and
# would mislabel housekeeping pathways as "growth-activated" and outcompete the specific stress signal.
CONDITION = {
    "HIF1A": "hypoxia (low O2)", "ATF4": "amino-acid starvation / integrated stress",
    "XBP1": "ER stress (unfolded-protein response)", "ATF6": "ER stress (unfolded-protein response)",
    "DDIT3": "ER stress / apoptosis (CHOP)", "NFE2L2": "oxidative stress", "TP53": "DNA damage / cellular stress",
    "NFKB1": "inflammation / immune (NF-kB)", "RELA": "inflammation / immune (NF-kB)",
    "STAT1": "interferon / cytokine", "STAT3": "IL-6-type cytokine", "SREBF1": "low sterol / lipogenic",
    "SREBF2": "low cholesterol", "SMAD3": "TGF-beta signalling", "HSF1": "heat shock / proteotoxic stress",
    "FOXO3": "growth-factor withdrawal / starvation",
}
THRESH = 2.0        # -log10(hypergeom p); >= this = a real condition, below = constitutive/always-on
MINK = 50           # ignore condition-TFs with fewer than this many curated targets


def _load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    n2i = {x: i for i, x in enumerate(names)}
    reg = collections.defaultdict(set)
    sout = collections.defaultdict(list)                 # TF name -> [(target idx, sign)]  (signed edges only)
    sin = collections.defaultdict(list)                  # target idx -> [(src name, sign)]  (signed edges only)
    for s, t, sg in D["reg"]:
        if s < len(names) and t < len(names):
            reg[names[s]].add(t)                          # TF name -> set of target indices
            if sg != 0:
                sout[names[s]].append((t, sg)); sin[t].append((names[s], sg))
    paths = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    sgn = {"out": sout, "in": sin, "n2i": n2i}
    return names, reg, paths, len(names), sgn


def _direction(tf, mem_set, sgn):
    """does the condition-TF ACTIVATE or REPRESS the pathway, and is there feedback from the pathway back to the TF?
    Uses the signed regulatory edges (~73% reliable, this project's measurement)."""
    edges = [(t, sg) for t, sg in sgn["out"].get(tf, []) if t in mem_set]
    pos = sum(1 for _, sg in edges if sg > 0)
    neg = sum(1 for _, sg in edges if sg < 0)
    if pos + neg == 0:
        direction, frac = "unsigned", None
    else:
        direction = "activates" if pos >= neg else "represses"
        frac = round(max(pos, neg) / (pos + neg), 2)
    tfi = sgn["n2i"].get(tf)
    fb = [(src, sg) for src, sg in sgn["in"].get(tfi, []) if sgn["n2i"].get(src) in mem_set]
    fb_neg = [s for s, sg in fb if sg < 0]
    fb_pos = [s for s, sg in fb if sg > 0]
    feedback = None
    if fb_neg:
        feedback = ("negative", fb_neg[:4])              # pathway represses its own activator -> self-limiting
    elif fb_pos:
        feedback = ("positive", fb_pos[:4])              # pathway reinforces its activator -> amplifying / switch
    return {"direction": direction, "sign_frac": frac, "n_signed": pos + neg,
            "feedback": feedback}


def _score(mem, reg, N):
    """hypergeometric enrichment (-log10 p) of each condition-TF's targets among a pathway's genes."""
    s = {i for i in mem if i < N}
    n = len(s)
    out = {}
    for tf in CONDITION:
        K = len(reg.get(tf, set()))
        k = len(s & reg.get(tf, set()))
        if K < MINK or k == 0 or n == 0:
            continue
        out[tf] = -math.log10(max(hypergeom.sf(k - 1, N, K, n), 1e-300))
    return sorted(out.items(), key=lambda x: -x[1])


def infer(pathway, names=None, reg=None, paths=None, N=None, sgn=None):
    """the activation-condition call for one pathway: constitutive (always-on) or a ranked list of candidate triggers,
    each with its DIRECTION (activates/represses) and any feedback loop back onto the trigger TF."""
    if names is None:
        names, reg, paths, N, sgn = _load()
    if pathway not in paths:
        return None
    mem_set = {i for i in paths[pathway] if i < N}
    ranked = _score(paths[pathway], reg, N)
    strong = []
    for tf, v in ranked:
        if v < THRESH:
            continue
        df = _direction(tf, mem_set, sgn) if sgn else {"direction": "unsigned", "feedback": None, "sign_frac": None}
        strong.append({"tf": tf, "score": round(v, 1), "condition": CONDITION[tf],
                       "direction": df["direction"], "sign_frac": df["sign_frac"], "feedback": df["feedback"]})
    return {"pathway": pathway, "n_genes": len(paths[pathway]),
            "constitutive": len(strong) == 0, "conditions": strong[:4],
            "top": [(tf, round(v, 1)) for tf, v in ranked[:4]]}


def validate(names=None, reg=None, paths=None, N=None):
    """does tracing recover the KNOWN condition? test on pathways whose stimulus is stated in their name."""
    if names is None:
        names, reg, paths, N, _sgn = _load()
    KW = {"hypoxia": ["HIF1A"], "unfolded protein": ["ATF4", "XBP1", "ATF6", "DDIT3"],
          "endoplasmic reticulum stress": ["ATF4", "XBP1", "ATF6", "DDIT3"], "oxidative": ["NFE2L2"],
          "DNA damage": ["TP53"], "interferon": ["STAT1", "NFKB1", "RELA"], "cholesterol": ["SREBF1", "SREBF2"],
          "sterol": ["SREBF1", "SREBF2"], "TGF": ["SMAD3"], "heat": ["HSF1"], "inflamm": ["NFKB1", "RELA", "STAT3"]}
    h1 = h3 = n = 0
    for p, mem in paths.items():
        expect = [tf for kw, tfs in KW.items() if kw.lower() in p.lower() for tf in tfs]
        if not expect or not (5 <= len(mem) <= 200):
            continue
        ranked = [tf for tf, _ in _score(mem, reg, N)]
        if not ranked:
            continue
        n += 1
        h1 += ranked[0] in expect
        h3 += any(tf in expect for tf in ranked[:3])
    return {"n": n, "top1": round(h1 / n, 2), "top3": round(h3 / n, 2)}


def scan(names=None, reg=None, paths=None, N=None, limit=12):
    """the leftover: pathways whose condition is NOT stated in their name but IS inferred — discovered triggers."""
    if names is None:
        names, reg, paths, N, _sgn = _load()
    stated = ["hypoxia", "unfolded protein", "endoplasmic reticulum", "oxidative", "dna damage", "interferon",
              "cholesterol", "sterol", "tgf", "heat", "inflamm", "cytokine", "starvation", "stress", "immune"]
    out = []
    for p, mem in paths.items():
        if not (8 <= len(mem) <= 120) or any(k in p.lower() for k in stated):
            continue
        r = _score(mem, reg, N)
        if r and r[0][1] >= 4.0:                          # a confident inferred condition on an un-named pathway
            out.append((round(r[0][1], 1), p, r[0][0], CONDITION[r[0][0]]))
    out.sort(reverse=True)
    return out[:limit]


def main():
    print("=" * 92)
    print("CELL-CONDITIONS — infer a pathway's activation trigger from the TFs that control its genes")
    print("=" * 92)
    names, reg, paths, N, sgn = _load()
    v = validate(names, reg, paths, N)
    print(f"\n  VALIDATION (n={v['n']} pathways whose condition is stated in the name):")
    print(f"    traced condition-TF recovered: top-1 {v['top1']:.0%}   top-3 {v['top3']:.0%}   (chance ~7% / 20%)")
    print("\n  CONSTITUTIVE (always-on — no condition-TF enrichment):")
    for p in ["Eukaryotic Translation Elongation", "rRNA processing", "Citric acid cycle (TCA cycle)"]:
        r = infer(p, names, reg, paths, N, sgn)
        if r:
            print(f"    {p[:42]:42s}: {'CONSTITUTIVE' if r['constitutive'] else r['conditions'][0]['condition']}")
    print("\n  CONDITIONAL (trigger + direction + feedback):")
    for p in ["Cellular response to hypoxia", "Regulation of cholesterol biosynthesis by SREBP (SREBF)",
              "TP53 Regulates Transcription of Cell Death Genes"]:
        r = infer(p, names, reg, paths, N, sgn)
        if r and r["conditions"]:
            c = r["conditions"][0]
            fb = f"  · {c['feedback'][0]} feedback ({','.join(c['feedback'][1])})" if c["feedback"] else ""
            print(f"    {p[:40]:40s}: {c['condition']} — {c['direction'].upper()} [{c['tf']} {c['score']}]{fb}")
    print("\n  DISCOVERED (condition NOT in the pathway name, but inferred — the 'leftover'):")
    for sc, p, tf, cond in scan(names, reg, paths, N, 8):
        print(f"    {p[:50]:50s} → {cond}  [{tf} {sc}]")
    out = {"validation": v, "n_conditions_curated": len(CONDITION),
           "note": "Infer a pathway's activation trigger by hypergeometric enrichment of its genes among condition-TF "
                   f"targets (curated directed reg edges). Validated top-1 {v['top1']}, top-3 {v['top3']} on "
                   f"{v['n']} condition-named pathways (chance ~0.07/0.20). Constitutive pathways show no enrichment "
                   "(always-on). Condition-CATEGORY resolution; ~20 curated condition-TFs; reg edges are annotation "
                   "(regulated-by, not measured-functional). A hypothesis generator, not a proven trigger."}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "cell_conditions.json", "w"), indent=1)
    print(f"\n  -> {OUT/'cell_conditions.json'}")
    return out


if __name__ == "__main__":
    main()
