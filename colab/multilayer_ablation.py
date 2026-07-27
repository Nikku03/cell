"""MULTI-LAYER REASONING DRIVEN BY THE CORRECTED ABLATION, not by the knockout gene's own annotation.

WHAT CHANGES FROM multilayer_reason.py, AND WHY IT SHOULD MATTER.

The incumbent (precision 0.0786, the best of seven methods) decides which stress sensors fire by reading the
KNOCKOUT GENE'S OWN annotation: if the gene is mitochondrial, the mito sensor fires. That is layer 1 wearing the
costume of layer 6. It cannot see a lesion whose consequence lands somewhere the knockout gene is never annotated
-- which is the entire point of having layers.

The corrected ablation supplies what was missing: for each gene, the specific reactions that STOP when it is
removed, the products that then go unmade, and the reactions downstream that starve. So the sensor can be driven by
WHERE THE DAMAGE LANDS rather than by where the gene lives:

    knockout gene  ->  reactions it stops  ->  products nothing else makes  ->  reactions that starve
                                                                                        |
                                            classify THOSE reactions (compartment + pathway of their own genes)
                                                                                        |
                                                                            sensor fires -> TF -> mRNA programme

THE CORRECTION IS THE EXPERIMENT, SO IT IS RUN BOTH WAYS. reaction_ablation.py found three defects that all
inflated STOPS: currency marked inconsistently per source (35,883 slots spuriously currency, O2 never currency),
disjoint id spaces between Reactome and Human-GEM, and "STOPS" counting species nothing in the network makes. If
the corrections matter for prediction, the corrected variant must beat the uncorrected one on the same sealed
protocol. Running only the corrected version would assume the answer.

    ABLATION       corrected currency set, cross-source name bridge, addressable removals only
    UNCORRECTED    exactly the pre-correction behaviour, same reasoner downstream
    SHUFFLED       corrected consequences permuted across genes -- if this matches, the ablation is contributing
                   nothing beyond the shape of its output
    (incumbent)    ko_predictions_multilayer.json, annotation-only, scored separately

WHAT WOULD FALSIFY THE WHOLE IDEA. If SHUFFLED matches ABLATION, the per-gene consequence sets carry no
information and the sensor is firing on set size alone. If UNCORRECTED matches ABLATION, the defects were real
bookkeeping errors but irrelevant to prediction, and the correction should be reported as hygiene rather than as
an improvement. Both outcomes are recorded as found.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilayer_reason import SENSORS, classify
from reaction_ablation import CURRENCY, norm

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAXPART = 60
MINFIRE = 1        # a sensor fires when at least this many broken reactions classify into it. Absolute, not a
                   # quantile: the top-quartile rule used by markov_multilayer guarantees 25% firing by
                   # construction, which is exactly the non-discrimination that sank it.


def consequences(steps, corrected):
    """Per gene: the reactions its removal stops, the products then unmade, and the reactions that starve.

    `corrected=False` reproduces the pre-correction behaviour exactly -- the currency flags as they came out of
    reaction_network.json (inconsistent between sources, unresolved numeric ids spuriously flagged), no
    cross-source name bridge, and no addressability filter. That is the control, so it must be the real old
    behaviour rather than a sketch of it.
    """
    def iscur(p):
        return (norm(p["name"]) in CURRENCY) if corrected else bool(p.get("currency"))

    def keys(p):
        k = [p["id"]]
        if corrected and p["type"] in ("METABOLITE", "DRUG", "OTHER"):
            k.append("nm:" + norm(p["name"]))
        return k

    produced_by, consumed_by = collections.defaultdict(set), collections.defaultdict(set)
    for sid, s in steps.items():
        for p in s["out"]:
            if not iscur(p):
                for k in keys(p):
                    produced_by[k].add(sid)
        for p in s["in"]:
            if not iscur(p):
                for k in keys(p):
                    consumed_by[k].add(sid)

    def others(p, sid, tbl):
        got = set()
        for k in keys(p):
            got |= tbl.get(k, set())
        return got - {sid}

    con = collections.defaultdict(lambda: {"stopped": set(), "lost": [], "starved": set()})
    for sid, s in steps.items():
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if len(gs) > MAXPART:
            continue
        cats = list(s["catalysts"])
        # which genes, if removed, stop THIS reaction?
        breakers = set()
        if len(cats) == 1:
            breakers |= set(cats)                       # sole catalyst
        for p in s["in"]:
            if iscur(p):
                continue
            if others(p, sid, produced_by):
                continue                                # buffered: another reaction makes it
            if corrected and not p["genes"]:
                continue                                # a small molecule: no knockout can remove it
            if corrected and p["type"] == "SET" and len(p["genes"]) > 1:
                continue                                # DefinedSet = alternatives, not required components
            breakers |= set(p["genes"]) if p["genes"] else {p["name"]}
        if not breakers:
            continue
        lost = [q for q in s["out"] if not iscur(q) and not others(q, sid, produced_by)]
        starved = set()
        for q in lost:
            starved |= others(q, sid, consumed_by)
        for g in breakers:
            c = con[g]
            c["stopped"].add(sid)
            c["lost"].extend(q["name"] for q in lost)
            c["starved"] |= starved
    return con


def main(cohort=""):
    # captured at entry: the loops below rebind a local named `tag` for the VARIANT, and an earlier version of
    # this file read the cohort out of that shadowed name, writing every holdout file as "...uncorrected.json".
    net = json.load(open(OUT / "reaction_network.json"))
    steps = {s["id"]: s for s in net["steps"]}
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    doss = json.loads((OUT / f"ko_pred_dossiers{cohort}.json").read_text())
    cmap = json.loads((OUT / "consequence_map.json").read_text())

    gene_pw = collections.defaultdict(list)
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            for g in p[2:]:
                gene_pw[g].append(p[0])

    # ---- lesion class of a REACTION: from the compartment it happens in and the pathways of its own genes ----
    rxn_class = {}
    for sid, s in steps.items():
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if len(gs) > MAXPART:
            continue
        where = comp.get(sid, {}).get("compartments", [])
        # sorted, not set-iteration order: an unsorted [:12] slice of a set of strings samples different genes on
        # every process because of hash randomisation, and the class counts moved by ~80 reactions between two
        # runs of the same code before this was pinned.
        pw = []
        for g in sorted(gs)[:12]:
            pw.extend(gene_pw.get(g, ())[:8])
        rxn_class[sid] = classify(sid, pw, where, None)
    cc = collections.Counter(c for v in rxn_class.values() for c in v)
    print(f"lesion class assigned to reactions: {dict(cc.most_common())}")
    print(f"  reactions with no class: {sum(1 for v in rxn_class.values() if not v):,}/{len(rxn_class):,}")

    variants = {}
    for tag, corrected in (("ablation", True), ("uncorrected", False)):
        con = consequences(steps, corrected)
        variants[tag] = con
        st = sorted(len(v["starved"]) for v in con.values())
        print(f"\n{tag.upper():<12} genes with a consequence {len(con):,}; "
              f"starved reactions median {st[len(st)//2]}, max {st[-1]:,}, "
              f"mean {sum(st)/max(len(st),1):.1f}")

    # SHUFFLED: corrected consequences reassigned to other genes. Set SIZES are preserved, identities destroyed.
    rng = np.random.default_rng(0)
    base = variants["ablation"]
    gk = sorted(base)
    perm = rng.permutation(len(gk))
    variants["shuffled"] = {gk[i]: base[gk[perm[i]]] for i in range(len(gk))}

    # ---- how many knockouts can the ablation see at all? ----
    seen = [k for k in doss if variants["ablation"].get(k, {}).get("stopped")
            or variants["ablation"].get(k, {}).get("starved")]
    print(f"\nCOVERAGE: the ablation sees a consequence for {len(seen)}/{len(doss)} sealed knockouts.")
    print(f"  blind to: {', '.join(sorted(set(doss) - set(seen))[:14])} ...")
    print("  These are spliceosome / nucleolar / nuclear-pore / complex-assembly subunits. They appear in the")
    print("  network only as INPUTS that some other complex-formation reaction also produces, so every ablation")
    print("  of them is BUFFERED and no reaction is ever recorded as stopping. The lesion-site view is therefore")
    print("  a supplement to the annotation view, not a replacement for it -- which the combined variant tests.")

    for tag, con in list(variants.items()) + [("combined", variants["ablation"]),
                                              ("fallback", variants["ablation"])]:
        preds, fired_n = {}, collections.Counter()
        for k, v in doss.items():
            c = con.get(k, {"stopped": set(), "lost": [], "starved": set()})
            broken = set(c["stopped"]) | set(c["starved"])
            weight = collections.Counter()
            for sid in broken:
                for cl in rxn_class.get(sid, ()):
                    weight[cl] += 1
            fired = sorted(cl for cl, w in weight.items() if w >= MINFIRE)
            if tag == "fallback":
                # POST-HOC RULE, DERIVED FROM n=8 AND THEREFORE WEAK. Of the 8 knockouts where the lesion site
                # fired a sensor the gene's own annotation did not, all 3 that gained hits (PCNA, POLE, RBBP5) had
                # NO annotation class at all, and all 5 that only added misses (PIAS4, PNO1, PPRC1, RAD21, RBM8A)
                # already had one. So: consult the lesion site only when annotation is silent. This is fitted to
                # the same 50 knockouts it is scored on -- it is a hypothesis to carry to fresh data, not a result.
                own = classify(k, (v.get("pathways") or []) + (cmap.get(k, {}).get("pathways_member") or []),
                               v.get("compartments") or cmap.get(k, {}).get("compartments") or [],
                               v.get("function"))
                fired = sorted(own) if own else fired
            if tag == "combined":
                # the incumbent's layer-1 view, UNIONED with the lesion site. Neither subsumes the other: the
                # annotation sees complex subunits the ablation is blind to, the ablation sees damage landing
                # where the knockout gene is not annotated.
                own = classify(k, (v.get("pathways") or []) + (cmap.get(k, {}).get("pathways_member") or []),
                               v.get("compartments") or cmap.get(k, {}).get("compartments") or [],
                               v.get("function"))
                for cl in own:
                    weight[cl] = weight.get(cl, 0) + MINFIRE
                fired = sorted(set(fired) | set(own))
            fired_n.update(fired)

            genes, seen = [], {k}

            def add(xs):
                for x in xs:
                    if x and x not in seen and "; " not in x:
                        seen.add(x)
                        genes.append(x)

            # LAYER 1 unchanged: a TF's own signed regulon is the most direct transcriptional consequence
            add(v["tf_targets_down"] + v["tf_targets_up"])
            n_reg = len(genes)
            # LAYER 6 driven by the LESION SITE: sensors ordered by how many broken reactions land in each class
            for cl in sorted(fired, key=lambda c_: -weight[c_]):
                add(SENSORS[cl]["programme"])
            n_sensor = len(genes) - n_reg
            if tag in ("combined", "fallback") and not fired:
                # the incumbent's layer-2-5 fallback, reproduced EXACTLY. Without it `combined` is not a superset
                # of the incumbent and any difference is uninterpretable -- the first run of this comparison had
                # PHF5A drop from 12 predictions to 1 for precisely this reason, which is not an ablation effect.
                add((cmap.get(k, {}).get("starved_genes") or [])[:15])
                add((v.get("reaction_partners") or [])[:15])
            preds[k] = {
                "variant": tag,
                "n_stopped": len(c["stopped"]), "n_starved": len(c["starved"]),
                "sensors_fired": fired,
                "class_weights": dict(weight.most_common(6)),
                "reasoning": (f"{len(c['stopped'])} reactions stop, {len(c['starved'])} starve; "
                              f"those reactions classify as {fired or 'nothing'}; "
                              f"{n_reg} regulon + {n_sensor} sensor-programme genes"),
                "n_regulon": n_reg, "n_sensor": n_sensor,
                "predicted_genes": genes[:60],
            }
        p = OUT / f"ko_predictions_abl_{tag}{cohort}.json"
        json.dump(preds, open(p, "w"), indent=1)
        sz = sorted(len(x["predicted_genes"]) for x in preds.values())
        nf = sorted(len(x["sensors_fired"]) for x in preds.values())
        print(f"\n  {tag:<12} sensors fired {dict(fired_n.most_common())}")
        print(f"  {'':<12} sensors/knockout median {nf[len(nf)//2]}, none {sum(1 for x in nf if x == 0)}/{len(nf)}"
              f"; prediction size median {sz[len(sz)//2]}, empty {sum(1 for s in sz if s == 0)}")
        print(f"  {'':<12} -> {p.name}")

    # a sensor that fires for nearly every knockout cannot discriminate; report the rate so it is visible
    print(f"\nFIRING RATES (a class firing on >half the knockouts is not a discriminator)")
    for tag in ("ablation", "uncorrected"):
        preds = json.loads((OUT / f"ko_predictions_abl_{tag}{cohort}.json").read_text())
        n = len(preds)
        r = collections.Counter(c for p_ in preds.values() for c in p_["sensors_fired"])
        print(f"  {tag:<12} " + ", ".join(f"{c} {v}/{n}" for c, v in r.most_common()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
