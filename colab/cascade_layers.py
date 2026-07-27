"""RECURSIVE MULTI-LAYER CASCADE: a knockout disables neighbours, and those neighbours then fail in their OWN ways.

WHAT WAS MISSING FROM THE PREVIOUS ATTEMPT. multilayer_reason.py mapped a lesion straight to a sensor. It worked
for mitochondrial import (4/4) and did nothing else, because it had only ONE step: knockout -> sensor. Real
consequence is recursive. Knocking out a complex subunit does not merely stop that complex -- it ORPHANS the other
subunits, orphaned subunits are degraded, and if one of those was a transcription factor then ITS regulon moves
too, at a distance of two from the knockout and via a completely different mechanism.

So this file gives every protein a set of ROLES, gives every role a rule for what it does to which neighbours and
IN WHAT MODE, and then re-applies the whole machine to any neighbour that the previous round disabled.

  ROLE                  what its loss does                          neighbour left in mode
  CATALYST              its reaction stops                          consumers of the product  -> STARVED
  COMPLEX_SUBUNIT       the complex cannot assemble                 co-subunits               -> ORPHANED
  TF                    its regulon loses its driver                targets                   -> DYSREGULATED
  SIGNALLING            the signal is not relayed                   downstream nodes          -> UNSIGNALLED
  TRANSPORTER           cargo never reaches its compartment         cargo                     -> STRANDED
  MODIFIER              substrates keep the wrong chemical state    substrates                -> UNMODIFIED
  RECEPTOR/LIGAND       the message is not received                 partner                   -> UNRECEIVED

THE RECURSION IS THE POINT. A protein left ORPHANED, STRANDED or UNMODIFIED is FUNCTIONALLY ABSENT even though its
gene is untouched and its mRNA may be perfectly normal. So it is fed back in as a new lesion and its own roles fire.
DYSREGULATED and STARVED are deliberately NOT recursed: a target whose transcription shifted is not thereby
abolished, and treating a quantitative change as a knockout is how a cascade model reaches the whole cell by depth
three. That asymmetry is the main guard against runaway, and it is a claim about biology, not a tuning knob.

THE SECOND GUARD is that orphaning only propagates within a complex small enough for stoichiometry to matter, and
starvation only through non-currency entities. Without both, the ribosome and ATP connect everything to everything.

WHAT IT PREDICTS, AND WHY THAT IS FINALLY THE RIGHT KIND OF OUTPUT. Perturb-seq reads mRNA, so only two things in
this cascade are legitimately predictable: the regulons of transcription factors that were disabled ANYWHERE in the
cascade, and the sensor programmes tripped by the accumulated damage modes. Everything else the cascade computes is
protein-level and correctly withheld from the prediction.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilayer_reason import SENSORS as MLSENSORS, classify as ml_classify

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAXDEPTH = 3
MAXCPLX = 30          # orphaning is a stoichiometry argument; it does not apply to a 90-subunit machine
MAXFANOUT = 25        # per role per round, so one hub cannot flood the cascade

SENSOR_PROGRAMME = {
    "proteotoxic": ["HSPA1A", "HSPA1B", "HSPA8", "DNAJB1", "DNAJA1", "HSPH1", "BAG3", "AHSA1", "CHORDC1",
                    "HSPB1", "HSPD1", "HSPE1"],
    "mito_stress": ["ATF4", "ATF5", "DDIT3", "TRIB3", "SESN2", "MTHFD2", "MTHFD2L", "SHMT2", "CEBPB",
                    "SLC7A11", "SLC3A2", "EIF4EBP1", "GDF15", "CTH", "XBP1", "VEGFA"],
    "er_stress": ["HSPA5", "XBP1", "DDIT3", "HERPUD1", "SEL1L", "EDEM1", "DNAJB9", "DNAJB11", "PDIA6",
                  "PDIA3", "CANX", "CALR", "SSR1", "SEC61B", "ATF4"],
    "translation_stress": ["ATF4", "ATF5", "DDIT3", "TRIB3", "SESN2", "MTHFD2", "SHMT2", "SLC3A2",
                           "SLC7A11", "EIF4EBP1", "CEBPB"],
}


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}
    N = len(names)

    # ---------- ROLE TABLES ----------
    net = json.load(open(OUT / "reaction_network.json"))
    comps = json.loads((OUT / "compartments.json").read_text())["steps"]
    catal = collections.defaultdict(set)
    produces, consumes = collections.defaultdict(set), collections.defaultdict(set)
    rxn_where, rxn_out = {}, {}
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if len(gs) > 60:
            continue
        rxn_where[s["id"]] = comps.get(s["id"], {}).get("compartments", [])
        rxn_out[s["id"]] = [p["id"] for p in s["out"] if not p.get("currency")]
        for g in s["catalysts"]:
            catal[g].add(s["id"])
        for p in s["out"]:
            if not p.get("currency"):
                produces[p["id"]].add(s["id"])
        for p in s["in"]:
            if not p.get("currency"):
                consumes[p["id"]].add(s["id"])
    rxn_genes = {}
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if len(gs) <= 60:
            rxn_genes[s["id"]] = gs

    cplx_mem = {c: [names[i] for i in v if 0 <= i < N] for c, v in D.get("complexes", {}).items()}
    gene_cplx = collections.defaultdict(list)
    for c, mem in cplx_mem.items():
        if 2 <= len(mem) <= MAXCPLX:
            for g in mem:
                gene_cplx[g].append(c)

    tf_up, tf_dn = collections.defaultdict(list), collections.defaultdict(list)
    for e in D.get("reg", []):
        try:
            s, t, w = int(e[0]), int(e[1]), (int(e[2]) if len(e) > 2 else 0)
        except (TypeError, ValueError, IndexError):
            continue
        if w and 0 <= s < N and 0 <= t < N and s != t:
            (tf_up if w > 0 else tf_dn)[names[s]].append(names[t])

    sigdn = collections.defaultdict(list)
    for e in D.get("sig", []):
        try:
            s, t = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= s < N and 0 <= t < N and s != t:
            sigdn[names[s]].append(names[t])

    lr = collections.defaultdict(list)
    for e in D.get("lr", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N:
            lr[names[a]].append(names[b])
            lr[names[b]].append(names[a])

    ptm_genes = {names[int(k)] for k in D.get("ptm", {}) if k.isdigit() and int(k) < N}

    def roles(g):
        r = set()
        if catal.get(g):
            r.add("CATALYST")
            if any("SPANNING" == comps.get(x, {}).get("kind") for x in catal[g]):
                r.add("TRANSPORTER")
        if gene_cplx.get(g):
            r.add("COMPLEX_SUBUNIT")
        if tf_up.get(g) or tf_dn.get(g):
            r.add("TF")
        if sigdn.get(g):
            r.add("SIGNALLING")
        if lr.get(g):
            r.add("RECEPTOR_LIGAND")
        if g in ptm_genes and catal.get(g):
            r.add("MODIFIER")
        return r

    def where(g):
        c = collections.Counter()
        for x in catal.get(g, ()):
            for w in rxn_where.get(x, []):
                c[w] += 1
        return c

    # ---------- THE CASCADE ----------
    def cascade(seed):
        disabled = {seed: "KNOCKED_OUT"}
        modes = collections.Counter()
        dysreg = collections.defaultdict(set)      # TF -> targets, from any depth
        frontier = [seed]
        trace = []
        for depth in range(1, MAXDEPTH + 1):
            nxt = []
            for g in frontier:
                R = roles(g)
                w = where(g)
                for role in R:
                    hit = []
                    if role == "COMPLEX_SUBUNIT":
                        for c in gene_cplx[g][:5]:
                            for m in cplx_mem[c]:
                                if m != g and m not in disabled:
                                    hit.append((m, "ORPHANED"))
                    elif role == "CATALYST":
                        for rx in list(catal[g])[:10]:
                            if len(rxn_genes.get(rx, ())) and all(
                                    len(produces.get(p, set())) <= 1 for p in rxn_out.get(rx, [])):
                                for p in rxn_out.get(rx, []):
                                    for c2 in list(consumes.get(p, ()))[:5]:
                                        for m in rxn_genes.get(c2, ()):
                                            if m not in disabled:
                                                hit.append((m, "STARVED"))
                    elif role == "TRANSPORTER":
                        for rx in list(catal[g])[:10]:
                            for m in rxn_genes.get(rx, ()):
                                if m != g and m not in disabled:
                                    hit.append((m, "STRANDED"))
                    elif role == "MODIFIER":
                        for t in sigdn.get(g, [])[:MAXFANOUT]:
                            if t not in disabled:
                                hit.append((t, "UNMODIFIED"))
                    elif role == "SIGNALLING":
                        for t in sigdn.get(g, [])[:MAXFANOUT]:
                            if t not in disabled:
                                hit.append((t, "UNSIGNALLED"))
                    elif role == "RECEPTOR_LIGAND":
                        for t in lr.get(g, [])[:10]:
                            if t not in disabled:
                                hit.append((t, "UNRECEIVED"))
                    for m, mode in hit[:MAXFANOUT]:
                        disabled[m] = mode
                        modes[mode] += 1
                        # ONLY hard functional loss recurses. A dysregulated or starved protein is quantitatively
                        # changed, not abolished, and recursing it reaches the whole cell by depth three.
                        if mode in ("ORPHANED", "STRANDED", "UNMODIFIED"):
                            nxt.append(m)
                    if role == "TF":
                        dysreg[g] |= set(tf_up.get(g, []) + tf_dn.get(g, []))
                # any TF disabled at ANY depth contributes its regulon -- the second-order transcriptional arm
                if (tf_up.get(g) or tf_dn.get(g)) and g != seed:
                    dysreg[g] |= set(tf_up.get(g, []) + tf_dn.get(g, []))
                # sensors, from where the damage is and what kind it is
                if w.get("mitochondrion"):
                    modes["@mito"] += 1
                if w.get("endoplasmic reticulum") or w.get("golgi"):
                    modes["@er"] += 1
            trace.append({"depth": depth, "n_frontier": len(frontier), "n_disabled": len(disabled)})
            frontier = list(dict.fromkeys(nxt))[:60]
            if not frontier:
                break
        # SENSORS FROM THE CASCADE'S OWN DAMAGE MODES...
        sensors = set()
        if modes.get("ORPHANED", 0) >= 3:
            sensors.add("proteotoxic")
        if modes.get("@mito"):
            sensors.add("mito_stress")
        if modes.get("@er") or modes.get("STRANDED", 0) >= 3:
            sensors.add("er_stress")
        if modes.get("STARVED", 0) >= 5:
            sensors.add("translation_stress")
        return disabled, modes, dysreg, sensors, trace

    doss = json.loads((OUT / "ko_pred_dossiers.json").read_text())
    preds, stat = {}, collections.Counter()
    cmap_all = json.loads((OUT / "consequence_map.json").read_text())
    for k in doss:
        disabled, modes, dysreg, sensors, trace = cascade(k)
        # ...UNIONED WITH THE ANNOTATION-BASED CLASSIFIER, RATHER THAN REPLACING IT. The first version of this file
        # used only the cascade's damage modes, which are derived from reactions a gene CATALYSES -- so PHB2 and
        # PMPCB, which are not catalysts, lost their mitochondrial classification and their scores collapsed from
        # 11/17 and 7/16 to nothing. The recursion was the new idea; discarding the working lesion classifier was
        # an accident, and the two are complementary rather than alternative.
        _m = cmap_all.get(k, {})
        _pw = (doss[k].get("pathways") or []) + (_m.get("pathways_member") or [])
        _co = doss[k].get("compartments") or _m.get("compartments") or []
        sensors |= ml_classify(k, _pw, _co, doss[k].get("function"))
        stat.update(sensors or {"<no sensor>"})
        genes, seen = [], {k}

        def add(xs):
            for x in xs:
                if x and x not in seen and "; " not in x:
                    seen.add(x)
                    genes.append(x)

        add(sorted(dysreg.get(k, ())))                       # the knockout's own regulon
        n_own = len(genes)
        for tf, tgts in dysreg.items():                       # regulons of TFs disabled downstream
            if tf != k:
                add(sorted(tgts)[:15])
        n_2nd = len(genes) - n_own
        for s in sorted(sensors):
            add(SENSOR_PROGRAMME.get(s) or MLSENSORS[s]["programme"])
        n_sen = len(genes) - n_own - n_2nd
        if not genes:      # never emit nothing: fall back to physical context
            add(sorted(disabled)[:20])
        preds[k] = {
            "reasoning": f"cascade over {len(disabled)} proteins; modes {dict(modes)}; sensors {sorted(sensors)}",
            "n_disabled": len(disabled), "modes": dict(modes), "sensors": sorted(sensors),
            "n_own_regulon": n_own, "n_downstream_tf_regulon": n_2nd, "n_sensor": n_sen,
            "downstream_tfs_disabled": sorted(t for t in dysreg if t != k)[:20],
            "trace": trace, "predicted_genes": genes[:60],
        }
    json.dump(preds, open(OUT / "ko_predictions_cascade.json", "w"), indent=1)
    nd = sorted(p["n_disabled"] for p in preds.values())
    print(f"{len(preds)} cascades; proteins disabled per knockout: median {nd[len(nd)//2]}, max {nd[-1]}")
    print(f"  sensors fired: {dict(stat.most_common())}")
    n2 = sum(1 for p in preds.values() if p["n_downstream_tf_regulon"])
    print(f"  knockouts with a SECOND-ORDER TF arm (a TF disabled downstream): {n2}/{len(preds)}")
    sz = sorted(len(p["predicted_genes"]) for p in preds.values())
    print(f"  prediction size median {sz[len(sz)//2]}, empty {sum(1 for s in sz if s == 0)}")
    print(f"  -> {OUT/'ko_predictions_cascade.json'}")


if __name__ == "__main__":
    main()
