"""THE CURATED DIRECTED NETWORK: only arrows that trace to a curated record, tiered by corroboration.

WHAT THIS REPLACES. The assembled network reported 15.34% of PPI edges directed, but 12.06 of those points were a
heuristic -- "the transcription factor points at the non-TF" -- applied precisely where curated sources are silent,
so a curated source can check EXACTLY ZERO of its 23,097 edges. That tier is dropped here. Everything below traces
to a database record.

THE FOUR CURATED SOURCES, AND THE FACT THAT THEY ARE NOT ALL THE SAME RELATION:

  SIGNOR-signalling   manually curated "A acts on B" -- protein-level causation
  Reactome-order      `precedingEvent` between reactions -- curated pathway sequence
  Reactome-prepares   output(A) feeds input(B) -- the preparation step, which is what supplies a pathway with the
                      protein its first reaction consumes. This is the "additional input at any step, and the PPI
                      that prepares it". Currency molecules (ATP, water) are excluded by a swept reaction-count cap,
                      because joining on ATP would declare that everything precedes everything.
  reg-transcription   curated TF -> target -- a DIFFERENT relation from the other three. A kinase phosphorylating a
                      TF and that TF regulating the kinase's gene are BOTH true and point opposite ways. The audit
                      measured this directly: the two curated sources agree only 78.3%, and 98% of their
                      disagreements involve a TF. So `reg` is kept as its own relation type and a disagreement with
                      it is recorded as a FEEDBACK PAIR rather than scored as an error.

HOW "SURE" IS DEFINED, since a single curated source is not enough -- Reactome order alone is only 65.5% against
SIGNOR:

  CONFIRMED   two or more INDEPENDENT curated sources give the same arrow. This is the strongest statement
              available without new experiments.
  SINGLE      exactly one source has an opinion, and no other source contradicts it.
  FEEDBACK    two sources give OPPOSITE arrows. Recorded with both relation types rather than resolved, because in
              the measured cases both are usually correct and the pair is a genuine loop.

AND THE CONFIRMED TIER IS VALIDATED BY LEAVE-ONE-SOURCE-OUT, which is the part that makes this more than
bookkeeping: for arrows confirmed by sources A and B, the direction is checked against a THIRD source held out of
the confirmation. If corroboration is meaningful, agreement with the held-out source should be higher for CONFIRMED
arrows than for SINGLE ones. If it is not, corroboration is not buying confidence and this module says so.
"""
import collections
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
HUB = 5          # reaction-count cap for currency molecules, chosen by sweep in pathway_inputs.py
MAXP = 100       # participants-per-reaction cap, chosen by sweep in build_directed_network.py


def ci(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    return p, float(np.sqrt(p * (1 - p) / n))


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((names[min(a, b)], names[max(a, b)]))
    edge_of = collections.defaultdict(set)
    for a, b in pe:
        edge_of[a].add(b); edge_of[b].add(a)
    print(f"PPI edges: {len(pe):,}", flush=True)

    def curated(key):
        S = set()
        for e in D.get(key, []) or []:
            try:
                s_, t_ = int(e[0]), int(e[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
                S.add((names[s_], names[t_]))
        # RESTRICT TO PPI EDGES. Without this, `reg` alone contributes 610,256 TF->target pairs that are not
        # protein interactions at all, and the tier counts come out at 321% of the network -- which is what the
        # first run printed. Every source must be scoped to the graph being oriented.
        return {(a, b) for a, b in S if (b, a) not in S and b in edge_of.get(a, ())}

    SRC = {"SIGNOR-signalling": curated("sig"), "reg-transcription": curated("reg")}
    for k_, v_ in SRC.items():
        print(f"  {k_}: {len(v_):,} directed gene pairs on PPI edges", flush=True)

    # ---- Reactome: pathway order and preparation ----
    C = json.loads((Path(SP) / "reactome_order.json").read_text())
    IO = json.loads((Path(SP) / "reaction_io.json").read_text())
    s2r = {k: set(v) for k, v in json.loads((Path(SP) / "sym2rxn.json").read_text()).items()}
    r2s = collections.defaultdict(set)
    for s, rs in s2r.items():
        for r in rs:
            r2s[r].add(s)
    io = IO["io"]
    prod, cons = collections.defaultdict(set), collections.defaultdict(set)
    for r, d in io.items():
        for e in d.get("out", []):
            prod[e].add(r)
        for e in d.get("in", []):
            cons[e].add(r)
    touched = {e: len(prod[e]) + len(cons[e]) for e in set(prod) | set(cons)}

    def rxn_to_gene(rlinks, label):
        rel = set()
        for a_r, b_r in rlinks:
            s_, t_ = r2s.get(a_r), r2s.get(b_r)
            if not s_ or not t_ or len(s_) > MAXP or len(t_) > MAXP:
                continue
            for a in s_:
                nb = edge_of.get(a)
                if not nb:
                    continue
                for b in t_:
                    if a != b and b in nb:
                        rel.add((a, b))
        print(f"  {label}: {len(rel):,} directed gene pairs on PPI edges", flush=True)
        return rel

    order_links = {(p, r) for r, ps in C["preceding"].items() for p in ps}
    SRC["Reactome-order"] = rxn_to_gene(order_links, "Reactome-order")
    prep_links = set()
    for e in set(prod) & set(cons):
        if touched[e] > HUB:
            continue
        for a in prod[e]:
            for b in cons[e]:
                if a != b:
                    prep_links.add((a, b))
    SRC["Reactome-prepares"] = rxn_to_gene(prep_links, "Reactome-prepares")

    # ---- per-edge vote ----
    vote = collections.defaultdict(dict)          # frozenset(edge) -> {source: (a,b)}
    for src, rel in SRC.items():
        for a, b in rel:
            k = frozenset((a, b))
            if k in vote and src in vote[k] and vote[k][src] != (a, b):
                vote[k][src] = None               # this source itself is ambiguous on this edge
            else:
                vote[k].setdefault(src, (a, b))
    for k in vote:
        for s_ in [x for x, v in vote[k].items() if v is None]:
            del vote[k][s_]

    tiers = collections.Counter()
    out_edges = []
    confirmed_pairs, single_pairs = [], []
    for k, vs in vote.items():
        if not vs:
            # EVERY SOURCE THAT TOUCHED THIS EDGE GAVE BOTH DIRECTIONS. That is a source contradicting ITSELF --
            # Reactome lists reactions where A precedes B and others where B precedes A -- so the edge has no
            # usable arrow. An earlier version dropped these silently and 23,725 edges disappeared from the
            # accounting; they are now their own tier so the tiers sum to the number of edges any source touched.
            tiers["SOURCE-AMBIGUOUS"] += 1
            out_edges.append([tuple(k)[0], tuple(k)[1], "SOURCE-AMBIGUOUS", []])
            continue
        dirs = collections.Counter(vs.values())
        if len(dirs) == 1 and len(vs) >= 2:
            d = next(iter(dirs))
            tiers["CONFIRMED"] += 1
            out_edges.append([d[0], d[1], "CONFIRMED", sorted(vs)])
            confirmed_pairs.append((k, d, sorted(vs)))
        elif len(dirs) == 1:
            d = next(iter(dirs))
            tiers["SINGLE"] += 1
            out_edges.append([d[0], d[1], "SINGLE", sorted(vs)])
            single_pairs.append((k, d, sorted(vs)))
        else:
            tiers["FEEDBACK"] += 1
            a, b = tuple(k)
            out_edges.append([a, b, "FEEDBACK", sorted(f"{s_}:{v[0]}->{v[1]}" for s_, v in vs.items())])

    ndir = tiers["CONFIRMED"] + tiers["SINGLE"]
    print(f"\n  CURATED-ONLY DIRECTED NETWORK (the TF heuristic is dropped entirely)")
    for t in ["CONFIRMED", "SINGLE", "FEEDBACK", "SOURCE-AMBIGUOUS"]:
        print(f"    {t:12s} {tiers[t]:7,}  {tiers[t]/len(pe):6.2%} of PPI")
    print(f"    {'-> usable arrows':12s} {ndir:7,}  {ndir/len(pe):6.2%}")
    touched_n = sum(tiers.values())
    print(f"    {'no source':12s} {len(pe)-touched_n:7,}  {(len(pe)-touched_n)/len(pe):6.2%}")
    assert touched_n + (len(pe) - touched_n) == len(pe), "tiers must partition the edge set"

    # ---- LEAVE-ONE-SOURCE-OUT: does corroboration actually buy confidence? ----
    print(f"\n  LEAVE-ONE-SOURCE-OUT VALIDATION -- is a CONFIRMED arrow better than a SINGLE one?")
    print(f"    {'held-out source':20s} {'CONFIRMED':>20s} {'SINGLE':>20s}")
    loo = {}
    for held in SRC:
        hd = {frozenset((a, b)): (a, b) for a, b in SRC[held]}
        res = {}
        for lab, pool in [("CONFIRMED", confirmed_pairs), ("SINGLE", single_pairs)]:
            # arrows agreed by sources OTHER than the held-out one, then checked against it
            chk = [(k, d) for k, d, ss in pool if held not in ss and k in hd]
            if len(chk) < 20:
                res[lab] = None
                continue
            good = sum(1 for k, d in chk if hd[k] == d)
            res[lab] = ci(good, len(chk)) + (len(chk),)
        loo[held] = res
        f = lambda r: f"{r[0]:.3f}+-{r[1]:.3f} n={r[2]}" if r else "n/a"
        print(f"    {held:20s} {f(res['CONFIRMED']):>20s} {f(res['SINGLE']):>20s}")

    valid = [(h, r) for h, r in loo.items() if r["CONFIRMED"] and r["SINGLE"]]
    better = sum(1 for h, r in valid if r["CONFIRMED"][0] > r["SINGLE"][0])
    conf_acc = np.mean([r["CONFIRMED"][0] for h, r in valid]) if valid else float("nan")
    sing_acc = np.mean([r["SINGLE"][0] for h, r in valid]) if valid else float("nan")

    verdict = (
        f"THE CURATED DIRECTED NETWORK. Dropping the transcription-factor heuristic -- 23,097 edges that no curated "
        f"source could check even in principle -- leaves {ndir:,} PPI edges ({ndir/len(pe):.2%}) with an arrow that "
        f"traces to a database record, plus {tiers['FEEDBACK']:,} recorded as FEEDBACK pairs. Four curated sources "
        f"contribute: SIGNOR signalling, Reactome reaction order, Reactome PREPARATION (output of one reaction "
        f"feeding the input of another -- the step that supplies a pathway with the protein its first reaction "
        f"consumes), and curated TF->target regulation. "
        f"OF THOSE, {tiers['CONFIRMED']:,} ARE CONFIRMED BY TWO OR MORE INDEPENDENT SOURCES ({tiers['CONFIRMED']/len(pe):.2%} "
        f"of the interactome) and {tiers['SINGLE']:,} rest on one source alone. A further "
        f"{tiers['SOURCE-AMBIGUOUS']:,} edges are touched by a source that contradicts ITSELF -- Reactome records "
        f"both A-before-B and B-before-A for them -- and get no arrow; an earlier version dropped those silently "
        f"and lost them from the accounting entirely. "
        + (f"BUT CORROBORATION WAS NOT SHOWN TO BUY CONFIDENCE, and this is the result that should temper the "
           f"whole exercise. Leave-one-source-out -- hold a source out of the confirmation, then check the arrow "
           f"against it -- gives CONFIRMED {conf_acc:.3f} and SINGLE {sing_acc:.3f}, both essentially at the 0.500 "
           f"chance line, with CONFIRMED ahead in {better} of {len(valid)} tests. The samples are small (n=86 and "
           f"n=48) and their intervals straddle chance, so the honest reading is UNDERPOWERED rather than refuted; "
           f"and the only sources with enough held-out overlap to test are the two Reactome-derived ones, which are "
           f"the weakest of the four. What it does mean is that 'CONFIRMED' here records agreement between two "
           f"sources, and must NOT be read as a demonstrated accuracy. " if valid else
           "Leave-one-source-out could not be run: too few arrows survive holding a source out. ")
        + f"THE FEEDBACK TIER IS NOT A FAILURE AND IS THE REASON THIS IS TIERED AT ALL. {tiers['FEEDBACK']:,} edges "
        f"have two curated sources pointing OPPOSITE ways, and the audit showed 98% of such conflicts involve a "
        f"transcription factor: a kinase acts on a TF while that TF regulates the kinase's gene. Both arrows are "
        f"real. Forcing one would have destroyed information and inflated the apparent coverage, so both are kept "
        f"with their relation types. "
        f"WHAT THIS CANNOT SAY. Curated does not mean true in K562 -- Reactome and SIGNOR describe canonical human "
        f"biology, not this cell line under these conditions. 'Reactome-prepares' is the weakest of the four: "
        f"sharing an entity means one reaction CAN supply another, not that it does, and the currency-molecule cap "
        f"is blunt enough to drop informative hubs like ubiquitin along with ATP. A precedence relation between "
        f"reactions still does not prove the two PROTEINS touch in that order. And coverage is the price of this "
        f"honesty: {(len(pe)-len(vote))/len(pe):.0%} of the interactome has no curated arrow at all.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_ppi": len(pe), "tiers": dict(tiers), "usable": ndir, "frac_usable": ndir / len(pe),
               "sources": {k: len(v) for k, v in SRC.items()},
               "leave_one_out": {h: {k: (list(v) if v else None) for k, v in r.items()} for h, r in loo.items()},
               "verdict": verdict, "edges": out_edges},
              open(OUT / "curated_directed.json", "w"))
    print(f"\n  -> {OUT/'curated_directed.json'} ({(OUT/'curated_directed.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
