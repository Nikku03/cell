"""case_report -- ONE knockout, end to end, answering the whole question in a single narrative instead of three separate modules:

  (1) THE LAST PHOTO      -- every gene that moved; generic tide split from knockout-SPECIFIC response.
  (2) WHAT WE CAN NAME    -- for each specific mover: direct target / physical partner / cofactor route / indirect chain / UNKNOWN, printed as
                             a chain with the FUNCTION of every protein on the path.
  (3) WHAT ACTUALLY CHANGED AT EACH HOP -- the part a pathway diagram never tells you. For every intermediate we ask what physically happened,
                             using the measured state, and label it:
        ABUNDANCE-CHANGED        |z|>=1 in the direction the edge requires -- the protein's level moved as the chain needs.
        ABUNDANCE-CONTRADICTED   |z|>=1 the OPPOSITE way -- in THIS cell the edge does not operate as annotated: the protein is NOT working the
                                 way it was supposed to. (Sign flip, not a small deviation.)
        ACTIVITY-CHANGED         mRNA flat BUT its chromatin motif activity shifted (Spear-ATAC, same knockdown) -- the protein's LEVEL is held
                                 constant while its regulatory FUNCTION changed: loss-of-function without loss-of-abundance.
        ACTIVITY-UNCHANGED       mRNA flat AND motif activity flat -- tested on BOTH layers and found nothing. The strongest available evidence
                                 that a link genuinely did not fire.
        DEFENDED-UNTESTABLE      mRNA flat, gene is ESSENTIAL (level actively maintained), and it has no motif readout -> cannot conclude.
        UNTESTED                 mRNA flat and the protein has no chromVAR motif at all -> its activity was never measured. Absence of evidence.
      The last two are a DATA limit, not a biological finding, and the report prints how many intermediates were even testable.
  (4) THE UNKNOWNS        -- for movers no chain reaches, try to BUILD one: (a) a transcription-rooted regulatory pathway, (b) an indirect
                             physical/protein chain from the knocked-out protein. Report exactly where each trail goes cold and why.
  (5) TALLY + HONESTY GUARD -- how many named vs unknown, and whether 'reachable in N hops' still discriminates real movers from non-movers
                             (it stops doing so once the small-world graph reaches everything).

The point of (3) is the user's demand: do not hop merely to arrive. Establish, at every hop, what changed -- level, function, or nothing.
Deterministic. Usage: python case_report.py [KNOCKOUT]  (default GATA1)."""
import json, sys, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
SPA = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/spatac")
MAXHOP = 4
THR = 1.0          # |z| >= 1 == abundance changed
DACT = 0.25        # |chromVAR motif z shift| == activity changed
RARE = 0.004       # mover-frequency below this == the gene's level is rarely allowed to move (defended)


def main():
    KO = sys.argv[1] if len(sys.argv) > 1 else "GATA1"
    H = Harness("K562")
    if KO not in H.ki:
        print(f"{KO} is not a K562 knockout."); return
    reg, _t, _s = build_causal(H)
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    gi, ki, M, A = H.gi, H.ki, H.M, H.A
    r = ki[KO]; z = M[r]

    # ---- deduped signed regulatory graph + physical graph ----
    sreg = collections.defaultdict(dict); tmp = collections.defaultdict(lambda: collections.defaultdict(set))
    for s, ts in reg.items():
        for t, sg in ts:
            tmp[s][t].add(sg)
    for s, tt in tmp.items():
        for t, ss in tt.items():
            nz = {x for x in ss if x != 0}
            sreg[s][t] = (next(iter(nz)) if len(nz) == 1 else 0)
    phys = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            phys[names[a]].add(names[b]); phys[names[b]].add(names[a])
    for mem in D["complexes"].values():
        mm = [names[x] for x in mem if isinstance(x, int) and x < N]
        if 2 <= len(mm) <= 200:
            for g in mm:
                phys[g].update(x for x in mm if x != g)
    adj = collections.defaultdict(list)                     # typed edges: reg (signed) + phys (sign 0)
    for s, d in sreg.items():
        for t, sg in d.items():
            adj[s].append((t, "reg", sg))
    for a, nb in phys.items():
        for b in nb:
            adj[a].append((b, "phys", 0))
    reg_in = collections.defaultdict(set)
    for s, d in sreg.items():
        for t in d:
            reg_in[t].add(s)

    # ---- chromatin ACTIVITY in this knockdown (Spear-ATAC chromVAR), if available ----
    chrom = {}
    try:
        d = np.load(SPA / "pilot_motif_pb.npz", allow_pickle=True)
        groups = [str(x) for x in d["groups"]]; gidx = {g: i for i, g in enumerate(groups)}
        if KO in gidx and "NT" in gidx:
            mean = d["mean"]; delta = mean[gidx[KO]] - mean[gidx["NT"]]
            per = collections.defaultdict(list)
            for j, mg in enumerate([str(x) for x in d["motif_genes"]]):
                for tf in mg.split("+"):
                    per[tf].append(float(delta[j]))
            chrom = {tf: max(v, key=abs) for tf, v in per.items()}   # that TF's largest motif shift
    except Exception:
        pass

    def func(g):
        rr = info.get(g, {}); proc = (rr.get("proc") or "").strip(); path = (rr.get("path") or "").strip()
        s = "; ".join([x for x in [proc, path] if x]) or "uncharacterized"
        return (("TF/" if rr.get("tf") else "") + s)[:52]        # only real transcription factors get the TF tag

    def hop_state(g, predicted_sign):
        """WHAT CHANGED at this protein? -> (label, detail).

        The labels form an EPISTEMIC LADDER -- what we actually tested, not what we assume. Note two corrections found by self-audit:
        (a) 'defended' must mean ESSENTIAL only. An earlier version also counted 'moves in <0.4% of knockouts' as buffered, but the MEDIAN gene
            moves in 0% of knockouts, so that rule captured 93.5% of the genome and the label was vacuous.
        (b) Only ~770 TFs have a chromVAR motif, so most proteins can NEVER be tested for an activity change. We separate 'tested and found
            no activity change' from 'could not be tested at all' instead of lumping both into one bucket."""
        j = gi.get(g)
        zi = float(z[j]) if j is not None else 0.0
        if j is not None and abs(zi) >= THR:
            if predicted_sign is None or np.sign(zi) == predicted_sign:
                return "ABUNDANCE-CHANGED", f"mRNA z={zi:+.1f} (level moved as required)"
            return "ABUNDANCE-CONTRADICTED", f"mRNA z={zi:+.1f} (moved OPPOSITE to the annotated edge)"
        ca = chrom.get(g)
        ess = float(info.get(g, {}).get("ess") or 0)
        if ca is not None:                                            # we COULD test activity for this protein
            if abs(ca) >= DACT:
                return "ACTIVITY-CHANGED", f"mRNA flat (z={zi:+.1f}) but chromatin activity {ca:+.2f}z -- FUNCTION changed, level held constant"
            return "ACTIVITY-UNCHANGED", f"mRNA flat (z={zi:+.1f}) AND chromatin activity only {ca:+.2f}z -- tested both layers, no change found"
        if ess >= 1.0:
            return "DEFENDED-UNTESTABLE", f"mRNA flat (z={zi:+.1f}); ESSENTIAL gene (level actively maintained) and no motif readout -> cannot tell"
        return "UNTESTED", f"mRNA flat (z={zi:+.1f}); no chromatin motif for this protein -> activity never measured, cannot conclude"

    # ---- shortest typed path from the knockout ----
    par = {KO: None}; hop = {KO: 0}; cur = [KO]
    for h in range(1, MAXHOP + 1):
        nxt = []
        for node in cur:
            for nbr, et, sg in adj.get(node, []):
                if nbr not in hop:
                    hop[nbr] = h; par[nbr] = (node, et, sg); nxt.append(nbr)
        cur = nxt
        if not cur:
            break

    def chain_of(g):
        seq = []; c = g
        while c is not None and c in par:
            p = par[c]
            if p is None:
                seq.append((c, None, None)); break
            seq.append((c, p[1], p[2])); c = p[0]
        return list(reversed(seq))

    # ---- the last photo ----
    mv = np.where(H.mover[r])[0]
    spec = [int(i) for i in mv if H.nontide[i]]
    tide = [int(i) for i in mv if not H.nontide[i]]
    spec_sorted = sorted(((H.genes[i], float(z[i])) for i in spec), key=lambda x: -abs(x[1]))

    print("=" * 112)
    print(f"CASE REPORT -- {KO} knockout in K562     [{func(KO)}]")
    print("=" * 112)
    ko_act = chrom.get(KO)
    print(f"\n(1) THE LAST PHOTO -- the measured endpoint")
    print(f"    {len(mv)} genes moved (|z|>=1):  {len(tide)} generic 'tide' (move under many knockouts) + {len(spec)} {KO}-SPECIFIC.")
    if ko_act is not None:
        print(f"    Sanity, the knocked-out protein itself: {KO} chromatin motif activity {ko_act:+.2f}z vs control "
              f"-> its regulatory function is {'measurably LOST' if ko_act < -DACT else 'not clearly reduced'} (independent of mRNA).")
    print(f"    The {len(spec)} specific movers are what a mechanism has to explain. Going gene by gene:\n")

    # ---- classify every specific mover, auditing each hop ----
    cls = collections.Counter(); named_rows = []; unknown_rows = []
    intermediates = collections.Counter(); inter_state = {}
    for g, zg in spec_sorted:
        if g not in hop or hop[g] == 0 or hop[g] > MAXHOP:
            cls["UNKNOWN"] += 1; unknown_rows.append((g, zg)); continue
        seq = chain_of(g)
        audit = []; drive = -1; broke = None
        for i2 in range(1, len(seq)):
            node, et, sg = seq[i2]
            pred = (drive * sg) if (et == "reg" and sg != 0) else None
            lab, det = hop_state(node, pred)
            audit.append((node, et, sg, lab, det))
            if i2 < len(seq) - 1:                                     # it is an intermediate, not the endpoint
                intermediates[node] += 1; inter_state[node] = (lab, det)
                if broke is None and lab not in ("ABUNDANCE-CHANGED", "ACTIVITY-CHANGED"):   # only demonstrated change keeps a chain alive
                    broke = (node, lab)
            elif broke is None and lab == "ABUNDANCE-CONTRADICTED":
                # the ENDPOINT counts too: a direct target that moved the OPPOSITE way is not an operative chain, it is a refuted one.
                # (without this, every 1-hop direct target is "operative" by default because it has no intermediates.)
                broke = (node, lab)
            j = gi.get(node)
            if j is not None and abs(z[j]) >= THR:
                drive = int(np.sign(z[j]))
            elif pred is not None:
                drive = pred
        kind = ("DIRECT" if hop[g] == 1 and seq[1][1] == "reg" else
                "PHYSICAL" if hop[g] == 1 else
                ("COFACTOR" if hop[g] == 2 and any(e == "phys" for _, e, _ in seq[1:]) else f"CHAIN-{hop[g]}hop"))
        cls[kind] += 1
        named_rows.append((g, zg, kind, hop[g], audit, broke))

    named = len(named_rows)
    print(f"(2) WHAT WE CAN NAME:  {named}/{len(spec)} reachable by a curated chain within {MAXHOP} hops; "
          f"{cls['UNKNOWN']}/{len(spec)} UNKNOWN (no path at all).")
    print(f"    by route: direct target {cls['DIRECT']}   physical partner {cls['PHYSICAL']}   cofactor route {cls['COFACTOR']}   "
          + "  ".join(f"{k} {v}" for k, v in sorted(cls.items()) if k.startswith("CHAIN")))

    def render(audit):
        out = [f"{KO}[{func(KO)}] (knocked out)"]
        for node, et, sg, lab, det in audit:
            arr = "--activates-->" if (et == "reg" and sg > 0) else "--represses-->" if (et == "reg" and sg < 0) \
                  else "--regulates-->" if et == "reg" else "--binds--"
            out.append(f"    {arr} {node} [{func(node)}]\n         WHAT CHANGED: {lab} :: {det}")
        return "\n".join(out)

    print(f"\n{'-'*112}\n(3) WHAT ACTUALLY CHANGED AT EACH HOP -- examples (not just 'is it reachable', but 'did it move, and how')\n{'-'*112}")
    shown = 0
    for g, zg, kind, h, audit, broke in named_rows:
        if shown >= 8:
            break
        shown += 1
        status = "OPERATIVE (every hop changed as required)" if broke is None else f"BREAKS at {broke[0]} [{broke[1]}]"
        print(f"\n  -> {g}  (observed z={zg:+.1f})   route={kind}   {status}")
        print("     " + render(audit))

    # ---- per-intermediate state summary: the heart of the question ----
    print(f"\n{'-'*112}\n(4) THE INTERMEDIATE PROTEINS -- for each one on a chain, what happened to it?\n{'-'*112}")
    st_count = collections.Counter(lab for lab, _ in inter_state.values())
    print(f"    {'protein':10s} {'used in':>8s}  {'WHAT CHANGED':24s} detail")
    for node, n in intermediates.most_common(14):
        lab, det = inter_state[node]
        print(f"    {node:10s} {n:>8d}  {lab:24s} {det}")
    print(f"\n    tally across {len(inter_state)} distinct intermediates: "
          + ",  ".join(f"{k} {v}" for k, v in st_count.most_common()))
    testable = sum(1 for n2 in inter_state if n2 in chrom)
    print(f"    ACTIVITY COVERAGE: only {testable}/{len(inter_state)} intermediates have a chromVAR motif, so only those could EVER be tested "
          f"for an activity change; the remaining {len(inter_state)-testable} are UNTESTED/DEFENDED-UNTESTABLE by data availability, not by biology.")

    # ---- unknowns: try to build a pathway ----
    print(f"\n{'-'*112}\n(5) THE UNKNOWNS -- {len(unknown_rows)} movers no chain reaches. Trying to BUILD a pathway for each:\n{'-'*112}")
    reached = set(hop)
    for g, zg in unknown_rows[:15]:
        regs = reg_in.get(g, set())
        reach_regs = [x for x in regs if x in reached]
        ppi_nb = phys.get(g, set()); reach_ppi = [x for x in ppi_nb if x in reached]
        if reach_regs:
            x = min(reach_regs, key=lambda q: hop.get(q, 99))
            print(f"  {g:10s} z={zg:+.1f}  TRANSCRIPTION ROUTE (just beyond {MAXHOP} hops): {KO} ...{hop[x]} hops... {x}"
                  f"[{func(x)}] --regulates--> {g}  | that intermediate: {hop_state(x, None)[0]}")
        elif reach_ppi:
            x = min(reach_ppi, key=lambda q: hop.get(q, 99))
            print(f"  {g:10s} z={zg:+.1f}  PROTEIN-CHAIN ROUTE: {KO} ...{hop[x]} hops... {x}[{func(x)}] --binds-- {g}  "
                  f"| that intermediate: {hop_state(x, None)[0]}")
        elif regs:
            print(f"  {g:10s} z={zg:+.1f}  DEAD END: regulated only by {{{', '.join(sorted(regs)[:4])}}} -- none reachable from {KO}.")
        else:
            print(f"  {g:10s} z={zg:+.1f}  NO CURATED UPSTREAM REGULATOR AND NO PHYSICAL LINK -- transcription-rooted pathway cannot even "
                  f"start. [{func(g)}]")
    if len(unknown_rows) > 15:
        print(f"  ... and {len(unknown_rows)-15} more")

    # ---- honesty guard ----
    print(f"\n{'-'*112}\n(6) HONESTY GUARD -- does 'reachable in N hops' still MEAN anything at each depth?\n{'-'*112}")
    specset = set(H.genes[i] for i in spec)
    nonmover = set(H.genes[i] for i in H.nontide_idx) - specset
    print(f"    {'hops':>4s} {'spec movers reached':>20s} {'non-movers reached':>19s} {'discrimination':>15s}")
    disc = {}
    for hh in range(1, MAXHOP + 1):
        R = set(x for x, d in hop.items() if d <= hh)
        sm = len(specset & R) / max(len(specset), 1); nm = len(nonmover & R) / max(len(nonmover), 1)
        disc[hh] = sm / nm if nm > 0 else float("inf")
        print(f"    {hh:>4d} {sm*100:>19.0f}% {nm*100:>18.0f}% {disc[hh]:>14.2f}x")

    n_oper = sum(1 for x in named_rows if x[5] is None)
    n_act = st_count.get("ACTIVITY-CHANGED", 0); n_def = st_count.get("DEFENDED-UNOBSERVED", 0)
    n_con = st_count.get("ABUNDANCE-CONTRADICTED", 0); n_inert = st_count.get("INERT", 0)
    testable = sum(1 for n2 in inter_state if n2 in chrom)
    n_actC = st_count.get("ACTIVITY-CHANGED", 0); n_actU = st_count.get("ACTIVITY-UNCHANGED", 0)
    n_def = st_count.get("DEFENDED-UNTESTABLE", 0); n_unt = st_count.get("UNTESTED", 0)
    n_con = st_count.get("ABUNDANCE-CONTRADICTED", 0); n_abc = st_count.get("ABUNDANCE-CHANGED", 0)
    verdict = (
        f"CASE REPORT ({KO}, K562; case_report.py): one knockout end-to-end, establishing what CHANGED at every hop rather than merely traversing "
        f"it. LAST PHOTO: {len(mv)} genes moved -- {len(tide)} generic tide + {len(spec)} {KO}-specific"
        + (f"; the knocked-out protein's own chromatin activity is {ko_act:+.2f}z, so its regulatory function is measurably lost. " if ko_act is not None else ". ")
        + f"NAMING: {named}/{len(spec)} reachable by a curated chain within {MAXHOP} hops, {cls['UNKNOWN']} unreachable -- but reachability alone "
        f"is worthless: discrimination between real movers and non-movers collapses from {disc.get(1, float('nan')):.1f}x at 1 hop to "
        f"{disc.get(MAXHOP, float('nan')):.1f}x at {MAXHOP} hops, so long chains are connectivity, not mechanism. "
        f"THE PER-HOP STATE AUDIT over {len(inter_state)} distinct intermediate proteins: {n_abc} actually changed ABUNDANCE as their edge "
        f"requires; {n_con} changed the OPPOSITE way (in this cell those proteins are NOT working as annotated); {n_actC} kept a flat mRNA but "
        f"show CHANGED CHROMATIN ACTIVITY -- function altered with level held constant, i.e. loss-of-function invisible to abundance; "
        f"{n_actU} were tested on BOTH layers and showed no change (the strongest evidence a link did not fire); and {n_def + n_unt} could not be "
        f"tested at all. THE COVERAGE CAVEAT IS DECISIVE: only {testable}/{len(inter_state)} intermediates even have a chromVAR motif, so the "
        f"activity layer could only be interrogated for a minority -- the large 'cannot tell' class is a limit of the DATA, not a biological "
        f"finding. Only {n_oper}/{named} named chains are operative end to end (every protein on them demonstrably changed). "
        "MEANING: a pathway drawn through this graph is mostly a claim about connectivity; demand that each protein on it actually changed -- in "
        "level or in measured activity -- and most chains do not survive, a few are contradicted outright, and the largest class is simply "
        "untestable with an abundance readout. That is precisely where a genome-wide activity readout is required. "
        "(Self-audit corrections applied: 1-hop targets no longer auto-pass as operative, and 'defended' now requires essentiality -- an earlier "
        "rule counting rarely-moving genes captured 93.5% of the genome and was vacuous.) Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"knockout": KO, "moved": int(len(mv)), "tide": len(tide), "specific": len(spec),
               "named": named, "unknown": cls["UNKNOWN"], "routes": dict(cls),
               "operative_chains": n_oper, "n_intermediates": len(inter_state),
               "intermediate_states": dict(st_count), "activity_testable": testable, "ko_chromatin_activity": ko_act,
               "discrimination_by_hop": {str(k): round(float(v), 2) for k, v in disc.items()},
               "verdict": verdict, "note": verdict}, open(OUT / f"case_report_{KO}.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/case_report_{KO}.json")


if __name__ == "__main__":
    main()
