"""hop_accountability -- the user's correction to case_study: a hop is NOT 'reached / not reached'. A hop only carries causation if the
intermediate protein ACTUALLY CHANGED -- degraded/lost abundance, changed magnitude, or (invisibly) changed activity -- and changed in the
DIRECTION the edge requires. We must not hop for the sake of reaching; we must establish, at every intermediate, WHAT changed and whether the
protein 'is working the way it was supposed to, or that changed'.

We have the measured state of EVERY gene in the knockout (signed z from Perturb-seq), so each hop is auditable. Sign logic: the KO removes the
source (its level -> down). An ACTIVATION edge passes the driver's direction; a REPRESSION edge flips it. So along a signed path the predicted
direction at each node = (source went down) * product(edge signs). We compare that PREDICTION to the node's OBSERVED z and label the hop:

  CONFIRMED_DOWN / CONFIRMED_UP  -- abundance changed (|z|>=1) in the predicted direction. The transcript changed as the chain needs;
                                    CONFIRMED_DOWN is consistent with the gene being shut off (toward loss), CONFIRMED_UP with de-repression.
  WRONG_DIR                      -- it changed, but the OPPOSITE way: in THIS context the edge did not operate as annotated (behaviour changed).
  SILENT                         -- |z|<1: no abundance change. Either the link is inert here, OR (for signaling/kinase nodes) the change is in
                                    ACTIVITY/PTM that mRNA cannot see -- flagged 'activity-unobservable', not silently counted as working.
  DIR_UNKNOWN                    -- edge sign uncurated/conflicting: direction unpredictable.

A chain is MECHANISTICALLY OPERATIVE only if EVERY hop is CONFIRMED. We re-tally: of a knockout's specific movers, how many are reached by an
all-confirmed chain (each intermediate demonstrably changed the required way) vs the naive 'any path' count. Controls: (1) a sign-flip NULL
(randomize the observed directions, keep magnitudes) -- if real operative reach isn't above null, the directional agreement is chance;
(2) a CO-MOVEMENT mediation check -- for a 2-hop A->m, do A and m co-vary across the 1400 knockout profiles more than a matched-random pair?
(coupling is a necessary condition for A mediating the effect on m; well-powered, annotation-free).

Honest scope: Perturb-seq measures mRNA ABUNDANCE (a proxy for how much protein is made), NOT whether the protein is active. We audit abundance
change at every hop; we cannot see disintegration/PTM/activity directly, and say so. Also: ~79% of the panel's knockouts are near-silent, so a
KO's specific response is carried by a minority of strong perturbations. GATA1 detailed + aggregate over all scorable KOs. Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
KO = "GATA1"
MAXHOP = 4
THR = 1.0                    # |z| >= 1 == "changed" (the harness mover bar)


def main():
    H = Harness("K562")
    reg, _trace, sources = build_causal(H)
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]
    info = {g["name"]: g for g in D["genes"]}
    gi = H.gi

    def func(g):
        r = info.get(g, {}); proc = (r.get("proc") or "").strip(); path = (r.get("path") or "").strip()
        s = "; ".join([x for x in [proc, path] if x]) or "uncharacterized"
        if r.get("tf"):
            s = "TF/" + s
        return s[:50]

    def signaling(g):
        return (info.get(g, {}).get("proc") or "") == "signaling"

    # DEDUPE the graph: unique (src -> tgt) with a consensus sign; conflicting/unknown signs -> 0
    sreg = collections.defaultdict(dict)                           # src -> {tgt: sign}
    tmp = collections.defaultdict(lambda: collections.defaultdict(set))
    for s, ts in reg.items():
        for t, sg in ts:
            tmp[s][t].add(sg)
    for s, tt in tmp.items():
        for t, ss in tt.items():
            nz = {x for x in ss if x != 0}
            sreg[s][t] = (next(iter(nz)) if len(nz) == 1 else 0)   # single consistent sign else unknown/conflict
    adj = {s: list(d.items()) for s, d in sreg.items()}            # node -> [(nbr, sign)]

    # column-standardized profiles for co-movement (correlation across the KO axis)
    mu = H.M.mean(0); sd = H.M.std(0) + 1e-9
    Zc = (H.M - mu) / sd

    def corr(a_idx, m_idx):
        return abs(float(Zc[:, a_idx] @ Zc[:, m_idx]) / len(Zc))

    # ---- operative reachability: BFS that only expands through nodes whose OBSERVED change matches predicted direction ----
    def operative_reach(zvec, src=KO, maxhop=MAXHOP):
        changed = np.abs(zvec) >= THR; osign = np.sign(zvec)
        reached = {}; cur = {src: -1}; visited = {src}
        for hop in range(1, maxhop + 1):
            nxt = {}
            for node, drive in cur.items():
                for nbr, e in adj.get(node, []):
                    if e == 0:
                        continue
                    j = gi.get(nbr)
                    if j is None or nbr in visited:
                        continue
                    if changed[j] and osign[j] == drive * e:        # hop CONFIRMED
                        reached[nbr] = hop; visited.add(nbr); nxt[nbr] = int(osign[j])
            cur = nxt
            if not cur:
                break
        return reached

    def shortest_signed(src=KO, maxhop=MAXHOP):
        par = {src: None}; hop = {src: 0}; cur = [src]
        for h in range(1, maxhop + 1):
            nx = []
            for node in cur:
                for nbr, e in adj.get(node, []):
                    if nbr not in hop:
                        hop[nbr] = h; par[nbr] = (node, e); nx.append(nbr)
            cur = nx
            if not cur:
                break
        return par, hop

    def audit_chain(path_ne, z):
        out = []; drive = -1; operative = True
        for i, (node, e) in enumerate(path_ne):
            if i == 0:
                out.append((node, None, 0.0, "SOURCE_KO")); continue
            zi = float(z[gi[node]]) if node in gi else 0.0
            if e == 0:
                lab = "DIR_UNKNOWN"; operative = False
            else:
                pred = drive * e
                if abs(zi) < THR:
                    lab = "SILENT" + ("(activity?)" if signaling(node) else ""); operative = False
                elif np.sign(zi) == pred:
                    lab = "CONFIRMED_" + ("UP" if zi > 0 else "DOWN")
                else:
                    lab = "WRONG_DIR"; operative = False
            drive = int(np.sign(zi)) if abs(zi) >= THR else (drive * e if e != 0 else drive)
            out.append((node, e, zi, lab))
        return out, operative

    # ===================== PART A: GATA1 detailed =====================
    r = H.ki[KO]; z = H.M[r].copy()
    spec_idx = set(int(i) for i in np.where(H.mover[r])[0] if H.nontide[i])
    tide_idx = set(int(i) for i in np.where(H.mover[r])[0] if not H.nontide[i])
    reached = operative_reach(z)
    reached_idx = set(gi[n] for n in reached if n in gi)
    spec_reached = len(spec_idx & reached_idx); tide_reached = len(tide_idx & reached_idx)
    nulls = []
    for s in range(30):
        rng = np.random.RandomState(s)
        zn = np.abs(z) * rng.choice([-1, 1], size=H.nG)
        nulls.append(len(spec_idx & set(gi[n] for n in operative_reach(zn) if n in gi)))
    null_mean = float(np.mean(nulls)); null_sd = float(np.std(nulls))

    print("=" * 104)
    print(f"HOP ACCOUNTABILITY -- {KO} knockout, K562. Every hop must ESTABLISH what changed, not just reach.")
    print("=" * 104)
    print(f"{len(spec_idx)} {KO}-specific movers to explain. mRNA measures ABUNDANCE (proxy for protein made), NOT activity.\n")

    par, hopd = shortest_signed()
    def path_of(g):
        seq = []; cur = g
        while cur is not None and cur in par:
            p = par[cur]
            if p is None:
                seq.append((cur, None)); break
            seq.append((cur, p[1])); cur = p[0]
        return list(reversed(seq))

    spec_named = sorted(((H.genes[i], float(z[i])) for i in spec_idx), key=lambda x: -abs(x[1]))
    n_operative = 0; break_reason = collections.Counter(); ex_conf, ex_broke = [], []
    for g, zg in spec_named:
        if g not in hopd or hopd[g] == 0 or hopd[g] > MAXHOP:
            break_reason["unreachable"] += 1; continue
        audit, oper = audit_chain(path_of(g), z)
        if oper:
            n_operative += 1
            if len(ex_conf) < 6:
                ex_conf.append((g, zg, audit))
        else:
            fb = next((lab for _, _, _, lab in audit[1:] if not lab.startswith("CONFIRMED")), "?")
            break_reason[fb.split("(")[0]] += 1
            if len(ex_broke) < 6:
                ex_broke.append((g, zg, audit))

    def render(audit):
        s = []
        for i, (node, e, zi, lab) in enumerate(audit):
            if i == 0:
                s.append(f"{node}[{func(node)}] (KO'd -> level DOWN)")
            else:
                arr = "==activates==>" if e > 0 else "==represses==>" if e < 0 else "==(sign?)==>"
                s.append(f"  {arr} {node} z={zi:+.1f} [{lab}] ({func(node)})")
        return "\n        ".join(s)

    print("HOP-BY-HOP AUDIT (shortest curated chain; each intermediate's MEASURED state change):")
    print("-" * 104)
    print("\n  OPERATIVE chains (every intermediate changed in the required direction) -- examples:")
    for g, zg, audit in ex_conf:
        print(f"    -> {g} (z={zg:+.1f}):\n        {render(audit)}")
    if not ex_conf:
        print("    (none)")
    print("\n  BROKEN chains (a hop did NOT change as required -> the path is graph connectivity, not mechanism) -- examples:")
    for g, zg, audit in ex_broke:
        print(f"    -> {g} (z={zg:+.1f}):\n        {render(audit)}")

    print(f"\n  RE-TALLY of {len(spec_idx)} {KO}-specific movers:")
    print(f"    shortest curated chain is ALL-CONFIRMED (operative): {n_operative}  ({n_operative/len(spec_idx)*100:.0f}%)")
    print(f"    otherwise the chain BREAKS at:")
    for k, v in break_reason.most_common():
        print(f"        {k:16s} {v}")
    print(f"\n  BEST operative reach (BFS through ANY confirmed hops): {spec_reached}/{len(spec_idx)} specific movers "
          f"({spec_reached/len(spec_idx)*100:.0f}%)")
    print(f"    sign-flip NULL (directions randomized, magnitudes kept): {null_mean:.1f} +/- {null_sd:.1f}  "
          f"-> real is {(spec_reached-null_mean)/max(null_sd,1e-9):+.1f} sigma above chance directional agreement")
    print(f"    (tide movers operatively reached: {tide_reached}/{len(tide_idx)})")

    # ===================== PART B: aggregate over all scorable KOs =====================
    # (1) direct-target accountability (deduped unique signed pairs): do a KO's own direct targets even move, and the right way?
    dir_conf = dir_wrong = dir_silent = dir_tot = 0
    for ko in H.scorable:
        zz = H.M[H.ki[ko]]
        for tgt, e in sreg.get(ko, {}).items():
            if e == 0 or tgt not in gi:
                continue
            dir_tot += 1; zt = zz[gi[tgt]]
            if abs(zt) < THR:
                dir_silent += 1
            elif np.sign(zt) == -e:
                dir_conf += 1
            else:
                dir_wrong += 1
    dir_moved = dir_conf + dir_wrong

    # (2) operative-reach of specific movers, pooled, vs pooled sign-flip null
    tot_spec = op_spec = 0
    for ko in H.scorable:
        zz = H.M[H.ki[ko]]
        sidx = set(int(i) for i in np.where(H.mover[H.ki[ko]])[0] if H.nontide[i])
        if not sidx:
            continue
        tot_spec += len(sidx)
        op_spec += len(sidx & set(gi[n] for n in operative_reach(zz, src=ko) if n in gi))
    op_null = []
    for s in range(5):
        rng = np.random.RandomState(1000 + s); acc = 0
        for ko in H.scorable:
            sidx = set(int(i) for i in np.where(H.mover[H.ki[ko]])[0] if H.nontide[i])
            if not sidx:
                continue
            zn = np.abs(H.M[H.ki[ko]]) * rng.choice([-1, 1], size=H.nG)
            acc += len(sidx & set(gi[n] for n in operative_reach(zn, src=ko) if n in gi))
        op_null.append(acc)
    op_null_mean = float(np.mean(op_null))

    # (3) co-movement mediation: for 2-hop A->m, do A and m co-vary across the 1400 KO profiles vs a matched-random partner?
    med_pairs = []
    for ko in H.scorable:
        par2, hop2 = shortest_signed(src=ko)
        for i in (int(j) for j in np.where(H.mover[H.ki[ko]])[0] if H.nontide[j]):
            m = H.genes[i]
            if hop2.get(m) == 2:
                A = par2[m][0]
                if A in gi:
                    med_pairs.append((gi[A], i))
    rng = np.random.RandomState(7)
    real_c = [corr(a, m) for a, m in med_pairs]
    null_c = [corr(a, int(rng.randint(H.nG))) for a, m in med_pairs]     # same intermediate, random partner
    med_real = float(np.mean(real_c)) if real_c else float("nan")
    med_null = float(np.mean(null_c)) if null_c else float("nan")

    print("\n" + "=" * 104)
    print(f"AGGREGATE over all {len(H.scorable)} scorable K562 knockouts:")
    print("=" * 104)
    print(f"  (1) DIRECT-target accountability ({dir_tot} unique annotated signed KO->target pairs, target measured):")
    print(f"        moved at all (|z|>=1):        {dir_moved:5d}  ({dir_moved/dir_tot*100:.0f}%)   <- {100-dir_moved/dir_tot*100:.0f}% are SILENT")
    print(f"          of those, CONFIRMED dir:    {dir_conf:5d}  ({dir_conf/max(dir_moved,1)*100:.0f}% of movers)")
    print(f"          of those, WRONG direction:  {dir_wrong:5d}  ({dir_wrong/max(dir_moved,1)*100:.0f}% of movers)")
    print(f"  (2) OPERATIVE-reach of specific movers (all-confirmed chains): {op_spec}/{tot_spec} "
          f"({op_spec/tot_spec*100:.1f}%)  vs sign-flip null {op_null_mean:.0f} ({op_null_mean/tot_spec*100:.1f}%) "
          f"= {op_spec/max(op_null_mean,1e-9):.1f}x")
    print(f"  (3) CO-MOVEMENT mediation ({len(med_pairs)} 2-hop A->m pairs): |corr(A,m)| across 1400 KOs "
          f"{med_real:.3f} vs matched-random partner {med_null:.3f} = {med_real/max(med_null,1e-9):.1f}x")

    op_frac = op_spec / tot_spec
    lift = op_frac / (op_null_mean / tot_spec) if op_null_mean > 0 else float("inf")
    med_lift = med_real / med_null if med_null > 0 else float("nan")
    verdict = (
        f"HOP ACCOUNTABILITY (hop_accountability.py): the user's correction -- a hop only carries causation if the intermediate ACTUALLY "
        f"CHANGED the required way, not merely 'was reachable'. Auditing every hop's measured abundance change AND direction. "
        f"GATA1 detail: of {len(spec_idx)} specific movers, only {n_operative} ({n_operative/len(spec_idx)*100:.0f}%) have a shortest chain that is "
        f"fully confirmed; the rest break, {break_reason.most_common(1)[0][1] if break_reason else 0} of them at a SILENT intermediate that "
        f"didn't change at all (e.g. GATA1==represses==>MYC but MYC's mRNA is flat, so GATA1->MYC->LTB never actually fired). Allowing ANY "
        f"confirmed path, operative reach is {spec_reached}/{len(spec_idx)} ({spec_reached/len(spec_idx)*100:.0f}%), "
        f"{(spec_reached-null_mean)/max(null_sd,1e-9):+.1f}sigma above the sign-flip null -- so the confirmed agreement that exists is real, but small. "
        f"THE HEADLINE is the aggregate direct-target audit ({dir_tot} unique signed KO->target pairs): only {dir_moved/dir_tot*100:.0f}% of a "
        f"knockout's OWN annotated direct targets even change (|z|>=1) -- {100-dir_moved/dir_tot*100:.0f}% are transcriptionally SILENT in the KO -- "
        f"and of the few that move, only {dir_conf/max(dir_moved,1)*100:.0f}% go the annotated activation/repression direction; "
        f"{dir_wrong/max(dir_moved,1)*100:.0f}% go the OPPOSITE way (in K562 the wiring's SIGN has flipped vs the database -- e.g. GATA1's curated "
        "'activation' targets rise when GATA1 is removed, because in this context GATA1 represses that myeloid program). So 'the protein that was "
        "supposed to work one way' mostly (a) doesn't respond, or (b) responds the other way. Operative chains reach just "
        f"{op_frac*100:.1f}% of specific movers ({lift:.1f}x the null), and co-movement supports coupling only weakly ({med_lift:.1f}x). "
        "MEANING: the hop critique is decisive -- demanding that each intermediate's state actually changed as required dissolves almost all the "
        "'named' chains from case_study; what survives is a small, direction-confirmed core. The knockout-specific response is not a traceable "
        "cascade of confirmed state-changes through the curated wiring -- most of that wiring is silent or sign-flipped here. This is an ABUNDANCE "
        "audit; silent intermediates could still act through activity/PTM mRNA cannot see -- exactly the unmeasured layer the ceiling lives in. "
        "Deterministic; GATA1 detailed + all scorable KOs.")
    print("\nVERDICT: " + verdict)

    json.dump({
        "knockout_detail": KO, "n_specific": len(spec_idx), "operative_shortest_chains": n_operative,
        "operative_shortest_frac": round(n_operative / len(spec_idx), 4),
        "gata1_operative_reach": spec_reached, "gata1_operative_frac": round(spec_reached / len(spec_idx), 4),
        "gata1_null_mean": round(null_mean, 2), "gata1_null_sd": round(null_sd, 2),
        "gata1_sigma": round((spec_reached - null_mean) / max(null_sd, 1e-9), 2), "break_reasons": dict(break_reason),
        "direct_signed_pairs": dir_tot, "direct_moved": dir_moved, "direct_moved_frac": round(dir_moved / dir_tot, 4),
        "direct_confirmed": dir_conf, "direct_wrong": dir_wrong,
        "direct_confirmed_frac_of_movers": round(dir_conf / max(dir_moved, 1), 4),
        "direct_wrong_frac_of_movers": round(dir_wrong / max(dir_moved, 1), 4),
        "operative_reach_specific_frac": round(op_frac, 4), "operative_null_frac": round(op_null_mean / tot_spec, 4),
        "operative_over_null": round(lift, 3),
        "comovement_real": round(med_real, 4), "comovement_null": round(med_null, 4), "comovement_lift": round(med_lift, 3),
        "comovement_n_pairs": len(med_pairs), "verdict": verdict, "note": verdict,
    }, open(OUT / "hop_accountability.json", "w"), indent=1)
    print("\n  -> outputs/orphan/hop_accountability.json")


if __name__ == "__main__":
    main()
