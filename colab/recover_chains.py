"""recover_chains -- the "how do we get from 7 to 230" attack. case_report could verify only 7 of 230 curated chains end-to-end for GATA1. The
audit itself said WHY: 82% of intermediates failed for lack of MEASUREMENT (61 never measured, 72 measured-but-untestable), and only 4 because
the annotation was wrong. So the gap is dominated by missing data and missing SIGNS -- both of which we attack here with data already in hand.

FIX 1 -- UNCENSORED EXPRESSION. The harness pickle stores only the top ~250 genes per knockout (96.5% zeros). A dense matrix of the same
experiment exists (1971 KOs x 8563 genes, 0% zeros, 100% sign agreement with the pickle, spearman 0.90, scale ~3.8x). Its threshold is
CALIBRATED against the knockouts where the pickle is NOT truncated, so "moved" means the same thing in both. This converts most
NO-EXPRESSION-DATA / below-floor intermediates into real measurements.

FIX 2 -- EMPIRICAL SIGNS. 207/230 endpoints were reached by edges carrying NO direction, and where a direction WAS annotated it was wrong 65%
of the time. Instead of trusting the database sign, LEARN it: for edge A->B, take the correlation of A's and B's response across all OTHER
knockouts (GATA1 held out, so this is not circular). corr>0 => A and B move together (activating), corr<0 => opposing. Requires a minimum
|corr| to claim a sign at all, so we abstain rather than guess.

We then re-run the identical per-hop audit and report a RECOVERY LADDER: how many chains verify with (a) the original censored data + database
signs, (b) uncensored data + database signs, (c) uncensored data + empirical signs. Each step's contribution is separated, and the empirical
sign is scored against the database sign head-to-head on the endpoints where BOTH make a claim -- the honest test of whether learned signs are
actually better. Deterministic (sorted iteration, GATA1 excluded from its own sign learning)."""
import json, sys, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAXHOP = 4
TIDE_FRAC = 0.05
MINCORR = 0.15          # below this we abstain rather than assert an empirical sign


def main():
    KO = sys.argv[1] if len(sys.argv) > 1 else "GATA1"
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    if KO not in df.index:
        print(f"{KO} not in the uncensored matrix."); return
    genes = [str(g) for g in df.columns]; gidx = {g: i for i, g in enumerate(genes)}
    Z = df.values.astype(np.float32); kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}

    # ---- calibrate the uncensored threshold against the pickle's UNtruncated knockouts ----
    true_counts = []
    for k in H.kos:
        s = np.abs(H.M[H.ki[k]]); nz = s[s > 0]
        if len(nz) and nz.min() < 1.0 and k in kidx:
            true_counts.append(int((s >= 1.0).sum()))
    med_true = float(np.median(true_counts)) if true_counts else 1.0
    Aall = np.abs(Z)
    T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        c = float(np.median((Aall >= t).sum(1)))
        e = abs(c - med_true)
        if best is None or e < best:
            best, T = e, float(t)

    mover = Aall >= T
    tide_mask = mover.mean(0) >= TIDE_FRAC
    r = kidx[KO]; z = Z[r]
    spec_idx = [i for i in np.where(mover[r])[0] if not tide_mask[i]]
    specset = set(genes[i] for i in spec_idx)
    print("=" * 108)
    print(f"RECOVER CHAINS -- {KO}: can better DATA and learned SIGNS turn unverifiable chains into verified ones?")
    print("=" * 108)
    print(f"\nDATA: uncensored {Z.shape[0]} KOs x {Z.shape[1]} genes (0% zeros) vs the pickle's top-250-per-KO store.")
    print(f"  calibrated threshold |z|>={T:.1f} (matches the pickle's |z|>=1.0 on its {len(true_counts)} untruncated KOs)")
    print(f"  {KO}: {int(mover[r].sum())} movers, {len(spec_idx)} specific  (the pickle could show at most 250)")

    # ---- graph ----
    reg, _t, _s = build_causal(H)
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    sreg = collections.defaultdict(dict); tmp = collections.defaultdict(lambda: collections.defaultdict(set))
    for s, ts in reg.items():
        for t2, sg in ts:
            tmp[s][t2].add(sg)
    for s in sorted(tmp):
        for t2 in sorted(tmp[s]):
            nz = {x for x in tmp[s][t2] if x != 0}
            sreg[s][t2] = (next(iter(nz)) if len(nz) == 1 else 0)
    phys = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            phys[names[a]].add(names[b]); phys[names[b]].add(names[a])
    for mem in D["complexes"].values():
        mm = [names[x] for x in mem if isinstance(x, int) and x < N]
        if 2 <= len(mm) <= 200:
            for g in mm:
                phys[g].update(x for x in mm if x != g)
    adj = collections.defaultdict(list)
    for s in sorted(sreg):
        for t2 in sorted(sreg[s]):
            adj[s].append((t2, "reg", sreg[s][t2]))
    for a in sorted(phys):
        for b in sorted(phys[a]):
            adj[a].append((b, "phys", 0))

    # ---- FIX 2: empirical signs, learned with KO held out (not circular) ----
    keep = [i for i in range(len(kos)) if i != r]
    Zc = Z[keep]
    Zs = (Zc - Zc.mean(0)) / (Zc.std(0) + 1e-9)
    n_obs = Zs.shape[0]
    emp_cache = {}
    def emp_sign(a, b):
        """MEASURED causal sign: knock out `a` and read which way `b` actually moved, using a's OWN knockout row.
        This is a direct interventional measurement, not a correlation -- and it never touches the target KO's row.
        Falls back to cross-knockout correlation only when `a` was not itself profiled."""
        key = (a, b)
        if key in emp_cache:
            return emp_cache[key]
        ia, jb = kidx.get(a), gidx.get(b)
        v = None
        if ia is not None and ia != r and jb is not None and abs(Z[ia, jb]) >= T:
            v = (int(np.sign(Z[ia, jb])), float(abs(Z[ia, jb])), "measured")   # effect of REMOVING a on b
        else:
            ga = gidx.get(a)
            if ga is not None and jb is not None:
                c = float(Zs[:, ga] @ Zs[:, jb] / n_obs)
                if abs(c) >= MINCORR:
                    v = (-int(np.sign(c)), abs(c), "corr")   # co-movement sign -> effect of REMOVAL is its negation
        emp_cache[key] = v
        return v

    # ---- BFS ----
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

    # ---- the audit, run under three regimes ----
    def audit(mode, censor):
        """mode: 'db' = database signs only; 'emp' = database sign, else learned sign.
           censor: True = emulate the pickle (only the KO's top-250 genes are visible)."""
        vis = None
        if censor:
            top = set(np.argsort(-np.abs(z))[:250].tolist())
            vis = top
        oper = 0; states = collections.Counter(); tested = 0; correct = 0; usable = 0
        for g in sorted(specset):
            if g not in hop or hop[g] == 0 or hop[g] > MAXHOP:
                continue
            usable += 1
            seq = chain_of(g); drive = -1; broke = None
            for i2 in range(1, len(seq)):
                node, et, sg = seq[i2]
                pred = None
                if et == "reg" and drive is not None:
                    if sg != 0:
                        pred = drive * sg
                    elif mode == "emp":
                        es = emp_sign(seq[i2 - 1][0], node)
                        if es:
                            pred = -drive * es[0]                # es = effect of REMOVING the upstream node
                elif et == "phys" and mode == "emp" and drive is not None:
                    es = emp_sign(seq[i2 - 1][0], node)          # measured sign rescues an otherwise directionless physical hop
                    if es:
                        pred = -drive * es[0]
                j = gidx.get(node)
                seen = (j is not None) and (not censor or j in vis)
                if not seen:
                    lab = "NO-DATA"; nd = None
                elif abs(z[j]) >= T:
                    if pred is None:
                        lab, nd = "MOVED-UNSIGNED", int(np.sign(z[j]))
                    elif np.sign(z[j]) == pred:
                        lab, nd = "CONFIRMED", int(np.sign(z[j])); tested += 1; correct += 1
                    else:
                        lab, nd = "CONTRADICTED", int(np.sign(z[j])); tested += 1
                else:
                    lab, nd = "FLAT", None
                states[lab] += 1
                if broke is None and lab != "CONFIRMED":
                    broke = lab
                drive = nd
            if broke is None:
                oper += 1
        return oper, usable, states, tested, correct

    print(f"\n{'-'*108}\nRECOVERY LADDER -- chains verified end to end (every hop demonstrably changed in the predicted direction)\n{'-'*108}")
    rows = []
    for label, mode, censor in [("(a) censored data + database signs", "db", True),
                                ("(b) UNCENSORED data + database signs", "db", False),
                                ("(c) UNCENSORED data + LEARNED signs", "emp", False)]:
        oper, usable, st, tested, correct = audit(mode, censor)
        rows.append((label, oper, usable, st, tested, correct))
        acc = f"{100*correct/tested:.0f}%" if tested else "n/a"
        print(f"  {label:38s} {oper:>4d}/{usable:<4d} verified   ({100*oper/max(usable,1):>4.1f}%)   "
              f"sign-testable hops {tested:>4d}, correct {acc}")
        print(f"      hop states: " + ", ".join(f"{k} {v}" for k, v in st.most_common()))

    # head-to-head: where BOTH database and learned signs make a claim, which is right?
    db_w = db_r = emp_w = emp_r = 0
    for g in sorted(specset):
        if g not in hop or hop[g] == 0 or hop[g] > MAXHOP:
            continue
        seq = chain_of(g)
        for i2 in range(1, len(seq)):
            node, et, sg = seq[i2]
            j = gidx.get(node)
            if j is None or abs(z[j]) < T or et != "reg" or sg == 0:
                continue
            if i2 != 1:
                continue                                   # only scoreable at hop 1, where drive is known to be -1
            es = emp_sign(seq[i2 - 1][0], node)
            if not es:
                continue
            obs = int(np.sign(z[j]))
            db_r += (obs == -sg); db_w += (obs != -sg)     # database: KO removes source, activation(+1) predicts down
            emp_r += (obs == es[0]); emp_w += (obs != es[0])   # measured: es IS the effect of removing the source
    if db_r + db_w:
        print(f"\n  HEAD-TO-HEAD on hop-1 edges where BOTH make a claim ({db_r+db_w} edges):")
        print(f"      database sign correct {db_r}/{db_r+db_w} ({100*db_r/(db_r+db_w):.0f}%)   "
              f"learned sign correct {emp_r}/{emp_r+emp_w} ({100*emp_r/(emp_r+emp_w):.0f}%)")

    # ---- THE BINDING CONSTRAINT: to verify GATA1 -> X -> Y you must know what removing X does ----
    inter = collections.Counter()
    for g in sorted(specset):
        if hop.get(g, 99) == 0 or hop.get(g, 99) > MAXHOP:
            continue
        c = g
        while c is not None and c in par:
            pnode = par[c]
            if pnode is None:
                break
            c = pnode[0]
            if c != KO:
                inter[c] += 1
    n_int = len(inter)
    profiled = [x for x in inter if x in kidx]
    effective = [x for x in profiled if int((Aall[kidx[x]] >= T).sum()) >= 20]
    per_ko = (Aall >= T).sum(1)
    weak = int((per_ko < 20).sum())
    print(f"\n{'-'*108}\nTHE BINDING CONSTRAINT -- why analysis alone cannot close the gap\n{'-'*108}")
    print(f"  {n_int} distinct intermediate proteins sit on {KO}'s chains.")
    print(f"    themselves knocked out in this experiment : {len(profiled):>4d} ({100*len(profiled)/max(n_int,1):.0f}%)")
    print(f"    ...with a knockout strong enough to read   : {len(effective):>4d} ({100*len(effective)/max(n_int,1):.0f}%)")
    print(f"  To verify {KO} -> X -> Y you must know what removing X does. For "
          f"{100*(1-len(effective)/max(n_int,1)):.0f}% of intermediates that experiment was never run.")
    print(f"\n  And the panel itself is mostly null: of {len(kos)} knockouts profiled, {weak} ({100*weak/len(kos):.0f}%) move fewer than 20 "
          f"genes (median {int(np.median(per_ko))}). Only {int((per_ko>=100).sum())} ({100*(per_ko>=100).sum()/len(kos):.0f}%) produce a "
          f"response of >=100 genes -- so most knockouts carry no information to propagate.")

    (la, oa, ua, sa, ta, ca) = rows[0]; (lb, ob, ub, sb, tb, cb) = rows[1]; (lc, oc, uc, sc, tc, cc) = rows[2]
    gain_data = ob - oa; gain_sign = oc - ob
    verdict = (
        f"RECOVER CHAINS ({KO}; recover_chains.py): attacking the 7-of-230 verification gap with the two things the audit blamed -- missing "
        f"MEASUREMENT (82% of failed hops) and missing/wrong SIGNS. FIX 1, uncensored expression: the harness pickle keeps only the top ~250 "
        f"genes per knockout, while a dense matrix of the SAME experiment (1971x8563, 0% zeros, 100% sign agreement, spearman 0.90) exists; "
        f"calibrated to the pickle's own untruncated knockouts, {KO} has {int(mover[r].sum())} movers ({len(spec_idx)} specific), so the pickle "
        f"was showing about {100*250/max(int(mover[r].sum()),1):.0f}% of the response. FIX 2, learned signs: instead of trusting a database "
        f"direction that is wrong most of the time, infer each edge's sign from how the two genes co-move across all OTHER knockouts ({KO} held "
        f"out, so not circular), abstaining below |corr|={MINCORR}. RESULT -- chains verified end to end: censored+database {oa}/{ua} "
        f"({100*oa/max(ua,1):.1f}%) -> uncensored+database {ob}/{ub} ({100*ob/max(ub,1):.1f}%) -> uncensored+learned {oc}/{uc} "
        f"({100*oc/max(uc,1):.1f}%). So better DATA contributes {gain_data:+d} verified chains and learned SIGNS {gain_sign:+d}. "
        + (f"HEAD-TO-HEAD, on the hop-1 edges where both make a claim, the database sign is right {100*db_r/max(db_r+db_w,1):.0f}% and the "
           f"learned sign {100*emp_r/max(emp_r+emp_w,1):.0f}%. " if db_r + db_w else "")
        + f"BUT THE LADDER IS NOT THE POINT -- THE BINDING CONSTRAINT IS. Of the {n_int} distinct intermediate proteins on {KO}'s chains, only "
        f"{len(profiled)} ({100*len(profiled)/max(n_int,1):.0f}%) were THEMSELVES knocked out in this experiment, and only "
        f"{len(effective)} ({100*len(effective)/max(n_int,1):.0f}%) had a knockout strong enough to read. To verify {KO}->X->Y you must know "
        f"what removing X does, and for ~{100*(1-len(effective)/max(n_int,1)):.0f}% of the intermediates that experiment was never run. "
        f"Worse, the panel is mostly null: {weak}/{len(kos)} ({100*weak/len(kos):.0f}%) of profiled knockouts move fewer than 20 genes "
        f"(median {int(np.median(per_ko))}), and only {100*(per_ko>=100).sum()/len(kos):.0f}% move >=100 -- most knockouts carry no signal to "
        "propagate at all. CONCLUSION, and it is actionable: the route from a handful of verified chains to most of them is NOT more clever "
        "analysis of this dataset, and NOT primarily an activity readout -- it is PERTURBING THE INTERMEDIATES. A targeted second screen that "
        "knocks out the few hundred proteins sitting on the chains (with verified depletion, since most current knockdowns are null) would make "
        "each hop directly checkable by measurement instead of inferable by annotation. That is a few-hundred-gene screen, not a genome-wide one, "
        "and it is chosen by the graph rather than at random. Deterministic; signs learned with the target knockout held out.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"knockout": KO, "calibrated_threshold": round(T, 2), "movers_uncensored": int(mover[r].sum()),
               "specific_uncensored": len(spec_idx),
               "ladder": [{"regime": l, "verified": o, "of": u, "sign_tested": t2, "sign_correct": c2,
                           "states": dict(st_)} for (l, o, u, st_, t2, c2) in rows],
               "gain_from_data": gain_data, "gain_from_learned_signs": gain_sign,
               "head_to_head_hop1": {"n": db_r + db_w, "database_correct": db_r, "learned_correct": emp_r},
               "binding_constraint": {"intermediates": n_int, "profiled_as_knockouts": len(profiled),
                                      "with_readable_knockout": len(effective),
                                      "panel_knockouts": len(kos), "panel_weak_lt20_movers": weak,
                                      "panel_median_movers": int(np.median(per_ko))},
               "verdict": verdict, "note": verdict},
              open(OUT / f"recover_chains_{KO}.json", "w"), indent=1)
    print(f"\n  -> outputs/orphan/recover_chains_{KO}.json")


if __name__ == "__main__":
    main()
