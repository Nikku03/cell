"""screen_design -- item #4 made concrete. "Perturb the intermediates" is a wet-lab action, but the computational half is choosing WHICH genes,
and that is fully determined by the data we already have. This emits the ranked target list for a targeted second screen, with the reason and the
expected payoff for each gene, so the experiment is specified rather than gestured at.

SELECTION LOGIC. A chain KO -> X -> Y can only be verified if we know what removing X does. So we rank candidate X by how much verification each
one unlocks:
   chain_load     how many measured specific movers currently route through X on a shortest curated chain (across the hub knockouts we can audit)
   already_done   whether X was itself knocked out effectively in an existing panel (if so it needs no new experiment -- and we say so)
   is_regulator   whether X is a TF/regulator, since the honesty guard showed the regulatory layer keeps discrimination at depth while the
                  physical layer is at chance -- so regulatory intermediates are worth more per slot
   measurable     whether X is even detected in the expression readout (an undetected gene cannot be read out as a target either)
The output is a CSV-style table, ordered by unlocked chain-load, restricted to genes that are NOT already effectively perturbed -- i.e. exactly
the experiment that does not yet exist. Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MAXHOP, TIDE_FRAC = 4, 0.05
TOPHUBS = 25          # audit the strongest knockouts, which is where verifiable chains actually exist


def main():
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]; gidx = {g: i for i, g in enumerate(genes)}
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); Aall = np.abs(Z)
    true_counts = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
                   if k in kidx and (np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0
                                     if (np.abs(H.M[H.ki[k]]) > 0).any() else False)]
    med = float(np.median(true_counts)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((Aall >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (Aall >= T).sum(1)
    tide = ((Aall[[i for i in range(len(kos)) if per[i] >= 20]] >= T).mean(0) >= TIDE_FRAC)

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
            adj[s].append((t2, "reg"))
    for a in sorted(phys):
        for b in sorted(phys[a]):
            adj[a].append((b, "phys"))

    hubs = [kos[i] for i in np.argsort(-per)[:TOPHUBS]]
    print(f"designing the screen from the {len(hubs)} strongest knockouts (threshold |z|>={T:.1f}): {', '.join(hubs[:8])} ...")
    load = collections.Counter(); via_reg = collections.Counter(); serves = collections.defaultdict(set)
    for ko in hubs:
        par = {ko: None}; hop = {ko: 0}; cur = [ko]
        for h in range(1, MAXHOP + 1):
            nxt = []
            for node in cur:
                for nbr, et in adj.get(node, []):
                    if nbr not in hop:
                        hop[nbr] = h; par[nbr] = (node, et); nxt.append(nbr)
            cur = nxt
            if not cur:
                break
        r = kidx[ko]
        for j in np.where(Aall[r] >= T)[0]:
            if tide[j]:
                continue
            g = genes[j]
            if hop.get(g, 99) > MAXHOP or hop.get(g, 99) < 2:
                continue
            c = g
            while c is not None and c in par and par[c] is not None:
                p = par[c]
                c = p[0]
                if c != ko:
                    load[c] += 1; serves[c].add(ko)
                    if p[1] == "reg":
                        via_reg[c] += 1

    eff_kos = {k for i, k in enumerate(kos) if per[i] >= 20}
    rows = []
    for x, n in load.most_common():
        rows.append({"gene": x, "chains_unlocked": n, "hub_knockouts_served": len(serves[x]),
                     "via_regulatory": via_reg[x], "already_perturbed_effectively": x in eff_kos,
                     "measurable_as_target": x in gidx, "is_TF": bool(info.get(x, {}).get("tf")),
                     "essential": float(info.get(x, {}).get("ess") or 0)})
    todo = [r for r in rows if not r["already_perturbed_effectively"]]
    print(f"\n{len(rows)} distinct intermediates carry chain-load; {len(todo)} of them have NO effective knockout yet "
          f"({100*len(todo)/max(len(rows),1):.0f}%) -- that is the screen.")
    print(f"\n  {'rank':>4s} {'gene':10s} {'chains':>7s} {'hubs':>5s} {'via reg':>8s} {'TF':>3s} {'ess':>4s} {'measurable':>11s}")
    for i, r in enumerate(todo[:30], 1):
        print(f"  {i:>4d} {r['gene']:10s} {r['chains_unlocked']:>7d} {r['hub_knockouts_served']:>5d} {r['via_regulatory']:>8d} "
              f"{'Y' if r['is_TF'] else '-':>3s} {r['essential']:>4.0f} {'Y' if r['measurable_as_target'] else 'no':>11s}")
    cum = np.cumsum([r["chains_unlocked"] for r in todo])
    tot = cum[-1] if len(cum) else 1
    for frac in (0.5, 0.8, 0.9):
        k = int(np.searchsorted(cum, frac * tot)) + 1
        print(f"  -> the top {k} genes cover {100*frac:.0f}% of all currently-unverifiable chain load")
    n50 = int(np.searchsorted(cum, 0.5 * tot)) + 1; n90 = int(np.searchsorted(cum, 0.9 * tot)) + 1
    pd.DataFrame(todo).to_csv(OUT / "screen_design.csv", index=False)

    verdict = (
        f"SCREEN DESIGN (screen_design.py): the targeted screen specified rather than gestured at. Auditing the {TOPHUBS} strongest K562 "
        f"knockouts, {len(rows)} distinct intermediate proteins carry chain-load (a measured specific mover routes through them), and "
        f"{len(todo)} ({100*len(todo)/max(len(rows),1):.0f}%) have NO effective knockout in any existing panel -- that gap IS the experiment. "
        f"The load is concentrated: the top {n50} genes cover 50% of all currently-unverifiable chain load and the top {n90} cover 90%, so this "
        f"is a few-hundred-gene screen, not a genome-wide one, and it is chosen by the measured graph rather than by annotation. Each row carries "
        "why it was picked (chains unlocked, hub knockouts served, whether the route is regulatory -- which the honesty guard showed is the layer "
        "that retains discrimination at depth -- plus TF status, essentiality, and whether the gene is even detectable as a readout target). "
        "Full ranked list written to outputs/orphan/screen_design.csv. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_intermediates": len(rows), "n_needing_perturbation": len(todo),
               "top50pct_genes": n50, "top90pct_genes": n90,
               "top30": todo[:30], "verdict": verdict, "note": verdict},
              open(OUT / "screen_design.json", "w"), indent=1)
    print("  -> outputs/orphan/screen_design.json + screen_design.csv")


if __name__ == "__main__":
    main()
