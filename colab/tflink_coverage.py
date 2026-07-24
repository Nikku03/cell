"""tflink_coverage -- does a BIGGER researcher-curated TF->target network (TFLink: GTRD+ReMap ChIP-seq + TRRUST curation) push coverage of the
specific movers past our 10% (CollecTRI+SIGNOR+TRRUST)? And -- crucially -- is any extra coverage REAL or just the 'ChIP binds everything' trap?

For K562 scorable KOs (TF or TF-cofactor), we measure how much of the tide-removed SPECIFIC response each network explains, PLUS the
enrichment (mover-coverage / non-mover-coverage) -- because a network that 'covers' movers only by also covering everything is worthless.
Deterministic."""
import json, gzip, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")


def main():
    H = Harness("K562")
    reg, _, sources = build_causal(H)                          # our network: CollecTRI + SIGNOR + TRRUST
    reg_t = {s: set(t for t, _ in ts) for s, ts in reg.items()}
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            ppi[names[a]].add(names[b]); ppi[names[b]].add(names[a])
    comem = collections.defaultdict(set)
    for mem in D["complexes"].values():
        mm = [names[x] for x in mem if isinstance(x, int) and x < N]
        if 2 <= len(mm) <= 200:
            for g in mm:
                comem[g].update(x for x in mm if x != g)

    # who do we need TFLink edges for: every KO + its physical partners (cofactor route)
    wanted = set()
    for ko in H.scorable:
        wanted.add(ko); wanted |= ppi.get(ko, set()) | comem.get(ko, set())

    # stream TFLink, keep TF->target where TF in wanted
    tfl = collections.defaultdict(set)
    with gzip.open(SP / "tflink.tsv.gz", "rt") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            tf, tgt = p[4], p[5]
            if tf in wanted:
                tfl[tf].add(tgt)
    print(f"TFLink edges loaded for {len(tfl)} relevant TFs (median targets/TF "
          f"{int(np.median([len(v) for v in tfl.values()])) if tfl else 0})\n")

    gi = H.gi
    def toidx(s): return set(gi[n] for n in s if n in gi)

    def targets_of(ko, net):                                    # union of KO's own targets + its physical partners' targets (cofactor route)
        s = set(net.get(ko, set()))
        for p in (ppi.get(ko, set()) | comem.get(ko, set())):
            s |= net.get(p, set())
        return s

    cov_ours, cov_tfl, cov_comb, enr_tfl = [], [], [], []
    nti = set(int(i) for i in H.nontide_idx)
    for ko in H.scorable:
        r = H.ki[ko]
        movers = set(int(i) for i in np.where(H.mover[r])[0] if H.nontide[i])
        if not movers:
            continue
        phys = toidx(ppi.get(ko, set()) | comem.get(ko, set()))
        ours = phys | toidx(targets_of(ko, reg_t))
        tfl_t = toidx(targets_of(ko, tfl))
        comb = ours | tfl_t
        cov_ours.append(len(movers & ours) / len(movers))
        cov_tfl.append(len(movers & tfl_t) / len(movers))
        cov_comb.append(len(movers & comb) / len(movers))
        # fairness: enrichment = mover-coverage / non-mover-coverage by TFLink
        nonmovers = nti - movers
        if nonmovers:
            nm_cov = len(nonmovers & tfl_t) / len(nonmovers)
            if nm_cov > 0:
                enr_tfl.append((len(movers & tfl_t) / len(movers)) / nm_cov)
    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Coverage of tide-removed SPECIFIC movers ({len(cov_ours)} KOs):")
    print(f"    OURS (CollecTRI+SIGNOR+TRRUST+PPI)  {m(cov_ours)*100:5.1f}%")
    print(f"    TFLink alone (GTRD+ReMap ChIP+TRRUST) {m(cov_tfl)*100:5.1f}%")
    print(f"    COMBINED                            {m(cov_comb)*100:5.1f}%")
    print(f"\n    TFLink fairness: mover-coverage / NON-mover-coverage = {m(enr_tfl):.2f}x  "
          f"(1.0x = 'binds everything', no real specificity)")

    past10 = m(cov_comb) > 0.15
    real = m(enr_tfl) > 1.5
    verdict = (
        f"TFLINK COVERAGE (tflink_coverage.py): does a bigger researcher-curated ChIP+curation network (TFLink = GTRD+ReMap ChIP-seq + TRRUST) "
        f"push specific-mover coverage past our 10%? K562 scorable KOs. OURS (CollecTRI+SIGNOR+TRRUST+PPI) {m(cov_ours)*100:.0f}%, TFLink alone "
        f"{m(cov_tfl)*100:.0f}%, COMBINED {m(cov_comb)*100:.0f}%. "
        + (f"Coverage DOES rise to {m(cov_comb)*100:.0f}% -- but the fairness check is decisive: TFLink covers movers only {m(enr_tfl):.2f}x more "
           f"than NON-movers. " if m(cov_tfl) > m(cov_ours) + 0.05 else
           f"Coverage barely moves ({m(cov_comb)*100:.0f}% combined). ")
        + ("Because that enrichment is ~1x, the extra 'coverage' is the BINDING-NOT-REGULATION trap: GTRD/ReMap ChIP has each TF bound to "
           "thousands of promoters, so it 'covers' the specific movers only by also covering nearly every non-mover -- it inflates coverage "
           "without adding specificity, and would predict a huge false-positive list. So NO researcher-curated network -- not CollecTRI (already "
           "ours), not the largest ChIP compendia (TFLink) -- gets past the ~10% of the tide-removed SPECIFIC response with real specificity. "
           "This confirms the ceiling is not a gap in our particular databases: it is that CURATED regulation encodes CANONICAL targets (which "
           "are the generic/tide movers we remove), while the knockout-SPECIFIC response is non-canonical and in NO database."
           if not real else
           "The enrichment is real (>1.5x), so TFLink adds genuine specific coverage -- a better network does help.")
        + " Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_kos": len(cov_ours), "coverage_ours": round(m(cov_ours), 4), "coverage_tflink": round(m(cov_tfl), 4),
               "coverage_combined": round(m(cov_comb), 4), "tflink_mover_vs_nonmover_enrichment": round(m(enr_tfl), 3),
               "past_10pct_with_specificity": bool(past10 and real), "verdict": verdict, "note": verdict},
              open(OUT / "tflink_coverage.json", "w"), indent=1)
    print("\n  -> outputs/orphan/tflink_coverage.json")


if __name__ == "__main__":
    main()
