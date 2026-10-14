"""Loop 206a. The physics feature: thermodynamic NR3C1 occupancy computed from sequence alone.

WHAT THIS IS FOR. Loop 206 asks whether the missing regulatory gains -- 612,133 of them, 71% of the
whole parameter shortfall -- can be COMPUTED rather than measured. This module builds the computed
side of that comparison and nothing else. It runs no test and it has no verdict; it fetches, it
calculates, it caches, and loop 206 scores it.

WHY A BINDING GAIN IS THE FAIR PLACE TO ASK. Loop 205 measured that rate constants cannot be
computed: perfect knowledge of the chemistry pins kcat only to 7.8x median error, because k is
exponential in a barrier nobody can compute to better than a kcal/mol. A transcription-factor
binding gain is NOT that problem. It is an EQUILIBRIUM, and equilibrium binding energies are the
one thing sequence-based biophysics has a principled route to:

    Berg and von Hippel (1987): for a factor at equilibrium with its site, the log-odds score of a
    position weight matrix IS a binding free energy in kT units, up to an additive constant.

        E(s) = sum_i  -ln( p(b_i, i) / p_bg(b_i) )        [kT]

    Occupancy then follows from the grand-canonical form with chemical potential mu, which absorbs
    the free factor concentration:

        theta(s) = 1 / (1 + exp(E(s) - mu))

    and a promoter carrying several sites is a partition function over them:

        Z = sum_s exp(mu - E(s)),        theta_promoter = Z / (1 + Z)

    No barrier, no transition state, no simulation. This is the strongest case physics has.

WHAT IS FETCHED, AND WHY EACH.
    JASPAR MA0113.4  -- the NR3C1 position frequency matrix, 15 bp. NR3C1 is the glucocorticoid
                        receptor and dexamethasone is its ligand, so on the A549 course it is the
                        factor the perturbation acts on directly. If a computed occupancy is going
                        to work anywhere, it is here.
    UCSC hg38        -- promoter sequence, TSS +/- PROM_PAD, for the genes loop 198 scored.

MU IS NOT CHOSEN HERE. The chemical potential sets the effective factor concentration and it is the
one free parameter in the expression above. Choosing it against the target would be fitting, so
this module emits occupancy on a GRID of mu and loop 206 selects mu on training folds only. That is
declared here, in the module that produces the feature, so the choice cannot be made later and
described as physics.

THE BACKGROUND IS MEASURED, NOT ASSUMED. p_bg is taken from the base composition of the fetched
promoters themselves rather than set to 0.25 each, because a GC-rich promoter set scored against a
uniform background inflates every energy in the same direction and would show up as signal.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
A549 = SP / "grtc"
OUTDIR = ROOT / "colab" / "data" / "physics"
PWM_F = OUTDIR / "MA0113.4.json"
SEQ_F = OUTDIR / "promoter_seq_hg38.json"
OCC_F = OUTDIR / "nr3c1_occupancy.npz"

T_MIN, REPS, MIN_TPM, MIN_PLATEAU = 30.0, (1, 2, 3), 1.0, 0.5
PROM_PAD = L191.PROM_PAD
MU_GRID = np.arange(-4.0, 14.01, 0.5)
UCSC = "https://api.genome.ucsc.edu/getData/sequence"


def say(*a):
    print(" ".join(str(x) for x in a), flush=True)


def gene_set():
    """Exactly loop 198's scored set: responders with a promoter DNase peak."""
    import gzip
    z = np.load(A549 / "rna.npz", allow_pickle=True)
    tpm = z["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"].astype(int)
    allt = sorted(set(mins.tolist()))
    comp = {t: set(reps[mins == t].tolist()) for t in allt}
    grid = np.array([t for t in allt if set(REPS) <= comp[t] and t >= T_MIN], dtype=float)
    M, _ = L191.rep_trajectories(tpm, mins, reps, REPS, grid)
    e2s = L191.ensg_to_symbol(lambda *_: None)
    sym = np.array([e2s.get(g, "") for g in ensg])
    base = tpm[(mins == int(grid[0])) & np.isin(reps, REPS)].mean(0)
    pl = M[-3:].mean(0)
    resp = (base >= MIN_TPM) & (np.abs(pl) >= MIN_PLATEAU)
    tab = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    tssb = {}
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tab):
                tssb[str(tab[i]["name"]).upper()] = (q[0], int(q[2]))
    pt, PM = L191.promoter_track("DNase", [tssb.get(s) for s in sym], PROM_PAD, lambda *_: None)
    idx = [int(np.where(pt == t)[0][0]) for t in grid]
    A = PM[idx]
    keep = resp & (A > 0).any(0)
    return grid, M, A, sym, keep, tssb


def fetch_sequences(sym, keep, tssb):
    if SEQ_F.exists():
        seqs = json.load(open(SEQ_F))
        say(f"     promoter sequences cached: {len(seqs):,}")
        return seqs
    import urllib.request
    seqs = {}
    todo = [s for s in sym[keep] if tssb.get(s)]
    say(f"     fetching {len(todo):,} promoters from UCSC hg38 ({2*PROM_PAD} bp each)")
    for n, s in enumerate(todo, 1):
        ch, pos = tssb[s]
        lo, hi = max(0, pos - PROM_PAD), pos + PROM_PAD
        url = f"{UCSC}?genome=hg38;chrom={ch};start={lo};end={hi}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    seqs[s] = json.loads(r.read())["dna"].upper()
                break
            except Exception:
                time.sleep(2 ** attempt)
        if n % 200 == 0:
            say(f"       {n:,}/{len(todo):,}")
            json.dump(seqs, open(SEQ_F, "w"))
    json.dump(seqs, open(SEQ_F, "w"))
    say(f"     fetched {len(seqs):,}")
    return seqs


def energy_matrix(pfm, bg):
    """Berg-von Hippel: PWM log-odds is a binding energy in kT, up to an additive constant."""
    P = np.array([pfm[b] for b in "ACGT"], float)          # 4 x L counts
    P = (P + 0.25 * P.sum(0).mean() * 0.01) / (P + 0.25 * P.sum(0).mean() * 0.01).sum(0)
    return -np.log(P / bg[:, None])                        # 4 x L, kT units


def scan(seq, E, mu_grid):
    """Partition-function occupancy over both strands, for every mu on the grid."""
    B = {c: i for i, c in enumerate("ACGT")}
    L = E.shape[1]
    arr = np.array([B.get(c, -1) for c in seq], int)
    Erc = E[::-1, ::-1]                                    # reverse complement
    out = np.zeros(len(mu_grid))
    if len(arr) < L:
        return out
    win = np.lib.stride_tricks.sliding_window_view(arr, L)
    ok = (win >= 0).all(1)
    win = win[ok]
    if not len(win):
        return out
    cols = np.arange(L)
    e_f = E[win, cols].sum(1)
    e_r = Erc[win, cols].sum(1)
    e = np.minimum(e_f, e_r)                               # best strand per position
    for k, mu in enumerate(mu_grid):
        Z = np.exp(np.clip(mu - e, -50, 50)).sum()
        out[k] = Z / (1.0 + Z)
    return out


def main():
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    say("=" * 96)
    say("LOOP 206a -- thermodynamic NR3C1 occupancy from sequence alone")
    say("=" * 96)

    grid, M, A, sym, keep, tssb = gene_set()
    say(f"     loop 198's scored set: {int(keep.sum()):,} genes, grid {[int(x) for x in grid]}")

    seqs = fetch_sequences(sym, keep, tssb)
    pfm = json.load(open(PWM_F))["pfm"]
    L = len(pfm["A"])

    counts = np.zeros(4)
    B = {c: i for i, c in enumerate("ACGT")}
    for s in seqs.values():
        for c in s:
            if c in B:
                counts[B[c]] += 1
    bg = counts / counts.sum()
    say(f"     motif MA0113.4 (NR3C1), {L} bp")
    say(f"     MEASURED background from the fetched promoters: "
        + "  ".join(f"{b} {bg[i]:.4f}" for i, b in enumerate('ACGT')))
    say(f"     GC content {bg[1]+bg[2]:.4f}  -- a uniform 0.25 background would have biased every "
        f"energy in one direction")

    E = energy_matrix(pfm, bg)
    say(f"     energy matrix range {E.min():.2f} to {E.max():.2f} kT   "
        f"best possible site {E.min(0).sum():.2f} kT")

    names = [s for s in sym[keep] if s in seqs]
    OCC = np.zeros((len(names), len(MU_GRID)))
    for i, s in enumerate(names):
        OCC[i] = scan(seqs[s], E, MU_GRID)
        if (i + 1) % 400 == 0:
            say(f"       scanned {i+1:,}/{len(names):,}")
    say(f"     scanned {len(names):,} promoters over {len(MU_GRID)} values of mu "
        f"({MU_GRID[0]:.1f} to {MU_GRID[-1]:.1f} kT)")
    say(f"     occupancy at mu=0: median {np.median(OCC[:, MU_GRID==0]):.4f}   "
        f"at mu=8: median {np.median(OCC[:, MU_GRID==8]):.4f}")
    say("     mu is NOT selected here -- loop 206 picks it on training folds only")

    np.savez_compressed(OCC_F, occ=OCC, mu=MU_GRID, genes=np.array(names),
                        bg=bg, energy=E, motif="MA0113.4", pad=PROM_PAD)
    say(f"     wrote {OCC_F}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
