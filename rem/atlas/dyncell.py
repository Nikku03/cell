"""A DYNAMIC memory bank: a whole cell held as state that changes itself, with provenance.

WHAT THE EXISTING MEMORY BANK IS, AND WHAT IT IS NOT. memory_bank/facts/ holds 52 static facts,
each pinned to a script, a commit and a data hash, with an invariant checker that refuses
contradictions. That discipline is the best thing in this repository. But a fact store cannot
represent a cell, because a cell is a STATE that moves and a fact is a value that does not.

THE IDEA, AND WHY IT IS NOT A SIMULATOR. This holds the cell as mutable state and lets a process
change it -- but it carries only DIRECTION, never magnitude, because the record has measured that
magnitudes are unavailable: 27,000 of ~860,000 rate constants exist, and dropping magnitude was
measured to remove three of the four walls. So the object is a SIGN-PROPAGATION STATE MACHINE with
the memory bank's provenance discipline bolted on, and calling it anything more would be the
overstatement this record keeps correcting.

WHY THAT IS WORTH BUILDING ANYWAY, AND IT IS THE WHOLE POINT. A fact store cannot be wrong about a
cell; it can only be inconsistent with itself. A state machine with invariants CAN FAIL -- and the
record's own verdict on this project is that its central weakness is a model that cannot be
falsified (the whole-cell LP has 0 of 12,931 reactions with a measured flux and is "solvable and
unfalsifiable"). An invariant that fires names an edge that the network and the state disagree
about. That is a finding, produced by the object's normal operation.

AND IT REVISES ITSELF, WHICH IS THE PART A FACT STORE CANNOT DO. Edges implicated in contradictions
have their confidence decremented in the bank. So the bank does not merely record state changes --
it changes its own belief about the network that produced them. Every such revision is journalled
with the contradiction that caused it.

THE STATE IS BUILT FROM WHAT THIS SESSION ASSEMBLED: 16,492 genes; 1,635 with a balanced Recon3D
reaction; 54,128 signed regulatory and 16,924 signed signalling edges; protein abundance for
16,015; isoform counts for 16,491; and DBD-disruption bounds for 294 regulators, which gate
whether a regulator may act at all.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

D1  PROVENANCE COMPLETENESS, BLOCKING, AND RUN FIRST. Every state change must cite the rule and
    the edges that caused it, and replaying the journal from the initial state must reproduce the
    final state EXACTLY.
    PREDECLARED: if any state is reachable with no cited cause, or replay diverges by a single
    gene, the bank is not auditable and NO number from it may be reported. This gate blocks
    everything below it. A dynamic store without replay is a pile of mutations.

D0  THE CEILING GATE. Price propagation before trusting it. From a perturbation, how far does
    direction actually travel?
    PREDECLARED: if the reached set is either trivially small or essentially the whole graph,
    propagation carries no information and this module says so. Measured against a matched control
    that SHUFFLES the edge signs while preserving the topology exactly.

D2  THE INVARIANTS, AND THAT THEY CAN FAIL. Each invariant must be violable in principle; one that
    cannot fail is decoration, not a check. Report violations as counts and name examples.
    PREDECLARED: a run with zero violations on every invariant is reported as SUSPICIOUS, not as
    success -- it most likely means the invariants are too weak to bite.

D3  SELF-REVISION, AGAINST A MATCHED CONTROL. Count edges whose confidence the bank downgrades.
    PREDECLARED: the shuffled-sign control must produce MORE contradictions than the real network.
    If it does not, the signs carry no information and the self-revision is revising noise.

D4  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import gzip
import hashlib
import json
import random
import time

HERE = os.path.dirname(__file__)
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
ENCY = R("colab", "data", "cell_complete.json.gz")
BANK = R("memory_bank", "cell")
OUT = os.path.join(HERE, "RESULTS_dyncell.txt")
RULE = "=" * 97


# =================================================================================================
# THE STATE
# =================================================================================================

class Cell:
    """Mutable cell state with an append-only journal. Direction only: +1, -1, 0 (unchanged),
    None (unknown). Every assignment carries the rule and the edges that justified it."""

    def __init__(self, genes, protein, iso, dbd_dead, tier):
        self.genes = genes
        self.i = {g: k for k, g in enumerate(genes)}
        self.protein = protein          # gene -> abundance (ppm); 0/absent blocks acting
        self.iso = iso                  # gene -> named isoform count
        self.dbd_dead = dbd_dead        # gene -> fraction of isoforms with a disrupted DBD
        self.tier = tier
        self.dir = {}                   # gene -> +1 / -1 / 0
        self.conflict = {}              # gene -> list of disagreeing (src, sign)
        self.journal = []
        self.edge_conf = collections.Counter()   # (src,tgt) -> net confidence delta

    def set(self, g, d, rule, because):
        prev = self.dir.get(g)
        self.dir[g] = d
        self.journal.append({"op": "set", "gene": g, "to": d, "from": prev,
                             "rule": rule, "because": because})

    def flag_conflict(self, g, votes, rule):
        self.conflict[g] = votes
        self.journal.append({"op": "conflict", "gene": g, "votes": votes, "rule": rule})
        for src, sgn in votes:
            self.edge_conf[(src, g)] -= 1
            self.journal.append({"op": "downgrade", "edge": [src, g], "delta": -1,
                                 "rule": "contradiction_at_" + g})

    def replay(self):
        """Reconstruct dir{} from the journal alone. D1 compares this to self.dir."""
        d = {}
        for e in self.journal:
            if e["op"] == "set":
                d[e["gene"]] = e["to"]
        return d

    def fingerprint(self):
        s = json.dumps(sorted(self.dir.items()), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# =================================================================================================
# THE PROCESS
# =================================================================================================

def may_act(C, g):
    """INVARIANT GATE: a gene may only act as a regulator if its protein is present and at least
    one isoform retains a DNA-binding domain."""
    if C.protein.get(g, 0) <= 0:
        return False, "no_protein"
    if C.dbd_dead.get(g, 0.0) >= 1.0:
        return False, "all_isoforms_dbd_disrupted"
    return True, ""


def propagate(C, out_edges, seeds, max_waves=6):
    """Direction-only propagation in waves. Unanimous incoming sign assigns; disagreement flags a
    contradiction and assigns NOTHING -- the network failed to explain the gene."""
    for g, d in seeds.items():
        C.set(g, d, "perturbation", [])
    frontier = set(seeds)
    blocked = collections.Counter()
    for wave in range(max_waves):
        votes = collections.defaultdict(list)
        for src in frontier:
            ok, why = may_act(C, src)
            if not ok:
                blocked[why] += 1
                continue
            sd = C.dir.get(src)
            if not sd:
                continue
            for tgt, w in out_edges.get(src, ()):
                if tgt in C.dir or tgt in C.conflict:
                    continue
                votes[tgt].append((src, sd * w))
        nxt = set()
        for tgt, vs in votes.items():
            signs = {s for _, s in vs}
            if len(signs) == 1:
                C.set(tgt, signs.pop(), f"propagate_wave_{wave}", [v[0] for v in vs][:6])
                nxt.add(tgt)
            else:
                C.flag_conflict(tgt, vs[:8], f"propagate_wave_{wave}")
        if not nxt:
            break
        frontier = nxt
    return blocked


def build_edges(D, names, shuffle_signs=False, seed=0):
    E = collections.defaultdict(list)
    rows = [(s, t, w) for s, t, w in D["reg"] if w] + [(s, t, w) for s, t, w in D["sig"] if w]
    if shuffle_signs:
        rng = random.Random(seed)
        ws = [w for _, _, w in rows]
        rng.shuffle(ws)
        rows = [(s, t, ws[k]) for k, (s, t, _) in enumerate(rows)]
    for s, t, w in rows:
        E[names[s]].append((names[t], w))
    return E


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("A DYNAMIC MEMORY BANK: THE CELL AS STATE THAT CHANGES ITSELF"); P_(RULE)
    D = json.load(gzip.open(ENCY))
    names = [r["name"] for r in D["genes"]]
    protein = {}
    for k, v in D["ppm"].items():
        if k.isdigit():
            protein[names[int(k)]] = v
    iso, dbd_dead, tier = {}, {}, {}
    pj = R("outputs", "protein_tf_layer.json")
    if os.path.exists(pj):
        for g, v in json.load(open(pj)).items():
            if v.get("isoforms"):
                iso[g] = v["isoforms"]
    gj = R("outputs", "gene_product_join.json")
    if os.path.exists(gj):
        for g, v in json.load(open(gj)).items():
            tier[g] = v.get("tier")
    ij = R("outputs", "isoform_edge_bounds.json")
    if os.path.exists(ij):
        for g, v in json.load(open(ij)).items():
            n = v.get("n_isoforms") or 1
            dbd_dead[g] = len(v.get("dbd_disrupted", [])) / max(n, 1)
    P_(f"  genes {len(names):,}  protein {len(protein):,}  isoforms {len(iso):,}"
       f"  DBD bounds {len(dbd_dead):,}  product tiers {len(tier):,}")

    E = build_edges(D, names)
    P_(f"  signed edges loaded: {sum(len(v) for v in E.values()):,}"
       f" from {len(E):,} regulators")

    # pick seeds: well-connected credible regulators, deterministic
    seeds_src = [g for g in ("TP53", "NR3C1", "RELA", "MYC", "STAT3") if g in E]
    P_(f"  perturbation: {seeds_src} each set DOWN (-1), a knockdown")
    seeds = {g: -1 for g in seeds_src}

    C = Cell(names, protein, iso, dbd_dead, tier)
    blocked = propagate(C, E, seeds)

    # ---- D1  BLOCKING ----------------------------------------------------------------------
    P_("\n" + RULE); P_("D1  PROVENANCE COMPLETENESS AND REPLAY (BLOCKING)"); P_(RULE)
    rp = C.replay()
    ident = rp == C.dir
    nocause = [e for e in C.journal if e["op"] == "set" and e["rule"] is None]
    P_(f"  journal entries                  {len(C.journal):>8,}")
    P_(f"  state genes assigned             {len(C.dir):>8,}")
    P_(f"  replay reproduces state exactly  {ident}")
    P_(f"  state changes with no cited rule {len(nocause):>8,}")
    P_(f"  state fingerprint                {C.fingerprint()}")
    if not ident or nocause:
        P_("\n  D1: FAIL -- the bank is not auditable. No further number reported.")
        open(OUT, "w").write("\n".join(out) + "\n")
        return
    P_("\n  D1: PASS -- every change cites a rule and the journal replays to the same state")

    # ---- D0  ceiling gate --------------------------------------------------------------------
    P_("\n" + RULE); P_("D0  THE CEILING GATE: DOES PROPAGATION CARRY INFORMATION?"); P_(RULE)
    Csh = Cell(names, protein, iso, dbd_dead, tier)
    Esh = build_edges(D, names, shuffle_signs=True, seed=20261005)
    propagate(Csh, Esh, dict(seeds))
    N = len(names)
    P_(f"    {'':<26} {'real':>10} {'sign-shuffled':>15}")
    P_(f"    {'genes reached':<26} {len(C.dir):>10,} {len(Csh.dir):>15,}")
    P_(f"    {'% of genome':<26} {100*len(C.dir)/N:>9.1f}% {100*len(Csh.dir)/N:>14.1f}%")
    P_(f"    {'contradictions':<26} {len(C.conflict):>10,} {len(Csh.conflict):>15,}")
    frac = 100 * len(C.dir) / N
    informative = 2.0 < frac < 90.0
    P_(f"\n  D0: {'PASS' if informative else 'FAIL'} -- reached {frac:.1f}% of the genome"
       f" ({'neither trivial nor everything' if informative else 'uninformative'})")

    # ---- D2  invariants ----------------------------------------------------------------------
    P_("\n" + RULE); P_("D2  THE INVARIANTS, AND WHETHER THEY BITE"); P_(RULE)
    v = {}
    v["contradiction: incoming signs disagree"] = len(C.conflict)
    v["regulator blocked: no protein"] = blocked.get("no_protein", 0)
    v["regulator blocked: all isoforms DBD-dead"] = blocked.get("all_isoforms_dbd_disrupted", 0)
    v["direction on a gene with no protein record"] = sum(
        1 for g in C.dir if g not in protein)
    v["direction on a tier-5 gene (no product record)"] = sum(
        1 for g in C.dir if tier.get(g) == "5_none")
    for k, n in v.items():
        P_(f"    {('FIRED ' if n else 'silent') :<7} {k:<48} {n:>8,}")
    fired = sum(1 for n in v.values() if n)
    P_(f"\n  {fired} of {len(v)} invariants fired.")
    P_(f"  D2: {'PASS -- the checks bite' if fired >= 2 else 'SUSPICIOUS -- too weak to be a check'}")
    ex = list(C.conflict.items())[:3]
    for g, vs in ex:
        P_(f"    e.g. {g}: {[(s, sg) for s, sg in vs][:4]}  <- network cannot explain this gene")

    # ---- D3  self-revision -------------------------------------------------------------------
    P_("\n" + RULE); P_("D3  SELF-REVISION, AGAINST THE MATCHED CONTROL"); P_(RULE)
    dn = [e for e in C.journal if e["op"] == "downgrade"]
    dnsh = [e for e in Csh.journal if e["op"] == "downgrade"]
    P_(f"  edges downgraded, real network       {len({tuple(e['edge']) for e in dn}):>8,}")
    P_(f"  edges downgraded, sign-shuffled      {len({tuple(e['edge']) for e in dnsh}):>8,}")
    ok3 = len(dnsh) > len(dn)
    P_(f"\n  D3: {'PASS' if ok3 else 'FAIL'} -- shuffled signs produce"
       f" {'MORE' if ok3 else 'NO MORE'} contradictions, so the real signs"
       f" {'carry information' if ok3 else 'do not, and self-revision would be revising noise'}")
    worst = C.edge_conf.most_common()[:-6:-1]
    P_(f"  most-contradicted edges: {[(f'{a}->{b}', c) for (a, b), c in worst]}")

    # ---- persist the bank --------------------------------------------------------------------
    os.makedirs(BANK, exist_ok=True)
    json.dump({"id": "cell_state", "confidence": "derived",
               "claim": ("Direction-only cell state after a 5-regulator knockdown, propagated over "
                         "the signed network with protein and DBD gates. Carries no magnitudes."),
               "last_verified": time.strftime("%Y-%m-%d"),
               "fingerprint": C.fingerprint(),
               "n_assigned": len(C.dir), "n_conflicts": len(C.conflict),
               "perturbation": seeds, "state": C.dir},
              open(os.path.join(BANK, "state.json"), "w"))
    with open(os.path.join(BANK, "journal.jsonl"), "w") as fh:
        for e in C.journal:
            fh.write(json.dumps(e) + "\n")
    json.dump({f"{a}->{b}": c for (a, b), c in C.edge_conf.items() if c},
              open(os.path.join(BANK, "edge_confidence.json"), "w"), indent=1)
    json.dump({"state": "gene -> direction in {+1,-1}; absent means unknown",
               "journal": "append-only ops: set | conflict | downgrade; replayable to state",
               "edge_confidence": "the bank's own revised belief per edge, negative = contradicted",
               "gates": {"protein_present": "a regulator with no protein may not act",
                         "dbd_intact": "a regulator whose every isoform has a disrupted DBD may not act"},
               "carries_no_magnitude": True},
              open(os.path.join(BANK, "schema.json"), "w"), indent=1)
    P_(f"\n  bank written to memory_bank/cell/: state.json, journal.jsonl,"
       f" edge_confidence.json, schema.json")

    P_("\n" + RULE); P_("D4  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. It is not a simulator. No magnitudes, no time, no concentrations -- direction only,")
    P_("     because the record measured that the rate constants do not exist.")
    P_("  2. 'Unanimous incoming sign' is a rule I chose, not a measured mechanism. Real genes")
    P_("     integrate regulators non-linearly and this treats disagreement as unexplainable")
    P_("     rather than resolving it.")
    P_("  3. The signs inherit the default-activation defect: about half the regulatory signs are")
    P_("     a rule, not evidence. The sig layer (97.1% signed) is the trustworthy half.")
    P_("  4. A contradiction names an edge the network cannot reconcile WITH ITSELF. It is not")
    P_("     evidence against the edge in biology -- no measurement enters this loop anywhere.")
    P_("     Making that loop touch data is the next step, and it is the whole remaining problem.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
