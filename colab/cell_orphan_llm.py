"""CAN REASONING FIND THE MISSING PIECE WHERE COUNTING CANNOT?  A closed-book language-model arm on the
identical held-out reactions the tie-breaker arms were measured on.

WHERE THIS SITS.  `cell_orphan_ties_nn` closed out the model-capacity question: on 458 held-out orphan
reactions, ordering the top tie group gets recall@1 to

    random 0.314   freq 0.367   logistic 0.373   set-transformer 0.365   ORACLE 0.498

Six hand-built signals and an 8,929-parameter set transformer all land on the same 0.37, and 69% of the
tie-break headroom is unclaimed. Every one of those arms sees the candidates only as GRAPH STATISTICS -- how
often this gene catalyses anything, how many PPI partners it has, how many pathways it is in. Not one of them
knows what the reaction DOES. `dCDP + dGDP -> dCMP + dGTP` is a phosphate transfer between nucleotides, and
nothing in a frequency count can represent that.

So this arm asks the one question the feature-based arms structurally cannot: does knowing the CHEMISTRY pick
the right enzyme out of the tie group?

THE HONEST CAVEAT, stated before any number because it decides how the result may be read.  These are curated
Reactome and Human-GEM reactions, published long before the model's training cutoff. A language model has very
likely READ that `CMPK1` phosphorylates dCMP. So a high score here is NOT evidence of discovery -- it is
partly, perhaps mostly, recall of the literature the answer key was built from. That makes this arm useless as
a discovery claim and genuinely useful as two other things:

    1. a CEILING PROBE. If domain knowledge cannot claim the headroom either, the headroom is probably not
       claimable from this data at all, and the tie group is simply underdetermined.
    2. a measurement of how much of the missing-piece problem is ALREADY WRITTEN DOWN somewhere, which is
       exactly what you want to know before spending effort predicting it.

THE CONTAMINATION DIAGNOSTIC, so the caveat is measured rather than merely admitted.  Results are stratified
by how often the true catalyst appears elsewhere in the network. If the reasoning arm's advantage is
concentrated entirely in FAMOUS enzymes -- the ones a textbook names -- that is recall. If it holds for rare
one-reaction catalysts too, something more than memorisation is operating. This is reported unconditionally,
win or lose.

ARMS
    tie      choose one gene from the top tie group, presented ALPHABETICALLY. This is the exact task
             freq / logistic / set_transformer perform, on the exact same reactions, with the same 0.498
             ceiling. The only fair head-to-head.
    open     no candidate list at all -- name any human gene. Ceiling 1.000. This is not a tie-break, it is
             the whole missing-piece problem end to end, and it is the arm the caveat above bites hardest.

BASELINES, all recomputed on this exact sample rather than carried over, so every number shares a denominator:
    random | freq | logistic | set_transformer (retrained, 5 seeds) | oracle

NO-CHEATING PROTOCOL, enforced structurally rather than promised:
    1. `dump` writes the puzzles with the catalyst REMOVED, the reaction id REMOVED and participant database
       ids REMOVED, so a puzzle cannot be looked up by identifier.
    2. No answer key is ever written to disk. `score` recomputes the truth from the same deterministic
       `build()` the tie-breaker arms use, so there is no file for a solver to find.
    3. Predictions are committed to git BEFORE `score` is run. The commit order is the audit trail.

PREDECLARED, before any number:
    tie ~ freq (0.367)          -> chemistry knowledge does not order the tie group either. Combined with six
                                   hand-built signals and a transformer landing on the same value, the tie
                                   group is underdetermined and the 0.498 ceiling is not reachable from this
                                   data. This is the single most informative outcome.
    tie materially > freq       -> the headroom IS claimable, and what the feature arms lacked was reaction
                                   semantics rather than capacity. Read against the frequency stratification
                                   before calling it anything more than retrieval of known biochemistry.
    tie < freq                  -> reasoning is actively worse than counting, and popularity is a stronger
                                   prior than mechanism for this problem.
    open >> tie                 -> the bottleneck is retrieval, not ordering: the right enzyme is often not in
                                   the candidate list at all, and the chemistry-neighbour step is what to fix.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_ties_nn as T

OUT = A.OUT
PUZZLES = OUT / "llm_puzzles.json"
ANSWERS = OUT / "llm_answers.json"
N_SAMPLE = 200
SAMPLE_SEED = 7
KS = (1,)


def _sample(B):
    """The evaluated reactions. Deterministic, so `dump` and `score` cannot drift apart."""
    ev = list(B["ev_i"])
    r = np.random.default_rng(SAMPLE_SEED)
    take = r.choice(len(ev), min(N_SAMPLE, len(ev)), replace=False)
    return [int(ev[int(t)]) for t in sorted(take)]


def _clean(p):
    """A participant as the solver sees it: NAME ONLY. Database ids are stripped so the reaction cannot be
    recovered by grepping an identifier -- the whole point of the protocol.

    The subunit gene list is stripped TOO, and that is not fastidiousness. Measured on the first dump: in
    17 of 120 puzzles the true catalyst was itself listed as a subunit of one of the reaction's own
    participants -- a SUMO ligase inside the conjugation complex it acts on, a AAA-ATPase inside the
    machinery it disassembles. Handing that over would have inflated the reasoning arm by up to 0.14 for
    reading a field off, which is precisely the cheating this protocol exists to prevent. The participant
    NAME survives and carries the biology a curator would see anyway ("p21 RAS:GDP" says RAS out loud).
    """
    d = {"name": p.get("name") or "?", "type": p.get("type") or "?"}
    c = p.get("compartment")
    if c:
        d["compartment"] = c
    return d


def dump():
    B = T.build()  # noqa: F841  (used by the leak assertions below)
    ex, steps = B["examples"], B["steps"]
    sel = _sample(B)
    out = []
    for pid, j in enumerate(sel):
        e = ex[j]
        s = steps[e["rx"]]
        cand = sorted(e["tied"])
        out.append({
            "pid": pid,
            "source": s.get("source"),
            "category": s.get("category"),
            "reversible": bool(s.get("reversible")),
            "inputs": [_clean(p) for p in (s.get("in") or [])],
            "outputs": [_clean(p) for p in (s.get("out") or [])],
            "n_candidates": len(cand),
            "candidates": cand,
        })
    # Nothing I CONSTRUCT may carry the answer -- asserted structurally, on the KEYS I emit. Checking for the
    # substring "subunits" instead was my first attempt and it false-positived immediately: a participant is
    # named "...subunits...". Never search prose for your own field names.
    OKTOP = {"pid", "source", "category", "reversible", "inputs", "outputs", "n_candidates", "candidates"}
    OKPART = {"name", "type", "compartment"}
    for pid, j in enumerate(sel):
        assert set(out[pid]) <= OKTOP, f"unexpected puzzle field: {set(out[pid]) - OKTOP}"
        for w in ("inputs", "outputs"):
            for q in out[pid][w]:
                assert set(q) <= OKPART, f"unexpected participant field: {set(q) - OKPART}"
        assert steps[ex[j]["rx"]].get("id", "\0") not in json.dumps(out[pid]), "reaction id leaked"

    # What I cannot strip: a participant NAME that contains the catalyst's own symbol. That is not a
    # construction error, it is the biology -- a SUMO E2 is covalently loaded and appears as a participant of
    # the reaction it catalyses. Removing those puzzles would bias the sample toward reactions whose enzyme is
    # absent, so they STAY, get counted here, and are scored as a separate stratum. Score() recomputes this
    # mask independently; it is deliberately NOT written into the puzzle file, which would itself be a hint.
    gim = sum(1 for pid, j in enumerate(sel)
              if any(g in " ".join(q["name"] for w in ("inputs", "outputs") for q in out[pid][w])
                     for g in B["cats"][ex[j]["rx"]]))
    json.dump(out, open(PUZZLES, "w"), indent=1)
    sz = np.array([p["n_candidates"] for p in out])
    print(f"wrote {len(out)} puzzles -> {PUZZLES}")
    print(f"  candidates per puzzle: median {np.median(sz):.0f} mean {sz.mean():.1f} "
          f"p90 {np.percentile(sz,90):.0f} max {sz.max()}")
    print(f"  singleton candidate lists: {int((sz==1).sum())}/{len(out)} -- no tie-break can discriminate")
    print(f"  puzzles whose participant NAMES contain the true catalyst: {gim}/{len(out)} "
          f"({gim/len(out):.0%}) -- kept, and scored as a separate stratum")
    print(f"  NO catalyst, NO subunit lists, NO reaction id, NO database ids. No answer key written anywhere.")
    return 0


def _norm(g):
    return str(g or "").strip().upper().replace(" ", "")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        return dump()

    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import binomtest
    torch.set_num_threads(4)

    report("=" * 100)
    report("CAN REASONING FIND THE MISSING PIECE WHERE COUNTING CANNOT?")
    report("=" * 100)
    report("  Six hand-built signals and an 8,929-parameter set transformer all land on recall@1 ~0.37 against")
    report("  a 0.498 ceiling. Every one of them sees candidates only as graph statistics and none of them")
    report("  knows what the reaction DOES. This arm is given the chemistry and nothing else.")
    report("  CAVEAT DECLARED FIRST: these are curated reactions published before the training cutoff, so a")
    report("  high score is partly recall of the literature the answer key was built from, NOT discovery.")
    report("  It is still a valid CEILING PROBE, and results are stratified by catalyst fame to measure how")
    report("  much of any advantage is memorisation.")
    report("  PREDECLARED: tie ~ 0.367 => the tie group is underdetermined and the ceiling is unreachable.")
    report("  tie >> 0.367 => the feature arms lacked reaction semantics, not capacity. open >> tie =>")
    report("  the bottleneck is retrieval, not ordering.")

    B = T.build()
    ex, cats, freq = B["examples"], B["cats"], B["freq"]
    sel = _sample(B)
    pz = {p["pid"]: p for p in json.load(open(PUZZLES))}
    ans = json.load(open(ANSWERS))
    ans = {int(k): v for k, v in (ans.items() if isinstance(ans, dict) else
                                  ((a["pid"], a) for a in ans))}

    # the solver must not have been able to answer a puzzle it was never shown
    assert set(pz) == set(range(len(sel))), "puzzle file does not match the deterministic sample"

    n = len(sel)
    truth = [set(cats[ex[j]["rx"]]) for j in sel]
    tied = [list(ex[j]["tied"]) for j in sel]   # Xz rows align to THIS order -- never re-sort it
    report(f"\n  {n} held-out reactions | candidates median {np.median([len(t) for t in tied]):.0f} "
           f"| truth in candidate list {np.mean([bool(t & set(c)) for t, c in zip(truth, tied)]):.3f} "
           f"<- ceiling for every list-constrained arm")

    # ---------------- baselines, retrained and rescored on THIS sample so the denominator is shared ----------
    tr_i = B["tr_i"]
    Xtr = np.vstack([ex[k]["Xz"] for k in tr_i if ex[k]["has_truth"]])
    ytr = np.concatenate([ex[k]["y"] for k in tr_i if ex[k]["has_truth"]])
    XMU, XSD = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(max_iter=3000).fit((Xtr - XMU) / XSD, ytr)

    nets = []
    for s_ in T.SEEDS:
        m, _, _ = _train(torch, B, s_)
        nets.append(m)

    rs = np.random.default_rng(0)
    hit = {}

    def pick_scored(j, key):
        t = list(ex[j]["tied"])
        v = key(ex[j], t)
        return max(range(len(t)), key=lambda q: (v[q], t[q]))

    hit["random"] = [bool(truth[q] & {tied[q][int(rs.integers(len(tied[q])))]}) for q in range(n)]
    hit["freq"] = [bool(truth[q] & {tied[q][pick_scored(j, lambda e, t: [freq.get(g, 0) for g in t])]})
                   for q, j in enumerate(sel)]
    hit["logistic"] = [bool(truth[q] & {tied[q][pick_scored(
        j, lambda e, t: list(lr.predict_proba((e["Xz"] - XMU) / XSD)[:, 1]))]}) for q, j in enumerate(sel)]
    st_seed = []
    for m in nets:
        h = [bool(truth[q] & {tied[q][pick_scored(j, lambda e, t: list(_fwd(torch, m, e)))]})
             for q, j in enumerate(sel)]
        st_seed.append(float(np.mean(h)))
        hit["set_transformer"] = h                       # last seed kept for the paired table
    hit["oracle"] = [bool(truth[q] & set(tied[q])) for q in range(n)]

    # ---------------- the reasoning arms ----------------
    miss, offlist = Counter(), Counter()
    for arm in ("tie", "open", "scram"):
        h = []
        for q, pid in enumerate(range(n)):
            a = _norm((ans.get(pid) or {}).get(arm))
            if not a:
                miss[arm] += 1
            if arm in ("tie", "scram") and a and a not in {_norm(g) for g in tied[q]}:
                offlist[arm] += 1                        # answered off-list: scored as a miss, never silently
            h.append(bool(a) and a in {_norm(g) for g in truth[q]})
        hit["llm_" + arm] = h

    ARMS = ("random", "llm_scram", "freq", "logistic", "set_transformer", "llm_tie", "oracle", "llm_open")
    report(f"\n  RECALL@1 on the identical {n} reactions")
    report(f"    {'arm':<20}{'@1':>8}{'right':>9}    note")
    NOTE = {"random": "floor", "oracle": "ceiling for every list-constrained arm",
            "set_transformer": f"8,929 params, {len(T.SEEDS)}-seed mean "
                               f"{np.mean(st_seed):.3f}+-{np.std(st_seed):.3f}",
            "llm_scram": "SAME candidate lists, WRONG reaction -- chemistry-blind control",
            "llm_tie": "closed-book reasoning, same task as freq/logistic",
            "llm_open": "NO candidate list, ceiling 1.000 -- not a tie-break"}
    rec = {}
    for a in ARMS:
        v = float(np.mean(hit[a]))
        rec[a] = v
        report(f"    {a:<20}{v:>8.3f}{int(np.sum(hit[a])):>6}/{n}    {NOTE.get(a,'')}")
    if miss or offlist:
        report(f"    unanswered {dict(miss)} | answered off-list (scored as misses) {dict(offlist)}")

    # ---------------- paired tests against the arm each is trying to beat ----------------
    report(f"\n  PAIRED McNEMAR (n={n}; 1 reaction = {1/n:.3f} recall)")
    mcn = {}
    for a, b in (("llm_tie", "llm_scram"), ("llm_tie", "freq"), ("llm_tie", "logistic"),
                 ("llm_tie", "set_transformer"), ("llm_tie", "random"), ("llm_scram", "random"),
                 ("llm_open", "llm_tie")):
        X, Y = np.array(hit[a]), np.array(hit[b])
        w, l = int((X & ~Y).sum()), int((~X & Y).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
        mcn[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "p": float(p)}
        report(f"    {a:<14} vs {b:<18}{w:>4} win {l:>4} lose   p={p:.4f}"
               + ("  SIGNIFICANT" if p < 0.05 else ""))

    # ---------------- stratified diagnostics ----------------
    fame = np.array([max((freq.get(g, 0) for g in truth[q]), default=0) for q in range(n)])
    med = float(np.median(fame))
    nc = np.array([len(t) for t in tied])
    # a participant NAME containing the catalyst hands the answer over -- kept in, but never counted blind
    gim = np.array([any(g in " ".join(q["name"] for w in ("inputs", "outputs") for q in pz[i][w])
                        for g in truth[i]) for i in range(n)])
    strat = {}

    def block(title, note, rows):
        report(f"\n  {title}")
        report(f"  {note}")
        report(f"    {'stratum':<26}{'n':>5}{'freq':>9}{'llm_tie':>10}{'llm_open':>10}{'tie-freq':>11}")
        for nm, m_ in rows:
            k = int(m_.sum())
            if not k:
                continue
            f_, t_, o_ = (float(np.mean(np.array(hit[x])[m_])) for x in ("freq", "llm_tie", "llm_open"))
            # key on the FIRST WORD. Keying on the padded label and looking it up with a differently
            # padded literal is how the contamination verdict once printed "+0.000" and called the gain
            # memorisation, while the table directly above it showed +0.099 in the other direction.
            strat[nm.split()[0].lower()] = {"n": k, "freq": f_, "llm_tie": t_, "llm_open": o_}
            report(f"    {nm:<26}{k:>5}{f_:>9.3f}{t_:>10.3f}{o_:>10.3f}{t_-f_:>+11.3f}")

    block("CONTAMINATION DIAGNOSTIC -- is any advantage concentrated in FAMOUS enzymes?",
          f"Split by how many reactions the true catalyst runs elsewhere (median {med:.0f}). If the gain "
          f"lives only\n  in the famous half, that is recall of the source literature, not reasoning.",
          [("rare   (<= median)", fame <= med), ("famous (>  median)", fame > med)])

    block("WHERE A TIE-BREAK CAN ACTUALLY MATTER",
          f"{int((nc == 1).sum())} of {n} puzzles have a ONE-gene candidate list, so every list-constrained "
          f"arm is forced to\n  the same answer and contributes nothing to any difference. The bottom row is "
          f"the real contest.",
          [("singleton list", nc == 1), ("2+ candidates", nc >= 2)])

    block("THE GIVEAWAYS",
          f"In {int(gim.sum())} puzzles the true catalyst appears verbatim in a participant NAME -- a SUMO E2 "
          f"loaded onto\n  its own substrate. Real biology, so they stay, but the reasoning arm can read those "
          f"off.",
          [("answer named in reaction", gim), ("answer NOT named", ~gim)])

    # ---------------- reading ----------------
    report("\n  READING")
    lt, fq, lg = rec["llm_tie"], rec["freq"], rec["logistic"]
    sc = rec["llm_scram"]
    p_sc = mcn["llm_tie_vs_llm_scram"]["p"]
    report(f"  CONTROL FIRST. The scrambled arm saw the SAME candidate lists attached to the WRONG reaction")
    report(f"  and scored {sc:.3f}; the honest arm scored {lt:.3f} (paired p={p_sc:.4f}).")
    if lt - sc <= 0.02 or p_sc >= 0.05:
        report(f"  THE CHEMISTRY IS NOT BEING USED. Give the solver an unrelated reaction and it picks the same")
        report("  candidates about as often. Whatever the honest arm scores, it is choosing on properties of")
        report("  the LIST -- how familiar a gene looks -- not on what the reaction does. Read on accordingly.")
    else:
        report(f"  The reaction text IS doing work: {lt-sc:+.3f} over the same lists with the wrong chemistry.")
        report("  So the honest arm is reading the reaction, not just picking the most familiar-looking gene.")
    op, orc = rec["llm_open"], rec["oracle"]
    p_fq = mcn["llm_tie_vs_freq"]["p"]
    d_rare = strat["rare"]["llm_tie"] - strat["rare"]["freq"]
    d_fame = strat["famous"]["llm_tie"] - strat["famous"]["freq"]
    # the STRONGEST baseline is the one that matters, not the weakest. Beating freq while tying the
    # transformer is a much smaller claim than "reasoning beats counting" and must not be reported as one.
    best_b = max(("freq", fq), ("logistic", lg), ("set_transformer", rec["set_transformer"]),
                 key=lambda kv: kv[1])
    p_best = mcn.get(f"llm_tie_vs_{best_b[0]}", {}).get("p", 1.0)
    if lt > max(fq, lg) and p_fq < 0.05:
        report(f"  REASONING BEATS COUNTING. {lt:.3f} against freq {fq:.3f}, paired p={p_fq:.4f}, against a")
        report(f"  {orc:.3f} ceiling -- {(lt-rec['random'])/max(orc-rec['random'],1e-9):.0%} of the headroom "
               f"where six hand-built signals and a transformer claimed ~30%.")
        report(f"  What the feature arms lacked was reaction semantics, not capacity.")
        if p_best >= 0.05:
            report(f"  BUT NOT ESTABLISHED AGAINST THE STRONGEST BASELINE: {best_b[0]} reaches {best_b[1]:.3f}")
            report(f"  and the paired test cannot separate it from {lt:.3f} (p={p_best:.4f}). At n={n} the")
            report("  honest claim is 'beats frequency counting', not 'beats everything'.")
    elif abs(lt - fq) <= 0.05 or p_fq >= 0.05:
        report(f"  REASONING DOES NOT ORDER THE TIE GROUP EITHER. {lt:.3f} against freq {fq:.3f} "
               f"(paired p={p_fq:.4f}).")
        report(f"  That now makes eight approaches -- six hand-built signals, a set transformer and closed-book")
        report(f"  chemistry knowledge -- all landing near {fq:.2f} against a {orc:.3f} ceiling. The tie group")
        report("  is UNDERDETERMINED: the information needed to order it is not in the reaction.")
    else:
        report(f"  COUNTING BEATS REASONING. {lt:.3f} against freq {fq:.3f} (paired p={p_fq:.4f}). How often an")
        report("  enzyme already catalyses anything is a stronger prior than what the reaction chemically is.")
    report(f"\n  CONTAMINATION: the gain over freq is {d_rare:+.3f} on network-RARE catalysts and {d_fame:+.3f}")
    if d_rare >= d_fame:
        report("  on network-famous ones -- LARGER where the enzyme is rarer, which is the opposite of what")
        report("  pure memorisation of famous enzymes would produce.")
    else:
        report("  on network-famous ones -- concentrated in well-known enzymes, which is what memorisation of")
        report("  the source literature looks like.")
    report("  Read that with care in BOTH directions: 'rare' here means the catalyst runs few OTHER reactions")
    report("  in this network, which is not the same as rare in the literature. A one-reaction enzyme can still")
    report("  be a textbook enzyme, so this bounds the contamination loosely rather than ruling it out.")
    report(f"\n  OPEN-ENDED, no candidate list: {op:.3f}. The list-constrained ceiling is {orc:.3f}, so the")
    if op > orc:
        report(f"  reasoning arm names the right enzyme MORE often with no list than any arm can achieve with")
        report(f"  one ({op:.3f} vs {orc:.3f}). The candidate list is a CEILING, not a help: the chemistry-")
        report("  neighbour retrieval step is what limits this problem, and ranking work cannot fix it.")
    else:
        report(f"  candidate list is not the binding constraint ({op:.3f} <= {orc:.3f}); retrieval and")
        report("  reasoning fail on the same reactions.")
    cl = (nc >= 2) & ~gim
    lt_c, fq_c = float(np.mean(np.array(hit["llm_tie"])[cl])), float(np.mean(np.array(hit["freq"])[cl]))
    strat["clean_contest"] = {"n": int(cl.sum()), "freq": fq_c, "llm_tie": lt_c}
    report(f"\n  CLEANEST COMPARISON -- {int(cl.sum())} puzzles with 2+ candidates AND no answer named in the")
    report(f"  reaction: freq {fq_c:.3f} vs closed-book reasoning {lt_c:.3f} ({lt_c-fq_c:+.3f}).")
    report(f"\n  OUT OF 10: freq gets {10*fq:.1f}, the transformer {10*rec['set_transformer']:.1f}, "
           f"closed-book reasoning {10*lt:.1f} on the same task,")
    report(f"  and {10*op:.1f} with no candidate list at all. Perfect tie-breaking would get {10*orc:.1f}.")

    json.dump({"test": "cell_orphan_llm", "n": n, "recall": rec, "mcnemar": mcn, "strata": strat,
               "set_transformer_seeds": st_seed, "unanswered": dict(miss), "off_list": dict(offlist),
               "sample_seed": SAMPLE_SEED, "log": log},
              open(OUT / "cell_orphan_llm.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_llm.json'}")
    return 0


def _train(torch, B, seed):
    """The same SetRanker, retrained here so its number shares this sample's denominator."""
    import torch.nn as nn
    torch.manual_seed(seed)
    ex, tr_i = B["examples"], B["tr_i"]

    class SetRanker(nn.Module):
        def __init__(s, fd, d=T.D_MODEL, h=T.N_HEADS):
            super().__init__()
            s.proj = nn.Linear(fd, d)
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.attn = nn.MultiheadAttention(d, h, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            s.out = nn.Linear(d, 1)

        def forward(s, X):
            h = s.proj(X).unsqueeze(0)
            a, _ = s.attn(s.ln1(h), s.ln1(h), s.ln1(h), need_weights=False)
            h = h + a
            h = h + s.ff(s.ln2(h))
            return s.out(h.squeeze(0)).squeeze(-1)

    m = SetRanker(B["FD"])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    use = [ex[k] for k in tr_i if ex[k]["has_truth"] and len(ex[k]["tied"]) > 1]
    shuf = np.random.default_rng(seed)
    for _ in range(T.EPOCHS):
        shuf.shuffle(use)
        m.train()
        for e in use:
            opt.zero_grad()
            s_ = m(torch.tensor(e["Xz"], dtype=torch.float32))
            loss = -torch.log_softmax(s_, 0)[torch.tensor(e["y"]).bool()].mean()
            loss.backward()
            opt.step()
    m.eval()
    return m, None, None


def _fwd(torch, m, e):
    with torch.no_grad():
        return m(torch.tensor(e["Xz"], dtype=torch.float32)).numpy()


if __name__ == "__main__":
    raise SystemExit(main())
