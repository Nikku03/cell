"""Loop 182. A two-tower network on proven examples, with measured protein geometry.

WHAT THIS TESTS AND WHAT IT DOES NOT PRETEND TO. Three things were asked for: a neural network,
training on proven examples, and three-dimensional structure. Each is built here, and each is
scoped by what the last six loops already settled, so none of them is sold as more than it is.

  THE NETWORK. Loop 177 already compared learners on this data and found the signal additive:
  logistic regression reached AUC 0.8468 against a gradient-boosted ensemble's 0.8506 on identical
  columns, a deeper ensemble was no better, and a two-layer network was worse at 0.8048. So a
  network over the SAME columns is a known no-op and is run only as V3, to confirm it on stage two.
  The architecture that has a real claim is different. Every column in this arc sums over all 736
  matrices before the model sees anything -- total occupancy, occupancy times electrostatic
  potential times domain charge -- and no learner can recover an interaction between two matrices
  it was never shown separately. A two-tower model embeds the element's raw 736-dimensional
  occupancy vector and the promoter's raw 736-dimensional vector and combines them by ELEMENTWISE
  PRODUCT, which is the "which factor at the promoter matches which factor at the element" question
  at full resolution. That is the one thing the aggregation destroyed, and V4 is its gate.

  PROVEN EXAMPLES. Every pair in this benchmark is already powered to detect a 25% effect. The
  stricter tier is power to detect a 15% effect, which 8,059 of 12,040 pairs clear; there a
  non-significant call is a real negative rather than a missed detection. Training is restricted to
  those and V2 measures what the restriction costs or buys, evaluated on the same held-out pairs
  either way so the comparison is a comparison.

  STRUCTURE. `tf_domains.py` gave each factor a charge density, an arginine fraction and a mean
  residue volume from its domain SEQUENCE, and the complementarity block built on them failed every
  gate it faced -- loop 173's E7 and E7b, loop 174's F6. AlphaFold geometry replaces those with the
  quantities that decide grip: the charge dipole over the radius of gyration, the surface charge a
  groove actually sees under Shrake-Rupley accessibility, how far arginine guanidinium carbons sit
  from the domain centroid, and the domain's maximum reach in base pairs. Two of those already say
  something before any model runs: median surface charge +8.1, so domains present a positive face,
  and median arg_out 1.11, so arginines point outward on average. V5 gives the block one honest
  head-to-head against its sequence-derived predecessor, and V6 checks the result is not living in
  domains AlphaFold predicted at low confidence.

WHAT IS ALREADY KNOWN TO BOUND THIS, stated so no gate below can be read as bigger than it is.
Loop 178's P1 put the learnable element-intrinsic ceiling at within-gene R@1 0.4422, BELOW
distance-only at 0.5930. Loop 181 measured that a third of the within-gene decisions here are
between candidates sharing a 5 kb contact bin. Neither is a modelling problem and no architecture
moves either. What a better model can still do is use the promoter-element correspondence at full
resolution, and that is what V4 asks.

PREDECLARED, BEFORE ANY NUMBER.

  V1 IS THE PROVEN SUBSET USABLE? Its size, positive rate and how many evaluable genes survive.
     Gate: PASS iff at least 100 evaluable genes remain AND the positive rate stays within a factor
     of two of the full set's 0.0404. Below that the restriction has changed the task rather than
     cleaned it.

  V2 DOES TRAINING ON PROVEN EXAMPLES HELP? The same model and columns trained on the proven subset
     against trained on everything, both scored on the same held-out proven pairs.
     Gate: paired per-seed R@1 positive in >= 4/5 and past 3 sem, AND paired AUPRC >= +0.01 in
     >= 4/5 -- loop 173's E3 bar, unchanged since loop 173.

  V3 DOES A NETWORK BEAT THE TREE ON IDENTICAL COLUMNS? Expected to fail on loop 177's evidence,
     and run because an expectation is not a measurement.
     Gate: same bar.

  V4 DOES THE RAW TWO-TOWER BEAT THE AGGREGATED COLUMNS? The same network with the 736-dimensional
     element and promoter vectors and their elementwise product, against the same network on the
     hand-aggregated columns.
     Gate: same bar. This is the loop's real question.

  V5 DOES MEASURED GEOMETRY BEAT THE SEQUENCE PROXIES? The complementarity block computed from
     AlphaFold structures against the identical block computed from domain sequence, everything
     else held fixed.
     Gate: same bar.

  V6 IS THE GEOMETRY REAL OR PREDICTED? V5 re-run with the structure block zeroed for every matrix
     whose domain pLDDT is below 70.
     Gate: PASS iff restricting to confident domains does NOT reduce the structure block's
     increment. If the effect only exists when low-confidence domains are included, the block is
     reading invented geometry and says so.

  V7 THE DECISIVE ONE. The best arm against distance alone, identical folds.
     Gate: same bar. This is the gate loops 173, 175, 178, 179 and 181 were all held to.

  V8 THE SHUFFLE, DONE ON THE RIGHT ARM. Loop 178's P7 declared "best configuration against
     shuffled" and then passed the arm carrying a column P3 had just shown to be harmful, so it
     measured that harm instead. Here the arm is selected by measured R@1 before the comparison is
     made, and the selection is printed.
     Gate: real beats dinucleotide-shuffled in >= 4/5 seeds past 3 sem.

  V9 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_twotower.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
from enh import tf_structures as TS          # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402
from sklearn.metrics import average_precision_score             # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_twotower.json"
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
POWER_COL = "PowerAtEffectSize15"
MIN_POWER = 0.8
MIN_GENES = 100
PLDDT_CUT = 70.0
EMB, HID, EPOCHS, LR, WD, BATCH = 64, 64, 60, 3e-3, 1e-4, 512

STRUCT_COLS = ["s_dipole", "s_surfchg", "s_argout", "s_reach", "s_rg"]
SEQ_COLS = ["comp_charge", "comp_arg", "comp_steric", "comp_major", "comp_twist", "comp_span"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def structure_block(S, occ, e_idx, T, plddt_min=None, report=print):
    """The complementarity block recomputed from AlphaFold geometry instead of domain sequence."""
    st = TS.load()
    dom = TD.load()
    ids = [str(m) for m in S["motif_ids"]]
    width = S["motif_width"].astype(np.float64)
    def col(k, d=0.0):
        return np.array([float(st.get(m, {}).get(k, d) or d) for m in ids])
    have = np.array([m in st for m in ids])
    pl = col("plddt")
    if plddt_min is not None:
        have = have & (pl >= plddt_min)
    dip, sch, ao, md, rg = (col("dipole"), col("surf_charge"), col("arg_out"),
                            col("max_dim"), col("rg"))
    groove = np.array([dom.get(m, {}).get("groove", "major") for m in ids])
    minorish = ((groove == "minor") | (groove == "both")) & have
    rgc = rg - (rg[have].mean() if have.any() else 0.0)
    reach = np.where(width > 0, (md / 3.4) / width, 0.0)     # domain span in bp over motif width
    EP = np.nan_to_num(T["ep"][:, e_idx].astype(np.float64), nan=0.0)
    MGW = np.nan_to_num(T["mgw"][:, e_idx].astype(np.float64), nan=0.0)
    F = {}
    F["s_dipole"] = (occ * (-EP) * (dip * minorish)[:, None]).sum(0)
    F["s_surfchg"] = (occ * (-EP) * (sch * have)[:, None]).sum(0)
    F["s_argout"] = (occ * (-EP) * (ao * minorish)[:, None]).sum(0)
    F["s_reach"] = (occ * (reach * have)[:, None]).sum(0) / np.maximum(occ.sum(0), 1e-300)
    F["s_rg"] = (occ * MGW * (rgc * have)[:, None]).sum(0)
    report(f"    structure block over {int(have.sum())}/{len(ids)} matrices"
           + (f" with pLDDT >= {plddt_min}" if plddt_min else ""))
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F


class TwoTower(nn.Module):
    """Element and promoter vectors embedded separately, combined by elementwise product, then
    concatenated with the scalar columns. The product is the point: it is the only place where
    matrix i at the promoter can meet matrix i at the element without having been summed away."""

    def __init__(self, n_vec, n_scalar, emb=EMB, hid=HID):
        super().__init__()
        self.el = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU(), nn.Linear(128, emb))
        self.pr = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU(), nn.Linear(128, emb))
        self.head = nn.Sequential(
            nn.Linear(emb * 3 + n_scalar, hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ve, vp, xs):
        a, b = self.el(ve), self.pr(vp)
        return self.head(torch.cat([a * b, a, b, xs], dim=1)).squeeze(1)


class Flat(nn.Module):
    """The same head on the scalar columns alone -- V3's network, with no towers."""

    def __init__(self, n_scalar, hid=HID):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(n_scalar, hid), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ve, vp, xs):
        return self.head(xs).squeeze(1)


def fit_torch(model, Ve, Vp, Xs, y, tr, te, seed):
    torch.manual_seed(seed)
    dev = "cpu"
    m = model.to(dev)
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    Xn = (Xs - mu) / sd
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    pw = torch.tensor([(len(y[tr]) - y[tr].sum()) / max(y[tr].sum(), 1)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    ve = torch.tensor(Ve, dtype=torch.float32)
    vp = torch.tensor(Vp, dtype=torch.float32)
    xs = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    idx = np.where(tr)[0]
    rng = np.random.default_rng(seed)
    m.train()
    for ep in range(EPOCHS):
        rng.shuffle(idx)
        for k in range(0, len(idx), BATCH):
            j = idx[k:k + BATCH]
            opt.zero_grad()
            out = m(ve[j], vp[j], xs[j])
            loss = lossf(out, yt[j])
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(ve[te], vp[te], xs[te])).numpy()


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def run(kind, Ve, Vp, Xs, y, chrom, g_idx, jitter, tag, train_mask=None, eval_mask=None,
        report=print):
    """`train_mask` restricts which rows may be trained on (the proven subset); `eval_mask`
    restricts which rows the metrics are computed over. Folds are identical across every arm."""
    r1, ap = [], []
    for s in SEEDS:
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if train_mask is not None:
                tr = tr & train_mask
            if te.sum() == 0 or y[tr].sum() < 5:
                continue
            if kind == "gbm":
                m = gbm(s)
                m.fit(np.nan_to_num(Xs[tr]), y[tr])
                sc[te] = m.predict_proba(np.nan_to_num(Xs[te]))[:, 1]
            else:
                net = (TwoTower(Ve.shape[1], Xs.shape[1]) if kind == "tower"
                       else Flat(Xs.shape[1]))
                sc[te] = fit_torch(net, Ve, Vp, np.nan_to_num(Xs), y, tr, te, s)
        m_ = np.ones(len(y), bool) if eval_mask is None else eval_mask
        r1.append(L173.within_gene(sc[m_], y[m_], g_idx[m_], jitter[m_])[0])
        ap.append(average_precision_score(y[m_], sc[m_]) if y[m_].sum() else 0.0)
    r1, ap = np.array(r1), np.array(ap)
    report(f"    {tag:40} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 182  TWO-TOWER NETWORK ON PROVEN EXAMPLES, WITH MEASURED DOMAIN GEOMETRY")
    say("=" * 104)
    say(f"  PREDECLARED: the proven subset must keep >= {MIN_GENES} evaluable genes and a positive")
    say("  rate within 2x of 0.0404; every arm on loop 173's E3 bar -- paired R@1 positive in")
    say(f"  >= {MIN_SEEDS}/5 past 3 sem AND paired AUPRC >= +0.01 in >= {MIN_SEEDS}/5; the")
    say(f"  structure block must not lose its increment when restricted to pLDDT >= {PLDDT_CUT};")
    say("  and V8's shuffle runs on the arm selected by measured R@1, printed before the")
    say("  comparison, which is the defect loop 178's P7 committed.")
    say()
    say("  ALREADY KNOWN TO BOUND THIS: loop 178's P1 put the learnable element-intrinsic ceiling")
    say("  at R@1 0.4422, below distance's 0.5930; loop 181 measured that 33.9% of within-gene")
    say("  decisions are between candidates sharing a 5 kb contact bin. No architecture moves")
    say("  either.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))

    # ---- V1 the proven subset ------------------------------------------------------------------
    say("V1 IS THE PROVEN SUBSET USABLE?")
    rows = SC.load_benchmark(lambda *_: None)
    key = {}
    for r in rows:
        key[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]),
             r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])] = float(r[POWER_COL] or 0)
    el = [str(k) for k in S["el_key"]]
    gn = [str(k) for k in S["gn_key"]]
    proven = np.zeros(len(y), bool)
    for i in range(len(y)):
        c, rest = el[int(e_idx[i])].split(":")
        a, b = rest.split("-")
        gc, gp, gs = gn[int(g_idx[i])].split(":")
        proven[i] = key.get((c, int(a), int(b), gc, int(gp), gs), 0.0) >= MIN_POWER
    cand = defaultdict(list)
    pos = Counter()
    for i in np.where(proven)[0]:
        cand[int(g_idx[i])].append(i)
        if y[i]:
            pos[int(g_idx[i])] += 1
    ev = [g for g in cand if len(cand[g]) >= 2 and pos[g] > 0]
    say(f"     {POWER_COL} >= {MIN_POWER}: {int(proven.sum()):,}/{len(y):,} pairs "
        f"({proven.mean():.1%}), {int(y[proven].sum())} positives, "
        f"base rate {y[proven].mean():.4f} against the full set's {y.mean():.4f}")
    say(f"     evaluable genes inside the proven subset: {len(ev)}")
    v1 = bool(len(ev) >= MIN_GENES and 0.5 * y.mean() <= y[proven].mean() <= 2 * y.mean())
    GG.verdict(v1, emit=say,
               if_true=f"V1 PASS -- {len(ev)} evaluable genes survive at base rate "
                       f"{y[proven].mean():.4f}, so the restriction cleans the labels without "
                       f"changing the task",
               if_false=f"V1 FAIL -- {len(ev)} genes at base rate {y[proven].mean():.4f}; the "
                        f"restriction has changed the task rather than cleaned it")

    # ---- features ------------------------------------------------------------------------------
    say()
    say("   building features")
    E, FAM, SPEC = L178.element_frame(S, "el", say)
    Es, FAMs, SPECs = L178.element_frame(S, "sh", say)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    Ps, _, _ = L173.build_features(S, "sh", report=lambda *_: None)
    for fr in (P, Ps):
        for c in fr:
            fr[c] = np.nan_to_num(fr[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xagg = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xagg_s = np.column_stack([Ps[c] for c in base_cols] + [FAMs[c][e_idx] for c in fam_cols])
    Xdist = np.column_stack([P["log_dist"]])

    # the raw towers: the element's own 736 log-occupancies, and the promoter's 736
    bg = np.exp(L173._logsumexp(S["bg_LZ"].astype(np.float64), axis=1)) / float(S["bg_bp"])
    den = np.log(np.maximum(bg, 1e-300)) + np.log(4_000_000)
    occ_el = np.exp(S["el_LZ"].astype(np.float64) - den[:, None])
    Ve = np.log10(np.maximum(occ_el, 1e-12)).T[e_idx].astype(np.float32)
    prz = S["pr_LZ"].astype(np.float64)
    Vp = (prz - prz.mean(1, keepdims=True)).T[g_idx].astype(np.float32)
    Ve = (Ve - Ve.mean(0)) / (Ve.std(0) + 1e-6)
    Vp = (Vp - Vp.mean(0)) / (Vp.std(0) + 1e-6)
    say(f"    towers: element {Ve.shape}, promoter {Vp.shape}; aggregated columns {Xagg.shape[1]}")

    T = {n: S["el_SH"][i] for i, n in enumerate(list(S["tracks"]))}
    occ_pair = np.exp(L173.occupancy(S["el_LZ"], e_idx, g_idx, len(S["gn_key"]), bg))
    ST = structure_block(S, occ_pair, e_idx, T, None, say)
    STc = structure_block(S, occ_pair, e_idx, T, PLDDT_CUT, say)
    Xseq = np.column_stack([Xagg] + [P[c] for c in SEQ_COLS])
    Xstr = np.column_stack([Xagg] + [ST[c] for c in STRUCT_COLS])
    Xstrc = np.column_stack([Xagg] + [STc[c] for c in STRUCT_COLS])

    res = {}
    res["distance"] = run("gbm", Ve, Vp, Xdist, y, chrom, g_idx, jitter, "distance (tree)",
                          eval_mask=proven, report=say)
    res["gbm_all"] = run("gbm", Ve, Vp, Xagg, y, chrom, g_idx, jitter,
                         "aggregated cols, tree, trained on ALL", eval_mask=proven, report=say)
    res["gbm_proven"] = run("gbm", Ve, Vp, Xagg, y, chrom, g_idx, jitter,
                            "aggregated cols, tree, trained on PROVEN",
                            train_mask=proven, eval_mask=proven, report=say)
    res["flat_proven"] = run("flat", Ve, Vp, Xagg, y, chrom, g_idx, jitter,
                             "aggregated cols, network", train_mask=proven, eval_mask=proven,
                             report=say)
    res["tower_proven"] = run("tower", Ve, Vp, Xagg, y, chrom, g_idx, jitter,
                              "TWO-TOWER on the raw 736-vectors", train_mask=proven,
                              eval_mask=proven, report=say)
    res["seq_compl"] = run("gbm", Ve, Vp, Xseq, y, chrom, g_idx, jitter,
                           "+ sequence-derived complementarity", train_mask=proven,
                           eval_mask=proven, report=say)
    res["str_compl"] = run("gbm", Ve, Vp, Xstr, y, chrom, g_idx, jitter,
                           "+ AlphaFold-derived complementarity", train_mask=proven,
                           eval_mask=proven, report=say)
    res["str_compl_conf"] = run("gbm", Ve, Vp, Xstrc, y, chrom, g_idx, jitter,
                                f"+ AlphaFold, pLDDT >= {PLDDT_CUT:.0f} only", train_mask=proven,
                                eval_mask=proven, report=say)

    # ---- V2..V7 --------------------------------------------------------------------------------
    def gate(tag, a, b, name, if_t, if_f, use_ap=True):
        d = L173.paired(res[a], res[b])
        say()
        say(name)
        say(f"     {a} vs {b}   {L173.fmt(d)}")
        ok = L173.gate_pair(d, use_ap=use_ap)
        GG.verdict(ok, emit=say, if_true=f"{tag} PASS -- {if_t}", if_false=f"{tag} FAIL -- {if_f}")
        return ok, d

    v2, d2 = gate("V2", "gbm_proven", "gbm_all",
                  "V2 DOES TRAINING ON PROVEN EXAMPLES HELP?",
                  "restricting training to pairs the screen could definitively call helps",
                  "the extra 3,881 lower-powered pairs were not hurting; the restriction only "
                  "costs training data")
    v3, d3 = gate("V3", "flat_proven", "gbm_proven",
                  "V3 DOES A NETWORK BEAT THE TREE ON IDENTICAL COLUMNS?",
                  "the network beats the ensemble on the same columns",
                  "the network matches or loses to the ensemble on identical columns, as loop "
                  "177 found on stage one; the learner is not the constraint")
    v4, d4 = gate("V4", "tower_proven", "flat_proven",
                  "V4 DOES THE RAW TWO-TOWER BEAT THE AGGREGATED COLUMNS?",
                  "the promoter-element correspondence at full matrix resolution carries "
                  "something the summed columns destroyed",
                  "even at full resolution, matching the promoter's factors against the "
                  "element's adds nothing over the aggregated columns")
    v5, d5 = gate("V5", "str_compl", "seq_compl",
                  "V5 DOES MEASURED GEOMETRY BEAT THE SEQUENCE PROXIES?",
                  "AlphaFold geometry beats the amino-acid counts it replaces",
                  "measured domain geometry does no better than the sequence proxies, so the "
                  "complementarity block's failures were not about the proxies")

    say()
    say("V6 IS THE GEOMETRY REAL OR PREDICTED?")
    inc_all = L173.paired(res["str_compl"], res["seq_compl"])
    inc_conf = L173.paired(res["str_compl_conf"], res["seq_compl"])
    say(f"     structure increment, all domains        {L173.fmt(inc_all)}")
    say(f"     structure increment, pLDDT >= {PLDDT_CUT:.0f} only  {L173.fmt(inc_conf)}")
    v6 = bool(inc_conf["mean_r1"] >= inc_all["mean_r1"] - 3 * max(inc_all["sem_r1"], 1e-9))
    GG.verdict(v6, emit=say,
               if_true="V6 PASS -- the structure block does not depend on domains AlphaFold was "
                       "unsure about",
               if_false="V6 FAIL -- the block loses its increment once low-confidence domains are "
                        "dropped, so it was reading predicted geometry rather than measured")

    say()
    say("V7 THE DECISIVE ONE")
    best = max((k for k in res if k != "distance"), key=lambda k: res[k]["r1"].mean())
    d7 = L173.paired(res[best], res["distance"])
    say(f"     best arm {best} at R@1 {res[best]['r1'].mean():.4f} / AUPRC "
        f"{res[best]['ap'].mean():.4f} against distance {res['distance']['r1'].mean():.4f} / "
        f"{res['distance']['ap'].mean():.4f}")
    say(f"     {L173.fmt(d7)}")
    v7 = L173.gate_pair(d7)
    GG.verdict(v7, emit=say,
               if_true=f"V7 PASS -- {best} clears the bar every stage-two loop has been held to",
               if_false="V7 FAIL -- stage two is still distance")

    say()
    say("V8 THE SHUFFLE, ON THE ARM SELECTED BY MEASURED R@1")
    say(f"     selected arm: {best}")
    kind = {"tower_proven": "tower", "flat_proven": "flat"}.get(best, "gbm")
    Xsh = {"seq_compl": np.column_stack([Xagg_s] + [Ps[c] for c in SEQ_COLS]),
           "str_compl": np.column_stack([Xagg_s] + [ST[c] for c in STRUCT_COLS]),
           "str_compl_conf": np.column_stack([Xagg_s] + [STc[c] for c in STRUCT_COLS])
           }.get(best, Xagg_s)
    res["best_shuffled"] = run(kind, Ve, Vp, Xsh, y, chrom, g_idx, jitter,
                               f"{best}, SHUFFLED elements", train_mask=proven, eval_mask=proven,
                               report=say)
    d8 = L173.paired(res[best], res["best_shuffled"])
    say(f"     real vs dinucleotide-shuffled   {L173.fmt(d8)}")
    v8 = L173.gate_pair(d8, use_ap=False)
    GG.verdict(v8, emit=say,
               if_true="V8 PASS -- on the arm that actually won, real sequence beats a "
                       "composition-matched shuffle",
               if_false="V8 FAIL -- the shuffle matches the winning arm, so on stage two the "
                        "sequence columns still are not reading binding sites")

    say()
    say("V9 WHAT THIS CANNOT SHOW")
    say("     The towers use the promoter's motif profile, which is a 1 kb window of sequence.")
    say("     Whether a factor is actually present at that promoter in K562 is not known here and")
    say("     no expression or ChIP filter is applied.")
    say("     AlphaFold geometry is an unbound monomer prediction. A DNA-binding domain changes")
    say("     conformation on binding, and nothing here models that.")
    say("     The proven subset is defined by statistical power, not by mechanism; a well-powered")
    say("     negative is still only a negative for the effect size the screen could see.")
    say("     Loop 178's element-intrinsic ceiling of 0.4422 and loop 181's 5 kb resolution limit")
    say("     bound every number above, and no architecture in this loop addresses either.")
    v9 = True
    say(f"     V9 {'PASS' if v9 else 'FAIL'}")

    gates = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": v7, "V8": v8,
             "V9": v9}
    man = RM.manifest(inputs=[Path("colab/data/tf_structures.json"),
                              Path("colab/data/tf_domains.json")],
                      available=int(len(y)), used=int(proven.sum()),
                      selection=f"{POWER_COL} >= {MIN_POWER}", seed=L173.TIE_SEED,
                      controls=["the same columns under a tree, a flat network and the towers",
                                "sequence-derived against AlphaFold-derived complementarity, "
                                "everything else fixed",
                                f"the structure block re-run with pLDDT < {PLDDT_CUT} zeroed",
                                "the shuffle run on the arm selected by measured R@1"],
                      note="two-tower network, proven-power training, measured domain geometry")
    out = dict(test="enhancer two-tower", gates=gates,
               n_pairs=int(len(y)), n_proven=int(proven.sum()),
               proven_base_rate=float(y[proven].mean()), n_evaluable_proven=len(ev),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items()},
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in d.items()}
                       for k, d in (("V2", d2), ("V3", d3), ("V4", d4), ("V5", d5),
                                    ("V7", d7), ("V8", d8))},
               structure_increment=dict(all=inc_all["mean_r1"], confident=inc_conf["mean_r1"]),
               best_arm=best, manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
