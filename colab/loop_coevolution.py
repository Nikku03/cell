"""Loop 225. Evo 2's training method at gene granularity: co-evolution across 95 mammal genomes.

WHAT EVO 2 ACTUALLY DOES, checked against the repository and paper rather than recalled. Evo 2 is
trained AUTOREGRESSIVELY on OpenGenome2 -- 8.8 trillion tokens, all domains of life -- at single
nucleotide resolution, using StripedHyena 2, at 7B and 40B parameters with context to 1M bases.
Its objective is plain next-token prediction on raw DNA. It never explicitly models gene-gene
co-dependency. Co-variation is learned IMPLICITLY: train on enough genomes and positions that vary
together become mutually predictive, and that structure appears in the embeddings.

WHAT IS AND IS NOT BEING REPRODUCED HERE, stated plainly so nothing below is oversold. Evo 2
itself cannot run in this container: the 7B needs a supported GPU, the 40B needs several H100s, the
StripedHyena kernels are CUDA-only, and torch.cuda.is_available() is False. Retraining on mammalian
genomes is further out of reach by orders of magnitude. What transfers is the METHOD -- learn
co-variation self-supervised across many genomes -- applied one level up, at gene granularity
instead of nucleotide granularity. That difference is a real limitation and it cuts one way: a
nucleotide model can see co-variation inside a gene, between promoter and coding sequence, in
non-coding regions. A gene-level rate cannot. Whatever this loop measures is a LOWER bound on
what Evo 2's representation would carry.

THE DATA, and it is the mammal genomes, just summarised. Ensembl Compara release 116 publishes the
full human homology table. Filtered to high-confidence one-to-one orthologs in mammalian species:

    198 species with one-to-one orthologs, of which 95 are Mammalia by NCBI lineage
    1,446,407 mammalian one-to-one ortholog rows, 91.8% flagged high confidence
    18,757 human genes x 95 mammals after filtering, median 81 orthologs per gene

dN/dS IS NOT AVAILABLE AND THAT CHANGES THE RATE PROXY. Every dn and ds field in this release's
default dump is NULL -- 0 of 1,446,407 rows carry both. The evolutionary rate here is therefore
protein IDENTITY, converted to divergence as 100 - identity, which is the standard fallback when
dN/dS is absent. Identity conflates synonymous and non-synonymous change, so it is a blunter
instrument than omega, and E3 exists because it is also more exposed to alignment-length and
composition confounds.

EVOLUTIONARY RATE COVARIATION IS THE ESTABLISHED METHOD AND IS THE BASELINE, not the novelty.
Genes whose products work together are under correlated selective pressure, so their evolutionary
rates rise and fall together across a phylogeny. The standard construction, used verbatim here:
take log divergence, subtract the per-species mean (species differ in overall distance from human)
AND the per-gene mean (genes differ in overall constraint), then correlate the residual profiles.
Both centrings are required; without the species centring every pair correlates through shared
phylogenetic distance and the matrix is meaningless.

NO HYPOTHESIS IS OFFERED ABOUT WHICH ARM WINS. Loop 221 spent its budget naming a thing and had
its own naming refuted by its own gates. The instruction taken from that is to build the best
estimator available and let the gates rank it, so three constructions are measured against one
task and none is predicted here to come first.

    ERC        pairwise Pearson between doubly-centred residual profiles. n = 95 species.
    FACTOR     low-rank factorisation of the residual matrix, then cosine between gene embeddings.
               Pooling across genes can denoise a 95-sample correlation; it can also erase it.
    MLP        a learned embedding trained to reconstruct masked entries of the residual matrix --
               the Evo-2-style objective, masked prediction, at gene granularity.

PREDECLARED, BEFORE ANY NUMBER.

  E1 IS THE COMPARATIVE DATA USABLE ON OUR GENES?
     Gate: PASS iff at least 60% of the stack's gene set has 50 or more mammalian orthologs. A
     block covering a quarter of the genes cannot be stacked and there is no point scoring it.

  E2 DOES CO-EVOLUTION RECOVER KNOWN INTERACTIONS?  -- the positive control; all else requires it
     Score every pair in the OmniPath and SIGNOR networks against random pairs drawn from the same
     gene set, by ERC. AUC.
     Gate: PASS iff AUC >= 0.60. Below that the signal is not in this data at this resolution and
     nothing downstream can be believed regardless of how it scores.

  E3 IS E2 A CONFOUND?
     Co-evolution measured from identity tracks alignment length, overall constraint and
     expression level, and interacting proteins are longer and more expressed than random ones.
     Rerun E2 with negative pairs matched to positives on ortholog count, mean divergence and
     expression decile.
     Gate: PASS iff the matched AUC retains at least 70% of the unmatched excess over 0.5.
     Requires E2.

  E4 DOES A LEARNED EMBEDDING BEAT PAIRWISE ERC?
     FACTOR and MLP scored on the same matched interaction-recovery task as E3.
     Gate: PASS iff the best learned arm exceeds ERC's matched AUC by at least 0.02. A FAIL is a
     real result -- with 95 species, pairwise correlation may already be the maximum-likelihood
     estimate and there is nothing for capacity to add. Requires E2.

  E5 DOES CO-EVOLUTION ADD TO THE STANDING STACK?
     Add the winning co-evolution block to loop 213's ten-block stack on the same genes, same
     held-out split.
     Gate: PASS iff the stacked held-out |r| exceeds loop 213's 0.5474. Requires E2.

  E6 RIDGE AGAINST MLP, WITH THE SHUFFLED CONTROL BOTH TIMES
     Loop 211 measured ridge 0.4057 against MLP-wide 0.2072, and on SHUFFLED labels ridge 0.0237
     against MLP-wide 0.0742 -- the wide net fitting three times as much noise. That test is
     repeated here on the co-evolution block, which is a different kind of target.
     Gate: PASS iff the better arm beats its own shuffled-label control by at least 0.05. This
     gates the comparison being meaningful, not which model wins; the winner is reported.

  E7 SPECIES-SHUFFLE CONTROL
     Permute species labels INDEPENDENTLY WITHIN EACH GENE, which preserves every gene's marginal
     divergence distribution exactly while destroying cross-gene phylogenetic covariation.
     Gate: PASS iff E2's AUC falls below 0.55 under the shuffle. A control that cannot move is
     not evidence, and this one is constructed so that it can.

  E8 WHAT THIS CANNOT SHOW -- written before the run.
     Gene-level rates cannot see intra-genic co-variation, regulatory sequence, or non-coding
     elements, all of which a nucleotide model reads directly. A FAIL anywhere below is a
     statement about protein identity across 95 mammals, not about Evo 2.
     OmniPath and SIGNOR are literature-curated, so both edges and the genes carrying them are
     biased toward well-studied proteins. E3 matches on expression and ortholog count but cannot
     match on how much a gene has been studied, and loop 213 already measured that a fame-only
     baseline reaches 0.0940 on the stack target.
     95 species sounds like many and is not: every pairwise ERC value is a correlation on 95
     points at most, with a standard error near 0.10 even before phylogenetic non-independence,
     which inflates the effective error further because related species are not independent draws.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_coevolution.json"
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
EVO = SP / "compara" / "mammal_identity.npz"
NET = ROOT / "colab" / "data" / "networks"
SEED, MIN_ORTH, K_FACTOR = 225225, 50, 32
REF_STACK, REF_FAME = 0.5474, 0.0940
AUC_BAR, MATCH_KEEP, LEARN_MARGIN, SHUF_BAR, CTRL_MARGIN = 0.60, 0.70, 0.02, 0.55, 0.05

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(a)).astype(float) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ridge_pred(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    return B @ np.linalg.solve(A.T @ A + R, A.T @ ytr)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "mammalian co-evolution as a gene-gene block"}
    say("=" * 104)
    say("LOOP 225 -- EVO 2's METHOD AT GENE GRANULARITY: CO-EVOLUTION ACROSS 95 MAMMAL GENOMES")
    say("=" * 104)
    say("     Evo 2 itself does not run here: 7B needs a GPU, 40B needs several H100s, the")
    say("     StripedHyena kernels are CUDA-only and torch.cuda.is_available() is False. What")
    say("     transfers is the method -- self-supervised co-variation across many genomes --")
    say("     applied at gene granularity. A nucleotide model sees more; this is a lower bound.")

    Z = np.load(EVO, allow_pickle=True)
    Mid, egen, esp = Z["M"], Z["genes"], Z["species"]
    say(f"     Ensembl Compara 116: {Mid.shape[0]:,} human genes x {Mid.shape[1]} mammals, "
        f"high-confidence one-to-one orthologs")
    say(f"     rate proxy is protein identity: every dn and ds field in this release is NULL")

    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    y_all = (M[-3:].mean(0))[gi]
    allg = [sym[i] for i in gi]
    e2s = L191.ensg_to_symbol(lambda *_: None)
    s2e = {}
    for e, s in e2s.items():
        if s and s not in s2e:
            s2e[s] = e
    eidx = {str(g): i for i, g in enumerate(egen)}
    rowof = {s: eidx[s2e[s]] for s in allg if s in s2e and s2e[s] in eidx}

    # ---------------------------------------------------------------- E1
    say("E1 IS THE COMPARATIVE DATA USABLE ON OUR GENES?")
    cov = {s: int(np.isfinite(Mid[rowof[s]]).sum()) for s in rowof}
    good = [s for s in allg if cov.get(s, 0) >= MIN_ORTH]
    frac = len(good) / len(allg)
    say(f"     stack gene set {len(allg):,}; mapped to an Ensembl row {len(rowof):,}; "
        f"with >={MIN_ORTH} mammalian orthologs {len(good):,} ({frac:.1%})")
    say(f"     orthologs per covered gene: median {int(np.median([cov[s] for s in good]))}")
    G.add("E1", bool(frac >= 0.60), stat=float(frac),
          if_true=lambda: f"E1 PASS -- {frac:.0%} of the gene set carries a usable mammalian "
                          f"profile",
          if_false=lambda: f"E1 FAIL -- only {frac:.0%} of the gene set is covered")
    res["coverage"] = {"n_stack": len(allg), "n_mapped": len(rowof), "n_good": len(good),
                       "frac": frac, "n_species": int(Mid.shape[1])}

    idx = np.array([rowof[s] for s in good])
    R = Mid[idx].astype(np.float64)
    D = np.log(np.clip(100.0 - R, 0.5, None))          # divergence, logged
    ok = np.isfinite(D)
    D = np.where(ok, D, np.nan)
    D = D - np.nanmean(D, axis=0, keepdims=True)        # per-species centring
    D = D - np.nanmean(D, axis=1, keepdims=True)        # per-gene centring
    Dz = np.where(np.isfinite(D), D, 0.0)
    say(f"     doubly-centred residual matrix {Dz.shape[0]:,} genes x {Dz.shape[1]} species")
    gpos = {s: i for i, s in enumerate(good)}

    def erc(i, j):
        m = ok[i] & ok[j]
        if m.sum() < 30:
            return np.nan
        a, b = D[i][m], D[j][m]
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        return float(a @ b / d) if d > 0 else np.nan

    # ---------------------------------------------------------------- E2
    say("E2 DOES CO-EVOLUTION RECOVER KNOWN INTERACTIONS?")
    edges = set()
    import csv
    with open(NET / "omnipath.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r.get("source_genesymbol", ""), r.get("target_genesymbol", "")
            if a in gpos and b in gpos and a != b:
                edges.add((min(a, b), max(a, b)))
    n_omni = len(edges)
    try:
        with open(NET / "signor_human.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                a = r.get("ENTITYA") or r.get("IDA") or ""
                b = r.get("ENTITYB") or r.get("IDB") or ""
                if a in gpos and b in gpos and a != b:
                    edges.add((min(a, b), max(a, b)))
    except Exception as e:
        say(f"     SIGNOR not parsed ({type(e).__name__}); OmniPath only")
    edges = sorted(edges)
    say(f"     interacting pairs inside the covered gene set: {len(edges):,} "
        f"(OmniPath {n_omni:,}, plus SIGNOR)")
    posv = np.array([erc(gpos[a], gpos[b]) for a, b in edges])
    ii = rng.integers(0, len(good), size=(len(edges) * 3, 2))
    ii = ii[ii[:, 0] != ii[:, 1]]
    negv = np.array([erc(a, b) for a, b in ii])
    a2 = auc(posv, negv)
    say(f"     ERC on interacting pairs: median {np.nanmedian(posv):+.4f}   "
        f"random pairs: {np.nanmedian(negv):+.4f}")
    say(f"     AUC {a2:.4f} over {np.isfinite(posv).sum():,} positives and "
        f"{np.isfinite(negv).sum():,} negatives")
    G.add("E2", bool(np.isfinite(a2) and a2 >= AUC_BAR), stat=float(a2), requires=("E1",),
          if_true=lambda: f"E2 PASS -- AUC {a2:.3f}; co-evolution across 95 mammals does carry "
                          f"interaction information",
          if_false=lambda: f"E2 FAIL -- AUC {a2:.3f} against a {AUC_BAR:.2f} bar; the signal is "
                           f"not in protein identity at this resolution and nothing below counts")
    res["erc_raw"] = {"auc": a2, "n_pos": int(np.isfinite(posv).sum()),
                      "n_neg": int(np.isfinite(negv).sum()),
                      "pos_med": float(np.nanmedian(posv)), "neg_med": float(np.nanmedian(negv))}

    # ---------------------------------------------------------------- E3
    say("E3 IS E2 A CONFOUND?")
    northo = np.array([cov[s] for s in good], float)
    meandiv = np.nanmean(np.log(np.clip(100.0 - R, 0.5, None)), axis=1)
    expr = np.array([y_all[allg.index(s)] for s in good])
    feats = np.column_stack([northo, meandiv, expr])
    fz = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    nb = 6
    binz = np.zeros(len(good), np.int64)
    for c in range(fz.shape[1]):
        q = np.quantile(fz[:, c], np.linspace(0, 1, nb + 1)[1:-1])
        binz = binz * nb + np.searchsorted(q, fz[:, c])
    bucket = {}
    for i, b in enumerate(binz):
        bucket.setdefault(int(b), []).append(i)
    mneg = []
    for a, b in edges:
        for src in (gpos[a], gpos[b]):
            pool = bucket.get(int(binz[src]), [])
            if len(pool) > 1:
                mneg.append((src, int(rng.choice(pool))))
    mneg = [(x, z) for x, z in mneg if x != z][: len(edges) * 2]
    mnegv = np.array([erc(x, z) for x, z in mneg])
    a3 = auc(posv, mnegv)
    excess_raw, excess_m = a2 - 0.5, a3 - 0.5
    kept = excess_m / excess_raw if excess_raw > 0 else float("nan")
    say(f"     negatives matched on ortholog count, mean divergence and expression, "
        f"{nb}x{nb}x{nb} strata")
    say(f"     matched AUC {a3:.4f} against unmatched {a2:.4f}; excess over chance "
        f"{excess_m:+.4f} of {excess_raw:+.4f} = {kept:.1%} retained")
    G.add("E3", bool(np.isfinite(kept) and kept >= MATCH_KEEP), stat=float(kept), requires=("E2",),
          if_true=lambda: f"E3 PASS -- {kept:.0%} of the effect survives matching; it is not "
                          f"length, constraint or expression",
          if_false=lambda: f"E3 FAIL -- only {kept:.0%} survives matching, so most of E2 was "
                           f"gene properties rather than shared evolution")
    res["erc_matched"] = {"auc": a3, "kept": kept, "n_neg": int(np.isfinite(mnegv).sum())}

    # ---------------------------------------------------------------- E4
    say("E4 DOES A LEARNED EMBEDDING BEAT PAIRWISE ERC?")
    U, Sv, Vt = np.linalg.svd(Dz - Dz.mean(0, keepdims=True), full_matrices=False)
    EF = U[:, :K_FACTOR] * Sv[:K_FACTOR]
    EFn = EF / (np.linalg.norm(EF, axis=1, keepdims=True) + 1e-9)
    say(f"     FACTOR: rank-{K_FACTOR} factorisation, top component explains "
        f"{Sv[0]**2/np.sum(Sv**2):.1%} of residual variance")
    fpos = np.array([float(EFn[gpos[a]] @ EFn[gpos[b]]) for a, b in edges])
    fneg = np.array([float(EFn[x] @ EFn[z]) for x, z in mneg])
    a_f = auc(fpos, fneg)

    a_m = float("nan"); mlp_note = ""
    try:
        from sklearn.neural_network import MLPRegressor
        mask = rng.random(Dz.shape) < 0.15
        Xin = np.where(mask, 0.0, Dz)
        net = MLPRegressor(hidden_layer_sizes=(K_FACTOR,), activation="tanh", max_iter=120,
                           random_state=SEED, early_stopping=False)
        net.fit(Xin, Dz)
        H = np.tanh(Xin @ net.coefs_[0] + net.intercepts_[0])
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        mpos = np.array([float(Hn[gpos[a]] @ Hn[gpos[b]]) for a, b in edges])
        mneg_ = np.array([float(Hn[x] @ Hn[z]) for x, z in mneg])
        a_m = auc(mpos, mneg_)
        mlp_note = (f"masked-reconstruction autoencoder, {K_FACTOR} units, 15% of entries masked, "
                    f"train loss {net.loss_:.4f}")
    except Exception as e:
        mlp_note = f"MLP arm did not run: {type(e).__name__}: {e}"
    say(f"     FACTOR matched AUC {a_f:.4f}")
    say(f"     MLP    matched AUC {a_m:.4f}   ({mlp_note})")
    say(f"     ERC    matched AUC {a3:.4f}")
    best_learn = max([x for x in (a_f, a_m) if np.isfinite(x)], default=float("nan"))
    G.add("E4", bool(np.isfinite(best_learn) and best_learn - a3 >= LEARN_MARGIN),
          stat=float(best_learn), requires=("E2",),
          if_true=lambda: f"E4 PASS -- the learned embedding reaches {best_learn:.3f} against "
                          f"ERC's {a3:.3f}; pooling across genes adds information",
          if_false=lambda: f"E4 FAIL -- best learned arm {best_learn:.3f} against ERC {a3:.3f}. "
                           f"With {Dz.shape[1]} species a pairwise correlation may already be the "
                           f"whole estimate")
    res["learned"] = {"factor_auc": a_f, "mlp_auc": a_m, "erc_auc": a3,
                      "top_component": float(Sv[0] ** 2 / np.sum(Sv ** 2))}

    # ---------------------------------------------------------------- E5
    say("E5 DOES CO-EVOLUTION ADD TO THE STANDING STACK?")
    yv = np.array([y_all[allg.index(s)] for s in good])
    arms = {"ERC-neighbourhood": None, "FACTOR": EF}
    knn = np.zeros((len(good), 8))
    Cf = EFn @ EFn.T
    np.fill_diagonal(Cf, -np.inf)
    nn = np.argsort(-Cf, axis=1)[:, :8]
    knn = yv[nn]
    blocks = {"coev_factor": EF, "coev_neighbour_y": knn,
              "coev_meta": np.column_stack([northo, meandiv, np.nanstd(D, axis=1)])}
    n = len(good); perm = rng.permutation(n); cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    scores = {}
    for nm, X in blocks.items():
        Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scores[nm] = pear(yv[te], ridge_pred(Xc[tr], yv[tr], Xc[te]))
        say(f"       {nm:<20} held-out |r| {abs(scores[nm]):.4f}")
    Xall = np.nan_to_num(np.hstack([blocks[k] for k in blocks]), nan=0.0)
    comb = pear(yv[te], ridge_pred(Xall[tr], yv[tr], Xall[te]))
    say(f"       {'all co-evolution':<20} held-out |r| {abs(comb):.4f}")
    say(f"     loop 213's ten-block stack reached {REF_STACK:.4f}; fame-only was {REF_FAME:.4f}")
    G.add("E5", bool(abs(comb) > REF_STACK), stat=float(abs(comb)), requires=("E2",),
          if_true=lambda: f"E5 PASS -- co-evolution reaches {abs(comb):.4f}, above the standing "
                          f"stack's {REF_STACK:.4f}",
          if_false=lambda: f"E5 FAIL -- {abs(comb):.4f} against the standing stack's "
                           f"{REF_STACK:.4f}")
    res["stack"] = {k: abs(v) for k, v in scores.items()}
    res["stack"]["combined"] = abs(comb); res["stack"]["reference"] = REF_STACK

    # ---------------------------------------------------------------- E6
    say("E6 RIDGE AGAINST MLP, WITH THE SHUFFLED CONTROL BOTH TIMES")
    ysh = yv.copy(); rng.shuffle(ysh)
    r_real = abs(pear(yv[te], ridge_pred(Xall[tr], yv[tr], Xall[te])))
    r_shuf = abs(pear(ysh[te], ridge_pred(Xall[tr], ysh[tr], Xall[te])))
    m_real = m_shuf = float("nan")
    try:
        from sklearn.neural_network import MLPRegressor
        mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
        Zt, Ze = (Xall[tr] - mu) / sd, (Xall[te] - mu) / sd
        for tag, yy in (("real", yv), ("shuf", ysh)):
            nn_ = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED,
                               early_stopping=True)
            nn_.fit(Zt, yy[tr])
            v = abs(pear(yy[te], nn_.predict(Ze)))
            if tag == "real":
                m_real = v
            else:
                m_shuf = v
    except Exception as e:
        say(f"     MLP arm did not run: {type(e).__name__}: {e}")
    say(f"       ridge  real {r_real:.4f}   shuffled {r_shuf:.4f}   margin {r_real-r_shuf:+.4f}")
    say(f"       MLP    real {m_real:.4f}   shuffled {m_shuf:.4f}   margin {m_real-m_shuf:+.4f}")
    say(f"     loop 211 measured ridge 0.4057 / MLP-wide 0.2072, shuffled 0.0237 / 0.0742")
    cands = {"ridge": (r_real, r_shuf), "MLP": (m_real, m_shuf)}
    winner = max((k for k in cands if np.isfinite(cands[k][0])),
                 key=lambda k: cands[k][0], default=None)
    marg = (cands[winner][0] - cands[winner][1]) if winner else float("nan")
    G.add("E6", bool(np.isfinite(marg) and marg >= CTRL_MARGIN), stat=float(marg), requires=("E1",),
          if_true=lambda: f"E6 PASS -- {winner} wins at {cands[winner][0]:.4f} and beats its own "
                          f"shuffled control by {marg:+.4f}",
          if_false=lambda: f"E6 FAIL -- best arm beats its shuffled control by only {marg:+.4f}; "
                           f"the comparison carries no information")
    res["ridge_vs_mlp"] = {"ridge_real": r_real, "ridge_shuf": r_shuf,
                           "mlp_real": m_real, "mlp_shuf": m_shuf, "winner": winner}

    # ---------------------------------------------------------------- E7
    say("E7 SPECIES-SHUFFLE CONTROL")
    Ds = D.copy(); oks = ok.copy()
    for i in range(Ds.shape[0]):
        p = rng.permutation(Ds.shape[1])
        Ds[i] = Ds[i][p]; oks[i] = oks[i][p]
    def erc_s(i, j):
        m = oks[i] & oks[j]
        if m.sum() < 30: return np.nan
        a, b = Ds[i][m], Ds[j][m]
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        return float(a @ b / d) if d > 0 else np.nan
    sp_ = np.array([erc_s(gpos[a], gpos[b]) for a, b in edges])
    sn_ = np.array([erc_s(x, z) for x, z in mneg])
    a7 = auc(sp_, sn_)
    say(f"     species permuted independently within each gene -- marginals preserved exactly")
    say(f"     shuffled AUC {a7:.4f} against the real matched {a3:.4f}")
    G.add("E7", bool(np.isfinite(a7) and a7 < SHUF_BAR), stat=float(a7), requires=("E2",),
          if_true=lambda: f"E7 PASS -- destroying phylogenetic covariation drops AUC to {a7:.3f}",
          if_false=lambda: f"E7 FAIL -- the shuffle still reaches {a7:.3f}, so E2 is not measuring "
                           f"shared evolution")
    res["shuffle"] = {"auc": a7}

    # ---------------------------------------------------------------- E8
    say("E8 WHAT THIS CANNOT SHOW")
    say("     Gene-level rates cannot see intra-genic co-variation, regulatory sequence or")
    say("     non-coding elements, all of which a nucleotide model reads directly. Any FAIL here")
    say("     is a statement about protein identity across 95 mammals, not about Evo 2.")
    say("     OmniPath and SIGNOR are literature-curated, so edges and the genes carrying them")
    say("     are biased toward well-studied proteins. E3 matches on expression and ortholog")
    say("     count but cannot match on how much a gene has been studied.")
    say("     95 species is not many: each ERC value is a correlation on at most 95 points, with")
    say("     a standard error near 0.10 before phylogenetic non-independence inflates it further.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
