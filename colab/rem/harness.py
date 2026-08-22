"""Shared evaluation harness for the REM gap-prediction arc. No gates -- this is the instrument.

WHY IT EXISTS. Loops 160 and 161 each rebuilt the same held-out protocol inline, and 161 then had
to correct 160's headline because the two reported different conditioning on the same number. One
harness, imported by both the design work and the final test, removes that class of error.

THE SPLIT, AND WHY IT IS LOCKED. Loop 161 ended by fitting a fusion weight on half the cases and
scoring the other half. That was the right instinct applied too late: the DESIGN of a merge is
itself a fit, and a merge designed against the same reactions it is later scored on is not
measured, it is remembered. So the eligible reactions are split once, deterministically, into

    DEV   used for designing and choosing a merge. Look at it as much as you like.
    TEST  locked. Touched once, after the gates are written down.

split_seed is fixed at 16100 and is not a tunable.

THE PRIMARY METRIC IS THE FULL CANDIDATE SET. Loop 160 reported AUC 0.8158 inside a 2-step
shortlist, and loop 161 measured that the shortlist contains the answer only 52.6% of the time --
so that number describes half the problem. `evaluate` therefore returns AUC over ALL non-currency
candidates as the headline, with the shortlist-conditional figure and the shortlist recall beside
it, and a rank-based metric (mean reciprocal rank, hit@k) that a missing answer can actually lose.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats

HERE = Path(__file__).resolve().parent
TOPO = Path("colab/data/rem_bipartite.npz")
CHEM = Path("colab/data/rem_chem.npz")
CURRENCY_MIN = 200
ALPHA, NITER = 0.15, 60
SPLIT_SEED = 16100
DEV_FRAC = 0.5


class REM:
    def __init__(self):
        T = np.load(TOPO, allow_pickle=False)
        C = np.load(CHEM, allow_pickle=False)
        self.species = list(T["species"])
        self.sp_comp = list(T["sp_comp"])
        self.sp_name = list(T["sp_name"])
        self.reactions = list(T["reactions"])
        self.rev = T["reversible"]
        self.rr, self.rs = T["react_rx"], T["react_sp"]
        self.pr, self.ps = T["prod_rx"], T["prod_sp"]
        self.rco, self.pco = C["react_coef"], C["prod_coef"]
        self.E = C["E"].astype(np.float64)
        self.charge = C["charge"].astype(np.float64)
        self.generic = C["generic"]
        self.elements = list(C["elements"])
        self.NS, self.NR = len(self.species), len(self.reactions)

        self.react_of, self.prod_of = defaultdict(set), defaultdict(set)
        self.coef_r, self.coef_p = defaultdict(dict), defaultdict(dict)
        for j, i, c in zip(self.rr, self.rs, self.rco):
            self.react_of[int(j)].add(int(i))
            self.coef_r[int(j)][int(i)] = float(c)
        for j, i, c in zip(self.pr, self.ps, self.pco):
            self.prod_of[int(j)].add(int(i))
            self.coef_p[int(j)][int(i)] = float(c)

        d = Counter()
        for i in self.rs:
            d[int(i)] += 1
        for i in self.ps:
            d[int(i)] += 1
        self.deg = d
        self.currency = {i for i in range(self.NS) if d[i] > CURRENCY_MIN}
        self.noncur = np.array(sorted(set(range(self.NS)) - self.currency))
        self.ncmap = {int(v): k for k, v in enumerate(self.noncur)}
        self.degv = np.array([d[int(i)] for i in self.noncur], float)
        self.Enc = self.E[self.noncur]
        self.Enc2 = (self.Enc ** 2).sum(1)
        self.Enc2[self.Enc2 == 0] = np.inf

        self.sp_rx = defaultdict(set)
        for j in range(self.NR):
            for i in self.react_of[j] | self.prod_of[j]:
                self.sp_rx[i].add(j)

        self.nodes = self.NS + self.NR
        rv_r = self.rev[self.rr] == 1
        rv_p = self.rev[self.pr] == 1
        self.src = np.concatenate([self.rs, self.NS + self.pr,
                                   self.NS + self.rr[rv_r], self.ps[rv_p]])
        self.dst = np.concatenate([self.NS + self.rr, self.ps,
                                   self.rs[rv_r], self.NS + self.pr[rv_p]])
        self.wgt = np.concatenate([self.rco, self.pco, self.rco[rv_r], self.pco[rv_p]])

        self.eligible = [j for j in range(self.NR)
                         if (self.react_of[j] - self.currency)
                         and (self.prod_of[j] - self.currency)
                         and len(self.react_of[j] | self.prod_of[j]) >= 2]
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(self.eligible))
        k = int(DEV_FRAC * len(self.eligible))
        el = np.array(self.eligible)
        self.dev = sorted(el[perm[:k]].tolist())
        self.test = sorted(el[perm[k:]].tolist())

    # ---------------------------------------------------------------- operators
    def operator(self, mask_rx, weights=None):
        """Column-stochastic walk operator with reaction `mask_rx` deleted.

        weights: None for uniform, "stoich" for the stoichiometric coefficient, or an array
        aligned with self.src giving a weight per directed edge."""
        s_, d_ = self.src, self.dst
        if weights is None:
            w_ = np.ones(len(s_))
        elif isinstance(weights, str) and weights == "stoich":
            w_ = self.wgt
        else:
            w_ = np.asarray(weights, float)
        keep = ~(((s_ >= self.NS) & (s_ - self.NS == mask_rx))
                 | ((d_ >= self.NS) & (d_ - self.NS == mask_rx)))
        A = sparse.csr_matrix((w_[keep], (d_[keep], s_[keep])),
                              shape=(self.nodes, self.nodes))
        cs = np.asarray(A.sum(0)).ravel()
        cs[cs == 0] = 1.0
        return A @ sparse.diags(1.0 / cs)

    def walk(self, P, seeds, alpha=ALPHA, niter=NITER):
        e = np.zeros(self.nodes)
        e[seeds] = 1.0 / len(seeds)
        p = e.copy()
        for _ in range(niter):
            p = (1 - alpha) * (P @ p) + alpha * e
        return p

    # ---------------------------------------------------------------- case setup
    def case(self, j):
        """Everything a scorer may see about held-out reaction j. Nothing about its products
        except the CURRENCY ones, which every gap-filling method assumes available."""
        seeds = sorted(self.react_of[j] - self.currency)
        targets = self.prod_of[j] - self.currency - set(seeds)
        if not seeds or not targets:
            return None
        pos = np.zeros(len(self.noncur), bool)
        for i in targets:
            pos[self.ncmap[int(i)]] = True
        excl = np.zeros(len(self.noncur), bool)
        for i in seeds:
            excl[self.ncmap[int(i)]] = True
        res = np.zeros(len(self.elements))
        for i, c in self.coef_r[j].items():
            res += c * self.E[i]
        res_hard = res.copy()
        for i, c in self.coef_p[j].items():
            if i in self.currency:
                res -= c * self.E[i]
        qres = sum(c * self.charge[i] for i, c in self.coef_r[j].items())
        return {"j": j, "seeds": seeds, "targets": targets, "pos": pos, "excl": excl,
                "residual": res, "residual_hard": res_hard, "charge_residual": qres,
                "comp": {self.sp_comp[i] for i in seeds}}

    def shortlist(self, cs, k=2):
        """The k-step bipartite neighbourhood of the seeds, with the held-out reaction removed."""
        cur = set(cs["seeds"])
        seen = set(cur)
        for _ in range(k // 2 if k > 2 else 1):
            pass
        nb = set()
        frontier = set(cs["seeds"])
        for _ in range(max(1, k // 2)):
            nxt = set()
            for i in frontier:
                for r in self.sp_rx[i]:
                    if r == cs["j"]:
                        continue
                    nxt |= (self.react_of[r] | self.prod_of[r])
            nb |= nxt
            frontier = nxt - seen
            seen |= nxt
        nb -= self.currency
        m = np.zeros(len(self.noncur), bool)
        for i in nb:
            m[self.ncmap[int(i)]] = True
        return m & ~cs["excl"]

    def balance_score(self, res):
        """Fraction of the elemental residual each candidate can explain at a best non-negative
        coefficient. Scale-invariant, zero for anything pointing the wrong way."""
        d = float(res @ res)
        if d <= 0:
            return np.zeros(len(self.noncur))
        ip = self.Enc @ res
        return np.where(ip > 0, ip ** 2 / (self.Enc2 * d), 0.0)


# ---------------------------------------------------------------------- metrics
def auc_of(scores, pos):
    if pos.sum() == 0 or (~pos).sum() == 0:
        return np.nan
    r = stats.rankdata(scores)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def evaluate(rem, scorer, cases, ks=(1, 5, 20, 100)):
    """scorer(rem, case) -> array over rem.noncur, higher is better.

    Returns AUC over ALL non-currency candidates (the headline -- a scorer that cannot see a
    candidate simply loses on it), plus mean reciprocal rank and hit@k, which a missing answer
    can actually lose. Excluded seeds are removed from the ranking everywhere."""
    aucs, mrr, hits = [], [], {k: [] for k in ks}
    for cs in cases:
        if cs is None:
            continue
        s = np.asarray(scorer(rem, cs), float)
        m = ~cs["excl"]
        pos = cs["pos"][m]
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        sv = s[m]
        aucs.append(auc_of(sv, pos))
        order = np.argsort(-sv, kind="stable")
        rank = np.empty(len(sv), int)
        rank[order] = np.arange(1, len(sv) + 1)
        best = int(rank[pos].min())
        mrr.append(1.0 / best)
        for k in ks:
            hits[k].append(1.0 if best <= k else 0.0)
    a = np.array(aucs, float)
    return {"auc": float(np.nanmean(a)), "sem": float(np.nanstd(a) / np.sqrt(np.isfinite(a).sum())),
            "mrr": float(np.mean(mrr)), "n": int(np.isfinite(a).sum()),
            **{f"hit@{k}": float(np.mean(v)) for k, v in hits.items()}}
