"""BIOBRAIN -- the brain's neuro engine (Nikku03/money, branch claude/practical-volta-mbc1b5, brain/neuro, docs/NEURO.md)
pointed at biology itself: can the memory bank and the evidence engine find out how the cell works inside, and is
what they record enough to build its responses again?

THE SAME RULES AS FOR A NETWORK (docs/NEURO.md), APPLIED TO A CELL
  network                                    cell
  entity NN-<sha of weights>                 a knockdown screen: the cell's response to each gene's loss, as
                                             records in the memory bank (rem/atlas/_cache/cell_v8; 128 components)
  probes write facts                         `look` writes facts: each complex's response programme (citing the
                                             records it averages), each subject's top programmes, its strength scale
  dev network (seed 0) for looking           DEV: K562 genome-wide screen A, TRAIN genes only
  ten fresh seeds for the verdict            ten screens nothing looked at: K562 A (TEST genes), K562-essential B,
                                             RPE1, HepG2, Jurkat (Replogle 2022 / Nadig 2025, CC BY), HCT116,
                                             HEK293T, iPSC-N, iPSC-Fsf, hESC-P (reprocessed, CC BY-NC-SA; local)
  'learnt the task'                          'qualifies': >= 100 genes in the analysis set with a complex mate
  null keeps all but the structure           gene <-> complex membership permuted among genes of the same record
                                             strength decile (200 draws): keeps every response and the strength
                                             confound (strong knockdowns share a generic stress response), removes
                                             only which gene is in which complex
  positive control: hand-built mechanism     `planted`: synthetic responses, one random programme per complex
  negative control: no mechanism             `unstructured`: Gaussian responses with the dev subject's covariance
  interventions / rebuild / transfer         a gene's response REPLACED by its complex's programme recorded from
                                             other genes (rebuild); the rule with K562 facts alone, in other cell
                                             types (transfer)
THE ANALYSIS SET, per subject: records with strength (norm) at or above the subject's median (weak records are
mostly noise), as unit vectors after removing the subject's own 3 largest programmes (top principal directions:
the generic responses shared by most strong knockdowns), re-normalised: 'specific responses'. Complexes: Complex
Portal human complexes (EBI, outputs/orphan/science_data, 2026-01-14), members mapped Ensembl -> symbol (HGNC).
Pathways: Reactome (ReactomePathways.gmt), pathways of 5-200 genes.

THE CLAIMS (B1-*; known answers: Replogle et al. 2022 Cell, genes of one complex give one transcriptional phenotype;
Nadig et al. 2025, core complexes' responses are conserved across cell types)
  alike        two subunits of the same complex respond more alike than two genes of different complexes:
               AUROC of the cosine, same-complex pairs vs other pairs of complex genes, >= threshold; beats null.
  rebuild      a gene's response rebuilt from the bank's facts alone (its complexes' programmes, recorded from the
               OTHER half of the subject's genes, split by sha256('bio-fold:'+gene)): median cosine with the
               measured response >= threshold; beats null. The gene's own measurement is never used.
  transfer     the rule with K562 facts alone: programmes recorded on the DEV subject (K562 A TRAIN genes; a gene's
               own K562 record left out of its programmes) rebuild each subject's responses: median cosine >=
               threshold; beats null.
  (pathway: dropped at the dev stage, before declaring. 'Two genes of one Reactome pathway that share no complex
   respond alike at the alike threshold' was meant as a declared-to-fail claim, but its positive control cannot
   pass even with no noise (a planted pathway mechanism reaches AUROC 0.734 < 0.75: genes sit in 4.5 pathways
   each, so 'shares a pathway' is too broad a label for this measure). An untestable claim is not declared;
   the pathway AUROC and its null p are reported per subject, descriptively, with no verdict.)
PER SUBJECT: measure >= threshold and null p < 0.05. PASS: >= 9 of 10 subjects qualify (else NO VERDICT), >= 90% of
those pass, positive control passes >= 90% of 10 runs, negative fails >= 90% of 10 runs, BH q over every claim in
rem/atlas/bio_ledger.jsonl < 0.10; a claim's p is the chance of that many passing subjects at one half each.
Thresholds were set on the DEV subject alone (`look`), recorded in the declaration with the dev values.
NOT: one complex resource; a response compressed to K562's 128 components (other cell types mapped onto K562's
genes); the reprocessed screens differ in pipeline; record strength stands in for 'has a phenotype'.

Run: python3 -P rem/atlas/biobrain.py look | declare | test     (declare, then commit, then test)
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("cellbank4", os.path.join(HERE, "cellbank4.py"))
cb4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb4)

BANK = os.path.join(HERE, "_cache", "cell_v8")
LEDGER = os.path.join(HERE, "bio_ledger.jsonl")
FACTS = os.path.join(HERE, "_cache", "biobrain", "facts.jsonl")
OUT = os.path.join(HERE, "RESULTS_biobrain.txt")
VERSION = "B1"
DEV = ("K562-A", "TRAIN")
SUBJECTS = [("K562-A", "A", "TEST"), ("K562-ess", "B", None), ("RPE1", "C", None), ("HepG2", "D", None), ("Jurkat", "E", None),
            ("HCT116", "S:HCT116", None), ("HEK293T", "S:HEK293T", None), ("iPSC-N", "S:iPSC-N", None),
            ("iPSC-Fsf", "S:iPSC-Fsf", None), ("hESC-P", "S:hESC-P", None)]
N_NULL, N_CTRL, N_PC, QUALIFY = 200, 10, 3, 100
PLANT_SIGNAL = 3.0   # the planted mechanism's programme size against unit noise (set on dev so planted passes cleanly)
GATE = {"min_qualify": 9, "pass_share": 0.9, "null_p_lt": 0.05, "bh_q": 0.10, "ctrl_share": 0.9}
# Thresholds set on the DEV subject alone (look, 2026-11-14): alike AUROC 0.913, rebuild 0.645 (pathway 0.561),
# transfer not measurable on dev (dev on itself, own record left out: 0.691). Margins as in docs/NEURO.md
# (0.8 for a dev 0.91). Transfer: set a priori at about half the within-cell rebuild, before any subject was scored.
CLAIMS = [
    {"id": "alike", "sets": "complex", "threshold": 0.75, "dev": 0.913, "expect": "pass",
     "claim": "Two subunits of the same protein complex respond to knockdown more alike than two genes of different "
              "complexes: AUROC of the specific-response cosine, same-complex pairs vs other pairs of complex genes, "
              ">= 0.75, beating the strength-matched membership null."},
    {"id": "rebuild", "sets": "complex", "threshold": 0.40, "dev": 0.645, "expect": "pass",
     "claim": "A gene's response rebuilt from the bank's facts alone (its complexes' programmes recorded from the other "
              "half of the subject's genes; its own measurement never used) has median cosine >= 0.40 with the "
              "measured response, beating the null."},
    {"id": "transfer", "sets": "complex", "threshold": 0.30, "dev": None, "expect": "pass",
     "claim": "The rule with K562 facts alone: complex programmes recorded on the DEV subject (K562 A TRAIN genes; a "
              "gene's own K562 record left out) rebuild each subject's responses with median cosine >= 0.30, beating "
              "the null."},
]


# ---------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------
def annotations(bank):
    import pandas as pd
    hg = pd.read_csv(os.path.join(HERE, "hgnc_complete_set.txt"), sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"])
    ens2sym = dict(zip(hg.ensembl_gene_id, hg.symbol))
    known = set(bank.by_gene)
    cm = pd.read_parquet(R("outputs", "orphan", "science_data", "complex_members.parquet"))
    comp = {}
    for cid, e in zip(cm.complex_id, cm.ensembl_gene_id):
        s = ens2sym.get(e)
        if s in known:
            comp.setdefault(cid, set()).add(s)
    comp = {c: m for c, m in comp.items() if len(m) >= 2}
    path = {}
    with zipfile.ZipFile(os.path.join(HERE, "_cache", "cellbank4", "data", "ReactomePathways.gmt.zip")) as z:
        for line in z.read(z.namelist()[0]).decode().splitlines():
            f = line.split("\t")
            g = {x for x in f[2:] if x in known}
            if 5 <= len(f) - 2 <= 200 and len(g) >= 2:
                path[f[1]] = g
    return comp, path


def subject(bank, screen, split=None):
    """Genes and raw 128-d responses of one screen (optionally one split)."""
    genes, ids = [], []
    for g, d in bank.by_gene.items():
        i = d.get(screen)
        if i is not None and (split is None or bank.split[i] == split):
            genes.append(g); ids.append(i)
    return genes, bank.z(ids).astype(np.float64)


def analysis(genes, Z):
    """Top half by strength; subject's own top programmes removed; unit length."""
    s = np.linalg.norm(Z, axis=1); keep = s >= np.median(s)
    G = [g for g, k in zip(genes, keep) if k]; X = Z[keep]; st = s[keep]
    U = X / st[:, None]
    _, _, Vt = np.linalg.svd(U - U.mean(0), full_matrices=False); P = Vt[:N_PC]
    Xs = U - (U @ P.T) @ P
    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
    return G, Xs, st, P


def incidence(G, sets):
    names = sorted(sets); col = {c: k for k, c in enumerate(names)}; pos = {g: i for i, g in enumerate(G)}
    M = np.zeros((len(G), len(names)), bool)
    for c, m in sets.items():
        for g in m:
            if g in pos:
                M[pos[g], col[c]] = True
    keep = M.sum(0) >= 2
    return M[:, keep], [n for n, k in zip(names, keep) if k]


def deciles(st, rng):
    """A permutation of genes within strength deciles."""
    q = np.quantile(st, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, st, side="right") - 1, 0, 9)
    perm = np.arange(len(st))
    for k in range(10):
        idx = np.nonzero(b == k)[0]; perm[idx] = rng.permutation(idx)
    return perm


# ---------------------------------------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------------------------------------
def _auroc(sc, lab):
    """Mann-Whitney AUROC of scores sc for lab True vs False (ranks taken within the given pairs)."""
    n1 = int(lab.sum()); n0 = len(lab) - n1
    if not n1 or not n0:
        return float("nan")
    rk = sc.argsort().argsort() + 1.0
    return float((rk[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def m_alike(Xs, st, M, rng, null=True, excl=None):
    """AUROC: same-set pairs vs other pairs among genes in >= 1 set. excl: pairs sharing an excl set are dropped."""
    idx = np.nonzero(M.any(1))[0]
    iu = np.triu_indices(len(idx), 1); sc = (Xs[idx] @ Xs[idx].T)[iu]
    Mi = M[idx].astype(np.float32)
    lab = (Mi @ Mi.T)[iu] > 0
    E = excl[idx].astype(np.float32) if excl is not None else None
    keep = ~((E @ E.T)[iu] > 0) if E is not None else np.ones(len(lab), bool)
    obs = _auroc(sc[keep], lab[keep]); out = dict(stat=obs, genes=int(len(idx)), pos_pairs=int((lab & keep).sum()))
    if null:
        sub = st[idx]; ge = 0
        for _ in range(N_NULL):
            p = deciles(sub, rng); Mp = Mi[p]
            lp = (Mp @ Mp.T)[iu] > 0
            kp = ~((E[p] @ E[p].T)[iu] > 0) if E is not None else keep
            ge += _auroc(sc[kp], lp[kp]) >= obs
        out["p"] = (1 + ge) / (1 + N_NULL)
    return out


def programmes(Xs, M, rows=None, leave_out=None):
    """Each set's programme: the mean of its members' specific responses (only `rows` used)."""
    W = M.astype(np.float64) if rows is None else M.astype(np.float64) * rows[:, None]
    S = W.T @ Xs; n = W.sum(0)
    return S, n


def _rebuild_cos(Xs, Mg, S, n, own=None):
    """For each gene: cosine of its own response with the sum of its sets' programmes (own contribution removed)."""
    S2 = Mg.astype(np.float64) @ S
    if own is not None:
        S2 = S2 - own
    v = (S2 * Xs).sum(1) / (np.linalg.norm(S2, axis=1) + 1e-12)
    ok = np.linalg.norm(S2, axis=1) > 1e-3        # an empty programme (only the gene itself) is not a rebuild
    return v, ok


def m_rebuild(G, Xs, st, M, rng, null=True):
    half = np.array([int(hashlib.sha256(("bio-fold:" + g).encode()).hexdigest(), 16) % 2 for g in G])
    def stat(Mg):
        vals = np.full(len(G), np.nan)
        for h in (0, 1):
            S, n = programmes(Xs, Mg, rows=(half != h))        # facts from the other half only
            v, ok = _rebuild_cos(Xs, Mg, S, n)
            sel = (half == h) & ok
            vals[sel] = v[sel]
        return vals
    v = stat(M); obs = float(np.nanmedian(v)); out = dict(stat=obs, genes=int(np.isfinite(v).sum()))
    if null:
        ge = 0
        for _ in range(N_NULL):
            ge += np.nanmedian(stat(M[deciles(st, rng)])) >= obs
        out["p"] = (1 + ge) / (1 + N_NULL)
    return out


def m_transfer(dev, G, Xs, st, sets, rng, null=True, facts=None):
    """Programmes recorded on the dev subject, applied to this subject's genes; a gene's own dev record left out.
    With `facts`, the programmes and their member lists are read from the bank's fact records only."""
    Gd, Xd = dev["G"], dev["Xs"]
    pos_d = {g: i for i, g in enumerate(Gd)}; pos_s = {g: i for i, g in enumerate(G)}
    if facts is not None:
        names = [f["attribute"].split("/")[1] for f in facts]
        S = np.array([f["value"] for f in facts], dtype=np.float64)
        dev_members = [set(f["members"]) for f in facts]
        all_members = [sets[c] for c in names]
    else:
        names = [c for c in sorted(sets) if any(g in pos_d for g in sets[c])]
        dev_members = [{g for g in sets[c] if g in pos_d} for c in names]
        all_members = [sets[c] for c in names]
        S = np.array([Xd[[pos_d[g] for g in m]].sum(0) for m in dev_members])
    Md = np.zeros((len(Gd), len(names)), bool); Ms = np.zeros((len(G), len(names)), bool)
    for k in range(len(names)):
        for g in dev_members[k]:
            Md[pos_d[g], k] = True
        for g in all_members[k]:
            if g in pos_s:
                Ms[pos_s[g], k] = True
    n = Md.sum(0)
    jd = np.array([pos_d.get(g, -1) for g in G])

    def own_part(Mg):
        """The gene's own dev record, once for each of its (assigned) complexes it is a dev member of: removed."""
        own = np.zeros((len(G), Xs.shape[1]))
        for i in np.nonzero(jd >= 0)[0]:
            k = float((Mg[i] & Md[jd[i]]).sum())
            if k:
                own[i] = Xd[jd[i]] * k
        return own

    def stat(Mg):
        v, ok = _rebuild_cos(Xs, Mg, S, n, own=own_part(Mg))
        return v[ok]
    v = stat(Ms); obs = float(np.median(v)); out = dict(stat=obs, genes=int(len(v)))
    if null:
        ge = 0
        for _ in range(N_NULL):
            ge += np.median(stat(Ms[deciles(st, rng)])) >= obs
        out["p"] = (1 + ge) / (1 + N_NULL)
    return out


# ---------------------------------------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------------------------------------
def planted(G, M, seed, noise_seed=None, signal=PLANT_SIGNAL):
    """One random programme per set; a gene's response = its sets' programmes + noise, random overall size."""
    rng = np.random.default_rng(1000 + seed)
    Pc = rng.normal(size=(M.shape[1], 128)); Pc /= np.linalg.norm(Pc, axis=1, keepdims=True)
    base = M.astype(np.float64) @ Pc
    nb = np.linalg.norm(base, axis=1, keepdims=True); base = np.where(nb > 0, base / np.maximum(nb, 1e-12), 0)
    rn = np.random.default_rng(3000 + (seed if noise_seed is None else noise_seed))
    Z = signal * base + rn.normal(size=(len(G), 128)) / np.sqrt(128)     # unit-size noise
    Z *= rn.lognormal(0, 0.5, size=(len(G), 1))
    return Z


def unstructured(Zdev, n, seed):
    rng = np.random.default_rng(2000 + seed)
    mu = Zdev.mean(0); C = np.cov(Zdev.T)
    return rng.multivariate_normal(mu, C, size=n)


# ---------------------------------------------------------------------------------------------------------
# look (dev only)
# ---------------------------------------------------------------------------------------------------------
def load():
    bank = cb4.Bank(BANK)
    comp, path = annotations(bank)
    return bank, comp, path


def dev_subject(bank):
    genes, Z = subject(bank, "A", "TRAIN")
    G, Xs, st, P = analysis(genes, Z)
    return dict(G=G, Xs=Xs, st=st, P=P, Zraw=Z, genes_all=genes)


def look():
    t0 = time.time(); rng = np.random.default_rng(0)
    bank, comp, path = load(); dev = dev_subject(bank)
    G, Xs, st = dev["G"], dev["Xs"], dev["st"]
    Mc, cn = incidence(G, comp); Mp, pn = incidence(G, path)
    print(f"DEV {DEV}: {len(dev['genes_all'])} records, analysis set {len(G)}; complexes with >= 2 members here {len(cn)}, "
          f"genes in one {int(Mc.any(1).sum())}; pathways {len(pn)}, genes in one {int(Mp.any(1).sum())}")
    a = m_alike(Xs, st, Mc, rng); print(f"  alike    AUROC {a['stat']:.3f}  p {a['p']:.3f}  genes {a['genes']}  same-complex pairs {a['pos_pairs']}")
    r = m_rebuild(G, Xs, st, Mc, rng); print(f"  rebuild  median cosine {r['stat']:.3f}  p {r['p']:.3f}  genes {r['genes']}")
    pw = m_alike(Xs, st, Mp, rng, excl=Mc); print(f"  pathway  AUROC {pw['stat']:.3f}  p {pw['p']:.3f}  genes {pw['genes']}  pathway-only pairs {pw['pos_pairs']}")
    # transfer on dev itself is not a test (facts and genes are the same); leave-one-out version, for scale only
    rt = m_transfer(dev, G, Xs, st, comp, rng, null=False); print(f"  transfer (dev on itself, own record left out) median cosine {rt['stat']:.3f}")
    for c in CLAIMS:
        for name in ("positive", "negative"):
            r_ = control(c, name, dev, comp, path, 0, rng)
            print(f"  control {c['id']:9s} {name:8s} stat {r_['stat']:.3f} p {r_['p']:.3f} -> {'passes' if r_['pass'] else 'fails'}")
    for nm, scr, sp in SUBJECTS:
        genes, Z = subject(bank, scr, sp); Gs, Xss, sts, _ = analysis(genes, Z)
        Ms_, _ = incidence(Gs, comp)
        print(f"  subject {nm:9s} records {len(genes):5d}, analysis set {len(Gs):5d}, genes with a complex mate {int(Ms_.any(1).sum()):4d}  (counts only)")
    print(f"look done ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------
# measuring a claim on a subject, controls, facts
# ---------------------------------------------------------------------------------------------------------
def _sets(c, comp, path):
    return comp if c["sets"] == "complex" else path


def measure(c, dev, G, Xs, st, comp, path, rng, facts=None):
    if c["id"] == "alike":
        M, _ = incidence(G, comp); return m_alike(Xs, st, M, rng)
    if c["id"] == "rebuild":
        M, _ = incidence(G, comp); return m_rebuild(G, Xs, st, M, rng)
    if c["id"] == "transfer":
        return m_transfer(dev, G, Xs, st, comp, rng, facts=facts)
    if c["id"] == "pathway":
        Mp, _ = incidence(G, path); Mc, _ = incidence(G, comp); return m_alike(Xs, st, Mp, rng, excl=Mc)
    raise KeyError(c["id"])


def control(c, kind, dev, comp, path, seed, rng):
    """positive: the planted mechanism on the dev genes; negative: unstructured responses. For transfer the dev
    facts come from a planted dev subject and the target shares its programmes (positive) or has none (negative)."""
    Gd = dev["G"]; sets = _sets(c, comp, path); Ms, _ = incidence(Gd, sets)
    if c["id"] == "transfer":
        Gc, Xc, sc, _ = analysis(Gd, planted(Gd, Ms, seed, noise_seed=seed))
        dev_c = dict(G=Gc, Xs=Xc)
        Zt = planted(Gd, Ms, seed, noise_seed=100 + seed) if kind == "positive" else unstructured(dev["Zraw"], len(Gd), seed)
        Gt, Xt, st_t, _ = analysis(Gd, Zt)
        r = m_transfer(dev_c, Gt, Xt, st_t, comp, rng)
    else:
        Z = planted(Gd, Ms, seed) if kind == "positive" else unstructured(dev["Zraw"], len(Gd), seed)
        Gc, Xc, sc, _ = analysis(Gd, Z)
        r = measure(c, dev, Gc, Xc, sc, comp, path, rng)
    r["pass"] = bool(r["stat"] >= c["threshold"] and r["p"] < GATE["null_p_lt"])
    return r


def write_facts(dev, comp):
    """The bank's facts the transfer claim reads: each complex's programme on the dev subject, citing its members."""
    os.makedirs(os.path.dirname(FACTS), exist_ok=True)
    names = sorted(comp); pos = {g: i for i, g in enumerate(dev["G"])}
    with open(FACTS, "w") as f:
        for c in names:
            mem = sorted(g for g in comp[c] if g in pos)
            if not mem:
                continue
            v = dev["Xs"][[pos[g] for g in mem]].sum(0)
            rec = dict(entity=f"subject/{DEV[0]}-{DEV[1]}", attribute=f"complex/{c}/programme_sum", members=mem,
                       value=[round(float(x), 6) for x in v])
            rec["id"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------------------------------------
# declare / test (the neuro ledger's rules)
# ---------------------------------------------------------------------------------------------------------
def code_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def spec(c):
    return dict(c, version=VERSION, dev=list(DEV), subjects=[s[0] for s in SUBJECTS], gate=GATE, n_null=N_NULL,
                n_ctrl=N_CTRL, n_pc=N_PC, qualify=QUALIFY, plant_signal=PLANT_SIGNAL, code_sha=code_sha())


def _read(path=LEDGER):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def _git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for c in CLAIMS:
            cid = f"{VERSION}-{c['id']}"
            if cid in have:
                print(f"{cid}: already declared"); continue
            rec = dict(op="declare", id=cid, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), spec=spec(c),
                       commit=_git("rev-parse", "HEAD").stdout.strip())
            f.write(json.dumps(rec, sort_keys=True) + "\n"); print(f"{cid}: declared")


def test():
    head = _git("show", "HEAD:rem/atlas/bio_ledger.jsonl").stdout
    committed = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
    for c in CLAIMS:
        cid = f"{VERSION}-{c['id']}"
        if cid not in committed:
            raise SystemExit(f"{cid}: its declaration is not in HEAD's rem/atlas/bio_ledger.jsonl; commit it first")
        if committed[cid]["spec"]["code_sha"] != code_sha():
            raise SystemExit(f"{cid}: the code changed since it was declared")
    if any(r["op"] == "result" and r["id"].startswith(VERSION) for r in _read()):
        raise SystemExit(f"{VERSION} already tested; a new test is a new VERSION")
    t0 = time.time(); rng = np.random.default_rng(7)
    bank, comp, path = load(); dev = dev_subject(bank); write_facts(dev, comp)
    subj = {}
    for nm, scr, sp in SUBJECTS:
        genes, Z = subject(bank, scr, sp); G, Xs, st, _ = analysis(genes, Z)
        Mc, _ = incidence(G, comp)
        subj[nm] = dict(G=G, Xs=Xs, st=st, qualifies=bool(int(Mc.any(1).sum()) >= QUALIFY))
    # the transfer claim reads the dev programmes from the facts file, not from memory
    facts = [json.loads(x) for x in open(FACTS)]
    print(f"facts written and read back: {len(facts)} complex programmes on the dev subject ({time.time() - t0:.0f}s)")
    out = []
    for c in CLAIMS:
        rows = []
        for nm, _, _ in SUBJECTS:
            s = subj[nm]
            if not s["qualifies"]:
                rows.append(dict(subject=nm, qualifies=False)); continue
            r = measure(c, dev, s["G"], s["Xs"], s["st"], comp, path, rng, facts=facts)
            r.update(subject=nm, qualifies=True, **{"pass": bool(r["stat"] >= c["threshold"] and r["p"] < GATE["null_p_lt"])})
            rows.append(r)
            print(f"  {c['id']:9s} {nm:9s} stat {r['stat']:+.3f} p {r['p']:.3f} genes {r['genes']:4d} -> {'pass' if r['pass'] else 'fail'}", flush=True)
        ctrl = {k: [control(c, k, dev, comp, path, s_, rng)["pass"] for s_ in range(N_CTRL)] for k in ("positive", "negative")}
        q = [r for r in rows if r["qualifies"]]; k = sum(r["pass"] for r in q)
        from scipy.stats import binomtest
        p = binomtest(k, len(q), 0.5, alternative="greater").pvalue if q else 1.0
        ok_ctrl = np.mean(ctrl["positive"]) >= GATE["ctrl_share"] and np.mean([not x for x in ctrl["negative"]]) >= GATE["ctrl_share"]
        gate = len(q) >= GATE["min_qualify"] and k >= GATE["pass_share"] * len(q) and ok_ctrl
        out.append(dict(op="result", id=f"{VERSION}-{c['id']}", subjects=rows, controls=ctrl, passing=k, qualifying=len(q),
                        p=float(p), controls_ok=bool(ok_ctrl), gate=bool(gate), expect=c["expect"],
                        at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_sha=code_sha()))
        print(f"  {c['id']}: {k}/{len(q)} subjects pass; controls positive {sum(ctrl['positive'])}/{N_CTRL}, negative failed "
              f"{N_CTRL - sum(ctrl['negative'])}/{N_CTRL} ({time.time() - t0:.0f}s)", flush=True)
    desc = []
    for nm, _, _ in SUBJECTS:
        s_ = subj[nm]; Mp, _ = incidence(s_["G"], path); Mc, _ = incidence(s_["G"], comp)
        r = m_alike(s_["Xs"], s_["st"], Mp, rng, excl=Mc); r["subject"] = nm; desc.append(r)
    # BH over every claim in this ledger
    ps = np.array([o["p"] for o in out]); order = ps.argsort(); m = len(ps); qv = np.empty(m)
    run = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        run = min(run, ps[i] * m / rank); qv[i] = run
    with open(LEDGER, "a") as f:
        for o, q_ in zip(out, qv):
            o["q"] = float(q_); o["verdict"] = ("PASS" if o["gate"] and q_ < GATE["bh_q"] else
                                                "NO VERDICT" if (o["qualifying"] < GATE["min_qualify"] or not o["controls_ok"]) else "FAIL")
            f.write(json.dumps(o, sort_keys=True) + "\n")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(op="descriptive", id=f"{VERSION}-pathway-auroc", subjects=desc, code_sha=code_sha()), sort_keys=True) + "\n")
    report(out, desc)


def report(out, desc):
    L = ["=" * 97, "BIOBRAIN -- the brain's neuro engine pointed at the cell: claims declared before they ran", "=" * 97,
         f"  dev subject {DEV[0]} {DEV[1]} genes (thresholds set there only); verdict on {len(SUBJECTS)} screens nothing looked at",
         f"  {'claim':9s} {'threshold':>9s} {'dev':>6s} {'pass/qualify':>13s} {'controls +/-':>13s} {'p':>8s} {'q':>8s}  verdict (expected)"]
    for o, c in zip(out, CLAIMS):
        L.append(f"  {c['id']:9s} {c['threshold']:9.2f} {('%.3f' % c['dev']) if c['dev'] is not None else '  -  ':>6s} "
                 f"{o['passing']:>6d}/{o['qualifying']:<6d} {sum(o['controls']['positive']):>5d}/{N_CTRL - sum(o['controls']['negative'])}{'':<6s} "
                 f"{o['p']:8.2g} {o['q']:8.2g}  {o['verdict']} ({o['expect']})")
    L.append("  per subject (measure, null p):")
    for o, c in zip(out, CLAIMS):
        L.append(f"    {c['id']:9s} " + "; ".join(f"{r['subject']} {r['stat']:+.3f} (p {r['p']:.3f})" for r in o["subjects"] if r.get("qualifies")))
    L.append("  descriptive, no verdict -- pathway-only pairs (same Reactome pathway, no shared complex), AUROC (null p):")
    L.append("    " + "; ".join(f"{r['subject']} {r['stat']:.3f} ({r['p']:.3f})" for r in desc))
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"look": look, "declare": declare, "test": test}[sys.argv[1]]()
