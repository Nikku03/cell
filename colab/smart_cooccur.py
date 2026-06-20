"""Smart cooccurrence integration: a calibrated stacker over the transformer's
abstention bucket that maximises NEW coverage at a fixed precision floor.

Why a stacker (not transformer features): adding cooccur as focal features was
flat (the MSA already encodes ranking signal). But on the genes the transformer
ABSTAINS on (brightness < 0.85), the cooccur channels carry orthogonal
information. Instead of the old hard '>=2 votes agree' rule, we:

  1. compute CONTINUOUS, leak-clean channel scores per abstained gene
     (backup, presence, co-essentiality, phyletic, dN/dS),
  2. fit a logistic stacker on [transformer_p, brightness, channels] with a
     CROSS-CLADE fit (score clade c using a stacker trained on the other 4),
  3. pick the single acceptance threshold that maximises rescued coverage
     subject to COMBINED precision >= PRECISION_FLOOR (default 0.90).

This directly optimises "most coverage add, least precision drop", and the
dN/dS channel is ablated (--no_dnds) so its marginal value is explicit.

Inputs : outputs/orphan/af_torch_preds.npz  (from af_torch.py)
         data/drive_import/labels/{labels,orthology_features,cooccurrence_features}.csv
         optional expanded presence matrix (build_presence.py)
Output : outputs/orphan/smart_cooccur_results.json
"""
import csv, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

from af_common import OUT, DATA, MAJOR_CLADES

PRECISION_FLOOR = 0.90
BRIGHT = 0.85


# ---- tiny standardised logistic regression (no sklearn dependency) ---------
def fit_logreg(X, y, l2=1.0, iters=300, lr=0.5):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.c_[np.ones(len(Xs)), Xs]
    w = np.zeros(Xs.shape[1]); n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ (p - y) / n + l2 * np.r_[0, w[1:]] / n
        w -= lr * g
    return (w, mu, sd)


def pred_logreg(model, X):
    w, mu, sd = model
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))


def load_orthology(presence_npz=None):
    """Build Y[O,G] essentiality, P[O,G] presence, plus index maps.
    If presence_npz is given (build_presence.py output), use its larger
    presence matrix for the cooccur partner search (essentiality stays from
    labelled orgs only)."""
    gene_og = {}; og_present = defaultdict(set)
    for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
        if r["og_id"]:
            gene_og[(r["organism"], r["locus_tag"])] = r["og_id"]
            og_present[r["og_id"]].add(r["organism"])
    labels = {(r["organism"], r["locus_tag"]): int(r["essential"])
              for r in csv.DictReader(open(DATA / "labels" / "labels.csv"))}
    all_orgs = sorted({o for s in og_present.values() for o in s})
    org_idx = {o: i for i, o in enumerate(all_orgs)}
    all_ogs = sorted(og_present); og_idx = {g: j for j, g in enumerate(all_ogs)}
    O, G = len(all_orgs), len(all_ogs)
    Y = np.full((O, G), np.nan, np.float32)
    for (o, l), e in labels.items():
        g = gene_og.get((o, l))
        if g:
            Y[org_idx[o], og_idx[g]] = e
    P = np.zeros((O, G), np.float32)
    for g, orgs_p in og_present.items():
        j = og_idx[g]
        for o in orgs_p:
            P[org_idx[o], j] = 1.0
    cofeat = {}
    for r in csv.DictReader(open(DATA / "labels" / "cooccurrence_features.csv")):
        try:
            cofeat[(r["organism"], r["locus_tag"])] = float(r["cooccur_essential_neighbor_frac"])
        except Exception:
            pass
    return dict(gene_og=gene_og, og_idx=og_idx, org_idx=org_idx,
                Y=Y, P=P, cofeat=cofeat, O=O, G=G, all_orgs=all_orgs)


def channel_features(preds, orth, use_dnds=True):
    """Return X (feature matrix) and y for every transformer-abstained gene,
    plus index bookkeeping. Channel scores are leak-clean per held clade and
    cached per (OG, clade) for speed."""
    p = preds["p"]; br = preds["brightness"]; y = preds["y"]
    clade = preds["clade"]; meta_org = preds["meta_org"]; lt = preds["lt"]
    orgs_cache = list(preds["orgs"]) if "orgs" in preds else None

    Y, P = orth["Y"], orth["P"]; O = orth["O"]
    org_idx = orth["org_idx"]; og_idx = orth["og_idx"]; gene_og = orth["gene_og"]; cofeat = orth["cofeat"]
    p_sum = P.sum(0); cand_p = np.where((p_sum >= 5) & (p_sum <= O - 5))[0]
    n_lab = (~np.isnan(Y)).sum(0); mean_e = np.nanmean(Y, axis=0)
    cand_e = np.where((n_lab >= 10) & (mean_e > 0.05) & (mean_e < 0.95))[0]
    Pc = P[:, cand_p] - P[:, cand_p].mean(0); Pn = Pc / (Pc.std(0) + 1e-9)

    # cache-org index -> org name (cache stores names list); fall back to str
    def org_name(i):
        return str(orgs_cache[i]) if orgs_cache is not None else str(i)

    feats = []; ys = []; meta = []
    base_phyl = np.nanmean([v for v in cofeat.values()]) if cofeat else 0.5

    for cl in MAJOR_CLADES:
        sel = np.where((clade == cl) & (br < BRIGHT))[0]   # abstained genes only
        if len(sel) == 0:
            continue
        # training orgs = orthology orgs NOT in this clade
        held_orgs = set()
        for i in np.where(clade == cl)[0]:
            nm = org_name(int(meta_org[i]))
            if nm in org_idx:
                held_orgs.add(org_idx[nm])
        train_mask = np.ones(O, bool)
        for oi in held_orgs:
            train_mask[oi] = False

        # precompute per-OG partner picks (depends only on j + clade)
        og_cache = {}
        def og_channels(j):
            if j in og_cache:
                return og_cache[j]
            res = dict(backup=(None, 0.0), presence=(None, 0.0), coess=(None, 0.0))
            e_tr = Y[train_mask, j]; v = ~np.isnan(e_tr)
            if v.sum() >= 8 and e_tr[v].std() >= 0.1:
                et = e_tr[v]; et_z = (et - et.mean()) / (et.std() + 1e-9)
                Pn_tr = Pn[train_mask][v]; Pnc = Pn_tr - Pn_tr.mean(0); Pnz = Pnc / (Pnc.std(0) + 1e-9)
                corr = (et_z @ Pnz) / len(et_z)
                if j in cand_p:
                    pos = np.searchsorted(cand_p, j)
                    if pos < len(cand_p) and cand_p[pos] == j:
                        corr[pos] = 2.0
                bb = int(np.argmax(-corr)); res["backup"] = (cand_p[bb], float(-corr[bb]))
                bp = int(np.argmax(corr));  res["presence"] = (cand_p[bp], float(corr[bp]))
                Y_tr = Y[train_mask][:, cand_e]; Ym = np.nanmean(Y_tr, axis=0); Ym[np.isnan(Ym)] = 0.5
                Yi = np.where(np.isnan(Y_tr), Ym[None, :], Y_tr)[v]
                Yc = Yi - Yi.mean(0); Yz = Yc / (Yc.std(0) + 1e-9)
                ce = (et_z @ Yz) / len(et_z)
                if j in cand_e:
                    pos = np.searchsorted(cand_e, j)
                    if pos < len(cand_e) and cand_e[pos] == j:
                        ce[pos] = -2.0
                bc = int(np.argmax(ce)); res["coess"] = (cand_e[bc], float(ce[bc]))
            og_cache[j] = res
            return res

        for i in sel:
            nm = org_name(int(meta_org[i])); lts = str(lt[i])
            oi = org_idx.get(nm, -1)
            og = gene_og.get((nm, lts)); j = og_idx.get(og, -1) if og else -1
            f = dict(backup=0.0, presence=0.0, coess=0.0, phyl=0.0,
                     dnds=0.0, has_dnds=0.0)
            if j >= 0 and oi >= 0:
                ch = og_channels(j)
                pog, sc = ch["backup"]
                if pog is not None and sc >= 0.4:
                    f["backup"] = sc * (1.0 if P[oi, pog] == 0 else -1.0)  # ess iff partner absent
                pog, sc = ch["presence"]
                if pog is not None and sc >= 0.4:
                    f["presence"] = sc * (1.0 if P[oi, pog] == 1 else -1.0)
                pog, sc = ch["coess"]
                if pog is not None and sc >= 0.4:
                    pe = Y[oi, pog]
                    if not np.isnan(pe):
                        f["coess"] = sc * (1.0 if pe >= 0.5 else -1.0)
            ph = cofeat.get((nm, lts))
            if ph is not None:
                f["phyl"] = ph - base_phyl
            # dN/dS straight from the transformer focal cache (col5=dnds, col6=has_dnds)
            if "foc" in preds:
                f["dnds"] = float(preds["foc"][i, 5]); f["has_dnds"] = float(preds["foc"][i, 6])
            row = [p[i], br[i], f["backup"], f["presence"], f["coess"], f["phyl"]]
            if use_dnds:
                row += [f["dnds"], f["has_dnds"]]
            feats.append(row); ys.append(float(y[i])); meta.append((cl, int(i)))
    return np.array(feats, np.float32), np.array(ys, np.float32), meta


def evaluate(preds, X, y, meta, label=""):
    """Cross-clade stacker fit + threshold selection to a precision floor."""
    p_all = preds["p"]; br_all = preds["brightness"]; y_all = preds["y"]; clade = preds["clade"]
    clades = np.array([m[0] for m in meta])

    # transformer-confident block (fixed)
    ev = ~np.isnan(p_all)
    hi = ev & (br_all >= BRIGHT)
    T_called = int(hi.sum())
    T_correct = int(((p_all[hi] > 0.5).astype(int) == y_all[hi]).sum())
    N_pooled = int(ev.sum())

    # cross-clade stacker predictions over the abstained pool
    stack_p = np.zeros(len(y))
    for cl in MAJOR_CLADES:
        te = clades == cl; trn = ~te
        if te.sum() == 0 or trn.sum() < 50:
            continue
        model = fit_logreg(X[trn], y[trn])
        stack_p[te] = pred_logreg(model, X[te])
    conf = np.maximum(stack_p, 1 - stack_p)
    call = (stack_p >= 0.5).astype(int)
    correct = (call == y).astype(int)

    # choose acceptance threshold maximising rescued count s.t. combined precision >= floor
    order = np.argsort(-conf)
    best = dict(tau=1.0, rescued=0, combined_cov=T_called / N_pooled,
                combined_prec=(T_correct / T_called) if T_called else 0.0)
    csum_correct = np.cumsum(correct[order])
    csum_n = np.arange(1, len(order) + 1)
    for k in range(len(order)):
        acc_correct = int(csum_correct[k]); acc_n = int(csum_n[k])
        comb_prec = (T_correct + acc_correct) / (T_called + acc_n)
        if comb_prec >= PRECISION_FLOOR:
            tau = float(conf[order[k]])
            cov = (T_called + acc_n) / N_pooled
            if acc_n > best["rescued"]:
                best = dict(tau=tau, rescued=acc_n, rescued_acc=acc_correct / acc_n,
                            combined_cov=cov, combined_prec=comb_prec)
    # tradeoff curve
    curve = []
    for tau in np.linspace(0.5, 0.99, 20):
        m = conf >= tau
        acc_n = int(m.sum()); acc_c = int(correct[m].sum())
        comb_prec = (T_correct + acc_c) / (T_called + acc_n) if (T_called + acc_n) else 0
        curve.append(dict(tau=round(float(tau), 3), rescued=acc_n,
                          combined_coverage=round((T_called + acc_n) / N_pooled, 4),
                          combined_precision=round(comb_prec, 4)))
    res = dict(
        label=label, n_pooled=N_pooled,
        transformer=dict(coverage=round(T_called / N_pooled, 4),
                         precision=round(T_correct / T_called, 4) if T_called else 0),
        smart_combined=dict(coverage=round(best["combined_cov"], 4),
                            precision=round(best["combined_prec"], 4),
                            rescued=best["rescued"], tau=round(best["tau"], 4),
                            rescued_accuracy=round(best.get("rescued_acc", 0), 4)),
        delta=dict(coverage_pp=round((best["combined_cov"] - T_called / N_pooled) * 100, 2),
                   precision_pp=round((best["combined_prec"] - (T_correct / T_called if T_called else 0)) * 100, 2)),
        tradeoff_curve=curve)
    return res


def main(presence_npz=None):
    preds = dict(np.load(OUT / "af_torch_preds.npz", allow_pickle=True))
    # attach focal cache (for dN/dS channel) + org names
    from af_common import Cache
    c = Cache()
    preds["foc"] = c.foc; preds["orgs"] = np.array(c.orgs, dtype=object)
    orth = load_orthology(presence_npz)

    out = {}
    for use_dnds, tag in [(True, "with_dnds"), (False, "no_dnds")]:
        X, y, meta = channel_features(preds, orth, use_dnds=use_dnds)
        print(f"\n[{tag}] abstained genes scored: {len(y)}")
        res = evaluate(preds, X, y, meta, label=tag)
        out[tag] = res
        t = res["transformer"]; s = res["smart_combined"]; d = res["delta"]
        print(f"  transformer: {t['coverage']:.3f} @ {t['precision']:.3f}")
        print(f"  smart combined: {s['coverage']:.3f} @ {s['precision']:.3f} "
              f"(rescued {s['rescued']} @ {s['rescued_accuracy']:.3f}, tau={s['tau']:.3f})")
        print(f"  delta: {d['coverage_pp']:+.2f}pp coverage, {d['precision_pp']:+.2f}pp precision")

    dn = out["with_dnds"]["smart_combined"]["coverage"] - out["no_dnds"]["smart_combined"]["coverage"]
    out["dnds_marginal_coverage_pp"] = round(dn * 100, 2)
    print(f"\ndN/dS marginal coverage: {dn*100:+.2f}pp")
    json.dump(out, open(OUT / "smart_cooccur_results.json", "w"), indent=2)
    print("wrote smart_cooccur_results.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--presence_npz", default=None, help="optional expanded presence matrix")
    a = ap.parse_args()
    main(a.presence_npz)
