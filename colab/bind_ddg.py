"""bind_ddg — the trained EXTRINSIC magnitude sensor: predict ΔΔG-of-binding for ANY interface mutation on a complex,
all 20 amino acids, from STRUCTURE-DERIVABLE physics features only (so it works on any mutation, not just SKEMPI-measured
ones). This turns NEXUS's extrinsic axis from a measured-value lookup into a real predictor.

Features (all computable from the WT complex structure): property deltas (hydropathy/volume/charge) + the WT sidechain's
interface contribution (LJ vdW / #partner-contacts / Coulomb / induction, from flex_physics) + the MUTANT sidechain's
best precomputed-rotamer clash (Dunbrack χ-wells) + a charge-superposition field for charge-changing mutations.

ESM is deliberately NOT a feature: ESM reads a single chain and is partner-blind, so it scores the FOLD/fitness axis,
which is orthogonal to binding (measured: ESM-alone r=0.09 on ΔΔG-binding vs physics 0.46). ESM belongs on the intrinsic
FOLD sensor, not here.

Honest accuracy, held-out BY COMPLEX. FULL SKEMPI (277 complexes, 4,417 muts): all-AA r 0.51, non-alanine 0.51,
alanine 0.51, hotspot(ΔΔG>1) AUC 0.77 — the substitution-class gap VANISHES at scale. (46-complex sandbox subset:
all-AA 0.46, non-alanine 0.30 — that low non-ala was DATA-STARVATION, ~358 non-ala examples, NOT intrinsic difficulty:
given enough non-alanine training data the physics+rotamer+charge features predict general substitutions as well as
alanine.) The committed bind_ddg_model.pkl is the sandbox fallback; bind_ddg_colab.ipynb trains the full-SKEMPI model.
  train(pdbs, cache) -> outputs/orphan/bind_ddg_model.pkl (+ bind_ddg.json)   |   Predictor.predict(pdb,ch,pos,wt,mut)
"""
import os, sys, json, pickle, warnings, re, csv
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "orphan")
MODEL = os.path.join(OUT, "bind_ddg_model.pkl")

VOL = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
HYD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHG = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}
AROM = set("FYWH")
FEAT_NAMES = ["dHyd", "dVol", "dChg", "absdChg", "wt_arom", "wt_vol", "wt_chg", "mut_vol", "mut_chg",
              "vdw", "contacts", "elec", "induc", "rot_bestwell", "charge_field"]
FEAT_NAMES_ALL = FEAT_NAMES + ["n_sites"]                # the deployed model is JOINT single+multi (n_sites)


def _mut_rotamer(t, ch, pos, mut):
    """best precomputed-χ-well soft-core clash of the mutant sidechain vs the partner, + charge-superposition field."""
    import flex_physics as fp
    from flex_physics import mut_sidechain, WELLS, CHI1_FAR, CHI2_FAR, _find, _dihedral, _rot_axis, _lj_sum, _RAD
    res = (t["ch"] == ch) & (t["rn"] == pos); nm0 = t["nm"][res]
    if not all(a in nm0 for a in ("N", "CA", "C")):
        return 0.0, 0.0
    N = t["co"][res][nm0 == "N"][0]; CA = t["co"][res][nm0 == "CA"][0]; C = t["co"][res][nm0 == "C"][0]
    co, mr, nm = mut_sidechain(mut, N, CA, C)
    if len(co) == 0:
        return 0.0, 0.0
    other = t["ch"] != ch
    dmin = np.linalg.norm(t["co"][:, None, :] - co[None, :, :], axis=2).min(1)
    part = other & (dmin < 10.0)
    if not part.any():
        return 0.0, 0.0
    oc, orr = t["co"][part], t["rad"][part]; rad = _RAD(np.array([n[0] for n in nm]))
    cb = _find(nm, ["CB"]); cg = _find(nm, CHI1_FAR); idx = np.arange(len(nm)); es = []
    for x1 in (WELLS if (cb >= 0 and cg >= 0) else (None,)):
        c1 = co.copy()
        if x1 is not None:
            cur = _dihedral(N, CA, co[cb], co[cg]); c1[idx != cb] = _rot_axis(c1[idx != cb], CA, co[cb], x1 - cur)
        cd = _find(nm, CHI2_FAR)
        for x2 in (WELLS if (cb >= 0 and cg >= 0 and cd >= 0) else (None,)):
            c2 = c1.copy()
            if x2 is not None:
                cur2 = _dihedral(CA, c1[cb], c1[cg], c1[cd]); m2 = (idx != cb) & (idx != cg)
                c2[m2] = _rot_axis(c2[m2], c1[cb], c1[cg], x2 - cur2)
            es.append(_lj_sum(c2, rad, oc, orr, soft=True))
    q = CHG.get(mut, 0.0)
    qf = q * (1.0 / (np.linalg.norm(co[:, None, :] - oc[None, :, :], axis=2) + 1e-6)).sum() if q else 0.0
    return float(min(es)), float(qf)


def features(t, ch, pos, wt, mut):
    """the structure-derivable feature vector for one interface mutation (returns None if it can't be built)."""
    import flex_physics as fp
    sc = fp.sidechain_vdw(t, ch, pos, wt=wt)
    if sc is None or wt not in VOL or mut not in VOL:
        return None
    rot, qf = _mut_rotamer(t, ch, pos, mut)
    return [HYD[mut] - HYD[wt], VOL[mut] - VOL[wt], CHG.get(mut, 0) - CHG.get(wt, 0), abs(CHG.get(mut, 0) - CHG.get(wt, 0)),
            1 if wt in AROM else 0, VOL[wt], CHG.get(wt, 0), VOL[mut], CHG.get(mut, 0),
            sc["vdw"], sc["contacts"], sc["elec"], sc["induc"], rot, qf]


def features_agg(t, sites):
    """aggregate the per-site features of a SINGLE- or MULTI-point mutation: element-wise SUM of the per-site vectors
    (physics energies are ~additive) plus n_sites. Returns a FEAT_NAMES_ALL-length vector, or None if any site fails."""
    fs = []
    for ch, pos, wt, mut in sites:
        f = features(t, ch, int(pos), wt, mut)
        if f is None:
            return None
        fs.append(f)
    return list(np.sum(fs, axis=0)) + [len(sites)]


def _load_all(skempi_path):
    """parse SKEMPI single AND multi-point mutations -> [(pdb, [(chain,pos,wt,mut),...], ddg_bind), ...]."""
    import interface_hotspots as ih
    site_re = re.compile(r"^([A-Z])([A-Za-z0-9])(\d+)([A-Z])$")
    out = []
    for r in list(csv.reader(open(skempi_path), delimiter=";"))[1:]:
        if not r or not r[1]:
            continue
        sites, ok = [], True
        for mm in r[1].split(","):
            m = site_re.match(mm)
            if not m:
                ok = False; break
            sites.append((m.group(2), int(m.group(3)), m.group(1), m.group(4)))
        if not ok or not sites:
            continue
        try:
            km, kw = float(r[7]), float(r[9])
        except (ValueError, IndexError):
            continue
        if km <= 0 or kw <= 0:
            continue
        T = 298.0; mt = re.match(r"\s*(\d+)", r[13] or "")
        if mt:
            T = float(mt.group(1))
        out.append((r[0].split("_")[0], sites, float(ih.R * T * np.log(km / kw))))
    return out


def _dataset(pdbs, cache):
    """joint single+multi dataset. Returns X (FEAT_NAMES_ALL), y, grp(pdb), nsite, mutl (single mut letter or '*')."""
    import flex_physics as fp, interface_hotspots as ih
    fp.PDBDIR = cache
    have = set(pdbs)
    X, y, grp, nsite, mutl = [], [], [], [], []
    for pdb, sites, ddg in _load_all(ih.SKEMPI):
        if pdb not in have:
            continue
        t = fp.table(pdb)
        if t is None:
            continue
        f = features_agg(t, sites)
        if f is None:
            continue
        X.append(f); y.append(ddg); grp.append(pdb); nsite.append(len(sites))
        mutl.append(sites[0][3] if len(sites) == 1 else "*")
    return np.array(X), np.array(y), np.array(grp), np.array(nsite), np.array(mutl)


def train(pdbs, cache, model_out=MODEL):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr
    X, y, grp, nsite, mutl = _dataset(pdbs, cache)
    single = nsite == 1; ala = mutl == "A"
    print(f"  {len(y)} mutations / {len(set(grp))} complexes  (single {int(single.sum())} [ala {int(ala.sum())}], multi {int((~single).sum())})", flush=True)
    if len(set(grp)) < 5:
        return {"error": "need >=5 complexes"}
    pred = cross_val_predict(RandomForestRegressor(150, max_depth=10, random_state=0, n_jobs=-1), X, y, groups=grp, cv=GroupKFold(5))

    def rr(m):
        return round(float(pearsonr(pred[m], y[m])[0]), 3) if m.sum() > 20 and len(set(grp[m])) > 1 else None
    r_all, r_non, r_ala, r_multi = rr(single), rr(single & ~ala), rr(single & ala), rr(~single)
    yc = (y > 1.0).astype(int)
    auc = round(float(roc_auc_score(yc, pred)), 3) if len(set(yc)) == 2 else None
    # fit the deployed JOINT model on ALL data (single + multi, n_sites feature)
    final = RandomForestRegressor(150, max_depth=10, random_state=0, n_jobs=-1).fit(X, y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    pickle.dump({"model": final, "feat_names": FEAT_NAMES_ALL}, open(model_out, "wb"))
    out = {"n_mutations": int(len(y)), "n_complexes": int(len(set(grp))), "n_single": int(single.sum()), "n_multi": int((~single).sum()),
           "r_all": r_all, "r_nonala": r_non, "r_alanine": r_ala, "r_multi": r_multi, "hotspot_auc": auc,
           "model": os.path.basename(model_out), "features": FEAT_NAMES_ALL, "esm_used": False,
           "note": "JOINT single+multi model (n_sites feature). r_all/r_nonala/r_alanine are SINGLE-mutation (NEXUS's use); "
                   "r_multi is multi-point (a free scope-add — neutral to single). Structure-only physics+rotamer+charge; "
                   "ESM excluded (partner-blind/fold-axis, r=0.09 on binding). Held-out by complex."}
    json.dump(out, open(os.path.join(OUT, "bind_ddg.json"), "w"), indent=1, default=float)
    print(f"  held-out BY COMPLEX (single): all-AA r={r_all}  non-ala r={r_non}  alanine r={r_ala}   |   multi r={r_multi}   |   hotspot AUC {auc}", flush=True)
    print(f"  -> {model_out}", flush=True)
    return out


class Predictor:
    def __init__(self, model_path=MODEL):
        self.ok = os.path.exists(model_path)
        if self.ok:
            d = pickle.load(open(model_path, "rb")); self.m = d["model"]

    def predict(self, pdb, ch, pos, wt, mut, cache=None):
        """predicted ΔΔG-binding (kcal/mol, >0 = weaker binding) for one single-point mutation."""
        return self.predict_multi(pdb, [(ch, pos, wt, mut)], cache=cache)

    def predict_multi(self, pdb, sites, cache=None):
        """predicted ΔΔG-binding for a SINGLE- or MULTI-point mutation: sites = [(chain,pos,wt,mut), ...]. None if
        features unavailable. Multi is a free scope-add (held-out multi r~0.38 / hotspot AUC 0.76); single is r~0.51."""
        if not self.ok:
            return None
        import flex_physics as fp
        if cache:
            fp.PDBDIR = cache
        t = fp.table(pdb)
        if t is None:
            return None
        f = features_agg(t, sites)
        if f is None:
            return None
        return float(self.m.predict(np.array(f).reshape(1, -1))[0])


if __name__ == "__main__":
    import flex_physics as fp
    cache = sys.argv[1] if len(sys.argv) > 1 else fp.PDBDIR
    have = sorted({os.path.splitext(f)[0] for f in os.listdir(cache) if f.endswith(".pdb")})
    print(f"training bind_ddg on {len(have)} cached complexes in {cache}", flush=True)
    train(have, cache)
