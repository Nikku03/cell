"""POST-HOC (written after cellbank4 score): where do the RPE1 and partner inputs help, and the screen-B ceiling
on the same genes. Not a predeclared test. Run from the repo root: python3 -P rem/atlas/cellbank4_posthoc.py"""
import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "rem", "atlas"))
import importlib.util
s = importlib.util.spec_from_file_location("cb", "rem/atlas/cellbank4.py"); cb = importlib.util.module_from_spec(s); s.loader.exec_module(cb)
G = cb.notebook(); bank = cb.Bank(cb.BANK); torch, np = G["torch"], G["np"]
from scipy.stats import binomtest
preds = torch.load(os.path.join(cb.CACHE, "engine_preds.pt")); Vy = G["Vy"].cpu()
R = {a: torch.nan_to_num(G["pearson_rows"](preds[a] @ Vy.T, G["A_te"])).numpy() for a in cb.ARMS}
KN, rte = G["KN"], G["rte"]
has_c = np.array([bank.get(KN[int(rte[t])], "C") is not None for t in range(len(rte))])
has_p = np.array([any(p in bank.by_gene and bank.get(p, "A") is not None or bank.get(p, "B") is not None for p in bank.genes[KN[int(rte[t])]]["partners"]) for t in range(len(rte))])
def cmp(a, b, m, tag):
    d = R[a][m] - R[b][m]; w, l = int((d > 0).sum()), int((d < 0).sum())
    print(f"  {tag:52s} n {m.sum():5d}  {a} {np.median(R[a][m]):+.3f} vs {b} {np.median(R[b][m]):+.3f}  per-gene median {np.median(d):+.4f}  better {w} / worse {l}  p {binomtest(w, w + l).pvalue:.2g}")
cmp("E2", "E1", has_c, "TEST genes WITH their own RPE1 record")
cmp("E2", "E1", ~has_c, "TEST genes without")
cmp("E2", "E0", has_c, "TEST genes WITH their own RPE1 record")
cmp("E1", "E0", has_p, "TEST genes with a partner record in the bank")
cmp("E1", "E0", ~has_p, "TEST genes without")
# noise ceiling (screen B replicate, raw, as notebook section 10) on the same genes
A, TE, TR = G["A"], G["TE"], G["TR"]
Bs = G["load_screen"](G["P"]["ess.h5ad"]); bk = {p: i for i, p in enumerate(Bs["kos"])}; bg = {}
for j, g in enumerate(Bs["genes"]): bg.setdefault(g, j)
cols = [j for j, g in enumerate(A["genes"]) if g in bg]; bcols = [bg[A["genes"][j]] for j in cols]
btr = [bk[A["kos"][i]] for i in TR if A["kos"][i] in bk]; btide = torch.nanmean(Bs["X"][btr][:, bcols], 0)
both = [t for t in range(len(TE)) if has_c[t] and A["kos"][TE[t]] in bk]
Bp = torch.nan_to_num(Bs["X"][[bk[A["kos"][TE[t]]] for t in both]][:, bcols] - btide)
ceil = torch.nan_to_num(G["pearson_rows"](Bp, G["A_te"][both][:, cols])).numpy()
print(f"  CEILING on {len(both)} TEST genes with an RPE1 record and a screen-B replicate: median {np.median(ceil):+.3f}; E2 on the same genes {np.median(R['E2'][both]):+.3f}; E1 {np.median(R['E1'][both]):+.3f}")
