"""grn — the cell as RUNNING software: a gene-regulatory dynamical system you can boot and watch flow.

Everything else in the model is either a photograph (measured snapshot) or a destination without a journey
(FBA solves for the steady operating point directly). This is the one piece with a CLOCK: a state that starts
somewhere and evolves, tick by tick, under the genome's own regulatory logic, until it settles into a stable
pattern — an ATTRACTOR. In dynamical-systems terms an attractor is a cell state/type; "running the cell" is
letting the rules pull an initial state into one.

The program (the logic):  54k SIGNED transcriptional edges (SIGNOR/CollecTRI) — TF ->(activate|repress) target.
The runtime (the update):  a continuous-state recurrent map over the 7,480 genes that HAVE regulators
    x_next[g] = (1-lam)*x[g] + lam * sigmoid( beta * ( (W x)[g] - theta ) )
  W is the signed regulatory matrix, ROW-NORMALIZED per target so no hub dominates and the sum stays bounded;
  theta is a NEGATIVE-leaning threshold so a gene decays OFF unless it gets net activation (this is the fix for
  "activation domination": the edge set is 4.5:1 activating, so without a threshold everything trivially turns on).
  Genes with no regulators are inputs — clamped to their seed (boundary conditions / the environment).

Honesty discipline (this is a FRONTIER build, unvalidated going in):
  * the ONLY thing calibrated is the operating point (global theta set so the ON-fraction matches the ~40% of
    genes a real cell expresses). Nothing is tuned to the essentiality target.
  * the real test (grn_validate.py) is BLIND: does the dynamical fragility of the attractor when you knock a gene
    out predict MEASURED essentiality (dep_frac) — and does it beat the trivial connectivity (out-degree) baseline?
    If dynamics add nothing over raw hub-ness, that is the finding and we report it.

-> used by grn_validate.py; wired into CellOS as `boot`/`exec` only if it earns it.
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp

OUT = "outputs/orphan"
EDGES = f"{OUT}/causal_edges.json"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class GRN:
    def __init__(self, edge_path=EDGES, C=None, unsigned_weight=0.5, target_on_frac=0.40,
                 beta=4.0, lam=0.5, seed=0):
        """Build the signed, row-normalized regulatory dynamical system over the genes that have regulators."""
        self.beta, self.lam = beta, lam
        d = json.load(open(edge_path))
        edges = d["edges"]                                   # [reg_idx, tgt_idx, sign, source]
        # node universe = every gene that appears as a regulator or a target
        nodes = sorted({e[0] for e in edges} | {e[1] for e in edges})
        self.nodes = np.array(nodes)                         # local index -> global gene index
        self.g2l = {g: i for i, g in enumerate(nodes)}       # global gene index -> local
        n = len(nodes)
        self.n = n

        # signed weight matrix W[target, regulator] as sparse CSR; unsigned edges get a small positive weight
        rows, cols, vals = [], [], []
        indeg = np.zeros(n, dtype=np.float32)
        for reg, tgt, sign, _src in edges:
            r, t = self.g2l[reg], self.g2l[tgt]
            w = float(sign) if sign in (1, -1) else unsigned_weight
            rows.append(t); cols.append(r); vals.append(w)
            indeg[t] += 1.0
        W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        W.sum_duplicates()                                   # collapse parallel edges (same reg->tgt) into one weight
        # ROW-NORMALIZE by in-degree so a gene with 300 regulators isn't 100x louder than one with 3 (hub control)
        nz = indeg > 0
        inv = np.zeros(n, dtype=np.float32); inv[nz] = 1.0 / indeg[nz]
        W = sp.diags(inv) @ W
        self.W = W.tocsr()
        self.indeg = indeg
        self.is_input = ~nz                                  # genes with no incoming edge = clamped inputs
        # out-degree (# targets each gene regulates) for the connectivity-confound baseline
        self.outdeg = np.asarray((self.W != 0).sum(0)).ravel().astype(np.float32)

        # names, for reporting
        self.name = None
        if C is not None:
            self.name = np.array([C.genes[g].get("name", str(g)) if g < len(C.genes) else str(g)
                                  for g in nodes])

        # calibrate the global threshold theta so the settled ON-fraction ~ target_on_frac (the ONLY tuned knob)
        self.theta = 0.0
        self._calibrate_theta(target_on_frac, seed)

    # ---------------------------------------------------------------- dynamics
    def step(self, x):
        drive = self.W @ x                                   # net signed regulatory input per gene
        upd = _sigmoid(self.beta * (drive - self.theta))
        x_new = (1 - self.lam) * x + self.lam * upd
        x_new[self.is_input] = x[self.is_input]              # inputs are boundary conditions — held fixed
        return x_new

    def run(self, x0, steps=300, tol=1e-4, clamp=None):
        """Iterate to an attractor. clamp = {local_idx: value} genes pinned every step (knockouts / inputs)."""
        x = x0.astype(np.float32).copy()
        if clamp:
            for i, v in clamp.items():
                x[i] = v
        traj_norm = []
        conv_step = steps
        for s in range(steps):
            xn = self.step(x)
            if clamp:
                for i, v in clamp.items():
                    xn[i] = v
            delta = float(np.abs(xn - x).max())
            traj_norm.append(delta)
            x = xn
            if delta < tol:
                conv_step = s + 1
                break
        return x, dict(converged=conv_step < steps, steps=conv_step, final_delta=traj_norm[-1] if traj_norm else 0.0)

    # ---------------------------------------------------------------- operating point
    def _seed(self, seed=0, frac=0.5):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0, 1, self.n).astype(np.float32)
        x[self.is_input] = (rng.uniform(0, 1, self.is_input.sum()) < frac).astype(np.float32)
        return x

    def _calibrate_theta(self, target_on_frac, seed):
        """Pick the global threshold so the settled state has ~target_on_frac genes ON (0/1 free operating knob)."""
        x0 = self._seed(seed)
        best, best_gap = 0.0, 1e9
        for theta in np.linspace(-0.2, 0.6, 17):             # scan; higher theta => harder to turn on => fewer ON
            self.theta = float(theta)
            att, _ = self.run(x0, steps=200, tol=1e-4)
            on = float((att > 0.5).mean())
            gap = abs(on - target_on_frac)
            if gap < best_gap:
                best_gap, best = gap, float(theta)
        self.theta = best
        return best

    def baseline_attractor(self, seed=0, steps=400):
        x0 = self._seed(seed)
        att, info = self.run(x0, steps=steps)
        info["on_frac"] = float((att > 0.5).mean())
        return att, info

    # ---------------------------------------------------------------- perturbation
    def knockout_fragility(self, base_att, genes_local, restart_from_base=True, steps=200):
        """Clamp each gene OFF, re-run, return displacement ||A_ko - A_base|| (how much the cell-state moved)."""
        fr0 = base_att if restart_from_base else self._seed(0)
        out = {}
        for gl in genes_local:
            att, _ = self.run(fr0.copy(), steps=steps, clamp={gl: 0.0})
            out[gl] = float(np.linalg.norm(att - base_att))
        return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from complete_cell import CompleteCell
    C = CompleteCell()
    g = GRN(C=C)
    print(f"GRN: {g.n} genes in the dynamical core "
          f"({int(g.is_input.sum())} clamped inputs, {int((~g.is_input).sum())} dynamic)")
    print(f"calibrated theta = {g.theta:.3f}")
    att, info = g.baseline_attractor()
    print(f"baseline attractor: converged={info['converged']} in {info['steps']} steps, "
          f"ON-fraction={info['on_frac']:.1%}, final_delta={info['final_delta']:.1e}")
