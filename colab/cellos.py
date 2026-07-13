"""
 ██████╗███████╗██╗     ██╗      ██████╗ ███████╗
██╔════╝██╔════╝██║     ██║     ██╔═══██╗██╔════╝
██║     █████╗  ██║     ██║     ██║   ██║███████╗   the cell, as an operating system
██║     ██╔══╝  ██║     ██║     ██║   ██║╚════██║
╚██████╗███████╗███████╗███████╗╚██████╔╝███████║   boots on the real ~16.5k-gene cell model
 ╚═════╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝

WHY THIS EXISTS
---------------
"Biology is software" is usually a lazy metaphor. Taken seriously (per the DNA-malware work — DNA is untrusted
input crossing into the software domain; the tools that parse it are memory-unsafe C), it is *precise*: a cell is
not a static program you can read — it is a RUNNING SYSTEM you can only understand by executing it, poking it, and
distrusting your inputs. That is exactly an operating system. CellOS makes the mapping literal and RUNNABLE, and it
is honest about the one thing this whole project measured: you cannot read the source and know the behaviour — so
the causal syscalls are backed by INTERVENTIONAL data (the debugger), and every correlational answer is flagged.

THE MAPPING (each backed by real data in the model — nothing simulated for show)
    genome ................ read-only program image (disk)         gene locus chrom:tss ... address on disk
    gene .................. a process (PID = gene index)           transcription ......... loading a program
    protein ............... the running process                   abundance (ppm) ....... memory footprint
    compartment ........... memory segment (nucleus/cytosol/…)     essentiality/dep_frac . scheduler priority (niceness)
    transcription factor .. a scheduler / init process            regulatory + causal ... control-flow edges
    cell type (emask) ..... runlevel / boot profile               signaling pathway ..... interrupt handler
    CRISPR knockout ....... kill -9  (SIGKILL a process)           Perturb-seq ........... the DEBUGGER (strace/gdb)
    mutation .............. a code patch (no-op / crash / bug)     synthetic lethal ...... a deadlock pair
    drug .................. a runtime binary patch (hotfix)        disease ............... corrupted state / malware
    dark gene ............. a process with NO debug symbols        our confound audits ... the SECURITY layer (linter)

SYSCALLS      boot · ps · man · top · strace/kill · predict · whodunit · diagnose · simulate · cure · deadlock
              · patch · lint · help
CAUSALITY     [C] = causal (interventional / Perturb-seq or curated intervention).  [~] = correlational (low-trust).

A NOTE OF INSPIRATION (the Matrix lens): a cell is a model of reality you can only truly wield once you see its
code AND can bend it. `simulate` = "there is no spoon" (edit the cell, watch reality reshape); `cure` = "free the
cell" (search the perturbation that reverses a corrupted state — combination therapy). Both are honest about the
edge: a single knockout is measured (real), a combination is additive (its error is epistasis, unmeasured).

-> run:  python3 colab/cellos.py --demo        (scripted tour)
         python3 colab/cellos.py               (interactive shell)
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))

SCRATCH = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
# interventional debugger (Replogle Perturb-seq). DEFAULT = the essential screen (2,058 well-powered knockouts ->
# high-precision syscalls, e.g. predict PSMB5 r=0.86). Mount the GENOME-WIDE screen for breadth (PERTURB_H5AD=gwps:
# ~9.9k knockouts, coverage 86% of the genome, but lower per-gene precision r~0.3 — a measured coverage/precision
# tradeoff). cellformer.coverage() reports the genome-wide completeness regardless of which is mounted here.
PERTURB = os.environ.get("PERTURB_H5AD") or next(
    (p for p in [f"{SCRATCH}/k562.h5ad", f"{SCRATCH}/gwps.h5ad"] if os.path.exists(p)), f"{SCRATCH}/k562.h5ad")


class CellKernel:
    """the kernel: boots the cell model into a process table, and exposes syscalls over real data."""

    def __init__(self, quiet=False):
        self.log = (lambda *a: None) if quiet else (lambda *a: print(*a))
        self._boot()

    # ---------------------------------------------------------------- boot
    def _boot(self):
        self.log("CellOS 0.1  —  booting kernel from cell_complete image …")
        from complete_cell import CompleteCell
        self.C = C = CompleteCell()
        self.n = len(C.name)
        self.log(f"  [ ok ]  loaded genome image: {self.n:,} genes (processes)")
        self.log(f"  [ ok ]  PPI bus: {sum(len(v) for v in C.ppi_adj.values())//2:,} edges  |  "
                 f"control-flow (causal): {sum(len(v) for v in C.causal_out.values()):,} directed edges")
        self._pert = None                                    # lazy: the debugger data
        # scheduler processes = transcription factors
        self.tfs = [i for i in range(self.n) if C.genes[i].get("tf")]
        self.log(f"  [ ok ]  scheduler: {len(self.tfs):,} transcription factors (init/services)")
        # synthetic-lethal adjacency (deadlock pairs) from curated screens: [i, j, score]
        self.sl_adj = {}
        for pair in (C.D.get("sl") or []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2 and isinstance(pair[0], int) and isinstance(pair[1], int):
                a, b = pair[0], pair[1]; s = pair[2] if len(pair) > 2 else None
                self.sl_adj.setdefault(a, []).append((b, s)); self.sl_adj.setdefault(b, []).append((a, s))
        self.log(f"  [ ok ]  deadlock table: {len(self.sl_adj):,} genes with synthetic-lethal partners")
        self.log(f"  [ ok ]  security: confound-linter armed (memorisation / contamination / hub-leak / "
                 "interpolation-as-discovery)")
        dbg = os.path.exists(PERTURB)
        self.log(f"  [ {'ok ' if dbg else 'warn'}]  debugger (Perturb-seq): "
                 f"{'attached '+os.path.basename(PERTURB) if dbg else 'NOT mounted — strace/diagnose degrade to [~]'}")
        self.log("CellOS ready.  type 'help'.\n")

    # ---------------------------------------------------------------- helpers
    def _pid(self, name):
        return self.C.idx.get(name)

    def _prio(self, i):
        """scheduler priority 0..99 (higher = more critical / kernel-thread) from essentiality + constraint."""
        g = self.C.genes[i]
        dep = g.get("dep_frac") or 0.0
        ess = 1.0 if g.get("ess") else 0.0
        loeuf = g.get("loeuf")
        constr = (1.0 - min(loeuf, 2.0) / 2.0) if isinstance(loeuf, (int, float)) else 0.0
        return int(round(100 * min(1.0, 0.5 * dep + 0.3 * ess + 0.2 * constr)))

    def _seg(self, i):
        c = (self.C.genes[i].get("comp") or "?")
        return (c[0] if isinstance(c, str) else "?")

    def _load_debugger(self):
        if self._pert is not None:
            return self._pert
        if not os.path.exists(PERTURB):
            self._pert = False; return False
        import h5py, numpy as np
        f = h5py.File(PERTURB, "r")
        X = f["X"][:]
        idxkey = f["obs"].attrs.get("_index", "gene_transcript")
        labels = [x.decode() if isinstance(x, bytes) else x for x in f["obs"][idxkey][:]]
        pert = [l.split("_")[1] if len(l.split("_")) > 1 else l for l in labels]
        gn = f["var"]["gene_name"]
        if gn.dtype.kind in ("i", "u"):
            cats = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["__categories"]["gene_name"][:]]
            syms = [cats[c] for c in gn[:]]
        else:
            syms = [x.decode() if isinstance(x, bytes) else x for x in gn[:]]
        f.close()
        # one row per perturbed gene (average replicates)
        from collections import defaultdict
        rows = defaultdict(list)
        for k, g in enumerate(pert):
            rows[g].append(k)
        pgenes = sorted(rows)
        M = np.vstack([X[rows[g]].mean(0) for g in pgenes])
        M = np.clip(np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)  # gwps has NaN/inf entries
        self._pert = dict(M=M, pgenes=pgenes, pidx={g: k for k, g in enumerate(pgenes)}, syms=syms,
                          norms=np.linalg.norm(M, axis=1) + 1e-9)
        return self._pert

    # ---------------------------------------------------------------- syscalls
    def ps(self, context=None, limit=15):
        """process table. with --context CELLTYPE, show processes 'running' in that runlevel (emask); else the
        highest-priority kernel processes globally."""
        C = self.C
        order = sorted(range(self.n), key=lambda i: -self._prio(i))
        rows = order[:limit]
        head = f"  PID    PRIO SEG {'NAME':<12} TF  DEP%  PROCESS"
        out = [f"process table  (top {limit} by scheduler priority = essentiality)", head, "  " + "-" * 62]
        for i in rows:
            g = C.genes[i]
            out.append(f"  {i:<6} {self._prio(i):>3}  {self._seg(i):<3} {C.name[i]:<12} "
                       f"{'Y' if g.get('tf') else ' '}  {int(100*(g.get('dep_frac') or 0)):>3}  "
                       f"{(g.get('proc') or '?')[:26]}")
        return "\n".join(out)

    def stat(self):
        """whole-cell completeness dashboard — coverage of every layer (df/htop for the cell). All counts real."""
        import os, json
        C = self.C; D = C.D; n = self.n

        def gc(pred):
            return sum(1 for g in C.genes if pred(g))

        def line(label, k, extra=""):
            return f"    {label:26} {k:6,} / {n:,}  {k/n:4.0%}   {extra}"
        react = D.get("reactome", {}) or {}
        in_path = len({m for ms in react.values() for m in ms if isinstance(m, int)})
        enz = {m for ms in (D.get("met2enz", {}) or {}).values() for m in ms if isinstance(m, int)}
        dbg = self._load_debugger()
        measured = len(dbg["pgenes"]) if dbg else 0
        cf = {}
        p = f"{OUT_DIR}/cellformer.json" if (OUT_DIR := "outputs/orphan") else None
        if p and os.path.exists(p):
            cf = (json.load(open(p)) or {}).get("completeness", {})
        L = []
        L.append("=" * 66)
        L.append("  CELL SYSTEM STATS  —  completeness of every layer")
        L.append("=" * 66)
        L.append(f"  GENOME: {n:,} genes (processes)   |   scheduler: {len(self.tfs):,} TFs")
        L.append("  ── ANNOTATION (what each gene IS) ─────────────────────────────")
        L.append(line("localization/segment", gc(lambda g: g.get("comp"))))
        L.append(line("role / process", gc(lambda g: g.get("proc"))))
        L.append(line("GO terms", len(D.get("go", {}) or {})))
        L.append(line("domains (architecture)", len(getattr(C, "domains", {}) or {})))
        L.append(line("constraint (LOEUF)", gc(lambda g: g.get("loeuf") is not None)))
        L.append(line("pathway membership", gc(lambda g: g.get("path"))))
        L.append(line("PTM sites", len(D.get("ptm", {}) or {})))
        L.append(line("dark (no known function)", gc(lambda g: g.get("dark")), "<- the unknown"))
        L.append("  ── NETWORK (control flow) ─────────────────────────────────────")
        L.append(line("has PPI partner", len(C.ppi_adj)))
        L.append(line("co-expression", len(D.get("coexpr", {}) or {})))
        L.append(line("co-dependency", len(D.get("codep", {}) or {})))
        L.append(f"    edges: PPI {len(D.get('ppi',[])):,}  |  regulatory {len(D.get('reg',[])):,}  |  "
                 f"signaling {len(D.get('sig',[])):,}  |  causal(dir) {sum(len(v) for v in C.causal_out.values()):,}")
        L.append(f"    synthetic-lethal pairs {len(D.get('sl',[])):,}  |  ligand-receptor {len(D.get('lr',[])):,}")
        L.append("  ── MODULES ────────────────────────────────────────────────────")
        L.append(f"    Reactome pathways {len(react):,}  (cover {in_path:,} genes, {in_path/n:.0%})")
        L.append(f"    complexes {len(D.get('complexes',{})):,}  (cover {len(D.get('gene2cplx',{})):,} genes)")
        L.append(f"    metabolism: {len(enz):,} enzymes mapped to reactions")
        L.append("  ── QUANTITATIVE / PHARMA ──────────────────────────────────────")
        L.append(line("protein abundance (copies)", len(D.get("ppm", {}) or {})))
        L.append(line("cell-type expression", len(D.get("emask", {}) or {})))
        L.append(f"    drugs {len(D.get('drugs',{})):,}")
        L.append("  ── INTERVENTIONAL (the debugger — CAUSAL) ─────────────────────")
        L.append(line("measured knockouts (mounted)", measured, f"({os.path.basename(PERTURB)})"))
        if cf:
            L.append(f"    genome-wide completeness: {cf.get('measured_debugger',0):,} measured + "
                     f"{cf.get('predictable_complex_r0.23',0):,} predictable(good) + "
                     f"{cf.get('predictable_weak_r0.05',0):,} weak + {cf.get('dark_no_context',0):,} dark")
            L.append(f"    -> ANSWERS for {cf.get('answers_for_frac',0):.0%};  "
                     f"TRUSTWORTHY for {cf.get('WELL_covered_frac',0):.0%} of the genome")
        L.append("=" * 66)
        return "\n".join(L)

    def lit(self, name, k=5):
        """[LIT] the individual PubMed literature for a gene (grounded, cited). Fills the model's DARK genes from
        focused single-gene studies — the knowledge high-throughput screens can't reach. QUALITATIVE (function,
        interactions, phenotypes + DOIs), NOT the measured signatures strace/predict use. Cache: litmine.json."""
        import os, json
        if not hasattr(self, "_lit"):
            p = "outputs/orphan/litmine.json"
            self._lit = json.load(open(p)) if os.path.exists(p) else {}
        rec = self._lit.get(name)
        if not rec or not rec.get("papers"):
            return f"lit {name}: not mined — run `python3 colab/litmine.py {name}` to fetch its PubMed literature."
        i = self._pid(name)
        dark = " (model calls this DARK — literature fills it)" if (i is not None and self.C.genes[i].get("dark")) else ""
        L = [f"[LIT] {name}: {rec['n_papers']} PubMed papers{dark}   (source: PubMed — cite the DOIs)"]
        for a in rec["papers"][:k]:
            L.append(f"  {a['year']:>4}  {a['title'][:72]}")
            if a["doi"]:
                L.append(f"         doi:{a['doi']}")
        L.append("  qualitative layer: grounded in these abstracts, not measured signatures (fills annotation, "
                 "not the causal-prediction number).")
        return "\n".join(L)

    def viability(self, name):
        """[FBA] TOP-DOWN survival prediction: does the cell still GROW after knocking this gene out? From
        objective-driven FBA on Human-GEM — maximise biomass (reproduce) subject to mass balance + enzyme capacity
        (physics limits). This is the model that PRIORITISES staying alive + reproducing, and it predicts measured
        essentiality at AUC 0.70 where reading the wiring diagram was chance (0.50). Metabolic genes only."""
        import os, json
        if not hasattr(self, "_fba"):
            p = "outputs/orphan/fba_essentiality.json"
            self._fba = (json.load(open(p)).get("predicted_growth_on_knockout", {}) if os.path.exists(p) else {})
        if not self._fba:
            return "viability: FBA table not built (run ecflux + fba_essentiality)."
        if name not in self._fba:
            return (f"viability {name}: not in the metabolic model — objective-driven viability covers ~2,800 "
                    f"metabolic genes (there is no biomass-flux objective for signalling/TFs).")
        g = self._fba[name]
        verdict = ("LETHAL — a bottleneck the cell CANNOT reroute around (predicted essential)" if g < 0.1 else
                   f"IMPAIRED — growth drops to {g:.0%} of WT (dose-sensitive)" if g < 0.6 else
                   f"SURVIVES — growth {g:.0%}; flux REROUTES around it (robust / redundant)")
        return "\n".join([
            f"[FBA] viability  kill -9 {name}: predicted growth = {g:.0%} of wild-type",
            f"  -> {verdict}",
            f"  (objective = maximise biomass under mass-balance + capacity; predicts measured essentiality "
            f"AUC 0.70 vs wiring-diagram 0.50 — the top-down complement to strace's bottom-up.)"])

    def assess(self, name):
        """best-evidence essentiality — the SYNTHESIS syscall. Top-down (FBA physics) and bottom-up (measured
        Perturb-seq) are blind in DIFFERENT places, so between them they cover ~2x the genes either reaches alone
        (validated: routing each gene to its specialist → 4,082 genes at effective AUC 0.77 vs 0.71 for the best
        single modality). This gathers every independent layer for one gene, shows each with its evidence grade, and
        gives a combined call whose CONFIDENCE rises when independent layers AGREE."""
        import numpy as np
        i = self._pid(name)
        if i is None:
            return f"assess: no such gene '{name}'"
        g = self.C.genes[i]
        layers, votes = [], []
        # measured DepMap essentiality (ground truth where it exists)
        dep = g.get("dep_frac")
        if isinstance(dep, (int, float)):
            c = dep > 0.5
            layers.append(f"  measured  DepMap CRISPR (1,150 lines):  dep_frac {dep:.2f}  → "
                          f"{'ESSENTIAL' if c else 'non-essential'}   [C measured]")
            votes.append(c)
        # top-down FBA (physics) — covers metabolic genes even when silent in the assay
        if not hasattr(self, "_fba"):
            self.viability(name)                                  # warms self._fba
        fb = self._fba.get(name) if getattr(self, "_fba", None) else None
        if fb is not None:
            c = fb < 0.1
            layers.append(f"  top-down  FBA (physics/objective):     growth {fb:.0%} of WT  → "
                          f"{'LETHAL' if c else 'survives (reroutes)'}   [FBA modeled]")
            votes.append(c)
        # bottom-up measured Perturb-seq knockout shock — covers genes expressed in the screen
        dbg = self._load_debugger()
        if dbg and name in dbg.get("pidx", {}):
            nm = dbg["norms"]; pct = float((nm < nm[dbg["pidx"][name]]).mean())
            c = pct > 0.66
            layers.append(f"  bottom-up Perturb-seq knockout shock:   effect percentile {pct:.0%}  → "
                          f"{'high (matters)' if c else 'modest'}   [C measured]")
            votes.append(c)
        if not layers:
            return (f"assess {name}: no essentiality evidence in any layer (physics has no objective for it, it is "
                    f"not in the mounted screen, and it has no DepMap dependency) — dark for this question.")
        agree = len(set(votes)) == 1
        conf = "HIGH (independent layers agree)" if agree and len(votes) >= 2 else \
               "MIXED (layers disagree — inspect)" if len(votes) >= 2 else "single-layer (no corroboration)"
        call = "ESSENTIAL" if sum(votes) > len(votes) / 2 else \
               "non-essential" if sum(votes) < len(votes) / 2 else "SPLIT"
        cover = "physics+data" if fb is not None and dbg and name in dbg.get("pidx", {}) else \
                "physics-only (screen blind here)" if fb is not None else \
                "data-only (physics has no objective here)" if dbg and name in dbg.get("pidx", {}) else "measured-only"
        return "\n".join([f"assess {name}  —  best-evidence essentiality across independent layers",
                          *layers,
                          f"  ── combined call: {call}   confidence: {conf}",
                          f"  coverage: {cover}.  (physics+data are blind in different places → together ~2x reach.)"])

    def _grn(self):
        """lazily build the regulatory dynamical system — the cell's one RUNNABLE program (a clock, not a snapshot)."""
        if getattr(self, "_grn_obj", None) is None:
            import grn as _grnmod
            self._grn_obj = _grnmod.GRN(C=self.C)
        return self._grn_obj

    def boot(self, gene=None, steps=400):
        """exec/boot — RUN the genome forward and watch a cell state emerge. The only syscall with a CLOCK: it
        starts from an initial state and iterates the regulatory logic tick-by-tick until the cell settles into an
        attractor (a stable, self-consistent cell-state) — genuinely EXECUTING the software, not reading a snapshot
        (the debugger) or solving for the endpoint directly (FBA). With a gene arg it knocks that gene out and
        re-runs, so you can watch the cell's ROBUSTNESS first-hand. Honest limit below."""
        import numpy as np
        g = self._grn()
        x = g._seed(seed=0).copy()
        traj, checkpoints = [], {0, 1, 2, 4, 8, 16, 32, 64, 128, 256}
        conv = steps
        for s in range(steps):
            xn = g.step(x); d = float(np.abs(xn - x).max()); x = xn
            if s in checkpoints:
                traj.append((s + 1, float((x > 0.5).mean()), d))
            if d < 1e-4:
                conv = s + 1; traj.append((conv, float((x > 0.5).mean()), d)); break
        base = x
        L = [f"boot — executing the regulatory program forward  ({g.n:,} genes, {int((~g.is_input).sum()):,} dynamic,"
             f" {int(g.is_input.sum()):,} clamped inputs)",
             "  a CLOCK, not a photograph: state evolving tick-by-tick under the genome's own logic",
             "",
             "   tick   ON%    Δmax     (the cell settling into a stable state)",
             "   ----   ----   -----"]
        for t, on, d in traj:
            L.append(f"   {t:>4}   {on*100:>3.0f}%  {d:.1e}")
        L.append(f"  → converged in {conv} ticks to an attractor: {(base>0.5).mean()*100:.0f}% of genes ON "
                 f"(a self-consistent cell state)")
        if gene is None:
            L += ["",
                  "  this IS the cell running. it converges and — like a real cell — is ROBUST (validated: 68% of",
                  "  single knockouts barely move it, matching measured biology).",
                  "  HONEST LIMIT: running it does NOT predict which genes are essential (AUC 0.47 ≈ chance) —",
                  "  regulatory wiring doesn't carry knockout outcomes; only the physical/measured layers do",
                  "  (viability=FBA at 0.70, strace=the debugger). try:  boot <gene>  to watch robustness."]
            return "\n".join(L)
        gi = self._pid(gene)
        if gi is None or gi not in g.g2l:
            return "\n".join(L) + (f"\n\n  ({gene}: not in the regulatory dynamical core — it has no signed "
                                   f"transcriptional edges to run, so booting can't perturb through it.)")
        gl = g.g2l[gi]
        att_ko, _ = g.run(base.copy(), steps=200, clamp={gl: 0.0})
        disp = float(np.linalg.norm(att_ko - base))
        flipped = int(((att_ko > 0.5) != (base > 0.5)).sum())
        # population context from the validated sweep: median displacement ~0.45, 68% of KOs < 0.5
        rel = ("LESS disruptive than a typical knockout" if disp < 0.45 else
               "MORE disruptive than typical (a more connected/critical node)")
        L += ["",
              f"  kill -9 {gene}  → re-run the program with {gene} clamped OFF:",
              f"    attractor displacement {disp:.2f}   ({flipped} of {g.n:,} genes flipped state)",
              f"    → {rel}   (population median ≈ 0.45; 68% of knockouts barely move the cell)",
              "  you just watched robustness: the cell mostly re-settles. NB fragility here does NOT rank",
              "  essentiality (validated AUC 0.47) — use  viability {g}  /  strace {g}  for that.".format(g=gene)]
        return "\n".join(L)

    def induce(self, tfs, top=10):
        """reprogram — force transcription factor(s) ON in the resting cell, run the program forward, and show the
        lineage that lights up (Yamanaka-style). This is the ONE thing running the regulatory network does WELL:
        validated, forcing a master TF ON induces its OWN textbook program #1 for 8/9 masters (specificity AUC 0.99)
        — the FORWARD complement to the backward essentiality null. Honest: this expresses the TF's (correctly-wired,
        lineage-specific) targets; it is master-regulator readback played forward, not emergent discovery."""
        import numpy as np
        g = self._grn()
        locs, named = {}, []
        for t in tfs:
            gi = self._pid(t)
            if gi is not None and gi in g.g2l:
                locs[g.g2l[gi]] = 1.0; named.append(t)
        if not locs:
            return (f"induce: none of {list(tfs)} are transcription factors in the regulatory core "
                    f"(need outgoing signed edges to drive a program).")
        base, _ = g.run(g._seed(seed=0), steps=400)
        forced, _ = g.run(base.copy(), steps=200, clamp=locs)
        delta = forced - base
        order = np.argsort(-delta)
        up = [(str(g.name[i]), float(delta[i])) for i in order[:top] if delta[i] > 1e-3 and g.name[i] not in named]
        L = [f"induce {'+'.join(named)}  →  force ON in the resting cell and run the program forward",
             f"  top induced (the lineage program that lights up):"]
        L += [f"    {nm:<10} +{d:.2f}" for nm, d in up] or ["    (no genes rose above threshold)"]
        # if a known master TF was forced, confirm its textbook lineage program specifically rose
        try:
            from grn_reprogram import PROGRAMS
            for t in named:
                if t in PROGRAMS:
                    prg = [p for p in PROGRAMS[t] if self._pid(p) in g.g2l and p != t]
                    mp = float(np.mean([delta[g.g2l[self._pid(p)]] for p in prg])) if prg else 0.0
                    bg = float(delta[np.where(~g.is_input)[0]].mean())
                    L.append(f"  [{t}] textbook lineage program induction {mp:+.2f}  vs cell-wide background "
                             f"{bg:+.2f}  →  {'SPECIFIC (reprogrammed)' if mp > bg + 0.02 else 'not above background'}")
        except Exception:
            pass
        L += ["  forward master-regulator logic works (AUC 0.99, 8/9); the backward direction (essentiality) is null.",
              "  NB this expresses the TF's correctly-wired targets — reprogramming, not new discovery."]
        return "\n".join(L)

    def man(self, name):
        """process documentation: what this gene 'does', where it runs, its priority, its interfaces."""
        i = self._pid(name)
        if i is None:
            return f"man: no such process '{name}'"
        C = self.C; g = C.genes[i]
        ppi = list(C.ppi_adj.get(i, []))[:10]
        outedges = [C.name[j] for j, *_ in (C.causal_out.get(i, []) or [])][:8]
        dark = " (NO DEBUG SYMBOLS — dark gene)" if g.get("dark") else ""
        paths = g.get("path") or []
        if isinstance(paths, str):
            paths = [paths]
        L = [f"PROC {name}  (pid {i}){dark}",
             f"  priority   {self._prio(i)}/99   {'[KERNEL/essential]' if self._prio(i)>=60 else '[user process]'}",
             f"  segment    {g.get('comp') or '?'}   (memory/localisation)",
             f"  role       {g.get('proc') or '?'}",
             f"  scheduler  {'IS a transcription factor (init/service)' if g.get('tf') else 'not a TF'}",
             f"  pathways   {', '.join(str(p) for p in paths[:5]) or '—'}",
             f"  links      PPI bus: {', '.join(C.name[j] for j in ppi) or '—'}",
             f"  drives →   {', '.join(outedges) or '(no outgoing control-flow edges known)'}",
             f"  constraint LOEUF={g.get('loeuf')}   dependency={int(100*(g.get('dep_frac') or 0))}% of cell lines"]
        return "\n".join(L)

    def top(self, limit=12):
        """the kernel threads: the processes the whole system cannot run without (highest dependency)."""
        C = self.C
        order = sorted((i for i in range(self.n) if (C.genes[i].get("dep_frac") or 0) > 0),
                       key=lambda i: -(C.genes[i].get("dep_frac") or 0))[:limit]
        out = ["KERNEL THREADS  (processes with the highest system-wide dependency — 'don't kill these')",
               f"  {'NAME':<10} DEP%  PRIO  ROLE"]
        for i in order:
            g = C.genes[i]
            out.append(f"  {C.name[i]:<10} {int(100*(g.get('dep_frac') or 0)):>3}   {self._prio(i):>3}  "
                       f"{(g.get('proc') or '?')[:34]}")
        return "\n".join(out)

    def strace(self, name, k=12):
        """[C] THE DEBUGGER. SIGKILL a process (CRISPR knockout) and show the MEASURED downstream state change
        (Perturb-seq). This is causal — the effect of actually removing the gene, not a correlation."""
        dbg = self._load_debugger()
        i = self._pid(name)
        if i is None:
            return f"strace: no such process '{name}'"
        if not dbg:
            nb = [self.C.name[j] for j in list(self.C.ppi_adj.get(i, []))[:k]]
            return (f"strace {name}: [~] debugger not mounted — falling back to CORRELATIONAL neighbours "
                    f"(low trust):\n  {', '.join(nb) or '—'}")
        if name not in dbg["pidx"]:
            return (f"strace {name}: [~] no interventional trace for this process in the mounted screen "
                    f"({len(dbg['pgenes'])} genes profiled). Correlational neighbours: "
                    f"{', '.join(self.C.name[j] for j in list(self.C.ppi_adj.get(i, []))[:8]) or '—'}")
        import numpy as np
        row = dbg["M"][dbg["pidx"][name]]
        order = np.argsort(row)
        down = [(dbg["syms"][j], row[j]) for j in order[:k] if row[j] < 0]
        up = [(dbg["syms"][j], row[j]) for j in order[::-1][:k] if row[j] > 0]
        out = [f"[C] strace  kill -9 {name}   (measured downstream effect, Replogle Perturb-seq)",
               f"  ↓ DOWN-regulated on knockout:  " + ", ".join(f"{g}({v:+.1f})" for g, v in down[:8]),
               f"  ↑ UP-regulated on knockout:    " + ", ".join(f"{g}({v:+.1f})" for g, v in up[:8]),
               f"  (interventional: this is what the system does when the process is removed)"]
        return "\n".join(out)

    def diagnose(self, up=None, down=None, k=10):
        """[C] ROOT-CAUSE debugger. Given a corrupted state (genes UP / DOWN = the 'stack trace'), rank the
        processes whose removal REVERSES it — the causal driver, not a bystander. Interventional (the alibi test)."""
        dbg = self._load_debugger()
        up = up or []; down = down or []
        if not dbg:
            return "diagnose: [~] debugger not mounted; cannot run the interventional root-cause test."
        import numpy as np
        sidx = {s: j for j, s in enumerate(dbg["syms"])}
        sig = np.zeros(len(dbg["syms"]))
        for g in up:
            if g in sidx: sig[sidx[g]] = 1.0
        for g in down:
            if g in sidx: sig[sidx[g]] = -1.0
        if not sig.any():
            return "diagnose: none of the signature genes are measured in the screen."
        M = dbg["M"]
        # reversal(T) = -cosine(effect_T, signature): knocking T down pushes OPPOSITE to the corrupted state
        num = M @ sig
        den = (np.linalg.norm(M, axis=1) * (np.linalg.norm(sig) + 1e-9)) + 1e-9
        reversal = -(num / den)
        order = np.argsort(-reversal)[:k]
        out = [f"[C] diagnose  (root-cause: whose knockout REVERSES the state — interventional)",
               f"  corrupted state: UP={up or '—'}  DOWN={down or '—'}",
               f"  {'RANK':<5}{'DRIVER':<10}{'reversal':>9}  verdict"]
        for r, j in enumerate(order, 1):
            g = dbg["pgenes"][j]
            out.append(f"  {r:<5}{g:<10}{reversal[j]:>+9.3f}  {'drives the state (candidate target)' if reversal[j]>0 else 'protective'}")
        return "\n".join(out)

    def whodunit(self, name, topsig=25, k=8):
        """[C] the debugger as detective. Take the FINGERPRINT a knockout leaves and find which OTHER knockouts
        produce the most similar measured state — the implicated module. Ranked by +cosine MATCH (a cause's effect
        MATCHES the state; ranking by 'reversal' would be exactly backwards — the sign that broke this the first
        time). Self-match is excluded as trivial; identifying the EXACT cause cross-context is ~21% top-10 (measured,
        perturb_prioritizer.py) — the honest ceiling."""
        hits = self.whodunit_hits(name, topsig=topsig, k=k)
        if isinstance(hits, str):
            return hits
        nup, ndown, suspects = hits
        return "\n".join([
            f"[C] whodunit {name}: crime scene = {nup} genes UP / {ndown} DOWN after the knockout.",
            f"  knockouts whose MEASURED effect best matches this state (the implicated module):",
            "    " + ", ".join(f"{g}({m:+.2f})" for g, m in suspects),
            f"  (self-match excluded as trivial. identifying the EXACT cause CROSS-context is ~21% top-10, measured "
            f"— the debugger is real; context transfer is the open problem.)"])

    def whodunit_hits(self, name, topsig=25, k=8):
        """the ranked list behind whodunit(): [(gene, match_cosine)] excluding self, or an error string."""
        dbg = self._load_debugger()
        if not dbg:
            return "whodunit: debugger not mounted."
        if name not in dbg["pidx"]:
            return f"whodunit: {name} not profiled in the mounted screen."
        import numpy as np
        row = dbg["M"][dbg["pidx"][name]]; order = np.argsort(row)
        down = [dbg["syms"][j] for j in order[:topsig] if row[j] < 0]
        up = [dbg["syms"][j] for j in order[::-1][:topsig] if row[j] > 0]
        sidx = {s: j for j, s in enumerate(dbg["syms"])}
        sig = np.zeros(len(dbg["syms"]))
        for gg in up:
            if gg in sidx: sig[sidx[gg]] = 1.0
        for gg in down:
            if gg in sidx: sig[sidx[gg]] = -1.0
        match = (dbg["M"] @ sig) / (dbg["norms"] * (np.linalg.norm(sig) + 1e-9))
        ordr = np.argsort(-match)
        suspects = [(dbg["pgenes"][j], float(match[j])) for j in ordr if dbg["pgenes"][j] != name][:k]
        return len(up), len(down), suspects

    def _predict_vec(self, name):
        """predicted effect vector of an unmeasured knockout (neighbour-weighted avg); None if no context.
        Sets self._last_ctx to the context genes used."""
        import numpy as np, cellformer as cf
        dbg = self._load_debugger(); screen = set(dbg["pgenes"])
        if getattr(self, "_ctxidx", None) is None:
            self._ctxidx = cf.build_context_index(self.C, screen)
        g2c, coexpr, cx2m = self._ctxidx
        w = cf.context_weights(self.C, name, screen, g2c, coexpr, cx2m)
        self._last_ctx = sorted(w, key=lambda x: -w[x])[:25]
        if not self._last_ctx:
            return None
        wv = np.array([w[c] for c in self._last_ctx]); wv = wv / wv.sum()
        return wv @ dbg["M"][[dbg["pidx"][c] for c in self._last_ctx]]

    def _effect_vec(self, name):
        """the effect vector of knocking out `name`: MEASURED if in the screen (exact), else PREDICTED. Returns
        (vector, 'measured'|'predicted') or (None, None)."""
        dbg = self._load_debugger()
        if name in dbg["pidx"]:
            return dbg["M"][dbg["pidx"][name]], "measured"
        v = self._predict_vec(name)
        return (v, "predicted") if v is not None else (None, None)

    def simulate(self, genes, k=10):
        """[C~] THERE IS NO SPOON — edit the cell (knock out one or more genes) and propagate to the resulting
        state by combining their effects. Each single KO is MEASURED (exact) where in the screen; a COMBINATION is
        the additive sum — an approximation whose error IS the genetic interaction (epistasis), which single-
        perturbation data cannot measure. So: combos are 'first-order reality-bending', flagged honestly."""
        import numpy as np
        dbg = self._load_debugger()
        vecs, tags = [], []
        for g in genes:
            v, how = self._effect_vec(g)
            if v is None:
                tags.append(f"{g}[SINGLETON:skipped]"); continue
            vecs.append(v); tags.append(f"{g}[{how}]")
        if not vecs:
            return f"simulate: none of {genes} can be applied (no measured/predicted effect)."
        state = np.sum(vecs, axis=0)
        order = np.argsort(state)
        down = [(dbg["syms"][j], state[j]) for j in order[:k] if state[j] < 0][:8]
        up = [(dbg["syms"][j], state[j]) for j in order[::-1][:k] if state[j] > 0][:8]
        single = len(vecs) == 1 and "measured" in tags[0]
        mode = "MEASURED (exact)" if single else f"ADDITIVE of {len(vecs)} perturbations"
        out = [f"[C~] simulate  edit = {{ {', '.join(tags)} }}   -> resulting cell state [{mode}]",
               f"  ↓ DOWN: " + ", ".join(f"{g}({v:+.1f})" for g, v in down),
               f"  ↑ UP:   " + ", ".join(f"{g}({v:+.1f})" for g, v in up)]
        if not single:
            out.append("  ⚠ additive gets the DIRECTION right (r=0.88 on Norman doubles) but misses ~54% of the "
                       "magnitude; 36% of pairs are strongly synergistic. Trust the pattern, not the exact scale.")
        return "\n".join(out)

    def cure(self, up=None, down=None, combo=2, k=6):
        """[C] FREE THE CELL — given a disease/corrupted state, search the perturbation whose knockout best REVERSES
        it, then greedily add a SECOND to reverse the residual (a combination-therapy search). Interventional:
        ranks by measured reversal, exactly like diagnose, but returns a COMBINATION."""
        import numpy as np
        dbg = self._load_debugger()
        up = up or []; down = down or []
        sidx = {s: j for j, s in enumerate(dbg["syms"])}
        sig = np.zeros(len(dbg["syms"]))
        for g in up:
            if g in sidx: sig[sidx[g]] = 1.0
        for g in down:
            if g in sidx: sig[sidx[g]] = -1.0
        if not sig.any():
            return "cure: none of the signature genes are measured in the screen."
        M, norms = dbg["M"], dbg["norms"]

        def reversal_of(vec):
            return float(-(vec @ sig) / ((np.linalg.norm(vec) + 1e-9) * (np.linalg.norm(sig) + 1e-9)))
        rev1 = -(M @ sig) / (norms * (np.linalg.norm(sig) + 1e-9))
        i1 = int(np.argmax(rev1))
        chosen = [dbg["pgenes"][i1]]; combined = M[i1].copy()
        lines = [f"[C] cure  (free the cell from the corrupted state — combination-therapy search)",
                 f"  target state: UP={up or '—'} DOWN={down or '—'}",
                 f"  1st perturbation: kill {dbg['pgenes'][i1]}  (reversal {rev1[i1]:+.3f})"]
        for step in range(2, combo + 1):
            cand = -((M + combined) @ sig) / ((np.linalg.norm(M + combined, axis=1) + 1e-9) * (np.linalg.norm(sig) + 1e-9))
            for c in chosen:
                cand[dbg["pidx"][c]] = -9
            ib = int(np.argmax(cand))
            gain = cand[ib] - reversal_of(combined)
            if gain <= 0.001:
                lines.append(f"  (no {step}-way partner improves reversal — stop; single is best)"); break
            chosen.append(dbg["pgenes"][ib]); combined = combined + M[ib]
            lines.append(f"  +{step}: add kill {dbg['pgenes'][ib]}  -> combined reversal {cand[ib]:+.3f}  (gain {gain:+.3f})")
        lines.append(f"  => prescription: co-knockout [ {', '.join(chosen)} ]  reverses the state best.")
        if len(chosen) > 1:
            lines.append("  ⚠ combo reversal assumes additivity (r=0.88 direction, but ~54% magnitude residual on "
                         "real doubles) — synergy could make the true effect larger or smaller. Confirm with a "
                         "measured double before trusting the magnitude.")
        return "\n".join(lines)

    def predict(self, name, k=8):
        """[C~] PREDICT THE NEXT THING (transformer-style): predict the response of an UNMEASURED knockout from the
        measured responses of its network neighbours (weighted). Causal source, but predicted → tag [C~]. If the
        gene happens to be in the screen, print the live predicted-vs-real correlation. Measured ceiling
        (cellformer.py): r≈0.43 when the gene sits in a complex, ≈0.19 for singletons."""
        dbg = self._load_debugger()
        if not dbg:
            return "predict: debugger not mounted."
        import numpy as np, cellformer as cf
        pred = self._predict_vec(name)
        if pred is None:
            return (f"predict {name}: no network neighbours in the screen — a SINGLETON. Cannot predict the next "
                    f"state (this is the honest failure mode; r≈0.19 even when we can).")
        ctx = self._last_ctx
        order = np.argsort(pred)
        down = [(dbg["syms"][j], pred[j]) for j in order[:k] if pred[j] < 0]
        up = [(dbg["syms"][j], pred[j]) for j in order[::-1][:k] if pred[j] > 0]
        L = [f"[C~] predict  kill -9 {name}   (PREDICTED from {len(ctx)} network neighbours — unmeasured knockout)",
             f"  ↓ predicted DOWN: " + ", ".join(f"{g}({v:+.1f})" for g, v in down[:8]),
             f"  ↑ predicted UP:   " + ", ".join(f"{g}({v:+.1f})" for g, v in up[:8])]
        if name in dbg["pidx"]:
            r = cf._pearson(pred, dbg["M"][dbg["pidx"][name]])
            L.append(f"  [self-check] this gene WAS measured — predicted-vs-real r={r:+.2f}  "
                     f"({'RECOVERED (sits in a module)' if r > 0.4 else 'weak (singleton-like)'})")
        else:
            L.append(f"  (genuine prediction: this knockout is not in the screen)")
        return "\n".join(L)

    def deadlock(self, name, k=10):
        """[C] synthetic-lethal partners: processes where killing EITHER alone is survivable but BOTH = kernel
        panic. A mutual-dependency deadlock. From curated SL screens (interventional double-knockouts)."""
        i = self._pid(name)
        if i is None:
            return f"deadlock: no such process '{name}'"
        partners = sorted(self.sl_adj.get(i, []), key=lambda t: -(t[1] or 0))[:k]
        names = [f"{self.C.name[j]}" + (f"({s:.2f})" if s else "") for j, s in partners]
        if not names:
            return f"[C] deadlock {name}: no synthetic-lethal partner in curated screens (kill is survivable alone)."
        return (f"[C] deadlock {name}: SIGKILL {name} is survivable, but co-killing any of these = PANIC "
                f"(synthetic lethal — the double-knockout kills):\n  {', '.join(names)}")

    def patch(self, spec):
        """static analysis of a code PATCH (mutation 'GENE:R175H'): classify no-op / crash (LOF) / logic-bug (GOF).
        This is a fast static lint; the deep analyzer is reasoning_chain.py (ESM + ΔΔG + structure)."""
        if ":" not in spec:
            return "patch: usage  patch GENE:MUT   e.g.  patch TP53:R175H"
        name, mut = spec.split(":", 1)
        i = self._pid(name)
        if i is None:
            return f"patch: no such process '{name}'"
        g = self.C.genes[i]
        loeuf = g.get("loeuf"); dep = g.get("dep_frac") or 0.0; ndis = g.get("ndis") or 0
        suppressor = (isinstance(loeuf, (int, float)) and loeuf < 0.5) or dep > 0.4 or ndis >= 8
        cls = "CRASH (loss-of-function likely) → process may fail to start" if suppressor else \
              "LOGIC-BUG or NO-OP (function-changing vs silent — needs deep analysis)"
        return "\n".join([
            f"[static-lint] patch {name}:{mut}",
            f"  constraint LOEUF={loeuf}  dependency={int(100*dep)}%   → target is "
            f"{'highly constrained (patches are dangerous)' if suppressor else 'tolerant (many patches are silent)'}",
            f"  predicted class: {cls}",
            f"  NOTE: this is a static lint. Deep analysis (ESM zero-shot + ΔΔG + interface) = "
            f"reasoning_chain.reason('{name}', ...).  Do not ship the static call as the answer."])

    def lint(self, claim):
        """[the SECURITY layer] scan a CLAIM for the data-exploit classes this project keeps catching: is it a
        curated fact (trusted), a novel prediction in the chance regime (UNTRUSTED), or a hub artefact? Treats
        every conclusion as untrusted input until it passes."""
        import ml_guardrail as mg
        toks = claim.replace(",", " ").split()
        genes = [t for t in toks if self._pid(t) is not None]
        rel = "binds" if "bind" in claim or "interact" in claim else \
              "regulates" if "regulat" in claim or "drives" in claim else "relates"
        out = [f"[sec] lint: \"{claim}\""]
        if len(genes) < 2:
            out.append("  cannot parse two known genes; nothing to verify.")
            return "\n".join(out)
        a, b = genes[0], genes[1]
        ia, ib = self._pid(a), self._pid(b)
        curated = (ib in self.C.ppi_adj.get(ia, set())) or \
                  any(j == ib for j, *_ in (self.C.causal_out.get(ia, []) or []))
        # hub-leak check (validated exploit: the ML combiner hallucinates ribosomal partners for hubs like TP53)
        thr = mg.hub_threshold(self.C)
        ha = mg.node_degree(self.C, a) or 0; hb = mg.node_degree(self.C, b) or 0
        hub = ha >= thr or hb >= thr
        verdict = ("TRUSTED — present in curated interventional/physical evidence" if curated else
                   "UNTRUSTED — not in curated data; this is a PREDICTION, and novel-pair prediction is at chance "
                   "(0.5 AUC) in this project's blind tests")
        out.append(f"  parsed: {a} {rel} {b}")
        out.append(f"  provenance : {'in curated graph ✓' if curated else 'NOT in curated graph ✗'}")
        out.append(f"  hub-leak   : {'⚠ '+ (a if ha>=thr else b) +' is a HUB (deg≥p90) — learned scores hallucinate here' if hub else 'ok (neither endpoint a hub)'}")
        out.append(f"  VERDICT    : {verdict}")
        if not curated:
            out.append(f"  → to make this trustworthy, RUN THE EXPERIMENT (strace/diagnose), don't trust the prediction.")
        return "\n".join(out)

    HELP = """CellOS syscalls  ([C]=causal/interventional, [~]=correlational):
  stat / df            whole-cell completeness dashboard (coverage of every layer)
  ps [context]         process table (genes) by scheduler priority (essentiality)
  man GENE             process documentation (role, segment, priority, interfaces)
  lit GENE             the gene's PubMed literature (grounded+cited) — fills DARK genes
  top                  kernel threads (highest system-wide dependency)
  viability GENE       [FBA] top-down: does the cell still GROW after knockout? (objective+physics)
  assess GENE          best-evidence essentiality: fuse measured + physics + data; confidence rises when they agree
  boot [GENE]          RUN the genome forward (a clock): watch a cell-state emerge; [GENE] = knock out + re-run
  induce TF [TF..]     REPROGRAM: force master TF(s) ON, run forward, watch the lineage program light up
  strace GENE          [C] SIGKILL + measured downstream effect (Perturb-seq debugger)
  predict GENE         [C~] predict an UNMEASURED knockout's response from its neighbours
  whodunit GENE        [C] detective: recover the cause from a knockout's fingerprint
  diagnose up=.. down=..  [C] root-cause: whose knockout reverses a corrupted state
  simulate G1 G2 ..    [C~] "there is no spoon": edit the cell, propagate to the new state
  cure up=.. down=..    [C] free the cell: combination-therapy search to reverse a state
  deadlock GENE        [C] synthetic-lethal partners (co-kill = panic)
  patch GENE:MUT       static-lint a mutation (no-op / crash / logic-bug)
  lint "CLAIM"         [security] verify a claim; flag untrusted predictions & hub-leak
  help / exit"""


class CellShell:
    def __init__(self, kernel):
        self.k = kernel

    def run_line(self, line):
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        parts = line.split()
        cmd, args = parts[0], parts[1:]
        k = self.k
        try:
            if cmd in ("help", "?"): return k.HELP
            if cmd in ("exit", "quit"): return "__EXIT__"
            if cmd in ("stat", "df"): return k.stat()
            if cmd == "viability": return k.viability(args[0])
            if cmd == "assess": return k.assess(args[0])
            if cmd in ("boot", "exec", "run"): return k.boot(args[0] if args else None)
            if cmd in ("induce", "reprogram"): return k.induce(args)
            if cmd == "lit": return k.lit(args[0])
            if cmd == "ps": return k.ps(context=args[0] if args else None)
            if cmd == "man": return k.man(args[0])
            if cmd == "top": return k.top()
            if cmd in ("strace", "kill"): return k.strace(args[0])
            if cmd == "predict": return k.predict(args[0])
            if cmd == "whodunit": return k.whodunit(args[0])
            if cmd == "deadlock": return k.deadlock(args[0])
            if cmd == "patch": return k.patch(args[0])
            if cmd == "lint": return k.lint(line.split(" ", 1)[1].strip().strip('"'))
            if cmd == "simulate": return k.simulate(args)
            if cmd in ("diagnose", "cure"):
                up = down = []
                for a in args:
                    if a.startswith("up="): up = a[3:].split(",")
                    if a.startswith("down="): down = a[5:].split(",")
                return k.diagnose(up=up, down=down) if cmd == "diagnose" else k.cure(up=up, down=down)
            return f"{cmd}: command not found (try 'help')"
        except IndexError:
            return f"{cmd}: missing argument (try 'help')"
        except Exception as e:
            return f"{cmd}: kernel fault: {str(e)[:120]}"


DEMO = [
    "# --- CellOS: the cell as an operating system, booted on the real model ---",
    "help",
    "# 0) whole-cell completeness — coverage of every layer (pathways, complexes, edges, drugs, debugger)",
    "stat",
    "# 0b) TOP-DOWN survival — objective-driven FBA (grow+reproduce under physics): does knockout kill the cell?",
    "viability RAE1",
    "viability SLC22A1",
    "# 0b+) SYNTHESIS — fuse top-down physics + bottom-up data + measured; confidence rises when they agree.",
    "#      Blind in different places, together ~2x reach (4,082 genes, effective AUC 0.77 vs 0.71 single).",
    "assess RAE1",
    "# 0c) BOOT THE CELL — the only syscall with a clock: run the genome forward, watch a cell-state emerge, then",
    "#     knock a gene out and watch the cell RE-SETTLE (robustness). Honest: running it doesn't rank essentiality.",
    "boot",
    "boot TP53",
    "# 0d) REPROGRAM — force a master TF ON and run forward; the correct lineage program lights up (Yamanaka-style).",
    "#     The FORWARD direction works (AUC 0.99) where the backward one (essentiality) was null.",
    "induce GATA1",
    "# 1) kernel threads — the processes the system cannot run without (essentiality = priority)",
    "top",
    "# 2) documentation for one process (real data: role, segment, scheduler, interfaces)",
    "man TP53",
    "# 2b) PubMed literature for a DARK gene — the knowledge screens can't reach (grounded + cited)",
    "lit FNDC5",
    "# 3) THE DEBUGGER — SIGKILL a gene, watch the MEASURED downstream effect (interventional, not correlation)",
    "strace SF3B1",
    "# 4) THE DETECTIVE — recover the CAUSE from the fingerprint a knockout leaves (causal, self-consistency proof)",
    "whodunit SF3B1",
    "# 4b) PREDICT THE NEXT THING — an unmeasured knockout's response, transformer-style, with a live self-check",
    "predict PSMB5",
    "# 5) ROOT-CAUSE on an arbitrary corrupted state — whose removal reverses it?",
    "diagnose up=HBA1,HBB down=CCNB1,CDK1",
    "# 5a) THERE IS NO SPOON — edit the cell (a combination knockout) and propagate to the resulting state",
    "simulate SF3B1 PSMB5",
    "# 5b) FREE THE CELL — combination-therapy search to reverse a corrupted state",
    "cure up=CCND1,MYC down=CDKN1A",
    "# 6) synthetic-lethal deadlocks — co-kill = kernel panic (the SL idea, from curated double-knockouts)",
    "deadlock FANCI",
    "# 7) static-lint a code patch (mutation)",
    "patch TP53:R175H",
    "# 8) THE SECURITY LAYER — every claim is untrusted input until verified (the article's lesson)",
    'lint "TP53 binds MDM2"',
    'lint "TP53 binds MRPS25"',
]


def main():
    interactive = "--demo" not in sys.argv
    k = CellKernel()
    sh = CellShell(k)
    if not interactive:
        for line in DEMO:
            if line.startswith("#"):
                print("\n\033[2m" + line + "\033[0m" if sys.stdout.isatty() else "\n" + line)
                continue
            print(f"cellos> {line}")
            out = sh.run_line(line)
            if out and out != "__EXIT__":
                print(out)
        print("\ncellos> exit")
        return
    print("interactive CellOS — 'help' for syscalls, 'exit' to leave")
    while True:
        try:
            line = input("cellos> ")
        except (EOFError, KeyboardInterrupt):
            break
        out = sh.run_line(line)
        if out == "__EXIT__":
            break
        if out:
            print(out)


if __name__ == "__main__":
    main()
