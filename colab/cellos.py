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

    @staticmethod
    def _read_pert(path):
        """read a Replogle bulk h5ad -> (M [gene x measured], pgenes, syms), averaging guide replicates per gene."""
        import h5py, numpy as np
        f = h5py.File(path, "r")
        X = f["X"][:]
        idxkey = f["obs"].attrs.get("_index", "gene_transcript")
        if isinstance(idxkey, bytes):
            idxkey = idxkey.decode()
        labels = [x.decode() if isinstance(x, bytes) else x for x in f["obs"][idxkey][:]]
        pert = [l.split("_")[1] if len(l.split("_")) > 1 else l for l in labels]
        gn = f["var"]["gene_name"]
        if gn.dtype.kind in ("i", "u"):
            cats = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["__categories"]["gene_name"][:]]
            syms = [cats[c] for c in gn[:]]
        else:
            syms = [x.decode() if isinstance(x, bytes) else x for x in gn[:]]
        f.close()
        from collections import defaultdict
        rows = defaultdict(list)
        for k, g in enumerate(pert):
            rows[g].append(k)
        pgenes = sorted(rows)
        M = np.vstack([X[rows[g]].mean(0) for g in pgenes])
        M = np.clip(np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)  # gwps has NaN/inf entries
        return M, pgenes, syms

    def _load_debugger(self):
        if self._pert is not None:
            return self._pert
        import numpy as np
        # PRIMARY = the essential screen (high per-gene precision). EXTENSION = genome-wide gwps (breadth, lower
        # precision). Merging keeps essential-precision for its 2,058 genes and adds the other ~7,800 as measured
        # (if noisier) removals -> the causal syscalls reach ~9,867 pieces instead of 2,058, without degrading the
        # high-precision answers (essential rows win on overlap). Disable the merge with PERTURB_MERGE=0.
        primary = PERTURB if os.path.exists(PERTURB) else None
        ext = f"{SCRATCH}/gwps.h5ad"
        merge = os.environ.get("PERTURB_MERGE", "1") != "0" and os.path.exists(ext) and ext != primary
        if primary is None and not (os.path.exists(ext)):
            self._pert = False; return False
        if primary is None:                                   # only gwps available
            M, pgenes, syms = self._read_pert(ext); hi = set()
        else:
            M, pgenes, syms = self._read_pert(primary); hi = set(pgenes)
            if merge:
                gM, gpg, gsy = self._read_pert(ext)
                common = [s for s in syms if s in set(gsy)]    # shared response-gene columns
                ec = {s: j for j, s in enumerate(syms)}; gc = {s: j for j, s in enumerate(gsy)}
                eci = np.array([ec[s] for s in common]); gci = np.array([gc[s] for s in common])
                gpidx = {g: k for k, g in enumerate(gpg)}
                extra = [g for g in gpg if g not in hi]        # genes only in the genome-wide screen
                M = np.vstack([M[:, eci], gM[np.array([gpidx[g] for g in extra])][:, gci]])
                pgenes = pgenes + extra; syms = common
        self._pert = dict(M=M, pgenes=pgenes, pidx={g: k for k, g in enumerate(pgenes)}, syms=syms,
                          norms=np.linalg.norm(M, axis=1) + 1e-9, hi_prec=hi)
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

    def check(self, name, kcat):
        """put in a kcat and let the running cell SANITY-CHECK it against the flux it must carry. An enzyme's kcat
        (max turnover) must be >= the rate it operates at (kapp = FBA flux ÷ abundance) — a too-slow value can't
        carry the flux. HONEST caveat (kcat_invivo_validate.py, tested on 148 measured kcats vs an INDEPENDENT
        flux kapp from davidi_kcat.json): this is a WEAK, one-sided, aggregate signal, NOT a reliable per-enzyme
        verdict — the flux/abundance estimates are noisy (~92x), so ~30% of correct kcats sit below their own kapp
        and a 100x-too-slow value is caught only ~65% of the time. (Earlier 99%/0.93 claims were CIRCULAR: they
        used incell_rate = kcat*sigma; see kcat_invivo_validate.json.) So: a flag here is a HINT to re-check, not a
        proof of impossibility."""
        import json, os
        # non-circular operating rate: flux-derived kapp (NO kcat input), from the Drive davidi set
        if not hasattr(self, "_kapp"):
            p = "outputs/orphan/davidi_kcat.json"
            self._kapp = json.load(open(p)).get("davidi_kcat", {}) if os.path.exists(p) else {}
        rec = self._kapp.get(name)
        rate = rec.get("kcat_max_per_s") if rec else None
        if not isinstance(rate, (int, float)) or rate <= 0:
            return (f"check {name}: no independent flux-derived kapp for this enzyme (not flux-carrying in the "
                    f"NCI-60 FBA set) — can't sanity-check a kcat against a demand it doesn't have.")
        try:
            kcat = float(kcat)
        except (TypeError, ValueError):
            return "check: kcat must be a number (turnover per second, e.g. `check MGAT2 1.8`)."
        ratio = kcat / rate
        if kcat < rate:
            v = (f"FLAG (weak). kcat {kcat:g}/s is BELOW the flux-derived operating rate ({rate:.3g}/s, kapp = "
                 f"FBA flux ÷ abundance). If the kapp estimate is right, this enzyme can't carry its flux — re-check "
                 f"the value. But kapp is noisy (~30% of even CORRECT kcats land here), so treat as a hint, not proof.")
        elif ratio < 3:
            v = (f"TIGHT. kcat {kcat:g}/s is only {ratio:.1f}x the flux-derived rate ({rate:.3g}/s) — near saturation; "
                 f"plausible, little headroom.")
        else:
            v = (f"CONSISTENT. kcat {kcat:g}/s exceeds the flux-derived rate ({rate:.3g}/s, {ratio:.0f}x headroom) — "
                 f"no contradiction (can't rule out 'too fast' from this side).")
        return "\n".join([f"check {name}: proposed kcat = {kcat:g}/s  vs  flux-derived operating rate {rate:.3g}/s",
                          f"  → {v}",
                          "  (WEAK/one-sided signal: on 148 measured kcats the flag is ~70% consistent, 30% false-flag; "
                          "a hint to re-check, not a per-enzyme verdict — kcat_invivo_validate.json)"])

    def reason(self, name):
        """REASON across independent evidence lines to a conclusion with CALIBRATED confidence — the layer above the
        data. Validated (reason.py): weighing independent lines predicts essentiality at AUC 0.80 (> best single
        layer 0.75), and confidence is calibrated — P(essential) rises 10%→88% as more lines agree. Not one
        measurement, but the convergence of independent ones, and knowing when to trust it."""
        i = self._pid(name)
        if i is None:
            return f"reason: no such gene '{name}'"
        import numpy as np, json, os
        C = self.C; g = C.genes[i]
        lines = []                                              # (label, grade, vote 0/1, detail)
        dbg = self._load_debugger()
        if dbg and name in dbg.get("pidx", {}):
            nm = dbg["norms"]; pct = float((nm < nm[dbg["pidx"][name]]).mean())
            lines.append(("Perturb-seq shock", "measured", int(pct > 0.66), f"{pct:.0%} percentile"))
        if not hasattr(self, "_fba"):
            self.viability(name)
        fb = self._fba.get(name) if getattr(self, "_fba", None) else None
        if fb is not None:
            lines.append(("FBA viability", "modeled", int(fb < 0.1), f"growth {fb:.0%} of WT"))
        lo = g.get("loeuf")
        if isinstance(lo, (int, float)):
            lines.append(("LOEUF constraint", "genomic", int(lo < 0.35), f"LOEUF {lo}"))
        g2c = C.D.get("gene2cplx", {}) or {}
        inc = (i in g2c) or (str(i) in g2c)
        lines.append(("complex member", "structural", int(bool(inc)), "in a physical complex" if inc else "no complex"))
        od = len(C.causal_out.get(i, []) or [])
        lines.append(("network hub", "network", int(od >= 20), f"drives {od} genes"))
        comps = self._localization().get(name)              # L6 localization (science-run layer)
        if comps:
            prim = comps[0]
            if prim in ("Nucleus", "Mitochondrion"):
                lines.append(("localization", "compartment", 1, f"{prim} (essential-leaning)"))
            elif prim in ("Secreted", "Cell membrane", "Membrane"):
                lines.append(("localization", "compartment", 0, f"{prim} (non-essential-leaning)"))
        sd = self._string().get(name)                        # L7 STRING physical connectivity (science-run layer)
        if sd:
            d = sd.get("physical_degree", 0)
            lines.append(("STRING hub", "network", int(d >= self._string_thr()), f"{d} physical partners"))

        nyes = sum(v for _, _, v, _ in lines); navail = len(lines)
        # calibrated confidence from the validated calibration curve
        calib = {}
        p = f"outputs/orphan/reason.json"
        if os.path.exists(p):
            calib = {int(k): v["p_essential"] for k, v in json.load(open(p)).get("calibration", {}).items()}
        pe = calib.get(min(nyes, max(calib) if calib else 3)) if calib else nyes / max(navail, 1)
        L = [f"reason {name} — chaining independent evidence to a conclusion"]
        for lab, grade, v, det in lines:
            mark = "essential" if v else "—"
            L.append(f"  [{grade:<10}] {lab:<18} {det:<22} → {mark}")
        agree = nyes / max(navail, 1)
        if agree >= 0.5:
            concl, conf = "ESSENTIAL", "HIGH" if agree >= 0.6 else "MEDIUM"
        elif agree <= 0.2:
            concl, conf = "NON-ESSENTIAL", "HIGH" if agree == 0 else "MEDIUM"
        else:
            concl, conf = "UNCERTAIN (lines split — inspect)", "LOW"
        L.append(f"  ── {nyes} of {navail} independent lines say essential")
        L.append(f"  CONCLUSION: {concl}   confidence {conf}"
                 + (f"  (calibrated P≈{pe:.0%} at this agreement)" if calib else ""))
        L.append("  reasoning validated: weighing independent lines → AUC 0.89 > any single layer (0.75); "
                 "confidence calibrated 1%→67% (localization + STRING-PPI lines added from the science run).")
        return "\n".join(L)

    def perturb(self, name):
        """the cell as a RUNNING program: propagate a perturbation through the physical network and see the blast
        radius + checkpoint. Validated (biosim.py): the propagation recovers the functional MODULE (PSMB5→proteasome,
        RPL13→ribosome) — the near-field is real — but does NOT reproduce the whole transcriptional response
        (Spearman +0.02, downstream/regulatory, off the physical graph). Shows WHO is in the blast radius, not HOW
        the whole cell responds."""
        if not hasattr(self, "_bio"):
            try:
                from biosim import BioSim
                self._bio = BioSim(kernel=self)
            except Exception as e:
                self._bio = None; self._bio_err = str(e)[:80]
        if self._bio is None:
            return f"perturb: runtime unavailable ({getattr(self, '_bio_err', '?')})"
        return self._bio.run(name)

    def _string(self):
        """lazily load the STRING physical-PPI layer (string_degree.json)."""
        if not hasattr(self, "_str"):
            import json, os
            p = "outputs/orphan/string_degree.json"
            self._str = json.load(open(p)).get("layer", {}) if os.path.exists(p) else {}
        return self._str

    def _string_thr(self):
        if not hasattr(self, "_strthr"):
            import numpy as np
            d = np.array([v.get("physical_degree", 0) for v in self._string().values()], float)
            d = d[d > 0]
            self._strthr = float(np.percentile(d, 75)) if len(d) else 1e9
        return self._strthr

    def ppi(self, name):
        """top physical interaction PARTNERS from STRING v12.0 (the science-run network layer)."""
        sd = self._string().get(name)
        if not sd or not sd.get("partners"):
            return f"ppi {name}: no STRING physical interactions on record."
        ps = ", ".join(f"{p['gene']}({p['score']})" for p in sd["partners"][:8])
        return (f"ppi {name}: {sd.get('physical_degree', 0)} physical partners; top by confidence: {ps}\n"
                f"  (STRING v12.0 physical/curated channels; connectivity lifts reason 0.86→0.89)")

    def disease(self, name):
        """a SECOND reasoning engine (disease_reason.py): how likely is this gene DISEASE-associated? Reasoned from
        independent lines (constraint, complex, hub, pleiotropy, localization) — a DIFFERENT axis from essentiality
        (they overlap only 11% vs 9%). HONEST: modest — fused AUC 0.65 (vs essentiality 0.86); useful as a CALIBRATED
        PRIOR (P rises 22%→62% with agreement), not a strong classifier. Disease-association is a harder axis."""
        import json, os
        if not hasattr(self, "_dis"):
            p = "outputs/orphan/disease_reason.json"
            self._dis = json.load(open(p)).get("priors", {}) if os.path.exists(p) else {}
        r = self._dis.get(name)
        if not r or r.get("p_disease") is None:
            return f"disease {name}: too few independent lines to reason a disease prior."
        known = "  (already a known disease gene)" if r["known_disease"] else ""
        return (f"disease {name}: {r['n_agree']}/{r['n_lines']} independent lines agree → "
                f"calibrated P(disease-associated) ≈ {r['p_disease']:.0%}{known}\n"
                f"  (a second reasoner_core engine; MODEST AUC 0.65 — a prior for triage, not a strong call)")

    def _localization(self):
        """lazily load the per-gene subcellular compartments (localization.json)."""
        if not hasattr(self, "_loc"):
            import json, os
            p = "outputs/orphan/localization.json"
            self._loc = json.load(open(p)).get("labels", {}) if os.path.exists(p) else {}
        return self._loc

    def loc(self, name):
        """WHERE in the cell does this protein sit? subcellular compartment(s) from reviewed UniProt (the
        science-run layer). Compartment carries essentiality signal (nuclear/mito enriched, secreted/membrane
        depleted) and lifts the reason engine 0.80→0.86."""
        comps = self._localization().get(name)
        if not comps:
            return f"loc {name}: no reviewed-UniProt subcellular annotation."
        lean = ("essential-leaning" if comps[0] in ("Nucleus", "Mitochondrion")
                else "non-essential-leaning" if comps[0] in ("Secreted", "Cell membrane", "Membrane") else "mixed")
        return f"loc {name}: {', '.join(comps[:4])}  (primary {comps[0]}, {lean})"

    def mutate(self, args):
        """run the cell's reasoning WITH a mutation. Fuses the VARIANT layer (does it break the protein? —
        AlphaMissense call + ΔΔG/functional-site mechanism + a sickle-cell-style gain-of-function override) with
        the CELL layer (does the cell depend on it? — reason, AUC 0.80, calibrated). NAMES the regime, because
        cell-fitness essentiality and disease pathogenicity are DIFFERENT axes (they agree for loss-of-function in
        load-bearing genes, diverge for tumor-suppressors / GOF / tissue-specific).
        usage: mutate GENE UNIPROT POS WT MUT   e.g.  mutate TP53 P04637 175 R H   (needs net for AlphaMissense)."""
        if len(args) < 5:
            return "mutate GENE UNIPROT POS WT MUT   e.g.  mutate TP53 P04637 175 R H"
        try:
            gene, up, pos, wt, mut = args[0], args[1], int(args[2]), args[3].upper(), args[4].upper()
        except ValueError:
            return "mutate: POS must be an integer.  usage: mutate GENE UNIPROT POS WT MUT"
        if not hasattr(self, "_mr"):
            from reason_mutation import MutationReasoner
            self._mr = MutationReasoner(kernel=self)     # reuse this kernel for the cell layer (no second boot)
        o = self._mr.reason(gene, up, pos, wt, mut, verbose=False)
        v, c = o["variant_layer"], o["cell_layer"]
        return "\n".join([
            f"mutate {o['mutation']}  —  reasoning a variant through the whole cell",
            f"  A) VARIANT : {v['call'].upper()}  (AlphaMissense {v['score']})  — {', '.join(v['mechanisms'])}",
            f"       why: {v['why']}",
            f"  B) CELL    : {c['essentiality']}" + (f"  (calibrated P≈{c['p_essential']:.0%})" if c['p_essential'] is not None else "")
            + "   — does the cell depend on this protein?",
            f"  ── REGIME  : {o['regime']}",
            f"     → {o['cell_level_conclusion']}",
            f"     joint confidence: {o['joint_confidence']}",
            "  (fuses two validated reasoners; essentiality is a cell-FITNESS lens, not pathogenicity — regimes named)"])

    def level(self, name):
        """WHERE in its pathway does this gene act? metabolic STEP (tier-1, ordered substrate chain) or signaling
        upstream→downstream TIER (tier-2, SIGNOR directed graph + SCC feedback handling). Honest by construction:
        genes in a feedback loop, or whose level differs across pathways, are FLAGGED — not given a fake number.
        (cell_levels.py: ~5,300 genes get a trustworthy level — ~750 metabolic KEGG steps + ~4,500 signaling
        tiers — the rest flagged context-dependent/feedback or abstained.)"""
        import json, os
        if not hasattr(self, "_levels"):
            p = "outputs/orphan/cell_levels.json"
            self._levels = json.load(open(p)).get("labels", {}) if os.path.exists(p) else {}
        r = self._levels.get(name)
        if not r:
            return f"level {name}: not a pathway-membership gene in the map — no pathway level to give."
        st = r["status"]
        if st == "metabolic-step":
            return (f"level {name}: METABOLIC — {r['detail']} in {r['pathway']} "
                    f"({r['bucket']}, level {r['level']}) — an ordered substrate chain  [{r['source']}]")
        if st == "signaling-tier":
            return (f"level {name}: SIGNALING {r['bucket'].upper()} — {r['detail']}; e.g. {r['pathway']}  "
                    f"[{r['source']}]")
        if st == "context-dependent":
            return (f"level {name}: CONTEXT-DEPENDENT — {r['detail']}. No single level; it depends which pathway "
                    f"you mean (a hub acting at different points).")
        if st == "feedback-module":
            return (f"level {name}: FEEDBACK MODULE — {r['detail']} (e.g. {r['pathway']}). Mutually-regulating "
                    f"genes have no upstream/downstream — the honest answer is 'a loop', not a rank.")
        return f"level {name}: pathway member but no orderable context — abstaining (no trustworthy level)."

    def needs(self):
        """the software's own BILL OF MATERIALS: what it still needs to reach the known complete-cell state, tagged
        DATA (a findable measurement — where a data hunt pays), METHOD (need an algorithm), or HARD (biology has no
        single answer — must stay flagged, not faked). Grounded in the committed coverage artifacts (needs.py)."""
        import json, os
        n = json.load(open("outputs/orphan/needs.json")) if os.path.exists("outputs/orphan/needs.json") else None
        if not n:
            return "needs: run needs.py first to compute the gap manifest."
        tag = {"DATA": "[DATA]", "METHOD": "[METHOD]", "HARD": "[HARD]", "METHOD/HARD": "[METH/HD]"}
        out = ["needs — what the cell software still needs to reach the complete state:"]
        for r in n["rows"]:
            out.append(f"  {tag.get(r['limiter'],'[?]'):8} {r['layer']}: {r['gap']}")
            out.append(f"           → {r['closes']}")
        out.append(f"  ── {n['headline']}")
        return "\n".join(out)

    def investigate(self, args):
        """build a PREDICTED investigative dossier for a gene we lack a full record for — the way you profile a person
        from partial records: textbook RECORD, FAMILY (co-dependent + interacting relatives), DESTINATION (compartment),
        PATH (pathway), TIMING (pathway tier), INTERACTORS (PPI), SURVEILLANCE (measured or predicted removal effect),
        and the REQUIRED SUPPORT its job implies (translocation, folding, transport, energy). Validated: family-consensus
        recovers a held-out gene's JOB ~55% and DESTINATION ~54% (≈2× the majority baseline) for the closest relatives.
        'investigate underworld' lists DARK genes embedded in a coherent characterized module (hidden crew: EMC7 in the
        EMC complex, MTG2 in mito-translation, NCLN in the ER Nicalin complex).
        usage: investigate GENE | investigate underworld"""
        if not hasattr(self, "_inv"):
            import investigate as _I
            self._invmod = _I
            self._inv = _I.Investigator(kernel=self)
        if args and args[0].lower() == "underworld":
            uw = self._inv.underworld(k=15)
            out = ["investigate underworld — DARK genes embedded in a coherent characterized module (hidden crew):"]
            for u in uw:
                tag = "measured" if u["measured"] else "unmeasured"
                out.append(f"  {u['dark_gene']:12} → {u['predicted_job']}/{u['predicted_place']} "
                           f"(coherence {u['coherence']}, {tag}) — crew: {', '.join(u['crew'][:4])}; "
                           f"likely role: {u['likely_role']}")
            return "\n".join(out)
        if not args:
            return "investigate GENE   (or 'investigate underworld')"
        g = args[0]
        if g not in self._inv.name2i:
            return f"investigate {g}: no such gene in the cell image."
        dark_or_unlabeled = self._inv._g(g).get("dark") or not self._inv._proc(g)
        return self._invmod.render(self._inv.profile(g, predict_mode=dark_or_unlabeled))

    def network(self, args):
        """the whole cell wired into ONE multi-layer graph — INTERCONNECTED (physical: ppi/complex/ligand-receptor)
        and INTERDEPENDENT (functional: causal/regulatory/signaling/co-dependency/co-expression/synthetic-lethal) —
        queryable in the running software (network.py). The two families are complementary (a functional edge is
        who-NEEDS-whom, ~94% not a physical edge) yet coherent (still 46x the chance rate of recovering a real
        complex). Edges are observed/curated relationships, not predictions.
          network GENE           — the gene's complete wiring, split physical vs functional
          network link A B       — every relationship (and its direction) between two genes
          network path A B [fam] — a shortest chain (fam = interconnected | interdependent)
          network hubs [fam]     — the most-wired genes in a family
        usage: network GENE | network link A B | network path A B [interconnected|interdependent] | network hubs [fam]"""
        if not hasattr(self, "_net"):
            import network as _N
            self._net = _N.CellNetwork(kernel=self)
        G = self._net
        if args and args[0].lower() == "wire" and len(args) > 1:
            args = args[1:]                                   # 'network wire GENE' -> wire GENE
        sub = (args[0].lower() if args else "")
        if sub == "hubs":
            fam = args[1] if len(args) > 1 else "interconnected"
            fam = "interdependent" if fam.startswith("interd") else "interconnected"
            hs = G.hubs(fam, 15)
            return f"network hubs ({fam}):\n  " + ", ".join(f"{h['gene']}({h['degree']})" for h in hs)
        if sub == "link":
            if len(args) < 3:
                return "network link A B"
            lk = G.link(args[1], args[2])
            if lk["status"] != "ok":
                return f"network link {args[1]} {args[2]}: unknown gene."
            if not lk["relationships"]:
                return f"{args[1]} and {args[2]}: no wired relationship in any layer."
            out = [f"network link {lk['a']} ↔ {lk['b']} — {lk['n_links']} relationship(s):"]
            for r in lk["relationships"]:
                out.append(f"  [{r['family']:15}] {r['layer']:16} {r['direction']}  (w={r['w']})")
            return "\n".join(out)
        if sub == "path":
            if len(args) < 3:
                return "network path A B [interconnected|interdependent]"
            fam = args[3] if len(args) > 3 else "interconnected"
            fam = "interdependent" if fam.startswith("interd") else "interconnected"
            p = G.path(args[1], args[2], family=fam)
            if p["status"] != "ok":
                return f"network path {args[1]} -> {args[2]} ({fam}): {p['status']}"
            return f"network path ({fam}, {p['hops']} hops): " + " -> ".join(p["path"])
        if not args:
            import network as _N
            n_ic = sum(G.layer_edges(L) for L in _N.INTERCONNECTED)
            n_id = sum(G.layer_edges(L) for L in _N.INTERDEPENDENT)
            return (f"network — {G.n:,} genes wired into one graph:\n"
                    f"  INTERCONNECTED (physical)  {n_ic:,} edges: " + ", ".join(_N.INTERCONNECTED) + "\n"
                    f"  INTERDEPENDENT (functional) {n_id:,} edges: " + ", ".join(_N.INTERDEPENDENT) + "\n"
                    f"  try: network GENE | network link A B | network path A B [fam] | network hubs [fam]")
        w = G.wire(args[0])
        if w["status"] != "ok":
            return f"network {args[0]}: no such gene."
        out = [f"network — {w['gene']} wired into the cell (total degree {w['degree']['total']}):",
               "  INTERCONNECTED (who it physically touches):"]
        if not w["interconnected"]:
            out.append("    (none recorded)")
        for L, v in w["interconnected"].items():
            out.append(f"    {L:16}: " + ", ".join(x["gene"] for x in v))
        out.append("  INTERDEPENDENT (who it needs / controls):")
        if not w["interdependent"]:
            out.append("    (none recorded)")
        for L, v in w["interdependent"].items():
            if isinstance(v, dict):
                for side, lst in v.items():
                    out.append(f"    {L}.{side:11}: " + ", ".join(x["gene"] for x in lst))
            else:
                out.append(f"    {L:16}: " + ", ".join(x["gene"] for x in v))
        return "\n".join(out)

    def knockout(self, args):
        """the MEASURED downstream effect of removing a gene — its Perturb-seq blast radius (what actually goes up/down),
        for the ~9,871 knocked-out genes (dependency.py). Distinct from the wired `network` (curated edges): this is the
        real measured transcriptome response. HONEST: the direct effect is real data, but it does NOT propagate — you
        cannot chain it to compute an unmeasured knockout's cascade (transitivity ~chance).
          knockout GENE        — the measured up/down blast radius of removing GENE
          knockout impact      — genes ranked by measured blast-radius size (a load-bearing score)
        usage: knockout GENE | knockout impact"""
        if not hasattr(self, "_dep"):
            import dependency as _D
            self._dep = _D.Dependency(kernel=self)
        D = self._dep
        if not args:
            return "knockout GENE   (or 'knockout impact')"
        if args[0].lower() == "impact":
            imp = D.impact_ranking(20)
            out = ["knockout impact — genes ranked by measured blast-radius size (# genes strongly perturbed):"]
            for r in imp:
                tag = "" if r["hi_precision"] else "  (genome-wide screen; lower per-gene precision)"
                out.append(f"  {r['gene']:10} {r['genes_perturbed']:5} genes moved{tag}")
            return "\n".join(out)
        g = args[0]
        ko = D.knockout(g, topn=10)
        if ko["status"] != "measured":
            return f"knockout {g}: {ko.get('note', 'not knocked out in the screen — try a measured gene')}"
        prec = "high-precision (essential screen)" if ko["hi_precision"] else "genome-wide screen (lower precision)"
        imp = D.impact_of(g)
        out = [f"knockout {g} — MEASURED blast radius ({prec}); {ko['n_dependents_strong']} genes strongly moved"
               + (f", top {imp['percentile']:.0%} impact" if imp else "") + ":",
               "  goes DOWN (needs " + g + "): " + (", ".join(f"{x}({v})" for x, v in ko["down"]) or "—"),
               "  goes UP (released by " + g + "): " + (", ".join(f"{x}({v})" for x, v in ko["up"]) or "—")]
        return "\n".join(out)

    def protein(self, args):
        """knock out the PROTEIN, not the gene. `knockout` deletes the gene and measures the transcriptional response;
        this DEGRADES the protein (PROTAC-style) — the structural effect: a protein is a shared component wired into
        several machines, so removing it pulls it from EVERY complex at once, leaving the co-subunits expressed but
        their assembly incomplete. Grounded: complex parts are co-essential (46% share a top co-dependency vs 0.04%
        chance); the transcriptional measurement is mostly BLIND to it (post-transcriptional) except where a machine
        is feedback-wired (proteasome, spliceosome) — which is why this structural view is not the measured knockout.
        usage: protein GENE   (alias: degrade GENE)"""
        if not hasattr(self, "_pko"):
            import protein_knockout as _P
            self._pko = _P.ProteinKnockout(kernel=self)
        if not args:
            return "protein GENE   (degrade the protein product — disassemble its machines)"
        d = self._pko.degrade(args[0], topn=10)
        if d["status"] != "ok":
            return f"protein {args[0]}: no such protein in the cell image."
        p = d["protein"]
        if d["n_machines"] == 0:
            return (f"degrade {p} — not a curated subunit of any complex, so no machine-level disassembly to show. "
                    f"{d['n_interactions_severed']} physical interactions would be severed: "
                    f"{', '.join(d['interactions_severed'][:10]) or '—'}. (For its transcriptional effect use "
                    f"'knockout {p}'.)")
        out = [f"degrade {p} (PROTEIN knockout) — {d['n_machines']} machine(s) lose a subunit, "
               f"{d['n_orphaned_subunits']} co-subunits orphaned, {d['n_interactions_severed']} interactions severed:"]
        for m in d["machines_disrupted"]:
            out.append(f"  ✗ {m['complex']}")
            out.append(f"      orphaned co-subunits: {', '.join(m['orphaned_co_subunits']) or '—'}")
        c = d["measured_confirmation"]
        if c and c.get("mates_measured"):
            out.append(f"  measured check: complex-mates move {c['fold']}x the average gene in {p}'s own Perturb-seq "
                       f"knockout — {'strong feedback' if c['fold'] >= 1.5 else 'weak (protein KO is post-transcriptional)'}"
                       f"; top: {', '.join(f'{s}({v})' for s, v in c['top_moved_mates'][:5])}")
        out.append(f"  (vs 'knockout {p}' = the measured TRANSCRIPTIONAL response of deleting the gene.)")
        return "\n".join(out)

    def cascade(self, args):
        """use the perturbation data as a directed INFLUENCE network (what a knockout AFFECTS) and reconstruct the
        ROUTE of each effect from what we know — without predicting any unmeasured effect. `cascade X` stages X's whole
        blast radius; `cascade X P` reconstructs the single route KO(X)->P. Each effect is DIRECT (X physically/causally
        contacts P), MEASURED-MEDIATED (a specific non-hub stepping-stone M that is itself a measured knockout: X->M->P,
        both hops real data), or UNRESOLVED. Honest, measured on this screen: ~1% direct (6.5x chance), ~14% mediated
        (43x a placebo), ~85% unresolved (the far-field that doesn't compose). The influence network is 100% real; the
        route is reconstructable for a validated minority.
        usage: cascade GENE | cascade GENE TARGET"""
        if not hasattr(self, "_infl"):
            import influence as _I
            self._infl = _I.InfluenceNetwork(kernel=self)
        G = self._infl
        if not args:
            return "cascade GENE   (or 'cascade GENE TARGET' to reconstruct one route)"
        X = args[0]
        if X not in G.pidx:
            return f"cascade {X}: not knocked out in the screen (no measured influence to trace)."
        if len(args) >= 2:                                        # explain a single route X -> P
            P = args[1]; e = G.explain(X, P)
            if e["status"] == "not-measured":
                return f"cascade {X} -> {P}: {P} was not measured."
            if e.get("route") == "no-effect":
                return f"cascade {X} -> {P}: no strong measured effect (Δ={e['value']})."
            if e["route"] == "direct":
                return f"cascade {X} -> {P}: DIRECT ({e['via']}), stage 1  (Δ={e['value']})."
            if e["route"] == "mediated":
                s = e["stepping_stone"]
                return (f"cascade {X} -> {P}: MEASURED-MEDIATED, stage 2 — route {e['path']}  (Δ={e['value']}; "
                        f"X→M {s['x_to_m']}, M→P {s['m_to_p']}, M is a measured non-hub knockout).")
            return (f"cascade {X} -> {P}: UNRESOLVED (Δ={e['value']}) — no direct contact or specific measured "
                    f"stepping-stone; distal/regulatory/diffuse.")
        c = G.cascade(X)                                          # stage the whole blast radius
        out = [f"cascade {X} — measured influence network, route reconstructed for each effect:",
               f"  {c['n_affected']} gene-products affected → {c['n_direct']} DIRECT entry, {c['n_mediated']} "
               f"MEASURED-MEDIATED, {c['n_unresolved']} UNRESOLVED  ({c['explained_frac']:.0%} explained)"]
        if c["stage1_direct"]:
            out.append("  STAGE 1 (direct contacts — where the effect enters):")
            for d in c["stage1_direct"][:6]:
                out.append(f"    {d['gene']:10} via {d['via']:20} Δ={d['value']}")
        if c["stage2_mediated"]:
            out.append("  STAGE 2 (via a measured stepping-stone M — X→M→P, both hops real):")
            for m in c["stage2_mediated"][:6]:
                out.append(f"    {m['gene']:10} via {m['via']:20} Δ={m['value']}")
        out.append(f"  ({c['n_unresolved']} effects unresolved — far-field that doesn't compose; honestly unrouted.)")
        return "\n".join(out)

    def pertseq(self, args):
        """the four canonical Perturb-seq applications attempted on this data (K562), each validated & honestly graded
        (pertseq_apps.py). 'pertseq' shows the scorecard; 'pertseq function GENE' does a live fingerprint lookup — the
        KNOWN genes whose knockout response most matches GENE's (the validated 84%/4.1x pathway-recovery use).
          [1] GRN wiring    — does NOT work on this data (co-response, not causal; needs time-resolved data)
          [2] gene function — WORKS by transcriptomic fingerprint (84%, 4.1x chance)
          [3] disease conv. — method validated, but thin + wrong cell context
          [4] immune logic  — not feasible (needs +/- stimulus axis and immune cells)
        usage: pertseq | pertseq function GENE"""
        if args and args[0].lower() == "function":
            if len(args) < 2:
                return "pertseq function GENE"
            if not hasattr(self, "_psa"):
                import pertseq_apps as _P
                self._psa = _P.PertSeqApps(kernel=self)
            r = self._psa.nearest_known(args[1])
            if r["status"] != "ok":
                return f"pertseq function {args[1]}: not knocked out in the screen (no fingerprint)."
            tag = " [dark]" if r["is_dark"] else ""
            out = [f"pertseq function {r['gene']}{tag} — nearest KNOWN genes by KO fingerprint "
                   f"(consensus: {r['consensus_role']}, {r['consensus_support']}):"]
            for x in r["neighbours"]:
                out.append(f"    {x['gene']:10} cos {x['cos']:<5} {str(x['role'])[:52]}")
            out.append("  (a validated SIMILARITY hint — 84% share a pathway, 4.1x chance — not proof; a shared "
                       "growth-arrest phenotype can mimic a shared pathway.)")
            return "\n".join(out)
        import json, os
        p = "outputs/orphan/pertseq_apps.json"
        if not os.path.exists(p):
            return "pertseq: run pertseq_apps.py first to compute the four-application scorecard."
        d = json.load(open(p))
        g = d["grn"]; f = d["fingerprint_function"]; di = d["disease_convergence"]
        kk = [x for x in f if x.startswith("validation")][0]
        spl = di["positive_controls"].get("spliceosom", {}).get("z")
        return "\n".join([
            "PERTURB-SEQ — the four canonical applications on this K562 data, honestly graded:",
            f"  [1] GRN WIRING          DOES NOT (on this data): inferred TF->TF edges recover curated direct "
            f"regulation only {g['curated_fold']}x chance, {g['sign_agreement_on_recovered']:.0%} sign (=chance).",
            f"                          co-response network, not causal wiring — needs time-resolved / nascent-RNA data.",
            f"  [2] GENE FUNCTION       WORKS: fingerprint-kNN recovers a held-out gene's pathway {f[kk]:.0%} "
            f"({f['lift']}x chance); {f['n_dark_confident_gt0.3cos']} confident dark-gene calls.  ('pertseq function GENE')",
            f"  [3] DISEASE CONVERGENCE method validated (spliceosome control z={spl}), but thin: "
            f"{di['n_diseases_testable']} disease testable here, and K562 is the wrong context for neuro/immune disease.",
            f"  [4] IMMUNE BRAKES/GAS   NOT feasible: needs a +/- stimulus axis and immune cells; this is one "
            f"unstimulated leukemia condition.",
            "  -> one works, one is a validated hint, one is a method awaiting the right data, one is out of scope."])

    def xcell(self, args):
        """CROSS-CELL-TYPE check using the second screen (Xaira X-Atlas/Orion HCT116) against K562: which of a
        knockout's responders reproduce in BOTH cell types (cell-type-ROBUST — the trustworthy core of a cascade) vs
        only one (cell-type-specific). Validated: strong effects agree at Pearson ~0.5-0.6 across the two cancer types
        (vs ~0.04 shuffled). Needs hct116.h5ad staged (python3 colab/fetch_xaira.py).
        usage: xcell GENE"""
        if not hasattr(self, "_xc"):
            try:
                import crosscell as _X
                self._xc = _X.CrossCell(kernel=self)
            except (FileNotFoundError, RuntimeError) as e:
                self._xc = None; self._xc_err = str(e)
        if self._xc is None:
            return f"xcell: second cell type not staged — {self._xc_err}"
        if not args:
            return "xcell GENE   (cross-cell-type robustness of a knockout's effect)"
        c = self._xc.compare(args[0])
        if c["status"] == "not-in-both":
            return f"xcell {args[0]}: measured in {c['measured_in']} only — need it in both K562 and HCT116 to compare."
        if c["status"] != "ok":
            return f"xcell {args[0]}: {c['status']}"
        g = c["gene"]
        out = [f"xcell {g} — K562 vs HCT116 (consistency r={c['consistency']}):",
               f"  ROBUST (moved in BOTH cell types, same direction — {c['n_robust']}): the trustworthy core",
               "    " + ", ".join(f"{s}(K{k}/H{h})" for s, k, h in c["robust"][:8])]
        out.append(f"  K562-specific ({c['n_K562_only']}): " + ", ".join(c["K562_only"][:8]))
        out.append(f"  HCT116-specific ({c['n_hct116_only']}): " + ", ".join(c["hct116_only"][:8]))
        out.append("  (a responder confirmed in both cell types is a property of the gene, not the cell line.)")
        return "\n".join(out)

    def fieldsim(self, args):
        """the four-layer continuous 'protein-field' whole-cell engine (fieldsim.py), assembled end-to-end and tested
        HONESTLY: SH field overlap (L1) + mutation->ΔΔG->γ (L2) + SDF reaction-diffusion PDE (L3) + Hill source (L4),
        run as a closed loop. It RUNS and is internally consistent, but the tests confirm it is a MECHANISTIC
        DEMONSTRATOR, not a validated predictor (the SH-translation seam is real; γ=exp(ΔΔG/RT) diverges; and per every
        measured result here the whole-cell mutation->phenotype map does not compose). 'fieldsim run' executes it live.
        usage: fieldsim | fieldsim run"""
        if args and args[0].lower() == "run":
            import fieldsim as _F
            _F.main()
            return "(fieldsim ran live — see output above)"
        if args and args[0].lower() in ("validate", "test", "measured"):
            import fieldsim_test as _T
            _T.main()
            return "(fieldsim measured-validation ran live — see output above)"
        import json, os
        p = "outputs/orphan/fieldsim.json"
        if not os.path.exists(p):
            return "fieldsim: run 'fieldsim run' (or python3 colab/fieldsim.py) to build the four-layer engine first."
        d = json.load(open(p))
        g = d.get("L2_gamma", {})
        t = json.load(open("outputs/orphan/fieldsim_test.json")) if os.path.exists("outputs/orphan/fieldsim_test.json") else None
        out = [
            "fieldsim — four-layer protein-field engine, assembled & tested (a mechanistic DEMONSTRATOR, not a predictor):",
            f"  L1 overlap: coefficient-contraction == direct integral to {d.get('L1_overlap_rel_err',0):.1%}  (co-centred only)",
            f"  L1 SEAM: a co-centred source is 1 coefficient but needs ~121 once 0.34 off-centre -> diffusing the SH",
            f"           object is the wrong move (re-centring is an O(L^3) FMM translation).",
            f"  L2 gamma: spec exp(ΔΔG/RT) explodes ({g.get('10.0',{}).get('spec_exp','?')} at ΔΔG=10) -> bounded form used",
            f"  L3 SDF gate: field 100% confined without an export motif vs 28% with one (boundary works)",
            f"  L4 Hill cascade: TF knockout collapses the enzyme field {d.get('L4_tf_knockout_enzyme_drop',0):.0%}"]
        if t:
            out += [
            "  --- vs MEASURED Perturb-seq (the real test, 'fieldsim validate') ---",
            f"  predicting which genes move: AUC {t['median_auc_fieldsim']} (chance 0.5); static-network baseline {t['median_auc_static_targets']}",
            f"  -> the dynamics add NOTHING over topology; reaches only {t['median_recall_of_movers']:.0%} of real movers (huge false reach)"]
        out += [
            "  VERDICT: runs end-to-end & internally consistent, but its knockout prediction does NOT beat chance or a",
            "           trivial network lookup — long-range dynamics don't compose (transitivity ~0.009). Study tool, not",
            "           a predictor; the measured 'knockout'/'cascade' engines remain the trustworthy ones.",
            "  ('fieldsim run' = build+physics tests live; 'fieldsim validate' = score vs measured live.)"]
        return "\n".join(out)

    def latent(self, args):
        """test the load-bearing assumption of a Mechanistic-Latent-Mechanistic architecture BEFORE wiring a real
        foundation model: can a Geneformer/scGPT-style latent predict a held-out knockout's response, and does it beat
        'predict the mean perturbation response'? (latent_bridge_test.py). Result: barely (+0.018 AUC), and for NOVEL
        perturbations — the claimed moat — it is WORSE than the mean; the wall relocates into latent space, it isn't
        crossed. 'latent run' executes it live.
        usage: latent | latent run"""
        if args and args[0].lower() in ("run", "test"):
            import latent_bridge_test as _L
            _L.main()
            return "(latent-bridge test ran live — see output above)"
        import json, os
        p = "outputs/orphan/latent_bridge_test.json"
        if not os.path.exists(p):
            return "latent: run 'latent run' (or python3 colab/latent_bridge_test.py) first."
        d = json.load(open(p)); nov = d["by_novelty"].get("novel (far from training)", {})
        return "\n".join([
            "latent — does a foundation-model latent beat 'predict the mean' on held-out knockouts? (M-L-M Phase 2 test)",
            f"  predict which genes move (AUC, chance 0.5):  LATENT {d['auc_latent']}  vs  MEAN baseline {d['auc_mean']}  "
            f"(gain {d['latent_gain']})",
            f"  correlation on moved genes:                  LATENT {d['pear_latent']}  vs  MEAN {d['pear_mean']}",
            f"  NOVEL perturbations (the claimed moat):      LATENT {nov.get('auc_latent')}  vs  MEAN {nov.get('auc_mean')}  "
            f"(gain {nov.get('latent_gain')} — worse or flat)",
            "  -> the latent bridge is ~the generic mean response; it helps only when a close twin is already in training,",
            "     and does NOT help for novel perturbations. Phase 2 relocates the transitivity wall, it doesn't cross it.",
            "  ('latent run' to execute live.)"])

    def pstruct(self, args):
        """structural dossier for the steps of a known pathway (glycolysis) from REAL data (pathway_struct.py): per
        ordered step — gene/isoenzymes, reaction (substrate->product, before/after), protein FAMILY/fold, residue-level
        ACTIVE-SITE + binding residues (UniProt), PPI partners (cell DB), oligomeric state + allostery, and an AlphaFold
        structural analysis with a cross-checked INDUCED-FIT (domain-closure) signal (HK1/PGK1 top; TPI loop-closure
        honestly missed). The near-field structural layer that this session found trustworthy.
        usage: pstruct   (regenerate/extend with: python3 colab/pathway_struct.py)"""
        import json, os
        p = "outputs/orphan/pathway_struct.json"
        if not os.path.exists(p):
            return "pstruct: run 'python3 colab/pathway_struct.py' first to build the pathway structural dossier."
        d = json.load(open(p))
        out = [f"pstruct — {d['pathway']} step-by-step structural dossier (gene/reaction/family/active-site/PPI/induced-fit):"]
        for s in d["steps"]:
            st = s.get("structure") or {}
            fit = " [DOMAIN-CLOSURE induced fit]" if st.get("domain_closure_induced_fit") else ""
            allo = " [allosteric]" if s.get("activity_regulation") else ""
            asr = ", ".join(f"{(desc or 'cat')}@{pos}" for pos, desc in (s.get("active_sites") or [])[:3]) or "—"
            out.append(f"  {s['step']:>2}. {s['gene']:6} {s['rxn']:38} [{s['family'] or '?'}]{fit}{allo}")
            out.append(f"        active site: {asr};  PPI deg {s.get('ppi_degree')};  "
                       f"{'essential' if s.get('essential') else 'non-essential'}")
        out.append("  induced-fit ranking: HK1 & PGK1 (domain-closure) top; PGAM1 demoted (compact); TPI1 loop-closure missed.")
        return "\n".join(out)

    def enzyme(self, args):
        """RESEARCH finding: what governs metabolic enzyme essentiality? (enzyme_patterns.py, multidimensional over
        2,511 enzymes, held-out). Two independent axes: CENTRALITY (in a complex / many pathways / PPI hub / specialist)
        is dominant, and PARALOG BUFFERING (lacking a sequence-family isoenzyme backup) is a real orthogonal second
        axis. Full model AUC 0.862 vs null 0.51. The run self-corrected: a confounded reaction-redundancy proxy gave a
        WRONG answer until true sequence-family paralogs were measured.
        usage: enzyme | enzyme run"""
        if args and args[0].lower() in ("run", "rerun"):
            import enzyme_patterns as _E
            _E.main()
            return "(enzyme-patterns research ran live — see output above)"
        import json, os
        p = "outputs/orphan/enzyme_patterns.json"
        if not os.path.exists(p):
            return "enzyme: run 'enzyme run' (or python3 colab/enzyme_patterns.py) first."
        d = json.load(open(p)); u = d["univariate"]
        top = sorted(u.items(), key=lambda kv: -abs(kv[1]["effect"]))[:5]
        out = [f"enzyme — what makes a metabolic enzyme essential? ({d['n_enzymes']:,} enzymes, held-out AUC "
               f"{d['auc_full']} vs null {d['auc_null']}):",
               "  dimensions by effect size (essential vs not):"]
        for f, v in top:
            arrow = "↑" if v["effect"] > 0 else "↓"
            out.append(f"    {f:18} {arrow} effect {v['effect']:+.2f}  (AUC alone {d['solo_auc'][f]})")
        out += ["  TWO INDEPENDENT AXES:",
                "    1. CENTRALITY (dominant): in a complex, many pathways, PPI hub, and a SPECIALIST (fewer reactions)",
                "    2. PARALOG BUFFERING (real): essential enzymes lack a sequence-family isoenzyme backup (median 0 vs 2)",
                "  self-correction: the reaction-redundancy proxy was CONFOUNDED and reversed buffering's sign until",
                "                   true sequence families were fetched — a confident wrong answer, fixed by measurement."]
        return "\n".join(out)

    def interface(self, args):
        """RESEARCH: the pattern INSIDE PPIs at the amino-acid level (interface_hotspots.py) — which interface residues
        carry a PPI so a mutation there changes binding. From SKEMPI 2.0 (4,956 MEASURED single-point interface
        mutations, 345 complexes): the buried CORE carries the energy (mean |ΔΔG| 1.92, 38% hotspots) vs rim/surface
        (~0); hotspot residues are the large/charged/aromatic ones (L,K,W,Y,F,R,D); a mutation acts ~68% by
        destabilising the bound state and ~32% by raising the association barrier; and ΔΔG is predictable from residue
        change + interface depth at complex-held-out r 0.36 (triage, not exact). The near-field structural handle on the
        single-nucleotide-mutation problem.
        usage: interface | interface run"""
        if args and args[0].lower() in ("run","rerun"):
            import interface_hotspots as _I; _I.main(); return "(interface-hotspot research ran live — see above)"
        import json, os
        p="outputs/orphan/interface_hotspots.json"
        if not os.path.exists(p): return "interface: run 'interface run' (or python3 colab/interface_hotspots.py) first."
        d=json.load(open(p)); bp=d["by_position"]; k=d["kinetics"]
        out=[f"interface — which amino acids carry a PPI, and will a mutation there change it? ({d['n_mutations']:,} measured, {d['n_complexes']} complexes):",
             "  POSITION carries the energy (mean |ΔΔG| kcal/mol, %hotspot>2):"]
        for loc in ("core","support","rim","surface"):
            if loc in bp: out.append(f"    {loc:8} {bp[loc]['mean_abs_ddg']:>5}  ({bp[loc]['frac_hotspot_gt2']:.0%} hotspot)")
        out+=["  HOTSPOT residues (mutating them hurts most): "+", ".join(d["alanine_hotspot_residues"][:6])+" (aromatic/charged)",
              f"  a mutation weakens binding {1-k['barrier_dominated_frac']:.0%} via bound-state STABILITY, {k['barrier_dominated_frac']:.0%} via association BARRIER",
              f"  PREDICT ΔΔG from residue-change + interface-depth: complex-held-out r {d['predict_ddg_pearson']} (triage, not exact; headroom with full structure)",
              "  -> flag load-bearing interface positions (core, aromatic/charged) where a variant will change the interaction."]
        return "\n".join(out)

    def flex(self, args):
        """RESEARCH: the physics ΔΔG-of-binding node (flex_physics.py) with clash relief, sp3 ROTAMER sampling and
        ELECTROSTATICS + INDUCTION, tested on measured SKEMPI. Rigid Lennard-Jones r^-12 EXPLODES on bulky interface
        mutations; jumping to the lowest-clash sp3 rotamer makes the energy physical (explosions 33%->4%) — the honest
        fix for the deeply-buried clash minimisation alone can't solve. Chemistry (Coulomb + Debye induction) adds real
        held-out signal on polar/charged mutations: the full physics+research stack reaches held-out ΔΔG r ~0.52.
        usage: flex | flex run | flex rosetta"""
        if args and args[0].lower() in ("run","rerun"):
            import flex_physics as _F; _F.main(); return "(flex-physics ran live — see above)"
        if args and args[0].lower() in ("rosetta","compare","vs"):
            import flex_vs_rosetta as _R; _R.main(); return "(flex-vs-Rosetta comparison ran live — see above)"
        import json, os
        p="outputs/orphan/flex_physics.json"
        if not os.path.exists(p): return "flex: run 'flex run' (or python3 colab/flex_physics.py) first."
        d=json.load(open(p)); a=d["alanine"]; tb=d["to_bigger"]; ef=tb["explosion_frac"]; sp=tb["spearman"]
        out=["flex — physics ΔΔG-of-binding (clash relief + sp3 rotamers + electrostatics/induction) on measured SKEMPI:",
             "  [1] rigid Lennard-Jones EXPLODES on a bulky (->Trp) interface mutation (1e6-1e10 kcal/mol) — the r^-12 wall.",
             "  [3] clash relief on larger-residue mutations (explosion rate / rank-corr vs measured):",
             f"      rigid {ef['rigid']:.0%} expl (r {sp['rigid']:+.2f});  soft-core {ef['softcore']:.0%} (r {sp['softcore']:+.2f});  "
             f"ROTAMER {ef.get('rotamer',0):.0%} (r {sp.get('rotamer',0):+.2f});  flex {ef['flex']:.0%} (r {sp['flex']:+.2f})",
             "      -> rotamer sampling fixes the PHYSICS (physical energies, 4% explosions) but the raw clash magnitude still RANKS best.",
             f"  [2] PATCH w/ research + chemistry: held-out ΔΔG r {a['r_empirical']} -> +sterics {a.get('r_steric','?')} "
             f"-> +electrostatics/induction {a['r_combined']} (chem +{a.get('chem_gain_over_steric','?')}, total +{a['physics_gain']}).",
             "  net: rotamers fixed the buried-clash physics; electrostatics+induction added real held-out accuracy on polar/charged muts."]
        return "\n".join(out)

    def screen(self, args):
        """RESEARCH: the ΔΔG node as a PARTNER-SCREENING pass (ppi_screen.py) — given a mutation, score it against a
        panel of candidate partners to flag LOSS of a known PPI at the right partner (and spare the ones it doesn't
        touch). Honest decomposition: LOCALISATION (which partner) is near-exact but that's structure-reading;
        MAGNITUDE (how much) needs the full ΔΔG model — burial alone is r~0 across substitutions; NOVEL-partner GAIN is
        NOT claimed (needs docking + experiment).
        usage: screen | screen run"""
        if args and args[0].lower() in ("run", "rerun"):
            import ppi_screen as _S; _S.main(); return "(ppi-screen ran live — see above)"
        import json, os
        p = "outputs/orphan/ppi_screen.json"
        if not os.path.exists(p): return "screen: run 'screen run' (or python3 colab/ppi_screen.py) first."
        d = json.load(open(p)); L = d["localisation"]; M = d["magnitude"]; G = d["gain"]
        return "\n".join([
            "screen — ΔΔG node as a PPI partner-screening pass, decomposed honestly:",
            f"  LOCALISATION (which partner): top-1 {L['top1_partner']:.0%} vs {L['random_baseline']:.0%} random "
            f"— reliable, but it's near-field STRUCTURE (reading geometry), not the hard part.",
            f"  MAGNITUDE (how much): burial ALONE r {M['burial_vs_absddg_all_r']:+.2f} across substitutions (~0 — ignores "
            f"what you mutate to); controlled X->Ala r {M['burial_vs_absddg_ala_r']:+.2f}; core |ΔΔG| {M['core_absddg']} "
            f"vs surface {M['surface_absddg']}. Magnitude needs the full ΔΔG model (~{M['full_ddg_model_pearson']}).",
            f"  GAIN: {G['affinity_increasing_frac']:.0%} of muts increase affinity at existing interfaces; NOVEL-partner "
            f"gain NOT claimed (needs docking + experiment).",
            "  net: a good LOSS/specificity screen (localise damage to the right partner x score it with the ΔΔG model); honest NO on new partners."])

    def surface(self, args):
        """RESEARCH: a REAL trained MaSIF-style surface-complementarity encoder (surface_fingerprint.py) — the honest
        version of the surface-DL idea, built from REAL coordinates/chemistry of REAL complexes with a REAL contrastive
        net, validated on HELD-OUT complexes. Two-sided result: learns interface-patch complementarity (discrimination
        AUC ~0.66 vs 0.50 untrained -- training earns it) but exact partner-patch retrieval is only WEAKLY above chance
        (residue-level features, no MSMS surface / APBS electrostatics -> coarser than real MaSIF).
        usage: surface | surface run"""
        if args and args[0].lower() in ("run", "rerun"):
            import surface_fingerprint as _S; _S.main(); return "(surface-fingerprint trained live — see above)"
        if args and args[0].lower() in ("apbs", "msms", "electrostatics"):
            import surface_apbs as _A; _A.main(); return "(surface-APBS controlled run ran live — see above)"
        import json, os
        p = "outputs/orphan/surface_fingerprint.json"
        if not os.path.exists(p): return "surface: run 'surface run' (or python3 colab/surface_fingerprint.py) first."
        d = json.load(open(p))
        out = [
            "surface — REAL trained surface-complementarity encoder (not the random-weight mock):",
            f"  data: {d['n_patches']:,} REAL patches from {d['n_complexes']} interface-labelled complexes; held out BY complex.",
            f"  WORKS: discrimination AUC {d['held_out_auc']} vs {d['untrained_baseline_auc']} untrained baseline "
            f"(training earns +{d['training_gain']}) — real, generalising surface-complementarity signal.",
            f"  MOSTLY DOESN'T: exact partner-patch retrieval top-1 {d['retrieval_top1']:.1%} (chance {d['retrieval_chance_top1']:.1%}), "
            f"top-5 {d['retrieval_top5']:.1%} (chance {d['retrieval_chance_top5']:.1%}) — only weakly above chance.",
            "  limit: residue-level features; coarse run has NO electrostatics/surface (see 'surface apbs')."]
        pa = "outputs/orphan/surface_apbs.json"
        if os.path.exists(pa):
            a = json.load(open(pa))
            out.append(f"  + APBS/SURFACE (real pdb2pqr->APBS PB + marching-cubes surface, controlled on {a['n_complexes']} complexes): "
                       f"AUC {a['auc_coarse']} -> {a['auc_apbs_surface']} ({a['gain']:+.3f}) — real electrostatics+surface HELP ('surface apbs').")
        pd_ = "outputs/orphan/dmasif.json"
        if os.path.exists(pd_):
            m = json.load(open(pd_))
            out.append(f"  + GEODESIC (dMaSIF: surface POINTS + learned geodesic conv, {m['n_complexes']} complexes): AUC "
                       f"{m['discrimination_auc']} (vs residue-patch {m['residue_patch_auc']}); retrieval {m['retrieval_lift_top5']}x chance — see 'dmasif'.")
        return "\n".join(out)

    def regsign(self, args):
        """RESEARCH: the REGULATORY-SIGN annotation (regsign.py) — the piece that lets the structural sensors reach
        GAIN-of-function. Sensors detect an interface BREAK; the sign says whether that interface is an ON switch or an
        OFF switch. Break an INHIBITORY/autoinhibitory brake -> GOF (activity UP), not LOF. Built from UniProt activity-
        regulation text, tested vs proto-oncogene(GOF)/tumor-suppressor(LOF) labels: when a brake is annotated it calls
        GOF at 86% precision (5.4x enrichment) — high-precision, recall-limited (annotation-sparse). Wires into NEXUS via
        directed_activity(inhibitory=True). usage: regsign | regsign run"""
        if args and args[0].lower() in ("run", "rerun"):
            import regsign as _R; _R.main(); return "(regsign ran live — see above)"
        import json, os
        p = "outputs/orphan/regsign.json"
        if not os.path.exists(p): return "regsign: run 'regsign run' (or python3 colab/regsign.py) first."
        d = json.load(open(p))
        return "\n".join([
            "regsign — regulatory-sign annotation (breakable brake => GAIN-of-function lever):",
            f"  built from UniProt activity-regulation text; tested vs {d['n_oncogene']} proto-oncogenes (GOF) / {d['n_tsg']} tumor-suppressors (LOF).",
            f"  WHEN THE SIGN FIRES (brake annotated): PRECISION {d['precision_brake_is_gof']:.0%} are GOF "
            f"({d['n_flagged_gof']} onco vs {d['n_flagged_lof']} TSG); {d['odds_ratio']}x enrichment — textbook cases (ABL1, KIT, BRAF, PDGFRA, FGFR2).",
            f"  LIMIT: RECALL {d['recall_gof_flagged']:.0%} (annotation-sparse; AUC {d['auc']} dragged by un-annotated 0s) — an INFORMATION gap, not compute.",
            "  NEXUS hook: a mutation breaking an inhibitory brake -> activity UP (GOF) via directed_activity(inhibitory=True), still gated by folding.",
            "  UNCHANGED: neomorphic GOF (a brand-new interface) stays out of reach — needs docking + experiment."])

    def nexus(self, args):
        """RESEARCH: the dual-sensor enzyme-health node (nexus.py) — INTRINSIC stability (folding ΔΔG -> folded_fraction)
        AND EXTRINSIC binding (interface ΔΔG -> bound_fraction) combined by a SOFT AND (graded product) into an activity
        that drives the FBA. Validated non-circular on measured SKEMPI: the two failure modes are ORTHOGONAL (r 0.15),
        so a stability-only node MISSES 94% of interface breakers; adding the extrinsic sensor lifts held-out AUC 0.56 ->
        0.75. Honest boundary: sensors are near-field structure (solid); the FBA->phenotype step is far-field (buffered).
        LIVE (integrated into the cell): nexus GENE [UNIPROT [POS WT MUT [PDB CHAIN]]] — reason a real mutation through
        DIRECTION (regsign LOF/GOF) + the structural dual-sensor (fold ΔΔG, bind ΔΔG when a complex+interface residue is
        given) + the cell's own essentiality. e.g.  nexus BRAF P15056   |   nexus GHR P10912 304 W A 1A22 B
        usage: nexus | nexus run | nexus GENE [UNIPROT [POS WT MUT [PDB CHAIN]]]"""
        if args and args[0].lower() in ("run", "rerun"):
            import nexus as _N; _N.main(); return "(nexus dual-sensor ran live — see above)"
        if args and args[0].lower() not in ("report",):
            # LIVE cell query: reason a mutation through the integrated node (regsign + dual-sensor + cell essentiality)
            if not hasattr(self, "_nexuscell"):
                from nexus_cell import NexusCell
                self._nexuscell = NexusCell(kernel=self)
            gene = args[0]; up = args[1] if len(args) > 1 else None
            pos = wt = mut = pdb = chain = None
            if len(args) >= 5:
                try:
                    pos = int(args[2])
                except ValueError:
                    return "nexus: POS must be an integer.  usage: nexus GENE [UNIPROT [POS WT MUT [PDB CHAIN]]]"
                wt, mut = args[3].upper(), args[4].upper()
            if len(args) >= 7:
                pdb, chain = args[5].upper(), args[6].upper()
            r = self._nexuscell.query(gene, uniprot=up, pos=pos, wt=wt, mut=mut, pdb=pdb, chain=chain)
            return self._nexuscell.render(r)
        import json, os
        p = "outputs/orphan/nexus.json"
        if not os.path.exists(p): return "nexus: run 'nexus run' (or python3 colab/nexus.py) first."
        d = json.load(open(p))
        out = ["nexus — dual-sensor enzyme health (intrinsic stability AND extrinsic binding), soft-AND -> FBA:",
               f"  WHY BOTH (measured, non-circular, n={d['n']}): the two failure modes are ORTHOGONAL — "
               f"Pearson(predicted ΔΔG_fold, measured ΔΔG_bind) {d['orthogonality_pearson']} — so the stability sensor "
               f"MISSES {d['intrinsic_miss_rate']:.0%} of interface breakers.",
               f"  FUSION (complex-held-out): intrinsic-only AUC {d['auc_intrinsic_only']} (~chance) -> +extrinsic "
               f"{d['auc_nexus_dualsensor']} ({d['fusion_gain']:+.2f}) — the 2nd sensor catches what the 1st is blind to.",
               "  SOFT-AND: activity = folded_fraction(ΔΔG_fold) × bound_fraction(ΔΔG_bind), graded not Boolean; binding is LOSS-only (no neomorphic gain)."]
        if d.get("fba_demo"):
            cc = {c["case"]: c for c in d["fba_demo"]["cases"]}
            sev = cc.get("severe both (catastrophic)", {})
            out.append(f"  FBA (demo, essential rxn): catastrophic damage -> activity {sev.get('activity')} -> biomass "
                       f"{sev.get('biomass_frac')} (propagates); moderate damage BUFFERED (thermo saturation + excess capacity).")
        out.append("  HONEST: sensors = near-field STRUCTURE (solid); FBA->phenotype = far-field (buffered/does-not-compose) — a demo, not a validated phenotype predictor.")
        return "\n".join(out)

    def _impact_systems(self, i):
        """which top-level cell systems (cell_pathway_map) a gene index belongs to."""
        import cell_pathway_map as cpm
        import json as _j
        if not hasattr(self, "_cpm_sys"):
            P = _j.load(open("outputs/orphan/reactome_pathways.json"))["pathways"]
            self._cpm_sys = [(sh, set(P[nm])) for nm, sh, _ in cpm.SYSTEMS if nm in P]
        return [sh for sh, mem in self._cpm_sys if i in mem]

    def _impact_complexes(self, i):
        """the complexes a gene is a subunit of + their co-members — the STRUCTURAL disassembly, readable from the graph
        with no measurement (validated: complex-mates are co-essential ~1000x chance)."""
        D = self.C.D
        g2c, plx = D.get("gene2cplx", {}), D.get("complexes", {})
        nm = [g["name"] for g in D["genes"]]
        out = []
        for cn in g2c.get(str(i), []):
            mem = [nm[m] for m in plx.get(cn, []) if isinstance(m, int) and m < len(nm)]
            out.append((cn, mem))
        return out

    def _impact_contribution(self, gene):
        """what the protein CONTRIBUTES to each partner, and in what DIRECTION — from the SIGNED edges (sig + reg,
        ~73% reliable). A partner it ACTIVATES goes DOWN when it's removed; a partner it INHIBITS goes UP (released)."""
        D = self.C.D
        nm = [g["name"] for g in D["genes"]]
        if not hasattr(self, "_sigmap"):
            import collections as _c
            sm = _c.defaultdict(dict)
            for src, tgt, sg in (D.get("sig", []) + D.get("reg", [])):     # sig first (higher quality), reg fills
                if src < len(nm) and tgt < len(nm) and sg != 0:
                    sm[nm[src]].setdefault(nm[tgt], sg)
            self._sigmap = sm
        edges = self._sigmap.get(gene, {})
        act = sorted(t for t, sg in edges.items() if sg > 0)
        inh = sorted(t for t, sg in edges.items() if sg < 0)
        return act, inh

    def impact(self, args):
        """MUTATE or REMOVE a gene and see how far the effect is KNOWABLE — the honest whole-cell chain. It layers only
        what's validated and ABSTAINS at the wall: [1] near-field (NEXUS: does the mutation break the protein? ~0.75) or
        full removal; [2] the DIRECT cell effect as MEASURED Perturb-seq data (real, for the ~9,871 knocked-out genes) —
        or an honest 'unobserved' if there's no measurement; [3] descriptive context (which cell systems, essentiality);
        [4] the FAR-FIELD wall — propagating to a whole-cell phenotype is MEASURED not to compose (transitivity ~0.009,
        forward-model AUC ~0.50), so we abstain rather than fake a cascade. usage: impact GENE [UNIPROT POS WT MUT [PDB CHAIN]]"""
        if not args:
            return "impact GENE [UNIPROT POS WT MUT [PDB CHAIN]]   — mutate/remove a gene; see how far the effect is knowable"
        gene = args[0]
        i = self._pid(gene)
        if i is None:
            return f"impact: no such gene '{gene}'"
        mut = len(args) >= 5
        out = [f"impact {gene}" + (f" {' '.join(args[1:])}" if len(args) > 1 else "  (full removal / knockout)") + " — how far we can see:\n"]
        # [1] NEAR-FIELD (validated)
        if mut:
            out.append("[1] NEAR-FIELD — does the mutation break the protein?  (VALIDATED, ~0.75 hotspot AUC)")
            out.append("    " + self.nexus(args).replace("\n", "\n    "))
        else:
            g = self.C.genes[i]
            out.append("[1] NEAR-FIELD — full removal = complete loss of function"
                       + (" (a DARK gene — little structural detail)" if g.get("dark") else "") + ".")
        # [2] DIRECT cell effect (measured only)
        if not hasattr(self, "_dep"):
            import dependency as _D
            self._dep = _D.Dependency(kernel=self)
        ko = self._dep.knockout(gene, topn=8)
        out.append("")
        if ko.get("status") == "measured":
            prec = "high-precision" if ko.get("hi_precision") else "genome-wide, lower precision"
            out.append(f"[2] DIRECT cell effect — MEASURED (real Perturb-seq, {prec}): {ko['n_dependents_strong']} genes strongly moved")
            out.append("    goes DOWN (need " + gene + "): " + (", ".join(f"{x}({v})" for x, v in ko["down"]) or "—"))
            out.append("    goes UP  (released):     " + (", ".join(f"{x}({v})" for x, v in ko["up"]) or "—"))
        else:
            out.append("[2] DIRECT cell effect — NO measured knockout for this gene → the transcriptional response is UNOBSERVED.")
        # [2b] STRUCTURAL disassembly — from the graph, no measurement needed (this is what the graph CAN tell you)
        cxs = self._impact_complexes(i)
        if cxs:
            show = "; ".join(f"{cn} (with {', '.join([m for m in mem if m != gene][:4]) or '—'})" for cn, mem in cxs[:4])
            out.append("")
            out.append(f"[2b] STRUCTURAL — from the graph, NO measurement: removing {gene} disassembles {len(cxs)} complex(es): {show}"
                       + (" …" if len(cxs) > 4 else ""))
            out.append("     co-subunits lose their assembly partner (VALIDATED: complex-mates co-essential ~1000x chance). This IS graph-predictable.")
        # [2c] DIRECTIONAL contribution — from signed edges: which partners this protein activates vs inhibits
        act, inh = self._impact_contribution(gene)
        if act or inh:
            out.append("")
            out.append(f"[2c] CONTRIBUTION & DIRECTION (signed edges) — {gene} regulates {len(act)+len(inh)} partners; the sign gives the "
                       f"DIRECTION of the effect for any partner that DOES respond (~73% reliable); WHICH respond is the [4] wall.")
            if act:
                out.append(f"     ACTIVATES {len(act)} (a responder goes DOWN when {gene} removed): {', '.join(act[:8])}" + (" …" if len(act) > 8 else ""))
            if inh:
                out.append(f"     INHIBITS  {len(inh)} (a responder goes UP / released): {', '.join(inh[:8])}" + (" …" if len(inh) > 8 else ""))
        # [3] descriptive context
        systems = self._impact_systems(i)
        dep = self.C.genes[i].get("dep_frac")
        ess = (f"dep_frac {dep:.2f} → {'ESSENTIAL' if isinstance(dep, (int, float)) and dep > 0.5 else 'non-essential'}"
               if isinstance(dep, (int, float)) else "essentiality not measured")
        out.append("")
        out.append(f"[3] CONTEXT (descriptive) — cell system(s): {', '.join(systems) or '(none — no pathway / dark gene)'};  {ess}")
        # [4] the wall
        out.append("")
        out.append("[4] FAR-FIELD (the wall) — the transcriptional CASCADE is NOT graph-predictable. Measured: graph")
        out.append("    neighbours (PPI+complex+pathway, the labelled interconnections) are 4x-enriched for real movers but")
        out.append("    capture only ~3% of them (miss ~97%; a typical knockout: 0 of its movers are graph neighbours).")
        out.append("    So we ABSTAIN on the whole-cell phenotype — the structural break [2b] is the honest limit of the graph.")
        return "\n".join(out)

    def dmasif(self, args):
        """RESEARCH: Geodesic Surface Learning (dMaSIF approach; dmasif.py) — the fine surface-POINT representation that
        was the remaining gap to MaSIF-grade. Surface points (not residue centres) + a learnable QUASI-GEODESIC
        CONVOLUTION in each point's local tangent frame + real APBS electrostatics per point, trained contrastively,
        held-out. Beats the residue-patch model on discrimination (AUC ~0.85 vs 0.76) and, decisively, FIXES retrieval —
        pinpointing the actual contacting point at ~13x chance where the residue model was ~1.4x.
        usage: dmasif | dmasif run"""
        if args and args[0].lower() in ("run", "rerun"):
            import dmasif as _D; _D.main(); return "(dMaSIF trained live — see above)"
        import json, os
        p = "outputs/orphan/dmasif.json"
        if not os.path.exists(p): return "dmasif: run 'dmasif run' (or python3 colab/dmasif.py) first."
        d = json.load(open(p))
        return "\n".join([
            "dmasif — Geodesic Surface Learning (surface points + learnable quasi-geodesic convolutions):",
            f"  trained on {d['n_complexes']} complexes, held out BY complex. Surface POINTS + local-tangent-frame conv + real APBS potential per point.",
            f"  DISCRIMINATION: AUC {d['discrimination_auc']} — vs residue-patch+APBS {d['residue_patch_auc']}, vs coarse residue 0.65.",
            f"  RETRIEVAL (the test the residue model FAILED): top-1 {d['retrieval_top1']:.1%} (chance {d['retrieval_chance_top1']:.1%}), "
            f"top-5 {d['retrieval_top5']:.1%} (chance {d['retrieval_chance_top5']:.1%}) = {d['retrieval_lift_top5']}x chance (residue model was ~1.4x).",
            "  net: finer surface-point resolution + geodesic convolution genuinely pinpoints the contacting point. Limit: subsampled points, 2-layer CPU net, Gaussian surface not exact MSMS."])

    def desolv(self, args):
        """RESEARCH: retest of an implicit-DESOLVATION term done the RIGHT way (desolv_eef1.py) — real Shrake-Rupley SASA
        (bound vs unbound) + the 'hydrophobic-illusion' SATISFACTION correction (a buried polar/charged atom is only
        taxed when it has NO H-bond/salt partner across the interface). Honest outcome: the physics validates
        (hydrophobic-buried r ~0.38; satisfaction separates satisfied +0.25 vs unsatisfied -0.05 polar burial) and it
        fixes the crude version's flaw (no longer HURTS), but on the alanine held-out task it adds ~0.00 because the
        burial signal is already captured by vdW+contacts. Correct term, not a headline gain here.
        usage: desolv | desolv run"""
        if args and args[0].lower() in ("run", "rerun"):
            import desolv_eef1 as _E; _E.main(); return "(desolv-EEF1 retest ran live — see above)"
        import json, os
        p = "outputs/orphan/desolv_eef1.json"
        if not os.path.exists(p): return "desolv: run 'desolv run' (or python3 colab/desolv_eef1.py) first."
        d = json.load(open(p))
        return "\n".join([
            "desolv — implicit desolvation retested with REAL SASA + satisfaction correction (EEF1):",
            f"  physics validates: hydrophobic-buried r {d['r_hydro']:+.2f}; satisfaction separates polar burial "
            f"(satisfied r {d['r_sat_polar']:+.2f} vs UNSATISFIED r {d['r_unsat_polar']:+.2f}).",
            f"  held-out alanine Pearson: baseline {d['r_base']} -> +EEF1-desolv {d['r_eef1']} ({d['eef1_gain']:+.3f}); "
            f"crude occlusion desolv {d['r_crude']}.",
            f"  verdict: {'folds in — real gain' if d['eef1_helps'] else 'correct + more physical + no longer HURTS, but ~0 gain on alanine (burial already captured by vdW+contacts) -> NOT folded into headline'}."])

    def kg(self, args):
        """the LABELED multi-relational cell knowledge graph (graph_label.py): nodes = genes, edges TYPED by relation —
        ppi (physical interaction), path (same Reactome pathway), lit (co-mentioned in a paper). 'kg GENE' shows a gene's
        typed neighbourhood (HOW it connects, not just THAT it does); 'kg' shows the whole-graph label stats. HONEST: a
        DESCRIPTIVE near-field KG — these edge types predict a knockout's effect at ~floor (r 0.03–0.09), so it is the
        substrate for representation / link-prediction / querying, NOT a causal/phenotype predictor. usage: kg | kg GENE"""
        import graph_label as gl
        if not hasattr(self, "_kg"):
            self._kg = gl.build()
        if args and args[0] not in ("run", "stats"):
            nb = gl.neighbors(self._kg, args[0])
            if nb is None:
                return f"kg {args[0]}: not a gene in the graph."
            lab = {"ppi": "physical interaction ", "path": "same pathway        ", "lit": "co-mentioned in paper"}
            out = [f"kg {args[0]} — labeled neighbourhood (HOW it connects):"]
            for r, v in nb.items():
                out.append(f"  [{r:4s} · {lab[r]}] {', '.join(v)}")
            return "\n".join(out) if nb else f"kg {args[0]}: in the graph but has no labeled edges."
        s = gl.stats(self._kg)
        out = ["kg — the labeled multi-relational cell graph (ppi + pathway + literature):",
               f"  {s['n_nodes']:,} gene nodes; {s['genes_with_any_edge']:,} have ≥1 labeled edge; "
               f"{s['total_labeled_edges']:,} unique labeled pairs"]
        for r in s["relations"]:
            out.append(f"    {r:5s}: {s['edges_by_type'][r]:>8,} edges  over {s['genes_covered_by_type'][r]:,} genes")
        out.append(f"  cross-type overlap {s['cross_type_overlap_frac']} → the relations are COMPLEMENTARY (each says a different thing).")
        out.append("  HONEST: descriptive near-field KG (edges predict knockout at ~floor) — a substrate for "
                   "representation/link-prediction, not phenotype.")
        return "\n".join(out)

    def ladder(self, args):
        """the compartment LAYOUT of a pathway (spatial_ladder.py): each protein placed in every compartment it's
        annotated in (localization, ~100%), grouped membrane→nucleus. 'ladder PATHWAY' (fuzzy name) shows one pathway's
        layers; 'ladder GENE' shows a gene's compartments + the pathways it's in; 'ladder' shows the reality-check.
        For an ORPHAN gene (not in pathway data) it still reads the compartment DIRECTLY and adds a literature-inferred
        candidate function (lit_place.py: IDF-weighted co-mention vote — a HYPOTHESIS, ~4x chance / top-5 ~1-in-5).
        HONEST: the DIRECTIONAL membrane→nucleus ladder was MEASURED not to hold (regulatory tier ≠ spatial depth), so
        this is an occupancy PROFILE read straight from localization, NOT a route and NOT a phenotype predictor.
        usage: ladder | ladder PATHWAY | ladder GENE"""
        import spatial_ladder as sl
        if not hasattr(self, "_sl"):
            self._sl = sl._load()                                  # names, tf, lv, loc, paths
        names, tf, lv, loc, paths = self._sl
        if not args:
            v = sl.validate(names, tf, lv, loc)
            return ("ladder — pathways laid out by compartment (localization × pathway):\n"
                    f"  reality check: nuclear vs membrane signalling genes differ in tier by {v['nuclear_more_downstream_by']:+.3f} "
                    f"(≈0), and MORE TFs are upstream ({v['upstream_tf_frac']:.0%}) than downstream ({v['downstream_tf_frac']:.0%}).\n"
                    "  → the directional membrane→nucleus ladder is NOT supported; delivering the compartment LAYOUT instead.\n"
                    "  try: ladder Signaling by EGFR | ladder WNT | ladder TP53")
        query = " ".join(args)
        hits = sl.find_pathways(query, paths)
        if hits:
            hits = sorted(hits, key=lambda p: -len(paths[p]))[:3]   # biggest matches first, cap the spew
            out = []
            for pw in hits:
                L = sl.ladder(pw, names, tf, lv, loc, paths)
                span = " | ".join(f"{s} {L['stage_counts'][s]}" for s in L["compartments_present"])
                out.append(f"ladder — {pw}  ({L['n_localized']}/{L['n_genes']} localized)\n  [{span}]")
                for s, genes in L["stages"].items():
                    out.append(f"    {s:14s}: {', '.join(genes[:10])}{' …' if len(genes) > 10 else ''}")
            out.append("  (* = TF · proteins counted in EVERY compartment they're annotated in · occupancy, not a route)")
            return "\n".join(out)
        if query in names:                                          # a gene
            deps = sorted(sl._depths(loc, query))
            comps = ", ".join(sl.DEPTH_NAME[d] for d in deps) or "(no mapped compartment)"
            tfmark = "  *TF" if query in tf else ""
            pws = sl.gene_pathways(query, names, paths)
            if pws:                                                 # in pathway data: known placement
                return (f"ladder — {query}: localized to [{comps}]{tfmark}\n"
                        f"  in {len(pws)} pathway(s): " + "; ".join(pws[:6]) + (" …" if len(pws) > 6 else ""))
            # ORPHAN gene (no pathway): compartment is still KNOWN; infer functional context from LITERATURE
            import lit_place as lp
            if not hasattr(self, "_lp"):
                self._lp = lp.build()
            p = lp.place(query, self._lp)
            out = [f"ladder — {query}: localized to [{comps}]{tfmark}   (NOT in pathway data — orphan)"]
            if p and p["co_mentioned"]:
                out.append(f"  literature co-mentions: {', '.join(p['co_mentioned'][:10])}")
                if p["candidate_pathways"]:
                    out.append("  candidate function — HYPOTHESIS from lit co-mention (validated ~4x chance, right in top-5 ~1-in-5):")
                    for pw in p["candidate_pathways"][:5]:
                        out.append(f"    · {pw}")
                else:
                    out.append("  (co-mentioned genes carry no capped-pathway membership — no candidate)")
            elif p:
                out.append("  no known-gene literature co-mentions found — compartment only.")
            else:
                out.append("  no literature data — compartment only.")
            return "\n".join(out)
        return f"ladder {query}: no matching pathway or gene."

    def connect(self, args):
        """CONNECT the orphan-gene signals into one mechanistic hypothesis (connect.py): literature (candidate pathway)
        + PPI (physical partners' pathways) + spatial (compartments), cross-checked. The confidence comes from AGREEMENT:
        a pathway that BOTH the literature co-mention AND a physical partner point to is the high-confidence tier
        (held-out top-1 ~0.71, vs 0.54 ppi-only, 0.13 lit-only). Then it checks locations (PPI partners co-localize 1.83x
        chance) and, where the gene and its inferred reaction sit in DIFFERENT compartments, flags a transport/trafficking
        partner that could bridge the gap (a HEURISTIC lead). HONEST: descriptive near-field triangulation — a strong
        starting mechanism, not a proven annotation or phenotype predictor. usage: connect GENE | connect"""
        import connect as cn
        if not hasattr(self, "_cn"):
            self._cn = cn.build()
        if not args:
            v = cn.validate(self._cn)
            return ("connect — triangulate literature + PPI + spatial for an orphan gene:\n"
                    f"  held-out pathway top-1:  lit-only {v['lit_only_top1']}   ppi-only {v['ppi_only_top1']}   "
                    f"CORROBORATED {v['corroborated_top1']} (fires {v['corroborated_coverage']:.0%}) = {v['corroboration_lift']}x lit\n"
                    f"  PPI partners co-localize {v['ppi_colocalize']} vs {v['random_colocalize']} random ({v['coloc_enrichment']}x) — location cross-check is grounded\n"
                    "  try: connect WDHD1 | connect CSE1L | connect UNC13C")
        c = cn.connect(args[0], self._cn)
        return cn.render(c)

    def conditions(self, args):
        """infer WHEN a pathway turns on (cell_conditions.py): trace its genes back to the stress/signal TFs that control
        them (curated directed reg edges, hypergeometric enrichment). Pathways with NO condition-TF enrichment are
        CONSTITUTIVE (always-on housekeeping); enriched ones report the trigger (HIF1A→hypoxia, SREBF→sterol, ATF4/XBP1→ER
        stress, TP53→DNA damage, NF-kB→inflammation, STAT→cytokine, HSF1→heat). VALIDATED top-1 ~0.67 / top-3 ~0.76 on
        condition-named pathways (chance 0.07/0.20). Backward regulator-tracing is a STRUCTURAL/annotation query (not the
        forward dynamical propagation that fails at chance) — a hypothesis for the trigger, not a proven condition.
        usage: conditions PATHWAY | conditions scan | conditions"""
        import cell_conditions as cc
        if not hasattr(self, "_cc"):
            self._cc = cc._load()
        names, reg, paths, N, sgn = self._cc
        if args and args[0].lower() in ("scan", "discover"):
            out = ["conditions — pathways whose trigger is NOT stated in their name but is inferred (the 'leftover'):"]
            for sc, p, tf, cond in cc.scan(names, reg, paths, N, 12):
                out.append(f"  {p[:54]:54s} → {cond}  [{tf} {sc}]")
            return "\n".join(out)
        if not args:
            v = cc.validate(names, reg, paths, N)
            return ("conditions — infer a pathway's activation trigger from the TFs that control its genes:\n"
                    f"  VALIDATED: traced condition-TF recovers the known trigger top-1 {v['top1']:.0%} / top-3 {v['top3']:.0%} "
                    f"on {v['n']} condition-named pathways (chance ~7%/20%).\n"
                    "  Constitutive pathways show no enrichment (always-on); conditional ones report the stimulus.\n"
                    "  try: conditions hypoxia | conditions cholesterol | conditions scan")
        query = " ".join(args)
        hits = [p for p in paths if query.lower() in p.lower()]
        if not hits:
            return f"conditions {query}: no matching pathway."
        out = []
        for p in sorted(hits, key=lambda p: -len(paths[p]))[:4]:
            r = cc.infer(p, names, reg, paths, N, sgn)
            if r["constitutive"]:
                out.append(f"conditions — {p}: CONSTITUTIVE (always-on; no stress/signal TF enrichment) · top {r['top'][:2]}")
                continue
            out.append(f"conditions — {p}:")
            for c in r["conditions"][:3]:
                dirn = {"activates": "→ ON", "represses": "→ OFF", "unsigned": "(dir?)"}[c["direction"]]
                fb = f"  ⟲ {c['feedback'][0]} feedback via {', '.join(c['feedback'][1])}" if c["feedback"] else ""
                out.append(f"    {c['condition']} {dirn} [{c['tf']} {c['score']}]{fb}")
        return "\n".join(out)

    def tfbs(self, args):
        """score how a DNA variant changes a TF's BINDING SITE (tfbs_score.py) — the honest protein-DNA analogue of the
        NEXUS binding node, built the right way (JASPAR PWM log-odds over 743 TFs, neighbour-aware motif window) rather
        than retrofitting NEXUS's amino-acid physics. 'tfbs GENE' shows the motif; 'tfbs GENE SEQ' scores a window;
        'tfbs GENE SEQ POS ALT' scores a variant's effect (Δ log-odds, ± = strengthen/weaken). HONEST: predicts INTRINSIC
        in-vitro sequence preference (validated by construction), NOT in-vivo binding/regulation (chromatin/cofactor/conc =
        the context wall); DNA-shape ('rotation') is the documented Rohs add-on. usage: tfbs GENE [SEQ [POS ALT]]"""
        import tfbs_score as tb
        if not hasattr(self, "_pwm"):
            self._pwm = tb._load()
        if not args:
            return f"tfbs GENE [SEQ [POS ALT]]   — TF binding-site variant scorer ({len(self._pwm)} JASPAR TFs)"
        tf = args[0]
        if tf not in self._pwm:
            return f"tfbs {tf}: no JASPAR motif (have {len(self._pwm)} TFs)."
        lo = self._pwm[tf]["lo"]
        cons = "".join("ACGT"[max(range(4), key=lambda b: lo[p][b])] for p in range(len(lo)))
        if len(args) == 1:
            return f"tfbs {tf}: motif {self._pwm[tf]['id']}, {len(lo)}bp, consensus {cons}"
        seq = args[1]
        if len(args) >= 4:
            v = tb.variant_effect(tf, seq, int(args[2]), args[3], self._pwm)
            return (f"tfbs {tf} ({v['motif']}, consensus {cons}) — variant {seq[int(args[2])]}->{args[3].upper()} @ {args[2]}:\n"
                    f"  best score {v['ref_score']} -> {v['alt_score']}  (Δ {v['delta']:+.2f})  → {v['call']}\n"
                    "  [intrinsic in-vitro preference; not in-vivo binding — chromatin/cofactor/conc is the wall]")
        return f"tfbs {tf} on {seq.upper()}: best motif log-odds {tb.score(tf, seq, self._pwm):.2f}  (consensus {cons})"

    def cofactor(self, args):
        """the cofactor SWITCH — a TF binds a DIFFERENT sequence depending on which partner it dimerises with (cofactor.py).
        Solved the DATA way: JASPAR composite (heterodimer) motifs = the measured site the PAIR binds, gated by which
        cofactor is EXPRESSED in a cell (emask). 'cofactor TF' lists partners with a composite motif; 'cofactor TF CELLTYPE'
        keeps only those expressed there; 'cofactor TF1 TF2' shows the switch (composite vs each solo site). HONEST: delivers
        the pair's in-vitro binding site (the concrete switch), NOT de-novo complex formation (the AF-Multimer route we skip)
        nor in-vivo regulation. ~479 measured pairs. usage: cofactor TF [CELLTYPE] | cofactor TF1 TF2"""
        import cofactor as cf
        if not hasattr(self, "_cof"):
            self._cof = cf._load()
        G = self._cof
        if not args:
            return f"cofactor TF [CELLTYPE] | cofactor TF1 TF2   — the cofactor switch ({len(G['comp'])} composite motifs)"
        tf = args[0]
        # two TFs both with a composite -> show the switch
        if len(args) == 2 and frozenset((args[0], args[1])) in G["pair2mid"]:
            s = cf.switch(args[0], args[1], G)
            solo = "  ".join(f"{t}={c}" for t, c in s["solo"].items() if c)
            return (f"cofactor SWITCH {s['pair']} ({s['composite_id']}):\n"
                    f"  composite (pair binds): {s['composite_consensus']}  [{s['composite_len']}bp]\n"
                    f"  solo (each alone):      {solo}\n"
                    "  [measured HT-SELEX site of the dimer; in-vitro specificity, not in-vivo binding]")
        ct = args[1] if len(args) >= 2 else None
        c = cf.cofactors(tf, ct, G)
        if not c["cofactors"]:
            where = f" expressed in '{c['celltype']}'" if ct else ""
            return f"cofactor {tf}: no partner with a composite motif{where} (have {len(G['comp'])} pairs)."
        head = f"cofactor {tf}" + (f" in '{c['celltype']}'" if c["celltype"] else "") + f": {c['n']} partner(s) with a measured composite site"
        rows = []
        for x in c["cofactors"][:12]:
            flags = []
            if x["physical_pair"]:
                flags.append("PPI")
            if x["expressed"]:
                flags.append("expressed")
            fl = f" [{', '.join(flags)}]" if flags else ""
            rows.append(f"    {x['cofactor']}{fl}  → switch: cofactor {tf} {x['cofactor']}")
        return head + "\n" + "\n".join(rows)

    def _discover(self):
        """lazily load the discovery results (discover.py → discover.json)."""
        if not hasattr(self, "_disc"):
            import json, os
            p = "outputs/orphan/discover.json"
            self._disc = json.load(open(p)) if os.path.exists(p) else {}
        return self._disc

    def discover(self, args):
        """aim the reasoning machinery at the UNKNOWNS, not at essentiality (already measured). Four engines, each
        honestly graded (discover.py):
          discover selective   — genes with a BIMODAL (selective) dependency profile = the drug-target window.
                                  VALIDATED: recovers 20/22 known oncology targets in the top-500 (33.5x enrichment).
          discover druggable   — selective AND surface-accessible (membrane/secreted) — a tractable shortlist (partial:
                                  no Pharos/DGIdb tractability loaded).
          discover disease     — genes not yet disease-linked scoring in the top disease-prior bucket. HONEST: coarse
                                  (6 buckets), modest axis (0.65) — a candidate SET, not a ranking.
          discover function G  — dark-gene function by label transfer. HONEST NULL for the truly dark proteome (neighbours
                                  scatter); only coherent-module genes (e.g. rRNA) get a real call.
        usage: discover [selective|druggable|disease|function GENE]"""
        d = self._discover()
        if not d:
            return "discover: run discover.py first to compute the discovery results."
        sub = args[0] if args else "summary"
        if sub == "selective":
            se = d["selective"]
            out = [f"discover selective — {se['n_selective']} genes with a bimodal (selective) dependency profile",
                   f"  VALIDATED: {se['known_recovered_top500']}/{se['known_total']} known oncology targets in top-500 "
                   f"({se['enrichment_top500_vs_chance']}x enrichment vs chance)",
                   "  top selective dependencies (skew = strength of the bimodal tail):"]
            for t in se["top_selective_targets"][:12]:
                tag = "  [known]" if t["known"] else ""
                out.append(f"    {t['gene']:10} skew {t['skew']:5}  strong in {t['n_strong_dep_lines']:3} lines  "
                           f"min effect {t['min_effect']}{tag}")
            out.append("  novel (not in the known set): " + ", ".join(t["gene"] for t in se["top_novel_selective"][:12]))
            return "\n".join(out)
        if sub == "druggable":
            dg = d["druggable"]
            out = [f"discover druggable — {dg['n']} selective + surface-accessible targets ({dg['note']})"]
            for t in dg["shortlist"]:
                tag = "  [known]" if t["known"] else ""
                out.append(f"    {t['gene']:10} strong in {t['n_strong_dep_lines']:3} lines, {t['compartment']}{tag}")
            return "\n".join(out)
        if sub == "disease":
            dn = d["disease_new"]
            gs = ", ".join(c["gene"] for c in dn["top_novel_disease_candidates"][:15])
            return (f"discover disease — {dn['n_in_top_bucket']} not-yet-disease genes in the top prior bucket "
                    f"(P≈{dn['top_bucket_probability']}); axis AUC {dn['axis_auc']}, {dn['resolution']}\n  {gs}")
        if sub == "function":
            fn = d["function"]
            if len(args) < 2:
                return "discover function GENE   (e.g. discover function NOL10)"
            g = args[1]
            hit = next((p for p in fn["top_predictions"] if p["gene"] == g), None)
            if hit:
                return (f"discover function {g}: → {hit['predicted_function']}  (vote margin {hit['margin']}, "
                        f"a coherent-module call)")
            return (f"discover function {g}: no concentrated call — {fn['verdict']}")
        # summary
        fn, se, dn, dg = d["function"], d["selective"], d["disease_new"], d["druggable"]
        return "\n".join([
            "discover — the reasoning machinery aimed at the UNKNOWNS (grade honestly attached to each):",
            f"  1. selective  [WIN]   {se['n_selective']} bimodal targets; recovered "
            f"{se['known_recovered_top500']}/{se['known_total']} known oncology targets in top-500 "
            f"({se['enrichment_top500_vs_chance']}x). Top novel: " + ", ".join(t["gene"] for t in se["top_novel_selective"][:6]),
            f"  2. druggable  [PART]  {dg['n']} selective + surface targets (no Pharos tractability yet): "
            + ", ".join(t["gene"] for t in dg["shortlist"][:6]),
            f"  3. disease    [WEAK]  {dn['n_in_top_bucket']} novel candidates in the top bucket, but coarse (axis "
            f"{dn['axis_auc']}).",
            f"  4. function   [NULL]  label transfer is null on the dark proteome (margin "
            f"{fn['dark_median_vote_margin']}); only {fn['n_dark_high_margin']} coherent-module genes get a real call.",
            "  → the one validated discovery engine is SELECTIVE dependencies. 'discover selective' for the ranked list."])

    def _pathway_decoder(self):
        """lazily load the co-dependency matrix + pathway labels for data-driven pathway decoding."""
        if getattr(self, "_pwd", None) is None:
            import numpy as np, os
            p = "outputs/orphan/depmap_vecs.npz"
            if not os.path.exists(p):
                self._pwd = False; return False
            z = np.load(p, allow_pickle=True)
            syms = [str(s) for s in z["syms"]]
            Z = np.nan_to_num(z["Z"].astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
            nz = np.linalg.norm(Z, axis=1); v = nz > 1e-6; Z[v] /= nz[v][:, None]
            idx = {s: i for i, s in enumerate(syms)}
            labels = {}
            for s in syms:
                i = self.C.idx.get(s)
                if i is None or not v[idx[s]]:
                    continue
                pp = self.C.genes[i].get("path") or []
                pp = [pp] if isinstance(pp, str) else pp
                if pp:
                    labels[s] = [str(x) for x in pp]
            lab = list(labels); li = np.array([idx[s] for s in lab])
            self._pwd = dict(Zn=Z, idx=idx, labels=labels, lab=lab, li=li, valid=v)
        return self._pwd

    def pathway(self, name, k=15):
        """decode a gene's pathway from DATA (its co-dependency neighbours across 1,150 cell lines), not a curated
        label. Validated (pathway_decode.py): recovers the exact Reactome pathway at 21% top-1 = 8x chance — a strong
        SHORTLIST engine, especially for UNANNOTATED genes; corroborates the annotation where one exists."""
        import numpy as np
        D = self._pathway_decoder()
        if not D:
            return "pathway: co-dependency matrix not available."
        if name not in D["idx"] or not D["valid"][D["idx"][name]]:
            return f"pathway {name}: no co-dependency vector — can't decode its pathway from data."
        gi = D["idx"][name]; sims = D["Zn"][D["li"]] @ D["Zn"][gi]
        top = np.argsort(-sims)[:80]; votes, nb, cnt = {}, [], 0
        for j in top:
            h = D["lab"][j]
            if h == name or sims[j] <= 0:
                continue
            for pw in D["labels"][h]:
                votes[pw] = votes.get(pw, 0.0) + float(sims[j])
            if len(nb) < 6:
                nb.append(h)
            cnt += 1
            if cnt >= k:
                break
        if not votes:
            return f"pathway {name}: no confident functional neighbours."
        ranked = sorted(votes.items(), key=lambda x: -x[1])[:5]
        tot = sum(v for _, v in ranked) + 1e-9
        i = self.C.idx.get(name)
        known = {str(x) for x in (self.C.genes[i].get("path") or []) if x} if i is not None else set()
        L = [f"pathway {name} — decoded from co-dependency (top functional neighbours: {', '.join(nb)})"]
        for pw, w in ranked:
            tag = "  [matches annotation]" if pw in known else ("  [NEW — gene was unannotated]" if not known else "")
            L.append(f"  {w/tot:>4.0%}  {pw[:54]}{tag}")
        L.append("  data-decoded (validated 21% exact top-1 = 8x chance); a shortlist, not a verdict.")
        return "\n".join(L)

    def coverage(self):
        """the honest whole-cell coverage map: '55%' is only the DEEPEST axis (predict the full knockout response,
        data-limited); this counts every trustworthy answer the software gives — measured essentiality, response,
        FBA viability, TF reprogramming, grounded literature — and the fraction with >=1 answer vs truly dark."""
        import json, os
        p = "outputs/orphan/coverage.json"
        if not os.path.exists(p):
            return "coverage: map not built yet (run colab/coverage.py)."
        R = json.load(open(p)); N = R["genome"]
        ax = R["axes"]
        def row(label, d):
            return f"  {label:<34}{d['n']:>8}{d['frac']*100:>8.1f}%   [{d['grade'].split()[0]}]"
        L = [f"whole-cell coverage — {N:,} genes, every trustworthy answer counted (not just the deepest)",
             f"  {'axis':<34}{'genes':>8}{'genome':>9}",
             "  " + "-" * 60,
             row("essentiality (measured DepMap)", ax["essentiality_measured"]),
             row("RESPONSE prediction ← the 55%", ax["response_predictable_55pct"]) + "  data-limited ceiling",
             row("viability (FBA physics)", ax["viability_physics_FBA"]),
             row("reprogrammable (TF, forward 0.99)", ax["reprogrammable_TF"]),
             row("documented (grounded literature)", ax["documented"]),
             "  " + "-" * 60,
             f"  {'≥1 TRUSTWORTHY ANSWER':<34}{R['union_any_trustworthy_answer']['n']:>8}"
             f"{R['union_any_trustworthy_answer']['frac']*100:>8.1f}%   ← real reach",
             f"  {'truly dark (nothing)':<34}{R['truly_dark']['n']:>8}{R['truly_dark']['frac']*100:>8.1f}%",
             "  the deepest answer (full response) is stuck at 55% (needs new wet-lab screens); the SYSTEM's",
             "  total reach is 99% because it answers many DIFFERENT trustworthy questions — 55% was never the whole story."]
        return "\n".join(L)

    def cellsim(self, gene, frac=0.2, rank=60):
        """checkpoint-anchored simulation: reveal a fraction of a knockout's MEASURED response as data checkpoints,
        reconstruct the REST from the measured co-response structure. This is the honest 'top-down sim with data
        helpers': validated (cellsim.py), from 10–50% observed you recover the held-out response at r=0.42–0.50,
        while the mechanistic regulatory simulation adds ~nothing (r=0.01) — the signal is the DATA anchors, not the
        forward-run. The sim runs on the rails the data lays."""
        import numpy as np
        dbg = self._load_debugger()
        if not dbg or gene not in dbg.get("pidx", {}):
            return (f"cellsim {gene}: not in the mounted measured screen — reconstruction needs a real knockout "
                    f"response to anchor to (try a gene from the screen, e.g. SF3B1).")
        M, i, syms = dbg["M"], dbg["pidx"][gene], dbg["syms"]
        mask = np.ones(M.shape[0], bool); mask[i] = False            # exclude the queried perturbation (no leakage)
        Mtr = M[mask]; mu = Mtr.mean(0)
        _, _, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
        V = Vt[:rank].T
        truth = M[i]
        perm = np.random.RandomState(0).permutation(M.shape[1]); nobs = int(frac * M.shape[1])
        obs, hid = perm[:nobs], perm[nobs:]
        z, *_ = np.linalg.lstsq(V[obs], truth[obs] - mu[obs], rcond=None)
        pred = mu + V @ z
        a, b = pred[hid] - pred[hid].mean(), truth[hid] - truth[hid].mean()
        r = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        top = hid[np.argsort(-np.abs(pred[hid]))[:8]]
        showed = ", ".join(f"{syms[j]}{'↑' if pred[j] > 0 else '↓'}" for j in top)
        return "\n".join([
            f"cellsim {gene}  —  reconstruct the knockout response from {int(frac*100)}% measured checkpoints",
            f"  observed {nobs:,} of {M.shape[1]:,} genes (the anchors); reconstructed the other {len(hid):,}",
            f"  → reconstruction accuracy r = {r:.2f}   (validated sweep: 10%→0.42, 50%→0.50; baseline 0.25)",
            f"  strongest reconstructed responders: {showed}",
            "  the data checkpoints ARE the signal: the mechanistic regulatory sim adds ~nothing (r=0.01, validated).",
            "  this is top-down simulation held on the rails by measured anchors — assimilation, not free-running."])

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

    def _prec_note(self, name):
        """low-confidence flag for a gene measured ONLY in the genome-wide screen (not the high-precision essential
        set). The merge extends reach 2,058 -> 9,871 pieces, but the new genes carry a noisier fingerprint (r~0.3 vs
        the essential screen's ~0.86; biology-recovery 1% vs 16%), so their causal answers are weak leads, not calls."""
        dbg = self._pert if isinstance(self._pert, dict) else None
        if dbg and dbg.get("hi_prec") and name in dbg["pidx"] and name not in dbg["hi_prec"]:
            return "\n  [LOW-PRECISION: measured only in the genome-wide screen (r~0.3) — a weak lead; verify with strace/diagnose]"
        return ""

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
        return "\n".join(out) + self._prec_note(name)

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
            f"— the debugger is real; context transfer is the open problem.)"]) + self._prec_note(name)

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
  coverage             honest coverage map: every trustworthy answer, not just the 55% deepest axis
  ps [context]         process table (genes) by scheduler priority (essentiality)
  man GENE             process documentation (role, segment, priority, interfaces)
  pathway GENE         DECODE the gene's pathway from data (co-dependency), not the curated label (8x chance)
  lit GENE             the gene's PubMed literature (grounded+cited) — fills DARK genes
  top                  kernel threads (highest system-wide dependency)
  viability GENE       [FBA] top-down: does the cell still GROW after knockout? (objective+physics)
  assess GENE          best-evidence essentiality: fuse measured + physics + data; confidence rises when they agree
  reason GENE          REASON essentiality across independent lines, calibrated confidence (AUC 0.86)
  disease GENE         a 2nd reasoning engine: calibrated prior that a gene is disease-associated (modest, 0.65)
  discover [sub]       aim the engine at UNKNOWNS: selective [WIN, 33x] / druggable / disease / function GENE
  investigate GENE     predicted DOSSIER (job/place/path/timing/interactors/support) for an under-recorded gene;
                       'investigate underworld' = dark genes hidden inside characterized modules (job ~55%, dest ~54%)
  network GENE         the whole cell wired into one graph — INTERCONNECTED (physical) + INTERDEPENDENT (functional);
                       'network link A B' / 'network path A B [fam]' / 'network hubs [fam]'  (complementary + coherent)
  knockout GENE        the MEASURED Perturb-seq blast radius of removing a gene (what goes up/down); 'knockout impact'
                       ranks genes by blast-radius size. Direct effect real; does NOT propagate to unmeasured cascades
  protein GENE         knock out the PROTEIN (degrade it, PROTAC-style), not the gene: the STRUCTURAL disassembly —
                       every machine it is a subunit of loses a part (alias: degrade GENE). Complements 'knockout'
  cascade GENE         perturbation as an INFLUENCE network (what a KO affects) with each effect's route reconstructed:
                       direct / measured-mediated / unresolved. 'cascade X P' traces one route. ~16% routed, honest
  pertseq              the four canonical Perturb-seq applications on this data, honestly graded (GRN/function/disease/
                       immune); 'pertseq function GENE' = unknown-gene function by KO fingerprint [WORKS, 84%/4.1x]
  xcell GENE           CROSS-CELL-TYPE: which of a knockout's responders reproduce in BOTH K562 and HCT116 (Xaira
                       X-Atlas/Orion) — cell-type-robust vs cell-type-specific (needs fetch_xaira.py staged)
  fieldsim             the 4-layer continuous protein-field whole-cell engine (SH overlap + ΔΔG->γ + SDF PDE + Hill),
                       assembled & tested HONESTLY — runs end-to-end but is a demonstrator, not a predictor ('fieldsim run')
  latent               does a foundation-model latent (scGPT/Geneformer-style) beat 'predict the mean' on held-out
                       knockouts? tested: barely, and WORSE for novel perturbations — the M-L-M Phase-2 premise fails
  pstruct              structural dossier for a known pathway's steps (glycolysis): gene/reaction-order/family/active-
                       site residues/PPI/allostery + a cross-checked induced-fit (domain-closure) signal — real data
  enzyme               RESEARCH: what makes a metabolic enzyme essential? two axes — CENTRALITY (dominant) + PARALOG
                       buffering (real); AUC 0.86. self-corrected a confounded proxy ('enzyme run')
  interface            RESEARCH: which amino acids carry a PPI so a mutation there changes binding? (SKEMPI, 4,956
                       measured) — buried CORE + aromatic/charged residues; barrier-vs-stability; predict ΔΔG r 0.36
  flex                 RESEARCH: physics ΔΔG-binding node w/ clash relief + sp3 ROTAMER sampling + electrostatics/
                       induction vs measured SKEMPI — rotamers fix buried-clash physics (33%->4% expl), chem lifts r->0.52
  screen               RESEARCH: ΔΔG node as a PPI partner-screen — localise a mutation's LOSS to the right partner
                       (top-1 98%); magnitude needs the full model (burial alone ~0); NOVEL-partner gain NOT claimed
  surface              RESEARCH: REAL trained MaSIF-style surface-complementarity encoder (vs the random-weight mock) —
                       AUC 0.65 vs 0.50 untrained; real APBS electrostatics+surface lifts 0.65->0.76 on 333 cplx ('surface apbs')
  dmasif               RESEARCH: Geodesic Surface Learning (dMaSIF) — surface points + learned geodesic conv + APBS;
                       discrimination AUC 0.85 (vs residue-patch 0.76) and retrieval ~13x chance (residue was ~1.4x)
  nexus [GENE ...]     dual-sensor mutation node, LIVE in the cell: 'nexus GENE [UNIPROT [POS WT MUT [PDB CHAIN]]]' →
                       DIRECTION (regsign LOF/GOF) + fold/bind ΔΔG dual-sensor + cell essentiality. 'nexus'=report, 'nexus run'
  impact GENE [MUT]    mutate/remove a gene, see HOW FAR the effect is knowable: near-field (NEXUS) + DIRECT measured
                       effect (real Perturb-seq) + system context, and ABSTAINS at the far-field wall (doesn't fake a cascade)
  regsign              RESEARCH: regulatory-sign annotation = the GAIN-of-function lever — break an inhibitory brake ->
                       GOF; UniProt-derived, 86% precision / 5.4x enrichment vs oncogene-TSG (recall-limited: info gap)
  desolv               RESEARCH: desolvation retested w/ REAL SASA + satisfaction correction — physics validates
                       (hydrophobic r 0.38) but ~0 held-out gain on alanine (burial already captured); honest, not folded in
  mutate GENE UP POS WT MUT  reason a MUTATION through the cell: variant-damage × cell-dependency, regime-named
  level GENE            where in its pathway the gene acts: metabolic step / signaling tier / feedback / abstain
  loc GENE              subcellular compartment(s) from reviewed UniProt (the science-run localization layer)
  ppi GENE             top physical interaction partners (STRING v12.0; connectivity lifts reason to 0.89)
  perturb GENE         RUN it: propagate a perturbation through the network — blast radius + checkpoint (near-field)
  needs                 the software's own bill of materials: what it still needs (DATA / METHOD / HARD)
  check ENZYME KCAT    put in a kcat; the cell sanity-checks it vs the flux it must carry — WEAK/one-sided hint (~70%)
  boot [GENE]          RUN the genome forward (a clock): watch a cell-state emerge; [GENE] = knock out + re-run
  induce TF [TF..]     REPROGRAM: force master TF(s) ON, run forward, watch the lineage program light up
  cellsim GENE         top-down sim held on the rails by data: reconstruct a KO response from measured checkpoints
  strace GENE          [C] SIGKILL + measured downstream effect (Perturb-seq debugger)
  predict GENE         [C~] predict an UNMEASURED knockout's response from its neighbours
  whodunit GENE        [C] detective: recover the cause from a knockout's fingerprint
  diagnose up=.. down=..  [C] root-cause: whose knockout reverses a corrupted state
  simulate G1 G2 ..    [C~] "there is no spoon": edit the cell, propagate to the new state
  cure up=.. down=..    [C] free the cell: combination-therapy search to reverse a state
  deadlock GENE        [C] synthetic-lethal partners (co-kill = panic)
  patch GENE:MUT       static-lint a mutation (no-op / crash / logic-bug)
  lint "CLAIM"         [security] verify a claim; flag untrusted predictions & hub-leak
  kg [GENE]            the LABELED cell knowledge graph — edges TYPED ppi/pathway/literature; 'kg GENE' = typed
                       neighbourhood (HOW it connects). Descriptive near-field substrate (not a phenotype predictor)
  ladder [PATHWAY|GENE] a pathway laid out by COMPARTMENT (membrane→nucleus) from localization; occupancy PROFILE, not a
                       route — the directional ladder was MEASURED not to hold (tier ≠ spatial depth). For an ORPHAN gene
                       (no pathway) it adds a literature-inferred candidate function (co-mention HYPOTHESIS, ~4x chance)
  connect GENE         one mechanistic hypothesis for an orphan gene: literature + PPI + spatial, CROSS-CHECKED — the
                       pathway that lit AND a physical partner agree on is the confident tier (~0.71); + a transport lead
  conditions PATHWAY   infer WHEN a pathway turns on — trace its genes to the stress/signal TFs that control them;
                       'always-on' (constitutive) vs a named trigger (hypoxia/sterol/DNA-damage/…). 'conditions scan' = discovered
  tfbs GENE [SEQ POS ALT] score how a DNA variant changes a TF's binding site (JASPAR PWM, 743 TFs, neighbour-aware);
                       intrinsic in-vitro preference, NOT in-vivo binding (chromatin/cofactor = the wall)
  cofactor TF [CELLTYPE]  the cofactor SWITCH — the DIFFERENT site a TF binds with each dimer partner (479 composite
  cofactor TF1 TF2     motifs, gated by which cofactor is expressed); 'TF1 TF2' shows composite-vs-solo. In-vitro site,
                       not de-novo complex formation (AF-Multimer route skipped) nor in-vivo regulation
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
            if cmd == "coverage": return k.coverage()
            if cmd in ("pathway", "decode"): return k.pathway(args[0])
            if cmd == "viability": return k.viability(args[0])
            if cmd == "assess": return k.assess(args[0])
            if cmd in ("reason", "explain"): return k.reason(args[0])
            if cmd == "check": return k.check(args[0], args[1])
            if cmd in ("boot", "exec", "run"): return k.boot(args[0] if args else None)
            if cmd in ("induce", "reprogram"): return k.induce(args)
            if cmd in ("cellsim", "reconstruct"): return k.cellsim(args[0])
            if cmd == "lit": return k.lit(args[0])
            if cmd == "ps": return k.ps(context=args[0] if args else None)
            if cmd == "man": return k.man(args[0])
            if cmd == "top": return k.top()
            if cmd in ("strace", "kill"): return k.strace(args[0])
            if cmd == "predict": return k.predict(args[0])
            if cmd == "whodunit": return k.whodunit(args[0])
            if cmd in ("mutate", "variant"): return k.mutate(args)
            if cmd in ("level", "where"): return k.level(args[0])
            if cmd in ("loc", "compartment"): return k.loc(args[0])
            if cmd in ("ppi", "partners"): return k.ppi(args[0])
            if cmd in ("perturb", "live"): return k.perturb(args[0])
            if cmd in ("disease", "risk"): return k.disease(args[0])
            if cmd in ("discover", "unknown"): return k.discover(args)
            if cmd in ("investigate", "dossier", "profile"): return k.investigate(args)
            if cmd in ("network", "wire", "graph"): return k.network(args)
            if cmd in ("kg", "labeled"): return k.kg(args)
            if cmd in ("ladder", "compartments"): return k.ladder(args)
            if cmd in ("connect", "mechanism"): return k.connect(args)
            if cmd in ("conditions", "trigger"): return k.conditions(args)
            if cmd in ("tfbs", "motif"): return k.tfbs(args)
            if cmd in ("cofactor", "dimer"): return k.cofactor(args)
            if cmd in ("knockout", "ko"): return k.knockout(args)
            if cmd in ("protein", "degrade"): return k.protein(args)
            if cmd in ("cascade", "influence", "trace"): return k.cascade(args)
            if cmd in ("pertseq", "perturbseq"): return k.pertseq(args)
            if cmd in ("xcell", "crosscell"): return k.xcell(args)
            if cmd in ("fieldsim", "fields"): return k.fieldsim(args)
            if cmd in ("latent", "bridge"): return k.latent(args)
            if cmd in ("pstruct", "pathstruct"): return k.pstruct(args)
            if cmd in ("enzyme", "enzymes"): return k.enzyme(args)
            if cmd in ("interface", "hotspot"): return k.interface(args)
            if cmd in ("flex", "physics"): return k.flex(args)
            if cmd in ("screen", "ppi_screen"): return k.screen(args)
            if cmd in ("surface", "masif"): return k.surface(args)
            if cmd in ("dmasif", "geodesic"): return k.dmasif(args)
            if cmd in ("nexus", "dualsensor"): return k.nexus(args)
            if cmd in ("impact", "effect"): return k.impact(args)
            if cmd in ("regsign", "gof"): return k.regsign(args)
            if cmd in ("desolv", "eef1"): return k.desolv(args)
            if cmd in ("needs", "todo"): return k.needs()
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
    "# 0e) CHECKPOINT-ANCHORED SIM — a top-down sim held on the rails by measured data: reveal part of a knockout's",
    "#     response, reconstruct the rest (r=0.42–0.50). Data checkpoints are the signal; the mechanistic sim isn't.",
    "cellsim SF3B1",
    "# 1) kernel threads — the processes the system cannot run without (essentiality = priority)",
    "top",
    "# 2) documentation for one process (real data: role, segment, scheduler, interfaces)",
    "man TP53",
    "# 2a) DECODE a gene's pathway from DATA (co-dependency neighbours), not a curated label — works on DARK genes",
    "pathway NEPRO",
    "# 2b) PubMed literature for a DARK gene — the knowledge screens can't reach (grounded + cited)",
    "lit FNDC5",
    "# 3) THE DEBUGGER — SIGKILL a gene, watch the MEASURED downstream effect (interventional, not correlation)",
    "strace SF3B1",
    "# 4) THE DETECTIVE — recover the CAUSE from the fingerprint a knockout leaves (causal, self-consistency proof)",
    "whodunit SF3B1",
    "# 4b) PREDICT THE NEXT THING — an unmeasured knockout's response, transformer-style, with a live self-check",
    "predict PSMB5",
    "# 4c) THE WIRED CELL — the whole complete-cell database as ONE graph: INTERCONNECTED (physical, who touches whom)",
    "#     + INTERDEPENDENT (functional, who needs/controls whom). Complementary (~6% overlap) yet coherent (46-214x).",
    "network TP53",
    "network link MDM2 TP53",
    "network path EGFR TP53 interdependent",
    "network hubs interdependent",
    "# 4d) MEASURED KNOCKOUT EFFECT — the real Perturb-seq blast radius of removing a gene (up/down), + impact rank.",
    "#     Direct effect is real; it does NOT propagate to unmeasured cascades (transitivity ~chance).",
    "knockout SF3B1",
    "knockout impact",
    "# 4e) KNOCK OUT THE PROTEIN, NOT THE GENE — degrade the molecule (PROTAC-style): the STRUCTURAL disassembly,",
    "#     every machine it is a subunit of loses a part at once. Complements the transcriptional gene knockout above.",
    "protein SMARCA4",
    "protein PSMA1",
    "# 4f) THE INFLUENCE NETWORK + CASCADE — perturbation as 'what a knockout AFFECTS' (we know the ends), then the",
    "#     route of each effect reconstructed from structure + OTHER measured knockouts: direct / mediated / unresolved.",
    "cascade GATA1",
    "cascade SF3B1 RBM22",
    "# 4g) THE FOUR CANONICAL PERTURB-SEQ APPLICATIONS, attempted on this data and honestly graded: GRN wiring (fails",
    "#     on single-condition data), gene function by fingerprint (WORKS), disease convergence (thin), immune (n/a).",
    "pertseq",
    "pertseq function TIMM23B",
    "# 4c) REASON across independent lines to a conclusion, with CALIBRATED confidence (the layer above data)",
    "reason RPL13",
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
    "# 7b) VALIDATE A PARAMETER — put in a kcat; the RUNNING CELL flags it if impossibly slow (can't carry flux)",
    "check SLC2A8 0.001",
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
