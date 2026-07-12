"""cellos_test — honest test of CellOS: not 'does it run' but 'are its answers RIGHT'.

Three layers:
  A) SMOKE — every syscall + edge cases (unknown gene, missing arg, gene not in screen) returns without crashing.
  B) CORRECTNESS ASSERTIONS — the debugger recovers known biology (SF3B1->spliceosome, PSMB5->proteasome,
     FANCI->Fanconi), the mutation lint calls a suppressor a CRASH, and the security layer classifies curated vs
     novel + flags hubs.
  C) QUANTIFIED BENCHMARK — whodunit functional-coherence over a large sample: what fraction of each knockout's
     top-8 implicated genes share a complex/pathway with it, versus a random baseline (enrichment).
-> prints PASS/FAIL summary + the coherence number.
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cellos import CellKernel, CellShell


def main():
    k = CellKernel(quiet=True)
    sh = CellShell(k)
    C = k.C
    fails = []

    # ---------------------------------------------------------------- A) smoke + edge cases
    print("A) SMOKE TESTS (runs without crashing, returns output)")
    smoke = {
        "help": "help", "stat": "stat", "top": "top",
        "viability lethal": "viability RAE1", "viability robust": "viability SLC22A1",
        "viability non-metabolic": "viability TP53",
        "boot bare": "boot", "boot knockout": "boot TP53", "boot off-core": "boot NOTAGENE",
        "induce master TF": "induce GATA1", "reprogram alias": "reprogram MYOD1", "induce non-TF": "induce TTN",
        "lit dark gene": "lit FNDC5", "lit unmined": "lit ZZZ",
        "man known": "man TP53", "man unknown": "man NOTAGENE",
        "strace in-screen": "strace SF3B1", "strace not-in-screen": "strace TP53",
        "whodunit in-screen": "whodunit SF3B1", "whodunit not-in-screen": "whodunit TP53",
        "predict in-complex": "predict PSMB5", "predict singleton": "predict TTN",
        "simulate combo": "simulate SF3B1 PSMB5", "cure": "cure up=CCND1,MYC down=CDKN1A",
        "diagnose": "diagnose up=HBA1,HBB down=CDK1,CCNB1",
        "deadlock has-partners": "deadlock FANCI", "deadlock none": "deadlock ACTB",
        "patch suppressor": "patch TP53:R175H", "patch tolerant": "patch TTN:V100A",
        "lint curated": 'lint "TP53 binds MDM2"', "lint novel": 'lint "TP53 binds MRPS25"',
        "lint unparseable": 'lint "the cell is complex"',
        "bad command": "frobnicate X", "missing arg": "man",
    }
    for name, line in smoke.items():
        try:
            out = sh.run_line(line)
            ok = out is not None and len(str(out)) > 0 and "kernel fault" not in str(out)
        except Exception as e:
            ok, out = False, f"EXCEPTION {e}"
        if not ok:
            fails.append(f"smoke:{name}")
        print(f"   [{'ok ' if ok else 'FAIL'}] {name:22} {str(out).splitlines()[0][:58]}")

    # ---------------------------------------------------------------- B) correctness assertions
    print("\nB) CORRECTNESS ASSERTIONS (does it recover KNOWN biology?)")

    def check(name, cond, detail=""):
        print(f"   [{'ok ' if cond else 'FAIL'}] {name:32} {detail}")
        if not cond:
            fails.append(name)

    # whodunit recovers the functional complex
    def hits(gene, n=8):
        h = k.whodunit_hits(gene, k=n)
        return [g for g, _ in h[2]] if isinstance(h, tuple) else []
    sf = hits("SF3B1"); spliceo = sum(1 for g in sf if g.startswith(("SF3", "SNRP", "SLU", "PRPF", "U2AF", "SF1")))
    check("whodunit SF3B1 -> spliceosome", spliceo >= 3, f"{spliceo}/8 splicing genes: {sf[:5]}")
    ps = hits("PSMB5"); proteas = sum(1 for g in ps if g.startswith(("PSM", "POMP")))
    check("whodunit PSMB5 -> proteasome", proteas >= 3, f"{proteas}/8 proteasome genes: {ps[:5]}")
    rp = hits("RPL13"); ribo = sum(1 for g in rp if g.startswith(("RPL", "RPS")))
    check("whodunit RPL13 -> ribosome", ribo >= 3, f"{ribo}/8 ribosomal genes: {rp[:5]}")

    # predict: an unmeasured knockout's response, self-checked. PSMB5 (proteasome) should recover strongly.
    pp = k.predict("PSMB5")
    check("predict PSMB5 self-check r>0.4", "RECOVERED" in pp, "predicted-vs-real correlation")
    check("predict PSMB5 -> proteostasis stress", any(s in pp for s in ("HSPA1A", "UBC", "SQSTM1")),
          "predicted UP = heat-shock/ubiquitin stress")

    # simulate: single measured KO is EXACT (matches strace); combo is additive
    sim1 = k.simulate(["PSMB5"])
    check("simulate PSMB5 exact (measured)", "MEASURED (exact)" in sim1 and "HSPA1A" in sim1, "single = measured")
    simc = k.simulate(["SF3B1", "PSMB5"])
    check("simulate combo is additive+flagged", "ADDITIVE" in simc and "synergistic" in simc)
    # cure returns a combination prescription
    cu = k.cure(up=["CCND1", "MYC"], down=["CDKN1A"])
    check("cure -> combination prescription", "prescription: co-knockout" in cu)

    # viability: objective-driven FBA calls a bottleneck lethal, a redundant gene survivable
    check("viability RAE1 -> LETHAL", "LETHAL" in k.viability("RAE1"))
    check("viability SLC22A1 -> SURVIVES", "SURVIVES" in k.viability("SLC22A1"))
    check("viability TP53 -> non-metabolic", "not in the metabolic model" in k.viability("TP53"))

    # lit: PubMed literature for a dark gene, grounded + cited
    lg = k.lit("FNDC5")
    check("lit FNDC5 -> papers + DOI", "PubMed papers" in lg and "doi:" in lg)

    # boot: the RUNNING cell converges to an attractor, and a hub TF perturbs it more than an off-core gene
    bb = k.boot()
    check("boot -> converges to attractor", "converged in" in bb and "self-consistent cell state" in bb)
    check("boot -> honest essentiality caveat", "does NOT predict which genes are essential" in bb)
    b_hub = k.boot("TP53"); b_leaf = k.boot("SLC22A1")
    def _disp(s):
        import re
        m = re.search(r"displacement ([\d.]+)", s); return float(m.group(1)) if m else -1.0
    check("boot TP53 (hub) more disruptive than a leaf", _disp(b_hub) > _disp(b_leaf),
          f"TP53 disp {_disp(b_hub):.2f} vs SLC22A1 {_disp(b_leaf):.2f}")

    # induce: forcing a master TF ON specifically reprograms its lineage; a structural gene gets no lineage
    ig = k.induce(["GATA1"])
    check("induce GATA1 -> erythroid SPECIFIC", "SPECIFIC (reprogrammed)" in ig and "[GATA1]" in ig)
    it = k.induce(["TTN"])
    check("induce non-master TTN -> no lineage label", "SPECIFIC (reprogrammed)" not in it)

    # deadlock recovers the pathway
    dl = k.deadlock("FANCI")
    check("deadlock FANCI -> Fanconi pathway", "FANCD2" in dl and "FANCA" in dl, "contains FANCD2/FANCA")

    # patch: suppressor is a CRASH, tolerant is not
    check("patch TP53:R175H -> CRASH/LOF", "CRASH" in k.patch("TP53:R175H"))

    # security layer
    lc = k.lint("TP53 binds MDM2"); ln = k.lint("TP53 binds MRPS25")
    check("lint curated -> TRUSTED", "TRUSTED" in lc and "in curated" in lc)
    check("lint novel -> UNTRUSTED", "UNTRUSTED" in ln)
    check("lint flags hub (TP53)", "HUB" in lc)
    # a non-hub curated pair should NOT be flagged as hub
    lo = k.lint("PAH binds GCH1") if k._pid("PAH") and k._pid("GCH1") else "HUB"
    check("lint non-hub not flagged", "HUB" not in lo or "neither endpoint a hub" in lo, "(PAH/GCH1)")

    # strace fallback is honestly labelled when a gene is not in the screen
    st = k.strace("TP53")
    check("strace off-screen -> flagged [~]", "[~]" in st, "correlational fallback labelled")

    # ---------------------------------------------------------------- C) quantified coherence benchmark
    print("\nC) QUANTIFIED BENCHMARK — whodunit functional coherence vs random")
    dbg = k._load_debugger()
    g2c = {int(x): set(v) for x, v in (C.D.get("gene2cplx", {}) or {}).items()}

    def paths_of(name):
        i = C.idx.get(name)
        if i is None:
            return set()
        p = C.genes[i].get("path") or []
        p = [p] if isinstance(p, str) else p
        cx = g2c.get(i, set())
        return set(str(x) for x in p) | set(f"cplx{c}" for c in cx)

    screen = [g for g in dbg["pgenes"] if paths_of(g)]           # screen genes with any annotation
    rng = random.Random(0)
    sample = rng.sample(screen, min(120, len(screen)))
    coh, base, hitrate = [], [], 0
    allscreen = dbg["pgenes"]
    for g in sample:
        gp = paths_of(g)
        top = hits(g, 8)
        shared = [t for t in top if paths_of(t) & gp]
        coh.append(len(shared) / max(len(top), 1))
        if shared:
            hitrate += 1
        # random baseline: 8 random screen genes
        rnd = rng.sample(allscreen, 8)
        base.append(sum(1 for t in rnd if paths_of(t) & gp) / 8)
    mc, mb = float(np.mean(coh)), float(np.mean(base))
    print(f"   sampled {len(sample)} annotated knockouts")
    print(f"   whodunit top-8 share a complex/pathway with the query:  {mc:.1%}")
    print(f"   random baseline (8 random knockouts):                    {mb:.1%}")
    print(f"   enrichment: {mc/max(mb,1e-6):.1f}x   |   >=1 coherent hit in top-8: {hitrate/len(sample):.0%}")
    check("whodunit coherence >> random", mc > max(3 * mb, 0.2), f"{mc:.0%} vs {mb:.0%}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 60)
    if fails:
        print(f"RESULT: {len(fails)} FAILURE(S): {fails}")
    else:
        print("RESULT: ALL PASS — CellOS runs, and its causal answers recover known biology.")
    print("=" * 60)
    return not fails


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
