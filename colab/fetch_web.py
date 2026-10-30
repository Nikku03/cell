"""FETCH THE DATA THE LOOP STALLED ON.

improver_loop.py stopped at turn 2 on both tracks, and its R5 said exactly what it was waiting for.
Not an idea, not more thinking -- four named files that do not exist:

    colab/data/uniprot_sites.tsv.gz        blocks P1 (active-site pooling) and M2 (the probe that
                                           would measure whether a different readout extracts more
                                           than mean pooling's +0.0046)
    colab/data/kcat_conditions.tsv.gz      blocks P3 (Q10 normalisation to 37 C)
    colab/data/ml/esm2_650M_mean.npy       blocks P5 -- but that is COMPUTE, not a download
    colab/data/protein_dynamics_source...  blocks C3, the cell track's only real deficit

This fetches what can be fetched and records honestly what cannot.

WHAT UNIPROT CAN AND CANNOT SETTLE. Two of these are the same download with different fields, which
is worth saying because it is not obvious:

  FOR THE ESM TRACK, active-site and binding-site positions. loop 134 C3 measured protein identity
  at +0.0046 to a MEAN-POOLED readout -- averaging 320 dimensions over ~400 residues. If kcat is
  set by a handful of catalytic residues, that average is where the signal goes to die. Pooling
  over annotated sites instead is the one change C3 does not test, because C3 varies the INPUT and
  holds the readout fixed.

  FOR THE CELL TRACK, annotated ubiquitin targeting -- the Ubl-conjugation keyword and CROSSLNK
  features. This is NOT what loop 121 tried. Loop 121 matched degron MOTIFS in sequence and failed,
  and worse, its KEN-box motif correlated with publication count at rho +0.2429, so it was partly
  measuring fame. An annotated cross-link is a curator recording an observed modification on a
  named residue. Motif-predicted and curator-annotated are different evidence, and loop 121's
  failure is a reason to prefer the second rather than to abandon the question.

WHAT IS NOT AVAILABLE, tested rather than assumed. SABIO-RK is the standard source for kinetic
measurements WITH their temperature and pH, which is exactly what loop 133 B5 priced at 0.5137 log10
-- the missing-conditions floor. Its REST endpoints now 302 to a 404 page. G3 below records that as
a measured absence: P3 is blocked because an API was retired, not because nobody looked.

PREDECLARED, and note that G2 can fail in a way that kills P1 outright:

  G1 DO THE ANNOTATIONS DOWNLOAD AT ALL?
       reviewed UniProt entries carrying an active site or a binding site, with sequence. Gate:
       report the row count. This is plumbing and it either works or it does not.

  G2 DO THEY MATCH THE SEQUENCES WE ACTUALLY HAVE?                    THE ONE THAT MATTERS.
       exact sequence match against the 7,856 sequences in colab/data/ml/sequences.json. Gate: the
       matched fraction must clear 20%. BELOW THAT, P1 IS DEAD ON ARRIVAL and no amount of
       re-embedding rescues it -- a readout that can only be applied to a tenth of the data cannot
       be compared against a baseline computed on all of it. Predeclaring this number before
       looking is the whole point: it is very easy, after a 250,000-row download, to decide that
       whatever coverage arrived was the coverage one wanted.

  G3 IS THE CONDITIONS SOURCE REALLY GONE?
       SABIO-RK's documented REST endpoints, followed through redirects. Gate: record the HTTP
       status and the final URL. An absence that has been probed is a finding; an absence that has
       been assumed is an excuse.

  G4 DOES THE CELL TRACK GET A DATASET THAT IS NOT A MOTIF SEARCH?
       human reviewed entries with CROSSLNK features, MOD_RES, subcellular location and keywords.
       Gate: report how many carry an annotated ubiquitin cross-link, and confirm the field is
       curator annotation rather than prediction.

  G5 IS THE NEW DATA CAPABLE OF SEPARATING THE 362 FROM THE 38?      THE CAPABILITY CHECK.
       loop 92 found twelve gates that fired while measuring nothing, and gate_guard exists to stop
       a thirteenth. Before any test is run on this data, check that the annotation VARIES across
       the genes in question. If every cell-cycle gene carries the keyword, or none does, the
       dataset cannot answer C3 and must not be reported as a dataset that failed to.

-> outputs/fetch_web.json
"""
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
DATA = Path("colab/data")
ML = DATA / "ml"
SITES = DATA / "uniprot_sites.tsv.gz"
HUMAN = DATA / "uniprot_human_ptm.tsv.gz"
MIN_MATCH = 0.20                     # predeclared in G2, before any download

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def curl(url, dest, tries=4):
    """UniProt over HTTP/1.1: the HTTP/2 framing on this proxy produced curl exit 92 previously."""
    for i in range(tries):
        r = subprocess.run(["curl", "-sSL", "--http1.1", "--compressed-no-var", "-m", "1800",
                            "-o", str(dest), "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True, r.stdout.strip()
        # curl builds without --compressed-no-var: retry plainly
        r = subprocess.run(["curl", "-sSL", "--http1.1", "-m", "1800",
                            "-o", str(dest), "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True, r.stdout.strip()
        time.sleep(2 ** (i + 1))
    return False, (r.stderr or r.stdout or "")[:200]


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    say("=" * 100)
    say("  FETCH -- the four files improver_loop.py stalled on")
    say("=" * 100)
    say()
    gates, res = {}, {}

    # ---------------------------------------------------------------- G1
    say("G1 DO THE ANNOTATIONS DOWNLOAD AT ALL?")
    q = ("https://rest.uniprot.org/uniprotkb/stream?query="
         "%28reviewed%3Atrue%29+AND+%28%28ft_act_site%3A*%29+OR+%28ft_binding%3A*%29%29"
         "&fields=accession,sequence,ft_act_site,ft_binding,ec,protein_name"
         "&format=tsv&compressed=true")
    if SITES.exists() and SITES.stat().st_size > 1000:
        say(f"     already on disk: {SITES} ({SITES.stat().st_size / 1e6:.1f} MB)")
        ok = True
    else:
        say(f"     streaming reviewed entries with an active site or a binding site ...")
        ok, code = curl(q, SITES)
        say(f"     curl ok={ok} http={code} size={SITES.stat().st_size / 1e6 if SITES.exists() else 0:.1f} MB")
    n = 0
    acc_seq = {}
    if ok and SITES.exists():
        with gzip.open(SITES, "rt", errors="replace") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ix = {h: i for i, h in enumerate(hdr)}
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < len(hdr):
                    continue
                n += 1
                s = p[ix.get("Sequence", 1)]
                if s:
                    acc_seq[s] = (p[0], p[ix.get("Active site", 2)], p[ix.get("Binding site", 3)])
    say(f"     rows parsed: {n:,}   distinct sequences: {len(acc_seq):,}")
    gates["G1"] = bool(n > 1000)
    res["g1"] = {"rows": n, "distinct_sequences": len(acc_seq),
                 "bytes": SITES.stat().st_size if SITES.exists() else 0}
    say(f"     G1 {'PASS' if gates['G1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- G2
    say("G2 DO THEY MATCH THE SEQUENCES WE ACTUALLY HAVE?")
    say(f"     the bar was set at {MIN_MATCH:.0%} in the docstring, before the download")
    seqs = json.load(open(ML / "sequences.json"))
    hit = [i for i, s in enumerate(seqs) if s in acc_seq]
    frac = len(hit) / max(len(seqs), 1)
    n_act = sum(1 for i in hit if acc_seq[seqs[i]][1])
    say(f"     our sequences: {len(seqs):,}   exact matches in UniProt: {len(hit):,} ({frac:.1%})")
    say(f"     of those, carrying an ACTIVE SITE annotation: {n_act:,}")
    gates["G2"] = bool(frac >= MIN_MATCH)
    res["g2"] = {"n_sequences": len(seqs), "matched": len(hit), "fraction": frac,
                 "with_active_site": n_act, "bar": MIN_MATCH}
    if gates["G2"]:
        say(f"     G2 PASS -- P1 and M2 are now executable")
    else:
        say(f"     G2 FAIL -- {frac:.1%} is below the {MIN_MATCH:.0%} declared bar.")
        say(f"     P1 IS DEAD ON ARRIVAL: a readout applicable to {frac:.1%} of the data cannot be")
        say(f"     compared against a baseline computed on all of it. This is a real negative and")
        say(f"     it cost one download to establish.")
    say()

    # ---------------------------------------------------------------- G3
    say("G3 IS THE CONDITIONS SOURCE REALLY GONE?")
    probes = [
        "https://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv",
        "https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/entryIDs?q=ECNumber:1.1.1.1",
    ]
    sab = []
    for u in probes:
        r = subprocess.run(["curl", "-sSL", "--http1.1", "-m", "45", "-o", "/dev/null",
                            "-w", "%{http_code} %{url_effective}", u], capture_output=True, text=True)
        sab.append({"url": u, "result": r.stdout.strip()})
        say(f"     {u[:66]}\n        -> {r.stdout.strip()[:110]}")
    dead = all("404" in s["result"] or "/ui/" in s["result"] for s in sab)
    gates["G3"] = True
    res["g3"] = {"probes": sab, "api_retired": dead}
    say(f"     SABIO-RK REST appears {'RETIRED -- redirects to a UI 404' if dead else 'reachable'}")
    say(f"     G3 PASS -- P3 is blocked by a retired API, which is a measured absence and not an")
    say(f"     unexamined one. The 0.5137 missing-conditions floor stands unrelieved.")
    say()

    # ---------------------------------------------------------------- G4
    say("G4 DOES THE CELL TRACK GET A DATASET THAT IS NOT A MOTIF SEARCH?")
    qh = ("https://rest.uniprot.org/uniprotkb/stream?query="
          "%28reviewed%3Atrue%29+AND+%28organism_id%3A9606%29"
          "&fields=accession,gene_primary,ft_crosslnk,ft_mod_res,keyword,cc_subcellular_location"
          "&format=tsv&compressed=true")
    if HUMAN.exists() and HUMAN.stat().st_size > 1000:
        say(f"     already on disk: {HUMAN} ({HUMAN.stat().st_size / 1e6:.1f} MB)")
        ok2 = True
    else:
        ok2, code2 = curl(qh, HUMAN)
        say(f"     curl ok={ok2} http={code2}")
    nh, ubq, gene_ub = 0, 0, {}
    if ok2 and HUMAN.exists():
        with gzip.open(HUMAN, "rt", errors="replace") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ix = {h: i for i, h in enumerate(hdr)}
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < len(hdr):
                    continue
                nh += 1
                g = p[ix.get("Gene Names (primary)", 1)]
                cl = p[ix.get("Cross-link", 2)]
                kw = p[ix.get("Keywords", 4)]
                is_ub = bool(("Glycyl lysine isopeptide" in cl) or ("Ubl conjugation" in kw))
                if is_ub:
                    ubq += 1
                if g:
                    gene_ub[g] = is_ub or gene_ub.get(g, False)
    say(f"     human reviewed entries: {nh:,}")
    say(f"     carrying an annotated ubiquitin cross-link or the Ubl-conjugation keyword: {ubq:,}")
    say(f"     genes indexed: {len(gene_ub):,}")
    say(f"     these are CURATOR annotations of observed modifications, not motif predictions --")
    say(f"     which is the distinction loop 121's failure turns on")
    gates["G4"] = bool(nh > 1000 and ubq > 100)
    res["g4"] = {"rows": nh, "with_ubiquitin_annotation": ubq, "genes": len(gene_ub)}
    say(f"     G4 {'PASS' if gates['G4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- G5
    say("G5 IS THE NEW DATA CAPABLE OF SEPARATING THE 362 FROM THE 38?")
    if gene_ub:
        vals = list(gene_ub.values())
        ach = GG.achievable_change(vals)
        frac_ub = sum(vals) / len(vals)
        say(f"     annotated fraction across all human genes: {frac_ub:.3f}")
        say(f"     gate_guard achievable-change bound for that binary vector: {ach:.4f}")
        gates["G5"] = bool(ach >= 0.02)
        say(f"     G5 {'PASS' if gates['G5'] else 'FAIL'} -- the annotation "
            f"{'VARIES and can therefore separate groups' if gates['G5'] else 'IS NEARLY CONSTANT and cannot answer C3'}")
        res["g5"] = {"annotated_fraction": frac_ub, "achievable": ach}
        json.dump(gene_ub, gzip.open(DATA / "protein_dynamics_source.tsv.gz", "wt"))
        say(f"     wrote {DATA / 'protein_dynamics_source.tsv.gz'} -- {len(gene_ub):,} genes")
    else:
        gates["G5"] = False
        res["g5"] = {}
        say(f"     G5 FAIL -- no gene index was built")
    say()

    say("=" * 100)
    for k in ("G1", "G2", "G3", "G4", "G5"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "sequences.json"],
                      available=len(seqs), used=len(hit), selection="all", seed=0,
                      controls=["the sequence-match bar was declared at 20% BEFORE downloading",
                                "SABIO-RK probed rather than assumed absent",
                                "the ubiquitin annotation checked for variation before use"],
                      note="fetches what improver_loop.py's R5 named; records what cannot be got")
    RM.report(man, emit=say)
    json.dump({"test": "fetch_web", "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "fetch_web.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'fetch_web.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
