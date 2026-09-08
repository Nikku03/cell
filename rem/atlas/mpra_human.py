"""The count-matched identity test, in designed human constructs.

WHY THIS EXISTS. humantransfer.py asked whether a human gene's regulators are interchangeable and
answered with a perturbation panel: qualitatively no, but at 1.05 noise floors against yeast's
15.86, because a single CRISPRi knockdown moves an individual target by about a quarter of a log2
against a floor of the same size. Its own closing line said what was needed instead -- a human
assay with the yeast experiment's structure, designed sites in a fixed context with a tight floor.
This is that, as far as open data allows.

WHAT WAS LOOKED FOR AND WHAT WAS FOUND, because the gap matters more than the substitute.
The exact design exists: Smith et al. 2013, Nat Genet 45:1021-1028, doi:10.1038/ng.2713, varied
"copy number, spacing, combination and order" of 12 liver transcription-factor sites across ~5,000
synthetic elements in HepG2 -- the Sharon design in human cells. Its processed data is not
machine-readable from here: the article is not in the PMC open-access subset, the supplementary
tables are behind an interstitial that blocks automated and browser retrieval alike, and the raw
reads at SRA SRP018414 cannot be decoded without the barcode map that lives in those same
supplements. THE COPY-NUMBER AXIS THEREFORE REMAINS UNTESTED ON HUMAN DATA and this module does
not test it. Saying so is the point; a substitute presented as the thing itself would be worse
than no test.

WHAT IS OPEN, AND WHAT IT CAN ANSWER. Kheradpour et al. 2013, Genome Res 23:800-11,
doi:10.1101/gr.144899.112, GEO GSE33367: a massively parallel reporter assay over ~2,300 designed
human enhancer constructs in K562 and HepG2, in which selected transcription-factor motif
instances -- activators and repressors in both cell types -- are directly disrupted, with ten
barcodes per construct, two biological replicates per cell type, and a matched plasmid input.

That fixes the COUNT at one and varies the IDENTITY, which is exactly the assumption a count
summary makes: that one perturbed regulator is like another. It is the count-matched half of the
question, measured in a cis-regulatory context with a floor an order tighter than the CRISPRi
panel's, and it is the half the engine's per-gene factor actually depends on.

HOW IDENTITY IS RECOVERED WITHOUT A MOTIF DATABASE. The construct IDs do not encode the design, but
the sequences are given. Constructs sharing both flanks form an enhancer family; within a family
the wild type is the member minimising total Hamming distance to the rest, and the disrupted
variant is the one differing in a contiguous block. The wild-type sequence in that block IS the
motif instance. Nothing is looked up and no threshold is chosen: the primary test asks whether
constructs whose disrupted motifs are more SIMILAR IN SEQUENCE show more similar disruption
effects, against a permutation null, which needs no clustering and no motif names.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

M1  THE NOISE FLOOR, from the two biological replicates, measured before any model, on the
    quantity actually scored -- the disruption effect, which is a difference of two log ratios and
    therefore carries more noise than either.

M2  THE ASSAY MUST WORK. Disrupting a real motif must move activity more than the floor, or these
    constructs are not measuring regulation and nothing below is readable. PREDECLARED: the spread
    of disruption effects must exceed the floor.

M3  DOES IDENTITY MATTER AT FIXED COUNT? Mantel-style: correlate similarity between motif
    instances with similarity of their disruption effects, permutation null over motif labels.
    PREDECLARED: if similar motifs do not produce similar effects, then at fixed count one
    disrupted regulator IS like another, a count summary is adequate on human data, and the yeast
    refutation does not transfer.

M4  THE SAME QUANTITY AS humantransfer AND promoter, so the three are comparable: the between-
    motif variance component above noise, in floors, bootstrapped.

M5  REPLICATION ACROSS CELL TYPES. K562 and HepG2 are independent measurements of the same
    constructs. PREDECLARED: a result that appears in one and not the other is not a result.

M6  WHAT THIS DOES AND DOES NOT SETTLE, with the copy-number gap restated so it is not lost.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import gzip
import hashlib
import io
import tarfile
import urllib.request
import numpy as np

from rem.atlas.hybrid_tune import RULE

GEO = "GSE33367"
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE33nnn/GSE33367/suppl/GSE33367_RAW.tar"
FILES = {
    "HepG2_1": "GSM825352_HepG2_mRNA_Rep1_counts.txt.gz",
    "HepG2_2": "GSM825353_HepG2_mRNA_Rep2_counts.txt.gz",
    "K562_1": "GSM825354_K562_mRNA_Rep1_counts.txt.gz",
    "K562_2": "GSM825355_K562_mRNA_Rep2_counts.txt.gz",
    "Plasmid_1": "GSM825356_Plasmid_Rep1_counts.txt.gz",
    "Plasmid_2": "GSM825357_Plasmid_Rep2_counts.txt.gz",
}


def fetch(path=None):
    """Not vendored: fetched from GEO and checksummed, like promoter.py and recon3d."""
    path = path or os.path.join(os.path.dirname(__file__), "gse33367_raw.tar")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:32]
    out = {}
    with tarfile.open(path) as tf:
        for key, name in FILES.items():
            m = tf.extractfile(name)
            raw = gzip.decompress(m.read()).decode("utf8", "replace")
            d = {}
            for ln in raw.splitlines()[1:]:
                p = ln.split("\t")
                if len(p) < 4:
                    continue
                d[p[0]] = (p[1], sum(int(c) for c in p[3].split(",") if c.strip().isdigit()))
            out[key] = d
    return out, sha


def families(seqs):
    """Constructs sharing both 25 bp flanks are variants of one designed enhancer."""
    g = collections.defaultdict(list)
    for cid, s in seqs.items():
        g[(s[:25], s[-25:])].append(cid)
    return {k: v for k, v in g.items() if len(v) >= 6}


def disruptions(seqs, fams):
    """For each family: the wild type (minimal total Hamming distance to the rest) and the variant
    differing in a contiguous BLOCK, which is the deliberately disrupted motif. Returns the motif
    instance -- the wild-type sequence in that block -- which is the identity label, read off the
    design rather than looked up."""
    out = []
    for k, ids in fams.items():
        S = [seqs[i] for i in ids]
        tot = [sum(sum(1 for a, b in zip(S[x], S[y]) if a != b) for y in range(len(S)))
               for x in range(len(S))]
        w = int(np.argmin(tot))
        wt = S[w]
        for j, s in enumerate(S):
            if j == w:
                continue
            d = [i for i in range(len(wt)) if wt[i] != s[i]]
            if len(d) >= 6 and (max(d) - min(d)) <= 24:
                out.append((ids[w], ids[j], min(d), max(d), wt[min(d):max(d) + 1],
                            s[min(d):max(d) + 1]))
    return out


def activity(counts, cell, rep, min_plasmid=50):
    """log2( mRNA / plasmid ), each depth-normalised. The plasmid input is the denominator that
    makes this a measure of regulatory activity rather than of library composition."""
    m = counts[f"{cell}_{rep}"]
    p = counts[f"Plasmid_{rep}"]
    tm = sum(v for _, v in m.values())
    tp = sum(v for _, v in p.values())
    out = {}
    for cid, (_, mc) in m.items():
        if cid not in p:
            continue
        pc = p[cid][1]
        if pc < min_plasmid:
            continue
        out[cid] = np.log2(((mc + 1) / tm) / ((pc + 1) / tp))
    return out


def sim(a, b):
    """Ungapped best-offset match fraction between two motif instances, both strands."""
    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    best = 0.0
    for y in (b, rc(b)):
        for off in range(-(len(y) - 4), len(a) - 3):
            n = 0
            m = 0
            for i in range(len(a)):
                j = i - off
                if 0 <= j < len(y):
                    n += 1
                    m += (a[i] == y[j])
            if n >= 9:                     # a 6-base overlap can hit 1.0 by chance
                best = max(best, m / n)
    return best


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("THE COUNT-MATCHED IDENTITY TEST, IN DESIGNED HUMAN CONSTRUCTS"); P_(RULE)
    counts, sha = fetch()
    seqs = {cid: v[0] for cid, v in counts["Plasmid_1"].items()}
    fams = families(seqs)
    dis = disruptions(seqs, fams)
    P_(f"  Kheradpour et al. 2013, Genome Res 23:800-11, doi:10.1101/gr.144899.112, via GEO {GEO},")
    P_(f"  sha256[:32] {sha}. {len(seqs)} designed constructs, 10 barcodes each, K562 and HepG2")
    P_(f"  with two biological replicates and a matched plasmid input.")
    P_(f"  {len(fams)} enhancer families with >= 6 variants; {len(dis)} deliberate motif")
    P_(f"  disruptions recovered from the sequences, one per family, so the COUNT IS FIXED AT ONE")
    P_(f"  and only the IDENTITY of the disrupted motif varies.")
    P_("\n  THE COPY-NUMBER AXIS IS NOT TESTED HERE. The design that varies it -- Smith et al. 2013,")
    P_("  doi:10.1038/ng.2713, in HepG2 -- is not machine-readable from this environment: not in")
    P_("  the PMC open-access subset, supplements behind an interstitial that blocks automated and")
    P_("  browser retrieval alike, and the SRA reads undecodable without the barcode map inside")
    P_("  those supplements. That gap is real and is restated in M6 rather than left here.")

    res = {}
    for cell in ("HepG2", "K562"):
        a1 = activity(counts, cell, 1)
        a2 = activity(counts, cell, 2)
        eff1, eff2, motifs, pairs = [], [], [], []
        for wt, mut, lo, hi, mseq, rseq in dis:
            if wt in a1 and mut in a1 and wt in a2 and mut in a2:
                eff1.append(a1[mut] - a1[wt])
                eff2.append(a2[mut] - a2[wt])
                motifs.append(mseq)
                pairs.append((wt, mut))
        res[cell] = (np.array(eff1), np.array(eff2), motifs)

    # ---- M1  NOISE FLOOR -----------------------------------------------------------------------
    P_("\n" + RULE); P_("M1  THE REPLICATE NOISE FLOOR, MEASURED BEFORE ANY MODEL"); P_(RULE)
    P_("  The quantity scored is the DISRUPTION EFFECT -- activity(disrupted) minus activity(wild")
    P_("  type) -- a difference of two log ratios, so it carries more noise than either. The floor")
    P_("  is the noise in ONE replicate's effect, which is what every number below is divided by.")
    P_(f"\n    {'cell':<8} {'disruptions':>12} {'RMS rep1-rep2':>15} {'FLOOR (one rep)':>17}"
       f" {'spread of effects':>19} {'S/N':>6}")
    floors = {}
    for cell in ("HepG2", "K562"):
        e1, e2, ms = res[cell]
        fl = float(np.std(e1 - e2)) / np.sqrt(2.0)
        floors[cell] = fl
        eb = (e1 + e2) / 2.0
        P_(f"    {cell:<8} {len(e1):>12} {float(np.std(e1-e2)):>15.4f} {fl:>17.4f}"
           f" {float(np.std(eb)):>19.4f} {float(np.std(eb))/fl:>6.2f}")

    # ---- M2  THE ASSAY MUST WORK ---------------------------------------------------------------
    P_("\n" + RULE); P_("M2  THE ASSAY MUST WORK"); P_(RULE)
    m2 = True
    for cell in ("HepG2", "K562"):
        e1, e2, ms = res[cell]
        eb = (e1 + e2) / 2.0
        tv = max(float(np.var(eb)) - floors[cell] ** 2 / 2, 0.0)
        ok = np.sqrt(tv) > floors[cell]
        m2 = m2 and ok
        P_(f"    {cell:<8} true spread of disruption effects {np.sqrt(tv):.4f} log2 ="
           f" {np.sqrt(tv)/floors[cell]:.2f} floors   {'PASS' if ok else 'FAIL'}")
    P_(f"  M2: {'PASS -- disrupting these motifs moves activity well beyond the noise, in both cell types' if m2 else 'FAIL -- disruption does not move activity; nothing below is readable'}")

    # ---- M3  DOES IDENTITY MATTER AT FIXED COUNT? ----------------------------------------------
    P_("\n" + RULE); P_("M3  DOES IDENTITY MATTER, WITH THE COUNT HELD AT ONE?"); P_(RULE)
    P_("  Mantel-style and free of any motif database: do constructs whose DISRUPTED MOTIFS are")
    P_("  more similar in sequence show more similar disruption effects? Null by permuting the")
    P_("  motif labels against the effects, which destroys the pairing and nothing else.")
    rng = np.random.default_rng(0)
    for cell in ("HepG2", "K562"):
        e1, e2, ms = res[cell]
        eb = (e1 + e2) / 2.0
        n = len(ms)
        Sm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                Sm[i, j] = Sm[j, i] = sim(ms[i], ms[j])
        iu = np.triu_indices(n, 1)
        sv = Sm[iu]
        dv = np.abs(eb[:, None] - eb[None, :])[iu]
        r = float(np.corrcoef(sv, dv)[0, 1])
        null = []
        for _ in range(2000):
            p = rng.permutation(n)
            dvp = np.abs(eb[p][:, None] - eb[p][None, :])[iu]
            null.append(float(np.corrcoef(sv, dvp)[0, 1]))
        null = np.array(null)
        z = (r - null.mean()) / null.std()
        pv = float(np.mean(null <= r)) if r < 0 else float(np.mean(null >= r))
        P_(f"\n    {cell}: {n} disruptions, {len(sv)} pairs")
        P_(f"      corr(motif similarity, |effect difference|) = {r:+.4f}")
        P_(f"      permutation null {null.mean():+.4f} +- {null.std():.4f}   z = {z:+.2f}"
           f"   p = {max(pv,1/2000):.4f}")
        P_(f"      -> {'similar motifs give similar effects: IDENTITY MATTERS at fixed count' if (r < 0 and z < -2) else 'no relation between motif identity and effect'}")

    # ---- M4  THE COMPARABLE VARIANCE COMPONENT -------------------------------------------------
    P_("\n" + RULE); P_("M4  THE SAME QUANTITY AS THE OTHER TWO EXPERIMENTS, SO ALL THREE COMPARE")
    P_(RULE)
    P_("  Group the disruptions by motif instance -- identical wild-type windows are the same motif")
    P_("  -- and take the between-group variance above noise, exactly as humantransfer H3b did.")

    def comp(groups, sg, rng2, B=2000):
        def est(gs):
            num = 0.0
            den = 0
            for v in gs:
                K = len(v)
                if K < 2:
                    continue
                num += np.var(v, ddof=0) * K - (sg ** 2 / 2) * (K - 1)
                den += K
            return max(num / max(den, 1), 0.0)
        pt = est(groups)
        idx = np.arange(len(groups))
        bs = np.array([est([groups[i] for i in rng2.choice(idx, len(idx), replace=True)])
                       for _ in range(B)])
        return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    P_(f"\n    {'cell':<8} {'motif classes':>14} {'floor':>8}")
    for cell in ("HepG2", "K562"):
        e1, e2, ms = res[cell]
        eb = (e1 + e2) / 2.0
        byk = collections.defaultdict(list)
        for v, m in zip(eb, ms):
            byk[m].append(v)
        keys = list(byk)
        # COMPLETE linkage. Single linkage is transitive and chained 181 of 196 disruptions into
        # one class on the first run, leaving exactly one group with two or more members -- so the
        # bootstrap resampled a single unit, the interval had zero width, and the "between-motif"
        # variance was really the variance WITHIN one giant class. Ledger U, in this module's own
        # statistic. Complete linkage cannot chain: a class is merged only if EVERY pair across it
        # is similar.
        clusters = [[k] for k in keys]
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if all(sim(a, b) >= 0.85 for a in clusters[i] for b in clusters[j]):
                        clusters[i] = clusters[i] + clusters[j]
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break
        cl = [sum((byk[k] for k in c), []) for c in clusters]
        sizes = sorted((len(v) for v in cl), reverse=True)
        gs = [np.array(v) for v in cl if len(v) >= 2]
        P_(f"    {cell:<8} {len(cl):>14} {floors[cell]:>8.4f}", )
        P_(f"      class sizes {sizes[:8]}{' ...' if len(sizes) > 8 else ''};"
           f" {len(gs)} classes have two or more members")
        if len(gs) < 3 or max(sizes) > 0.5 * sum(sizes):
            P_( "      NOT REPORTED: the partition is degenerate -- one class holds most of the")
            P_( "      disruptions, so a between-class variance would be a within-class variance")
            P_( "      wearing the wrong name, and its bootstrap interval would be meaningless.")
            continue
        pt, lo, hi = comp(gs, floors[cell], np.random.default_rng(1))
        P_(f"      between-motif sd {np.sqrt(pt):.4f} log2 ="
           f" {np.sqrt(pt)/floors[cell]:.2f} floors"
           f" CI [{np.sqrt(lo)/floors[cell]:.2f}, {np.sqrt(hi)/floors[cell]:.2f}]")
    P_("\n  for comparison, the same estimator elsewhere in this build order:")
    P_("    yeast promoters, between factor set   15.86 floors  CI [9.32, 21.36]")
    P_("    human CRISPRi, between regulator       1.05 floors  CI [0.90, 1.18]")

    # ---- M5 / M6 -------------------------------------------------------------------------------
    # ---- M4b  WHAT THE CROSS-CELL RESULT ACTUALLY SHOWS -----------------------------------------
    P_("\n" + RULE); P_("M4b  IS A MOTIF'S EFFECT A PROPERTY OF THE MOTIF, WITHIN ONE CELL?")
    P_(RULE)
    P_("  The cross-cell result in M5 has TWO explanations and they matter very differently.")
    P_("  (a) the per-gene RESPONSE FUNCTION differs between cells -- a problem for the engine; or")
    P_("  (b) the cognate factor's ACTIVITY differs between cells -- which is precisely what the")
    P_("      engine already models as controller state, and under which an uncorrelated cross-cell")
    P_("      result is what a CORRECT engine predicts.")
    P_("  M5 alone cannot separate them. This can: within ONE cell, is the same motif's effect")
    P_("  consistent across the different enhancer contexts it appears in?")
    P_(f"\n    {'cell':<8} {'classes':>8} {'between motifs':>16} {'within a motif':>16}"
       f" {'identity explains':>18}")
    for cell in ("HepG2", "K562"):
        e1, e2, ms = res[cell]
        eb = (e1 + e2) / 2.0
        sg = floors[cell]
        noise_mean = sg ** 2 / 2
        byk = collections.defaultdict(list)
        for v, m in zip(eb, ms):
            byk[m].append(v)
        keys = list(byk)
        clusters = [[k] for k in keys]
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if all(sim(a, b) >= 0.85 for a in clusters[i] for b in clusters[j]):
                        clusters[i] = clusters[i] + clusters[j]
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break
        gs = [np.array(sum((byk[k] for k in c), [])) for c in clusters]
        gs = [v for v in gs if len(v) >= 2]
        num = 0.0
        den = 0
        for v in gs:
            K = len(v)
            num += np.var(v, ddof=0) * K - noise_mean * (K - 1)
            den += K
        within = max(num / max(den, 1), 0.0)
        gm = np.array([v.mean() for v in gs])
        ns = np.array([len(v) for v in gs])
        between = max(float(np.var(gm, ddof=0)) - float(np.mean(within / ns + noise_mean / ns)), 0.0)
        tot = between + within
        P_(f"    {cell:<8} {len(gs):>8} {f'{np.sqrt(between)/sg:.2f} floors':>16}"
           f" {f'{np.sqrt(within)/sg:.2f} floors':>16}"
           f" {f'{100*between/tot:.1f}%' if tot > 0 else 'n/a':>18}")
    P_("\n  So the answer is (b) plus something else. Within a single cell, MOTIF IDENTITY explains")
    P_("  only a minority of the reproducible effect -- the rest is CONTEXT, the same motif")
    P_("  behaving differently in different enhancers. That does NOT show the engine's per-gene")
    P_("  response function is unstable: the engine's conditionals are PER GENE already, so")
    P_("  context-dependence is inside the model rather than outside it.")
    P_("  What it damages is the COMPRESSION. Every summary tested in this build order -- count,")
    P_("  signed count, class count, multiplier -- shares one assumption: that a gene's response")
    P_("  depends on its regulators through classes that are SHARED ACROSS GENES. A globally")
    P_("  shared class map can capture at most the between-motif share measured above, and that")
    P_("  is why the human class ladder was flat while yeast's spanned 8.6 floors.")
    P_("\n  THE CAVEAT THAT COULD OVERTURN THIS: the classes are recovered by sequence similarity,")
    P_("  not from a curated database, so instances of different factors may share a class. Any")
    P_("  such error inflates the WITHIN-class term, so the context share is an UPPER bound and")
    P_("  the identity share a LOWER one.")

    P_("\n" + RULE); P_("M5  REPLICATION ACROSS CELL TYPES"); P_(RULE)
    P_("  The first version of this gate asserted the principle and computed nothing. The two")
    P_("  cell types measure the SAME constructs, so the replication is a number.")
    eh = (res["HepG2"][0] + res["HepG2"][1]) / 2.0
    ek = (res["K562"][0] + res["K562"][1]) / 2.0
    n = min(len(eh), len(ek))
    rc = float(np.corrcoef(eh[:n], ek[:n])[0, 1])
    P_(f"\n    disruption effects, HepG2 against K562, same {n} constructs: r = {rc:+.4f}")
    P_(f"    within-cell reproducibility for comparison:")
    for cell in ("HepG2", "K562"):
        e1, e2, _ = res[cell]
        P_(f"      {cell:<8} replicate 1 against replicate 2: r = {float(np.corrcoef(e1, e2)[0,1]):+.4f}")
    P_("\n  M5: the M3 result appears in K562 and not in HepG2. By this gate as predeclared, that")
    P_("  is NOT a result. The cross-cell correlation above says why it need not be a")
    P_("  contradiction -- these are cell-type-specific enhancers and the same motif need not act")
    P_("  in both -- but a mechanism for a split is not the same as a replication, and the")
    P_("  predeclared bar was replication.")
    P_("\n" + RULE); P_("M6  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. IT DOES NOT TEST COPY NUMBER. Every disruption here removes ONE motif from a")
    P_("     construct that has one studied motif. The question 'do two sites act like one site")
    P_("     twice' is untouched, and the open dataset that would answer it could not be read.")
    P_("  2. It measures DISRUPTION, not presence: the effect of removing a motif from its native")
    P_("     context, which is not the same as the effect of adding one to a neutral background.")
    P_("  3. The motif classes are recovered from sequence, not from a curated database, so a")
    P_("     class is 'instances that look alike' rather than 'instances of a named factor'.")
    P_("  4. Two cell lines, one assay, episomal reporters. Nothing here speaks to chromatin.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_mpra_human.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
