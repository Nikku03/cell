"""THE CHAIN -- a point mutation in DNA, through the chromosome fold, to a residue, into nexus.

WHAT WAS ASKED FOR.  Link a point mutation in DNA with the chromosome fold, carry it to the amino-acid
sequence, and employ nexus to see its effects. Four links, each of which this repository has some piece
of, and which have never been connected end to end on real variants:

    1  DNA        a ClinVar single-nucleotide variant: chromosome, position, reference base, alt base
    2  FOLD       measured bTMP torsion at that exact coordinate (loop 8's tracks, 20 kb bins)
    3  RESIDUE    genomic position -> CDS offset -> codon -> translate WT and MUT amino acid
    4  NEXUS      AlphaFold structure -> folding ddG -> nexus.directed_activity

Link 3 is the one that has to be BUILT rather than looked up. It is computed here from Ensembl's CDS
exon structure and CDS sequence: walk the transcript's coding exons in transcription order, convert the
genomic coordinate to an offset in the coding sequence, take the codon, substitute the base (complemented
on a minus-strand gene), and translate both. Nothing about the protein consequence is read from an
annotation. That is the point -- a chain whose middle link is a lookup is a join, not a model.

THE FAILURE MODE THIS IS BUILT AGAINST.  A chain like this can run end to end, produce a number for
every variant, and be worthless in a specific way: if any link is CONSTANT WITHIN A GENE, then every
mutation in that gene gets the same answer, and what looks like "any point mutation" is really "which
gene". loop 5 declared exactly this limit about itself and loop 23 measured it. So every link here is
scored for whether it varies BETWEEN TWO MUTATIONS OF THE SAME GENE, and every separation is tested
against WITHIN-GENE shuffles rather than global ones. A global shuffle would let the chain rediscover
which genes are pathogenic, which loop 5 already did and which is not the question.

PREDECLARED, before any number:

    C1 THE CHAIN RUNS         at least 500 variants traverse all four links and come out the far end
                with a gene, a residue position, a WT and MUT amino acid, a torsion value and a nexus
                activity.
    C2 THE TRANSLATION IS RIGHT   the consequence computed from the codon agrees with ClinVar's OWN
                molecular-consequence field on at least 99% of variants.
                THIS IS THE GATE THAT SEPARATES A REAL LINK FROM A PLAUSIBLE ONE. The MC field is never
                used to build the codon -- it is held back and used only to mark it. A frame error, a
                strand error, or an off-by-one destroys this immediately and cannot be argued with.
    C3 THE CHAIN KEEPS VARIANT RESOLUTION   measured per link as the fraction of multi-variant genes in
                which two variants of that gene receive DIFFERENT values. The gate is on the FAR END:
                nexus activity must vary within at least 50% of multi-variant genes.
                A link at 0% does not answer "any point mutation" at all. It answers "which gene", and
                should be reported that way.
    C4 THE FOLD LINK EARNS ITS PLACE   torsion at the variant's own coordinate separates pathogenic from
                benign WITHIN gene, against 200 within-gene label shuffles.
                PREDECLARED EXPECTATION: FAIL. The track is binned at 20 kb, so for any gene shorter
                than a bin every variant gets an identical value and the link is constant by
                construction. It is written down here, before the run, so that a null reads as the
                measured limit it is rather than a disappointment -- and so that a PASS is required to
                survive the naked-DNA control below before it is believed.
    C5 NEXUS SEES SOMETHING   where its sensors fire, nexus activity separates pathogenic from benign
                within gene, above 200 within-gene shuffles.
    C6 IT BEATS THE SUBSTITUTION MATRIX   nexus activity beats BLOSUM62 scored on the same variants.
                A 20x20 table from 1992 needs no structure, no fold and no cell model. If it does as
                well, the three links in front of it bought nothing and the honest report is that.

CONTROLS: ClinVar's molecular-consequence field held back as an independent check on the translation;
the reference base checked against the CDS at every variant; 200 WITHIN-GENE label shuffles; the
naked-DNA bTMP control track alongside the torsion track, as in loop 8; BLOSUM62 as the sequence-only
comparator; and per-link coverage reported at every stage so attrition is visible rather than hidden in
a final N.

-> outputs/loop_chain.json
"""
import collections
import gzip
import json
import os
import pickle
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
CLINVAR = SC / "clinvar.vcf.gz"
GTF = SC / "ens_gtf.gz"
CDSFA = SC / "ens_cds.fa.gz"
AFDIR = SC / "af_cache"
TORSION = ("GSM8523629_supercoil_rep1_hg38.bw", "GSM8523630_supercoil_rep2_hg38.bw")
NAKED = "GSM8523628_RPE1_bTMP_genomic_control_hg38.bw"
SEED = 1904
N_SHUFFLE = 200
GATE_RUNS = 500
GATE_TRANSLATION = 0.99
GATE_RESOLUTION = 0.50
MAX_GENES = 400          # genes taken to the structural stage, most-variants-first
COMP = str.maketrans("ACGT", "TGCA")

PATHO = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}


def cached(name, fn):
    p = SC / f"_chain_{name}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    v = fn()
    pickle.dump(v, open(p, "wb"))
    return v


def transcript_model():
    """One coding transcript per gene: MANE Select, else Ensembl canonical, else longest CDS.

    Picking by hand would be a judgement buried in a regex; this is the standard order and it is
    written down. The chosen transcript's CDS exons carry strand and genomic span, which is everything
    the coordinate walk below needs."""
    cds = collections.defaultdict(list)
    meta = {}
    for line in gzip.open(GTF, "rt"):
        if line[0] == "#":
            continue
        f = line.rstrip("\n").split("\t")
        if f[2] != "CDS":
            continue
        a = f[8]
        gn = re.search(r'gene_name "([^"]+)"', a)
        tx = re.search(r'transcript_id "([^"]+)"', a)
        if not gn or not tx:
            continue
        gn, tx = gn.group(1), tx.group(1)
        cds[tx].append((int(f[3]), int(f[4])))
        meta[tx] = (gn, f[0], f[6],
                    0 if 'tag "MANE_Select"' in a else (1 if 'tag "Ensembl_canonical"' in a else 2))
    best = {}
    for tx, (gn, ch, st, rank) in meta.items():
        key = (rank, -sum(b - a + 1 for a, b in cds[tx]))
        if gn not in best or key < best[gn][0]:
            best[gn] = (key, tx, ch, st)
    return {"best": best, "cds": {t: cds[t] for _, t, _, _ in best.values()}}


def cds_sequences(want):
    seq, cur, buf = {}, None, []
    for line in gzip.open(CDSFA, "rt"):
        if line[0] == ">":
            if cur and cur in want:
                seq[cur] = "".join(buf)
            cur, buf = line[1:].split()[0].split(".")[0], []
        else:
            buf.append(line.strip())
    if cur and cur in want:
        seq[cur] = "".join(buf)
    return seq


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    from Bio.Seq import Seq
    say("=" * 100)
    say("THE CHAIN -- point mutation in DNA -> chromosome fold -> residue -> nexus")
    say("=" * 100)
    say("  Four links connected end to end on real variants. The middle one is COMPUTED from CDS exon")
    say("  structure, not read from an annotation, and ClinVar's own consequence field is held back to")
    say("  mark it.")

    # ---- link 1: the variants --------------------------------------------------------------------
    t0 = time.time()

    def _variants():
        rows = []
        for line in gzip.open(CLINVAR, "rt"):
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            if len(f) < 8 or len(f[3]) != 1 or len(f[4]) != 1:
                continue
            if f[3] not in "ACGT" or f[4] not in "ACGT":
                continue
            info = f[7]
            m = re.search(r"CLNSIG=([^;]+)", info)
            g = re.search(r"GENEINFO=([^:;|]+)", info)
            mc = re.search(r"MC=([^;]+)", info)
            if not (m and g and mc):
                continue
            sig = m.group(1)
            y = 1 if sig in PATHO else (0 if sig in BENIGN else None)
            if y is None:
                continue
            cons = {x.split("|")[-1] for x in mc.group(1).split(",")}
            rows.append((f[0], int(f[1]), f[3], f[4], g.group(1), y, ",".join(sorted(cons))))
        return rows

    V = cached("variants", _variants)
    say(f"\n  LINK 1  DNA -- {len(V):,} ClinVar single-nucleotide variants with a definite "
        f"pathogenic/benign call ({time.time()-t0:.0f}s)")

    M = cached("txmodel", transcript_model)
    seq = cached("cdsseq", lambda: cds_sequences({t.split(".")[0] for _, t, _, _ in M["best"].values()}))
    say(f"          transcript model: {len(M['best']):,} genes, one coding transcript each "
        f"(MANE Select > Ensembl canonical > longest CDS)")

    # ---- link 3: genomic coordinate -> codon -> residue -------------------------------------------
    def residue(gene, chrom, pos, ref, alt):
        e = M["best"].get(gene)
        if e is None:
            return None
        _, tx, ch, st = e
        if str(ch) != str(chrom):
            return None
        off = 0
        for a, b in sorted(M["cds"][tx], key=lambda x: x[0], reverse=(st == "-")):
            if a <= pos <= b:
                off += (pos - a) if st == "+" else (b - pos)
                break
            off += b - a + 1
        else:
            return None
        S = seq.get(tx.split(".")[0])
        if not S:
            return None
        ci, cp = off // 3, off % 3
        cod = S[3 * ci:3 * ci + 3]
        if len(cod) < 3:
            return None
        r = ref if st == "+" else ref.translate(COMP)
        a2 = alt if st == "+" else alt.translate(COMP)
        wt = str(Seq(cod).translate())
        mu = str(Seq(cod[:cp] + a2 + cod[cp + 1:]).translate())
        return ci + 1, wt, mu, bool(cod[cp] == r)

    def _translate():
        out = []
        for chrom, pos, ref, alt, gene, y, cons in V:
            r = residue(gene, chrom, pos, ref, alt)
            if r is None:
                continue
            aap, wt, mu, refok = r
            out.append((chrom, pos, ref, alt, gene, y, cons, aap, wt, mu, refok))
        return out

    T = cached("translated", _translate)
    say(f"\n  LINK 3  RESIDUE -- {len(T):,} of {len(V):,} variants fall inside the chosen transcript's "
        f"coding sequence")
    refok = np.mean([r[10] for r in T])
    say(f"          reference base matches the CDS at {refok:.2%} of them "
        f"(built-in check: a strand or frame error would collapse this)")
    T = [r for r in T if r[10]]

    # ---- C2 the independent check on the translation ---------------------------------------------
    def implied(wt, mu):
        """ClinVar's OWN Sequence Ontology spellings, which are not the obvious ones.

        A stop-gain is `nonsense` in the MC field, not `nonsense_variant`. The first run of this gate
        emitted `nonsense_variant` and tested it as a substring, so 55k correct translations scored as
        disagreements and C2 read 92.59% instead of passing. The reference base matched the CDS at
        100.00% of those same variants, which is what said the codon walk was right and the comparator
        was wrong. Membership is exact now rather than substring, so a vocabulary drift fails loudly
        instead of quietly counting matches."""
        if mu == "*":
            return "nonsense"
        if wt == "*":
            return "stop_lost"
        return "synonymous_variant" if wt == mu else "missense_variant"

    agree = [implied(r[8], r[9]) in set(r[6].split(",")) for r in T]
    c2_rate = float(np.mean(agree))
    c2 = bool(c2_rate >= GATE_TRANSLATION)
    say(f"\n  C2 THE TRANSLATION IS RIGHT -- agrees with ClinVar's own consequence field on "
        f"{c2_rate:.2%} of {len(T):,} variants")
    dis = collections.Counter(f"{implied(r[8], r[9])} vs {r[6]}" for r, a in zip(T, agree) if not a)
    for k, n in dis.most_common(3):
        say(f"          disagreement: {k}  x{n}")
    say(f"          C2 {'PASS' if c2 else 'FAIL'}  (gate {GATE_TRANSLATION:.0%})")

    D = pd.DataFrame([{"chrom": r[0], "pos": r[1], "gene": r[4], "y": r[5], "cons": r[6],
                       "aa_pos": r[7], "wt": r[8], "mut": r[9]} for r in T])
    D = D[D.cons.str.contains("missense_variant") & (D.wt != D.mut) & (D.mut != "*")].reset_index(drop=True)
    say(f"\n  restricting to missense: {len(D):,} variants, {int(D.y.sum()):,} pathogenic, "
        f"{len(D)-int(D.y.sum()):,} benign, {D.gene.nunique():,} genes")

    # ---- link 2: the fold at the variant's own coordinate ----------------------------------------
    import pyBigWig
    bws = {n: pyBigWig.open(str(SC / f)) for n, f in
           [("rep1", TORSION[0]), ("rep2", TORSION[1]), ("naked", NAKED)]}
    chroms = set(bws["rep1"].chroms())

    def _fold():
        vals = []
        for chrom, pos in zip(D.chrom, D.pos):
            cc = f"chr{chrom}"
            if cc not in chroms:
                vals.append((np.nan, np.nan))
                continue
            v = []
            for k in ("rep1", "rep2"):
                try:
                    s = bws[k].stats(cc, int(pos) - 1, int(pos), type="mean")[0]
                except Exception:
                    s = None
                v.append(np.nan if s is None else float(s))
            try:
                nk = bws["naked"].stats(cc, int(pos) - 1, int(pos), type="mean")[0]
            except Exception:
                nk = None
            vals.append((float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan,
                         np.nan if nk is None else float(nk)))
        return vals

    F = cached("fold", _fold)
    D["torsion"] = [x[0] for x in F]
    D["naked"] = [x[1] for x in F]
    say(f"\n  LINK 2  FOLD -- measured bTMP torsion at the variant's exact coordinate for "
        f"{np.isfinite(D.torsion).sum():,} of {len(D):,} variants (20 kb bins)")

    # ---- link 4: nexus ---------------------------------------------------------------------------
    import ddg_predictor as dp
    import nexus
    AFDIR.mkdir(parents=True, exist_ok=True)
    counts = D.gene.value_counts()
    genes = list(counts.index[:MAX_GENES])
    say(f"\n  LINK 4  NEXUS -- structural stage on the {len(genes)} genes with the most variants "
        f"({counts[genes].sum():,} of {len(D):,} variants)")

    def _acc(gene):
        u = ("https://rest.uniprot.org/uniprotkb/search?query=(gene:" + gene +
             ")+AND+organism_id:9606+AND+reviewed:true&fields=accession&format=tsv&size=1")
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                t = r.read().decode().strip().split("\n")
            return t[1].strip() if len(t) > 1 else None
        except Exception:
            return None

    def _accs():
        out = {}
        for i, g in enumerate(genes):
            out[g] = _acc(g)
            if i % 50 == 0:
                say(f"          uniprot {i}/{len(genes)}")
        return out

    ACC = cached("accs", _accs)
    have = sum(1 for v in ACC.values() if v)
    say(f"          UniProt accession found for {have}/{len(genes)} genes")

    def _structs():
        ok = {}
        for i, g in enumerate(genes):
            a = ACC.get(g)
            if not a:
                continue
            p = AFDIR / f"AF-{a}.pdb"
            if not p.exists():
                try:
                    urllib.request.urlretrieve(
                        f"https://alphafold.ebi.ac.uk/files/AF-{a}-F1-model_v6.pdb", p)
                except Exception:
                    continue
            if p.exists() and p.stat().st_size > 1000:
                ok[g] = str(p)
            if i % 50 == 0:
                say(f"          structures {i}/{len(genes)}")
        return ok

    STR = cached("structs", _structs)
    say(f"          AlphaFold structure for {len(STR)}/{len(genes)} genes")

    def _ddg():
        d = dp.DDGPredictor(str(ROOT / "outputs" / "orphan" / "ddg_model.pkl"))
        out = {}
        cur = None
        for i, r in D[D.gene.isin(STR)].iterrows():
            p = STR.get(r.gene)
            if not p:
                continue
            try:
                v, ok = d.predict_from_structure(p, "A", int(r.aa_pos), r.wt, r.mut)
                out[i] = (float(v), bool(ok))
            except Exception:
                out[i] = (np.nan, False)
            cur = i
            if len(out) % 2000 == 0:
                say(f"          ddG {len(out)} scored")
        return out

    G = cached("ddg", _ddg)
    D["ddg"] = [G.get(i, (np.nan, False))[0] for i in D.index]
    D["ddg_ok"] = [G.get(i, (np.nan, False))[1] for i in D.index]
    # the WT residue from Ensembl CDS must be the residue in the UniProt-numbered AlphaFold model.
    # Two entirely separate sources; where they disagree the sensor abstains rather than guessing.
    D.loc[~D.ddg_ok, "ddg"] = np.nan
    n_ddg = int(np.isfinite(D.ddg).sum())
    say(f"          folding ddG computed for {n_ddg:,} variants "
        f"({n_ddg/max(len(D),1):.1%} of the missense set)")
    D["activity"] = [nexus.nexus_activity(v, 0.0) if np.isfinite(v) else np.nan for v in D.ddg]

    # ---- C1 -------------------------------------------------------------------------------------
    full = D[np.isfinite(D.torsion) & np.isfinite(D.activity)]
    c1 = bool(len(full) >= GATE_RUNS)
    say(f"\n  C1 THE CHAIN RUNS -- {len(full):,} variants have all four links: DNA coordinate, torsion, "
        f"residue, nexus activity   C1 {'PASS' if c1 else 'FAIL'}  (gate {GATE_RUNS})")

    # ---- C3 variant resolution, per link ---------------------------------------------------------
    say(f"\n  C3 THE CHAIN KEEPS VARIANT RESOLUTION -- fraction of multi-variant genes in which two")
    say(f"     variants of that gene get DIFFERENT values. A link at 0% answers 'which gene'.")
    res = {}
    for col, label in (("pos", "1 DNA coordinate"), ("torsion", "2 chromosome fold"),
                       ("aa_pos", "3 residue position"), ("ddg", "4 nexus folding ddG"),
                       ("activity", "4 nexus activity")):
        sub = D[np.isfinite(D[col])] if col != "pos" else D
        gg = sub.groupby("gene")[col]
        multi = gg.count() >= 2
        if multi.sum() == 0:
            res[label] = float("nan")
            continue
        varies = gg.nunique()[multi] >= 2
        res[label] = float(varies.mean())
        say(f"     {label:<24}{res[label]:>8.1%}   ({int(multi.sum()):,} multi-variant genes)")
    c3 = bool(np.isfinite(res.get("4 nexus activity", np.nan))
              and res["4 nexus activity"] >= GATE_RESOLUTION)
    say(f"     C3 {'PASS' if c3 else 'FAIL'}  (gate: the far end of the chain varies within "
        f"{GATE_RESOLUTION:.0%} of multi-variant genes)")

    # ---- within-gene scoring ---------------------------------------------------------------------
    def within_gene(col, frame=None):
        """AUC over variants of genes that contain BOTH a pathogenic and a benign variant, with the
        null from shuffling labels INSIDE each gene. Between-gene signal is held fixed by construction,
        so what is measured is only the part that could ever answer 'any point mutation'."""
        S = (frame if frame is not None else D)
        S = S[np.isfinite(S[col])]
        keep = S.groupby("gene").y.agg(["min", "max", "count"])
        keep = keep[(keep["min"] == 0) & (keep["max"] == 1)].index
        S = S[S.gene.isin(keep)]
        if len(S) < 100:
            return float("nan"), float("nan"), 0, 0
        v, y, g = S[col].to_numpy(float), S.y.to_numpy(), S.gene.to_numpy()
        a = auc(v, y)
        null = []
        idx = {gg: np.where(g == gg)[0] for gg in set(g)}
        for _ in range(N_SHUFFLE):
            ys = y.copy()
            for gg, ii in idx.items():
                ys[ii] = rng.permutation(y[ii])
            null.append(auc(v, ys))
        null = np.array(null)
        p95 = float(np.percentile(np.abs(null - 0.5), 95)) + 0.5
        return a, p95, len(S), len(keep)

    say(f"\n  scoring WITHIN gene from here: only genes holding both a pathogenic and a benign variant,")
    say(f"  and the null shuffles labels inside each gene so between-gene signal cannot leak in.")

    # ---- C4 the fold link ------------------------------------------------------------------------
    a_t, p_t, n_t, g_t = within_gene("torsion")
    a_n, p_n, _, _ = within_gene("naked")
    c4 = bool(np.isfinite(a_t) and abs(a_t - 0.5) > p_t - 0.5)
    say(f"\n  C4 THE FOLD LINK EARNS ITS PLACE -- {n_t:,} variants in {g_t:,} genes")
    say(f"     torsion at the variant coordinate   {a_t:>8.4f}   within-gene null 95th {p_t:.4f}")
    say(f"     naked-DNA control track             {a_n:>8.4f}   within-gene null 95th {p_n:.4f}")
    say(f"     C4 {'PASS' if c4 else 'FAIL'}   (predeclared expectation was FAIL: 20 kb bins cannot "
        f"separate two mutations in one gene)")

    # ---- C5 / C6 nexus ---------------------------------------------------------------------------
    try:
        from Bio.Align import substitution_matrices
        B = substitution_matrices.load("BLOSUM62")

        def blos(w, m):
            try:
                return -float(B[w, m])     # negated: a LOW blosum score means a disruptive swap
            except Exception:
                return np.nan
        D["blosum"] = [blos(w, m) for w, m in zip(D.wt, D.mut)]
    except Exception:
        D["blosum"] = np.nan

    scored = D[np.isfinite(D.activity)]
    a_a, p_a, n_a, g_a = within_gene("activity", scored)
    a_d, p_d, _, _ = within_gene("ddg", scored)
    a_b, p_b, _, _ = within_gene("blosum", scored)
    c5 = bool(np.isfinite(a_a) and abs(a_a - 0.5) > p_a - 0.5)
    say(f"\n  C5 NEXUS SEES SOMETHING -- {n_a:,} variants in {g_a:,} genes, on the same variants "
        f"throughout")
    say(f"     nexus activity                      {a_a:>8.4f}   within-gene null 95th {p_a:.4f}")
    say(f"     folding ddG alone                   {a_d:>8.4f}   within-gene null 95th {p_d:.4f}")
    say(f"     BLOSUM62 (sequence lookup)          {a_b:>8.4f}   within-gene null 95th {p_b:.4f}")
    say(f"     C5 {'PASS' if c5 else 'FAIL'}")
    c6 = bool(np.isfinite(a_a) and np.isfinite(a_b) and abs(a_a - 0.5) > abs(a_b - 0.5))
    say(f"  C6 IT BEATS THE SUBSTITUTION MATRIX -- {abs(a_a-0.5):.4f} vs BLOSUM62's "
        f"{abs(a_b-0.5):.4f}   C6 {'PASS' if c6 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"C1 the chain runs": c1, "C2 the translation is right": c2,
             "C3 keeps variant resolution": c3, "C4 the fold link earns its place": c4,
             "C5 nexus sees something": c5, "C6 beats the substitution matrix": c6}
    for k, v in gates.items():
        say(f"  {k:<38}{'PASS' if v else 'FAIL'}")
    if c1 and c2:
        say(f"  THE CHAIN IS BUILT AND THE MIDDLE LINK IS VERIFIED. {len(full):,} point mutations go")
        say(f"  from a genomic coordinate, through measured torsion, to a computed residue change, into")
        say(f"  nexus -- and the residue step agrees with ClinVar's own consequence call on "
            f"{c2_rate:.2%},")
        say("  having never been shown it.")
    else:
        say("  THE CHAIN DOES NOT STAND UP. Nothing downstream of a broken translation is worth reading.")
    if not c4:
        say(f"  THE FOLD LINK IS GENE-LEVEL, AS PREDECLARED. Torsion varies within only "
            f"{res.get('2 chromosome fold', float('nan')):.1%} of")
        say("  multi-variant genes and does not separate pathogenic from benign inside a gene. It is a")
        say("  real measurement attached at the right coordinate, and it cannot tell two mutations in")
        say("  one gene apart, which is what the question asked for.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CLINVAR), str(GTF), str(CDSFA), str(SC / TORSION[0])],
                      available=len(D), used=len(full), selection="filtered", seed=SEED,
                      controls=["ClinVar molecular consequence held back as an independent check",
                                "reference base checked against the CDS at every variant",
                                f"{N_SHUFFLE} WITHIN-GENE label shuffles",
                                "naked-DNA bTMP control track",
                                "BLOSUM62 as the sequence-only comparator",
                                "per-link coverage reported so attrition is visible"],
                      note="the residue link is computed from CDS exon structure, not read from an "
                           "annotation; every separation is scored within gene because a link that is "
                           "constant inside a gene answers 'which gene', not 'which mutation'")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_chain", "manifest": man, "gates": gates,
               "n_variants": len(V), "n_translated": len(T), "n_missense": len(D),
               "n_full_chain": len(full), "ref_base_ok": float(refok),
               "translation_agreement": c2_rate, "resolution": res,
               "within_gene": {"torsion": a_t, "torsion_null95": p_t, "naked": a_n,
                               "activity": a_a, "activity_null95": p_a, "ddg": a_d,
                               "blosum": a_b, "n": n_a, "n_genes": g_a},
               "n_ddg": n_ddg, "n_genes_struct": len(STR), "log": log},
              open(OUT / "loop_chain.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_chain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
