"""splice_nexus — wire the real pretrained SpliceAI (spliceai_torch) into NEXUS as a SPLICING mutation mode.

NEXUS so far converts a mutation into a graded protein ACTIVITY via structure (ΔΔG_fold x ΔΔG_bind, soft-AND). But a variant can
be protein-silent yet destroy or create a SPLICE SITE -- changing the isoform, truncating the protein, or triggering nonsense-
mediated decay (NMD). That is a different worker (the spliceosome reading the pre-mRNA 5'/3' splice signals), so it needs its own
sensor. SpliceAI is that sensor: a deep net trained on the human transcriptome that reads +-5kb of pre-mRNA context and outputs,
per base, P(acceptor)/P(donor). The variant-effect readout is the SpliceAI DELTA score: run ref vs alt, take the max change in
splice-site probability within +-50 nt (acceptor-gain / acceptor-loss / donor-gain / donor-loss); delta in [0,1].

Chain into NEXUS:  splice delta -> predicted consequence -> activity multiplier
  delta < 0.2  : likely no splice effect            -> activity ~ 1        (protein made normally)
  delta 0.2-0.5: possible / weak splice alteration  -> activity graded
  delta > 0.5  : likely splice-disrupting (exon skip / cryptic site / intron retention) -> truncation/NMD -> activity -> ~0 (LOF)
So splice_activity = 1 - delta (clamped): a strong splice-disrupting variant drives the protein toward loss-of-function, exactly
the same [0,1] activity currency the rest of NEXUS/propagate consumes -- a splice-disrupting TF variant then flows through the
regulation -> PPI -> pathway stack like any other LOF.

HONEST SCOPE: SpliceAI is the field-standard and strong on CANONICAL splice-disrupting variants (published ~0.9 top-k accuracy /
high auPRC on ClinVar splice variants); it is weaker on deep-intronic, tissue-specific, and weak-effect splicing, and it predicts
splice-site USAGE change, not the exact resulting isoform ratio in a given cell (that needs RNA-seq junctions). Direction (does it
disrupt a site) is reliable; the precise isoform outcome and its quantitative penetrance are not. Sequence context is fetched
on-demand from Ensembl REST (GRCh38); no local genome needed.
"""
import numpy as np
import spliceai_torch as sa

FLANK = sa.CONTEXT // 2          # 5000 each side
ENSEMBL = "https://rest.ensembl.org"
_RC = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def _rc(s):
    return "".join(_RC.get(c, "N") for c in s.upper()[::-1])


def delta_from_sequences(ref_seq, alt_seq, var_target_pos, dist=50):
    """core delta-score: ref_seq/alt_seq each = target + FLANK on both sides, differing at one position. var_target_pos is the
    variant's index within the TARGET (0-based). Returns the 4 SpliceAI delta components + the overall delta score."""
    pr = sa.predict_probs(ref_seq)          # (L,3)
    pa = sa.predict_probs(alt_seq)
    L = pr.shape[0]
    lo, hi = max(0, var_target_pos - dist), min(L, var_target_pos + dist + 1)
    da = pa[lo:hi, 1] - pr[lo:hi, 1]        # acceptor change
    dd = pa[lo:hi, 2] - pr[lo:hi, 2]        # donor change
    ag, al = float(max(da.max(), 0)), float(max((-da).max(), 0))
    dg, dl = float(max(dd.max(), 0)), float(max((-dd).max(), 0))
    delta = max(ag, al, dg, dl)
    return {"acceptor_gain": round(ag, 3), "acceptor_loss": round(al, 3),
            "donor_gain": round(dg, 3), "donor_loss": round(dl, 3), "delta_score": round(delta, 3)}


def splice_activity(delta_score):
    """map SpliceAI delta -> NEXUS activity multiplier (1 = protein made normally, ->0 = splice-disrupted LOF)."""
    a = max(0.0, 1.0 - delta_score)
    call = ("splice-disrupting (LOF)" if delta_score >= 0.5 else
            "possible splice alteration" if delta_score >= 0.2 else "no substantial splice effect")
    return {"activity": round(a, 3), "call": call, "delta_score": round(delta_score, 3)}


def _fetch(chrom, lo, hi):
    import requests
    r = requests.get(f"{ENSEMBL}/sequence/region/human/{chrom}:{lo}..{hi}:1",
                     headers={"Content-Type": "text/x-fasta"}, timeout=90)
    r.raise_for_status()
    return "".join(r.text.split("\n")[1:]).upper()


def _gene_locus(gene):
    """resolve a gene symbol -> (chrom, strand) via Ensembl (GRCh38)."""
    import requests
    r = requests.get(f"{ENSEMBL}/lookup/symbol/homo_sapiens/{gene}",
                     headers={"Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    return d["seq_region_name"], d["strand"]


def variant_delta(chrom, pos, ref, alt, strand=1, dist=50):
    """SpliceAI delta for a SNV at 1-based genomic `pos` on `chrom`. Fetches +-(FLANK+dist) context from Ensembl, builds ref/alt
    windows (strand-corrected), scores. ref/alt are the plus-strand alleles as given; strand flips to transcript orientation."""
    lo, hi = pos - FLANK - dist, pos + FLANK + dist
    g = _fetch(chrom, lo, hi)                                  # plus-strand genomic, length 2*(FLANK+dist)+1
    vi = pos - lo                                              # variant index in g (plus strand)
    if g[vi] != ref.upper():
        return {"error": f"ref mismatch at {chrom}:{pos}: genome has {g[vi]}, variant says {ref}"}
    g_alt = g[:vi] + alt.upper() + g[vi + 1:]
    if strand == -1:
        ref_seq, alt_seq = _rc(g), _rc(g_alt)
        var_target_pos = (len(g) - 1 - vi) - FLANK
    else:
        ref_seq, alt_seq = g, g_alt
        var_target_pos = vi - FLANK
    d = delta_from_sequences(ref_seq, alt_seq, var_target_pos, dist)
    d.update({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "strand": strand})
    return d


def nexus_splice(gene, pos, ref, alt, dist=50):
    """the cross-layer call: a genomic SNV in `gene` -> SpliceAI delta -> NEXUS splice activity (LOF if it disrupts a site).
    Resolves the gene's chrom/strand from Ensembl, scores the variant, returns the activity the rest of NEXUS/propagate consumes."""
    try:
        chrom, strand = _gene_locus(gene)
    except Exception as e:
        return {"error": f"could not resolve {gene}: {e}"}
    d = variant_delta(chrom, pos, ref, alt, strand, dist)
    if "error" in d:
        return d
    act = splice_activity(d["delta_score"])
    return {"gene": gene, **d, **act}


if __name__ == "__main__":
    # validation: destroy the HBB exon-2 DONOR (the invariant +1 G of the GT) and confirm a large donor-loss delta.
    import requests
    S = ENSEMBL
    t = requests.get(f"{S}/lookup/id/ENST00000335295?expand=1", headers={"Content-Type": "application/json"}, timeout=60).json()
    chrom = t["seq_region_name"]; strand = t["strand"]
    exons = sorted([(e["start"], e["end"]) for e in t["Exon"]])
    ex_s, ex_e = exons[1]                                       # internal exon 2
    # exon2 DONOR is at its 3' (transcript) end. On the minus strand that is genomic coord ex_s-1/ex_s side.
    # Build transcript-oriented ref window and mutate the first intronic base after the donor (the G of GT).
    lo = ex_s - FLANK - 50; hi = ex_e + FLANK + 50
    g = _fetch(chrom, lo, hi)
    seq = _rc(g) if strand == -1 else g
    exlen = ex_e - ex_s + 1
    donor_pos = FLANK + 50 + exlen           # first intronic base after exon2 (transcript coords, +5000 flank in seq)
    # confirm ref donor fires, then mutate donor +1 (G->C) and rescore
    pr = sa.predict_probs(seq)
    ref_donor = float(pr[:, 2].max()); ref_donor_idx = int(pr[:, 2].argmax())
    alt = seq[:donor_pos] + ("C" if seq[donor_pos] == "G" else "A") + seq[donor_pos + 1:]
    d = delta_from_sequences(seq, alt, ref_donor_idx, dist=50)
    print(f"HBB exon2 donor test: ref max donor prob {ref_donor:.2f} at target idx {ref_donor_idx}")
    print(f"  disrupt donor dinucleotide -> delta components {d}")
    print(f"  -> splice_activity: {splice_activity(d['delta_score'])}")
