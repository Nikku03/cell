"""Session 26: select biologically-motivated gene pairs for the
synthetic-lethality pilot screen.

The full Syn3A pair space is C(458, 2) = ~105k pairs. The pilot tests
~250 pairs across five biologically-motivated categories so the per-
category synthetic-lethality rate is interpretable instead of an
unweighted noise floor:

    A — paralog pairs (cosine similarity in ESM-2 space)
        textbook synthetic-lethal candidates: shared function,
        one can substitute for the other
    B — same-pathway sequential pairs
        NEGATIVE control: if the same pathway is already broken by
        knocking out either gene, the joint pair shouldn't add
        lethality. A high category-B rate would falsify the
        methodology
    C — random different-pathway pairs of v15-non-essential genes
        BASELINE rate control: most should not be synthetic lethal
    D — transporter / substrate-consumer pairs
        exploratory: redundancy under specific medium conditions
    E — manually curated cofactor / redox / nucleotide-pool pairs
        biological-knowledge-based hypothesis seeds

Output:
    ``outputs/synthlet/pilot_pairs.csv`` with schema
    ``locus_a, locus_b, category, biological_rationale``.

Usage:
    python scripts/synthlet_pilot_pairs.py
    python scripts/synthlet_pilot_pairs.py --n-per-category 10  # quick

Hard non-negotiables (from spec):
    - All thresholds and selections derive from observable repo data
      (ESM-2 parquet, SBML reactions, v15 predictions). No fabricated
      pair lists.
    - Output CSV is deterministic given the same seed.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))

from cell_sim.layer3_reactions.sbml_parser import (  # noqa: E402
    parse_sbml, sbml_gene_to_locus,
)

ESM_PARQUET = REPO_ROOT / "cell_sim/features/cache/esm2_650M.parquet"
SBML_PATH = REPO_ROOT / (
    "cell_sim/data/Minimal_Cell_ComplexFormation/input_data/"
    "Syn3A_updated.xml"
)
V15_PRED = REPO_ROOT / (
    "outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4"
    "_composed_all455_v15_round2_priors.csv"
)

# Threshold for ESM-2 cosine paralog detection. The default 0.85 is
# defensible: it sits well above the typical inter-protein cosine
# baseline (~0.5-0.6 for unrelated bacterial proteins in ESM-2 space)
# but below 0.95 (a near-self/duplicate region). Adjustable via CLI.
PARALOG_COS_DEFAULT = 0.85

# Transporter detection: SBML reactions with a single substrate in the
# extracellular compartment ('_e') and a single product in the cytosol
# ('_c') with the same metabolite root.
EXT_SUFFIX = "_e"
CYT_SUFFIX = "_c"


def _normalize_locus(g: str) -> str:
    """Map SBML gene id to JCVISYN3A locus form."""
    return sbml_gene_to_locus(g) or g


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_v15_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["essential"] = df["essential"].astype(int)
    return df


def load_sbml_genes_and_pathways():
    """Return:
      gene_to_rxns:    {sbml_gene_id -> [rxn_short_name, ...]}
      gene_to_locus:   {sbml_gene_id -> JCVISYN3A_NNNN}
      rxn_to_genes:    {rxn_short_name -> [sbml_gene_id, ...]}
      rxn_to_metabolites: {rxn_short_name -> {"reactants": [...],
                                                "products": [...]}}
    """
    sbml = parse_sbml(SBML_PATH)
    gene_to_rxns: dict[str, list[str]] = defaultdict(list)
    rxn_to_genes: dict[str, list[str]] = defaultdict(list)
    rxn_to_mets: dict[str, dict[str, list[str]]] = {}
    for rxn in sbml.reactions.values():
        sname = rxn.short_name
        rxn_to_mets[sname] = {
            "reactants": list(rxn.reactants.keys()),
            "products": list(rxn.products.keys()),
        }
        for g in rxn.gene_associations:
            gene_to_rxns[g].append(sname)
            rxn_to_genes[sname].append(g)
    gene_to_locus = {g: _normalize_locus(g) for g in gene_to_rxns}
    return gene_to_rxns, gene_to_locus, rxn_to_genes, rxn_to_mets


def load_esm2(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------
# Category A — paralogs by ESM-2 cosine similarity
# ---------------------------------------------------------------------


def select_paralog_pairs(
    esm: pd.DataFrame, n_pairs: int, threshold: float, seed: int,
) -> list[dict]:
    """Pairs of Syn3A loci whose ESM-2 mean-pooled embeddings have
    cosine similarity > threshold. Computed across all O(N^2) pairs
    and ranked; the top n_pairs hits with cos > threshold are taken.
    """
    rng = random.Random(seed)
    M = esm.values.astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    Mn = M / norms
    locus_tags = list(esm.index)
    n = len(locus_tags)
    cos_pairs: list[tuple[float, str, str]] = []
    block = 32
    for i in range(0, n, block):
        i_end = min(n, i + block)
        block_sims = Mn[i:i_end] @ Mn.T  # (block, n)
        for li in range(i, i_end):
            row = block_sims[li - i]
            # only j > li to avoid duplicates and self-cosines
            for j in range(li + 1, n):
                if row[j] >= threshold:
                    cos_pairs.append(
                        (float(row[j]), locus_tags[li], locus_tags[j])
                    )
    # Sort by descending similarity, take top n_pairs
    cos_pairs.sort(reverse=True)
    if len(cos_pairs) > n_pairs * 2:
        # Mix top-similar with mid-tier so we sample across the
        # threshold space; otherwise we'd only see near-identical
        # paralogs and miss the 0.85-0.92 band.
        top = cos_pairs[: n_pairs // 2]
        rest = cos_pairs[n_pairs // 2 : n_pairs * 4]
        rng.shuffle(rest)
        cos_pairs = top + rest[: n_pairs - len(top)]
    elif len(cos_pairs) > n_pairs:
        cos_pairs = cos_pairs[:n_pairs]
    rows = []
    for sim, a, b in cos_pairs[:n_pairs]:
        rows.append({
            "locus_a": a, "locus_b": b, "category": "A_paralog",
            "biological_rationale": (
                f"ESM-2 cosine similarity {sim:.3f}; candidate paralog "
                f"pair where one may substitute for the other"
            ),
        })
    return rows


# ---------------------------------------------------------------------
# Category B — same-pathway sequential pairs (NEGATIVE control)
# ---------------------------------------------------------------------


def select_same_pathway_pairs(
    rxn_to_genes: dict[str, list[str]],
    rxn_to_mets: dict[str, dict[str, list[str]]],
    gene_to_locus: dict[str, str],
    n_pairs: int, seed: int,
) -> list[dict]:
    """Pairs (gene_in_rxn1, gene_in_rxn2) where rxn1's product is
    rxn2's reactant — they catalyse sequential steps of the same
    pathway. By construction, knocking out either gene already
    breaks the pathway: a real synthetic-lethality methodology
    should NOT flag these as synthetic lethal at high rates.
    """
    rng = random.Random(seed)
    # Build product -> [rxn] index
    product_to_rxns: dict[str, list[str]] = defaultdict(list)
    for rxn, mets in rxn_to_mets.items():
        for p in mets["products"]:
            product_to_rxns[p].append(rxn)
    sequential_pairs: list[tuple[str, str, str]] = []
    for r2, mets2 in rxn_to_mets.items():
        for s in mets2["reactants"]:
            r1_list = product_to_rxns.get(s, [])
            for r1 in r1_list:
                if r1 == r2:
                    continue
                for ga in rxn_to_genes.get(r1, []):
                    for gb in rxn_to_genes.get(r2, []):
                        if ga == gb:
                            continue
                        la = gene_to_locus[ga]
                        lb = gene_to_locus[gb]
                        if la and lb and la < lb:
                            sequential_pairs.append((la, lb, s))
    # dedupe
    seen = set()
    deduped = []
    for la, lb, s in sequential_pairs:
        key = (la, lb)
        if key not in seen:
            seen.add(key)
            deduped.append((la, lb, s))
    rng.shuffle(deduped)
    rows = []
    for la, lb, shared_met in deduped[:n_pairs]:
        rows.append({
            "locus_a": la, "locus_b": lb,
            "category": "B_same_pathway",
            "biological_rationale": (
                f"sequential reactions sharing metabolite {shared_met}; "
                f"NEGATIVE control"
            ),
        })
    return rows


# ---------------------------------------------------------------------
# Category C — random different-pathway non-essential pairs
# ---------------------------------------------------------------------


def select_random_different_pathway_pairs(
    v15: pd.DataFrame,
    gene_to_rxns: dict[str, list[str]],
    rxn_to_mets: dict[str, dict[str, list[str]]],
    gene_to_locus: dict[str, str],
    n_pairs: int, seed: int,
) -> list[dict]:
    """Random pairs of genes both predicted non-essential at v15, that
    don't share a metabolite within their immediate reaction
    neighborhood. Imperfect heuristic — some pairs may still touch
    distant pathways — but the goal is a baseline rate, not a clean
    pathway-disjoint guarantee.
    """
    rng = random.Random(seed)
    locus_to_neighbor_mets: dict[str, set[str]] = defaultdict(set)
    for g, rxns in gene_to_rxns.items():
        locus = gene_to_locus.get(g)
        if not locus:
            continue
        for r in rxns:
            for m in rxn_to_mets[r]["reactants"]:
                locus_to_neighbor_mets[locus].add(m)
            for m in rxn_to_mets[r]["products"]:
                locus_to_neighbor_mets[locus].add(m)
    nonessential_loci = sorted(
        v15[v15["essential"] == 0]["locus_tag"].dropna().unique()
    )
    nonessential_loci = [
        g for g in nonessential_loci if g in locus_to_neighbor_mets
    ]
    rng.shuffle(nonessential_loci)
    rows = []
    tried = 0
    while len(rows) < n_pairs and tried < n_pairs * 50:
        tried += 1
        a, b = rng.sample(nonessential_loci, 2)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        shared = locus_to_neighbor_mets[a] & locus_to_neighbor_mets[b]
        if shared:
            continue
        rows.append({
            "locus_a": a, "locus_b": b, "category": "C_random",
            "biological_rationale": (
                "random pair of v15-nonessentials with no shared "
                "metabolic-neighborhood metabolite"
            ),
        })
    return rows


# ---------------------------------------------------------------------
# Category D — transporter / substrate consumer
# ---------------------------------------------------------------------


def select_transporter_substrate_pairs(
    rxn_to_genes: dict[str, list[str]],
    rxn_to_mets: dict[str, dict[str, list[str]]],
    gene_to_locus: dict[str, str],
    n_pairs: int, seed: int,
) -> list[dict]:
    """Pairs (transporter_gene, consumer_gene) where the transporter
    moves M_X_e -> M_X_c and the consumer's reaction has M_X_c as a
    reactant. Captures functional redundancy under medium-rich
    conditions: cell can survive losing the transporter if metabolite
    is produced internally, or the consumer if external substrate is
    abundant — but losing both could break the metabolite supply.
    """
    rng = random.Random(seed)
    transporter_pairs: list[tuple[str, str, str]] = []
    for trans_rxn, mets in rxn_to_mets.items():
        ext = [m for m in mets["reactants"] if m.endswith(EXT_SUFFIX)]
        cyt = [m for m in mets["products"] if m.endswith(CYT_SUFFIX)]
        if not ext or not cyt:
            continue
        # find single metabolite root that matches across e and c
        ext_roots = {m.removesuffix(EXT_SUFFIX) for m in ext}
        cyt_roots = {m.removesuffix(CYT_SUFFIX) for m in cyt}
        shared_roots = ext_roots & cyt_roots
        if not shared_roots:
            continue
        for root in shared_roots:
            cyt_met = root + CYT_SUFFIX
            for trans_gene in rxn_to_genes.get(trans_rxn, []):
                tlocus = gene_to_locus.get(trans_gene)
                if not tlocus:
                    continue
                # find consumer reactions of cyt_met
                for cons_rxn, cmets in rxn_to_mets.items():
                    if cons_rxn == trans_rxn:
                        continue
                    if cyt_met not in cmets["reactants"]:
                        continue
                    for cons_gene in rxn_to_genes.get(cons_rxn, []):
                        clocus = gene_to_locus.get(cons_gene)
                        if not clocus or clocus == tlocus:
                            continue
                        a, b = sorted((tlocus, clocus))
                        transporter_pairs.append((a, b, cyt_met))
    seen = set()
    deduped = []
    for a, b, met in transporter_pairs:
        if (a, b) not in seen:
            seen.add((a, b))
            deduped.append((a, b, met))
    rng.shuffle(deduped)
    rows = []
    for a, b, met in deduped[:n_pairs]:
        rows.append({
            "locus_a": a, "locus_b": b,
            "category": "D_transporter_substrate",
            "biological_rationale": (
                f"transporter-consumer pair across {met} cytosolic pool"
            ),
        })
    return rows


# ---------------------------------------------------------------------
# Category E — manual curation
# ---------------------------------------------------------------------

# Each entry:
#   (gene_pattern_a, gene_pattern_b, rationale)
# The pair selector will resolve the patterns against the SBML genes
# that actually exist in this Syn3A model. Patterns are gene-name
# substrings (case-insensitive).
MANUAL_PATTERNS = [
    ("dha", "gly", "DhaK / glycerol kinase — alternate glycerol-source routes"),
    ("nox", "ldh", "NADH oxidase / lactate dehydrogenase — redox redundancy"),
    ("trxA", "tpx", "thioredoxin / thiol-peroxidase — alternate redox couples"),
    ("ackA", "pta", "acetate kinase / phosphotransacetylase — sequential acetyl-P"),
    ("acsA", "ackA", "acs / ackA — acetyl-CoA salvage"),
    ("rnhA", "rnhB", "RNase H1 / RNase H2 paralogs"),
    ("recA", "recF", "RecA / RecF — recombination repair"),
    ("dnaB", "dnaG", "helicase / primase — replication priming"),
    ("metG", "metK", "Met-tRNA / SAM synthase — methionine handoff"),
    ("rpsB", "rpsA", "rpsB / rpsA ribosomal small-subunit"),
    ("rplA", "rplB", "rplA / rplB ribosomal large-subunit"),
    ("groL", "groS", "GroEL / GroES chaperonin pair"),
    ("dnaJ", "dnaK", "DnaJ / DnaK chaperone pair"),
    ("infA", "infB", "translation initiation factors"),
    ("fusA", "tufA", "elongation factors G / Tu"),
    ("trmD", "truA", "tRNA methylation / pseudouridine modification"),
    ("ftsZ", "ftsA", "divisome FtsZ / FtsA"),
    ("secA", "secY", "Sec translocon ATPase / channel"),
    ("ATPb", "ATPa", "F1Fo ATP synthase paired subunits"),
    ("pta", "ldh", "phosphotransacetylase / lactate dehydrogenase — fermentation alternates"),
]


def select_manual_pairs(
    gene_to_locus: dict[str, str],
    gene_name_lookup: dict[str, str],
    n_pairs: int, seed: int,
) -> list[dict]:
    """Resolve manual gene-name patterns against actual Syn3A genes.

    gene_name_lookup: {locus_tag -> gene_name} from GenBank.
    """
    name_to_locus: dict[str, str] = {}
    for locus, gn in gene_name_lookup.items():
        if isinstance(gn, str) and gn:
            name_to_locus.setdefault(gn.lower(), locus)
    rng = random.Random(seed)
    rows = []
    for pat_a, pat_b, rationale in MANUAL_PATTERNS:
        a_hits = [
            l for l, gn in gene_name_lookup.items()
            if isinstance(gn, str) and gn and pat_a.lower() in gn.lower()
        ]
        b_hits = [
            l for l, gn in gene_name_lookup.items()
            if isinstance(gn, str) and gn and pat_b.lower() in gn.lower()
        ]
        if not a_hits or not b_hits:
            continue
        for a in a_hits:
            for b in b_hits:
                if a >= b:
                    continue
                rows.append({
                    "locus_a": a, "locus_b": b, "category": "E_manual",
                    "biological_rationale": rationale,
                })
                break  # one pair per pattern
            else:
                continue
            break
    rng.shuffle(rows)
    return rows[:n_pairs]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/synthlet/pilot_pairs.csv",
    )
    ap.add_argument("--n-per-category", type=int, default=50)
    ap.add_argument(
        "--paralog-cos", type=float, default=PARALOG_COS_DEFAULT,
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[load] SBML + v15 predictions + ESM-2 parquet ...")
    gene_to_rxns, gene_to_locus, rxn_to_genes, rxn_to_mets = (
        load_sbml_genes_and_pathways()
    )
    v15 = load_v15_predictions(V15_PRED)
    print(f"  v15 predictions: {len(v15)} rows  "
          f"({int((v15['essential']==1).sum())} essential / "
          f"{int((v15['essential']==0).sum())} non-essential)")

    name_lookup = dict(zip(v15["locus_tag"], v15["gene_name"]))

    if ESM_PARQUET.exists():
        esm = load_esm2(ESM_PARQUET)
    else:
        esm = None
        print(f"  [warn] {ESM_PARQUET} missing; Category A skipped")

    rows: list[dict] = []

    if esm is not None:
        print(f"\n[A] paralog selection (cos > {args.paralog_cos}) ...")
        a_rows = select_paralog_pairs(
            esm, args.n_per_category, args.paralog_cos, args.seed,
        )
        print(f"  -> {len(a_rows)} pairs")
        rows.extend(a_rows)

    print(f"\n[B] same-pathway sequential (NEGATIVE control) ...")
    b_rows = select_same_pathway_pairs(
        rxn_to_genes, rxn_to_mets, gene_to_locus,
        args.n_per_category, args.seed,
    )
    print(f"  -> {len(b_rows)} pairs")
    rows.extend(b_rows)

    print(f"\n[C] random different-pathway non-essentials ...")
    c_rows = select_random_different_pathway_pairs(
        v15, gene_to_rxns, rxn_to_mets, gene_to_locus,
        args.n_per_category, args.seed,
    )
    print(f"  -> {len(c_rows)} pairs")
    rows.extend(c_rows)

    print(f"\n[D] transporter / substrate-consumer ...")
    d_rows = select_transporter_substrate_pairs(
        rxn_to_genes, rxn_to_mets, gene_to_locus,
        args.n_per_category, args.seed,
    )
    print(f"  -> {len(d_rows)} pairs")
    rows.extend(d_rows)

    print(f"\n[E] manual curation ...")
    e_rows = select_manual_pairs(
        gene_to_locus, name_lookup, args.n_per_category, args.seed,
    )
    print(f"  -> {len(e_rows)} pairs")
    rows.extend(e_rows)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["locus_a", "locus_b"])
    print(f"\ntotal pairs: {len(df)}")
    print(df.groupby("category").size().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
