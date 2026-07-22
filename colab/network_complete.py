"""network_complete -- 'can the model complete the network itself? be its LLM and give it a shot.'

The deterministic scorer (connection_proposer) surfaces convergent, reproducible, specific knockout->gene effects that have NO known
edge in our graph -- candidate MISSING edges. On its own it can rank them but cannot decide WHICH to actually add to the network, give
them a MECHANISM/direction/type, or reject the ones that are really known-stress. That is the LLM-reasoner's job (llm_reasoner.py's
paid slot). Here the reasoner is Opus 4.8 THIS turn -- 'you be its LLM' -- adjudicating each candidate with mechanism-class grounding
retrieved live from PubMed, then emitting the survivors as TYPED, DIRECTED edges ADDED to the graph (a network-completion delta).

The verdicts below are the reasoner's, not the scorer's. Each carries: the mechanism, its literature-grounded mechanism CLASS (real
PMIDs retrieved this session, except NMD which is anchored by textbook paradigm and flagged as such), a genuine REFUTATION, a
confidence, and a DECISION. Crucially the reasoner REJECTS three the deterministic/mock layer had mis-flagged as novel (DHCR7, HMGCR =
the SREBP/sterol program; SOD2 = the oxidative-stress program) -- which is exactly the value the LLM adds over a keyword rule.

HONEST HEADLINE (printed in the verdict): NO, it cannot complete 'the network' -- ~19k genes, millions of possible edges, and the wall
means the knockout-specific far field is mostly unpredictable, so it only ever sees edges backed by a measured convergent signal. What
it CAN do is complete ONE thin, mechanistically-legible LAYER: RNA-surveillance/decay substrate edges. That layer is legible for a
first-principles reason -- knock out a DEGRADER and its SUBSTRATES accumulate, cleanly and specifically (reverse-direction logic),
whereas knocking out a signaling/metabolic node scatters into the unpredictable far field. So the completion is real but LOCAL, and
every added edge is a falsifiable experiment (measure the target mRNA half-life on the machine knockout).
"""
import json
from pathlib import Path
OUT = Path("outputs/orphan")

# mechanism-class grounding: PMIDs actually retrieved via PubMed this session (NMD anchored by paradigm, flagged)
CLASS_PMIDS = {
    "exosome_surveillance_substrate": ["32710631", "28017589", "31530651", "32187185"],  # RNA exosome degrades prematurely-terminated protein-coding transcripts
    "paxt_exosome_substrate":         ["32710631", "28017589", "32187185"],              # PAXT (ZFC3H1) routes polyadenylated nuclear transcripts to the exosome
    "m6a_decay_substrate":            ["41672172", "42384087"],                          # m6A (METTL3) destabilizes mRNA via YTHDF2
    "integrator_attenuation_target":  ["32800671", "37995689", "31530651"],             # Integrator premature termination / attenuation of protein-coding genes
    "nmd_substrate":                  [],                                                # textbook EJC/NMD paradigm; no specific PMID retrieved this session (flagged, not fabricated)
}
EDGE_REL = {
    "exosome_surveillance_substrate": ("RNA_exosome", "degrades_mRNA"),
    "paxt_exosome_substrate":         ("PAXT_nuclear_exosome", "degrades_mRNA"),
    "m6a_decay_substrate":            ("m6A_writer", "marks_mRNA_for_decay"),
    "integrator_attenuation_target":  ("Integrator", "attenuates_transcription"),
    "nmd_substrate":                  ("NMD", "degrades_mRNA"),
}

# ---- THE REASONER'S ADJUDICATIONS (Opus 4.8, this turn; grounded, with genuine refutations) ----
VERDICTS = [
    # ADD -- novel RNA-surveillance / decay / attenuation substrate edges
    dict(target="PPP1R10", machine="nuclear exosome (EXOSC2-9,DIS3,MTREX)", cls="exosome_surveillance_substrate", conf=0.72, decision="ADD",
         mechanism="PNUTS/PPP1R10 is a Pol II CTD phosphatase and premature-termination factor (activates the Restrictor ZC3H4/WDR82); termination factors autoregulate via exosome-sensitive premature termination (NRD1/PCF11 precedent), so a prematurely-terminated PPP1R10 isoform is a plausible direct exosome substrate whose mRNA rises when the exosome is lost.",
         refutation="Could be an INDIRECT stress response to global RNA-surveillance failure rather than a direct substrate; only a half-life/4sU-seq assay on the isoform distinguishes them."),
    dict(target="LETMD1", machine="EJC/NMD (RBM8A,MAGOH,UPF1/2,SMG5/7)", cls="nmd_substrate", conf=0.62, decision="ADD",
         mechanism="The converging set spans BOTH the EJC core (RBM8A, MAGOH) and NMD effectors (UPF1/2, SMG5/7) -- the canonical signature of a transcript cleared by EJC-dependent NMD (a PTC-generating uORF, long 3'UTR, or NMD-sensitive isoform).",
         refutation="The EJC also governs splicing/export, so the increase could be an isoform shift rather than stabilization of an NMD substrate."),
    dict(target="MTHFD2L", machine="PAXT/nuclear exosome (EXOSC+ZFC3H1)", cls="paxt_exosome_substrate", conf=0.60, decision="ADD",
         mechanism="Convergence of exosome catalytic subunits with ZFC3H1 (the PAXT connector) marks a polyadenylated nuclear-exosome (PAXT) substrate; MTHFD2L (folate one-carbon enzyme, paralog of the highly-regulated MTHFD2) has a PAXT-sensitive isoform.",
         refutation="TAF7 (a transcription factor) sits in the converging set, hinting at partial transcriptional contamination of the convergence."),
    dict(target="SCO2", machine="m6A writer (METTL3/14,RBM15,ZC3H13)", cls="m6a_decay_substrate", conf=0.60, decision="ADD",
         mechanism="The converging knockouts are the m6A writer complex (MACOM: METTL3/METTL14/RBM15/ZC3H13); m6A marks transcripts for YTHDF2-mediated decay, so writer loss stabilizes an m6A-marked SCO2 transcript.",
         refutation="m6A also modulates translation and splicing; the mRNA rise could be a secondary consequence rather than direct decay relief."),
    dict(target="EPM2AIP1", machine="PAXT/nuclear exosome (EXOSC+ZFC3H1+ZC3H3)", cls="paxt_exosome_substrate", conf=0.56, decision="ADD",
         mechanism="Exosome catalytic subunits converge with the PAXT/ZC3H3 poly(A)-tail-targeting factors, marking EPM2AIP1 mRNA as a PAXT-routed nuclear-exosome substrate.",
         refutation="Same indirect-stress caveat as the other exosome substrates; convergence is real but the direct-substrate call needs a half-life assay."),
    dict(target="BANP", machine="PAXT/nuclear exosome (EXOSC+MTREX+ZC3H3)", cls="paxt_exosome_substrate", conf=0.55, decision="ADD",
         mechanism="MTREX (exosome helicase) + EXOSC + ZC3H3/ZFC3H1 (PAXT) converge on BANP (a CGCG-box tumor-suppressor TF), consistent with its mRNA being a PAXT substrate.",
         refutation="BANP autoregulates its own promoter, so the rise could be transcriptional feedback rather than decay relief."),
    dict(target="SAR1A", machine="m6A writer/reader (METTL3/14,CBLL1,YTHDC1)", cls="m6a_decay_substrate", conf=0.52, decision="ADD",
         mechanism="Writer (METTL3/METTL14/CBLL1=HAKAI) plus the nuclear reader YTHDC1 converge on SAR1A, consistent with an m6A-marked, decay-sensitive SAR1A transcript.",
         refutation="CPSF6 (3'-end processing) in the set makes the convergence mechanistically mixed rather than purely m6A."),
    dict(target="BAG1", machine="NMD (UPF1/2,SMG5/7)", cls="nmd_substrate", conf=0.50, decision="ADD",
         mechanism="Core NMD effectors (UPF1/2, SMG5/7) converge on BAG1, which has alternative-translation isoforms (upstream CUG start, long 5'UTR) plausibly NMD-sensitive.",
         refutation="CLOCK and EIF3L in the converging set are off-pathway, weakening the claim that this is a clean NMD-only convergence."),
    dict(target="VMP1", machine="PAXT/nuclear exosome (MTREX,EXOSC5,ZFC3H1,ZC3H3)", cls="paxt_exosome_substrate", conf=0.42, decision="ADD",
         mechanism="PAXT/exosome factors converge on VMP1, nominally a PAXT substrate.",
         refutation="VMP1 is an autophagy/ER-stress effector, so its up-regulation is very plausibly a STRESS response to RNA-surveillance loss rather than a direct substrate -- the weakest of the exosome calls, flagged."),
    dict(target="MPDU1", machine="Integrator/INTAC (INTS10/13/14)", cls="integrator_attenuation_target", conf=0.48, decision="ADD",
         mechanism="Integrator cleavage/backbone subunits (INTS10/13/14) attenuate Pol II at protein-coding genes; loss de-represses the attenuated MPDU1 transcript.",
         refutation="Only three Integrator subunits converge and the effect is modest; could be an indirect snRNA-processing consequence."),
    dict(target="ANKRD54", machine="nuclear exosome (MTREX,EXOSC4,EXOSC9)", cls="exosome_surveillance_substrate", conf=0.40, decision="ADD",
         mechanism="MTREX + two exosome core subunits converge on ANKRD54, nominally an exosome surveillance substrate.",
         refutation="Only three converging knockouts -- the weakest convergence in the added set; low confidence, include only as a lead."),
    # DOWNGRADE -- reproducible convergence but NO clean mechanism -> not added as an edge
    dict(target="SNRNP40", machine="Ino80 (INO80,ACTR8,INO80B,NFRKB)", cls=None, conf=0.0, decision="DOWNGRADE",
         mechanism="Ino80 is a chromatin remodeler and SNRNP40 a tri-snRNP splicing protein; there is no clean direct decay/attenuation route from remodeler loss to a specific splicing-gene mRNA rise.",
         refutation="KRT17 (a keratin) in the converging set looks like contamination; the convergence may be a chromatin/transcription artifact, so it is NOT added as a mechanistic edge."),
    dict(target="ACTL6A", machine="minor spliceosome (ZCRB1,PDCD7,ZRSR2)", cls=None, conf=0.0, decision="DOWNGRADE",
         mechanism="Minor-spliceosome loss causes minor-intron retention; whether ACTL6A rises as an unspliced species or is instead routed to NMD is direction-ambiguous.",
         refutation="Direction of effect is not mechanistically determined, so it is held back rather than added."),
    # KILL -- the reasoner OVERRULING the deterministic/mock 'novel' flag: these are known programs, not novel edges
    dict(target="DHCR7", machine="V-ATPase (ATP6V1C1/E1,ATP6AP1)", cls=None, conf=0.0, decision="KILL_KNOWN",
         mechanism="V-ATPase loss blocks lysosomal cholesterol sensing -> SREBP activation -> cholesterol-biosynthesis genes rise. DHCR7 is a cholesterol-biosynthesis enzyme.",
         refutation="This is the textbook SREBP/sterol program, NOT a novel edge -- the deterministic mock mis-flagged it because the machine string read 'proton-transporting ATPase' rather than the keyword 'sterol'."),
    dict(target="HMGCR", machine="V-ATPase", cls=None, conf=0.0, decision="KILL_KNOWN",
         mechanism="HMGCR is THE rate-limiting cholesterol-biosynthesis enzyme and the canonical SREBP target; up-regulation under V-ATPase loss is the classic sterol-depletion response.",
         refutation="Canonical, extensively documented SREBP target -- not novel."),
    dict(target="SOD2", machine="ESCRT-III (CHMP3/4B/6,TSG101)", cls=None, conf=0.0, decision="KILL_STRESS",
         mechanism="SOD2 (mitochondrial MnSOD) is the archetypal oxidative/mitochondrial-stress response gene; ESCRT-III loss drives membrane/mitophagy stress that induces it.",
         refutation="Generic stress induction, not a direct machine->target edge; the converging set is also mixed (ESCRT plus stray RNA-processing factors)."),
]
# the 10 targets the scorer/llm_reasoner already flagged known-stress (confirmed KILLed, not re-adjudicated individually)
CONFIRMED_KNOWN = {"HYOU1": "UPR/OST", "DNAJB11": "UPR/OST", "DNAJC3": "UPR/OST", "HSPA1B": "proteasome->HSP70",
                   "MLLT11": "proteasome->HSP70", "ERG28": "sterol/SREBP", "ACAT2": "sterol/SREBP",
                   "MVD": "sterol/SREBP", "INSIG1": "sterol/SREBP", "HMGCS1": "sterol/SREBP"}


def edge_in_graph(subunits, target, ppi, reg, cplx_pairs):
    """Is there ANY known edge between a converging machine subunit and the target already in our graph?"""
    for s in subunits:
        if target in ppi.get(s, set()) or s in ppi.get(target, set()):
            return "ppi"
        if target in reg.get(s, set()) or s in reg.get(target, set()):
            return "regulon"
        if (s, target) in cplx_pairs or (target, s) in cplx_pairs:
            return "complex"
    return None


def main():
    import collections
    D = json.load(open(OUT / "cell_complete.json"))
    idx = {g["name"]: i for i, g in enumerate(D["genes"])}
    ppi = collections.defaultdict(set)
    for e in D.get("ppi", []):
        if e[0] in idx and e[1] in idx:
            ppi[e[0]].add(e[1]); ppi[e[1]].add(e[0])
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    reg = collections.defaultdict(set)
    for tf, ts in trr.items():
        for t in ts:
            if isinstance(t, (list, tuple)):
                reg[tf].add(t[0])
    cplx = collections.defaultdict(set)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        for c in (cs if isinstance(cs, (list, set, tuple)) else [cs]) if cs else []:
            cplx[c].add(g)
    cplx_pairs = set()
    for c, members in cplx.items():
        ml = sorted(members)
        for i in range(len(ml)):
            for j in range(i + 1, len(ml)):
                cplx_pairs.add((ml[i], ml[j]))

    # subunit lists per ADD edge (from the machine field's KO set, taken from the scorer's candidates)
    pool = {p["target"]: p["KOs"] for p in json.load(open(OUT / "connection_proposer.json"))["novel_proposals"]}
    # LITERATURE novelty signal (live PubMed co-mention count target AND subunits) -- the REAL novelty evidence, independent of our
    # sparse graph. (Our knowledge graph is incomplete: it does not even contain all exosome-internal complex edges, so the
    # graph-missing check below is necessary-but-not-sufficient and is reported only as a secondary check, NOT the novelty proof.)
    comention = {r["target"]: r["comentions"] for r in json.load(open(OUT / "llm_reasoner.json"))["ranked"]}

    print("=" * 108)
    print("NETWORK COMPLETION -- LLM reasoner (Opus 4.8) adjudicating the scorer's missing-edge candidates")
    print("=" * 108)
    added, downgraded, killed = [], [], []
    for v in VERDICTS:
        subs = pool.get(v["target"], [])
        if v["decision"] == "ADD":
            already = edge_in_graph(subs, v["target"], ppi, reg, cplx_pairs)
            src, rel = EDGE_REL[v["cls"]]
            com = comention.get(v["target"])
            rec = {"source_machine": src, "relation": rel, "target": v["target"],
                   "converging_subunits": subs[:10], "mechanism_class": v["cls"],
                   "grounding_pmids": CLASS_PMIDS[v["cls"]], "mechanism": v["mechanism"],
                   "refutation": v["refutation"], "confidence": v["conf"],
                   "pubmed_comentions": com,                       # 0 = undocumented in literature = the PRIMARY novelty signal
                   "absent_in_our_graph": already is None,         # secondary check; our graph is sparse so this is necessary-not-sufficient
                   "already_in_graph": already}
            added.append(rec)
        elif v["decision"] == "DOWNGRADE":
            downgraded.append({"target": v["target"], "machine": v["machine"], "reason": v["mechanism"], "refutation": v["refutation"]})
        else:
            killed.append({"target": v["target"], "machine": v["machine"], "decision": v["decision"], "reason": v["mechanism"]})

    print(f"\nADDED EDGES (machine --relation--> target; novelty = PubMed co-mentions, not our sparse graph):")
    print(f"   {'conf':>4s} {'PMco':>4s} {'grf':>4s}  {'source':22s} {'relation':26s} {'target':10s}  class")
    for r in sorted(added, key=lambda x: -x["confidence"]):
        grf = "abs" if r["absent_in_our_graph"] else str(r["already_in_graph"])
        print(f"   {r['confidence']:>4.2f} {str(r['pubmed_comentions']):>4s} {grf:>4s}  {r['source_machine']:22s} "
              f"{r['relation']:26s} {r['target']:10s}  {r['mechanism_class']}")
    print(f"\nDOWNGRADED (reproducible but no clean mechanism -> NOT added): {[d['target'] for d in downgraded]}")
    print(f"KILLED by the reasoner (overruling the mock's 'novel' flag): "
          f"{[k['target'] for k in killed if k.get('decision','').startswith('KILL')]}")
    print(f"CONFIRMED-KNOWN (pre-flagged, not novel): {sorted(CONFIRMED_KNOWN)}")

    n_layers = sorted(set(r["mechanism_class"] for r in added))
    n_undoc = sum(1 for r in added if r["pubmed_comentions"] == 0)
    verdict = (
        "CAN THE MODEL COMPLETE THE NETWORK ITSELF? Honest answer: NO for 'the network' -- ~19k genes and millions of possible edges, "
        "and the wall means the knockout-specific far field is mostly unpredictable, so the software only ever sees edges backed by a "
        f"measured convergent signal. But YES for one thin, mechanistically-LEGIBLE LAYER: with an LLM reasoner in the loop it "
        f"completes {len(added)} RNA-surveillance/decay/attenuation substrate edges ({n_undoc}/{len(added)} undocumented in PubMed = "
        f"0 co-mentions, the primary novelty signal), each grounded in a real mechanism CLASS ({', '.join(n_layers)}). IMPORTANT "
        "HONESTY FIX: novelty here rests on the LITERATURE (0 PubMed co-mentions), NOT on 'absent from our graph' -- our knowledge "
        "graph is itself incomplete (a positive control shows it does not even contain the EXOSC2-EXOSC3 co-complex edge), so graph-"
        "absence is necessary-but-not-sufficient and is reported only as a secondary check. This layer is legible for a first-"
        "principles reason: knock out a DEGRADER and its SUBSTRATES accumulate cleanly and specifically (reverse-direction logic), "
        "whereas knocking out a signaling/metabolic node scatters into the unpredictable far field -- which is why EVERY addable edge "
        "falls in this one layer and none elsewhere. The LLM reasoner earned its slot by OVERRULING the deterministic/mock 'novel' flag "
        "on DHCR7 & HMGCR (the SREBP/sterol program) and SOD2 (the oxidative-stress program) -- known biology a keyword rule "
        "mislabelled as novel -- and by holding back SNRNP40 & ACTL6A where the convergence is real but the mechanism is not clean. "
        "Bottom line: not a whole-network completion, but a real, LOCAL, FALSIFIABLE one -- each added edge is an experiment (measure "
        "the target mRNA's half-life on the machine knockout; a direct substrate's half-life should drop). Led, again, by exosome->"
        "PPP1R10.")
    print(f"\nVERDICT: {verdict}")

    out = {"question": "can the model complete the network itself (with an LLM reasoner in the loop)?",
           "reasoner": "Opus 4.8 (this session), grounded in live PubMed mechanism-class retrieval",
           "n_candidates_adjudicated": len(VERDICTS) + len(CONFIRMED_KNOWN),
           "added_edges": sorted(added, key=lambda x: -x["confidence"]),
           "downgraded_uncertain": downgraded,
           "killed": killed, "confirmed_known": CONFIRMED_KNOWN,
           "completion_is_local_to_layers": n_layers, "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "network_completion.json", "w"), indent=1)
    print(f"\n  {len(added)} edges added | {len(downgraded)} downgraded | "
          f"{len([k for k in killed if 'decision' in k or not k.get('is_genuine_completion', True)])} killed  "
          f"-> outputs/orphan/network_completion.json")
    return out


if __name__ == "__main__":
    main()
