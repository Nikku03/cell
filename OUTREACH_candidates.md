# Candidates — framed as a question, not a claim

**The question we are asking, in one sentence:** *when the clinical endpoint is "did any survive",
rather than "how many survived", can it be that averaging a cycling drug concentration doesn't
approximate the answer but deletes the variable the answer depends on?*

Every entry below was verified against a real indexed paper. Bibliographic data are from PubMed.
Two near-misses are recorded because they are the kind of error that ruins an outreach email:
**"Coombes BK" resolves to three different researchers** (Brian K at McMaster, Brooke K at Griffith
in sports medicine, Jeff S at Queensland), and one prominent persistence paper found en route
(PMID 25848049) is **RETRACTED** and must not be cited.

---

## What we can put on the table, and what we cannot

**Can, and it is validated:**

1. **Exact probability that zero cells survive** a given schedule — the deep tail, computed rather
   than sampled. Validated by reproducing, without being told it, the result that the
   survival-maximising lag equals the drug-exposure duration, at 4 of 4 exposure durations tested
   (1.5→1.5, 2.5→2.0, 3.5→3.0, 5.0→5.0 h).
2. **How much a time-averaged drug model misses.** In our test it returned the *identical* number,
   0.131839, for every schedule from one 9-hour exposure to thirty 18-minute ones, while the exact
   answer varied 4.9× — because averaging destroys the schedule. Up to 0.90 orders wrong in the
   probability of treatment failure.
3. **An uncertainty band on any rare-event number.** Headline 7.5× ratio between schedules becomes
   median 6.37×, IQR [3.26, 8.64] under 40% rate uncertainty. It never inverted.
4. **Which measured rate matters most for the tail** — i.e. which parameter is worth the effort of
   measuring precisely, and which is nearly free to get wrong.
5. **Whether a proposed model is structurally incomplete**, given data at ≥6 conditions — with the
   detection limit printed alongside, because "no gap detected" only means "no gap above this size".

**Cannot, and this belongs in the first email:**

- We cannot supply rate constants. Every number above is conditional on parameters *they* would
  provide; ours are illustrative.
- We have never validated a prediction against a real patient or a real infection. The validation
  is against one published in-vitro evolution result.
- Nothing here addresses pharmacokinetics, tissue penetration, immune clearance, or resistance
  evolution. Those are most of the problem.
- The detector has two blind spots: a mechanism below its detection limit, and any error that is
  the same at every condition. The second is not usually stated and is the more dangerous.

---

## McMaster University

**Gerard D. Wright** — David Braley Centre for Antibiotic Discovery; Michael G. DeGroote Institute
for Infectious Disease Research.
*Verified:* Jacques et al., Genetics 214:1103-1120 (2020), PMID 32094149,
[DOI](https://doi.org/10.1534/genetics.119.302851) — a compound that "enhanced the ability of
rifampin to eradicate antibiotic-tolerant persister cells". Also co-senior author on
Gordzevich et al., Nature 655:478-486 (2026), PMID 42343126,
[DOI](https://doi.org/10.1038/s41586-026-10647-9).
**Question for him:** your adjuvant is scored on how *many* persisters it eradicates. Can the same
compound be scored on the probability that *none* remain — and does that reorder which schedule you
would pair it with?

**Eric D. Brown** — same centre; co-senior author on the Nature 2026 megacluster paper above, which
reports four natural-product families acting **synergistically** in a mouse MDR *E. coli* model.
**Question for him:** synergy is conventionally scored by FIC index on population kill, which is a
mean. Eradication is a tail. Can two combinations with the same FIC differ in eradication
probability — and if so, is FIC ranking the wrong thing?

**Lori L. Burrows** — Department of Biochemistry and Biomedical Sciences; DeGroote Institute.
*Verified:* Burrows, ACS Infect Dis 4:1041-1047 (2018), PMID 29771109,
[DOI](https://doi.org/10.1021/acsinfecdis.8b00112) — *Pseudomonas* pipeline with explicit attention
to antimicrobial-tolerant biofilms.
**Question for her:** a biofilm's tolerance looks mathematically like a distribution of wake-up
times. If that distribution were measured, can we compute the exposure duration that maximally
*mismatches* it, rather than the concentration that maximally kills?

**Brian K. Coombes** — Department of Biochemistry and Biomedical Sciences; DeGroote Institute.
*Verified:* Massicotte et al., PLoS Pathog 21:e1013132 (2025), PMID 40924764,
[DOI](https://doi.org/10.1371/journal.ppat.1013132) — the Salmonella Psp system "supports bacterial
persistence in host tissues and survival within macrophages", with temporal transcriptomics across
four infection stages.
**Question for him:** you already have expression across multiple infection stages. Can that series
be used to ask whether a two-state persistence model is *structurally sufficient*, or whether a
component is missing — with the detection limit stated, so a null result means something?

---

## University of Toronto

**Karen L. Maxwell and Alan R. Davidson** — Department of Biochemistry.
*Verified:* Cook et al., Int J Antimicrob Agents 66:107613 (2025), PMID 40930191,
[DOI](https://doi.org/10.1016/j.ijantimicag.2025.107613) — phage therapy for chronic UTI in which
the patient **relapsed two weeks after clinical improvement, with the same isolate**.
**Strongest single match on this list.** Relapse-after-apparent-clearance *is* the quantity we
compute: P(not eradicated). **Question for them:** for that documented case, can we ask what the
survival probability looked like as a function of the dosing interval — and how much of the answer
rests on the wake-up-time distribution nobody measured?

**Catherine A. O'Brien** — Princess Margaret Cancer Centre; Departments of Surgery and Medical
Biophysics.
*Verified:* co-author, Russo et al., Nat Rev Cancer 24:694-717 (2024), PMID 39223250,
[DOI](https://doi.org/10.1038/s41568-024-00737-z) — roadmap on cancer drug-tolerant persister cells.
**Question for her:** cancer DTPs and bacterial persisters are the same mathematical object — a
dormant subpopulation that survives a cycling drug. Does relapse probability versus treatment
schedule transfer across that boundary?

---

## York University (Toronto)

**Gerald F. Audette** — Department of Chemistry, 4700 Keele Street, Toronto.
*Verified affiliation:* Bragagnolo & Audette, Acta Crystallogr D 80:834-849 (2024), PMID 39607821,
[DOI](https://doi.org/10.1107/S205979832401132X) — *Pseudomonas aeruginosa* type IV pilin.
*Also:* Rodriguez & Audette, Struct Dyn 13:024701 (2026), PMID 41858832,
[DOI](https://doi.org/10.1063/4.0001201) — F-plasmid Type IV Secretion System, "a complex nanomachine
central to antibiotic resistance dissemination".
**Note this is a different question, and worth being honest about.** His work is resistance
*spread*, not persistence. But conjugative transfer at low donor density is itself a rare
stochastic event. **Question for him:** the transfer *rate* is measured as a population average.
Can we instead compute the probability that *at least one* transfer event occurs in a given
window — which is what actually determines whether resistance establishes?

---

## The people whose result we validated against

**Nathalie Q. Balaban** — Racah Institute of Physics, The Hebrew University of Jerusalem.
*Verified:* Fridman, Goldberg, Ronin, Shoresh & Balaban, Nature 513:418-421 (2014), PMID 25043002,
[DOI](https://doi.org/10.1038/nature13469) — evolved lag time matches the antibiotic-exposure
interval. Also co-author on the cancer DTP roadmap above (PMID 39223250), so she is already working
across the bacteria/cancer boundary.
**Question for her — and this one should be asked first, because she can refute it fastest:** we
reproduced your lag-matching computationally without encoding it. Does the eradication-probability
version of your framework tell you anything your survival-fraction version doesn't? Our result that
fewer, longer exposures beat many short ones at fixed total drug is monotone, not resonant — is that
consistent with what you see?

---

## Intracellular persisters and adjuvants

All four verified from Lu et al., Nat Microbiol 10:3013-3025 (2025), PMID 41073665,
[DOI](https://doi.org/10.1038/s41564-025-02124-2) — a host-directed compound (KL1) that sensitizes
intracellular *S. aureus*, *Salmonella* and *M. tuberculosis* persisters to antibiotics.

- **Brian P. Conlon** — Department of Microbiology and Immunology, UNC Chapel Hill (senior author)
- **Sophie Helaine** — Department of Microbiology, Harvard Medical School
- **Sarah E. Rowe** — UNC Chapel Hill
- **Vance G. Fowler** — Division of Infectious Diseases, Duke University School of Medicine

**Question for this group:** KL1 works by *waking* persisters into a killable state. That is exactly
the lag-time knob in our model, and our validated result says the optimum depends on the exposure
window. Can the adjuvant's benefit be schedule-dependent — largest at some exposure durations and
absent at others — and is that testable with the assays you already run?

---

## How to open

Lead with the failure, not the capability. The most credible opening sentence is that the
time-averaged model returned **the same number for every schedule we tried** while the exact answer
moved 4.9× — that is a checkable statement about a method most of them already use, and it invites
correction rather than agreement. Attach the reproduction of the Fridman result as evidence the
machinery works, and be first to say what it cannot do.
