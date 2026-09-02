# Gyrase inhibitors and phage — Maxwell & Davidson

**Their area.** DNA gyrase mechanism and inhibition (Maxwell); phage biology and anti-CRISPR
(Davidson). Relevant local work: Cook *et al.*, *International Journal of Antimicrobial Agents*
66:107613 (2025), PMID 40930191, [DOI](https://doi.org/10.1016/j.ijantimicag.2025.107613) — phage
therapy for chronic UTI.

## The honest assessment: phage breaks the model's central assumption

Everything REM has validated assumes a **fixed or externally scheduled** killing pressure. A phage
**self-amplifies**: the killing agent's concentration is a state variable driven by the bacterial
population it is killing. That makes the system nonlinear and the drug schedule endogenous, and
none of the schedule results transfer.

**No claim is made that REM handles phage dynamics. It has not been tested on them.**

## Where there is real contact — gyrase

Gyrase inhibition **is** a small, well-mixed, exactly-solvable circuit, and resistance emergence
under inhibition is a **rare-event** question of the right shape: what is the probability that a
resistant lineage establishes during a course, given a mutation rate and a population trajectory?
That is a tail, and the mean-field answer to it is a fractional mutant, which is not a probability.

The relevant transferable result is the same one throughout this set: **treating a cycling drug as
its time average returned the identical number for every schedule tested**, while the exact answer
moved 4.9× and 31× (the second with a band: median 25.9×, IQR [14.5, 48.4], full range
[1.9, 381.3]). For resistance emergence the schedule matters for the same reason it matters
for eradication — the tail is decided by the excursions, not by the mean.

## What is NOT claimed

- No phage capability.
- Nothing about their data. No inputs retrieved, nothing computed.

## What would be needed

A per-generation mutation rate to resistance, a population trajectory under the inhibitor (or the
two slopes of a time-kill curve), and the dosing schedule. Then: exact P(resistance establishes),
with a band, and how it moves with the schedule at fixed total dose.

## The question, as a question

*For a gyrase inhibitor at a known mutation rate and dosing schedule, is the probability that a
resistant lineage establishes ever computed exactly — or is it estimated from a mean mutant count?
Those are different quantities, and at the population sizes that matter the difference is not
small.*
