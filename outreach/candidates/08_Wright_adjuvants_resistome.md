# Antibiotic adjuvants and the resistome — Gerard Wright (McMaster)

**Verified work.** According to PubMed, Jacques *et al.*, *Genetics* 214:1103–1120 (2020),
PMID 32094149, [DOI](https://doi.org/10.1534/genetics.119.302851) — a compound that "enhanced the
ability of" an antibiotic. Also Gordzevich *et al.*, *Nature* 655:478–486 (2026), PMID 42343126,
[DOI](https://doi.org/10.1038/s41586-026-10647-9).

## The honest assessment: the fit is weak, and here is why

REM's strength is **deep tails of small stochastic circuits** — probabilities of complete
elimination, rare switching, schedule dependence. Adjuvant discovery is a **combinatorial search**
problem over compound × strain × mechanism. Those are different mathematics and pretending
otherwise would not survive one conversation.

**This document exists to say the offer is weak, not to dress it up.**

## The one place there is a genuine seam

An adjuvant that works by **suppressing persister formation** rather than by killing changes a rate
constant in exactly the two-state model that document 02 analyses. For such a compound the
question "does it reduce the probability that any cell survives a course" is a tail question, and
the mean-field answer is provably blind to schedule.

The transferable measured result: replacing a cycling drug by its time average returned the
**identical** number for every schedule, while the exact answer varied **4.9×** and **31×** in two
parameterisations (the second with a band: median 25.9×, IQR [14.5, 48.4], range [1.9, 381.3]). The **direction flipped** between them — so it is a per-compound calculation,
never a rule.

## What is NOT claimed

- Nothing about their data. No inputs were retrieved and nothing was computed.
- No claim that REM helps with screening, hit-finding, resistome annotation, or mechanism
  assignment. It does not.

## What would be needed for the seam to be real

A candidate adjuvant with a **measured effect on persister formation or waking rate** (not just on
MIC), plus a biphasic time-kill curve with both slopes. Then: the exact probability of eradication
under their dosing schedule versus the standard mean-field estimate, with a band.

## The question, as a question

*Do any of your adjuvants act on persister formation rather than on killing — and if so, is there a
time-kill curve for one of them with both the fast and slow slopes reported? That is enough to
compute an upper bound on the formation rate for free, and to ask whether the compound's benefit
survives the actual dosing schedule.*
