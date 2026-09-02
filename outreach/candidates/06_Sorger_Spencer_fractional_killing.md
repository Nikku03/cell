# Fractional killing and drug-tolerant persisters — Sorger & Spencer

**Their area.** Single-cell variability in apoptotic response; fractional killing; drug-tolerant
persister states in cancer. Sabrina Spencer and Peter Sorger.

## Why this is the same mathematics as bacterial persistence

Fractional killing is the same shape of problem as antibiotic tolerance: a **deterministic model of
the average cell returns a fractional survivor and therefore no probability at all**, while the
clinically decisive question is whether **zero** cells survive. REM was built for exactly that
gap, and the machinery is organism-agnostic — it is a master equation over a small circuit, not a
bacterial model.

## The one transferable result already measured

Replacing a **cycling** drug schedule by its time average — the standard approximation — returned
the *identical* number for every schedule tested, while the exact answer varied by **4.9×** in one
parameterisation and **31×** in another — the latter with a band, under 40% lognormal
uncertainty on every fitted rate over 40 replicates: median 25.9×, IQR [14.5, 48.4], full range
[1.9, 381.3], direction surviving at the 25th percentile. Mean-field is not approximately right for a periodically
driven tail; it is blind to the schedule entirely.

**And a negative that must travel with it:** the *direction* of the schedule effect **flipped**
between hand-picked and fitted parameters — fewer-and-longer exposures won in one, more-and-shorter
won by 31× in the other. So this is a **per-system calculation, not a rule**. Anyone offering a
general "pulse better than continuous" claim is overselling; the calculation has to be run on the
system's own measured rates.

## UNRETRIEVED

No rate constants, single-cell trajectories, or stated gap were retrieved for this group. Nothing
is computed here and nothing is claimed about their data.

## What would be needed

| What | Why |
|---|---|
| Single-cell time-to-death distribution under a fixed drug exposure | Gives the two-state switching rates, as in the Van Bambeke case |
| The exposure schedule actually used (on/off durations) | The tail depends on the schedule, not on the time-averaged dose |
| The population size at which "cure" is being asked about | Sets how deep the tail has to go |

## The question, as a question

*If the same total drug exposure were redistributed across a different on/off schedule, does your
system's exact probability of complete elimination change — and in which direction? We can compute
it from a measured single-cell death-time distribution, but the direction is not predictable
without your numbers.*
