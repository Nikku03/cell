# Corrections to the first nexus x profiler report

The first synthesis of the nexus profiling run was written from an in-flight workflow summary
rather than from `outputs/loop_nexus_catalyst_interactions.json`. Two reviewers disputed it. Both
were partly right and partly wrong, and the artefact settles every point. This file is the record.

## What was wrong in the first report

| claimed | actual, from the artefact | where |
|---|---|---|
| docking is 23,305 of **28,420** core-s, **82.0%** | 23,304.7 of **25,209.5** core-s, **92.4%** | `n6` |
| all FFT docking **88.9%** | rotation loops alone **92.4%** | `n6.docking` |
| **44.1** CPU-s per candidate dock | **38.82** s/candidate | `n6.docking.s_per_candidate_full` |
| ablation worth **6.4x** | **13.23x** (25,209.5 / 1,904.8) | `n6` |
| candidate vocabulary **5,282** | 5,282 named, **5,251** with a sequence; every downstream number uses 5,251 | `n6.vocabulary` |
| adding docking makes the model **worse by 0.0021** | docking is worth **+0.0000** over the whole space, **+0.0010** in the artefact-free subspace | `n5` |
| profiler needs `n_references>=12` and consumes **128 unique evaluations** | R* is a random variable: [12, 12, 12, 48, 12, 48, 12, 6] over 8 draws. The 128-configuration claim was an inference, and it is now **measured and true from R=6 upward** | loop 158, `space_a` |

The 59.7 core-year figure for docking the full vocabulary against every orphan reaction survives:
9,186 x 5,251 x 38.82 s = 59.3 core-years, and the artefact's own line says 59.

## What the reviewers got wrong

`verify:2` alleged that three numbers were "lifted from a different run
(`outputs/loop_feature_interactions.json`, loop 157) and relabelled". They were not. The strings
`0.532`, `5.47`, `0.549`, `0.450` and `0.0021` do not occur in that file at all. The docking-feature
AUCs and the mean rank are `outputs/orphan/nexus_catalyst_pilot.json`, which is where the first
report attributed them:

```
auc/mean        0.4495     auc/clash      0.4525     auc/n_z2      0.5284
auc/top50cell   0.4501     auc/top10      0.4879     auc/std       0.5452
auc/best        0.4982     auc/skew       0.5494     auc/size_only 0.5318
rank_mean       5.4667   (chance 5.5)
```

So "every docking feature in AUC [0.450, 0.549] against a size-only control at 0.532, true-catalyst
mean rank 5.47 vs chance 5.5" was correct and correctly sourced.

`verify:0` proposed 7.99x -> 4.00x for the compute saving. Neither number is derivable from the
artefact; the measured figure is 13.23x. The reviewer was right that the first report's compute
arithmetic was wrong, and wrong about what the right answer is.

## What the structural thesis was, and whether it held

It held. The claim was that the profiler is not a search economy on this problem but that the
ablation it finds is worth a great deal. Loop 158 gates that claim rather than asserting it: P2
FAILS on space A (no configurations saved at the correct setting), P3 PASSES on space B (84 of 256),
P4 PASSES (the schedule beats 2^n from n=7-10 depending on density and reference count), P6 PASSES
(zero false positives at every setting, so the error is one-sided and only a reported ABSENCE needs
the expensive setting).

## Process note

The first report was written from a workflow's summary of its own agents rather than from the
artefact those agents wrote. That is the defect, and it is the same shape as the three
"prose contradicting its own gate" defects that produced `gate_guard.verdict` in loops 148-150:
narration produced beside the measurement instead of from it. `gate_guard.verdict` does not catch
this class either -- it conditions prose on a gate in the same process, and here the prose was
written in a different process from the gate. The rule that does catch it is the one applied
afterwards: read the JSON before writing the sentence.
