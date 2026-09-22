# extrude_gate.json IS SUPERSEDED -- DO NOT ACT ON ITS VERDICT

`extrude_gate.json` (395 pairs, CPU, commit a7657e2) carries the verdict string:

    "UNDERPOWERED -- net +0.0101 but 1 se is 0.0084 ... Raise EXT_NSUB and re-run."

**That instruction has already been executed and the gate is closed.** Commit e1af01b ran 2,984 pairs on an
A100 and wrote `colab_runs/extrude_gate_a100_3000.json`:

    SIM_shuffled_plus_count  0.6806   <- the CONTROL
    SIM_plus_count           0.6779   <- the real simulation
    count_contact            0.6757
    delta_net_of_control    -0.0027 +/- 0.0052       NO-GO
    delta_shape_vs_moment   +0.0000416   (<d^-3> 0.62594 vs <d> 0.62589)

Confirmed a second time on a different GPU generation (RTX PRO 6000 Blackwell); the pipeline is seeded and
deterministic, so device-dependence is ruled out.

**The "raise EXT_NSUB" string is itself a fossil of a fixed bug.** The verdict ladder tested `not consistent`
before the sign of `net`, so when the control beat the simulation it asked for more pairs instead of returning
NO-GO. e1af01b fixed the ladder. The stale file predates the fix.

The JSON is deliberately left byte-identical -- a measurement is not edited after the fact. This marker exists
because the stale verdict was read as a live instruction on 3 Aug and cost a redundant CPU run.

**Cost note for anyone tempted anyway:** the Langevin is 41 s/pair on 4 CPU cores (64 replicas x 3000 steps),
against 278 ms/pair on GPU. Windows are >= 201 beads by construction (`off = lo - 200 kb`), so there is no cheap
CPU path. ~2,000 pairs is ~20 h of CPU against ~14 min of GPU.

See UPDATES.md line ~13415 ("Stage 2+3 at scale on an A100") for the full write-up.
