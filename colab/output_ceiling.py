"""OUTPUT CEILING: the highest score this architecture could reach if its reasoning were perfect.

WHY THIS IS THE DIAGNOSTIC THAT MATTERED. After the genome-wide rebuild, every method scored 4-10x below a
frequency baseline that uses no biology, and the multi-layer reasoner was indistinguishable from its own shuffled
control. The natural next move is to improve the reasoning. This file measures whether that could possibly help,
and the answer is no.

The multi-layer model emits genes from a FIXED MENU: the nine hand-written sensor programmes, 89 genes in total.
So its score has a hard ceiling that has nothing to do with how well it reasons -- fire every sensor correctly on
every knockout, never make a mistake, and it still cannot name a gene outside those 89.

    ORACLE = predict the union of all programmes for every knockout.

That is an upper bound on the architecture, not on the implementation. Measured on the gwps cohort it is 4.4%
recall, and the deployed model is already near it. No amount of better firing logic closes that gap, because the
gap is not in the firing.

THREE NUMBERS, EACH ANSWERING A DIFFERENT OBJECTION.

  "the vocabulary is small but well chosen"   -> compare against the best possible list of the SAME SIZE, chosen
                                                 with the answers in hand. It reaches ~40% recall against the
                                                 oracle's 4.4%, so the 89 genes are roughly 9x worse than 89
                                                 genes could be. Small AND badly chosen.
  "responses share a common programme"        -> count how many moved genes move for exactly ONE knockout. Over
                                                 half do. A fixed menu cannot represent a bespoke response.
  "the sensor genes are at least the right    -> count how many of the 89 ever move at all. A third. The rest are
   kind of gene"                                 genes the model can emit and the readout never shows moving.

WHAT THIS DOES NOT SAY. It does not say the sensor biology is wrong -- ATF4 really is downstream of mitochondrial
import failure. It says a lookup table with 89 entries is the wrong SHAPE for a problem whose answer space is
thousands of genes and largely knockout-specific. The fix is a per-knockout output space, not a longer table.
"""
import collections
import json
import sys
from pathlib import Path

OUT = Path("outputs/orphan")


def main(tag=""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from multilayer_reason import SENSORS

    ans = json.loads((OUT / f"ko_pred_answers{tag}.json").read_text())
    truth = {k: set(v["movers"]) for k, v in ans.items()}
    if not truth:
        print("no answers file -- run ko_predict.py first")
        return
    prog = {g for v in SENSORS.values() for g in v["programme"]}
    allt = set().union(*truth.values())
    tot = sum(len(v) for v in truth.values())

    print(f"cohort{tag or ' 1'}: {len(truth)} knockouts, {tot:,} mover calls over {len(allt):,} distinct genes")
    print(f"model's entire output vocabulary (9 sensor programmes): {len(prog)} genes")
    print(f"  of which ever move at all: {len(prog & allt)}  "
          f"({len(prog) - len(prog & allt)} are genes it can emit and the readout never shows moving)")

    hits = sum(len(prog & v) for v in truth.values())
    print(f"\nORACLE CEILING -- every sensor fires correctly on every knockout, no reasoning errors:")
    print(f"  recall    {hits:,}/{tot:,} = {hits/tot:.1%}")
    print(f"  precision {hits:,}/{len(prog)*len(truth):,} = {hits/(len(prog)*len(truth)):.4f}")
    print("  -> this bounds the ARCHITECTURE. Better firing logic cannot exceed it.")

    cnt = collections.Counter(g for v in truth.values() for g in v)
    best = {g for g, _ in cnt.most_common(len(prog))}
    hb = sum(len(best & v) for v in truth.values())
    print(f"\nBEST POSSIBLE fixed list of {len(prog)} genes, chosen WITH the answers in hand:")
    print(f"  recall {hb:,}/{tot:,} = {hb/tot:.1%}   -- so the chosen 89 are ~{hb/max(hits,1):.0f}x worse than "
          f"89 genes could be")

    once = sum(1 for g, c in cnt.items() if c == 1)
    print(f"\nIS THE RESPONSE A SHARED PROGRAMME? {once:,}/{len(cnt):,} ({once/len(cnt):.0%}) of moved genes move "
          f"for exactly ONE knockout.")
    print("  A fixed menu cannot represent a bespoke response, however well it is fired.")

    json.dump({"n_knockouts": len(truth), "n_mover_calls": tot, "answer_space": len(allt),
               "vocab": len(prog), "vocab_that_moves": len(prog & allt),
               "oracle_recall": hits / tot, "oracle_precision": hits / (len(prog) * len(truth)),
               "best_same_size_recall": hb / tot, "frac_genes_moving_once": once / len(cnt)},
              open(OUT / f"output_ceiling{tag}.json", "w"))
    print(f"\n  -> {OUT/('output_ceiling'+tag+'.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
