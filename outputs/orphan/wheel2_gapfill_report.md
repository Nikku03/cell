# Wheel 2 (proper): gap-directed identification with a dimension filter

Your version of Wheel 2, built literally on the iML1515 E. coli model.

**The cell assembles itself, hits holes it can only describe coarsely, gets
"trucks" of same-family candidates from the soup, then narrows within each
truck by the doorway's measured dimensions.** We measured how much that
narrowing saves.

## The funnel — 115 essential-reaction "holes"

Each hole is an essential metabolic reaction the assembled cell needs but whose
gene we pretend is unknown (in the soup). The truck = every gene in the model
whose enzyme is the same coarse EC-1 class ("I need an oxidoreductase here").
Then we filter within the truck by dimensions read off the flanking
metabolites — **never** from the answer.

| narrowing step | median candidates | expected KOs to hit | true gene kept |
|---|---|---|---|
| **T0** truck — coarse class only ("a door") | 258 | ~129 | 100% |
| **T1** + cofactor (NAD/NADP/FAD/ATP/CoA) | 130 | ~65 | 100% |
| **T2** + redox direction (oxidize/reduce) | 130 | ~65 | 100% |
| **T3** + substrate carbon-size | **47** | ~24 | 100% |
| **T4** + EC sub-subclass (exact style) | **8** | **~4** | **100%** |

**Median shrink 258 → 8 = 32×, and the true gene survives every single filter
(115/115).** The dimensions throw out wrong-shape doors without ever discarding
the one that fits.

The honest split:
- **T3 (cofactor + direction + size)** uses *only* what the network gap tells
  you — the metabolites the cell already has on each side of the hole. That
  alone goes 258 → 47 (**5.5×**), true gene always kept. This is the pure
  "doorway measurements" result.
- **T4** adds the EC 3-level (e.g. "phosphatase acting on a sugar-phosphate",
  3.1.3.x). That's the most informative dimension and the closest to naming the
  function; it takes you to a median of 8. Use it when the gap's chemistry is
  specific enough to pin the sub-subclass.

## Worked examples (doorway → dimensions → candidates)

```
PPA   Inorganic diphosphatase           [hydrolase]
  dims: cofactor=none, dir=n/a, size=small(<=4C), EC=3.6.1.1
  264 -> +cof 136 -> +size 30 -> +EC3 8        (truck of 272 cut to 8)

GTPCII2  GTP cyclohydrolase II            [hydrolase]
  dims: cofactor=ATP/NTP, size=med(5-10C), EC=3.5.4.25
  272 -> +cof 146 -> +size 53 -> +EC3 3        (cut to 3)

USHD  UDP-sugar hydrolase                [hydrolase]
  dims: cofactor=none, size=large(>10C), EC=3.6.1.54
  272 -> +size 45 -> +EC3 6                     (cut to 6)
```

A wet lab staring at "something hydrolase-shaped is missing here" would face
~270 knockouts. Knowing the doorway is *small, cofactor-free, 3.6.1.x* cuts
that to **8 targeted tests** — and the right gene is guaranteed to be in those 8.

## Why this is the missing piece (and not circular)

The dimensions are **read from the assembled network, not from the answer**:
the metabolites flanking the hole are ones the cell already makes/needs, so
their identity, carbon count, and the cofactor the step must use are known
*before* you know which gene fills it. That's legitimate gap geometry — the
doorway's width and height — not peeking at the door.

This is what turns Wheel 2 from "the soup is a big undifferentiated bag" into
"here are the 4–8 genes that physically fit this specific hole — test these
first." It's the within-family **dimension filter** the earlier
`metabolic_gapfill.py` was missing: that one matched soup genes to holes by
family; this one narrows inside the family by chemistry, collapsing the
hit-and-trial set ~30×.

## Where it pays and where it doesn't

- **Pays most** where the truck is large (broad enzyme classes — hydrolases,
  transferases, oxidoreductases): 270 → single digits.
- **Pays least** where the coarse class is already small, or where EC codes are
  missing (the model can't describe the doorway). 115 of the essential
  reactions had a usable EC class; the rest can't be triaged this way.
- **Bound:** this identifies *which gene fills a known required reaction*. It
  assumes the hole itself is correctly flagged as required (that's Wheel 3 /
  FBA's job). The two compose: FBA says "a reaction is missing here," Wheel 2
  says "and here are the 8 genes shaped to fill it."
