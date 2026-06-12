# Essential-bucket rarefaction across 59 bacterial datasets

**Data:** 48,525 essential gene labels with og_id assigned, over 48 organisms, 5,658 distinct essential OGs.

## Cumulative essential-OG count vs organisms added

Median + 25/75-percentile band over 50 random orderings.

| n_orgs | mean OGs | p25 | p75 |
|--------|---------:|----:|----:|
|   1 |      721 |  604 |  788 |
|   2 |     1027 |  921 | 1151 |
|   3 |     1262 | 1101 | 1412 |
|   5 |     1685 | 1504 | 1808 |
|  10 |     2471 | 2323 | 2598 |
|  15 |     3061 | 2845 | 3301 |
|  20 |     3640 | 3452 | 3857 |
|  25 |     4129 | 3955 | 4326 |
|  30 |     4531 | 4384 | 4683 |
|  35 |     4898 | 4785 | 5018 |
|  40 |     5242 | 5176 | 5326 |
|  45 |     5503 | 5471 | 5560 |
|  48 |     5658 | 5658 | 5658 |

## Saturation diagnostics

- **Observed essential OGs at n=48:** 5,658
- **Slope over last 10 orgs added:** +477 OGs total, **48 per org**
- **Chao1 estimated asymptote:** 8084 OGs (observed = 70% of estimated total)
- **Singleton OGs (essential in only 1 org):** 2,441 (43% of all essential OGs)

## Core vs accessory essentialome

| breadth (essential in N orgs) | n OGs | fraction |
|---|---:|---:|
| ≥  1 |  5658 | 100.0% |
| ≥  2 |  3217 |  56.9% |
| ≥  3 |  1989 |  35.2% |
| ≥  5 |  1284 |  22.7% |
| ≥ 10 |   745 |  13.2% |
| ≥ 20 |   502 |   8.9% |
| ≥ 30 |   366 |   6.5% |
| ≥ 40 |   213 |   3.8% |

## Verdict

- Tail slope **48 new essential OGs per organism** added at the end.
- Chao1 says we've seen **70%** of the true essential-OG space.
- **43% of essential OGs are essential in only one organism** -- the open-ended tail.
- **Slowing but NOT saturated.** Substantial long tail of organism-specific essentials; the core is bounded but the accessory is open.