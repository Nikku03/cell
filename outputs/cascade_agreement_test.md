# Cascade agreement test: conservation x FBA

1,378 genes (2 organisms with FBA models), essential base rate 0.269, corr(streams)=0.395.

## Decision rules vs true essentiality

| rule | precision | recall | coverage | n_called |
|---|---:|---:|---:|---:|
| conservation@0.5 | 0.756 | 0.792 | 0.282 | 389 |
| FBA | 0.551 | 0.380 | 0.186 | 256 |
| AGREE (both) | 0.828 | 0.337 | 0.110 | 151 |
| UNION (either) | 0.628 | 0.836 | 0.358 | 494 |

## Error independence

- conservation errors: 172, FBA errors: 345
- both-wrong-on-same-gene: 87 (expected if independent: 43)
- conservation-only errors: 85, FBA-only errors: 258

## Verdict

- precision lift from agreement: **+0.072**
- agreement precision 0.828 vs best single 0.756