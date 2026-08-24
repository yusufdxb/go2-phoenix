# Causal Viability Replication Result

Sign convention: positive unshielded-minus-oracle means the fallback reduces post-onset falls. Negative means the fallback increases post-onset falls.

Formal gate: **FAILED**. The pooled pre-onset negative-control intervals did not all include zero at the policy-fault cell level.

## Primary outcome

| Policy | Fault | Independent disturbed blocks | Eligible pairs | Unshielded falls | Oracle falls | Unshielded rate | Oracle rate | Block-paired effect, pp | 95% block-bootstrap CI, pp | Process effects, pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Standing | Motor degradation | 96 | 1509 | 172 | 389 | 11.40% | 25.78% | -14.47 | [-17.00, -11.98] | -10.48, -14.12, -18.82 |
| Standing | Observation corruption | 96 | 1505 | 100 | 6 | 6.64% | 0.40% | +6.21 | [+4.91, +7.56] | +5.31, +5.74, +7.57 |
| Walking | Motor degradation | 96 | 1337 | 666 | 751 | 49.81% | 56.17% | -6.24 | [-9.26, -3.28] | -1.78, -9.00, -7.94 |
| Walking | Observation corruption | 96 | 1388 | 219 | 58 | 15.78% | 4.18% | +11.57 | [+9.40, +13.76] | +14.37, +10.10, +10.24 |

The primary counts above use the jointly onset-eligible environment pairs. Across all raw trials before eligibility filtering, disturbed fall counts were 183/411, 130/10, 824/909, and 319/169 for unshielded/oracle in the four table rows. Nominal fall counts were 2/3, 6/6, 83/77, and 70/79 out of 768 trials per arm and cell.

## Leave-one-process-out effects

| Cell | Leave out process 1, pp | Leave out process 2, pp | Leave out process 3, pp |
|---|---:|---:|---:|
| Standing, motor | -16.47 [-19.82, -13.21] | -14.65 [-17.86, -11.50] | -12.30 [-14.98, -9.66] |
| Standing, observation | +6.66 [+4.98, +8.38] | +6.44 [+4.96, +7.95] | +5.53 [+3.86, +7.28] |
| Walking, motor | -8.47 [-11.80, -5.12] | -4.86 [-8.72, -1.00] | -5.39 [-9.00, -1.81] |
| Walking, observation | +10.17 [+7.99, +12.27] | +12.30 [+9.57, +15.19] | +12.23 [+9.37, +15.26] |

## Fault-family interaction

| Quantity | Independent blocks | Effect, pp | 95% block-bootstrap CI, pp |
|---|---:|---:|---:|
| Motor degradation | 192 | -10.36 | [-12.33, -8.44] |
| Observation corruption | 192 | +8.89 | [+7.62, +10.16] |
| Observation-minus-motor interaction | 384 | +19.24 | [+16.94, +21.59] |

## Secondary outcomes

| Cell | Task completion U/O | Return until first fall U/O | Oracle dose | Treated falls | Latency ticks min/median/max | Nominal false handoffs |
|---|---|---|---:|---:|---|---:|
| Standing, motor | 88.60% / 74.22% | 3.280 / 4.036 | 0.563 | 389 | 0 / 0 / 0 | 0 |
| Standing, observation | 93.36% / 99.60% | -1698.601 / -34.536 | 0.686 | 6 | 0 / 0 / 0 | 0 |
| Walking, motor | 50.19% / 43.83% | 1.331 / 1.338 | 0.390 | 751 | 0 / 0 / 0 | 0 |
| Walking, observation | 84.22% / 95.82% | -4.233 / 1.606 | 0.670 | 58 | 0 / 0 / 0 | 0 |

There were zero missed engagements among onset-eligible disturbed trials. Treated falls are failures that occurred after correct oracle engagement, not trigger misses.

## Failed negative-control check

| Cell | Pre-onset U-minus-O effect, pp | 95% block-bootstrap CI, pp |
|---|---:|---:|
| Standing, motor | -0.65 | [-1.30, -0.07] |
| Standing, observation | +1.69 | [+1.11, +2.34] |
| Walking, motor | +0.39 | [-1.37, +2.15] |
| Walking, observation | -1.76 | [-3.19, -0.33] |

The motor-family pre-onset effect was -0.13 pp [-1.07, +0.81]. The observation-family pre-onset effect was -0.03 pp [-0.78, +0.75]. Family-level balance does not override the frozen cell-level fail-closed criterion.
