# Causal Viability Replication Result

positive unshielded-minus-oracle means fallback reduces post-onset falls; negative means fallback increases post-onset falls

Formal gate: **PASSED**.

| Check | Result |
|---|---|
| direction reproduces in all process cells | pass |
| leave one process out preserves direction | pass |
| pooled fault intervals exclude zero | pass |
| pre onset negative controls include zero | pass |

Registry hash: `3f4af8c891aa16a5b9965b242d5dd664f72b6f28b38a05fda8fcbc854c80014d`

## Primary outcome

| Policy | Fault | Independent disturbed blocks | Eligible pairs | Unshielded falls | Oracle falls | Unshielded rate | Oracle rate | Block-paired effect, pp | 95% block-bootstrap CI, pp | Process effects, pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Standing | Motor degradation | 96 | 1528 | 188 | 551 | 12.30% | 36.06% | -23.73 | [-26.32, -21.20] | -19.64, -27.87, -23.70 |
| Standing | Observation corruption | 96 | 1528 | 154 | 7 | 10.08% | 0.46% | +9.61 | [+8.09, +11.23] | +9.04, +9.62, +10.18 |
| Walking | Motor degradation | 96 | 1456 | 711 | 797 | 48.83% | 54.74% | -5.95 | [-7.87, -4.03] | -5.49, -6.43, -5.92 |
| Walking | Observation corruption | 96 | 1460 | 301 | 46 | 20.62% | 3.15% | +17.52 | [+15.31, +19.88] | +16.50, +18.39, +17.68 |

## Leave-one-process-out effects

| Cell | Leave out process 1, pp | Leave out process 2, pp | Leave out process 3, pp |
|---|---:|---:|---:|
| Standing, motor | -25.78 [-28.73, -22.86] | -21.67 [-24.90, -18.42] | -23.75 [-26.99, -20.59] |
| Standing, observation | +9.90 [+8.03, +11.86] | +9.61 [+7.73, +11.56] | +9.33 [+7.46, +11.31] |
| Walking, motor | -6.18 [-8.48, -3.86] | -5.71 [-8.21, -3.20] | -5.96 [-8.14, -3.81] |
| Walking, observation | +18.03 [+14.96, +21.33] | +17.09 [+14.69, +19.59] | +17.45 [+14.94, +20.12] |

## Fault-family interaction

| Quantity | Independent blocks | Effect, pp | 95% block-bootstrap CI, pp |
|---|---:|---:|---:|
| Motor degradation | 192 | -14.84 | [-16.47, -13.26] |
| Observation corruption | 192 | +13.57 | [+12.21, +14.97] |
| Observation-minus-motor interaction | 384 | +28.41 | [+26.30, +30.51] |

## Secondary outcomes

| Cell | Task completion U/O | Return until first fall U/O | Oracle dose | Treated falls | Latency ticks min/median/max | Nominal false handoffs |
|---|---|---|---:|---:|---|---:|
| Standing, motor | 87.70% / 63.94% | 1.932 / 1.318 | 0.510 | 551 | 0 / 0 / 0 | 0 |
| Standing, observation | 89.92% / 99.54% | -1674.716 / -35.438 | 0.685 | 7 | 0 / 0 / 0 | 0 |
| Walking, motor | 51.17% / 45.26% | 1.607 / 1.549 | 0.391 | 797 | 0 / 0 / 0 | 0 |
| Walking, observation | 79.38% / 96.85% | -3.832 / 1.937 | 0.670 | 46 | 0 / 0 / 0 | 0 |

## Pre-onset negative control

| Cell | Pre-onset U-minus-O effect, pp | 95% block-bootstrap CI, pp |
|---|---:|---:|
| Standing, motor | +0.00 | [+0.00, +0.00] |
| Standing, observation | +0.00 | [+0.00, +0.00] |
| Walking, motor | +0.07 | [+0.00, +0.20] |
| Walking, observation | +0.07 | [+0.00, +0.20] |

The motor-family pre-onset effect was +0.03 pp [+0.00, +0.10]. The observation-family pre-onset effect was +0.03 pp [+0.00, +0.10].
