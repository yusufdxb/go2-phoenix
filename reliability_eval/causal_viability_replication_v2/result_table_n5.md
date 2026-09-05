# Causal Viability Replication Result

positive unshielded-minus-oracle means fallback reduces post-onset falls; negative means fallback increases post-onset falls

Formal gate: **PASSED**.

| Check | Result |
|---|---|
| direction reproduces in all process cells | pass |
| leave one process out preserves direction | pass |
| pooled fault intervals exclude zero | pass |
| pre onset negative controls include zero | pass |

Registry hash: `995e28070f0c3d838d46efd499e588253eef48012d3c7caba1f9d67f74e2e02c`

## Primary outcome

| Policy | Fault | Independent disturbed blocks | Eligible pairs | Unshielded falls | Oracle falls | Unshielded rate | Oracle rate | Block-paired effect, pp | 95% block-bootstrap CI, pp | Process effects, pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Standing | Motor degradation | 160 | 2546 | 373 | 972 | 14.65% | 38.18% | -23.53 | [-25.43, -21.66] | -19.64, -27.87, -23.70, -22.50, -23.97 |
| Standing | Observation corruption | 160 | 2546 | 265 | 13 | 10.41% | 0.51% | +9.89 | [+8.67, +11.18] | +9.04, +9.62, +10.18, +9.49, +11.13 |
| Walking | Motor degradation | 160 | 2430 | 1282 | 1441 | 52.76% | 59.30% | -6.60 | [-8.12, -5.11] | -5.49, -6.43, -5.92, -8.65, -6.52 |
| Walking | Observation corruption | 160 | 2443 | 487 | 68 | 19.93% | 2.78% | +17.21 | [+15.60, +18.87] | +16.50, +18.39, +17.68, +16.67, +16.79 |

## Leave-one-process-out effects

| Cell | Leave out process 01, pp | Leave out process 02, pp | Leave out process 03, pp | Leave out process 04, pp | Leave out process 05, pp |
|---|---|---|---|---|---|
| Standing, motor | -24.51 [-26.55, -22.47] | -22.45 [-24.65, -20.32] | -23.49 [-25.68, -21.38] | -23.79 [-25.96, -21.65] | -23.43 [-25.63, -21.25] |
| Standing, observation | +10.11 [+8.71, +11.57] | +9.96 [+8.57, +11.40] | +9.82 [+8.39, +11.28] | +9.99 [+8.67, +11.37] | +9.58 [+8.14, +11.05] |
| Walking, motor | -6.88 [-8.63, -5.18] | -6.64 [-8.45, -4.89] | -6.77 [-8.45, -5.15] | -6.09 [-7.74, -4.44] | -6.62 [-8.35, -4.93] |
| Walking, observation | +17.38 [+15.44, +19.37] | +16.91 [+15.27, +18.62] | +17.09 [+15.41, +18.85] | +17.34 [+15.55, +19.27] | +17.31 [+15.44, +19.25] |

## Fault-family interaction

| Quantity | Independent blocks | Effect, pp | 95% block-bootstrap CI, pp |
|---|---:|---:|---:|
| Motor degradation | 320 | -15.07 | [-16.28, -13.86] |
| Observation corruption | 320 | +13.55 | [+12.53, +14.60] |
| Observation-minus-motor interaction | 640 | +28.62 | [+27.04, +30.23] |

## Secondary outcomes

| Cell | Task completion U/O | Return until first fall U/O | Oracle dose | Treated falls | Latency ticks min/median/max | Nominal false handoffs |
|---|---|---|---:|---:|---|---:|
| Standing, motor | 85.35% / 61.82% | 1.833 / 1.210 | 0.504 | 972 | 0 / 0 / 0 | 0 |
| Standing, observation | 89.59% / 99.49% | -1689.165 / -35.259 | 0.692 | 13 | 0 / 0 / 0 | 0 |
| Walking, motor | 47.24% / 40.70% | 1.499 / 1.469 | 0.359 | 1441 | 0 / 0 / 0 | 0 |
| Walking, observation | 80.07% / 97.22% | -3.825 / 2.001 | 0.675 | 68 | 0 / 0 / 0 | 0 |

## Pre-onset negative control

| Cell | Pre-onset U-minus-O effect, pp | 95% block-bootstrap CI, pp |
|---|---:|---:|
| Standing, motor | +0.00 | [+0.00, +0.00] |
| Standing, observation | +0.00 | [+0.00, +0.00] |
| Walking, motor | +0.04 | [+0.00, +0.12] |
| Walking, observation | +0.04 | [+0.00, +0.12] |

The motor-family pre-onset effect was +0.02 pp [+0.00, +0.06]. The observation-family pre-onset effect was +0.02 pp [+0.00, +0.06].
