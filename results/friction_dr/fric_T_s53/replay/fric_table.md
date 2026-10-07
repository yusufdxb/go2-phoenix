| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.29 (107%) | 75.5 (88%) | 0.61 | no |
| fmeas=1.0 | 1.06 (89%) | 59.7 (69%) | 0.49 | no |
| fmeas=1.25 | 0.99 (83%) | 54.5 (63%) | 0.45 | no |
| fmeas=1.5 | 0.93 (78%) | 46.9 (55%) | 0.41 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.29 | 0.10 | 2.7 | -1.281 | -1.308 | -1.0 | -0.9 | 1.13 |
| nominal | yaw05_3s | 0.11 | -0.01 | 75.5 | -1.281 | -1.308 | -1.3 | -0.9 | 0.96 |
| nominal | vx03_2s | 0.61 | 0.04 | 3.2 | -1.281 | -1.308 | -1.2 | -0.9 | 1.13 |
| fmeas=1.0 | vx03_4s | 1.06 | 0.04 | 2.5 | -1.283 | -1.307 | -1.1 | -0.9 | 0.96 |
| fmeas=1.0 | yaw05_3s | 0.08 | 0.00 | 59.7 | -1.283 | -1.307 | -1.2 | -0.9 | 0.81 |
| fmeas=1.0 | vx03_2s | 0.49 | 0.02 | -0.0 | -1.281 | -1.307 | -1.2 | -0.9 | 0.93 |
| fmeas=1.25 | vx03_4s | 0.99 | 0.03 | 1.4 | -1.282 | -1.307 | -1.1 | -0.9 | 0.90 |
| fmeas=1.25 | yaw05_3s | 0.07 | 0.01 | 54.5 | -1.281 | -1.307 | -1.2 | -0.9 | 0.77 |
| fmeas=1.25 | vx03_2s | 0.45 | 0.02 | -0.0 | -1.281 | -1.307 | -1.2 | -0.9 | 0.88 |
| fmeas=1.5 | vx03_4s | 0.93 | 0.02 | -0.6 | -1.282 | -1.307 | -1.1 | -0.9 | 0.85 |
| fmeas=1.5 | yaw05_3s | 0.05 | 0.01 | 46.9 | -1.274 | -1.307 | -1.3 | -0.9 | 0.70 |
| fmeas=1.5 | vx03_2s | 0.41 | 0.01 | 0.9 | -1.281 | -1.307 | -1.2 | -0.9 | 0.82 |

gate step-command forward_0p3: tracked mean 0.342 / 0.3 = 114 % fell=False
gate step-command yaw_0p6: tracked mean 0.603 / 0.6 = 100 % fell=False
