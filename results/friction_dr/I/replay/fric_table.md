| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.06 (89%) | 65.6 (76%) | 0.50 | no |
| fmeas=1.0 | 0.79 (66%) | 40.1 (47%) | 0.35 | no |
| fmeas=1.25 | 0.72 (60%) | 23.0 (27%) | 0.31 | no |
| fmeas=1.5 | 0.66 (55%) | 20.6 (24%) | 0.30 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.06 | 0.04 | 2.8 | -1.304 | -1.317 | -0.9 | -1.1 | 0.96 |
| nominal | yaw05_3s | 0.11 | -0.04 | 65.6 | -1.320 | -1.317 | -1.3 | -1.1 | 0.82 |
| nominal | vx03_2s | 0.50 | 0.03 | -1.0 | -1.302 | -1.317 | -1.2 | -1.1 | 0.97 |
| fmeas=1.0 | vx03_4s | 0.79 | 0.02 | 1.4 | -1.305 | -1.318 | -1.0 | -1.1 | 0.74 |
| fmeas=1.0 | yaw05_3s | 0.04 | -0.01 | 40.1 | -1.314 | -1.318 | -1.5 | -1.1 | 0.50 |
| fmeas=1.0 | vx03_2s | 0.35 | 0.02 | 1.5 | -1.300 | -1.318 | -1.2 | -1.1 | 0.75 |
| fmeas=1.25 | vx03_4s | 0.72 | 0.02 | 1.3 | -1.304 | -1.318 | -1.0 | -1.1 | 0.67 |
| fmeas=1.25 | yaw05_3s | 0.04 | -0.02 | 23.0 | -1.318 | -1.318 | -1.7 | -1.1 | 0.29 |
| fmeas=1.25 | vx03_2s | 0.31 | 0.01 | 0.2 | -1.298 | -1.318 | -1.2 | -1.1 | 0.69 |
| fmeas=1.5 | vx03_4s | 0.66 | 0.01 | 0.5 | -1.303 | -1.318 | -1.0 | -1.1 | 0.61 |
| fmeas=1.5 | yaw05_3s | 0.03 | -0.01 | 20.6 | -1.318 | -1.318 | -1.5 | -1.1 | 0.30 |
| fmeas=1.5 | vx03_2s | 0.30 | 0.00 | 0.5 | -1.299 | -1.318 | -1.1 | -1.1 | 0.64 |

gate step-command forward_0p3: tracked mean 0.286 / 0.3 = 95 % fell=False
gate step-command yaw_0p6: tracked mean 0.524 / 0.6 = 87 % fell=False
