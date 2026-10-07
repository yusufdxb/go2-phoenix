| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.04 (87%) | 58.9 (69%) | 0.49 | no |
| fmeas=1.0 | 0.71 (59%) | 35.3 (41%) | 0.35 | no |
| fmeas=1.25 | 0.66 (55%) | 17.2 (20%) | 0.31 | no |
| fmeas=1.5 | 0.58 (49%) | 16.0 (19%) | 0.27 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.04 | 0.02 | -0.7 | -1.318 | -1.320 | -1.5 | -1.1 | 0.91 |
| nominal | yaw05_3s | 0.07 | -0.04 | 58.9 | -1.328 | -1.320 | -1.8 | -1.1 | 0.81 |
| nominal | vx03_2s | 0.49 | 0.02 | -3.4 | -1.317 | -1.320 | -1.5 | -1.1 | 0.90 |
| fmeas=1.0 | vx03_4s | 0.71 | -0.01 | -3.9 | -1.310 | -1.321 | -1.9 | -1.1 | 0.64 |
| fmeas=1.0 | yaw05_3s | 0.03 | -0.01 | 35.3 | -1.322 | -1.321 | -1.6 | -1.1 | 0.36 |
| fmeas=1.0 | vx03_2s | 0.35 | 0.01 | -0.8 | -1.314 | -1.321 | -1.3 | -1.1 | 0.70 |
| fmeas=1.25 | vx03_4s | 0.66 | 0.01 | -2.2 | -1.314 | -1.321 | -1.7 | -1.1 | 0.58 |
| fmeas=1.25 | yaw05_3s | 0.03 | -0.01 | 17.2 | -1.321 | -1.321 | -1.5 | -1.1 | 0.21 |
| fmeas=1.25 | vx03_2s | 0.31 | 0.01 | -0.4 | -1.313 | -1.321 | -1.2 | -1.1 | 0.64 |
| fmeas=1.5 | vx03_4s | 0.58 | 0.01 | -0.7 | -1.315 | -1.321 | -1.5 | -1.1 | 0.52 |
| fmeas=1.5 | yaw05_3s | 0.02 | -0.01 | 16.0 | -1.323 | -1.321 | -1.3 | -1.1 | 0.22 |
| fmeas=1.5 | vx03_2s | 0.27 | 0.01 | -0.4 | -1.313 | -1.321 | -1.1 | -1.1 | 0.58 |

gate step-command forward_0p3: tracked mean 0.273 / 0.3 = 91 % fell=False
gate step-command yaw_0p6: tracked mean 0.524 / 0.6 = 87 % fell=False
