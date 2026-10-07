| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.03 (86%) | 64.3 (75%) | 0.47 | no |
| fmeas=1.0 | 0.79 (66%) | 40.5 (47%) | 0.34 | no |
| fmeas=1.25 | 0.71 (59%) | 37.7 (44%) | 0.30 | no |
| fmeas=1.5 | 0.63 (52%) | 22.1 (26%) | 0.26 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.03 | 0.03 | -3.5 | -1.314 | -1.317 | -0.6 | -1.1 | 0.94 |
| nominal | yaw05_3s | 0.08 | -0.03 | 64.3 | -1.313 | -1.317 | -0.8 | -1.1 | 0.84 |
| nominal | vx03_2s | 0.47 | 0.02 | -3.7 | -1.311 | -1.317 | -0.8 | -1.1 | 0.95 |
| fmeas=1.0 | vx03_4s | 0.79 | 0.02 | -0.9 | -1.314 | -1.318 | -0.8 | -1.1 | 0.74 |
| fmeas=1.0 | yaw05_3s | 0.04 | -0.02 | 40.5 | -1.311 | -1.318 | -0.9 | -1.1 | 0.58 |
| fmeas=1.0 | vx03_2s | 0.34 | 0.02 | 0.4 | -1.310 | -1.318 | -1.0 | -1.1 | 0.76 |
| fmeas=1.25 | vx03_4s | 0.71 | 0.03 | 0.0 | -1.313 | -1.318 | -0.8 | -1.1 | 0.68 |
| fmeas=1.25 | yaw05_3s | 0.03 | -0.02 | 37.7 | -1.314 | -1.318 | -0.8 | -1.1 | 0.45 |
| fmeas=1.25 | vx03_2s | 0.30 | 0.02 | 0.6 | -1.309 | -1.318 | -1.0 | -1.1 | 0.71 |
| fmeas=1.5 | vx03_4s | 0.63 | 0.03 | 0.9 | -1.314 | -1.318 | -0.7 | -1.1 | 0.62 |
| fmeas=1.5 | yaw05_3s | 0.03 | -0.02 | 22.1 | -1.311 | -1.318 | -1.0 | -1.1 | 0.30 |
| fmeas=1.5 | vx03_2s | 0.26 | 0.02 | 0.1 | -1.308 | -1.318 | -0.9 | -1.1 | 0.66 |

gate step-command forward_0p3: tracked mean 0.280 / 0.3 = 93 % fell=False
gate step-command yaw_0p6: tracked mean 0.526 / 0.6 = 88 % fell=False
