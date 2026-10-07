| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.05 (88%) | 71.7 (83%) | 0.51 | no |
| fmeas=1.0 | 0.77 (64%) | 47.4 (55%) | 0.37 | no |
| fmeas=1.25 | 0.70 (58%) | 39.0 (45%) | 0.33 | no |
| fmeas=1.5 | 0.62 (52%) | 33.8 (39%) | 0.29 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.05 | 0.03 | 1.8 | -1.301 | -1.313 | -1.5 | -1.0 | 0.92 |
| nominal | yaw05_3s | 0.09 | -0.04 | 71.7 | -1.311 | -1.313 | -1.6 | -1.0 | 0.86 |
| nominal | vx03_2s | 0.51 | 0.02 | -0.2 | -1.300 | -1.313 | -1.6 | -1.0 | 0.92 |
| fmeas=1.0 | vx03_4s | 0.77 | -0.02 | -0.7 | -1.301 | -1.314 | -1.8 | -1.0 | 0.68 |
| fmeas=1.0 | yaw05_3s | 0.05 | -0.01 | 47.4 | -1.308 | -1.314 | -1.5 | -1.0 | 0.62 |
| fmeas=1.0 | vx03_2s | 0.37 | 0.00 | 0.2 | -1.298 | -1.314 | -1.5 | -1.0 | 0.70 |
| fmeas=1.25 | vx03_4s | 0.70 | -0.02 | -0.5 | -1.301 | -1.314 | -1.7 | -1.0 | 0.62 |
| fmeas=1.25 | yaw05_3s | 0.04 | -0.01 | 39.0 | -1.301 | -1.314 | -1.4 | -1.0 | 0.56 |
| fmeas=1.25 | vx03_2s | 0.33 | 0.00 | -0.0 | -1.297 | -1.314 | -1.4 | -1.0 | 0.65 |
| fmeas=1.5 | vx03_4s | 0.62 | -0.02 | -1.4 | -1.299 | -1.314 | -1.6 | -1.0 | 0.57 |
| fmeas=1.5 | yaw05_3s | 0.03 | -0.02 | 33.8 | -1.304 | -1.314 | -1.4 | -1.0 | 0.45 |
| fmeas=1.5 | vx03_2s | 0.29 | -0.00 | -0.7 | -1.295 | -1.314 | -1.5 | -1.0 | 0.61 |

gate step-command forward_0p3: tracked mean 0.275 / 0.3 = 92 % fell=False
gate step-command yaw_0p6: tracked mean 0.538 / 0.6 = 90 % fell=False
