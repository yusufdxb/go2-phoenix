| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.26 (105%) | 81.8 (95%) | 0.62 | no |
| fmeas=1.0 | 1.05 (88%) | 58.4 (68%) | 0.50 | no |
| fmeas=1.25 | 0.97 (81%) | 51.9 (60%) | 0.46 | no |
| fmeas=1.5 | 0.90 (75%) | 39.2 (46%) | 0.42 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.26 | 0.07 | 0.3 | -1.284 | -1.302 | -0.9 | -1.2 | 1.08 |
| nominal | yaw05_3s | 0.02 | 0.01 | 81.8 | -1.307 | -1.302 | -1.1 | -1.2 | 1.01 |
| nominal | vx03_2s | 0.62 | 0.04 | 1.8 | -1.284 | -1.302 | -0.9 | -1.2 | 1.07 |
| fmeas=1.0 | vx03_4s | 1.05 | 0.03 | 3.1 | -1.286 | -1.302 | -0.9 | -1.2 | 0.93 |
| fmeas=1.0 | yaw05_3s | 0.03 | 0.02 | 58.4 | -1.305 | -1.302 | -1.2 | -1.2 | 0.80 |
| fmeas=1.0 | vx03_2s | 0.50 | 0.01 | -0.5 | -1.285 | -1.302 | -0.9 | -1.2 | 0.94 |
| fmeas=1.25 | vx03_4s | 0.97 | 0.02 | 0.9 | -1.286 | -1.302 | -0.9 | -1.1 | 0.87 |
| fmeas=1.25 | yaw05_3s | 0.04 | 0.02 | 51.9 | -1.303 | -1.302 | -1.2 | -1.1 | 0.71 |
| fmeas=1.25 | vx03_2s | 0.46 | 0.01 | -0.5 | -1.285 | -1.302 | -0.8 | -1.1 | 0.88 |
| fmeas=1.5 | vx03_4s | 0.90 | 0.02 | 0.4 | -1.286 | -1.301 | -0.8 | -1.1 | 0.80 |
| fmeas=1.5 | yaw05_3s | 0.03 | 0.01 | 39.2 | -1.296 | -1.301 | -1.3 | -1.1 | 0.62 |
| fmeas=1.5 | vx03_2s | 0.42 | 0.01 | 0.0 | -1.284 | -1.301 | -0.8 | -1.1 | 0.83 |

gate step-command forward_0p3: tracked mean 0.327 / 0.3 = 109 % fell=False
gate step-command yaw_0p6: tracked mean 0.619 / 0.6 = 103 % fell=False
