| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.13 (94%) | 67.1 (78%) | 0.54 | no |
| fmeas=1.0 | 0.83 (69%) | 50.3 (59%) | 0.39 | no |
| fmeas=1.25 | 0.76 (63%) | 41.5 (48%) | 0.36 | no |
| fmeas=1.5 | 0.68 (57%) | 34.4 (40%) | 0.32 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.13 | 0.06 | 0.2 | -1.296 | -1.311 | -0.5 | -1.2 | 0.98 |
| nominal | yaw05_3s | 0.14 | -0.02 | 67.1 | -1.308 | -1.311 | -0.4 | -1.2 | 0.86 |
| nominal | vx03_2s | 0.54 | 0.03 | -1.1 | -1.296 | -1.311 | -0.7 | -1.2 | 0.99 |
| fmeas=1.0 | vx03_4s | 0.83 | 0.03 | 1.4 | -1.296 | -1.312 | -0.8 | -1.1 | 0.73 |
| fmeas=1.0 | yaw05_3s | 0.08 | -0.02 | 50.3 | -1.308 | -1.312 | -0.4 | -1.1 | 0.69 |
| fmeas=1.0 | vx03_2s | 0.39 | 0.02 | 1.7 | -1.295 | -1.312 | -0.9 | -1.1 | 0.77 |
| fmeas=1.25 | vx03_4s | 0.76 | 0.03 | 1.8 | -1.297 | -1.312 | -0.8 | -1.1 | 0.67 |
| fmeas=1.25 | yaw05_3s | 0.08 | -0.01 | 41.5 | -1.304 | -1.312 | -0.5 | -1.1 | 0.60 |
| fmeas=1.25 | vx03_2s | 0.36 | 0.02 | 1.8 | -1.295 | -1.312 | -0.8 | -1.1 | 0.71 |
| fmeas=1.5 | vx03_4s | 0.68 | 0.02 | 1.5 | -1.297 | -1.312 | -0.8 | -1.1 | 0.61 |
| fmeas=1.5 | yaw05_3s | 0.05 | -0.01 | 34.4 | -1.302 | -1.312 | -0.5 | -1.1 | 0.53 |
| fmeas=1.5 | vx03_2s | 0.32 | 0.01 | 1.8 | -1.295 | -1.312 | -0.8 | -1.1 | 0.65 |

gate step-command forward_0p3: tracked mean 0.292 / 0.3 = 97 % fell=False
gate step-command yaw_0p6: tracked mean 0.541 / 0.6 = 90 % fell=False
