| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.27 (106%) | 72.7 (85%) | 0.61 | no |
| fmeas=1.0 | 1.05 (87%) | 59.5 (69%) | 0.51 | no |
| fmeas=1.25 | 0.97 (81%) | 54.8 (64%) | 0.47 | no |
| fmeas=1.5 | 0.92 (76%) | 49.9 (58%) | 0.43 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.27 | 0.09 | 1.3 | -1.275 | -1.308 | -1.6 | -1.3 | 1.11 |
| nominal | yaw05_3s | 0.10 | -0.01 | 72.7 | -1.294 | -1.308 | -1.1 | -1.3 | 0.93 |
| nominal | vx03_2s | 0.61 | 0.05 | 1.6 | -1.275 | -1.308 | -1.8 | -1.3 | 1.10 |
| fmeas=1.0 | vx03_4s | 1.05 | 0.07 | 3.4 | -1.278 | -1.308 | -1.9 | -1.3 | 0.93 |
| fmeas=1.0 | yaw05_3s | 0.06 | -0.00 | 59.5 | -1.294 | -1.308 | -1.1 | -1.3 | 0.77 |
| fmeas=1.0 | vx03_2s | 0.51 | 0.03 | -0.3 | -1.276 | -1.308 | -1.8 | -1.3 | 0.96 |
| fmeas=1.25 | vx03_4s | 0.97 | 0.06 | 2.7 | -1.277 | -1.308 | -1.9 | -1.3 | 0.87 |
| fmeas=1.25 | yaw05_3s | 0.05 | -0.00 | 54.8 | -1.292 | -1.308 | -1.1 | -1.3 | 0.73 |
| fmeas=1.25 | vx03_2s | 0.47 | 0.03 | -0.2 | -1.276 | -1.308 | -1.8 | -1.3 | 0.90 |
| fmeas=1.5 | vx03_4s | 0.92 | 0.05 | 2.3 | -1.278 | -1.308 | -1.9 | -1.3 | 0.80 |
| fmeas=1.5 | yaw05_3s | 0.04 | -0.00 | 49.9 | -1.287 | -1.308 | -1.2 | -1.3 | 0.68 |
| fmeas=1.5 | vx03_2s | 0.43 | 0.03 | 0.7 | -1.275 | -1.308 | -1.8 | -1.3 | 0.85 |

gate step-command forward_0p3: tracked mean 0.331 / 0.3 = 110 % fell=False
gate step-command yaw_0p6: tracked mean 0.584 / 0.6 = 97 % fell=False
