| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.10 (92%) | 76.8 (89%) | 0.53 | no |
| fmeas=1.0 | 0.83 (69%) | 52.6 (61%) | 0.37 | no |
| fmeas=1.25 | 0.75 (63%) | 48.1 (56%) | 0.34 | no |
| fmeas=1.5 | 0.68 (57%) | 41.1 (48%) | 0.30 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.10 | 0.04 | 3.0 | -1.313 | -1.318 | -1.4 | -1.0 | 0.97 |
| nominal | yaw05_3s | 0.07 | 0.01 | 76.8 | -1.317 | -1.318 | -1.0 | -1.0 | 0.93 |
| nominal | vx03_2s | 0.53 | 0.03 | 1.2 | -1.312 | -1.318 | -1.4 | -1.0 | 0.96 |
| fmeas=1.0 | vx03_4s | 0.83 | -0.01 | 0.3 | -1.312 | -1.319 | -1.6 | -0.9 | 0.74 |
| fmeas=1.0 | yaw05_3s | 0.05 | -0.01 | 52.6 | -1.316 | -1.319 | -0.8 | -0.9 | 0.73 |
| fmeas=1.0 | vx03_2s | 0.37 | 0.00 | 0.9 | -1.310 | -1.319 | -1.4 | -0.9 | 0.69 |
| fmeas=1.25 | vx03_4s | 0.75 | -0.01 | 0.3 | -1.312 | -1.319 | -1.5 | -0.9 | 0.67 |
| fmeas=1.25 | yaw05_3s | 0.03 | -0.00 | 48.1 | -1.314 | -1.319 | -0.7 | -0.9 | 0.68 |
| fmeas=1.25 | vx03_2s | 0.34 | -0.00 | 0.2 | -1.309 | -1.319 | -1.4 | -0.9 | 0.64 |
| fmeas=1.5 | vx03_4s | 0.68 | -0.01 | -0.1 | -1.312 | -1.320 | -1.4 | -0.9 | 0.62 |
| fmeas=1.5 | yaw05_3s | 0.03 | -0.00 | 41.1 | -1.309 | -1.320 | -0.6 | -0.9 | 0.57 |
| fmeas=1.5 | vx03_2s | 0.30 | -0.00 | -0.5 | -1.306 | -1.320 | -1.3 | -0.9 | 0.61 |

gate step-command forward_0p3: tracked mean 0.292 / 0.3 = 97 % fell=False
gate step-command yaw_0p6: tracked mean 0.575 / 0.6 = 96 % fell=False
