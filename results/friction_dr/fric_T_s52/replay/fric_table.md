| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.27 (106%) | 83.2 (97%) | 0.61 | no |
| fmeas=1.0 | 0.99 (83%) | 60.9 (71%) | 0.48 | no |
| fmeas=1.25 | 0.92 (77%) | 55.5 (65%) | 0.44 | no |
| fmeas=1.5 | 0.86 (72%) | 52.1 (61%) | 0.40 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.27 | 0.08 | 4.7 | -1.273 | -1.304 | -1.3 | -0.9 | 1.12 |
| nominal | yaw05_3s | 0.11 | -0.02 | 83.2 | -1.288 | -1.304 | -1.1 | -0.9 | 1.01 |
| nominal | vx03_2s | 0.61 | 0.03 | 3.7 | -1.275 | -1.304 | -1.4 | -0.9 | 1.12 |
| fmeas=1.0 | vx03_4s | 0.99 | 0.02 | 3.4 | -1.273 | -1.303 | -1.6 | -0.9 | 0.89 |
| fmeas=1.0 | yaw05_3s | 0.11 | -0.01 | 60.9 | -1.283 | -1.303 | -1.1 | -0.9 | 0.83 |
| fmeas=1.0 | vx03_2s | 0.48 | 0.01 | 1.0 | -1.275 | -1.303 | -1.4 | -0.9 | 0.93 |
| fmeas=1.25 | vx03_4s | 0.92 | 0.01 | 1.2 | -1.274 | -1.303 | -1.6 | -0.9 | 0.83 |
| fmeas=1.25 | yaw05_3s | 0.08 | -0.01 | 55.5 | -1.283 | -1.303 | -1.1 | -0.9 | 0.79 |
| fmeas=1.25 | vx03_2s | 0.44 | 0.01 | 0.8 | -1.274 | -1.303 | -1.3 | -0.9 | 0.88 |
| fmeas=1.5 | vx03_4s | 0.86 | 0.00 | -0.0 | -1.275 | -1.303 | -1.5 | -0.9 | 0.77 |
| fmeas=1.5 | yaw05_3s | 0.07 | -0.00 | 52.1 | -1.284 | -1.303 | -1.0 | -0.9 | 0.74 |
| fmeas=1.5 | vx03_2s | 0.40 | 0.01 | 1.8 | -1.273 | -1.303 | -1.2 | -0.9 | 0.83 |

gate step-command forward_0p3: tracked mean 0.336 / 0.3 = 112 % fell=False
gate step-command yaw_0p6: tracked mean 0.615 / 0.6 = 103 % fell=False
