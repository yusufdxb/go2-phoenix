| condition | vx03_4s dist m (% of 1.2) | yaw05_3s yaw deg (% of 86) | vx03_2s dist m | fell |
|---|---|---|---|---|
| nominal | 1.25 (104%) | 85.0 (99%) | 0.59 | no |
| fmeas=1.0 | 1.04 (87%) | 69.6 (81%) | 0.49 | no |
| fmeas=1.25 | 0.97 (81%) | 63.6 (74%) | 0.45 | no |
| fmeas=1.5 | 0.91 (76%) | 57.0 (66%) | 0.42 | no |

Nominal-style detail (displacement at profile end, body-start heading; calf = mean of the 4 calf target-q, rad; pitch + = nose down, deg):

| condition | preset | fwd m | lat m | yaw deg | calf tgt moving | calf tgt standing | pitch moving | pitch standing | steady vx/yaw frac |
|---|---|---|---|---|---|---|---|---|---|
| nominal | vx03_4s | 1.25 | 0.08 | 6.8 | -1.274 | -1.307 | -0.9 | -1.0 | 1.09 |
| nominal | yaw05_3s | 0.11 | -0.04 | 85.0 | -1.306 | -1.307 | -1.0 | -1.0 | 1.05 |
| nominal | vx03_2s | 0.59 | 0.02 | 2.7 | -1.275 | -1.307 | -1.0 | -1.0 | 1.05 |
| fmeas=1.0 | vx03_4s | 1.04 | 0.08 | 5.8 | -1.276 | -1.306 | -1.1 | -1.0 | 0.92 |
| fmeas=1.0 | yaw05_3s | 0.12 | -0.00 | 69.6 | -1.309 | -1.306 | -0.9 | -1.0 | 0.88 |
| fmeas=1.0 | vx03_2s | 0.49 | 0.03 | 3.0 | -1.275 | -1.306 | -1.2 | -1.0 | 0.92 |
| fmeas=1.25 | vx03_4s | 0.97 | 0.07 | 4.9 | -1.275 | -1.306 | -1.1 | -1.0 | 0.87 |
| fmeas=1.25 | yaw05_3s | 0.11 | 0.00 | 63.6 | -1.307 | -1.306 | -1.0 | -1.0 | 0.83 |
| fmeas=1.25 | vx03_2s | 0.45 | 0.03 | 3.0 | -1.274 | -1.306 | -1.1 | -1.0 | 0.87 |
| fmeas=1.5 | vx03_4s | 0.91 | 0.06 | 4.1 | -1.276 | -1.306 | -1.0 | -1.0 | 0.81 |
| fmeas=1.5 | yaw05_3s | 0.10 | 0.00 | 57.0 | -1.304 | -1.306 | -1.0 | -1.0 | 0.78 |
| fmeas=1.5 | vx03_2s | 0.42 | 0.03 | 3.2 | -1.274 | -1.306 | -1.1 | -1.0 | 0.80 |

gate step-command forward_0p3: tracked mean 0.330 / 0.3 = 110 % fell=False
gate step-command yaw_0p6: tracked mean 0.647 / 0.6 = 108 % fell=False
