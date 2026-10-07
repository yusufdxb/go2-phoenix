# GO2 joint static friction, measured 2026-10-07 (FP probe, robot hung)

Method: `scripts/harness_preflight.sh FP` (bridge PD kp 25 / kd 0.5, one joint ramped at
0.05 rad/s each way until breakaway). Static friction Fs = (tau+ - tau-)/2 at breakaway;
kinetic from the slow-motion segment. Estimator validated in MuJoCo (set friction 0-1.5 N m
recovered within 0.03 N m). Unloaded measurement: legs hanging, feet off the ground.

| joint | Fs (N m) | kinetic (N m) | pose | run |
|---|---|---|---|---|
| FR_hip   | 0.353 | 0.410 | hung, folded | run2 |
| FR_thigh | 0.249 | 0.220 | hung, folded | run3 |
| FL_hip   | 0.338 | 0.336 | hung, folded | run4 |
| FL_thigh | 0.171 | 0.214 | hung, folded | run4 |
| RR_hip   | 0.261 | 0.364 | hung, folded | run4 |
| RR_thigh | 0.339 | 0.298 | hung, folded | run4 |
| RL_thigh | 0.331 | 0.263 | hung, folded | run4 |
| FR_calf  | 0.422 | 0.445 | hung, legs at stance (stand-up first) | run6 |
| FL_calf  | 0.622 | 0.776 | hung, legs at stance | run6 |
| RR_calf  | 0.962 | 0.980 | hung, legs at stance | run6 |
| RL_calf  | 0.607 | 0.707 | hung, legs at stance | run6 |
| RL_hip   | 0.241 | 0.195 | hung, legs at stance | run6 |

Excluded: run4 RL_hip (leg started splayed at -0.84 rad, one-sided); run3 FR_calf (folded, at
the joint limit); run5 (front feet touched the floor: front calves read 1.9-2.3 N m; rear
calves, which did not touch, matched run6 to 0.01 N m).

Sim replay (MuJoCo, same policy and command profiles) with these values: forward 66 % and
yaw 34-47 % of command vs 89 % / 76 % frictionless and 60 % / 20 % measured on the robot;
1.25-1.5x these values matches the robot (load-dependent friction is the leading, untested
explanation). Standing calf sag is not explained by joint friction in sim.
