# Arm comparison, vx03_4s (results/20261020_228555e8443b_abc12_2026-10-18/analysis/runs.json)

Tape = measured. Leg odometry = estimate (no-slip kinematics), never used in place of tape.

| block | arm | SoC % | entered by | cmd m | tape m | F | fwd err m | lat m | yaw drift deg | odom m | F_odom | odom/tape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 55 | driver_relaying_operator | 1.200 | 0.813 | 0.677 | 0.387 |  | 1.6 | 0.694 | 0.578 | 0.854 |
| 2 | B | 54 | driver_relaying_operator | 1.200 | 1.194 | 0.995 | 0.006 | 0.051 | 1.7 | 1.026 | 0.855 | 0.859 |
| 3 | C | 52 | driver_relaying_operator | 1.200 | 1.245 | 1.037 | -0.045 |  | 0.6 | 1.080 | 0.900 | 0.868 |
| 4 | A | 51 | driver_relaying_operator | 1.200 | 0.807 | 0.672 | 0.394 |  | 1.4 | 0.696 | 0.580 | 0.863 |
| 5 | C | 49 | driver_relaying_operator | 1.200 | 1.232 | 1.027 | -0.032 |  | 0.3 | 1.083 | 0.903 | 0.879 |
| 6 | B | 48 | driver_relaying_operator | 1.199 | 1.200 | 1.001 | -0.001 |  | 1.0 | 1.026 | 0.856 | 0.855 |
| 7 | C | 47 | driver_relaying_operator | 1.200 | 1.238 | 1.032 | -0.038 |  | -0.3 | 1.081 | 0.901 | 0.873 |
| 8 | B | 45 | driver_relaying_operator | 1.200 | 1.194 | 0.995 | 0.006 |  | 1.2 | 1.032 | 0.860 | 0.864 |
| 9 | A | 44 | driver_relaying_operator | 1.200 | 0.813 | 0.677 | 0.387 |  | 1.5 | 0.692 | 0.577 | 0.852 |
| 10 | B | 43 | driver_relaying_operator | 1.200 | 1.181 | 0.984 | 0.019 |  | 1.9 | 1.017 | 0.847 | 0.861 |
| 11 | C | 42 | driver_relaying_operator | 1.200 | 1.238 | 1.032 | -0.038 |  | 0.1 | 1.078 | 0.899 | 0.871 |
| 12 | A | 40 | driver_relaying_operator | 1.200 | 0.819 | 0.683 | 0.381 |  | 1.4 | 0.691 | 0.576 | 0.843 |

## Per arm (mean, std, n)

| arm | metric | mean | std | n |
|---|---|---|---|---|
| A | F | 0.677 | 0.004 | 4 |
| A | fwd_err_m | 0.387 | 0.005 | 4 |
| A | yaw_drift_deg | 1.483 | 0.130 | 4 |
| A | F_odom | 0.578 | 0.002 | 4 |
| A | odom_leg_lat_m | 0.013 | 0.003 | 4 |
| A | odom_leg_speed_rmse_mps | 0.122 | 0.001 | 4 |
| A | odom_over_tape | 0.853 | 0.008 | 4 |
| B | F | 0.994 | 0.007 | 4 |
| B | fwd_err_m | 0.007 | 0.008 | 4 |
| B | lat_tape_m | 0.051 |  | 1 |
| B | yaw_drift_deg | 1.416 | 0.437 | 4 |
| B | F_odom | 0.855 | 0.005 | 4 |
| B | odom_leg_lat_m | 0.069 | 0.004 | 4 |
| B | odom_leg_speed_rmse_mps | 0.081 | 0.002 | 4 |
| B | odom_over_tape | 0.860 | 0.004 | 4 |
| C | F | 1.032 | 0.004 | 4 |
| C | fwd_err_m | -0.038 | 0.005 | 4 |
| C | yaw_drift_deg | 0.196 | 0.361 | 4 |
| C | F_odom | 0.901 | 0.002 | 4 |
| C | odom_leg_lat_m | 0.072 | 0.004 | 4 |
| C | odom_leg_speed_rmse_mps | 0.083 | 0.001 | 4 |
| C | odom_over_tape | 0.873 | 0.005 | 4 |

## Versus A (difference of means, exact permutation p)

| arm | metric | diff | p | n (A, arm) |
|---|---|---|---|---|
| B | F | 0.316 | 0.029 | [4, 4] |
| B | fwd_err_m | -0.380 | 0.029 | [4, 4] |
| B | lat_tape_m |  |  | [0, 1] |
| B | yaw_drift_deg | -0.067 | 0.800 | [4, 4] |
| B | F_odom | 0.277 | 0.029 | [4, 4] |
| C | F | 0.355 | 0.029 | [4, 4] |
| C | fwd_err_m | -0.425 | 0.029 | [4, 4] |
| C | lat_tape_m |  |  | [0, 0] |
| C | yaw_drift_deg | -1.288 | 0.029 | [4, 4] |
| C | F_odom | 0.323 | 0.029 | [4, 4] |
