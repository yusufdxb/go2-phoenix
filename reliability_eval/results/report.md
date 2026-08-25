# Reliability Shield, Phase 3 (Isaac twin) Results

dt=0.02s, warmup=15 frames dropped/episode, operating point = 5% nominal FPR.

## Behavioral fall rate (env base-contact oracle)

| condition | fall rate (env-ever-fell) |
|---|---|
| nominal | 0.154 [0.123, 0.185] |
| friction_moderate | 0.307 [0.264, 0.351] |
| friction_severe | 0.521 [0.507, 0.534] |
| mass_moderate | 0.047 [0.023, 0.070] |
| mass_severe | 0.026 [0.006, 0.046] |
| motor_moderate | 0.003 [-0.003, 0.008] |
| motor_severe | 0.232 [0.160, 0.304] |

Unnecessary-fallback rate on nominal (latent shield): **1.000 [1.000, 1.000]** per episode.

## Detection: latent-Mahalanobis vs baselines (AUROC, mean [95% CI] over seeds)

| condition | latent_maha | obs_maha | obs_magnitude | value_signal | action_sat | random |
|---|---|---|---|---|---|---|
| friction_moderate | 0.957 [0.950, 0.965] | 0.876 [0.869, 0.882] | 0.535 [0.524, 0.545] | 0.287 [0.255, 0.319] | 0.511 [0.487, 0.535] | 0.500 [0.498, 0.501] |
| friction_severe | 0.980 [0.974, 0.986] | 0.929 [0.925, 0.934] | 0.592 [0.569, 0.615] | 0.278 [0.225, 0.331] | 0.505 [0.472, 0.538] | 0.499 [0.498, 0.501] |
| mass_moderate | 0.797 [0.772, 0.822] | 0.567 [0.544, 0.590] | 0.486 [0.480, 0.491] | 0.448 [0.411, 0.485] | 0.550 [0.517, 0.582] | 0.500 [0.498, 0.502] |
| mass_severe | 0.893 [0.884, 0.903] | 0.698 [0.687, 0.709] | 0.515 [0.498, 0.532] | 0.483 [0.457, 0.509] | 0.561 [0.542, 0.579] | 0.500 [0.498, 0.502] |
| motor_moderate | 0.912 [0.909, 0.916] | 0.771 [0.760, 0.783] | 0.526 [0.512, 0.540] | 0.588 [0.547, 0.629] | 0.569 [0.546, 0.593] | 0.500 [0.498, 0.502] |
| motor_severe | 0.963 [0.954, 0.972] | 0.886 [0.856, 0.916] | 0.609 [0.583, 0.636] | 0.619 [0.565, 0.673] | 0.556 [0.550, 0.562] | 0.500 [0.498, 0.502] |

## Lead time before fall (seconds, latent shield) and warning rate

| condition | n_fail | warn_rate | lead median (s) | shield intervention rate |
|---|---|---|---|---|
| friction_moderate | n/a | 1.000 [1.000, 1.000] | 1.303 [1.089, 1.518] | 0.998 [0.994, 1.002] |
| friction_severe | n/a | 1.000 [1.000, 1.000] | 0.747 [0.694, 0.799] | 0.995 [0.991, 1.000] |
| mass_moderate | n/a | 1.000 [1.000, 1.000] | 2.253 [1.463, 3.044] | 1.000 [1.000, 1.000] |
| mass_severe | n/a | 1.000 [1.000, 1.000] | 2.127 [1.238, 3.015] | 1.000 [1.000, 1.000] |
| motor_moderate | n/a | 1.000 | 1.780 | 1.000 |
| motor_severe | n/a | 1.000 [1.000, 1.000] | 1.393 [1.325, 1.462] | 0.998 [0.994, 1.002] |
