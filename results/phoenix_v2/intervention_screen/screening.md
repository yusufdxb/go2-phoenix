# Stage W intervention screening, result

Frozen W2, exact deploy stack, DR off, 384 nominal episodes over seeds [7001, 7002, 7003]. Preregistration: `docs/research/INTERVENTION_SCREENING.md` (amendment 13). Every threshold below was frozen before any cell ran.

**Nominal reference: walking success 0.9219**, primary score 0.9899, fidelity 1.000.

| family | joints | s | walk success | drop | min seed drop | prim score | fid | hold | att | prog | h05 | qualifies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 single joint | 1 | 0.80 | 0.9219 | +0.0000 | -0.0234 | 0.9896 | 1.000 | 0.000 | 0.005 | 0.942 | 0.264 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C1 single joint | 1 | 0.70 | 0.9271 | -0.0052 | -0.0156 | 0.9876 | 1.000 | 0.000 | 0.005 | 0.938 | 0.263 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C1 single joint | 1 | 0.60 | 0.9245 | -0.0026 | -0.0078 | 0.9853 | 1.000 | 0.000 | 0.008 | 0.932 | 0.262 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C1 single joint | 1 | 0.50 | 0.9115 | +0.0104 | -0.0156 | 0.9779 | 0.997 | 0.003 | 0.018 | 0.931 | 0.261 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C2 one leg | 3 | 0.90 | 0.8932 | +0.0286 | +0.0078 | 0.9817 | 1.000 | 0.000 | 0.016 | 0.939 | 0.260 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C2 one leg | 3 | 0.85 | 0.8906 | +0.0312 | +0.0078 | 0.9764 | 1.000 | 0.000 | 0.018 | 0.937 | 0.258 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C2 one leg | 3 | 0.80 | 0.8750 | +0.0469 | +0.0312 | 0.9666 | 0.995 | 0.005 | 0.023 | 0.940 | 0.258 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C2 one leg | 3 | 0.75 | 0.8542 | +0.0677 | +0.0469 | 0.9555 | 0.990 | 0.010 | 0.018 | 0.935 | 0.256 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C2 one leg | 3 | 0.70 | 0.7969 | +0.1250 | +0.1172 | 0.9267 | 0.979 | 0.021 | 0.042 | 0.934 | 0.150 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C3 both rear legs | 6 | 0.90 | 0.9010 | +0.0208 | -0.0078 | 0.9778 | 0.992 | 0.008 | 0.016 | 0.939 | 0.261 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C3 both rear legs | 6 | 0.85 | 0.9036 | +0.0182 | -0.0156 | 0.9727 | 0.992 | 0.008 | 0.008 | 0.936 | 0.259 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C3 both rear legs | 6 | 0.80 | 0.8438 | +0.0781 | +0.0625 | 0.9484 | 0.987 | 0.013 | 0.018 | 0.929 | 0.254 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C3 both rear legs | 6 | 0.75 | 0.7891 | +0.1328 | +0.1094 | 0.9176 | 0.971 | 0.029 | 0.021 | 0.912 | 0.207 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C3 both rear legs | 6 | 0.70 | 0.6328 | +0.2891 | +0.2734 | 0.8239 | 0.904 | 0.096 | 0.049 | 0.894 | 0.095 | **YES** |
| C4 global | 12 | 0.90 | 0.8646 | +0.0573 | +0.0469 | 0.9767 | 0.982 | 0.018 | 0.021 | 0.914 | 0.258 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C4 global | 12 | 0.85 | 0.8203 | +0.1016 | +0.0859 | 0.9626 | 0.969 | 0.031 | 0.023 | 0.895 | 0.229 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C4 global | 12 | 0.80 | 0.7214 | +0.2005 | +0.1797 | 0.9200 | 0.938 | 0.062 | 0.057 | 0.865 | 0.132 | **YES** |
| C4 global | 12 | 0.75 | 0.5781 | +0.3438 | +0.2969 | 0.8823 | 0.922 | 0.078 | 0.086 | 0.825 | 0.110 | **YES** |
| C4 global | 12 | 0.70 | 0.4271 | +0.4948 | +0.4297 | 0.7833 | 0.807 | 0.193 | 0.172 | 0.775 | 0.081 | not_catastrophic |

## Family monotonicity (rule 5, tolerance 0.05)

* C1 single joint: worst reversal +0.0052, PASS
* C2 one leg: worst reversal +0.0000, PASS
* C3 both rear legs: worst reversal +0.0026, PASS
* C4 global: worst reversal +0.0000, PASS

## Selection

**C3 both rear legs at s = 0.70** (6 joints: RR_hip_joint, RR_thigh_joint, RR_calf_joint, RL_hip_joint, RL_thigh_joint, RL_calf_joint), floor 0.7.

Walking success 0.6328, a drop of 0.2891 from nominal (every seed at least 0.2734), with 63.3% of episodes still succeeding.

Selected by the frozen rule: fewest affected joints among qualifying cells, then the least severe qualifying severity.

Held-out severities, reserved and not used for development: [0.75].

The global family also qualifies, so it is carried as the preregistered secondary intervention (section 5).
