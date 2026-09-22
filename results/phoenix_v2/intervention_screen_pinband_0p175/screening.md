# Stage W intervention screening, result

Frozen W2, exact deploy stack, DR off, 384 nominal episodes over seeds [7001, 7002, 7003]. Preregistration: `docs/research/INTERVENTION_SCREENING.md` (amendment 13). Every threshold below was frozen before any cell ran.

**Nominal reference: walking success 0.9219**, primary score 0.9899, fidelity 1.000.

| family | joints | s | walk success | drop | min seed drop | prim score | fid | hold | att | prog | h05 | qualifies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 single joint | 1 | 0.80 | 0.9089 | +0.0130 | +0.0078 | 0.9845 | 0.977 | 0.023 | 0.010 | 0.942 | 0.261 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C1 single joint | 1 | 0.70 | 0.9089 | +0.0130 | +0.0000 | 0.9781 | 0.971 | 0.029 | 0.013 | 0.937 | 0.222 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C1 single joint | 1 | 0.60 | 0.8984 | +0.0234 | +0.0156 | 0.9742 | 0.958 | 0.042 | 0.026 | 0.931 | 0.217 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C1 single joint | 1 | 0.50 | 0.8724 | +0.0495 | +0.0234 | 0.9599 | 0.938 | 0.062 | 0.031 | 0.929 | 0.146 | magnitude_pooled_drop_ge_0.15, reproducible_every_seed_drop_ge_0.15 |
| C2 one leg | 3 | 0.90 | 0.7578 | +0.1641 | +0.1328 | 0.8965 | 0.794 | 0.206 | 0.062 | 0.939 | 0.087 | reproducible_every_seed_drop_ge_0.15, not_catastrophic |
| C2 one leg | 3 | 0.85 | 0.7552 | +0.1667 | +0.1406 | 0.8940 | 0.792 | 0.208 | 0.042 | 0.930 | 0.088 | reproducible_every_seed_drop_ge_0.15, not_catastrophic |
| C2 one leg | 3 | 0.80 | 0.7318 | +0.1901 | +0.1797 | 0.8703 | 0.766 | 0.234 | 0.070 | 0.929 | 0.085 | not_catastrophic |
| C2 one leg | 3 | 0.75 | 0.6510 | +0.2708 | +0.2188 | 0.8424 | 0.688 | 0.312 | 0.086 | 0.920 | 0.085 | not_catastrophic |
| C2 one leg | 3 | 0.70 | 0.5729 | +0.3490 | +0.3125 | 0.7795 | 0.633 | 0.367 | 0.096 | 0.925 | 0.082 | not_catastrophic |
| C3 both rear legs | 6 | 0.90 | 0.7448 | +0.1771 | +0.1641 | 0.8978 | 0.781 | 0.219 | 0.052 | 0.934 | 0.092 | not_catastrophic |
| C3 both rear legs | 6 | 0.85 | 0.6667 | +0.2552 | +0.2500 | 0.8656 | 0.714 | 0.286 | 0.073 | 0.921 | 0.091 | not_catastrophic |
| C3 both rear legs | 6 | 0.80 | 0.6589 | +0.2630 | +0.2344 | 0.8453 | 0.711 | 0.289 | 0.073 | 0.916 | 0.086 | not_catastrophic |
| C3 both rear legs | 6 | 0.75 | 0.5807 | +0.3411 | +0.3203 | 0.7957 | 0.656 | 0.346 | 0.062 | 0.913 | 0.083 | not_catastrophic |
| C3 both rear legs | 6 | 0.70 | 0.3854 | +0.5365 | +0.5000 | 0.6599 | 0.492 | 0.508 | 0.109 | 0.873 | 0.079 | headroom_success_ge_0.40, not_catastrophic |
| C4 global | 12 | 0.90 | 0.5417 | +0.3802 | +0.3516 | 0.7867 | 0.573 | 0.427 | 0.091 | 0.887 | 0.081 | not_catastrophic |
| C4 global | 12 | 0.85 | 0.4948 | +0.4271 | +0.4062 | 0.7540 | 0.516 | 0.484 | 0.122 | 0.869 | 0.082 | not_catastrophic |
| C4 global | 12 | 0.80 | 0.3620 | +0.5599 | +0.5234 | 0.6585 | 0.406 | 0.594 | 0.141 | 0.795 | 0.080 | headroom_success_ge_0.40, not_catastrophic |
| C4 global | 12 | 0.75 | 0.2344 | +0.6875 | +0.6641 | 0.5705 | 0.294 | 0.706 | 0.133 | 0.687 | 0.079 | headroom_success_ge_0.40, not_catastrophic |
| C4 global | 12 | 0.70 | 0.1042 | +0.8177 | +0.7734 | 0.4237 | 0.159 | 0.841 | 0.154 | 0.598 | 0.079 | headroom_success_ge_0.40, not_catastrophic |

## Family monotonicity (rule 5, tolerance 0.05)

* C1 single joint: worst reversal +0.0000, PASS
* C2 one leg: worst reversal +0.0000, PASS
* C3 both rear legs: worst reversal +0.0000, PASS
* C4 global: worst reversal +0.0000, PASS

## Selection

**No cell qualifies. The Phoenix adaptation experiment stops.**

Finding: no safe actuator intervention within the tested envelope produced the required measurable degradation. By the stop rule no floor is lowered, no family added, and the endpoint is not changed again.
