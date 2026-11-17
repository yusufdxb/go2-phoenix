# Gantry A/B analysis (logs/hw_sessions/gantry_ab/20261020_228555e8443b_abc12_2026-10-18/campaign.jsonl)

Verdict (exploratory): **INSUFFICIENT**

| block | arm | slot | att | valid | Y | F | overshoot | calf sag | max tilt | why |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | G | 1 | True |  |  |  | 0.136 | 2.433 |  |
| 1 | A | vx03_4s | 1 | True |  | 0.677 |  |  |  |  |
| 2 | B | G | 1 | True |  |  |  | 0.135 | 1.655 |  |
| 2 | B | vx03_4s | 1 | True |  | 0.995 |  |  |  |  |
| 3 | C | G | 1 | True |  |  |  | 0.143 | 2.101 |  |
| 3 | C | vx03_4s | 1 | True |  | 1.037 |  |  |  |  |
| 4 | A | G | 1 | True |  |  |  | 0.136 | 2.335 |  |
| 4 | A | vx03_4s | 1 | False |  |  |  |  |  | not exactly one policy episode; LowState not fresh on every policy tick; operator: rope loaded / touched / aborted; deadman released (release_without_policy_episode): aborted run, from the logs; walk_cmd outcome None; no policy.parquet |
| 4 | A | vx03_4s | 2 | True |  | 0.672 |  |  |  |  |
| 5 | C | G | 1 | True |  |  |  | 0.147 | 2.124 |  |
| 5 | C | vx03_4s | 1 | True |  | 1.027 |  |  |  |  |
| 6 | B | G | 1 | True |  |  |  | 0.132 | 1.895 |  |
| 6 | B | vx03_4s | 1 | True |  | 1.001 |  |  |  |  |
| 7 | C | G | 1 | True |  |  |  | 0.146 | 2.129 |  |
| 7 | C | vx03_4s | 1 | True |  | 1.032 |  |  |  |  |
| 8 | B | G | 1 | True |  |  |  | 0.136 | 1.981 |  |
| 8 | B | vx03_4s | 1 | True |  | 0.995 |  |  |  |  |
| 9 | A | G | 1 | True |  |  |  | 0.137 | 2.238 |  |
| 9 | A | vx03_4s | 1 | True |  | 0.677 |  |  |  |  |
| 10 | B | G | 1 | True |  |  |  | 0.139 | 2.082 |  |
| 10 | B | vx03_4s | 1 | True |  | 0.984 |  |  |  |  |
| 10 | B | stop_drill | 1 | False |  |  |  | 0.138 | 1.874 | STOP drill: the deadman was not released during policy authority |
| 11 | C | G | 1 | True |  |  |  | 0.141 | 2.044 |  |
| 11 | C | vx03_4s | 1 | True |  | 1.032 |  |  |  |  |
| 11 | C | stop_drill | 1 | True |  |  |  |  |  |  |
| 12 | A | G | 1 | True |  |  |  | 0.139 | 2.299 |  |
| 12 | A | vx03_4s | 1 | True |  | 0.683 |  |  |  |  |
| 12 | A | stop_drill | 1 | True |  |  |  |  |  |  |

## Tracking (measured / commanded; mean +- std over valid repeats, n)

| arm | slot | metric | mean | std | n |
|---|---|---|---|---|---|
| A | vx03_4s | F | 0.677 | 0.004 | 4 |
| B | vx03_4s | F | 0.994 | 0.007 | 4 |
| C | vx03_4s | F | 1.032 | 0.004 | 4 |

## Stop (seconds, from the ticks)

| arm | measure | mean | max | n |
|---|---|---|---|---|
| A | stop_latency_s | 0.000 | 0.000 | 1 |
| A | release_rx_to_zero_s | 0.000 | 0.000 | 1 |
| A | window_end_to_sit_s | 1.005 | 1.007 | 9 |
| B | window_end_to_sit_s | 1.005 | 1.006 | 9 |
| C | stop_latency_s | 0.000 | 0.000 | 1 |
| C | release_rx_to_zero_s | 0.000 | 0.000 | 1 |
| C | window_end_to_sit_s | 1.005 | 1.006 | 9 |
