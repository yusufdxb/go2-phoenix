# SUPERSEDED: captured through the frame-inverse defect

These three trajectories were harvested on 2026-09-11 while
`snapshot_manager_state` removed only the env origin's Z and `restore_state`
added the full XYZ. Their recorded `base_pos` x/y are WORLD-GRID coordinates,
measured at [6.603, -8.670], [-8.344, -4.417] and [1.353, 1.134], so a seeded
robot would be placed up to about 10 m from its own tile.

DO NOT use them as curriculum seeds. The H0 verdicts and the 0.0 to 2.0 s offset
sweep computed from them do not stand.

They are kept unmodified as a record of what the defect produced. They are not
relabelled, because relabelling would require assuming the flat grid cloner sets
origin z = 0 and nobody measured that.

Fix: commit f4ccf08. Record: docs/CORRECTIVE_PASS_2026-09-11.md section 6.
