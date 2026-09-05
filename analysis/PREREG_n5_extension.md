# Pre-registration: extending the causal-viability replication from n=3 to n=5 process seeds

Written **before** the two new process seeds were run. Committed before the runs
were launched, so the git history proves the seed choice was not made after
seeing the outcome.

## Why

The registered v2 result (`reliability_eval/causal_viability_replication_v2`)
rests on 3 independent process seeds. Every selection-bias attack in
`analysis/selection_bias/` failed to break it, but one threat survives: with 3
process seeds the process-level (mean-of-process-means) interval on the
`stand_obs` **leak-free-subset** cell crosses zero. Block-level bootstrap
intervals cannot answer that, because blocks inside one process are not
independent replicates of the process. Only more processes can.

## What is added

Two additional process seeds, `process_04` and `process_05`, each running the
same four cells (`stand_motor`, `stand_obs`, `walk_motor`, `walk_obs`).
8 protocols, 16 arm runs.

Seeds continue the existing arithmetic sequence in
`scripts/reliability_replication_run.sh` verbatim, with no search over seeds:

| replicate  | process_seed | protocol_seeds (motor/obs stand, motor/obs walk) |
|------------|--------------|--------------------------------------------------|
| process_04 | 2026074104   | 2026074013, 2026074014, 2026074015, 2026074016   |
| process_05 | 2026074105   | 2026074017, 2026074018, 2026074019, 2026074020   |

`build_registry` independently rejects any reused protocol seed, block seed, or
scenario fingerprint, so seed collision with the first three processes is a hard
error rather than a judgement call.

## What is held fixed

Nothing in the experiment changes. The frozen experimental source snapshot
`d2f66d43173e8d4e367755c8f0878cda8f6990b82bfab1977d1b85f9e3d21396` (the 11 files
in `EXPERIMENT_SOURCE_PATHS`) is **bit-identical** to the snapshot the original
12 protocols pinned, verified before launch. Also unchanged:

- primary estimand: paired block-level post-onset fall-rate difference among
  jointly onset-eligible environment pairs, unshielded minus oracle
- eligibility rule: drop an env pair if either arm falls before the onset tick
- analysis unit: scenario block
- subset rule: a block is leak-free iff `max |onset_obs_u - onset_obs_o| == 0`
- gate thresholds and the four gate criteria in `analyze_registry`
- design: 16 envs/block, 32 disturbed + 16 nominal blocks, horizon 500,
  same checkpoints, ONNX, shield artifacts and env configs
- the same policies: `phoenix-stand-v3-h25-final`, `phoenix-flat-v4`

## Kill criterion, declared in advance

The flagship claim is the **registered fault-family result**: at the fault-family
level (motor, obs; the level the gate criterion is actually defined on), the
pooled effect intervals exclude zero with the registered signs, in every process.

- The claim **survives** if, at n=5: (a) the four gate criteria still pass on the
  full sample, and (b) the process-level interval (mean of 5 process means,
  bootstrapped over processes) on each fault family excludes zero with the
  registered sign.
- The claim **fails** if either fault-family process-level interval crosses zero
  or a sign flips.
- The `stand_obs` leak-free-subset cell is a **secondary** claim and is reported
  as replicating only if its own process-level interval excludes zero at n=5.
  If it does not, it is reported as not replicating at the process level. It
  will not be rescued by switching to block-level bootstrap, by dropping a
  process, or by redefining the subset.

No result already in hand is removed, and no cell is dropped, whatever the two
new seeds show.
