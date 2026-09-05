# Paper Outline: Phoenix reliability shield (ICRA 2027)

> STATUS 2026-08-24: the pre-onset characterization returned BRANCH (b), a real treatment leak in env.reset(). Study v1 cannot ship. This skeleton stands, but Section 5 needs the v2 re-run behind it. See reliability_eval/causal_viability_replication/NEGATIVE_CONTROL_ANALYSIS.md.

Owner artifact of paper-outline-architect. Written 2026-08-24. Governed by
`~/Documents/Obsidian Vault/Projects/go2-phoenix/contribution_contract.md` (LOCKED 2026-08-15).
Nothing in this skeleton may claim more than that file allows.

Status of inputs actually read this session (verified, not assumed):

- `contribution_contract.md` (vault) : read in full.
- `status.md`, `tasks.md` (vault) : read in full.
- `reliability_eval/causal_viability_replication/{result_table.md, combined_summary.json, registry.json}` : read in full.
- `~/Projects/go2-phoenix-litmap.md` : read in full (this is the literature map; no `paper_state/literature_map.md` exists).
- `reliability_eval/closed_loop/REGIME_CHARACTERIZATION.md`, `reliability_eval/closed_loop_walk/FINDINGS.md`, `reliability_eval/results_stand/FINDINGS.md` : read in full.
- NOT FOUND, therefore not used: `paper_state/framing_memo.md`, `paper_state/claim_ledger.md`, `drafts/`, any prior `outline.md`. Claim IDs below are defined here for the first time, derived from the contract's numbered "May claim" list.

---

## Target venue and section structure

- Venue: ICRA 2027. Deadline Sep 15, 2026, 23:59 PST (verified in the contract against the official CFP). 22 days from today.
- Chosen structure: IEEE conference measurement paper, not a systems paper and not a method paper. Order: Introduction, Related Work, Apparatus, Study Design, Results, Threats to Validity, Limitations, Conclusion.
- Why this structure fits: the contribution is a measurement, so the load-bearing sections are Study Design (why the measurement is credible) and Threats to Validity (why the one failed control does not void it). A systems-paper structure would spend its page budget on the shield implementation, which is not the contribution. A method-paper structure would imply we are proposing the shield, which the contract forbids (the shield harms in two of four cells).
- Page budget: 8 pages TOTAL, references included. VERIFIED 2026-08-24 from the official ICRA 2027 CFP (https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/): "The page limit is 8 pages for the complete paper (text, figures, tables, acknowledgement, bibliography/references)." Papers over 8 pages are returned without review, and no fee-based extension is offered. Budget references at about 1 page for 25 to 30 entries, leaving roughly 7 pages of body. The earlier 6-page assumption is retired.

---

## Argument spine (problem, gap, contribution, evidence, limitations)

Section 1 carries problem (runtime shields are deployed on learned locomotion policies on the assumption that a better failure detector buys a safer robot) and gap (nobody has separated detector quality from closed-loop benefit with an interventional design; the literature evaluates detectors observationally with AUROC tables). Section 2 establishes the gap against named neighbors and concedes the one sibling finding (Shojaei). Sections 3 and 4 build the instrument that closes the gap: an oracle detector plus dose-matched sham arms in a pre-registered 2x2 factorial. Section 5 carries the evidence: the sign of the shield's benefit flips with fault family at a perfect detector, interaction +19.24 pp, reproducing in all three processes. Section 6 carries the honest damage: under study v2 the pre-registered pre-onset negative control passes the frozen gate, but it passes by touching zero in the two walking cells rather than straddling it, and a smaller post-reset residual remains, positively measured as within-tick coupling in the shared physics batch and bounded at +0.065 pp. Section 7 carries scope (sim only, one policy family, one simulator, one fallback design). Section 8 restates the measurement, not a recommendation.

---

## Claim ledger for this outline

Derived verbatim from the contract's "May claim" list. Every claim lands in exactly one section.

| ID | Claim | Home section | Evidence artifact |
|---|---|---|---|
| C1 | The sign of the shield's closed-loop benefit flips with the fault family, not with detector quality; the contrast runs at an oracle detector so detection error cannot explain it | 5.1 | `combined_summary.json` -> `pooled_cells`, `fault_by_treatment_interaction_obs_minus_motor` |
| C2 | Mechanism: a static stand fallback has no recovery authority under actuator degradation and real authority under observation corruption, where the learned policy is the corrupted component | 5.3 | `combined_summary.json` -> `pooled_cells` fall rates; `REGIME_CHARACTERIZATION.md` for the mechanism statement |
| C3 | Method export: oracle detector plus dose-matched sham arms as the design that separates detection quality from closed-loop outcome | 4 (design), 5.4 (the null it produced) | `registry.json`; `closed_loop_walk/results.json` for the dose-matched sham null |
| C4 | Direction reproduces in all 3 processes and survives leave-one-process-out in every cell | 5.2 | `combined_summary.json` -> `process_effects`, `leave_one_process_out` |
| N1 | NEGATIVE, must be reported: the pre-onset negative control passes under v2 (all four cells include zero, gate passes 4/4) but the two walking cells pass at +0.065 pp with an interval touching zero, and a post-reset onset residual remains, identified by a positive test and bounded at 1.1 percent of the smallest primary effect | 6 | `combined_summary.json` -> `pooled_pre_onset_negative_control_cells`, `gate_checks`; `onset_residual_audit.json` |

Excluded by the contract and therefore absent from every section below: any monitor-timing benefit as a benefit, any hardware result, any effect smaller than about 2 pp, universality across policies or simulators or fallback designs, "the shield makes the robot safer" unqualified, any merge with Phantom-Braking, ODIN, or VLA-Blindspot.

---

## Sections

### 1. Introduction

- JOB (one sentence): Convince an ICRA reviewer in one page that "better detector implies safer shielded robot" is an unexamined assumption, and that we measured it and it is false in a specific, mechanistic way.
- Claims it carries: none stated as evidence. Forward-references C1 and C2 as the paper's result.
- Paragraph plan:
  - P1: Learned locomotion policies are shipped behind runtime shields; the standard engineering move when a shield underperforms is to improve the failure detector.
  - P2: That move presumes detector quality predicts closed-loop benefit, a link that observational detector benchmarks (AUROC, lead time) cannot test because they never intervene.
  - P3: We test it with a pre-registered oracle-arm 2x2 factorial on a Unitree Go2 policy pair in Isaac: two policies (stand, walk) crossed with two fault families (motor degradation, observation corruption), unshielded versus oracle-shielded, 576 independent blocks, 3 independent processes.
  - P4: The result: at a perfect detector the shield's benefit changes sign with the fault family. Motor degradation, stand: -14.47 pp. Observation corruption, walk: +11.57 pp. Interaction +19.24 pp [+16.94, +21.59].
  - P5: Because the detector is an oracle in every cell, detection error cannot explain the flip; what changes is whether the fallback retains recovery authority in the detected fault family.
  - P6: Contributions, stated as a measurement and a method, plus one sentence conceding the failed pre-onset control and the simulation-only scope up front so no reviewer discovers it late.
- Figures/tables placed here:
  - Figure 1: the flip. Supports C1. Placed at the top of page 1 or 2 so the reviewer sees the sign change before reading Section 4.

### 2. Related Work

- JOB (one sentence): Show that the runtime-assurance, OOD-monitor, and recovery-policy literatures each own a piece of this question and that none of them has run the interventional design that answers it.
- Claims it carries: none of C1 to C4. It establishes the gap that licenses them.
- Paragraph plan:
  - P1: Runtime assurance and Simplex for learned controllers (Hobbs et al., IEEE Control Systems 2023; Konighofer et al., CACM 2025). Delta: design-space and formal-guarantee traditions, no interventional measurement of detector quality versus outcome.
  - P2: Quadruped-specific runtime shields and safety supervision (Agile But Safe, RSS 2024; One Filter to Deploy Them All, TRO 2025; Safety supervision framework for legged robots, Robotica 2025). Delta: all evaluate the deployed filter on success and safety rates, all positive-only, none reports a regime where the filter harms and none ablates the detector to an oracle.
  - P3: Failure detectors and OOD monitors for policies (SAFE, arXiv:2506.09937; latent-space reachability, RSS 2025). Delta: these are detection benchmarks; better detection is the outcome variable, closed-loop benefit is assumed.
  - P4: Fault-tolerant locomotion and recovery policies (FT-Net, RA-L 2023; Fault Joint Detection, RA-L 2025; Recovery RL, RA-L 2021). Delta: adapt-in-place or train-a-better-recovery-policy contributions, no monitor-triggered switch to ablate. Recovery RL argues our mechanism qualitatively but never measures it.
  - P5: The one sibling finding, handled honestly: Shojaei, arXiv:2606.25371 (2026-06-24), shows a latching shield suppressing a controller that would have self-corrected. Concurrent, not prior art. Different mechanism (engaged too early against a recoverable controller) from ours (fallback lacks authority in the detected mode). State this explicitly rather than burying it.
  - P6: The gap in one sentence: no paper separates a detection metric from a closed-loop task-level outcome via a controlled interventional design on a robot policy.
- Figures/tables placed here: none. A contrast table is now affordable under the verified 8-page limit, but the deltas are still one clause each in prose; promote it only if a section-2 read makes the positioning unclear. If a reviewer asks for it, it goes in a supplementary appendix, not the body.
- OPEN FLAG carried from the litmap: `SafeRecovery`, doi `10.1007/978-981-92-3381-6_42`, ICIC 2026, abstract paywalled and unfetchable from OpenAlex, Crossref, arXiv, and Springer. Title ("safety-recovery benchmark with four-axis evaluation for quadruped locomotion") is close enough that it must be read before submission. Scheduled Day 6. If its four axes include an oracle arm or an engagement-matched sham, P5 and possibly the contribution boundary change, and the decision routes to paper-contribution-locker, not to this outline.

### 3. Apparatus

- JOB (one sentence): Describe the policies, the fallback, the fault families, and the shield precisely enough that the reader accepts the measured effects belong to a real system and not to a toy.
- Claims it carries: none directly. It is the setup C1 and C2 are measured on.
- Paragraph plan:
  - P1: Platform and policies. Unitree Go2 in Isaac; a standing policy and a velocity-commanded walking policy; both trained with domain randomization. State that this is one policy family, once, here.
  - P2: The fallback: a static default stand pose, blended toward zero action. State that it is the fallback under test, not a proposed design.
  - P3: The two fault families and why they are the factorial's second factor: motor degradation attacks the actuators the fallback itself depends on; observation corruption attacks the learned policy while leaving the body intact.
  - P4: The monitor in the deployed shield (squared Mahalanobis on post-ELU policy latents) and, critically, that the study does not use it: every arm in the factorial is unshielded or oracle, so the monitor's quality is held at its ceiling by construction.
- Figures/tables placed here:
  - Figure 2: apparatus and arm diagram. Supports C3 (the design is the method contribution) and makes the pre-onset versus post-onset window split legible before Section 6 needs it.

### 4. Study Design

- JOB (one sentence): Establish that this is a pre-registered interventional experiment with a real unit of analysis, real pairing, and pre-committed gates, so its numbers carry causal weight that an AUROC table cannot.
- Claims it carries: C3 (the design as method export).
- Paragraph plan:
  - P1: Unit of analysis is the pre-registered scenario block, not the episode. 96 independent disturbed blocks per cell, 384 disturbed and 192 nominal blocks total, 576 independent blocks, 16 environments per block.
  - P2: Arms. Unshielded versus oracle (engagement at true onset, zero false positives, measured latency 0 ticks at min, median, and max in every cell). Why the oracle arm is the instrument: it removes detection error from the comparison entirely.
  - P3: The dose-matched sham and why dose matching is necessary: a globally permuted sham confounds "when you switch" with "how often you switch". State the correction as a methods contribution, cite the clinical-trial lineage of the term.
  - P4: Pre-registration and freezing. `study_id phoenix_causal_viability_replication_v1`, registry hash `46fc4829...`, per-cell protocol hashes, per-arm trajectory SHA-256, 18 exploratory protocols excluded by path before analysis, 3 independent processes with distinct process seeds.
  - P5: The pre-committed gate, stated before the results so the reader knows it was not chosen after seeing them: direction reproduces in all process cells, leave-one-process-out preserves direction, pooled fault intervals exclude zero, and pre-onset negative controls include zero. Say here, not in Section 6, that the fourth criterion failed.
  - P6: Analysis: block-paired differences, block bootstrap 95% intervals, sign convention (positive unshielded minus oracle means the fallback reduces post-onset falls).
- Figures/tables placed here:
  - Table I: pre-registration and accounting. Supports C3.

### 5. Results

- JOB (one sentence): Show the sign flip, show it is not an artifact of one process or one cell, and name the mechanism, all within the contract's boundary.
- Claims it carries: C1 (5.1), C4 (5.2), C2 (5.3), C3's null (5.4).
- Paragraph plan:
  - 5.1 P1: The four cells at the oracle detector. Stand/motor -14.47 [-17.00, -11.98]; walk/motor -6.24 [-9.26, -3.28]; stand/obs +6.21 [+4.91, +7.56]; walk/obs +11.57 [+9.40, +13.76]. The shield harms in both motor cells and helps in both observation cells.
  - 5.1 P2: The interaction is the headline: observation minus motor is +19.24 pp [+16.94, +21.59] over 384 blocks. State explicitly that this contrast is between two arms that both had a perfect detector, therefore detection quality is held constant across the sign change.
  - 5.1 P3: Fall-rate context so the pp effects are readable: stand/motor unshielded 11.40 percent versus oracle 25.78 percent; walk/obs unshielded 15.78 percent versus oracle 4.18 percent.
  - 5.2 P1: Robustness. Direction reproduces in all 3 processes in every cell (per-process effects listed) and every leave-one-process-out estimate preserves sign in all 12 combinations.
  - 5.2 P2: One honesty note: in process 01, walk/motor the per-process interval includes zero (-1.78 pp) even though the direction holds. Report it rather than let a reviewer find it in the JSON.
  - 5.3 P1: Mechanism. Under motor degradation the fallback depends on the same weakened actuators, so freezing removes the active compensation the policy was providing; oracle-treated falls are failures after correct engagement, not trigger misses (389 and 751 treated falls, zero missed engagements).
  - 5.3 P2: Under observation corruption the learned policy is the corrupted component and the body is intact, so a static pose is a safe attractor and the fallback has real authority.
  - 5.3 P3: The secondary outcome that supports the mechanism read: task completion moves the same way (stand/motor 88.60 percent unshielded versus 74.22 percent oracle; walk/obs 84.22 versus 95.82).
  - 5.4 P1: The detector-quality side of the claim, stated as the null it is. A latent-OOD monitor at episode-level AUROC 0.87 to 1.00 buys +0.004 [-0.023, +0.033] over a dose-matched blind switcher in the positive regime. Frame as "no measurable benefit from detector timing", never as a benefit.
- STALE NUMBERS, FLAGGED 2026-09-04, NOT YET REWRITTEN: every figure in the 5.1 to 5.3 paragraph plan above is a study **v1** number. The registered result is now v2 (`reliability_eval/causal_viability_replication_v2/combined_summary.json`, gate 4/4). Directions are unchanged in all four cells; magnitudes are not. Replace with, in pp:
  - stand_motor -23.73 [-26.32, -21.20] (was -14.47); stand_obs +9.61 [+8.09, +11.23] (was +6.21); walk_motor -5.95 [-7.87, -4.03] (was -6.24); walk_obs +17.52 [+15.31, +19.88] (was +11.57).
  - Interaction, observation minus motor: +28.41 [+26.30, +30.51] (was +19.24). This number appears in the argument spine and in the abstract skeleton as well; fix it in every location, not only here.
  - Fall-rate context for 5.1 P3: stand_motor unshielded 12.30 percent versus oracle 36.06; walk_obs unshielded 20.62 percent versus oracle 3.15.
  - 5.2 P1 per-process effects: stand_motor [-19.64, -27.87, -23.70]; stand_obs [+9.04, +9.62, +10.18]; walk_motor [-5.49, -6.43, -5.92]; walk_obs [+16.50, +18.39, +17.68]. Every leave-one-process-out estimate preserves sign and excludes zero in all 12 combinations (`leave_one_process_out` in the same JSON).
  - 5.2 P2, the v1 honesty note about process 01 walk_motor including zero, DOES NOT APPLY to v2: `direction_reproduces_in_all_process_cells` and `leave_one_process_out_preserves_direction` both pass. Do not carry that sentence forward without re-checking per-process intervals against v2.
  - Do not transcribe by hand; render from the JSON via `scripts/reliability_result_table.py`.
- Figures/tables placed here:
  - Table II: primary outcome, all four cells. Supports C1.
  - Figure 3: per-process and leave-one-process-out robustness. Supports C4. FIRST TO CUT if the layout runs over 8 pages including references, because Table II already carries the per-process effect column. Under the verified limit this cut is unlikely to be needed.

### 6. Threats to Validity

- JOB (one sentence): Report the pre-onset negative control in full under study v2, name the residual that remains, show it was measured rather than argued away, and bound its influence on the primary result before the reviewer forms their own reading.
- Claims it carries: N1.
- SOURCE OF TRUTH: every number below comes from `reliability_eval/causal_viability_replication_v2/combined_summary.json` (registered estimand and gate) and `reliability_eval/causal_viability_replication_v2/onset_residual_audit.json` (residual audit). Draft prose is in `paper/onset_residual_limitation.md`. Do not transcribe or round these by hand; Table III and Table IIIb render from the JSON via `scripts/reliability_result_table.py`.
- Paragraph plan:
  - P1: What the control is. Pre-onset windows are identical-treatment by construction: the oracle has not engaged yet, so the true effect is exactly zero. Any non-zero estimate is residual imbalance or noise.
  - P2: What it returned under v2. stand_motor +0.000 pp [+0.000, +0.000]; stand_obs +0.000 pp [+0.000, +0.000]; walk_motor +0.065 pp [+0.000, +0.195]; walk_obs +0.065 pp [+0.000, +0.195]. The frozen gate passes all four checks, `pre_onset_negative_controls_include_zero` included. State explicitly that the two walking cells pass by touching zero, not by straddling it.
  - P3: What is and is not aligned across the paired arms. The batched-block harness gives each block its own 16 environments and one lifetime, so no block can inherit a predecessor's simulator state. Measured across all 12 process-cell arm pairs: reset states are bit-identical (max abs diff exactly 0.0, 12 of 12) and initial observations are bit-identical (12 of 12), but onset observations are not, diverging in 4 to 42 of 48 blocks per pair with per-pair max abs differences of 2.87 to 13.18. The v1 reset leak is closed; a smaller channel acting after reset is not.
  - P4: The residual is positively measured, not inferred by elimination. Prediction: if the channel is within-tick coupling through the single shared GPU physics batch, divergence must be an upward closed set in onset TICK, with one threshold per arm pair, and must not be ordered by block index, disturbance status, or environment index. It holds without exception: a single onset-tick threshold separates divergent from bit-identical blocks in 12 of 12 arm pairs, joint probability under the arbitrary-subset null on the order of 1e-124. The mechanism is therefore identified as within-tick spatial coupling, and temporal carryover is ruled out. One caveat stated in the same paragraph: the implied propagation delay is not a single constant, with per-pair brackets from (12, 18] to (86, 88] ticks and no common value.
  - P5: Magnitude, named exactly rather than characterised. Across all 12 pairs, pre-onset fall status differs for 6 of 9,216 environment pairs, and for 2 of 6,144 inside the disturbed blocks the registered estimand uses. The largest residual, +0.065 pp, is 1.1 percent of the smallest primary effect (walk_motor, -5.95 pp) and 0.4 percent of the largest (walk_obs, +17.52 pp).
  - P6: Contamination-free sensitivity analysis. Because divergence is exactly the upper tail in onset tick, the bit-identical blocks are a subset on which the arms are provably identical up to onset. Recomputing the registered primary estimand there: stand_motor -23.28 [-30.56, -16.57] (n=17); stand_obs +9.27 [+6.98, +11.70] (n=44); walk_motor -7.19 [-10.40, -4.00] (n=24); walk_obs +16.19 [+13.19, +19.28] (n=32). All four keep their sign, all four intervals exclude zero, and every registered point estimate falls inside its contamination-free interval, so the sign flip survives on data the residual cannot have touched. Report it as post hoc: the subsets are small and are the early-onset blocks rather than a random sample, so this bounds the residual's influence without being an unbiased estimate.
  - P7: What we do not claim, and why we report rather than fix. We do not claim the harness is bit-exact; 11 of 12 arm pairs diverge in a majority of blocks at onset. We claim the divergence enters after reset through a mechanism identified by a positive test, that its effect on the registered control is at most +0.065 pp with an interval touching zero, and that the primary effects reproduce on the blocks it provably did not reach. Eliminating it entirely requires one physics batch per block, a 48-fold increase in simulator launches, judged not worth the compute against a residual of this size.
- Figures/tables placed here:
  - Table III: pre-onset negative control, all four cells plus family pooling, under v2. Supports N1. This table is non-negotiable; the control is reported as a table, not as a sentence in a limitations list.
  - Table IIIb: contamination-free sensitivity analysis, registered estimand against the bit-identical-block subset, all four cells. Supports N1 and defends C1.
- SUPERSEDED HISTORY (kept for traceability, do not draft from it): this section previously carried the v1 failing control (`gate_passed: false`, three of four cells excluding zero), a noise-floor reading that was DELETED as unsourced, and the real v1 treatment leak diagnosed in `NEGATIVE_CONTROL_ANALYSIS.md` (commit 179d7f5), where `env.reset()` returned an observation stale from the previous block's terminal state. The v2 batched-block re-run (commit 64cf339, clean tree) is the design fix that branch (b) called for. None of the v1 numbers belong in the draft.

### 7. Limitations

- JOB (one sentence): Fence the claim to exactly the contract's boundary so no reviewer can read generality into it.
- Claims it carries: none. It removes claims.
- Paragraph plan:
  - P1: Simulation only. Every number is Isaac. No GO2 hardware figure exists. The CaresLab Gate 7 stand test has never been run. State this plainly, not in a subordinate clause.
  - P2: n=1 policy family, one simulator, one fallback design (a static stand pose). The flip is a statement about this fallback's recovery authority, not about fallbacks in general.
  - P3: Scope of the detector claim: we show that detector quality does not explain the sign of the benefit. We do not show that detector quality never matters, and for fault identification (as distinct from intervention benefit) it plainly does.
  - P4: What would falsify or extend this: a fallback with actuator-independent authority under motor degradation, or the same factorial on hardware.
- Figures/tables placed here: none.

### 8. Conclusion

- JOB (one sentence): Restate the measurement and the exportable design in five sentences without adding a recommendation the evidence does not support.
- Claims it carries: restates C1 and C3, adds nothing.
- Paragraph plan:
  - P1: One paragraph. The sign of a shield's benefit is set by the fallback's recovery authority in the detected fault family, measured at a perfect detector. The exportable piece is the design, not the shield. Explicitly refuse the sentence "the shield makes the robot safer".

### Appendix (only if ICRA allows overflow pages; otherwise a repo link)

- JOB (one sentence): Give a reproducer the exact hashes, seeds, and commands to regenerate every number in the body.
- Claims it carries: none.
- Paragraph plan:
  - P1: Study registry, protocol hashes, process seeds, trajectory SHA-256, and the excluded exploratory protocol list.
  - P2: Reproduce commands.
- BLOCKER CLEARED 2026-09-04: the study code and the v2 artifact tree are committed and pushed on `feat/causal-viability-replication` (f18219f through 719ea52). A repo link is now a valid appendix.

---

## Figure and table manifest

Every asset names the artifact file it is computed from. NONE of the paper's figures exist today. The `.png` files already in `reliability_eval/` (`auroc_by_condition.png`, `fall_rate_by_arm.png`, `operating_point_tradeoff.png`) belong to earlier studies and are NOT figures of the replication; none is reused.

| Asset | Section | What it must show | Claim | Source artifact | Status |
|---|---|---|---|---|---|
| Figure 1 | 1 | Forest plot: 4 cell effects with 95 percent block-bootstrap CIs on a signed axis, zero line marked, interaction +19.24 [+16.94, +21.59] as a fifth row; the two motor cells left of zero, the two observation cells right of zero | C1 | `reliability_eval/causal_viability_replication/combined_summary.json` keys `pooled_cells.*.{mean_difference,ci_low,ci_high}` and `fault_by_treatment_interaction_obs_minus_motor` | TO PRODUCE. No plotting script exists. Needs a new `scripts/paper_figures.py`. |
| Figure 2 | 3 | Apparatus and arm diagram: policy, monitor tap, fallback blend, the two arms, the block pairing, and the episode timeline with the pre-onset window and the registered onset tick marked | C3 | Structural, from `src/phoenix/reliability/{runtime.py,arbiter.py,study.py}` plus `registry.json` for the block accounting. No numeric claims on this figure. | TO PRODUCE. Must be a real vector diagram (Graphviz or SVG), never ASCII. |
| Figure 3 | 5.2 | Small multiples, one panel per cell: 3 per-process effects plus the 3 leave-one-process-out estimates with CIs, all on the same signed axis, showing sign preservation in 12 of 12 | C4 | `combined_summary.json` keys `pooled_cells.*.process_effects` and `pooled_cells.*.leave_one_process_out.*` | TO PRODUCE. CONDITIONAL: first asset cut if the paper exceeds 8 pages including references. |
| Table I | 4 | Pre-registration and accounting: study_id, registry hash, 3 processes, 4 cells, 96 disturbed blocks per cell, 576 independent blocks, 12 independent protocols, 18 exploratory protocols excluded, 4 frozen gate criteria with pass/fail | C3 | `reliability_eval/causal_viability_replication/registry.json` and `combined_summary.json` key `gate_checks` | TO PRODUCE (LaTeX table, values transcribed from the two JSONs). |
| Table II | 5.1 | Primary outcome: per cell, blocks, eligible pairs, unshielded and oracle fall rates, block-paired effect, 95 percent CI, and the 3 per-process effects | C1, C4 | `reliability_eval/causal_viability_replication/result_table.md` "Primary outcome" table, cross-checked against `combined_summary.json` | TO PRODUCE. Exists as markdown, needs LaTeX conversion plus an independent recompute from the JSON before it ships. |
| Table III | 6 | Pre-onset negative control: per-cell effect and CI for all four cells, plus the two family-level pooled rows, with the failed cells marked | N1 | `result_table.md` "Failed negative-control check" and `combined_summary.json` keys `pooled_pre_onset_negative_control_cells`, `pooled_pre_onset_negative_control_fault_families` | TO PRODUCE. |

Assets considered and deliberately NOT placed (flagged for cut, so nobody re-adds them without a role):

- Detector AUROC table from `reliability_eval/results_stand/FINDINGS.md` (latent-Mahalanobis 0.873 to 1.000). Role would be "the detector is good and it still does not predict benefit", but that is a single clause in 5.4 and a whole table costs a third of a column. Cut; keep the number inline.
- `closed_loop_walk/fall_rate_by_arm.png` (5-arm bar chart including the dose-matched sham). Genuinely relevant to 5.4 but it comes from a different study than the locked evidence base, and placing it would invite the reviewer to read the sham null as the paper's primary evidence. Cut; keep 5.4 to prose plus the interval.
- A related-work contrast table. Deprioritized, not budget-blocked, under the verified 8-page limit. Deltas live in Section 2 prose; promote only on demand.
- Any hardware photo or plot. None exists and none may be implied.

---

## Word and page budget against 8 pages including references (VERIFIED)

ASSUMPTION: IEEE two-column 10pt gives roughly 1,050 words per full text page. Figures and tables are charged at their estimated column fraction.

| Section | Target words | Asset charge | Estimated pages |
|---|---:|---|---:|
| Title, authors, abstract | 180 | none | 0.30 |
| 1 Introduction | 750 | Figure 1 (0.40 page) | 1.11 |
| 2 Related Work | 550 | none | 0.52 |
| 3 Apparatus | 500 | Figure 2 (0.35 page) | 0.83 |
| 4 Study Design | 800 | Table I (0.25 page) | 1.01 |
| 5 Results | 900 | Table II (0.30), Figure 3 (0.35) | 1.51 |
| 6 Threats to Validity | 650 | Table III (0.22), Table IIIb (0.20) | 1.04 |
| 7 Limitations | 250 | none | 0.24 |
| 8 Conclusion | 120 | none | 0.11 |
| Body total | 4,700 | 2.27 pages of assets | 6.67 body, plus about 1.0 for references = 7.7 of 8 |

Over budget by about 0.67 page as of 2026-09-04, up from 0.28 after Section 6 grew to seven paragraphs and gained Table IIIb. Pre-committed cut order, so the cut is a decision made now rather than a panic on Day 20:

1. Cut Figure 3 (saves 0.35 page). Table II's per-process column already carries C4; the leave-one-process-out numbers move to one sentence in 5.2.
2. If still over: compress Section 2 from 6 paragraphs to 4 by merging P3 into P1 and P4 into P2 (saves about 0.15 page).
3. If still over: cut the appendix entirely and point at the repo. The repo link is now valid: the study code and the v2 artifacts are committed and pushed on `feat/causal-viability-replication` (through 719ea52).
4. Never cut: Table III, Table IIIb, Section 6, or Section 7. The honest-reporting sections are load-bearing for the paper's credibility and for the contract.

References budget: 22 to 28 entries. The litmap supplies 14 verified neighbors with DOIs; the rest are Isaac Lab, Go2 platform, block bootstrap, and pre-registration methodology citations.

---

## Day-by-day schedule, 2026-08-24 to 2026-09-15 (22 days)

Ordered so that every blocking dependency (uncommitted code, the pre-onset characterization, the SafeRecovery flag) is hit in the first third, leaving the last third for cuts, review, and a genuine buffer.

| Day | Date | Work | Gate or dependency |
|---|---|---|---|
| 1 | Mon Aug 24 | Lock this outline. DONE 2026-08-24: page limit verified at 8 pages including references, deadline Sep 15 2026 11:59 PST, notification Jan 31 2027, all from the official CFP. Still owed: pull the IEEEtran template ICRA 2027 uses. Set up the IEEEtran skeleton and the bib file with the 14 litmap entries. | Page-limit assumption resolved before any layout |
| 2 | Tue Aug 25 | Commit the study code and the `causal_viability_replication` tree. Run the public-push guard. Confirm the result regenerates from a clean checkout. | P0 blocker in vault `tasks.md`. Appendix and any repo link depend on this |
| 3 | Wed Aug 26 | Write `scripts/paper_figures.py`. Produce Figure 1 from `combined_summary.json`. Independently recompute Table II from the JSON and diff against `result_table.md`. | Any mismatch here is a stop-work event |
| 4 | Thu Aug 27 | Draft Section 4 (Study Design) and Table I. This is the most artifact-bound section and the hardest to fake. | Needs Day 2 hashes |
| 5 | Fri Aug 28 | Draft Section 5 (Results) and Table III. | Needs Figure 1 and Table II |
| 6 | Sat Aug 29 | Close the SafeRecovery flag: Wayne State Springer access via GlobalProtect, read `10.1007/978-981-92-3381-6_42`. Draft Section 2 with the verdict folded in. | DECISION GATE A: if its four axes include an oracle arm or engagement-matched sham, stop and route to paper-contribution-locker |
| 7 | Sun Aug 30 | Draft Section 6 (Threats to Validity) against whichever branch the pre-onset characterization returned. | DECISION GATE B: branch (b), a real leak, means no ICRA submission in this form. Do not draft branch (a) as if it is settled |
| 8 | Mon Aug 31 | Draft Section 3 (Apparatus). Produce Figure 2 as a real vector diagram. | |
| 9 | Tue Sep 1 | Draft Section 1 (Introduction). Produce Figure 3 if budget allows. | |
| 10 | Wed Sep 2 | Draft Sections 7 and 8. First full assembly. Measure the real page count. | |
| 11 | Thu Sep 3 | Execute the pre-committed cut order until the body fits. Write the abstract and title last, from the assembled paper. | DECISION GATE C: go or no-go on ICRA versus the workshop fallback, decided on the assembled draft not on hope |
| 12 | Fri Sep 4 | Skeptical internal review round: `/critique` plus codex, read-only. Every number traced back to its artifact file. | |
| 13 | Sat Sep 5 | Fix round 1. | |
| 14 | Sun Sep 6 | Claim audit: read the contract line by line against the draft. Confirm no sentence exceeds the boundary and no excluded claim reappeared. Grep for em dashes and for internal codenames. | |
| 15 | Mon Sep 7 | Send to the external reader (Hisham) with a specific ask: attack the Section 6 noise-floor argument. | |
| 16 | Tue Sep 8 | RESERVED. If Gate B went to branch (b), this is day 1 of the design fix and re-run. Otherwise: figure polish, caption pass. | |
| 17 | Wed Sep 9 | RESERVED, same. Otherwise: reference completeness, every DOI resolved live. | |
| 18 | Thu Sep 10 | Incorporate external feedback. | |
| 19 | Fri Sep 11 | Final prose pass. Captions finalized (paper-figure-caption-editor, not this agent). | |
| 20 | Sat Sep 12 | Format compliance: IEEE PDF eXpress, page count, font embedding, anonymization if ICRA 2027 requires it. | |
| 21 | Sun Sep 13 | Buffer. No new content. | |
| 22 | Mon Sep 14 | Submit. One full day before the deadline. | |
| - | Tue Sep 15 | Deadline 23:59 PST. Reserved for a portal failure only. Not a working day. | |

Slack accounting: 2 reserved days (16, 17) plus 1 buffer day (21) plus the 1 day between submission and deadline. That is 4 days of real slack against 3 named risks (SafeRecovery collision, pre-onset leak, page overflow). If two of the three fire, the workshop fallback in the contract is the correct outcome, decided at Gate C.

---

## Contract-realization check

- Contribution sentence realized by: Section 4 (builds the instrument that makes the claim causal), Section 5.1 (the sign flip at an oracle detector, which is the "detector quality does not predict intervention benefit" half), and Section 5.3 (the recovery-authority mechanism, which is the "what predicts it" half). Sections 1, 2, and 8 frame; Sections 6 and 7 fence.
- Boundary check: PASS with three flags, none of which is a breach as skeletoned.
  1. Section 5.4 uses the dose-matched sham null (+0.004 [-0.023, +0.033]) and the detector AUROC range (0.873 to 1.000), both of which come from `closed_loop_walk` and `results_stand`, NOT from the contract's named evidence base (the replication). The contract's claim 3 explicitly names dose-matched sham arms as part of the contribution, so this is inside the claim boundary, but it widens the evidence base beyond the sentence "Evidence base = the pre-registered causal-viability replication". FLAGGED for paper-contribution-locker to confirm the evidence base admits these two auxiliary studies. If it does not, 5.4 is cut and Section 5 loses one paragraph, which the page budget welcomes. The outline does not resolve this unilaterally.
  2. Section 5.4 must state the monitor-timing result as a null and never as a benefit. The contract's exclusion list withdraws "any monitor-timing benefit". Reporting the null is what makes the contribution sentence true; claiming the null as a win would breach. The drafter must be held to this wording.
  3. Section 6 branch (b) is a live path to "this paper cannot be submitted as skeletoned". That is not a boundary breach; it is the boundary doing its job. The outline pre-commits to routing that outcome to a re-run or a re-lock rather than to softer prose.
- No section carries a hardware claim, an effect below 2 pp, a universality claim, or an unqualified "the shield makes the robot safer". Grep targets for the Day 14 audit: "safer", "in general", "hardware", "always", "robust to".
