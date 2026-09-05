# Paper Outline: Phoenix reliability shield (ICRA 2027)

> STATUS 2026-09-05: study **v2 at n = 5 process seeds** is the evidence base. The v1
> tree (`reliability_eval/causal_viability_replication/`) is SUPERSEDED and no number from
> it may enter the draft; it carried a real treatment leak in `env.reset()`. The live
> artifacts are `reliability_eval/causal_viability_replication_v2/{registry_n5.json,
> combined_summary_n5.json, onset_residual_audit_n5.json}` plus
> `analysis/selection_bias/process_level.json`.
>
> **Every number in the body is rendered by `scripts/paper_numbers.py` into
> `paper/numbers.md`.** Paste from there. Do not retype: the one hand-typed figure in this
> paper (the pre-onset residual) was wrong by a factor of two for weeks.
>
> `process_04` and `process_05` were added under `analysis/PREREG_n5_extension.md`, which
> was committed with its kill criterion before either seed ran.

Owner artifact of paper-outline-architect. Written 2026-08-24. Governed by
`~/Documents/Obsidian Vault/Projects/go2-phoenix/contribution_contract.md` (LOCKED 2026-08-15).
Nothing in this skeleton may claim more than that file allows.

Status of inputs actually read this session (verified, not assumed):

- `contribution_contract.md` (vault) : read in full.
- `status.md`, `tasks.md` (vault) : read in full.
- `reliability_eval/causal_viability_replication_v2/{registry_n5.json, combined_summary_n5.json, onset_residual_audit_n5.json}` : the live evidence base (n = 5).
- `analysis/selection_bias/{RESULTS_n5.txt, process_level.json}` : selection-bias suite and process-level inference.
- `paper/numbers.md` : generated; the only place body numbers may be copied from.
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

Section 1 carries problem (runtime shields are deployed on learned locomotion policies on the assumption that a better failure detector buys a safer robot) and gap (nobody has separated detector quality from closed-loop benefit with an interventional design; the literature evaluates detectors observationally with AUROC tables). Section 2 establishes the gap against named neighbors and concedes the one sibling finding (Shojaei). Sections 3 and 4 build the instrument that closes the gap: an oracle detector plus dose-matched sham arms in a pre-registered 2x2 factorial. Section 5 carries the evidence: the sign of the shield's benefit flips with fault family at a perfect detector, interaction +28.62 pp [+26.01, +31.22] at the process level, reproducing in all 5 processes and all 20 process-cells. Section 6 carries the honest damage: the pre-registered pre-onset negative control passes the frozen gate 4/4, a post-reset onset residual remains, positively measured as within-tick coupling in the shared physics batch and bounded at +0.039 pp, the threshold predicate that identifies it now has 4 exceptions in 960 blocks, and the observation-fault effects are horizon-conditional. Section 7 carries scope (sim only, one policy family, one simulator, one fallback design). Section 8 restates the measurement, not a recommendation.

---

## Claim ledger for this outline

Derived verbatim from the contract's "May claim" list. Every claim lands in exactly one section.

| ID | Claim | Home section | Evidence artifact |
|---|---|---|---|
| C1 | The sign of the shield's closed-loop benefit flips with the fault family, not with detector quality; the contrast runs at an oracle detector so detection error cannot explain it | 5.1 | `combined_summary_n5.json` -> `pooled_cells`, `fault_by_treatment_interaction_obs_minus_motor`; `process_level.json` |
| C2 | Mechanism: a static stand fallback has no recovery authority under actuator degradation and real authority under observation corruption, where the learned policy is the corrupted component | 5.3 | `combined_summary_n5.json` -> `pooled_cells` fall rates; `REGIME_CHARACTERIZATION.md` for the mechanism statement |
| C3 | Method export: oracle detector plus dose-matched sham arms as the design that separates detection quality from closed-loop outcome | 4 (design), 5.4 (the null it produced) | `registry_n5.json`; `closed_loop_walk/results.json` for the dose-matched sham null |
| C4 | Direction reproduces in all 5 processes and survives leave-one-process-out in every cell; every cell, both fault families and the interaction exclude zero under PROCESS-level inference, not only the within-process block bootstrap | 5.2 | `combined_summary_n5.json` -> `process_effects`, `leave_one_process_out`; `process_level.json` |
| N1 | NEGATIVE, must be reported: the pre-onset control passes 4/4 at n = 5, but a post-reset onset residual remains (4 of 10,240 disturbed environment pairs), the onset-threshold predicate that identifies its mechanism holds in 18 of 20 arm pairs rather than exceptionlessly, and the observation-fault effects shrink by 1.6 to 1.8 pp on a common 300-tick horizon | 6 | `combined_summary_n5.json` -> `pooled_pre_onset_negative_control_cells`, `gate_checks`; `onset_residual_audit_n5.json`; `process_level.json` |

| C5 | Inferential honesty: the frozen block bootstrap is conditional on the process seeds run; every interval is therefore reported at BOTH the block and process levels, and the paper's headline rests on the process level | 4, 5.2, 6 | `process_level.json`, `analysis/selection_bias/process_level.py` |

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
  - P4: The result: at a perfect detector the shield's benefit changes sign with the fault family. Motor degradation, stand: -23.53 pp. Observation corruption, walk: +17.21 pp. Interaction +28.62 pp [+26.01, +31.22], process-level interval over 5 independent process seeds. All numbers from `paper/numbers.md`.
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
  - P4: Pre-registration and freezing. `study_id phoenix_causal_viability_replication_v2`, registry hash `995e2807...` (n=5; the frozen n=3 registry `bf1e18db...` is unchanged and rebuilds byte-identically), per-cell protocol hashes, per-arm trajectory SHA-256, 18 exploratory protocols excluded by path before analysis, 5 independent processes with distinct process seeds, 20 protocols, 960 independent blocks. `build_registry` rejects any reused protocol seed, block seed or scenario fingerprint, and requires all 20 protocols to pin one experimental source snapshot, so the extension is verifiably the same experiment.
  - P4a: The sample-size extension, stated as a design decision rather than discovered mid-results. The study registered 3 process seeds. It was extended to 5 under `analysis/PREREG_n5_extension.md`, committed with its kill criterion before either new seed ran, because one reported quantity (the leak-free stand_obs cell) crossed zero under process-level inference at n = 3 and no quantity of additional blocks could settle it. Say in the paper that the estimand, thresholds, eligibility rule, subset rule and outcome definition were unchanged, and that the extension was pre-registered, so a reviewer can check the ordering in the repository history rather than take our word for it.
  - P5: The pre-committed gate, stated before the results so the reader knows it was not chosen after seeing them: direction reproduces in all process cells, leave-one-process-out preserves direction, pooled fault intervals exclude zero, and pre-onset negative controls include zero. Say here, not in Section 6, that the fourth criterion failed.
  - P6: Analysis: block-paired differences, block bootstrap 95% intervals, sign convention (positive unshielded minus oracle means the fallback reduces post-onset falls).
- Figures/tables placed here:
  - Table I: pre-registration and accounting. Supports C3.

### 5. Results

- JOB (one sentence): Show the sign flip, show it is not an artifact of one process or one cell, and name the mechanism, all within the contract's boundary.
- Claims it carries: C1 (5.1), C4 and C5 (5.2), C2 (5.3), C3's null (5.4).
- SOURCE OF TRUTH: `reliability_eval/causal_viability_replication_v2/combined_summary_n5.json` (registered estimand and gate) and `analysis/selection_bias/process_level.json` (process-level inference). Both render into `paper/numbers.md` via `scripts/paper_numbers.py`. Every figure below is pasted from that file. Nothing here is retyped.
- Paragraph plan:
  - 5.1 P1: The four cells at the oracle detector, 160 disturbed blocks each, 5 process seeds. stand_motor -23.53 pp; walk_motor -6.60 pp; stand_obs +9.89 pp; walk_obs +17.21 pp. The shield harms in both motor cells and helps in both observation cells. Quote the PROCESS-level interval as the headline interval in the running text (Table II carries both levels): -23.53 [-27.22, -19.85]; -6.60 [-8.11, -5.09]; +9.89 [+8.89, +10.89]; +17.21 [+16.21, +18.20].
  - 5.1 P2: The interaction is the headline: observation minus motor is +28.62 pp [+26.01, +31.22] at the process level over 640 disturbed blocks. State explicitly that this contrast is between two arms that both had a perfect detector, therefore detection quality is held constant across the sign change.
  - 5.1 P3: Fall-rate context so the pp effects are readable. Pull the unshielded and oracle post-onset fall rates per cell from `combined_summary_n5.json`; do not carry forward any v1 or n=3 rate.
  - 5.2 P1: Robustness. Direction reproduces in all 5 processes in every cell, i.e. all 20 process-cells, and every leave-one-process-out estimate preserves sign. Per-process effects, pp: stand_motor [-19.64, -27.87, -23.70, -22.50, -23.97]; stand_obs [+9.04, +9.62, +10.18, +9.49, +11.13]; walk_motor [-5.49, -6.43, -5.92, -8.65, -6.52]; walk_obs [+16.50, +18.39, +17.68, +16.67, +16.79].
  - 5.2 P2 (C5, and this paragraph is not optional): State the inferential level explicitly. The frozen bootstrap resamples blocks within a process and never resamples processes, so on its own it answers "would this hold on more blocks from these seeds", not "would this hold on new seeds". We therefore report both levels everywhere, and the claims above rest on the process-level one. Say plainly that the study was extended from 3 to 5 process seeds under a pre-registration precisely because one quantity, the leak-free stand_obs cell, crossed zero at the process level at n = 3 and could not be settled with more blocks.
  - 5.2 P3: What the two extra seeds changed. Nothing in sign or gate outcome; the point estimates moved by at most 0.65 pp per cell; and the one interval that crossed zero at n = 3 no longer does. Also report the one thing that got WORSE: the onset-threshold predicate behind the Section 6 mechanism argument went from 12 of 12 pairs exceptionless to 18 of 20, with 4 mismatching blocks in 960. Do not omit this to keep the story clean.
  - 5.3 P1: Mechanism. Under motor degradation the fallback depends on the same weakened actuators, so freezing removes the active compensation the policy was providing; oracle-treated falls are failures after correct engagement, not trigger misses. Quote the treated-fall and missed-engagement counts from `combined_summary_n5.json`.
  - 5.3 P2: Under observation corruption the learned policy is the corrupted component and the body is intact, so a static pose is a safe attractor and the fallback has real authority.
  - 5.3 P3: The secondary outcome that supports the mechanism read: task completion moves the same way. Re-derive the rates at n = 5; do not carry the n=3 or v1 percentages.
  - 5.4 P1: The detector-quality side of the claim, stated as the null it is. A latent-OOD monitor at episode-level AUROC 0.87 to 1.00 buys +0.004 [-0.023, +0.033] over a dose-matched blind switcher in the positive regime. Frame as "no measurable benefit from detector timing", never as a benefit. NOTE: this arm comes from the earlier closed-loop study, not from the n = 5 replication; say so in the sentence rather than letting the reader assume one sample.
- Figures/tables placed here:
  - Table II: primary outcome, all four cells, BOTH inferential levels, plus the 5 per-process effects. Supports C1, C4, C5. Rendered from `paper/numbers.md` Table II.
  - Table IIb: fault-family pooling and the interaction, both levels. This is the level the registered gate criterion is defined on, so it is not optional. Supports C1.
  - Figure 3: per-process and leave-one-process-out robustness. Supports C4. FIRST TO CUT if the layout runs over 8 pages including references, because Table II already carries the per-process effect column.

### 6. Threats to Validity

- JOB (one sentence): Report the pre-onset negative control in full at n = 5, name the residual that remains, show it was measured rather than argued away, bound its influence on the primary result, and concede the two places the evidence is weaker than the earlier draft claimed.
- Claims it carries: N1, C5.
- SOURCE OF TRUTH: `combined_summary_n5.json`, `onset_residual_audit_n5.json`, `analysis/selection_bias/process_level.json`, all rendered into `paper/numbers.md`. Draft prose is in `paper/onset_residual_limitation.md`. Tables III and IIIb are pasted, never transcribed.
- Paragraph plan:
  - P1: What the control is. Pre-onset windows are identical-treatment by construction: the oracle has not engaged yet, so the true effect is exactly zero. Any non-zero estimate is residual imbalance or noise.
  - P2: What it returned at n = 5. stand_motor and stand_obs +0.000 pp [+0.000, +0.000]; walk_motor and walk_obs +0.039 pp, block bootstrap [+0.000, +0.117], process level [-0.069, +0.148]. The frozen gate passes all four checks. Note that the walking cells touch zero under the block bootstrap and straddle it properly under process-level inference, which is the honest reading.
  - P3: What is and is not aligned across the paired arms. The batched-block harness gives each block its own 16 environments and one lifetime, so no block can inherit a predecessor's simulator state. Across all 20 process-cell arm pairs, reset states and initial observations are bit-identical; onset observations are not. The v1 reset leak is closed; a smaller channel acting after reset is not.
  - P4: The residual is positively measured, not inferred by elimination. Prediction: if the channel is within-tick coupling through the single shared GPU physics batch, divergence must be an upward closed set in onset TICK, with one threshold per arm pair, and must not be ordered by block index, disturbance status, or environment index. **It holds in 18 of 20 arm pairs exactly and in 956 of 960 blocks; 4 blocks in 2 pairs, both in process_05, are exceptions.** Joint probability under the arbitrary-subset null on the order of 1e-212. State the exceptions in this paragraph, not in a footnote: the n = 3 draft called the rule exceptionless and that statement did not survive two more seeds. Same-paragraph caveat: the implied propagation delay is not a single constant and the per-pair brackets admit no common value.
  - P5: Magnitude, named exactly rather than characterised. Pre-onset fall status differs for **4 of 10,240 disturbed environment pairs**, the pairs the registered estimand uses, and 6 of 15,360 counting the nominal blocks the estimand never touches. Three disturbed blocks affected, none of them in the leak-free subset. The two new processes contributed zero additional discrepancies. The largest residual, +0.039 pp, is 0.6 percent of the smallest primary effect (walk_motor, -6.60 pp) and 0.2 percent of the largest (walk_obs, +17.21 pp). NOTE FOR THE WRITER: an earlier draft said 2 disturbed pairs. That was hand-typed and wrong. The count is now derived in `onset_residual.audit_replicate` and regression-tested.
  - P6: Contamination-free sensitivity analysis. The bit-identical blocks are a subset on which the arms are provably identical up to onset. Membership is defined by a MEASURED bit-identity, not by the fitted threshold, so the 4 predicate exceptions cannot admit a contaminated block to the clean set; the count of leak-free disturbed pairs carrying a pre-onset discrepancy is 0. Recomputing the registered estimand there: stand_motor -22.80 [-27.49, -18.10] (n=35); stand_obs +9.90 [+5.42, +14.38] (n=91); walk_motor -7.06 [-8.62, -5.51] (n=38); walk_obs +17.08 [+12.65, +21.52] (n=53), all process-level. **All four keep their sign and all four exclude zero at BOTH inferential levels.** Report as post hoc: the subsets are the early-onset blocks rather than a random sample, so this bounds the residual's influence without being an unbiased estimate.
  - P6a: Selection into the subset. Membership equals 1{onset <= per-pair threshold} in 956 of 960 blocks; onset is randomised at design time, so this is selection on a coarsening of a pre-treatment covariate and opens no collider path. Say explicitly that the 4 exceptions mean we no longer claim membership is EXACTLY deterministic in a pre-treatment covariate. The single remaining channel, effect modification by onset, is null in all four cells with implied bias of -1.43, -0.69, -0.31 and +/-0.51 pp against effects of -23.53, +9.89, -6.60 and +17.21 pp. Three supporting checks: subset-minus-complement crosses zero in all four cells (+0.72, +1.06, -0.56, -0.05 pp); a threshold sweep from q=115 to q=195 is flat; arm-label permutation on the subset gives p = 2e-4 in all four cells, the resolution floor at 5,000 permutations.
  - P6b: The horizon qualification, stated by us rather than left for a reviewer. The observation-fault effects are horizon-conditional. On a common 300-tick post-onset window, the longest every block can supply in full, stand_obs moves +9.89 to +8.28 pp and walk_obs +17.21 to +15.46 pp, while both motor cells move 0.08 pp. At the fault-family level, motor -15.07 to -15.15 and obs +13.55 to +11.87, both still excluding zero at the process level. No sign and no gate outcome changes, but the observation effect size is a function of how long the episode is watched.
  - P7: What we do not claim, and why we report rather than fix. We do not claim the harness is bit-exact. We claim the divergence enters after reset through a mechanism identified by a positive test with 4 exceptions in 960 blocks that we report, that its effect on the registered control is at most +0.039 pp with a process-level interval straddling zero, and that the primary effects reproduce on the blocks it provably did not reach. Eliminating it entirely requires one physics batch per block, a 48-fold increase in simulator launches, judged not worth the compute against a residual of this size.
- Figures/tables placed here:
  - Table III: pre-onset negative control, all four cells, BOTH inferential levels. Supports N1. Non-negotiable; the control is reported as a table, not a sentence in a limitations list.
  - Table IIIb: contamination-free sensitivity analysis, registered estimand on the bit-identical-block subset, all four cells plus the two family rows, both levels. Supports N1 and defends C1.
  - Table IV: common-horizon check at W = 300. Supports N1's horizon clause. Small; can be folded into Table IIIb's caption if the page budget bites.
- SUPERSEDED HISTORY (kept for traceability, do not draft from it): this section previously carried the v1 failing control (`gate_passed: false`), a noise-floor reading DELETED as unsourced, the real v1 treatment leak in `env.reset()` diagnosed in `NEGATIVE_CONTROL_ANALYSIS.md` (commit 179d7f5), and an n = 3 reading in which the leak-free stand_obs cell did not survive process-level inference (+9.32 [-2.88, +21.52]) and the onset predicate was described as exceptionless. None of those belong in the draft.

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
| Figure 1 | 1 | Forest plot: 4 cell effects on a signed axis with zero marked, interaction +28.62 [+26.01, +31.22] as a fifth row; the two motor cells left of zero, the two observation cells right of zero. Must draw BOTH intervals per row (block bootstrap and process level) or state in the caption which one is drawn | C1, C5 | `combined_summary_n5.json` keys `pooled_cells.*.{mean_difference,ci_low,ci_high}` and `fault_by_treatment_interaction_obs_minus_motor`; `process_level.json` for the process-level interval | TO PRODUCE. No plotting script exists. Needs a new `scripts/paper_figures.py` reading `paper/numbers.md`'s sources. |
| Figure 2 | 3 | Apparatus and arm diagram: policy, monitor tap, fallback blend, the two arms, the block pairing, and the episode timeline with the pre-onset window and the registered onset tick marked | C3 | Structural, from `src/phoenix/reliability/{runtime.py,arbiter.py,study.py}` plus `registry_n5.json` for the block accounting. No numeric claims on this figure. | TO PRODUCE. Must be a real vector diagram (Graphviz or SVG), never ASCII. |
| Figure 3 | 5.2 | Small multiples, one panel per cell: 5 per-process effects plus the 5 leave-one-process-out estimates with CIs, all on the same signed axis, showing sign preservation in 20 of 20 process-cells | C4 | `combined_summary_n5.json` keys `pooled_cells.*.process_effects` and `pooled_cells.*.leave_one_process_out.*` | TO PRODUCE. CONDITIONAL: first asset cut if the paper exceeds 8 pages including references. |
| Table I | 4 | Pre-registration and accounting: study_id, registry hash, 5 processes, 4 cells, 160 disturbed blocks per cell, 960 independent blocks, 20 independent protocols, 18 exploratory protocols excluded, 4 frozen gate criteria with pass/fail, and the n=3-to-n=5 pre-registration reference | C3 | `registry_n5.json` and `combined_summary_n5.json` key `gate_checks`, rendered via `paper/numbers.md` | TO PRODUCE (LaTeX table, pasted from `paper/numbers.md`). |
| Table II | 5.1 | Primary outcome: per cell, blocks, eligible pairs, unshielded and oracle fall rates, block-paired effect, BOTH the block-bootstrap and the process-level 95 percent interval, and the 5 per-process effects | C1, C4, C5 | `paper/numbers.md` Table II, generated by `scripts/paper_numbers.py` from `combined_summary_n5.json` + `process_level.json` | TO PRODUCE. Needs LaTeX conversion only; the recompute is the generator. |
| Table IIb | 5.1 | Fault-family pooling (motor, obs) and the obs-minus-motor interaction, both inferential levels. This is the level the registered gate criterion is defined on | C1 | `paper/numbers.md` Table IIb | TO PRODUCE. |
| Table III | 6 | Pre-onset negative control: per-cell effect and both intervals for all four cells | N1 | `paper/numbers.md` Table III | TO PRODUCE. |
| Table IIIb | 6 | Contamination-free (leak-free) subset: registered estimand on the bit-identical blocks, four cells plus two family rows, both intervals, subset sizes | N1, defends C1 | `paper/numbers.md` Table IIIb | TO PRODUCE. |
| Table IV | 6 | Common-horizon check at W = 300 ticks, four cells and two families | N1 | `paper/numbers.md` Table IV | TO PRODUCE. Foldable into a caption if the page budget bites. |

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
| 3 | Wed Aug 26 | Write `scripts/paper_figures.py`. Produce Figure 1 from `combined_summary_n5.json` + `process_level.json`. Tables come from `scripts/paper_numbers.py`, so the recompute is the generator; the check is that no body number exists that `paper/numbers.md` does not contain. | Any hand-typed body number is a stop-work event |
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
