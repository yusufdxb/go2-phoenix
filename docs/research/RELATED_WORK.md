# Related work

Search of 2026-09-21/22 over arXiv, OpenAlex, Crossref and Semantic Scholar. Author
lists of the arXiv entries were re-checked against the arXiv abstract pages (this
corrected one first-author attribution). Every entry
below was retrieved from a primary record (arXiv page, DOI record or publisher page);
identifiers are given so each can be checked. "Abstract level" means only the abstract
was read; a claim about what such a paper does NOT do is weaker than one about what it
does. Semantic Scholar rate-limited several queries, so an absence below is an absence
in what was searched, not a proof.

Columns: **When** = online (inside a rollout) or between deployments. **Latent** =
estimates dynamics or a fault. **Retrains** = updates policy weights after deployment.
**Detects** = explicitly detects a hardware change. **Status** = whether Phoenix's claimed
novelty survives this work: SURVIVES, PARTIAL, COLLIDES.

## 1. Updating the simulator from real data, then retraining (the closest collisions)

| Work | When | Latent | Retrains | Detects | Status |
|---|---|---|---|---|---|
| SimOpt: Chebotar et al., "Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience", ICRA 2019, doi:10.1109/ICRA.2019.8793789 | between | Y | Y | N | COLLIDES on mechanism |
| DROPO: Tiboni, Arndt, Kyrki, "DROPO: Sim-to-real transfer with offline domain randomization", Robotics and Autonomous Systems 2023, doi:10.1016/j.robot.2023.104432, arXiv:2201.08434 | between, offline | Y | Y | N | COLLIDES on mechanism, PARTIAL on framing |
| BayesSim: Ramos, Possas, Fox, RSS 2019, doi:10.15607/RSS.2019.XV.029, arXiv:1906.01728 | between | Y (posterior) | downstream | N | COLLIDES on mechanism |
| ASAP: He et al., "Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills", RSS 2025, doi:10.15607/RSS.2025.XXI.066, arXiv:2502.01143 | between | delta-action model | Y | N | PARTIAL (abstract level) |
| GAT: Hanna, Stone, "Grounded Action Transformation for Robot Learning in Simulation", AAAI 2017, doi:10.1609/aaai.v31i1.11044 | between | action transform | Y | N | PARTIAL |
| Active Domain Randomization: Mehta et al., arXiv:1904.04762 (2019) | between | partial | Y | N | PARTIAL |
| TuneNet: arXiv:1907.11200 (2019) | one-shot | Y | downstream | N | SURVIVES |
| Du et al., "Auto-Tuned Sim-to-Real Transfer", ICRA 2021, doi:10.1109/ICRA48506.2021.9562091 | between | Y (from RGB) | Y | N | SURVIVES |
| Policy transfer with strategy optimization: Yu, Liu, Turk, arXiv:1810.05751 | between | Y (strategy search) | N | N | PARTIAL |

What these establish: *real data in, updated simulator distribution out, policy
retrained* is a known recipe. SimOpt interleaves real rollouts with training and fits the
distribution to whole-trajectory discrepancy; DROPO fits it offline from a fixed dataset,
the same between-deployments cadence Phoenix uses; ASAP runs deploy, collect, fine-tune on
a humanoid with a learned delta-action model. Phoenix differs in the signal (a per-joint
commanded-vs-measured response residual taken after the safety layer, on a deployed
legged policy), the trigger (a persistent, localised change, not continuous gap
minimisation), the direction (the distribution is narrowed to the measured change, not
fit to all of the reality gap) and the acceptance rule (degraded improvement AND nominal
non-inferiority). A reviewer will ask: why not DROPO with joint tracking error as the
discrepancy? The honest answer is that this is close to what Phoenix does for one joint;
the difference Phoenix must demonstrate is the localisation, the safety-layer accounting
and the gate, and whether narrowing beats broadening. ASAP's and DROPO's appendices were
not read; a per-joint localisation or nominal-regression check there would narrow the gap
further.

## 2. Online system identification and latent adaptation

| Work | When | Latent | Retrains | Detects | Status |
|---|---|---|---|---|---|
| UP-OSI: Yu, Tan, Liu, Turk, "Preparing for the Unknown: Learning a Universal Policy with Online System Identification", RSS 2017, doi:10.15607/RSS.2017.XIII.048 | online | Y | N | N | PARTIAL |
| RMA: Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots", RSS 2021, doi:10.15607/RSS.2021.XVII.011, arXiv:2107.04034 | online | Y | N | N | SURVIVES, but the key competitor |
| Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning quadrupedal locomotion over challenging terrain", Science Robotics 2020, doi:10.1126/scirobotics.abc5986 | online, implicit | Y | N | N | SURVIVES |
| Peng et al., "Learning Agile Robotic Locomotion Skills by Imitating Animals", RSS 2020, doi:10.15607/RSS.2020.XVI.064 | online | Y | N | N | SURVIVES |

RMA and UP-OSI adapt inside the rollout and never change the policy weights; they only
work for changes inside their training distribution. Phoenix retrains between deployments
around a change it measured. No online-adaptation arm is run in the Phoenix experiment, so no claim is made
against these methods; the conclusions are limited to policies without history input.

## 3. Legged sim-to-real, domain randomisation, actuator models

| Work | Relevance | Status |
|---|---|---|
| Tan et al., "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots", RSS 2018, doi:10.15607/RSS.2018.XIV.010 | broad dynamics randomisation, the baseline family | SURVIVES |
| Hwangbo et al., "Learning agile and dynamic motor skills for legged robots", Science Robotics 2019, doi:10.1126/scirobotics.aau5872 | learned actuator network fit once from real data | PARTIAL (static, before deployment) |
| Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep RL", CoRL 2021, arXiv:2109.11978 | training infrastructure | SURVIVES |
| Margolis, Agrawal, "Walk These Ways", arXiv:2212.03238 | behaviour multiplicity for generalisation | SURVIVES |

Hwangbo et al. is the closest actuator-fidelity precedent: a model of the actuator fit
from measured data before deployment. Phoenix measures a *change from* the nominal
actuator response during deployment.

## 4. Fault-tolerant legged locomotion

| Work | When | Latent | Retrains | Detects | Status |
|---|---|---|---|---|---|
| Gravina, Rossini, Rizzardo, Laurenzi, Tsagarakis, "Learning Fault-Tolerant Locomotion with Adaptive Gait Timing", arXiv:2608.07328 (2026, preprint) | none after deployment | Y (actor latent) | N | implicit | PARTIAL |
| Lee et al., "DreamFLEX: Learning Fault-Aware Quadrupedal Locomotion Controller for Anomaly Situation in Rough Terrains", ICRA 2025, doi:10.1109/ICRA55743.2025.11127805 | online | Y (fault vector) | N | Y (online estimate) | PARTIAL |
| FT-WBC, "Learning Fault-Tolerant Whole-Body Control for Legged Loco-Manipulation", arXiv:2606.24466 (2026, preprint) | online | Y | N | Y (fault estimator) | PARTIAL |
| Luo, Xiao, Lu, "FT-Net: Learning Failure Recovery and Fault-Tolerant Locomotion for Quadruped Robots", IEEE RA-L 2023, doi:10.1109/LRA.2023.3329766 | online | Y | N | implicit | PARTIAL |
| Kim, Shin, Kim, "Learning Quadrupedal Locomotion with Impaired Joints Using Random Joint Masking", ICRA 2024, doi:10.1109/ICRA57147.2024.10610088 | online | Y (joint state) | N | implicit | PARTIAL |
| Xu et al., "AcL: Action Learner for Fault-Tolerant Quadruped Locomotion Control", arXiv:2503.21401 (2025, preprint), Go2 hardware | online | Y | N | implicit | PARTIAL |

Gravina et al. train once over randomised per-joint torque-efficiency faults with a
severity curriculum and deploy once; that is Phoenix's broad baseline, already published
with hardware results. DreamFLEX and FT-WBC estimate a fault online and modulate the
policy in the same forward pass. None of these builds a new training distribution from a
fault observed after deployment, and none retrains. Phoenix cannot claim novelty for
training a policy that tolerates per-joint torque loss; that ground is occupied. Its
claim is the measure, narrow, retrain, gate cycle. Further works cited by Gravina et al.
(for example Okamoto et al. arXiv:2111.10005, Liu et al. IEEE CAI 2024, Fu et al. CoRL
2025, Hou et al. ICRA 2024, Zhang et al. RA-L 2025) were taken from its reference list
and not independently re-verified.

## 5. Damage recovery and self-modelling

| Work | Relevance | Status |
|---|---|---|
| Bongard, Zykov, Lipson, "Resilient Machines Through Continuous Self-Modeling", Science 2006, doi:10.1126/science.1133687 | a legged robot re-estimates its self-model after damage and re-plans: the conceptual ancestor of "notice the change, update the model, act differently" | PARTIAL (model-based, no RL, no gate) |
| Cully, Clune, Tarapore, Mouret, "Robots that can adapt like animals", Nature 2015, doi:10.1038/nature14422 | damage triggers a search over a precomputed behaviour map | SURVIVES |
| Chatzilygeroudis et al., "Reset-free Trial-and-Error Learning for Robot Damage Recovery", RAS 2018, doi:10.1016/j.robot.2017.11.010 | same lineage, online search | SURVIVES |

## 6. Execution fidelity: the safety layer changing the action

No paper was found whose subject is using the commanded-vs-executed gap *created by a
safety or execution layer* on a legged robot as a measurement to be separated from the
actuator response. This is the least covered point and also the one with the weakest
search behind it (rate-limited). Phoenix treats it as engineering hygiene that its
measurement depends on, not as a headline contribution.

## Verdict

The loop is not new. Phoenix's defensible contribution is narrow: a safety-layer-aware,
per-joint response residual on a deployed legged policy; a training distribution narrowed
to the measured change; a promotion gate on degraded improvement and nominal
non-inferiority; and a controlled degradation that is the same parameter on the robot
and in the simulator, which gives the loop a ground truth. It survives only if targeting
beats broad randomisation at matched compute. Before submission: read the ASAP and DROPO
appendices, and search specifically for per-joint actuator-residual-triggered retraining
on manipulators, which this pass did not cover.

Bibliography in BibTeX: `docs/research/references.bib`.
