"""Dependency-inverted scenario reset hook for fresh on-policy PPO rollouts."""

from __future__ import annotations

import math

from phoenix.replay.state_adapter import (
    VelocityCommandAdapter,
    resolve_command_hold,
    restore_state,
)


class FrictionScenarioAdapter:
    """Exact per-environment robot-shape friction, with reset isolation.

    These are robot material coefficients, not inferred surface friction. The
    terrain material and PhysX combine modes must be recorded by the experiment.
    Unsupported search dimensions are errors, never ignored. Supports the
    torch PhysX API and the newer Warp root_view API explicitly.
    """

    supported = frozenset({"static_friction", "dynamic_friction"})

    def __init__(self, env):
        self.env = env
        robot = env.scene["robot"]
        self.view = getattr(robot, "root_physx_view", None)
        self.warp = self.view is None
        if self.warp:
            self.view = getattr(robot, "root_view", None)
        if self.view is None:
            raise RuntimeError("No supported material physics view")
        self.original = {}

    def _get(self):
        data = self.view.get_material_properties()
        if self.warp:
            import warp as wp

            data = wp.to_torch(data)
        return data.clone()

    def _set(self, values, ids):
        import torch

        indices = torch.as_tensor(ids, dtype=torch.int32, device="cpu")
        if self.warp:
            import warp as wp

            self.view.set_material_properties(
                wp.from_torch(values, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
            )
        else:
            self.view.set_material_properties(values, indices)

    def apply(self, env_id, parameters):
        unknown = set(parameters) - self.supported
        if unknown:
            raise ValueError(f"Unsupported scenario dimensions: {sorted(unknown)}")
        for name, value in parameters.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}: {value}")
        values = self._get()
        if env_id not in self.original:
            self.original[env_id] = values[env_id].clone()
        for name, column in (("static_friction", 0), ("dynamic_friction", 1)):
            if name in parameters:
                values[env_id, :, column] = parameters[name]
        if (values[env_id, :, 1] > values[env_id, :, 0]).any():
            raise ValueError("dynamic friction must not exceed static friction")
        self._set(values, [env_id])
        readback = self._get()[env_id]
        import torch

        if not torch.allclose(readback, values[env_id], atol=1e-6, rtol=0):
            raise RuntimeError("Scenario friction readback does not match requested coefficients")
        return {"applied_parameters": dict(parameters), "material_scope": "robot_shapes"}

    def reset(self, env_ids):
        ids = [int(i) for i in env_ids if int(i) in self.original]
        if ids:
            values = self._get()
            for i in ids:
                values[i] = self.original.pop(i)
            self._set(values, ids)


def install_scenario_reset(
    env,
    sample,
    resolve_state,
    apply_parameters,
    *,
    reset_parameters,
    command_name="base_velocity",
    command_policy="match_control_process",
    command_hold_seconds=None,
    require_exact_replay=False,
    on_reset=None,
):
    """Install callbacks without importing any Ashfall research abstractions.

    sample(n, iteration) returns Scenario objects or None for nominal resets.
    resolve_state(scenario) returns (InitialState, metadata). All non-nominal
    scenarios must be marked split='train'; heldout and validation fail closed.
    Parameter restoration runs BEFORE the normal reset's domain randomization.
    The returned control's iteration is updated by the training orchestration.

    ``command_policy`` defaults to ``match_control_process``: the scenario's
    command VALUE is written but the reset's own resample clock is left alone,
    so a scenario env and a nominal env run the same velocity-command process
    and differ only in the thing under study. Controller history travels with
    the resolved state when the source carries it, and the per-reset telemetry
    reports the replay fidelity that was actually achieved.
    """
    target = getattr(env, "unwrapped", env)
    adapter = VelocityCommandAdapter(target, command_name)
    hold, command_telemetry = resolve_command_hold(
        command_policy, time_before_onset_seconds=None, hold_seconds=command_hold_seconds
    )
    original = target._reset_idx
    control = {"iteration": 0, "policy_id": None, "scenario_by_env": {}}

    def reset(env_ids):
        reset_parameters(env_ids)
        original(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        selections = sample(len(env_ids), control["iteration"])
        if len(selections) != len(env_ids):
            raise ValueError("Scenario sampler returned wrong assignment count")
        for local, scenario in enumerate(selections):
            env_id = int(env_ids[local])
            control["scenario_by_env"].pop(env_id, None)
            if scenario is None:
                continue
            if scenario.split != "train":
                raise ValueError(
                    f"Training leakage rejected: {scenario.scenario_id} split={scenario.split}"
                )
            state, metadata = resolve_state(scenario)
            applied = apply_parameters(env_id, scenario.parameters)
            restored = restore_state(
                target,
                state,
                env_id,
                command_adapter=adapter,
                command_hold_seconds=hold,
                command_telemetry=command_telemetry,
                controller_history=getattr(state, "controller_history", None),
                require_exact_replay=require_exact_replay,
            )
            control["scenario_by_env"][env_id] = scenario.scenario_id
            if on_reset is not None:
                on_reset(
                    {**metadata, **(applied or {}), **restored, "scenario_id": scenario.scenario_id}
                )

    target._reset_idx = reset
    return control


def learn_with_frontier_updates(runner, *, total_iterations, update_interval, reestimate, control):
    """Chunk fresh PPO iterations and update frontier using a separate evaluator.

    reestimate(runner, completed_iterations) must evaluate the current policy on
    training-only scenarios in a separate environment. It updates the external
    sampler and returns an identity for the evaluated current policy. No old
    trajectories enter the runner. Resetting the training environment after the
    callback prevents stale simulator state from being consumed by PPO.
    """
    if total_iterations < 1 or update_interval < 1:
        raise ValueError("Training iterations and update interval must be positive")
    completed = 0
    while completed < total_iterations:
        control["iteration"] = completed
        control["policy_id"] = reestimate(runner, completed)
        runner.env.reset()
        count = min(update_interval, total_iterations - completed)
        runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
        completed += count
    return completed
