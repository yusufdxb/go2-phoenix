"""Controller history: the part of a failure a single state row cannot carry.

A Phoenix GO2 episode is a delayed closed loop, not a memoryless one. Three
pieces of controller state persist across a control step and none of them is a
function of the robot's instantaneous pose:

* ``last_action``. The trained observation vector ends in the previous policy
  output (Isaac Lab ``velocity_env_cfg`` declares
  ``actions = ObsTerm(func=mdp.last_action)``), and ``ActionManager.reset``
  zeroes it. A reset that writes only pose, velocity and joints therefore hands
  the policy an observation that says "you have just been spawned", whatever the
  body is doing.
* The rate limiter in :mod:`phoenix.sim_env.rate_limited_action`. In the default
  ``measured_q`` clip mode it is stateless (it clips against the measured joint
  position, which the state row does restore). In ``prev_command`` mode it
  carries ``_prev_target``, the previously applied joint target.
* The delayed DC motor in :mod:`phoenix.sim_env.delayed_dc_motor`. Its
  ``DelayBuffer`` holds the last ``max_delay`` PHYSICS-step setpoints, and the
  per-environment lag is redrawn from ``[min_delay, max_delay]`` at every reset
  by design, because it is domain randomization. Neither the buffer contents at
  physics rate nor the lag that was in force during the recorded failure is
  recoverable from a control-rate trajectory log.

So this module restores what is genuinely restorable, and NAMES what is not,
in the reset telemetry. :data:`REPLAY_EXACT` is reported only when every
stateful component was reconstructed and verified by readback; anything else
reports :data:`REPLAY_APPROXIMATE` together with the list of components that
were re-initialised by the reset. A seed with no history at all reports
:data:`REPLAY_STATE_ONLY`. Nothing here silently approximates, and
``require_exact=True`` turns an unreachable exact replay into an exception
instead of a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Every stateful controller component was reconstructed and read back.
REPLAY_EXACT = "exact_state_and_controller_history"

#: Some controller state was reconstructed; the rest was re-initialised by the
#: reset and is named in ``controller_state_not_reconstructed``.
REPLAY_APPROXIMATE = "approximate_controller_history"

#: Only the kinematic state row was written. This is a seed, not a replay.
REPLAY_STATE_ONLY = "state_only_seed"

__all__ = [
    "ControllerHistory",
    "REPLAY_APPROXIMATE",
    "REPLAY_EXACT",
    "REPLAY_STATE_ONLY",
    "inspect_controller_state",
    "restore_controller_history",
]


@dataclass(frozen=True)
class ControllerHistory:
    """Applied control history ending at the seed row.

    ``actions`` is ordered oldest to newest and ``actions[-1]`` is the raw
    policy output applied at the seed row, so it is exactly what
    ``mdp.last_action`` would have returned on the following step.
    ``joint_targets`` (optional, same ordering) is the joint target actually
    applied after scaling and rate limiting; it is what the ``prev_command``
    rate limiter needs and what a control-rate trajectory Parquet does not
    carry.
    """

    actions: np.ndarray
    joint_targets: np.ndarray | None = None
    starts_at_episode_start: bool = False
    control_dt: float | None = None
    source_rows: tuple[int, int] = field(default=(0, 0))

    def __post_init__(self):
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise ValueError("Controller history actions must be a non-empty (rows, dim) array")
        if not np.isfinite(actions).all():
            raise ValueError("Controller history actions must be finite")
        object.__setattr__(self, "actions", actions)
        if self.joint_targets is not None:
            targets = np.asarray(self.joint_targets, dtype=np.float32)
            if targets.shape != actions.shape:
                raise ValueError("Joint target history must match the action history shape")
            if not np.isfinite(targets).all():
                raise ValueError("Joint target history must be finite")
            object.__setattr__(self, "joint_targets", targets)
        if self.control_dt is not None and (
            not np.isfinite(self.control_dt) or self.control_dt <= 0
        ):
            raise ValueError("control_dt must be finite and positive")
        rows = tuple(int(v) for v in self.source_rows)
        if len(rows) != 2 or rows[0] > rows[1] or rows[0] < 0:
            raise ValueError("source_rows must be an ordered, nonnegative (first, last) pair")
        object.__setattr__(self, "source_rows", rows)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def last_action(self) -> np.ndarray:
        return self.actions[-1]

    @property
    def previous_action(self) -> np.ndarray | None:
        return self.actions[-2] if len(self) >= 2 else None


def _action_terms(env) -> dict:
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return {}
    terms = getattr(manager, "_terms", None)
    if isinstance(terms, dict):
        return dict(terms)
    names = getattr(manager, "active_terms", None) or []
    get_term = getattr(manager, "get_term", None)
    if get_term is None:
        return {}
    return {name: get_term(name) for name in names}


def _actuators(env) -> dict:
    try:
        robot = env.scene["robot"]
    except (KeyError, TypeError, AttributeError):
        return {}
    actuators = getattr(robot, "actuators", None)
    return dict(actuators) if isinstance(actuators, dict) else {}


def _delay_steps(actuator) -> int:
    cfg = getattr(actuator, "cfg", None)
    delay = getattr(cfg, "max_delay", 0) if cfg is not None else 0
    try:
        return int(delay)
    except (TypeError, ValueError):
        return 0


def inspect_controller_state(env, history: ControllerHistory | None = None) -> dict:
    """Enumerate stateful controller components and whether history can restore them.

    Pure inspection: reads configuration and buffer presence, writes nothing.
    Returns ``{"restorable": [...], "not_reconstructed": [...]}`` with every
    entry naming a concrete component, so the reason an episode is not an exact
    replay is legible in the telemetry rather than implied.
    """
    restorable: list[str] = []
    missing: list[str] = []

    manager = getattr(env, "action_manager", None)
    if manager is None or getattr(manager, "_action", None) is None:
        missing.append("action_manager.action_buffers_unavailable")
    elif history is None:
        missing.append("action_manager.last_action_no_history")
    else:
        restorable.append("action_manager.action")
        if history.previous_action is None:
            missing.append("action_manager.prev_action_history_too_short")
        else:
            restorable.append("action_manager.prev_action")

    for name, term in _action_terms(env).items():
        cfg = getattr(term, "cfg", None)
        clip_mode = getattr(cfg, "clip_mode", None)
        if clip_mode != "prev_command":
            continue
        if history is not None and history.joint_targets is not None:
            restorable.append(f"rate_limited_action.{name}.prev_target")
        else:
            missing.append(f"rate_limited_action.{name}.prev_target_no_joint_target_history")

    for name, actuator in _actuators(env).items():
        if _delay_steps(actuator) > 0 or hasattr(actuator, "positions_delay_buffer"):
            # The lag itself is redrawn from [min_delay, max_delay] at every
            # reset, and the buffer holds physics-rate setpoints a control-rate
            # log never saw. This one is not reconstructable, only declarable.
            missing.append(f"delayed_actuator.{name}.delay_buffer_redrawn_at_reset")

    if history is not None and not history.starts_at_episode_start:
        missing.append("controller_history.window_does_not_start_at_episode_start")

    return {"restorable": sorted(restorable), "not_reconstructed": sorted(missing)}


def restore_controller_history(
    env,
    env_id: int,
    history: ControllerHistory,
    *,
    require_exact: bool = False,
) -> dict:
    """Write the reconstructable controller state for one environment.

    Returns telemetry naming what was restored, what was not, and the resulting
    replay fidelity. With ``require_exact=True`` an unreachable exact replay
    raises instead of being reported.
    """
    import torch

    if not isinstance(history, ControllerHistory):
        raise TypeError("controller_history must be a ControllerHistory")
    report = inspect_controller_state(env, history)
    restored: list[str] = []

    manager = getattr(env, "action_manager", None)
    action_buffer = getattr(manager, "_action", None) if manager is not None else None
    if action_buffer is not None:
        if action_buffer.shape[-1] != history.action_dim:
            raise ValueError(
                f"Recorded action width {history.action_dim} does not match the environment's "
                f"action dimension {int(action_buffer.shape[-1])}"
            )
        last = torch.as_tensor(
            history.last_action, dtype=action_buffer.dtype, device=action_buffer.device
        )
        action_buffer[env_id] = last
        restored.append("action_manager.action")
        previous = history.previous_action
        prev_buffer = getattr(manager, "_prev_action", None)
        if previous is not None and prev_buffer is not None:
            prev_buffer[env_id] = torch.as_tensor(
                previous, dtype=prev_buffer.dtype, device=prev_buffer.device
            )
            restored.append("action_manager.prev_action")
        readback = getattr(manager, "action", action_buffer)[env_id]
        if not torch.allclose(readback, last, atol=1e-6, rtol=0):
            raise RuntimeError("last_action restoration readback failed")

    if history.joint_targets is not None:
        target = history.joint_targets[-1]
        for name, term in _action_terms(env).items():
            cfg = getattr(term, "cfg", None)
            if getattr(cfg, "clip_mode", None) != "prev_command":
                continue
            buffer = getattr(term, "_prev_target", None)
            if buffer is None:
                offset = getattr(term, "_offset", None)
                if offset is None:
                    raise RuntimeError(
                        f"Rate limiter {name} exposes neither _prev_target nor _offset; "
                        "cannot restore its slew state"
                    )
                buffer = offset.detach().clone()
                term._prev_target = buffer
            if buffer.shape[-1] != target.shape[-1]:
                raise ValueError(
                    f"Recorded joint target width {int(target.shape[-1])} does not match "
                    f"rate limiter {name} width {int(buffer.shape[-1])}"
                )
            buffer[env_id] = torch.as_tensor(target, dtype=buffer.dtype, device=buffer.device)
            restored.append(f"rate_limited_action.{name}.prev_target")

    not_reconstructed = sorted(set(report["not_reconstructed"]))
    fidelity = REPLAY_EXACT if not not_reconstructed else REPLAY_APPROXIMATE
    if require_exact and not_reconstructed:
        raise RuntimeError(
            "Exact replay requested but these controller components cannot be reconstructed: "
            + ", ".join(not_reconstructed)
        )
    return {
        "replay_fidelity": fidelity,
        "controller_history_restored": sorted(set(restored)),
        "controller_state_not_reconstructed": not_reconstructed,
        "controller_history_rows": len(history),
        "controller_history_source_rows": list(history.source_rows),
        "controller_history_starts_at_episode_start": bool(history.starts_at_episode_start),
    }
