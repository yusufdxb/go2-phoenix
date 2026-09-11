"""Item 7: a seeded reset must not hand a recorded state to fresh random physics.

The reset bridge restored the robot but not the thing that caused the failure:
friction, mass, motor degradation, latency, the push schedule. A capsule that
declares its environment parameters now has to have them applied through a
scenario adapter, or the run refuses to start; a capsule that declares none is
recorded as such rather than being allowed to look like a preserved cause.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from phoenix.adaptation.curriculum import FailureCurriculum, TrajectoryPool
from phoenix.adaptation.reset_bridge import install, resolve_seed
from tests.test_failure_seed_v2 import capsule


class _RecordingAdapter:
    supported = frozenset({"static_friction", "dynamic_friction"})

    def __init__(self):
        self.applied = []
        self.released = []

    def apply(self, env_id, parameters):
        self.applied.append((env_id, dict(parameters)))
        return {"applied_parameters": dict(parameters), "material_scope": "robot_shapes"}

    def reset(self, env_ids):
        self.released.append([int(i) for i in env_ids])


def _curriculum(path):
    return FailureCurriculum(TrajectoryPool([path]), failure_reset_fraction=1.0)


def _declared(tmp_path, **overrides):
    return capsule(
        tmp_path,
        environment_parameters={"static_friction": 0.35, "dynamic_friction": 0.3},
        **overrides,
    )


# -------------------- declaration reading (no torch) -----------------------


def test_resolve_seed_reports_the_declared_cause(tmp_path):
    record = resolve_seed(_declared(tmp_path, disturbances=[{"kind": "push", "vx": 1.0}]))
    assert record["declared_environment_parameters"] == {
        "static_friction": 0.35,
        "dynamic_friction": 0.3,
    }
    assert record["declared_disturbances"] == [{"kind": "push", "vx": 1.0}]
    assert record["environment_context_restored"] is False
    assert record["capsule_schema_version"] == "1.1"


def test_undeclared_cause_is_reported_as_undeclared(tmp_path):
    record = resolve_seed(capsule(tmp_path))
    assert record["declared_environment_parameters"] == {}
    assert record["declared_disturbances"] == []
    assert record["environment_context_restored"] is False


@pytest.mark.parametrize("bad", [{"static_friction": "high"}, {"static_friction": float("nan")}, []])
def test_malformed_environment_parameters_rejected(tmp_path, bad):
    path = capsule(tmp_path)
    data = json.loads(path.read_text())
    data["environment_parameters"] = bad
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="environment_parameters"):
        resolve_seed(path)


# -------------------- install-time refusals (no torch) ---------------------


def test_declared_cause_without_an_adapter_refuses_to_start(tmp_path):
    with pytest.raises(ValueError, match="no scenario_adapter"):
        install(object(), _curriculum(_declared(tmp_path)))


def test_adapter_that_cannot_apply_a_declared_dimension_refuses(tmp_path):
    path = capsule(tmp_path, environment_parameters={"trunk_mass_kg": 1.2})
    with pytest.raises(ValueError, match="cannot apply declared environment parameters"):
        install(object(), _curriculum(path), scenario_adapter=_RecordingAdapter())


def test_declared_disturbance_without_an_applier_refuses(tmp_path):
    path = capsule(tmp_path, disturbances=[{"kind": "push", "vx": 1.0}])
    with pytest.raises(ValueError, match="no disturbance_applier"):
        install(object(), _curriculum(path))


def test_require_all_rejects_a_source_with_no_recorded_cause(tmp_path):
    with pytest.raises(ValueError, match="declares no environment_parameters"):
        install(object(), _curriculum(capsule(tmp_path)), environment_policy="require_all")


def test_record_only_is_an_explicit_opt_out(tmp_path):
    # No adapter, a declared cause, and no exception: the run is allowed only
    # because the caller named the policy that says "applying nothing".
    env = SimpleNamespace(unwrapped=SimpleNamespace(_reset_idx=lambda ids: None))
    with pytest.raises(RuntimeError, match="Cannot restore command term"):
        install(env, _curriculum(_declared(tmp_path)), environment_policy="record_only")


def test_unknown_environment_policy_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown environment_policy"):
        install(object(), _curriculum(capsule(tmp_path)), environment_policy="whatever")


def test_record_only_with_an_adapter_is_a_contradiction(tmp_path):
    with pytest.raises(ValueError, match="contradicts it"):
        install(
            object(),
            _curriculum(_declared(tmp_path)),
            environment_policy="record_only",
            scenario_adapter=_RecordingAdapter(),
        )


# -------------------- applied at reset (torch) -----------------------------


def test_declared_cause_is_applied_and_released_around_the_reset(tmp_path):
    torch = pytest.importorskip("torch")
    from tests.test_reset_bridge import _fake_env, _FakeRobot

    robot = _FakeRobot()
    env, inner_calls, target = _fake_env(robot, torch.zeros(1, 3), "cpu")
    adapter = _RecordingAdapter()
    order = []
    original = target._reset_idx

    def tracked(env_ids):
        order.append("env_reset")
        original(env_ids)

    target._reset_idx = tracked
    adapter_reset = adapter.reset

    def tracked_release(env_ids):
        order.append("scenario_release")
        adapter_reset(env_ids)

    adapter.reset = tracked_release
    log = tmp_path / "reset.jsonl"
    install(env, _curriculum(_declared(tmp_path)), scenario_adapter=adapter, telemetry_path=log)
    target._reset_idx(torch.tensor([0]))

    assert order == ["scenario_release", "env_reset"]
    assert adapter.released == [[0]]
    assert adapter.applied == [(0, {"static_friction": 0.35, "dynamic_friction": 0.3})]
    record = json.loads(log.read_text())
    assert record["environment_context_restored"] is True
    assert record["applied_parameters"] == {"static_friction": 0.35, "dynamic_friction": 0.3}
    assert record["material_scope"] == "robot_shapes"
    assert record["environment_policy"] == "require_declared"
    assert len(inner_calls) == 1


def test_undeclared_cause_seeds_but_records_that_physics_was_not_restored(tmp_path):
    torch = pytest.importorskip("torch")
    from tests.test_reset_bridge import _fake_env, _FakeRobot

    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.zeros(1, 3), "cpu")
    log = tmp_path / "reset.jsonl"
    install(env, _curriculum(capsule(tmp_path)), telemetry_path=log)
    target._reset_idx(torch.tensor([0]))
    record = json.loads(log.read_text())
    assert record["environment_context_restored"] is False
    assert record["declared_environment_parameters"] == {}
