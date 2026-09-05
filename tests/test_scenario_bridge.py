"""Training leak rejection, exact friction writes, and nominal reset isolation."""

from types import SimpleNamespace

import pytest

from phoenix.adaptation.scenario_bridge import (
    FrictionScenarioAdapter,
    install_scenario_reset,
    learn_with_frontier_updates,
)
from phoenix.replay.trajectory_reader import load_initial_state
from tests.test_failure_seed_v2 import capsule
from tests.test_reset_bridge import _fake_env, _FakeRobot


def test_material_parameter_writes_are_per_env_and_reversible():
    torch = pytest.importorskip("torch")
    materials = torch.tensor([[[0.8, 0.7, 0.0]], [[0.9, 0.8, 0.0]]])

    class View:
        def get_material_properties(self):
            return materials.clone()

        def set_material_properties(self, data, indices):
            materials[indices.long()] = data[indices.long()]

    env = SimpleNamespace(scene={"robot": SimpleNamespace(root_physx_view=View())})
    adapter = FrictionScenarioAdapter(env)
    adapter.apply(1, {"static_friction": 0.2, "dynamic_friction": 0.1})
    assert materials[0, 0].tolist() == pytest.approx([0.8, 0.7, 0])
    assert materials[1, 0].tolist() == pytest.approx([0.2, 0.1, 0])
    adapter.reset([1])
    assert materials[1, 0].tolist() == pytest.approx([0.9, 0.8, 0])
    with pytest.raises(ValueError, match="Unsupported scenario"):
        adapter.apply(0, {"slope": 0.1})


def test_scenario_hook_rejects_heldout_before_writing(tmp_path):
    torch = pytest.importorskip("torch")
    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.zeros(1, 3), "cpu")
    scenario = SimpleNamespace(scenario_id="heldout", split="heldout", parameters={})
    writes = []
    install_scenario_reset(
        env,
        lambda n, iteration: [scenario],
        lambda s: (load_initial_state(capsule(tmp_path), 40), {}),
        lambda env_id, parameters: writes.append(parameters),
        reset_parameters=lambda ids: None,
    )
    with pytest.raises(ValueError, match="Training leakage"):
        target._reset_idx(torch.tensor([0]))
    assert not writes and not robot.root_pose_calls


def test_scenario_hook_applies_train_and_clears_previous_identity_on_nominal(tmp_path):
    torch = pytest.importorskip("torch")
    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.zeros(1, 3), "cpu")
    selections = [SimpleNamespace(scenario_id="train", split="train", parameters={})]
    restored = []
    control = install_scenario_reset(
        env,
        lambda n, iteration: list(selections),
        lambda s: (load_initial_state(capsule(tmp_path), 40), {}),
        lambda env_id, parameters: {},
        reset_parameters=lambda ids: restored.append(list(ids)),
    )
    target._reset_idx(torch.tensor([0]))
    assert control["scenario_by_env"] == {0: "train"}
    selections[0] = None
    target._reset_idx(torch.tensor([0]))
    assert control["scenario_by_env"] == {}
    assert len(restored) == 2


def test_frontier_refresh_keeps_fresh_ppo_rollouts():
    iterations, updates = [], []
    runner = SimpleNamespace(
        env=SimpleNamespace(reset=lambda: None), learn=lambda **kw: iterations.append(kw)
    )

    def reestimate(runner, step):
        updates.append(step)
        return str(step)

    control = {}
    assert (
        learn_with_frontier_updates(
            runner, total_iterations=5, update_interval=2, reestimate=reestimate, control=control
        )
        == 5
    )
    assert updates == [0, 2, 4]
    assert [entry["num_learning_iterations"] for entry in iterations] == [2, 2, 1]
    assert all(not entry["init_at_random_ep_len"] for entry in iterations)
