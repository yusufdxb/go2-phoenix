"""phoenix.physics.joint_friction: measured values, hypotheses, training distributions."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.physics.joint_friction import FrictionDistribution, FrictionModel, joint_group, load_measured
from phoenix.velocity.contract import JOINT_ORDER
from phoenix.velocity.spec import (
    BROAD_FRICTION,
    HW_FRICTION,
    friction_distribution,
    spec_from_dict,
    vnext3_spec,
    vnext3broad_spec,
    vnext3hw_spec,
)


def _uniform(seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda shape: rng.random(shape)


def test_measured_values_match_probe_readme():
    m = load_measured()
    assert set(m) == set(JOINT_ORDER)
    assert m["RR_calf_joint"]["static"] == pytest.approx(0.962)
    assert m["FL_thigh_joint"]["static"] == pytest.approx(0.171)


def test_hypothesis_scales_measured_and_sets_viscous_by_group():
    fm = FrictionModel.hypothesis(1.25, {"calf": 0.05}, load_coeff=0.1)
    assert fm.coulomb_nm["RR_calf_joint"] == pytest.approx(1.25 * 0.962)
    assert fm.viscous_nms["FR_calf_joint"] == 0.05 and fm.viscous_nms["FR_hip_joint"] == 0.0
    assert fm.load_coeff == 0.1


def test_hardware_distribution_keeps_joint_identity_and_bounds():
    fd = FrictionDistribution("hardware", scale=(0.75, 1.6), jitter=0.15, load=(0.0, 0.1), viscous_calf=(0.0, 0.05))
    c, v, k = fd.sample(4096, JOINT_ORDER, _uniform())
    m = load_measured()
    base = np.array([m[n]["static"] for n in JOINT_ORDER])
    assert np.all(c >= base * 0.75 * 0.85 - 1e-9) and np.all(c <= base * 1.6 * 1.15 + 1e-9)
    # RR_calf (the stiffest measured joint) stays the largest on average
    assert int(np.argmax(c.mean(axis=0))) == JOINT_ORDER.index("RR_calf_joint")
    calf = [i for i, n in enumerate(JOINT_ORDER) if joint_group(n) == "calf"]
    hip = [i for i, n in enumerate(JOINT_ORDER) if joint_group(n) == "hip"]
    assert v[:, calf].max() <= 0.05 and v[:, hip].max() == 0.0
    assert 0.0 <= k.min() and k.max() <= 0.1


def test_broad_distribution_has_no_joint_identity():
    fd = FrictionDistribution("broad", broad_coulomb=(0.0, 1.6))
    c, _, _ = fd.sample(20000, JOINT_ORDER, _uniform(1))
    means = c.mean(axis=0)
    assert np.allclose(means, 0.8, atol=0.03)


def test_specs_round_trip_and_validate():
    for spec_fn, expected in ((vnext3hw_spec, HW_FRICTION), (vnext3broad_spec, BROAD_FRICTION)):
        spec = spec_fn()
        assert spec.validate() == []
        rt = spec_from_dict(spec.to_dict())
        fd = friction_distribution(rt.domain_randomization)
        assert fd is not None and fd.kind == expected["joint_friction_model"]
        assert fd.load == tuple(expected["joint_friction_load"])
    assert friction_distribution(vnext3_spec().domain_randomization) is None


def test_arms_differ_only_in_friction():
    a, b, c = vnext3_spec().to_dict(), vnext3hw_spec().to_dict(), vnext3broad_spec().to_dict()
    for d in (b, c):
        diff = {k for k in a if a[k] != d[k]}
        assert diff == {"domain_randomization"}
        dra, drd = a["domain_randomization"], d["domain_randomization"]
        changed = {k for k in dra if dra[k] != drd[k]}
        assert all(k.startswith("joint_friction") or k.startswith("joint_viscous") for k in changed)
