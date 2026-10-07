"""Joint-friction DR spec (vnext3f): off by default, one-change over vnext3, round-trips."""

from __future__ import annotations

import dataclasses
import json

from phoenix.velocity.spec import spec_from_dict, vnext3_spec, vnext3f_spec


def test_friction_off_by_default_and_old_specs_rebuild_without_it():
    base = vnext3_spec()
    assert base.domain_randomization.joint_friction_enabled is False
    blob = json.loads(base.to_json())
    for k in ("joint_friction_enabled", "joint_friction_hip_thigh_nm", "joint_friction_calf_nm"):
        blob["domain_randomization"].pop(k)
    rebuilt = spec_from_dict(blob)
    assert rebuilt.domain_randomization.joint_friction_enabled is False


def test_vnext3f_differs_from_vnext3_only_in_friction_dr():
    a, b = vnext3_spec(), vnext3f_spec()
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    dra, drb = da.pop("domain_randomization"), db.pop("domain_randomization")
    assert da == db
    diff = {k for k in dra if dra[k] != drb[k]}
    assert diff == {
        "joint_friction_enabled",
        "joint_friction_hip_thigh_nm",
        "joint_friction_calf_nm",
    }
    assert drb["joint_friction_hip_thigh_nm"] == (0.1, 0.6)
    assert drb["joint_friction_calf_nm"] == (0.3, 1.5)
    assert b.validate() == []


def test_vnext3f_round_trips_through_json():
    s = vnext3f_spec()
    assert spec_from_dict(json.loads(s.to_json())) == s


def test_invalid_friction_range_rejected():
    s = vnext3f_spec()
    bad = s.replace(
        domain_randomization=dataclasses.replace(
            s.domain_randomization, joint_friction_calf_nm=(1.0, 0.5)
        )
    )
    assert any("joint_friction_calf_nm" in p for p in bad.validate())
