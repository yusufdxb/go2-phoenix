"""GO2 joint friction model shared by the Isaac Lab training env and the MuJoCo evaluator.

One definition, two simulators. Per joint ``j`` the friction torque opposing motion is

    Coulomb   F_j(t) = F0_j + k_load * |tau_j(t)|      (N m; static = dynamic)
    viscous   b_j * qd_j                               (N m s/rad)

``F0_j`` comes from the hung-robot probe (configs/physics/go2_joint_friction_2026-10-07.yaml)
times a hypothesis scale. ``k_load`` is the load-sensitivity HYPOTHESIS: gearbox friction
that grows with transmitted torque (efficiency-like). It is zero unless a hypothesis sets it.
``tau_j`` is the actuator torque applied at the previous physics step (MuJoCo) or control
step (Isaac): the friction coefficient lags its load by one update.

Simulator semantics differ and are checked, not assumed:

* PhysX (Isaac Sim 6.0): ``joint_friction_coeff`` / ``joint_dynamic_friction_coeff`` are an
  effort bound in N m (positive control results/friction_dr/isaac_probe.json),
  ``joint_viscous_friction_coeff`` is checked by scripts/isaac_friction_probe.py --viscous.
* MuJoCo: ``dof_frictionloss`` is a soft (regularised) Coulomb constraint in N m and
  ``dof_damping`` is exact viscous damping in N m s/rad.

Direction dependence is not modelled: the probe cannot separate it from gravity.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
MEASURED_FRICTION_YAML = REPO / "configs/physics/go2_joint_friction_2026-10-07.yaml"
JOINT_GROUPS = ("hip", "thigh", "calf")


def joint_group(name: str) -> str:
    for g in JOINT_GROUPS:
        if f"_{g}_" in name:
            return g
    raise ValueError(f"not a GO2 leg joint: {name!r}")


def load_measured(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """{joint: {"static": N m, "kinetic": N m}} from the measured-friction YAML."""
    import yaml

    blob = yaml.safe_load(Path(path or MEASURED_FRICTION_YAML).read_text())
    out = {}
    for name, v in blob["joints"].items():
        joint_group(name)
        out[name] = {"static": float(v["static"]), "kinetic": float(v["kinetic"])}
    if len(out) != 12:
        raise ValueError(f"expected 12 joints, got {sorted(out)}")
    return out


@dataclasses.dataclass(frozen=True)
class FrictionModel:
    """One deterministic physics hypothesis (used for evaluation and identification)."""

    coulomb_nm: Mapping[str, float]
    viscous_nms: Mapping[str, float]
    load_coeff: float = 0.0
    label: str = ""

    @classmethod
    def frictionless(cls) -> FrictionModel:
        names = load_measured().keys()
        return cls({n: 0.0 for n in names}, {n: 0.0 for n in names}, 0.0, "frictionless")

    @classmethod
    def hypothesis(
        cls,
        coulomb_scale: float = 1.0,
        viscous_by_group: Mapping[str, float] | None = None,
        load_coeff: float = 0.0,
        measured: Mapping[str, Mapping[str, float]] | None = None,
        label: str = "",
    ) -> FrictionModel:
        """Measured static friction x ``coulomb_scale``, viscous per joint group, load term."""
        meas = measured or load_measured()
        vg = dict(viscous_by_group or {})
        return cls(
            {n: coulomb_scale * float(m["static"]) for n, m in meas.items()},
            {n: float(vg.get(joint_group(n), 0.0)) for n in meas},
            float(load_coeff),
            label,
        )

    def coulomb(self, order: Sequence[str]) -> list[float]:
        return [float(self.coulomb_nm[n]) for n in order]

    def viscous(self, order: Sequence[str]) -> list[float]:
        return [float(self.viscous_nms[n]) for n in order]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "coulomb_nm": dict(self.coulomb_nm),
            "viscous_nms": dict(self.viscous_nms),
            "load_coeff": self.load_coeff,
        }


Range = tuple[float, float]


@dataclasses.dataclass(frozen=True)
class FrictionDistribution:
    """Per-env training distribution over the friction model.

    ``hardware``: Coulomb_j = measured_j * s * u_j with ONE env-wide scale s ~ U(scale) (the
    load/temperature hypothesis moves every joint together) and an independent per-joint
    jitter u_j ~ U(1 - jitter, 1 + jitter); joint identity is preserved (RR_calf stays the
    stiffest). ``broad``: Coulomb_j ~ U(broad_coulomb) independently for every joint, no
    joint identity (conventional DR). Both: viscous_j ~ U(viscous[group]), k_load ~ U(load).
    """

    kind: str  # "hardware" | "broad"
    scale: Range = (1.0, 1.0)
    jitter: float = 0.0
    broad_coulomb: Range = (0.0, 0.0)
    viscous_hip: Range = (0.0, 0.0)
    viscous_thigh: Range = (0.0, 0.0)
    viscous_calf: Range = (0.0, 0.0)
    load: Range = (0.0, 0.0)

    def validate(self) -> list[str]:
        p = []
        if self.kind not in ("hardware", "broad"):
            p.append(f"friction distribution kind {self.kind!r} not in (hardware, broad)")
        for name in ("scale", "broad_coulomb", "viscous_hip", "viscous_thigh", "viscous_calf", "load"):
            lo, hi = getattr(self, name)
            if not (0.0 <= lo <= hi):
                p.append(f"friction {name} must satisfy 0 <= lo <= hi, got {(lo, hi)}")
        if not 0.0 <= self.jitter < 1.0:
            p.append("friction jitter must be in [0, 1)")
        if self.broad_coulomb[1] > 5.0:
            p.append("broad Coulomb above 5 N m is outside any plausible GO2 joint")
        return p

    def viscous_range(self, group: str) -> Range:
        return {"hip": self.viscous_hip, "thigh": self.viscous_thigh, "calf": self.viscous_calf}[group]

    def sample(
        self,
        n: int,
        order: Sequence[str],
        uniform: Callable[[tuple[int, ...]], Any],
        measured: Mapping[str, Mapping[str, float]] | None = None,
    ) -> tuple[Any, Any, Any]:
        """(coulomb [n, J], viscous [n, J], load_coeff [n]) as arrays of ``uniform``'s type.

        ``uniform(shape)`` returns U[0, 1) samples: ``torch.rand`` (Isaac) or
        ``np.random.default_rng(s).random`` (MuJoCo, tests). Only elementwise arithmetic is
        used, so the same code serves both.
        """
        n_joints = len(order)
        lo = [self.viscous_range(joint_group(nm))[0] for nm in order]
        hi = [self.viscous_range(joint_group(nm))[1] for nm in order]
        uv = uniform((n, n_joints))
        viscous = uv * 0.0
        for j in range(n_joints):
            viscous[:, j] = lo[j] + (hi[j] - lo[j]) * uv[:, j]
        ul = uniform((n,))
        load = self.load[0] + (self.load[1] - self.load[0]) * ul
        if self.kind == "broad":
            uc = uniform((n, n_joints))
            coulomb = self.broad_coulomb[0] + (self.broad_coulomb[1] - self.broad_coulomb[0]) * uc
            return coulomb, viscous, load
        meas = measured or load_measured()
        base = [float(meas[nm]["static"]) for nm in order]
        us = uniform((n,))
        s = self.scale[0] + (self.scale[1] - self.scale[0]) * us
        uj = uniform((n, n_joints))
        coulomb = uj * 0.0
        for j in range(n_joints):
            coulomb[:, j] = base[j] * s * (1.0 - self.jitter + 2.0 * self.jitter * uj[:, j])
        return coulomb, viscous, load

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------- MuJoCo


#: MuJoCo's frictionloss is a soft constraint that lets a joint creep at
#: v ~ F * solref / inertia; with the default solref (0.02 s) the light calf creeps against
#: well under the set force at probe speeds. Two timesteps is the stiffest MuJoCo allows and
#: makes the constraint behave like Coulomb friction (scripts/verify_friction_physics.py).
STIFF_SOLREF_TIMESTEPS = 2.0


class MujocoFriction:
    """Applies a :class:`FrictionModel` to a MuJoCo model; call :meth:`update` every step."""

    def __init__(
        self,
        fm: FrictionModel,
        model: Any,
        dof_adr: Sequence[int],
        order: Sequence[str],
        stiff: bool = True,
    ) -> None:
        import numpy as np

        self._np = np
        self.model = model
        self.dof = np.asarray(dof_adr, dtype=int)
        self.base = np.asarray(fm.coulomb(order), dtype=float)
        self.k = float(fm.load_coeff)
        model.dof_frictionloss[self.dof] = self.base
        model.dof_damping[self.dof] = np.asarray(fm.viscous(order), dtype=float)
        if stiff:
            model.dof_solref[self.dof, 0] = STIFF_SOLREF_TIMESTEPS * model.opt.timestep

    def update(self, tau_applied: Any) -> None:
        if self.k:
            self.model.dof_frictionloss[self.dof] = self.base + self.k * self._np.abs(tau_applied)


__all__ = [
    "MEASURED_FRICTION_YAML",
    "STIFF_SOLREF_TIMESTEPS",
    "FrictionDistribution",
    "FrictionModel",
    "MujocoFriction",
    "joint_group",
    "load_measured",
]
