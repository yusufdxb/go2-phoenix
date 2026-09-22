"""A one-joint model of why a nominal policy is brittle to actuator gain loss.

This is the explanatory model for Phoenix, not evidence about the GO2. It has the
same structure as the GO2 deploy path and nothing else:

* the motor driver runs a PD law ``tau = s * kp * (u - q) - s * kd * dq`` where
  ``s`` is the actuator-authority factor (1 nominal, <1 degraded). On the GO2 the
  firmware runs this law with the ``kp``/``kd`` the bridge sends, and in Isaac Lab
  the explicit ``DCMotor`` actuator runs the same law with ``stiffness``/``damping``.
  Scaling ``kp``/``kd`` of one joint on the robot and scaling that joint's
  stiffness/damping in sim are therefore the same parameter;
* a gravity-like load ``tau_g = m g l sin(q)`` that the PD law must hold;
* a policy that updates its target once per control period (50 Hz) from a
  one-period-old joint measurement, with a fixed parameter vector
  ``theta = (b, k)``: ``u = q_ref + b + k * (q_ref - q_meas)``.
  ``b`` is a learned feed-forward offset (the sag compensation a stand policy
  learns) and ``k`` a proportional correction. The policy cannot identify ``s``:
  it has no memory, as the deployed Phoenix actor has no history input.

Training a policy of that class under a distribution ``p(s)`` means choosing
``theta`` to minimise ``E_{s~p}[J(theta, s)]``. Because the class cannot adapt,
the optimum depends on ``p`` and a single ``theta`` cannot be optimal at every
``s``: a nominal distribution gives a policy whose sag compensation is wrong at
``s=0.6``; a broad distribution trades nominal and degraded cost against each
other; a distribution concentrated near the actual ``s`` is best there and pays at
``s=1``. That last cost is why Phoenix gates every candidate on nominal
performance. None of this says anything about policies that identify ``s`` online
(RMA-style), which is the comparison the paper must make on the robot.

The residual estimator :func:`gain_ratio_from_errors` is the one the live monitor
uses: under the same commanded motion and load, the steady tracking error of a
linear PD loop scales as ``1/s``, so ``s_hat = rms(e_nominal) / rms(e_now)``.
The model lets the tests check where that approximation holds and where it breaks
(it is biased once the policy compensates the error it is measuring).

Pure numpy, deterministic given a seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointParams:
    """Defaults are GO2-thigh-like orders of magnitude, not identified values."""

    inertia: float = 0.06  # kg m^2, reflected link inertia about the joint
    load: float = 3.0  # N m, peak gravity torque m*g*l
    kp: float = 25.0  # N m / rad, the Phoenix deploy kp
    kd: float = 0.5  # N m s / rad, the Phoenix deploy kd
    effort_limit: float = 23.5  # N m, GO2 DCMotor saturation in Isaac Lab
    control_dt: float = 0.02  # s, 50 Hz policy / bridge period
    physics_dt: float = 0.001  # s
    q_mean: float = 0.8  # rad, reference posture
    amplitude: float = 0.25  # rad, reference swing (a gait-like periodic command)
    freq_hz: float = 1.5
    duration_s: float = 4.0
    sensor_noise: float = 0.002  # rad


DEFAULT_JOINT = JointParams()


def rollout(
    theta: tuple[float, float],
    s: float,
    p: JointParams = DEFAULT_JOINT,
    seed: int = 0,
    amplitude: float | None = None,
) -> dict[str, np.ndarray]:
    """Simulate one episode. Returns time series of reference, target, position."""
    rng = np.random.default_rng(seed)
    b, k = float(theta[0]), float(theta[1])
    amp = p.amplitude if amplitude is None else float(amplitude)
    n_ctrl = int(round(p.duration_s / p.control_dt))
    n_sub = int(round(p.control_dt / p.physics_dt))
    q, dq = p.q_mean, 0.0
    q_meas_prev = q
    t_ctrl = np.arange(n_ctrl) * p.control_dt
    q_ref = p.q_mean + amp * np.sin(2.0 * np.pi * p.freq_hz * t_ctrl)
    u_hist = np.empty(n_ctrl)
    q_hist = np.empty(n_ctrl)
    for i in range(n_ctrl):
        # the policy acts on the measurement taken one control period ago
        u = q_ref[i] + b + k * (q_ref[i] - q_meas_prev)
        u_hist[i] = u
        for _ in range(n_sub):
            tau = s * p.kp * (u - q) - s * p.kd * dq
            tau = float(np.clip(tau, -p.effort_limit, p.effort_limit))
            ddq = (tau - p.load * np.sin(q)) / p.inertia
            dq += ddq * p.physics_dt
            q += dq * p.physics_dt
        q_hist[i] = q
        q_meas_prev = q + rng.normal(0.0, p.sensor_noise)
        if not np.isfinite(q) or abs(q) > 10.0:
            q_hist[i:] = np.nan
            u_hist[i:] = np.nan
            break
    return {"t": t_ctrl, "q_ref": q_ref, "u": u_hist, "q": q_hist}


def cost(
    theta: tuple[float, float],
    s: float,
    p: JointParams = DEFAULT_JOINT,
    seed: int = 0,
    effort_weight: float = 0.02,
) -> float:
    """Tracking cost plus a target-rate penalty (the action-rate term of the real reward).

    A diverged rollout costs a large constant, so the optimiser never prefers it.
    """
    r = rollout(theta, s, p, seed)
    if np.isnan(r["q"]).any():
        return 1e3
    burn = int(0.5 / p.control_dt)  # skip the first 0.5 s transient
    e = r["q_ref"][burn:] - r["q"][burn:]
    du = np.diff(r["u"][burn:])
    return float(np.mean(e**2) + effort_weight * np.mean(du**2))


def expected_cost(
    theta: tuple[float, float],
    s_samples: np.ndarray,
    p: JointParams = DEFAULT_JOINT,
) -> float:
    return float(np.mean([cost(theta, float(s), p, seed=i) for i, s in enumerate(s_samples)]))


def fit_policy(
    s_samples: np.ndarray,
    p: JointParams = DEFAULT_JOINT,
    b_grid: np.ndarray | None = None,
    k_grid: np.ndarray | None = None,
) -> tuple[tuple[float, float], float]:
    """Exhaustive grid search for ``theta`` minimising the expected cost over ``s_samples``.

    A grid rather than a gradient method keeps the result deterministic and easy to
    audit; the policy class has two parameters.
    """
    b_grid = np.linspace(0.0, 0.25, 26) if b_grid is None else b_grid
    k_grid = np.linspace(0.0, 1.5, 16) if k_grid is None else k_grid
    best: tuple[tuple[float, float], float] = ((0.0, 0.0), float("inf"))
    for b in b_grid:
        for k in k_grid:
            c = expected_cost((float(b), float(k)), s_samples, p)
            if c < best[1]:
                best = ((float(b), float(k)), c)
    return best


def tracking_errors(
    theta: tuple[float, float], s: float, p: JointParams = DEFAULT_JOINT, seed: int = 0
) -> np.ndarray:
    """Target-vs-measured error ``u[i-1] - q[i]`` (the monitor's raw residual)."""
    r = rollout(theta, s, p, seed)
    burn = int(0.5 / p.control_dt)
    err: np.ndarray = r["u"][burn:-1] - r["q"][burn + 1 :]
    return err


def gain_ratio_from_errors(e_nominal: np.ndarray, e_now: np.ndarray) -> float:
    """``s_hat = rms(e_nominal) / rms(e_now)``, the monitor's authority-ratio estimate.

    Valid when both windows see the same commanded motion and load, the loop is in
    its linear (unsaturated) range, and the policy does not itself compensate the
    change. Tests quantify the bias when that last condition fails.
    """
    rn = float(np.sqrt(np.mean(np.square(e_nominal))))
    rc = float(np.sqrt(np.mean(np.square(e_now))))
    if rc <= 0.0:
        return float("nan")
    return rn / rc


#: Distributions compared in the toy study. Each maps an observed s_hat to samples.
Distribution = Callable[[float, np.random.Generator, int], np.ndarray]


def nominal_dist(_s_hat: float, rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.95, 1.05, n)


def broad_dist(_s_hat: float, rng: np.random.Generator, n: int) -> np.ndarray:
    # Wider than the GO2 training DR ([0.85, 1.15]) so it covers the degraded
    # value: a fair "randomise broadly enough" baseline.
    return rng.uniform(0.4, 1.2, n)


def targeted_dist(s_hat: float, rng: np.random.Generator, n: int, half_width: float = 0.1):
    return rng.uniform(max(0.05, s_hat - half_width), min(1.0, s_hat + half_width), n)


def anchored_dist(s_hat: float, rng: np.random.Generator, n: int, nominal_fraction: float = 0.5):
    """Phoenix's design: the targeted range mixed with nominal, so the nominal gate can pass."""
    n_nom = int(round(nominal_fraction * n))
    return np.concatenate([nominal_dist(1.0, rng, n_nom), targeted_dist(s_hat, rng, n - n_nom)])


def run_study(
    s_true: float = 0.6,
    s_eval: tuple[float, ...] = (0.5, 0.6, 0.7, 1.0),
    n_train: int = 8,
    seed: int = 0,
    p: JointParams = DEFAULT_JOINT,
) -> dict[str, object]:
    """Fit one policy per distribution and evaluate each at every ``s_eval``.

    The targeted distribution is centred on the monitor's ESTIMATE of ``s_true``
    computed from the nominal policy's own tracking errors, not on ``s_true``, so
    estimator error propagates into the result.
    """
    rng = np.random.default_rng(seed)
    theta_nom, _ = fit_policy(nominal_dist(1.0, rng, n_train), p)
    e_nom = tracking_errors(theta_nom, 1.0, p, seed=101)
    e_deg = tracking_errors(theta_nom, s_true, p, seed=102)
    s_hat = gain_ratio_from_errors(e_nom, e_deg)
    policies = {
        "nominal": theta_nom,
        "broad": fit_policy(broad_dist(s_hat, rng, n_train), p)[0],
        "targeted": fit_policy(targeted_dist(s_hat, rng, n_train), p)[0],
        "anchored": fit_policy(anchored_dist(s_hat, rng, n_train), p)[0],
    }
    table = {
        name: {
            f"{s:.2f}": float(np.mean([cost(theta, s, p, seed=1000 + j) for j in range(4)]))
            for s in s_eval
        }
        for name, theta in policies.items()
    }
    return {
        "s_true": s_true,
        "s_hat": s_hat,
        "policies": {k: list(v) for k, v in policies.items()},
        "cost": table,
    }


__all__ = [
    "JointParams",
    "anchored_dist",
    "broad_dist",
    "cost",
    "fit_policy",
    "gain_ratio_from_errors",
    "nominal_dist",
    "rollout",
    "run_study",
    "targeted_dist",
    "tracking_errors",
]
