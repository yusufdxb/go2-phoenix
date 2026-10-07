#!/usr/bin/env python3
"""LOCALHOST-ONLY end-to-end rehearsal of the FREE_FLOOR choreography (no robot, no Jetson).

It runs the REAL ``scripts/harness_preflight.sh FREE_FLOOR ...`` with the real bridge, policy node,
deadman node, passive STOP-path recorder, walk_command, analyzers and state files, against
``scripts/rehearsal_fake_go2.py`` (a synthetic /lowstate and a GO2 remote) and the archived
model_2750 bundle. Everything it records is flagged ``rehearsal`` and never counts as evidence.

What it can and cannot show. The fake robot is a STATIC plant (a fixed standing pose, zero velocity, a level IMU; the
policy loop is open because no simulated physics sits behind it), so the process lifecycle, logs, state transitions, analyzers, latches and failure
handling are exercised for real; the gait, the floor and the people are not. A pass here says the
software choreography works, not that the robot is safe to walk.

Every scenario starts from a clean environment built from scratch (``env -i`` plus the sourced ROS
setup), so nothing inherited from the caller's shell can leak into a run.

    scripts/free_floor_rehearsal.py --list
    scripts/free_floor_rehearsal.py                 # all scenarios
    scripts/free_floor_rehearsal.py happy_reject drill_bridge_dies
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

HARNESS = REPO / "scripts/harness_preflight.sh"
FF = REPO / "scripts/free_floor_gate.py"
FAKE = REPO / "scripts/rehearsal_fake_go2.py"
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
UNITREE_SETUP = Path.home() / "workspace/unitree_ros2/cyclonedds_ws/install/setup.bash"
ARCHIVE = Path.home() / "Archive/phoenix-deploy-candidates-2026-10-03"
WALK_CFG = REPO / "configs/sim2real/proposed/deploy_walk_iter2750.yaml"
WALK_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750.lock.yaml"
STAND_CFG = REPO / "configs/sim2real/deploy_walk_iter2750_stand.yaml"
STAND_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"
HARNESS_TIMEOUT_S = 150


def available() -> str | None:
    """Why the rehearsal cannot run on this machine (None if it can)."""
    for p in (ROS_SETUP, UNITREE_SETUP, ARCHIVE / "eval/iter2750_gate/policy.onnx"):
        if not p.exists():
            return f"missing {p}"
    return None


# ------------------------------------------------------------------ the world
class World:
    """A scratch directory holding the bundle, the synthetic prior evidence and the sessions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hw = root / "hw"
        self.bundle = root / "bundle"
        self.n = 0
        self._ros_env: dict[str, str] | None = None

    # -- bundle: the same layout stage_payload_bundle + derive_walk_bundle produce, pinned in place
    def build_bundle(self) -> None:
        g = ARCHIVE / "eval/iter2750_gate"
        self.bundle.mkdir(parents=True)
        for name in ("policy.onnx", "policy.onnx.data", "policy.pt", "phoenix_manifest.json"):
            shutil.copyfile(g / name, self.bundle / name)
        shutil.copyfile(
            ARCHIVE / "checkpoints/walk_vnext3c_seed47/2026-09-26_21-06-20/model_2750.pt",
            self.bundle / "model_2750.pt",
        )
        shutil.copyfile(WALK_CFG, self.bundle / WALK_CFG.name)
        shutil.copyfile(REPO / "src/phoenix/sim2real/activation.py", self.bundle / "activation.py")
        for f in self.bundle.iterdir():
            f.chmod(0o644)
        run = lambda *a: subprocess.run(a, check=True, capture_output=True, text=True)  # noqa: E731
        run(
            sys.executable,
            str(self.bundle / "activation.py"),
            "pin",
            "--bundle",
            str(self.bundle),
            "--target",
            str(self.bundle),
        )
        sums = subprocess.run(
            "sha256sum ./* > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS",
            shell=True,
            cwd=self.bundle,
            check=True,
        )
        del sums
        run(
            sys.executable,
            str(self.bundle / "activation.py"),
            "verify",
            "--bundle",
            str(self.bundle),
        )

    # -- evidence: synthetic stage records written by the PRODUCTION writer (never counts live)
    def _stage(self, session: Path, stage: str, **extra) -> None:
        from phoenix.sim2real import preflight
        from phoenix.sim2real import preflight_eval as pe
        from phoenix.sim2real.deploy_contract import load_lock
        from phoenix.sim2real.provenance import resolve_code_identity

        run_dir = session / f"run_{stage}1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "bridge.jsonl").write_text('{"fixture": true}\n')
        ctx = types.SimpleNamespace(
            identity=resolve_code_identity(REPO),
            id_problems=[],
            cfg_path=STAND_CFG,
            lock_path=STAND_LOCK,
            lock=load_lock(STAND_LOCK),
            lock_sha=preflight.file_sha256(STAND_LOCK),
            session=session,
            label="walk-iter2750-stand",
        )
        buf = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            preflight.write_stage(
                ctx,
                stage,
                [pe.Check(name="fixture", ok=True, detail="rehearsal fixture")],
                {"run_dir": str(run_dir), **extra},
            )
        finally:
            sys.stdout = buf

    def prior_evidence(self) -> None:
        prior = self.hw / "20261006_prior"
        prior.mkdir(parents=True)
        old = os.environ.pop("PHOENIX_REHEARSAL", None)
        try:
            for stage in "CFGH":
                self._stage(prior, stage)
        finally:
            if old is not None:
                os.environ["PHOENIX_REHEARSAL"] = old

    def new_session(self) -> Path:
        self.n += 1
        s = self.hw / f"2026100{self.n % 10}_{self.n:02d}rehearsal"
        s.mkdir(parents=True)
        old = os.environ.get("PHOENIX_REHEARSAL")
        os.environ["PHOENIX_REHEARSAL"] = "1"
        try:
            self._stage(s, "A", mode="payload")
        finally:
            if old is None:
                os.environ.pop("PHOENIX_REHEARSAL", None)
            else:
                os.environ["PHOENIX_REHEARSAL"] = old
        return s

    # -- a clean environment, built from scratch
    def env(
        self, session: Path, confirm: str, extra: dict[str, str] | None = None
    ) -> dict[str, str]:
        if self._ros_env is None:
            out = subprocess.run(
                ["bash", "-c", f"set +u; source {ROS_SETUP}; source {UNITREE_SETUP}; env -0"],
                env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin:/usr/local/bin"},
                capture_output=True,
                check=True,
            ).stdout.decode()
            self._ros_env = dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        e = dict(self._ros_env)
        e.update(
            ROS_LOCALHOST_ONLY="1",
            RMW_IMPLEMENTATION="rmw_cyclonedds_cpp",
            PHOENIX_REHEARSAL="1",
            PHOENIX_REHEARSAL_WALKED="y",
            PHOENIX_HW_SESSIONS=str(self.hw),
            PHOENIX_HW_SESSIONS_TEST_ONLY="1",
            DEPLOY_CFG=str(self.bundle / WALK_CFG.name),
            DEPLOY_LOCK=str(WALK_LOCK),
            PHOENIX_BUNDLE=str(self.bundle),
            PHOENIX_EXPECT_SHA=sha,
            PHOENIX_SESSION=str(session),
            PHOENIX_UNITREE_SETUP=str(UNITREE_SETUP),
            PHOENIX_CONFIRM=confirm,
        )
        e.update(extra or {})
        return e


# ------------------------------------------------------------------ one harness invocation
class Run:
    """A harness invocation with a fake GO2, a deadman driver and an optional fault injector."""

    def __init__(
        self,
        world: World,
        session: Path,
        args: list[str],
        *,
        release_after_s: float | None = None,
        fault: Callable[[Run], None] | None = None,
        no_fake: bool = False,
        follow: bool = False,
        extra_env: dict[str, str] | None = None,
        confirm: str | None = None,
        timeout_s: float = HARNESS_TIMEOUT_S,
        stage: str = "FREE_FLOOR",
    ) -> None:
        self.world, self.session, self.args = world, session, args
        self.release_after_s, self.fault, self.no_fake = release_after_s, fault, no_fake
        self.follow = follow
        self.extra_env, self.timeout_s = extra_env, timeout_s
        self.stage = stage
        self.confirm = confirm or (
            "LIVE W " + " ".join(args) + " ON GANTRY"
            if stage == "W"
            else "LIVE FREE_FLOOR " + " ".join(args) + " NO GANTRY"
        )
        self.deadman_file = world.root / "deadman.txt"
        self.proc: subprocess.Popen | None = None
        self.fake: subprocess.Popen | None = None
        self.out = ""
        self.rc: int | None = None
        self._t0 = time.time()
        self.events: list[str] = []

    # newest run dir created by THIS invocation
    def run_dir(self) -> Path | None:
        cands = [
            d
            for d in self.session.glob("W_*" if self.stage == "W" else "FF_*")
            if d.is_dir() and d.stat().st_ctime >= self._t0 - 1
        ]
        return sorted(cands)[-1] if cands else None

    def policy_started(self) -> bool:
        d = self.run_dir()
        if d is None or not (d / "bridge.jsonl").is_file():
            return False
        try:
            return any(
                '"mode":"policy"' in line.replace(" ", "") for line in open(d / "bridge.jsonl")
            )
        except OSError:
            return False

    def policy_ticks(self) -> int:
        d = self.run_dir()
        if d is None or not (d / "bridge.jsonl").is_file():
            return 0
        try:
            return sum(
                '"mode":"policy"' in line.replace(" ", "") for line in open(d / "bridge.jsonl")
            )
        except OSError:
            return 0

    def kill(self, pattern: str, sig: int = signal.SIGKILL) -> int:
        """Kill this world's processes whose command line has ``pattern`` and the world root."""
        n = 0
        for pid_s in subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        ).stdout.split():
            try:
                cmd = Path(f"/proc/{pid_s}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                continue
            if str(self.world.root) in cmd and "pgrep" not in cmd and pattern in cmd:
                os.kill(int(pid_s), sig)
                n += 1
        self.events.append(f"killed {pattern} x{n}")
        return n

    def _driver(self) -> None:
        self.deadman_file.write_text("hold")
        while self.proc and self.proc.poll() is None and not self.policy_started():
            time.sleep(0.05)
        if self.proc and self.proc.poll() is None and self.release_after_s is not None:
            time.sleep(self.release_after_s)
            self.deadman_file.write_text("release")
            self.events.append(f"released at +{self.release_after_s}s")

    def _faulter(self) -> None:
        if self.fault is not None:
            self.fault(self)

    def go(self) -> Run:
        env = self.world.env(self.session, self.confirm, self.extra_env)
        self.deadman_file.write_text("hold")
        if not self.no_fake:
            self.fake = subprocess.Popen(
                [
                    sys.executable,
                    str(FAKE),
                    "--pose",
                    "stand",
                    "--deadman-file",
                    str(self.deadman_file),
                    *(["--follow-lowcmd"] if self.follow else []),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(
                3.0
            )  # the fake refuses a graph with other publishers after a 2 s discovery wait
        self.proc = subprocess.Popen(
            [
                "bash",
                str(HARNESS),
                *([] if self.stage == "W" else ["FREE_FLOOR"]),
                *(["W"] if self.stage == "W" else []),
                *self.args,
            ],
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        for target in (self._driver, self._faulter):
            threading.Thread(target=target, daemon=True).start()
        try:
            self.out, _ = self.proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.out, _ = self.proc.communicate()
            self.events.append("harness TIMED OUT and was killed")
        self.rc = self.proc.returncode
        if self.fake is not None:
            self.fake.terminate()
            try:
                self.fake.wait(5)
            except subprocess.TimeoutExpired:
                self.fake.kill()
        # nothing of this world may outlive the invocation
        leftover = self.leftovers()
        if leftover:
            self.events.append(f"LEFTOVER PROCESSES: {leftover}")
            for pid in leftover:
                os.kill(pid, signal.SIGKILL)
        return self

    def leftovers(self) -> list[int]:
        out = []
        for pid_s in subprocess.run(
            ["pgrep", "-f", str(self.world.root)], capture_output=True, text=True
        ).stdout.split():
            if int(pid_s) != os.getpid():
                try:
                    cmd = Path(f"/proc/{pid_s}/cmdline").read_bytes().decode(errors="replace")
                except OSError:
                    continue
                if "pgrep" not in cmd and "free_floor_rehearsal" not in cmd:
                    out.append(int(pid_s))
        return out


def ff_cli(
    world: World, session: Path, *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    env = world.env(session, "x", extra_env)
    return subprocess.run(
        [sys.executable, str(FF), *args], env=env, capture_output=True, text=True, cwd=REPO
    )


def harness(
    world: World, session: Path, *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """A harness call that must not start anything (gates, review, refusals)."""
    env = world.env(session, "x", extra_env)
    return subprocess.run(
        ["bash", str(HARNESS), "FREE_FLOOR", *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


# ------------------------------------------------------------------ scenario plumbing
class Check:
    def __init__(self, scenario: str) -> None:
        self.scenario, self.rows, self.notes = scenario, [], []

    def ok(self, name: str, cond: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(cond), detail))
        return bool(cond)

    @property
    def passed(self) -> bool:
        return all(r[1] for r in self.rows)


def rec(session: Path, name: str) -> dict | None:
    p = session / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def drill_rec(session: Path) -> dict | None:
    files = sorted(session.glob("free_floor_drill_FF_drill_*.json"))
    return json.loads(files[-1].read_text()) if files else None


def no_motion(out: str) -> bool:
    return "launched" not in out


def good_drill(w: World, s: Path, c: Check, release: float = 2.0) -> Run:
    r = Run(w, s, ["drill"], release_after_s=release).go()
    d = drill_rec(s)
    c.ok("drill: harness rc 0", r.rc == 0, f"rc={r.rc} {r.events}")
    c.ok(
        "drill: GO record written", bool(d) and d["verdict"] == "GO", str((d or {}).get("problems"))
    )
    c.ok("drill: no processes left", not any("LEFTOVER" in e for e in r.events), str(r.events))
    return r


def good_walk(w: World, s: Path, c: Check, **kw) -> Run:
    env = {"PHOENIX_REHEARSAL_FORWARD_CM": "41"}
    r = Run(w, s, ["walk", "vx03_2s"], extra_env=env, **kw).go()
    res = rec(s, "free_floor_result.json")
    c.ok("walk: harness rc 0", r.rc == 0, f"rc={r.rc} {r.events}")
    c.ok("walk: GO result", bool(res) and res["verdict"] == "GO", str((res or {}).get("problems")))
    c.ok("walk: started record exists", (s / "free_floor_started.json").is_file())
    c.ok("walk: no processes left", not any("LEFTOVER" in e for e in r.events), str(r.events))
    return r


def refused(w: World, s: Path, c: Check, what: str, *args: str, needle: str = "BLOCKED") -> None:
    p = harness(w, s, *args)
    text = p.stdout + p.stderr
    c.ok(
        f"{what}: refused, nothing launched",
        p.returncode != 0 and no_motion(text) and needle in text,
        text[-300:],
    )


# ------------------------------------------------------------------ scenarios
def sc_happy_reject(w: World, c: Check) -> None:
    """gate -> drill -> walk -> blocked second walk -> review reject -> session closed."""
    s = w.new_session()
    g = harness(w, s, "gate")
    c.ok(
        "gate: drill phase passes, walk phase needs the drill",
        "drill gates pass" in (g.stdout + g.stderr),
        (g.stdout + g.stderr)[-300:],
    )
    good_drill(w, s, c)
    g = harness(w, s, "gate")
    c.ok(
        "gate after a GO drill: NEXT is the walk",
        g.returncode == 0 and "FREE_FLOOR walk vx03_2s" in g.stdout,
        g.stdout[-300:],
    )
    good_walk(w, s, c)
    res = rec(s, "free_floor_result.json") or {}
    rep = res.get("report") or {}
    c.ok(
        "report: valid, planned end, no safety event",
        rep.get("valid") is True
        and rep.get("end_condition") == "planned_end"
        and rep.get("safety_event") is False,
        json.dumps(rep)[:300],
    )
    c.ok(
        "report: tape measurement recorded (41 cm)",
        (rep.get("operator_measurement") or {}).get("forward_cm") == 41.0,
        str(rep.get("operator_measurement")),
    )
    c.ok("report: odometry marked unreliable", (rep.get("odometry") or {}).get("reliable") is False)
    refused(w, s, c, "second walk, unreviewed", "walk", "vx03_2s")
    refused(
        w, s, c, "W is blocked by the unreviewed walk", "gate"
    )  # gate shows the unreviewed block
    w_run = subprocess.run(
        ["bash", str(HARNESS), "W", "vx03_2s"],
        env=w.env(s, "x"),
        capture_output=True,
        text=True,
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    c.ok(
        "gantry W refuses while a free-floor walk is unreviewed",
        w_run.returncode != 0 and "unreviewed free-floor walk" in (w_run.stdout + w_run.stderr),
        (w_run.stdout + w_run.stderr)[-250:],
    )
    p = harness(w, s, "review", "reject", "--by", "rehearsal", "--note", "rehearsal reject")
    c.ok("review reject accepted", p.returncode == 0, p.stderr[-200:])
    p = harness(w, s, "review", "accept", "--by", "rehearsal", "--note", "again")
    c.ok("a run cannot be reviewed twice", p.returncode != 0, p.stderr[-200:])
    refused(w, s, c, "walk after reject, same session", "walk", "vx03_2s")
    u = ff_cli(w, s, "unreviewed")
    c.ok(
        "after the review nothing is unreviewed (latch released, session still closed)",
        u.returncode == 0,
        u.stderr[-200:],
    )


def sc_happy_accept(w: World, c: Check) -> None:
    """A fresh campaign: gate -> drill -> walk -> review accept; accept unlocks nothing more."""
    s = w.new_session()
    good_drill(w, s, c)
    good_walk(w, s, c)
    p = harness(w, s, "review", "accept", "--by", "rehearsal", "--note", "rehearsal accept")
    c.ok("review accept accepted", p.returncode == 0, p.stderr[-200:])
    refused(w, s, c, "another walk in the accepted session", "walk", "vx03_2s")
    refused(w, s, c, "yaw preset", "walk", "yaw05_3s")
    refused(w, s, c, "longer preset", "walk", "vx03_4s")
    s2 = w.new_session()
    g = harness(w, s2, "gate")
    c.ok(
        "a NEW session has no drill: its walk gate is closed",
        "no STOP/deadman drill" in (g.stdout + g.stderr),
        (g.stdout + g.stderr)[-250:],
    )


def sc_stale_env(w: World, c: Check) -> None:
    s = w.new_session()
    for var, val in (
        ("PHOENIX_WALK_AUTHORITY_MARGIN_S", "9.0"),
        ("PHOENIX_STANDUP_S", "0.5"),
        ("PHOENIX_MIN_BATTERY_SOC", "1"),
    ):
        p = harness(w, s, "gate", extra_env={var: val})
        c.ok(
            f"{var}={val} is refused",
            p.returncode != 0 and var in (p.stdout + p.stderr),
            (p.stdout + p.stderr)[-200:],
        )
    p = harness(
        w, s, "gate", extra_env={"PHOENIX_REHEARSAL": "0"}
    )  # a LIVE invocation inheriting the test overrides
    t = p.stdout + p.stderr
    c.ok(
        "live invocation with the evidence-override variables is refused",
        p.returncode != 0 and "PHOENIX_HW_SESSIONS" in t,
        t[-300:],
    )


def sc_wrong_session(w: World, c: Check) -> None:
    outside = Path(tempfile.mkdtemp(prefix="ffreh_outside_", dir=w.root))
    p = harness(w, outside, "gate")
    t = p.stdout + p.stderr
    c.ok(
        "a session outside hw_sessions is refused (invisible to the review latch)",
        p.returncode != 0 and "not a direct child" in t,
        t[-250:],
    )


def _kill_at_policy(pattern: str, delay: float = 0.5):
    def f(r: Run) -> None:
        while r.proc and r.proc.poll() is None and not r.policy_started():
            time.sleep(0.05)
        time.sleep(delay)
        r.kill(pattern)

    return f


def _kill_when_cmd_running(pattern: str):
    """After the walk profile has certainly started (>= 3.5 s of policy ticks incl. the 2 s lead-in)."""

    def f(r: Run) -> None:
        while r.proc and r.proc.poll() is None and r.policy_ticks() < 175:
            time.sleep(0.05)
        r.kill(pattern)

    return f


def _drill_fail(
    w: World, c: Check, name: str, *, fault=None, release: float | None = 2.0, why: str = ""
) -> Run:
    s = w.new_session()
    r = Run(w, s, ["drill"], release_after_s=release, fault=fault).go()
    d = drill_rec(s)
    c.ok(
        f"{name}: drill is not GO",
        not d or d["verdict"] != "GO",
        f"rc={r.rc} {r.events} {(d or {}).get('problems')}",
    )
    c.ok(f"{name}: nothing left running", not any("LEFTOVER" in e for e in r.events), str(r.events))
    g = harness(w, s, "gate")
    c.ok(
        f"{name}: the walk phase of the gate stays closed",
        "walk: NO-GO" in g.stdout,
        g.stdout[-200:],
    )
    refused(w, s, c, f"{name}: walk", "walk", "vx03_2s")
    c.notes.append(f"{name}: rc={r.rc} problems={(d or {}).get('problems')} {why}")
    return r


def sc_drill_bridge_dies(w: World, c: Check) -> None:
    _drill_fail(w, c, "bridge dies", fault=_kill_at_policy("lowcmd_bridge_node"))


def sc_drill_policy_dies(w: World, c: Check) -> None:
    _drill_fail(w, c, "policy node dies", fault=_kill_at_policy("ros2_policy_node"))


def sc_drill_logger_dies(w: World, c: Check) -> None:
    _drill_fail(
        w, c, "STOP-path recorder dies", fault=_kill_at_policy("free_floor_remote_trace", 0.3)
    )


def sc_drill_no_release(w: World, c: Check) -> None:
    _drill_fail(w, c, "deadman never released", release=None)


def sc_drill_early_release(w: World, c: Check) -> None:
    _drill_fail(w, c, "deadman released too early (0.3 s)", release=0.3)


def sc_drill_late_release(w: World, c: Check) -> None:
    _drill_fail(w, c, "deadman released too late (4.6 s)", release=4.6)


def sc_drill_stale_lowstate(w: World, c: Check) -> None:
    def f(r: Run) -> None:
        while r.proc and r.proc.poll() is None and not r.policy_started():
            time.sleep(0.05)
        time.sleep(0.8)
        if r.fake:
            r.fake.send_signal(signal.SIGKILL)
            r.events.append("fake GO2 (LowState) killed")

    _drill_fail(w, c, "LowState goes stale", fault=f, release=None)


def sc_drill_retry(w: World, c: Check) -> None:
    """A NO-GO drill must not poison the session: a retry is analysed and a GO one unblocks the walk."""
    s = w.new_session()
    r = Run(w, s, ["drill"], release_after_s=None).go()  # never released -> NO-GO
    d1 = drill_rec(s)
    c.ok("first drill NO-GO", bool(d1) and d1["verdict"] == "NO-GO", f"{r.events}")
    refused(w, s, c, "walk after a NO-GO drill", "walk", "vx03_2s")
    time.sleep(1.1)  # run directories are named by the second
    good_drill(w, s, c)
    c.ok(
        "two drill records on disk (nothing overwritten)",
        len(list(s.glob("free_floor_drill_FF_drill_*.json"))) == 2,
    )
    g = harness(w, s, "gate")
    c.ok(
        "walk gate opens only on the newest GO drill",
        g.returncode == 0,
        (g.stdout + g.stderr)[-300:],
    )
    for _ in range(2):
        time.sleep(1.1)
        Run(w, s, ["drill"], release_after_s=2.0).go()
    refused(w, s, c, "fourth drill attempt (cap 3)", "drill", needle="drill attempts already")


def _walk_fail(
    w: World,
    c: Check,
    name: str,
    *,
    fault=None,
    release: float | None = None,
    expect_blocked: bool = True,
) -> tuple[World, Path, Run]:
    s = w.new_session()
    good_drill(w, s, Check("x"))
    time.sleep(1.1)
    r = Run(
        w,
        s,
        ["walk", "vx03_2s"],
        release_after_s=release,
        fault=fault,
        extra_env={"PHOENIX_REHEARSAL_FORWARD_CM": "30"},
    ).go()
    res = rec(s, "free_floor_result.json")
    c.ok(
        f"{name}: started record exists (written before motion)",
        (s / "free_floor_started.json").is_file(),
    )
    c.ok(
        f"{name}: not a GO result",
        not res or res["verdict"] != "GO",
        f"rc={r.rc} {r.events} {(res or {}).get('problems')}",
    )
    c.ok(f"{name}: nothing left running", not any("LEFTOVER" in e for e in r.events), str(r.events))
    refused(w, s, c, f"{name}: next walk", "walk", "vx03_2s")
    p = harness(w, s, "review", "accept", "--by", "r", "--note", "n")
    c.ok(f"{name}: accept refused", p.returncode != 0, p.stderr[-200:])
    p = harness(w, s, "review", "reject", "--by", "r", "--note", "n")
    c.ok(f"{name}: reject accepted", p.returncode == 0, p.stderr[-200:])
    c.notes.append(f"{name}: rc={r.rc} result={(res or {}).get('problems')}")
    return w, s, r


def sc_walk_bridge_dies_before_cmd(w: World, c: Check) -> None:
    _walk_fail(
        w, c, "bridge dies before the command", fault=_kill_at_policy("lowcmd_bridge_node", 0.2)
    )


def sc_walk_bridge_dies_during_cmd(w: World, c: Check) -> None:
    _walk_fail(
        w, c, "bridge dies during the command", fault=_kill_when_cmd_running("lowcmd_bridge_node")
    )


def sc_walk_policy_dies(w: World, c: Check) -> None:
    _walk_fail(
        w,
        c,
        "policy node dies during the command",
        fault=_kill_when_cmd_running("ros2_policy_node"),
    )


def sc_walk_cmd_dies(w: World, c: Check) -> None:
    _walk_fail(
        w,
        c,
        "walk_command exits early",
        fault=_kill_when_cmd_running("phoenix.sim2real.walk_command"),
    )


def sc_walk_deadman_released(w: World, c: Check) -> None:
    _walk_fail(w, c, "deadman released mid-walk", release=3.0)


def sc_walk_lowstate_stops(w: World, c: Check) -> None:
    def f(r: Run) -> None:
        while r.proc and r.proc.poll() is None and r.policy_ticks() < 175:
            time.sleep(0.05)
        if r.fake:
            r.fake.send_signal(signal.SIGKILL)

    _walk_fail(w, c, "LowState stops during the command", fault=f)


def sc_walk_harness_killed(w: World, c: Check) -> None:
    """SIGKILL of the harness itself (power loss analogue): no analysis, no cleanup, still latched."""

    def f(r: Run) -> None:
        while r.proc and r.proc.poll() is None and r.policy_ticks() < 175:
            time.sleep(0.05)
        if r.proc:
            os.killpg(r.proc.pid, signal.SIGKILL)
            r.events.append("harness process group SIGKILLed mid-walk")

    _, s, r = _walk_fail(w, c, "harness SIGKILLed mid-walk", fault=f)
    c.ok("no result record exists", not (s / "free_floor_result.json").exists())


def sc_crash_after_begin(w: World, c: Check) -> None:
    """Interrupted between 'run started' and the first process: the walk is still recorded."""
    s = w.new_session()
    good_drill(w, s, Check("x"))
    run = s / "FF_walk_20261008T000000Z"
    run.mkdir()
    p = ff_cli(
        w,
        s,
        "begin",
        "--kind",
        "walk",
        "--preset",
        "vx03_2s",
        "--run-dir",
        str(run),
        "--config",
        str(w.bundle / WALK_CFG.name),
        "--lock",
        str(WALK_LOCK),
        "--session",
        str(s),
        "--bundle",
        str(w.bundle),
    )
    c.ok(
        "begin succeeds and records the walk",
        p.returncode == 0 and (s / "free_floor_started.json").is_file(),
        p.stderr[-300:],
    )
    refused(w, s, c, "walk after a crash between begin and launch", "walk", "vx03_2s")
    q = ff_cli(
        w,
        s,
        "walk-eval",
        "--run-dir",
        str(run),
        "--config",
        str(w.bundle / WALK_CFG.name),
        "--lock",
        str(WALK_LOCK),
        "--session",
        str(s),
        "--bundle",
        str(w.bundle),
    )
    res = rec(s, "free_floor_result.json")
    c.ok(
        "analysing the empty run is a NO-GO, not a crash or a pass",
        q.returncode != 0 and bool(res) and res["verdict"] == "NO-GO",
        q.stderr[-200:],
    )
    p = harness(w, s, "review", "accept", "--by", "r", "--note", "n")
    c.ok("accept refused on the empty run", p.returncode != 0)


def sc_review_forgery(w: World, c: Check) -> None:
    s = w.new_session()
    good_drill(w, s, Check("x"))
    good_walk(w, s, Check("x"))
    (s / "free_floor_review.json").write_text("{}")
    u = ff_cli(w, s, "unreviewed")
    c.ok(
        "a hand-made empty review file does not release the latch",
        u.returncode != 0,
        u.stderr[-200:],
    )
    (s / "free_floor_review.json").unlink()
    harness(w, s, "review", "accept", "--by", "r", "--note", "n")
    other = w.new_session()
    shutil.copyfile(s / "free_floor_started.json", other / "free_floor_started.json")
    shutil.copyfile(s / "free_floor_review.json", other / "free_floor_review.json")
    u = ff_cli(w, s, "unreviewed")
    c.ok("the copy in another session stays latched", other.name in u.stderr, u.stderr[-200:])
    c.ok("a genuine review releases ITS session's latch", s.name not in u.stderr, u.stderr[-200:])
    shutil.rmtree(other)
    res = json.loads((s / "free_floor_result.json").read_text())
    res["utc"] = "tampered"
    (s / "free_floor_result.json").write_text(json.dumps(res))
    u = ff_cli(w, s, "unreviewed")
    c.ok("editing the result after the review re-latches", u.returncode != 0, u.stderr[-200:])


def sc_missing_raw_log(w: World, c: Check) -> None:
    s = w.new_session()
    good_drill(w, s, Check("x"))
    good_walk(w, s, Check("x"))
    run = sorted(s.glob("FF_walk_*"))[-1]
    (run / "bridge.jsonl").unlink()
    p = harness(w, s, "review", "accept", "--by", "r", "--note", "n")
    c.ok(
        "accept refused when the result's raw log is gone",
        p.returncode != 0 and "evidence no longer matches" in p.stderr,
        p.stderr[-250:],
    )


def sc_artifact_swap(w: World, c: Check) -> None:
    """The model changes after the gate and before the run: the run-start re-check refuses it."""
    s = w.new_session()
    good_drill(w, s, Check("x"))
    onnx = w.bundle / "policy.onnx"
    keep = onnx.read_bytes()
    onnx.write_bytes(keep + b"x")
    try:
        refused(w, s, c, "walk with a modified policy.onnx", "walk", "vx03_2s")
    finally:
        onnx.write_bytes(keep)
    link = w.bundle / "policy.onnx.data"
    real = link.read_bytes()
    elsewhere = w.root / "elsewhere.data"
    elsewhere.write_bytes(real)
    link.unlink()
    link.symlink_to(elsewhere)
    try:
        refused(w, s, c, "walk with policy.onnx.data replaced by a symlink", "walk", "vx03_2s")
    finally:
        link.unlink()
        link.write_bytes(real)
    # the same files in another directory than PHOENIX_BUNDLE: the bundle is not what loads
    p = harness(w, s, "gate", extra_env={"PHOENIX_BUNDLE": str(ARCHIVE / "eval/iter2750_gate")})
    t = p.stdout + p.stderr
    c.ok(
        "a bundle that is not the directory the config loads from is refused",
        p.returncode != 0 and ("outside the bundle" in t or "SHA256SUMS" in t),
        t[-250:],
    )


# ------------------------------------------------------------------ walking authority (grant)
def _live_grant(w: World, s: Path, mode: str, preset: str, name: str) -> Path:
    """A valid, unreleased grant sitting in another run directory of the session (an active run)."""
    from phoenix.sim2real import walk_grant as wg
    from phoenix.sim2real.provenance import resolve_code_identity

    run = s / name
    run.mkdir()
    cfg = w.bundle / WALK_CFG.name
    grant, why = wg.issue_grant(
        mode=mode,
        preset=preset,
        run_dir=run,
        code_sha=resolve_code_identity(REPO).sha or "",
        config_path=cfg,
        onnx_path=w.bundle / "policy.onnx",
        first_message_timeout_s=15.0,
        standup_s=2.0,
        rehearsal=True,
    )
    assert grant is not None, why
    return run


def gantry_w(w: World, s: Path, preset: str = "vx03_2s", **kw) -> Run:
    return Run(w, s, [preset], stage="W", **kw).go()


def sc_gantry_w_ok(w: World, c: Check) -> None:
    """The legitimate gantry W path still works, bounded by a GANTRY grant it issued itself."""
    s = w.new_session()
    r = gantry_w(w, s)
    run = r.run_dir()
    c.ok("W: harness rc 0", r.rc == 0, f"rc={r.rc} {r.events} {r.out[-300:]}")
    c.ok("W: run directory created", run is not None)
    g = rec(run, "walk_grant.json") if run else None
    c.ok(
        "W: a GANTRY grant for vx03_2s was issued in the run directory",
        bool(g) and g["mode"] == "GANTRY" and g["preset"] == "vx03_2s" and g["run_id"] == run.name,
        str(g),
    )
    c.ok(
        "W: the grant was released at the end",
        bool(run) and (run / "walk_grant.released").is_file(),
    )
    summary = None
    try:
        lines = [json.loads(x) for x in (run / "walk_cmd.jsonl").read_text().splitlines()]
        summary = next((x for x in lines if x.get("record") == "summary"), None)
    except (OSError, ValueError, TypeError):
        pass
    c.ok(
        "W: walk_command completed the preset",
        bool(summary) and summary["outcome"] == "complete",
        str(summary),
    )
    faults = set()
    try:
        for x in (run / "bridge.jsonl").read_text().splitlines():
            t = json.loads(x)
            for f in t.get("faults") or []:
                faults.add(f)
    except (OSError, ValueError, TypeError):
        pass
    c.ok("W: no grant fault in the bridge", not any("grant" in f for f in faults), str(faults))
    c.ok("W: no processes left", not any("LEFTOVER" in e for e in r.events), str(r.events))


def sc_gantry_w_bad_preset(w: World, c: Check) -> None:
    """W authorizes only the registered gantry presets: nothing else launches."""
    s = w.new_session()
    for preset in ("vx05_3s", "vxm03_2s", "vx09_9s", "../vx03_2s", ""):
        p = subprocess.run(
            ["bash", str(HARNESS), "W", preset],
            env=w.env(s, f"LIVE W {preset} ON GANTRY"),
            capture_output=True,
            text=True,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        t = p.stdout + p.stderr
        c.ok(
            f"W preset {preset!r}: refused, nothing launched",
            p.returncode != 0 and no_motion(t) and not list(s.glob("W_*")),
            t[-250:],
        )


def sc_grant_mutual_exclusion(w: World, c: Check) -> None:
    """One walking authority at a time, in both directions; a released grant frees the slot."""
    s = w.new_session()
    gdir = _live_grant(w, s, "GANTRY", "vx03_4s", "W_vx03_4s_active")
    r = Run(w, s, ["drill"], release_after_s=2.0).go()
    t = r.out
    c.ok(
        "FREE_FLOOR drill refused while a gantry grant is live, nothing launched, nothing latched",
        r.rc != 0
        and "another walking grant is live" in t
        and no_motion(t)
        and not (s / "free_floor_started.json").exists(),
        t[-300:],
    )
    (gdir / "walk_grant.released").write_text("done\n")
    good_drill(w, s, c)
    s2 = w.new_session()
    _live_grant(w, s2, "FREE_FLOOR", "vx03_2s", "FF_walk_active")
    r = gantry_w(w, s2)
    t = r.out
    c.ok(
        "gantry W refused while a free-floor grant is live, nothing launched",
        r.rc != 0 and "another walking grant" in t and "is live" in t and no_motion(t),
        t[-300:],
    )


def sc_hand_launch_refused(w: World, c: Check) -> None:
    """A bridge or policy node started by hand, without a valid grant, cannot walk."""
    from phoenix.sim2real import walk_grant as wg

    s = w.new_session()
    run = s / "HAND_1"
    run.mkdir()
    env = w.env(s, "x")
    env["PYTHONPATH"] = str(REPO / "src") + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cfg = str(w.bundle / WALK_CFG.name)
    sha = env["PHOENIX_EXPECT_SHA"]

    def bridge(grant: str | None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "phoenix.sim2real.lowcmd_bridge_node",
                "--live",
                "--config",
                cfg,
                "--lock",
                str(WALK_LOCK),
                "--telemetry",
                str(run / "bridge.jsonl"),
                "--stage",
                "HAND",
                "--expect-sha",
                sha,
                *(["--grant", grant] if grant else []),
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )

    def policy(grant: str | None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "phoenix.sim2real.ros2_policy_node",
                "--config",
                cfg,
                "--lock",
                str(WALK_LOCK),
                "--log-parquet",
                str(run / "policy.parquet"),
                *(["--grant", grant] if grant else []),
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )

    b = bridge(None)
    c.ok(
        "hand bridge, no grant: refuses to start",
        b.returncode == 2 and "walking grant" in b.stderr,
        b.stderr[-250:],
    )
    n = policy(None)
    c.ok(
        "hand policy node, no grant: refuses to start",
        n.returncode == 2 and "walking grant" in (n.stderr + n.stdout),
        (n.stderr + n.stdout)[-250:],
    )
    # an EXPIRED grant, issued in the past for this very run
    grant, why = wg.issue_grant(
        mode="GANTRY",
        preset="vx03_2s",
        run_dir=run,
        code_sha=sha,
        config_path=Path(cfg),
        onnx_path=w.bundle / "policy.onnx",
        first_message_timeout_s=15.0,
        standup_s=2.0,
        rehearsal=True,
        now=time.time() - 3600,
    )
    assert grant is not None, why
    b = bridge(str(run / "walk_grant.json"))
    c.ok(
        "hand bridge, expired grant: refuses to start",
        b.returncode == 2 and "expired" in b.stderr,
        b.stderr[-250:],
    )
    n = policy(str(run / "walk_grant.json"))
    c.ok(
        "hand policy node, expired grant: refuses to start",
        n.returncode == 2 and "expired" in (n.stderr + n.stdout),
        (n.stderr + n.stdout)[-250:],
    )
    # a valid grant for ANOTHER run directory
    other = s / "HAND_2"
    other.mkdir()
    g2, why = wg.issue_grant(
        mode="GANTRY",
        preset="vx03_2s",
        run_dir=other,
        code_sha=sha,
        config_path=Path(cfg),
        onnx_path=w.bundle / "policy.onnx",
        first_message_timeout_s=15.0,
        standup_s=2.0,
        rehearsal=True,
    )
    assert g2 is not None, why
    b = bridge(str(other / "walk_grant.json"))
    c.ok(
        "hand bridge, grant of another run: refuses to start",
        b.returncode == 2 and "is for run" in b.stderr,
        b.stderr[-250:],
    )
    n = policy(str(other / "walk_grant.json"))
    c.ok(
        "hand policy node, grant of another run: refuses to start",
        n.returncode == 2 and "is for run" in (n.stderr + n.stdout),
        (n.stderr + n.stdout)[-250:],
    )


SCENARIOS: dict[str, Callable[[World, Check], None]] = {
    "happy_reject": sc_happy_reject,
    "happy_accept": sc_happy_accept,
    "stale_env": sc_stale_env,
    "wrong_session": sc_wrong_session,
    "artifact_swap": sc_artifact_swap,
    "review_forgery": sc_review_forgery,
    "missing_raw_log": sc_missing_raw_log,
    "crash_after_begin": sc_crash_after_begin,
    "drill_retry": sc_drill_retry,
    "drill_bridge_dies": sc_drill_bridge_dies,
    "drill_policy_dies": sc_drill_policy_dies,
    "drill_logger_dies": sc_drill_logger_dies,
    "drill_no_release": sc_drill_no_release,
    "drill_early_release": sc_drill_early_release,
    "drill_late_release": sc_drill_late_release,
    "drill_stale_lowstate": sc_drill_stale_lowstate,
    "walk_bridge_dies_before_cmd": sc_walk_bridge_dies_before_cmd,
    "walk_bridge_dies_during_cmd": sc_walk_bridge_dies_during_cmd,
    "walk_policy_dies": sc_walk_policy_dies,
    "walk_cmd_dies": sc_walk_cmd_dies,
    "walk_deadman_released": sc_walk_deadman_released,
    "walk_lowstate_stops": sc_walk_lowstate_stops,
    "walk_harness_killed": sc_walk_harness_killed,
    "gantry_w_ok": sc_gantry_w_ok,
    "gantry_w_bad_preset": sc_gantry_w_bad_preset,
    "grant_mutual_exclusion": sc_grant_mutual_exclusion,
    "hand_launch_refused": sc_hand_launch_refused,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the scratch world for inspection")
    args = ap.parse_args(argv)
    if args.list:
        for name, fn in SCENARIOS.items():
            print(f"{name:32s} {(fn.__doc__ or '').strip().splitlines()[0] if fn.__doc__ else ''}")
        return 0
    why = available()
    if why:
        print(
            f"SKIP: rehearsal needs ROS 2, the unitree messages and the model_2750 archive ({why})"
        )
        return 77
    if subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip():
        print(
            "REFUSING: the tracked tree is dirty. The bridge refuses to go live on a dirty tree (as it must); commit first."
        )
        return 2
    names = args.scenarios or list(SCENARIOS)
    bad = [n for n in names if n not in SCENARIOS]
    if bad:
        print(f"unknown scenario(s): {bad}", file=sys.stderr)
        return 2
    results = []
    for name in names:
        root = Path(tempfile.mkdtemp(prefix="ffrehearsal_"))
        w = World(root)
        t0 = time.time()
        c = Check(name)
        try:
            w.build_bundle()
            w.prior_evidence()
            SCENARIOS[name](w, c)
        except Exception as exc:  # a crashing scenario is a failed scenario
            c.ok("scenario ran to completion", False, f"{type(exc).__name__}: {exc}")
        print(f"\n=== {name}  ({time.time() - t0:.0f} s) {'PASS' if c.passed else 'FAIL'}")
        for row_name, ok, detail in c.rows:
            print(
                f"  [{'ok' if ok else 'FAIL'}] {row_name}"
                + ("" if ok or not detail else f"\n         {detail}")
            )
        for n in c.notes:
            print(f"  note: {n}")
        results.append((name, c.passed))
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"  world kept: {root}")
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {n}" for n, ok in results))
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
