"""An Isaac entry point's exit status must mean what it says.

Isaac Sim's ``simulation_app.close()`` terminates the process with status 0, so
the ordinary ``try/except/raise/finally: close()`` shape reports SUCCESS for a
run that crashed. That is not hypothetical: on the first end-to-end GPU run of
``scripts/loop_closure.sh`` all three ``fine_tune`` invocations died in
``install_reset_bridge`` and exited 0, the script's ``set -e`` saw nothing, and
the run only failed two stages later with ``checkpoint ... not found: <empty>``.

These tests run the wrapper in a REAL subprocess, because that is the only way
to observe a process exit status; a stub ``simulation_app`` stands in for Isaac
and can pretend to exit 0 the way the real one does.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

PREAMBLE = """
import os, sys
sys.path.insert(0, {src!r})
from phoenix.sim_app_exit import run_isaac_main

class FakeApp:
    def __init__(self, hijack=False):
        self.hijack = hijack
        self.closed = False
    def close(self):
        self.closed = True
        print("CLOSED", flush=True)
        if self.hijack:
            # What the real Isaac does: exit 0 out from under the caller.
            os._exit(0)
"""


def run_snippet(body: str, src: str) -> subprocess.CompletedProcess:
    script = PREAMBLE.format(src=src) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )


def src_dir() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "src")


def test_a_raising_body_exits_nonzero_even_when_close_would_exit_zero():
    res = run_snippet(
        """
        app = FakeApp(hijack=True)
        run_isaac_main(lambda: (_ for _ in ()).throw(ValueError("boom")), app)
        """,
        src_dir(),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "ValueError: boom" in res.stderr
    # A crashing run must not reach Isaac's teardown, because that teardown is
    # exactly what would overwrite the status with 0.
    assert "CLOSED" not in res.stdout


def test_a_nonzero_return_survives_a_hijacking_close():
    res = run_snippet(
        """
        run_isaac_main(lambda: 2, FakeApp(hijack=True))
        """,
        src_dir(),
    )
    assert res.returncode == 2, res.stdout + res.stderr


def test_success_exits_zero_and_still_closes_isaac():
    res = run_snippet(
        """
        run_isaac_main(lambda: 0, FakeApp(hijack=False))
        """,
        src_dir(),
    )
    assert res.returncode == 0
    assert "CLOSED" in res.stdout


def test_a_body_returning_none_counts_as_success():
    res = run_snippet(
        """
        run_isaac_main(lambda: None, FakeApp(hijack=False))
        """,
        src_dir(),
    )
    assert res.returncode == 0


def test_stdout_written_by_the_body_is_not_lost_to_os_exit():
    # os._exit skips normal interpreter teardown, so buffered output would be
    # discarded. A run whose evidence vanished would be as bad as a wrong status.
    res = run_snippet(
        """
        def body():
            print("EVIDENCE LINE")
            return 3
        run_isaac_main(body, FakeApp(hijack=True))
        """,
        src_dir(),
    )
    assert res.returncode == 3
    assert "EVIDENCE LINE" in res.stdout


def test_a_keyboard_interrupt_is_also_a_failure_status():
    res = run_snippet(
        """
        run_isaac_main(lambda: (_ for _ in ()).throw(KeyboardInterrupt()), FakeApp(hijack=True))
        """,
        src_dir(),
    )
    assert res.returncode == 1


def test_the_label_appears_on_a_nonzero_exit():
    res = run_snippet(
        """
        run_isaac_main(lambda: 4, FakeApp(hijack=False), label="replay")
        """,
        src_dir(),
    )
    assert res.returncode == 4
    assert "replay" in res.stderr


def test_every_isaac_entry_point_uses_the_wrapper():
    """A new entry point that hand-rolls the old shape reintroduces the bug."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    expected = [
        "src/phoenix/replay/reconstruct.py",
        "src/phoenix/adaptation/fine_tune.py",
        "src/phoenix/training/evaluate.py",
        "src/phoenix/training/ppo_runner.py",
        "scripts/harvest_sim_failures.py",
    ]
    for rel in expected:
        text = (root / rel).read_text()
        assert "run_isaac_main" in text, f"{rel} does not use the shared exit wrapper"
