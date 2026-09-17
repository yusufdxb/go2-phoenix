"""Make an Isaac Sim entry point's exit status mean what it says.

MEASURED 2026-09-17 on Isaac Sim 6 / Isaac Lab 4.5.22: ``simulation_app.close()``
terminates the process with status 0. Neither a ``raise`` nor a ``return`` from
``main()`` survives it, so the standard shape

    try:
        return _run(args, simulation_app)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

reports SUCCESS for a run that crashed. That is not theoretical: on the first
end-to-end GPU run of ``scripts/loop_closure.sh``, all three
``phoenix.adaptation.fine_tune`` invocations died in ``install_reset_bridge``
and exited 0. The script's ``set -e`` and ``pipefail`` saw nothing, it went on
to the evaluation stage, and only failed there with
``checkpoint for adapted_seed42 not found: <empty>`` -- two stages and several
minutes after the actual fault, with the real traceback buried in a per-seed log.

``scripts/diag_post_reset_termination.py`` already worked around this locally.
This module is the shared version, so every Isaac entry point gets it.

Usage, replacing the whole try/except/finally above::

    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args, simulation_app), simulation_app)

The wrapper never returns: it calls :func:`os._exit`, which is the only exit
that Isaac's shutdown cannot override. Callers still write ``return`` in front
of it so the control flow reads normally and linters see a terminating branch.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from typing import NoReturn


def run_isaac_main(
    body: Callable[[], int | None],
    simulation_app,  # noqa: ANN001 - Isaac's SimulationApp, untyped upstream
    *,
    label: str = "isaac",
) -> NoReturn:
    """Run ``body`` and exit with a status that survives Isaac's shutdown.

    ``body`` returns the process status (``None`` counts as 0). Any exception
    is printed and becomes status 1.

    On failure Isaac is deliberately NOT shut down cleanly: the process is
    already aborting, and a truthful exit status matters more than a tidy
    teardown. On success ``simulation_app.close()`` is called first, so the
    normal path keeps whatever cleanup Isaac does.
    """

    try:
        rc = body()
        rc = 0 if rc is None else int(rc)
    except BaseException:  # noqa: BLE001 - reported, then converted to a status
        traceback.print_exc()
        _flush()
        os._exit(1)

    _flush()
    if rc != 0:
        print(f"[{label}] exiting with status {rc}", file=sys.stderr, flush=True)
        os._exit(rc)
    simulation_app.close()
    _flush()
    os._exit(0)


def _flush() -> None:
    """Flush both streams before os._exit, which skips normal teardown."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - a closed stream must not mask the status
            pass
