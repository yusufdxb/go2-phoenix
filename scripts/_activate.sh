#!/usr/bin/env bash
# Shared Isaac Sim venv activation + EULA auto-accept.
# Source this from the other scripts: `source "$(dirname "$0")/_activate.sh"`.

ISAAC_VENV="${ISAAC_VENV:-$HOME/isaac-sim-venv}"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$ISAAC_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ISAAC_VENV/bin/activate"
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
