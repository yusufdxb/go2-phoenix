#!/usr/bin/env bash
# Old name of scripts/hw_campaign.sh (N-arm hardware campaign runner); same arguments.
exec bash "$(dirname "${BASH_SOURCE[0]}")/hw_campaign.sh" "$@"
