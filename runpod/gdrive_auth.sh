#!/usr/bin/env bash
# One-time Google Drive authorization for checkpoint backups, stored on the
# volume. Pods are headless, so the OAuth step runs in a browser elsewhere:
#   1. on a machine with a browser and rclone:  rclone authorize "drive"
#   2. paste the token JSON it prints when this script asks for it.
set -euo pipefail
export RCLONE_CONFIG=/workspace/.config/rclone/rclone.conf
mkdir -p "$(dirname "$RCLONE_CONFIG")"
read -r -p "token JSON from 'rclone authorize \"drive\"': " TOKEN
rclone config create gdrive drive scope=drive token="$TOKEN" config_refresh_token=false >/dev/null
rclone mkdir gdrive:oflm/checkpoints
rclone lsd gdrive:oflm && echo "drive backup ready: gdrive:oflm/checkpoints"
