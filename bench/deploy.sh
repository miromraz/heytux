#!/usr/bin/env bash
# deploy.sh - push the bench harness (code only) to the phone at /home/miro/heytux/bench.
# Leaves the phone's own manifest.json, results/ and model files alone. Override the target with
# HEYTUX_SSH / HEYTUX_SCP_HOST (default: the `pixelw` ssh alias, root).
set -euo pipefail   # pipefail so a tar failure in the tar|ssh pipe is not masked by ssh's rc
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="${HEYTUX_SCP_HOST:-pixelw}"
SSH="${HEYTUX_SSH:-ssh $HOST}"
DST=/home/miro/heytux/bench

# No rsync on the phone; tar the code over ssh (scp would also work for these four files).
$SSH "mkdir -p $DST"
tar -C "$HERE" -czf - backends.py run.py report.py deploy.sh zram_tune.py | $SSH "tar -C $DST -xzf -"
$SSH "chown -R miro:miro $DST"
echo "deployed bench -> $HOST:$DST"
