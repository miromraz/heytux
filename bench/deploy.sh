#!/usr/bin/env bash
# deploy.sh - push the bench harness (code only) to the phone at /home/miro/heytux/bench.
# Leaves the phone's own manifest.json, results/ and model files alone. Override the target with
# HEYTUX_SSH / HEYTUX_SCP_HOST (default: the `pixelw` ssh alias, root).
set -euo pipefail   # pipefail so a tar failure in the tar|ssh pipe is not masked by ssh's rc
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="${HEYTUX_SCP_HOST:-pixelw}"
SSH="${HEYTUX_SSH:-ssh $HOST}"
DST=/home/miro/heytux/bench

# No rsync on the phone; tar the code over ssh (scp would also work for these files).
$SSH "mkdir -p $DST"
# backends.py moved to ../heytux_backends.py (deployed by the daemon's deploy.sh to /home/miro/heytux,
# which run.py/zram_tune.py add to sys.path); the bench no longer ships its own copy.
tar -C "$HERE" -czf - run.py report.py deploy.sh zram_tune.py toolcall.py hybrid_e2e.py | $SSH "tar -C $DST -xzf -"
$SSH "chown -R miro:miro $DST"
echo "deployed bench -> $HOST:$DST"
