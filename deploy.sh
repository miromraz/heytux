#!/usr/bin/env bash
# Deploy heytux to the phone, restart the --user service, smoke test.
# Usage: ./deploy.sh [ssh-host-args]   (defaults to the WiFi target below)
set -euo pipefail

SSH=${HEYTUX_SSH:-"ssh -o BatchMode=yes -o ConnectTimeout=8 -o UserKnownHostsFile=$HOME/.ssh/known_hosts.sunfish -o HostKeyAlias=172.16.42.1 root@192.168.100.195"}
HERE="$(cd "$(dirname "$0")" && pwd)"
RU="runuser -u miro -- env XDG_RUNTIME_DIR=/run/user/60466 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/60466/bus"

echo "== migrate from the old jarvis name (no-op once done) =="
$SSH "$RU systemctl --user disable --now jarvis jarvis-llm 2>/dev/null || true"
$SSH 'cd /home/miro && { [ -d jarvis ] && [ ! -e heytux ] && mv jarvis heytux; [ -d .config/jarvis ] && [ ! -e .config/heytux ] && mv .config/jarvis .config/heytux; rm -rf .config/systemd/user/jarvis.service .config/systemd/user/jarvis-llm.service .local/share/applications/org.bengalos.JarvisSettings.desktop .local/share/plasma/plasmoids/org.bengalos.jarvis heytux/jarvis.py heytux/jarvis_config.py heytux/jarvis_settings.py; true; }'

echo "== push daemon + unit =="
$SSH 'mkdir -p /home/miro/heytux /home/miro/.config/systemd/user /home/miro/.local/share/plasma/plasmoids /home/miro/.local/share/applications'
tar cf - -C "$HERE" heytux.py heytux_config.py heytux_settings.py hey.txt | $SSH 'tar xf - -C /home/miro/heytux'
tar cf - -C "$HERE" heytux.service heytux-llm.service | $SSH 'tar xf - -C /home/miro/.config/systemd/user'
tar cf - -C "$HERE" org.bengalos.HeyTuxSettings.desktop | $SSH 'tar xf - -C /home/miro/.local/share/applications'
tar cf - -C "$HERE/plasmoid" . | $SSH 'mkdir -p /home/miro/.local/share/plasma/plasmoids/org.bengalos.heytux && tar xf - -C /home/miro/.local/share/plasma/plasmoids/org.bengalos.heytux'

echo "== fix ownership =="
$SSH 'chown -R miro:miro /home/miro/heytux /home/miro/.config/systemd/user /home/miro/.local/share/applications /home/miro/.local/share/plasma/plasmoids/org.bengalos.heytux'

echo "== (re)start services =="
$SSH "$RU systemctl --user daemon-reload"
# not enabled: the user starts/stops it from the Hey Tux app (it costs ~1/3 of a core listening)
$SSH "$RU systemctl --user disable heytux-llm heytux 2>/dev/null || true"
$SSH "$RU systemctl --user restart heytux-llm heytux"

echo "== wait for load =="
$SSH "$RU sh -c 'for i in \$(seq 1 30); do python3 -c \"import urllib.request,sys; urllib.request.urlopen(sys.argv[1],timeout=2)\" http://127.0.0.1:8770/status 2>/dev/null && exit 0; sleep 1; done; exit 1'"

echo "== smoke: /text turn on/off the flashlight =="
$SSH "$RU python3 -" <<'PY'
import urllib.request, json
def post(p, d=None):
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8770"+p,
        data=json.dumps(d).encode() if d is not None else None,
        headers={"Content-Type":"application/json"}), timeout=30)
    return json.load(r)
print("on :", json.dumps(post("/text", {"text":"turn on the flashlight"})))
print("off:", json.dumps(post("/text", {"text":"turn off the flashlight"})))
print("status:", json.dumps(post("/status")))
PY
echo "== is-active =="
$SSH "$RU systemctl --user is-active heytux"
