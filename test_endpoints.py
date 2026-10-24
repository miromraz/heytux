#!/usr/bin/env python3
# test_endpoints.py - HTTP checks against the LIVE heytux daemon. SILENT: no TTS, no tool calls.
# Run ON the phone as miro:  python3 test_endpoints.py
import json, os, sys, urllib.request, urllib.error

BASE = os.environ.get("HEYTUX_URL", "http://127.0.0.1:8770")
FAILS = []

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=30) as fh:
            return fh.status, json.load(fh)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

def check(name, fn):
    try:
        fn()
        print("PASS %s" % name)
    except AssertionError as e:
        FAILS.append(name)
        print("FAIL %s: %s" % (name, e))

CONFIG_JSON = os.path.expanduser("~/.config/heytux/config.json")

def _read_config_file():
    try:
        with open(CONFIG_JSON) as fh:
            return fh.read()
    except FileNotFoundError:
        return None

def t_config_shape():
    code, r = req("GET", "/config")
    assert code == 200, code
    for k in ("values", "defaults", "schema", "pending_restart"):
        assert k in r, "missing %s" % k
    keys = {f["key"] for f in r["schema"]}
    assert keys == set(r["defaults"]), "schema and defaults must cover the same keys"

def t_rejects_bad_value():
    code, r = req("POST", "/config", {"VAD_RMS": 99999})
    assert code == 400, code
    assert "VAD_RMS" in r.get("errors", {}), r

def t_rejects_leaves_state_untouched():
    _, before = req("GET", "/config")
    before_file = _read_config_file()
    req("POST", "/config", {"VAD_RMS": 99999})
    _, after = req("GET", "/config")
    assert before["values"] == after["values"], "a rejected POST must change nothing"
    assert _read_config_file() == before_file, "a rejected POST must not touch config.json"

def t_applies_and_persists():
    _, before = req("GET", "/config")
    orig = before["values"]["VAD_RMS"]
    target = 700.0 if orig != 700.0 else 800.0
    code, r = req("POST", "/config", {"VAD_RMS": target})
    assert code == 200, r
    assert "VAD_RMS" in r["applied"], r
    _, after = req("GET", "/config")
    assert after["values"]["VAD_RMS"] == target, after["values"]
    stored = json.load(open(os.path.expanduser("~/.config/heytux/config.json")))
    assert stored.get("VAD_RMS") == target, stored
    req("POST", "/config", {"VAD_RMS": orig})              # restore

def t_default_value_is_not_stored():
    _, r = req("GET", "/config")
    d = r["defaults"]["SILENCE_STOP_MS"]
    req("POST", "/config", {"SILENCE_STOP_MS": d + 50})
    req("POST", "/config", {"SILENCE_STOP_MS": d})
    stored = json.load(open(os.path.expanduser("~/.config/heytux/config.json")))
    assert "SILENCE_STOP_MS" not in stored, "a value back at its default must not be stored"

def t_unknown_key_rejected():
    code, r = req("POST", "/config", {"NOPE": 1})
    assert code == 400 and "NOPE" in r.get("errors", {}), r

def t_text_dry_run_returns_route_and_empty_said():
    # dry=true decides the route (executes nothing, speaks nothing): a real command routes to
    # tool with said="". The wake_skip_s invariant is now unit-tested in test_heytux.py.
    code, r = req("POST", "/text", {"text": "turn on the flashlight", "dry": True})
    assert code == 200, r
    assert r["route"] == "tool", r
    assert r["said"] == "", r

def t_models_shape():
    code, r = req("GET", "/models")
    assert code == 200, code
    assert "busy" in r and "slots" in r, r
    for slot in ("chat", "router", "asr", "tts", "vad"):
        s = r["slots"].get(slot)
        assert s is not None, "missing slot %s" % slot
        for k in ("field", "available", "current", "loaded", "rss_mb"):
            assert k in s, "slot %s missing %s" % (slot, k)

def t_restart_class_isolation():
    """A restart-class field (PORT) must apply to CFG/disk and show up as pending, but must NOT
    take effect on the running process: the daemon has to still answer on its original port."""
    code, r = req("POST", "/config", {"PORT": 8999})
    assert code == 200, r
    assert "PORT" in r["applied"], r
    assert "PORT" in r["pending_restart"], r["pending_restart"]
    code2, _ = req("GET", "/config")            # still served on BASE (unchanged port)
    assert code2 == 200, code2
    req("POST", "/config", {"PORT": 8770})       # restore (back to default -> not persisted)

# Live test (needs the daemon): only runs as a script, so pytest can import it offline without
# hitting the network.
if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        check(name[2:], fn)
    print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
