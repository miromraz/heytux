#!/usr/bin/env python3
# heytux.py - offline push-to-talk voice assistant daemon for Pixel 4a / BengalOS.
# Pipeline: mic (Silero VAD, energy fallback) -> normalize -> Parakeet/Moonshine STT -> Needle3 -> execute -> Piper TTS ack.
# stdlib + numpy + sherpa_onnx + ctypes only. Runs as miro's systemd --user service,
# so all subprocesses (pw-record/pw-play/wpctl/gio/busctl/nmcli/bluetoothctl) inherit miro's session.
import contextlib, ctypes, fcntl, glob, hashlib, json, math, os, queue, re, subprocess, sys, \
    threading, time, traceback, unicodedata, urllib.request, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import heytux_config as JC
import heytux_backends as HB

# ---------- paths / constants (config-tunable values live in CFG, not here) ----------
NEEDLE_DIR = "/home/miro/needle-arm64"
NEEDLE_IDX_DIR = "/home/miro/heytux"     # idx name is derived from the merged tool table, see needle_idx_path()
THREADS = 4
SR = 16000
FRAME = 480                      # 30 ms @ 16 kHz
WAKE_CUE_NOTE_S = 0.07           # each wake-cue note's length; shared so wake_skip_s() and Models._cue() agree

FLASH = "/sys/class/leds/white:flash/brightness"
BACKLIGHT = "/sys/class/backlight/ae94000.dsi.0"
BATTERY = "/sys/class/power_supply/qcom_qg"
WIFI_DEV = "wld0"

KWS_DIR = "/home/miro/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KWS_KEYWORDS = "/home/miro/heytux/hey.txt"     # deployed with the daemon; "▁HE Y ▁LI N U X @HEY_LINUX"
KWS_CHUNK = 3200                               # bytes = 100 ms s16 mono @ 16 kHz
CAPTURE_DIR = "/home/miro/heytux/captures"     # fixed location; only the ring size (CAPTURE_KEEP) is configurable
MANIFEST_PATH = os.environ.get("HEYTUX_MANIFEST", "/home/miro/heytux/bench/manifest.json")  # model registry

# The live-swappable model slots and the config field that selects each. tts_alt is the optional
# second voice (TTS_MODEL_ALT="" means off); like vad it may resolve to no model.
MODEL_FIELDS = {"CHAT_MODEL": "chat", "ROUTER_MODEL": "router", "ASR_MODEL": "asr",
                "TTS_MODEL": "tts", "TTS_MODEL_ALT": "tts_alt", "VAD_MODEL": "vad"}
FIELD_FOR_SLOT = {v: k for k, v in MODEL_FIELDS.items()}
SWAP_BUSY = threading.Event()                   # set while a hot model swap is in progress (for /models)
SWAP_LOCK = threading.Lock()                    # serialises swaps; a second request gets 409, not a queue

WAKE_ON = threading.Event()
WAKE_ON.set()
KWS_RELEASE = threading.Event()                 # set => KWS must release the mic (for /listen)
KWS_IDLE = threading.Event()                    # set => KWS is not holding the mic
KWS_IDLE.set()
wake_hits = 0
last_pipeline_end = 0.0

def log(*a):
    print("[heytux]", *a, file=sys.stderr, flush=True)

@contextlib.contextmanager
def timed(ms, key):
    """Record elapsed ms of the block into ms[key]."""
    t = time.time()
    yield
    ms[key] = int(1000 * (time.time() - t))

# ---------- config: CFG is the one source of truth, read at every use site ----------
CFG = JC.defaults()              # replaced by JC.load() at startup; defaults keep `import heytux` usable off-device
CFG_LOCK = threading.Lock()      # guards CFG mutation + save
_STARTED_WITH = {}               # snapshot of restart/model-class values at process start

def set_wake(on):
    """The one place the WAKE_ON Event is synced to CFG. POST /wake and POST /config both go
    through here so the Event and CFG can never disagree about whether the spotter is enabled."""
    CFG["WAKE_ON"] = on
    WAKE_ON.set() if on else WAKE_ON.clear()

# ---------- model registry: the manifest, filtered to what is actually staged on disk ----------
REGISTRY = {}        # id -> manifest entry, present-on-disk only
SLOT_IDS = {}        # slot -> [ids] (manifest order), present-on-disk only

def _model_base(entry):
    """The directory a manifest entry's `files` are listed relative to: the path itself when it is a
    directory, else the path's parent (gguf entries point at the .gguf file inside a model dir)."""
    p = entry["path"]
    return p if os.path.isdir(p) else os.path.dirname(p)

def _entry_present(entry):
    """True if the model's path and every file it lists exist on disk."""
    if not os.path.exists(entry["path"]):
        return False
    base = _model_base(entry)
    return all(os.path.exists(os.path.join(base, f)) for f in entry.get("files", []))

def load_registry():
    """Read the manifest and index only the entries actually staged on disk. Never raises: a missing
    or malformed manifest leaves REGISTRY/SLOT_IDS empty and model building falls back accordingly."""
    global REGISTRY, SLOT_IDS
    REGISTRY, SLOT_IDS = {}, {}
    try:
        entries = json.load(open(MANIFEST_PATH))
    except Exception as e:
        log("manifest %s unreadable (%s); model registry empty" % (MANIFEST_PATH, e))
        return
    for e in entries:
        if not isinstance(e, dict) or "id" not in e or not _entry_present(e):
            continue
        REGISTRY[e["id"]] = e
        SLOT_IDS.setdefault(e.get("slot"), []).append(e["id"])

# Only families the sherpa adapters (heytux_backends) actually load are offered - run1 recorded the
# others crashing the worker (kitten/matcha/supertonic TTS, streaming zipformer ASR, TEN VAD all
# failed, some rc 255). Offering them would let a swap hard-fail or brick the daemon.
SUPPORTED_FAMILY = {"asr": {"Moonshine", "NeMo-Parakeet-TDT", "Whisper"},
                    "tts": {"Piper-VITS", "Kokoro"},
                    "vad": {"Silero-VAD"}}

def slot_ids(slot):
    """Installed ids for a slot, filtered to the architectures the adapters support (asr/tts/vad);
    chat/embedding/router are llama.cpp and pass through."""
    ids = SLOT_IDS.get(slot, [])
    fams = SUPPORTED_FAMILY.get(slot)
    return ids if fams is None else [i for i in ids if REGISTRY[i].get("family") in fams]

def router_ids():
    """ROUTER_MODEL candidates: every present router-slot entry (Needle3 and the function-calling
    GGUFs) plus every present llama.cpp embedding entry (each usable as an embedding-classifier
    router). ONNX embeddings are skipped - there is no in-daemon adapter to host them."""
    emb = [i for i in SLOT_IDS.get("embedding", []) if REGISTRY[i].get("runtime") == "llama.cpp"]
    return SLOT_IDS.get("router", []) + emb

def _id_for_path(path):
    """The id of the installed manifest entry whose path matches, for migrating an old STT_DIR/TTS_DIR."""
    p = (path or "").rstrip("/")
    for e in REGISTRY.values():
        if e["path"].rstrip("/") == p:
            return e["id"]
    return None

def resolve_id(field, ids):
    """The id to actually load for a model field: the configured id if installed, else the schema
    default if installed, else the first installed candidate; any substitution is logged. None when
    nothing for the slot is on disk."""
    want = CFG.get(field)
    if want in ids:
        return want
    dflt = JC.FIELDS[field]["default"]
    if dflt in ids:
        log("model %s: %r not installed, using default %r" % (field, want, dflt))
        return dflt
    if ids:
        log("model %s: neither %r nor default %r installed, using %r" % (field, want, dflt, ids[0]))
        return ids[0]
    log("model %s: no installed candidate (wanted %r)" % (field, want))
    return None

def discover_models():
    """(Re)load the registry and fill the model enum fields' value lists from what is staged on disk,
    so the settings app only ever offers installed models and validate() rejects uninstalled ids."""
    load_registry()
    JC.FIELDS["CHAT_MODEL"]["values"] = slot_ids("chat")
    JC.FIELDS["ASR_MODEL"]["values"] = slot_ids("asr")
    JC.FIELDS["TTS_MODEL"]["values"] = slot_ids("tts")
    JC.FIELDS["TTS_MODEL_ALT"]["values"] = slot_ids("tts") + [""]   # "" = off
    JC.FIELDS["VAD_MODEL"]["values"] = slot_ids("vad")
    JC.FIELDS["ROUTER_MODEL"]["values"] = router_ids()

def migrate_config(stored):
    """Map a pre-registry config.json onto the new fields: an old STT_DIR/TTS_DIR path becomes the
    ASR_MODEL/TTS_MODEL id of the installed manifest entry with that path; an unmapped path is
    dropped (the field then defaults). Returns a new dict with the old keys removed."""
    s = dict(stored or {})
    for old, field in (("STT_DIR", "ASR_MODEL"), ("TTS_DIR", "TTS_MODEL")):
        path = s.pop(old, None)
        if field in s or not isinstance(path, str):
            continue
        mid = _id_for_path(path)
        if mid:
            s[field] = mid
            log("migrated %s=%s -> %s=%s" % (old, path, field, mid))
        else:
            log("migrated %s=%s: no installed manifest entry, %s will default" % (old, path, field))
    return s

def sync_kws_text(text):
    """Persist KWS_TEXT to the keyword file the spotter loads. model-class: only takes effect on
    the next restart, so this just keeps the file in sync with CFG."""
    if not text.endswith("\n"):
        text += "\n"
    try:
        with open(KWS_KEYWORDS, "w") as f:
            f.write(text)
    except OSError as e:
        log("failed to write KWS keywords file: %r" % e)

def load_config():
    global CFG, MERGED_TOOLS, TOOL_VOCAB
    discover_models()
    CFG = JC.overlay(migrate_config(JC._read_json(JC.CONFIG_PATH, dict)))
    set_wake(CFG["WAKE_ON"])
    sync_kws_text(CFG["KWS_TEXT"])
    MERGED_TOOLS = merged_tools()
    TOOL_VOCAB = _build_tool_vocab(MERGED_TOOLS)
    _STARTED_WITH.update({k: CFG[k] for k, f in JC.FIELDS.items() if f["apply"] in ("restart", "model")})
    log("config loaded from %s" % JC.CONFIG_PATH)

def pending_restart():
    """Keys whose saved value differs from what this process started with. Covers both apply
    classes that need a restart: 'restart' (a new global takes effect only at startup) and
    'model' (KWS_TEXT - takes effect only when the spotter is rebuilt)."""
    return [k for k, f in JC.FIELDS.items()
            if f["apply"] in ("restart", "model") and CFG.get(k) != _STARTED_WITH.get(k)]

def live_values():
    return {k: (WAKE_ON.is_set() if k == "WAKE_ON" else CFG[k]) for k in JC.FIELDS}

def wake_skip_s():
    """How much of the just-opened mic stream to discard before it reaches the VAD/STT: the wake
    cue's own length (so the cue is never captured), plus WAKE_SKIP_S as a margin. Zero when the
    cue plays no tone (haptic/off - the mic can't hear those). Uses JC.parse_cue_freqs(), the same
    parser Models._cue() uses, so the two can't diverge."""
    if CFG["CUE_MODE"] not in ("sound", "both"):
        return 0.0
    freqs = JC.parse_cue_freqs(CFG["CUE_WAKE_HZ"])
    return len(freqs) * WAKE_CUE_NOTE_S + CFG["WAKE_SKIP_S"] if freqs else CFG["WAKE_SKIP_S"]

# ---------- session env for subprocesses ----------
def sess_env():
    e = dict(os.environ)
    e.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())
    e.setdefault("DBUS_SESSION_BUS_ADDRESS", "unix:path=%s/bus" % e["XDG_RUNTIME_DIR"])
    e.setdefault("WAYLAND_DISPLAY", "wayland-0")
    return e

def run(cmd, timeout=8, check=False):
    return subprocess.run(cmd, env=sess_env(), capture_output=True, text=True,
                          timeout=timeout, check=check)

def pw_record_cmd():
    cmd = ["pw-record", "--rate", str(SR), "--channels", "1", "--format", "s16"]
    if CFG["MIC_TARGET"]:
        cmd += ["--target", CFG["MIC_TARGET"]]
    return cmd + ["-"]

# ---------- overlay: one live notification banner per voice turn over org.freedesktop.Notifications ----------
def _gv_str(s):
    """Escape s as a GVariant text-format string literal. `gdbus call` re-parses every CLI argument
    through GVariant/GOption, which bit three ways (all measured with dbus-monitor): a naive quoted
    summary lost its quotes on the wire, an argument starting with '-' was read as a flag, and one
    ending in '\\' failed as "unknown keyword". Wrapping every argument as an escaped literal fixes
    all three: the token always starts with '"', and a real backslash/quote survives as \\\\ / \\"."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

def _notify(summary, body="", replaces_id=0, expire_ms=0, timeout=8):
    """Show/replace the banner. Returns the new notification id, or 0 on any failure."""
    try:
        out = run(["gdbus", "call", "--session", "--dest", "org.freedesktop.Notifications",
                   "--object-path", "/org/freedesktop/Notifications",
                   "--method", "org.freedesktop.Notifications.Notify",
                   "heytux", str(replaces_id), "audio-input-microphone",
                   _gv_str(summary), _gv_str(body), "[]", "{}", str(expire_ms)], timeout=timeout)
        m = re.search(r"uint32 (\d+)", out.stdout)
        return int(m.group(1)) if m else 0
    except Exception as e:
        log("overlay notify error:", repr(e))
        return 0

def _haptic(event):
    """Fire one feedbackd event by name via org.sigxcpu.Feedback.TriggerFeedback (the session
    service, so it respects the user's feedback profile/mute). timeout=-1 means play once and stop.
    Never raises into the pipeline: a missing feedbackd must degrade to silence."""
    try:
        run(["gdbus", "call", "--session", "--dest", "org.sigxcpu.Feedback",
             "--object-path", "/org/sigxcpu/Feedback",
             "--method", "org.sigxcpu.Feedback.TriggerFeedback",
             "heytux", event, "{}", "@i -1"], timeout=2)
    except Exception as e:
        log("haptic cue error:", repr(e))

def _close_notification(replaces_id, timeout=8):
    try:
        run(["gdbus", "call", "--session", "--dest", "org.freedesktop.Notifications",
             "--object-path", "/org/freedesktop/Notifications",
             "--method", "org.freedesktop.Notifications.CloseNotification",
             str(replaces_id)], timeout=timeout)
    except Exception as e:
        log("overlay notify error:", repr(e))

# One FIFO queue + one worker guarantees updates are sent in the order pipeline() enqueues them.
# Two separately-started `gdbus call` processes could resolve in either order (the gap is just
# needle+act, 130-500 ms), so a step-2 update landing after step-3 would leave a stuck banner.
_overlay_q = queue.Queue()

def _overlay_worker():
    while True:
        fn, args = _overlay_q.get()
        try:
            fn(*args)
        except Exception as e:
            log("overlay notify error:", repr(e))
        finally:
            _overlay_q.task_done()

def _notify_async(summary, body, replaces_id, expire_ms):
    """Queue a replace; the worker makes the blocking gdbus call, so this adds no pipeline latency.
    The put itself is guarded so a notify failure can never reach the pipeline."""
    try:
        _overlay_q.put_nowait((_notify, (summary, body, replaces_id, expire_ms)))
    except Exception as e:
        log("overlay notify error:", repr(e))

def overlay_clear(notif_id):
    """Queue a close for a turn that errored before showing an outcome. Queued (not fired directly)
    so it lands strictly after any update already queued for this turn."""
    if not notif_id:
        return
    try:
        _overlay_q.put_nowait((_close_notification, (notif_id,)))
    except Exception as e:
        log("overlay notify error:", repr(e))

def overlay_listening():
    """Overlay step 1: 'Listening...' the moment capture is about to start. Synchronous (short
    timeout) since the id is needed to replace this banner later. 0 when overlay off or notify failed."""
    if not CFG["OVERLAY_ON"]:
        return 0
    return _notify("Listening…", "", 0, 0, timeout=0.15)

# ---------- tools: each tool's spec and handler are registered together (<=one action per tool) ----------
APPS = {
    "browser": "firefox-esr", "camera": "org.gnome.Snapshot",
    "calculator": "org.gnome.Calculator", "calendar": "org.gnome.Calendar",
    "clock": "org.gnome.clocks", "terminal": "org.gnome.Console",
    "contacts": "org.gnome.Contacts", "settings": "org.gnome.Settings",
    "music": "com.github.neithern.g4music", "messages": "sm.puri.Chatty",
    "phone": "org.gnome.Calls", "photos": "org.gnome.Loupe",
    "weather": "org.gnome.Weather", "app store": "io.github.kolunmi.Bazaar",
}
def match_app(name):
    """Map a free-text spoken app name to (canonical name, desktop id). None desktop if unknown."""
    if not name:
        return name, None
    q = name.strip().lower().replace(" app", "").strip()
    if q in APPS:
        return q, APPS[q]
    for k in APPS:                          # substring either direction (e.g. "app store" / "store")
        if k in q or q in k:
            return k, APPS[k]
    return name, None

def _fn(name, desc, props=None, required=None):
    p = {"type": "object", "properties": props or {}}
    if required:
        p["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}

TOOLS = []          # ordered specs sent to Needle3 (order was tuned; must stay byte-identical)
_HANDLERS = {}      # name -> handler(args) -> spoken ack

def tool(name, desc, props=None, required=None):
    """Register a tool: append its spec to TOOLS (in file order) and bind its handler in _HANDLERS."""
    spec = _fn(name, desc, props, required)
    def deco(fn):
        TOOLS.append(spec)
        _HANDLERS[name] = fn
        return fn
    return deco

def _torch(value):
    """Write the torch LED brightness. Raises OSError like the bare open() it replaced (callers
    that must tolerate a missing LED catch it themselves)."""
    with open(FLASH, "w") as f:
        f.write(value)

def _pct(v):
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return None

# Split zero-argument on/off tools (flashlight/wifi/bluetooth) route far better on Needle3 than a
# parameterized set_radio(radio,state), which the model dumped into flashlight_off. See README.
@tool("flashlight_on", "Turn on the flashlight torch LED.")
def _flashlight_on(args):
    _torch("255"); return "Flashlight on."

@tool("flashlight_off", "Turn off the flashlight torch LED.")
def _flashlight_off(args):
    _torch("0"); return "Flashlight off."

@tool("wifi_on", "Turn on wifi.")
def _wifi_on(args):
    run(["nmcli", "radio", "wifi", "on"]); return "Wi-Fi on."

@tool("wifi_off", "Turn off wifi.")
def _wifi_off(args):
    run(["nmcli", "radio", "wifi", "off"]); return "Wi-Fi off."

@tool("bluetooth_on", "Turn on bluetooth.")
def _bluetooth_on(args):
    run(["bluetoothctl", "power", "on"]); return "Bluetooth on."

@tool("bluetooth_off", "Turn off bluetooth.")
def _bluetooth_off(args):
    run(["bluetoothctl", "power", "off"]); return "Bluetooth off."

@tool("set_brightness", "Set the screen brightness to a percent from 0 to 100.",
      {"percent": {"type": "integer", "minimum": 0, "maximum": 100}}, ["percent"])
def _set_brightness(args):
    pct = _pct(args.get("percent"))
    if pct is None:
        return "I need a brightness level."
    val = round(pct / 100 * int(_read(BACKLIGHT + "/max_brightness", "1023")))
    run(["busctl", "call", "org.freedesktop.login1",
         "/org/freedesktop/login1/session/" + display_session(),
         "org.freedesktop.login1.Session", "SetBrightness", "ssu",
         "backlight", os.path.basename(BACKLIGHT), str(val)])
    return "Brightness %d percent." % pct

@tool("set_volume", "Set the media volume percent, or mute or unmute it.",
      {"action": {"type": "string", "enum": ["set", "mute", "unmute"]},
       "percent": {"type": "integer", "minimum": 0, "maximum": 100}}, ["action"])
def _set_volume(args):
    act = args.get("action")
    if act == "mute":
        run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"]); return "Muted."
    if act == "unmute":
        run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"]); return "Unmuted."
    pct = _pct(args.get("percent"))
    if pct is None:
        return "I need a volume level."
    run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "%.2f" % (pct / 100)])
    return "Volume %d percent." % pct

@tool("battery_status", "Say the current battery charge level.")
def _battery_status(args):
    # read sysfs directly; never call phone_info() here (it would query wifi)
    return "Battery is at %s percent and %s." % (
        _read(BATTERY + "/capacity", "unknown"), _read(BATTERY + "/status", "unknown").lower())

@tool("phone_info", "Report this phone's status: battery, storage, memory, temperature and wifi network.")
def _phone_info(args):
    i = phone_info()
    wifi = (i["wifi"] or "off").replace("_", " ")
    return ("Battery %s percent, %s. %d gigabytes free, %d memory, %s degrees, Wi-Fi %s." % (
            i["battery_percent"], i["battery_status"], round(i["storage_free_gb"]),
            round(i["memory_free_gb"]),
            i["cpu_temp_c"] if i["cpu_temp_c"] is not None else "unknown", wifi))

@tool("set_timer", "Start a countdown timer for some minutes or seconds, optional label.",
      {"minutes": {"type": "integer", "minimum": 0, "maximum": 1440},
       "seconds": {"type": "integer", "minimum": 0, "maximum": 86400},
       "label": {"type": "string"}})
def _set_timer(args):
    try:
        secs = int(args.get("minutes") or 0) * 60 + int(args.get("seconds") or 0)
    except (TypeError, ValueError):
        secs = 0
    if secs <= 0:
        return "I need a timer length."
    label = (args.get("label") or "").strip()
    tmr = threading.Timer(secs, _timer_fire, [label]); tmr.daemon = True; tmr.start()
    m, s = divmod(secs, 60)
    parts = ["%d minute%s" % (m, "" if m == 1 else "s")] if m else []
    parts += ["%d second%s" % (s, "" if s == 1 else "s")] if s else []
    return "Timer set for %s%s." % (" ".join(parts), " for " + label if label else "")

@tool("what_time_is_it", "What is the time, what day is it, what is the date today.")
def _what_time_is_it(args):
    return time.strftime("It is %-I:%M %p on %A, %B %-d.")

@tool("open_app", "Launch an installed app by name, e.g. browser, camera, calculator, calendar, "
      "clock, terminal, contacts, settings, music, messages, phone, photos, weather.",
      {"app": {"type": "string"}}, ["app"])
def _open_app(args):
    app, desktop = match_app(args.get("app"))
    path = "/usr/share/applications/%s.desktop" % desktop if desktop else None
    if not path or not os.path.exists(path):
        return "I don't have that app."
    subprocess.Popen(["gio", "launch", path], env=sess_env(),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "Opening %s." % app

# No headless still-capture path on this image (no gst-launch, no libcamera CLI or python bindings,
# raw v4l2 STREAMON fails on the CSI nodes), so both photo tools just open the camera app.
def _open_camera(selfie):
    subprocess.Popen(["gio", "launch", "/usr/share/applications/org.gnome.Snapshot.desktop"],
                     env=sess_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "Opening the camera for a selfie." if selfie else "Opening the camera."

@tool("take_photo", "Photograph what is in front of you with the rear camera.")
def _take_photo(args):
    return _open_camera(False)

@tool("take_selfie", "Take a selfie of yourself with the front camera.")
def _take_selfie(args):
    return _open_camera(True)

@tool("lock_screen", "Lock the phone, blank the screen.")
def _lock_screen(args):
    run(["loginctl", "lock-session", display_session()]); return "Screen locked."

def _media(method, ack):
    player = mpris_player()
    if not player:
        return "Nothing is playing."
    run(["busctl", "--user", "call", player, "/org/mpris/MediaPlayer2",
         "org.mpris.MediaPlayer2.Player", method])
    return ack

@tool("play_pause", "Play or pause the current music or media playback.")
def _play_pause(args):
    return _media("PlayPause", "Done.")

@tool("next_track", "Skip to the next music track or song.")
def _next_track(args):
    return _media("Next", "Next track.")

@tool("find_my_phone", "Find the phone by playing a loud sound and flashing the light.")
def _find_my_phone_tool(args):
    threading.Thread(target=_find_my_phone, daemon=True).start()
    return "Playing a sound and flashing the light."

# ask_question has no handler: decide_route always sends it to the chat LLM, never to execute().
TOOLS.append(_fn("ask_question", "Ask the assistant a question or say something to it. Use for "
                 "questions, facts, jokes, stories, greetings and conversation. question is the "
                 "user's words verbatim.", {"question": {"type": "string"}}, ["question"]))

def execute(name, args):
    """Dispatch a fired tool to its handler. Only ever called for real tool names (grounding
    rejects confabulated ones), so the fallback is defensive."""
    h = _HANDLERS.get(name)
    return h(args or {}) if h else "I can't do that."

def _timer_fire(label):
    with LOCK:
        M.speak("%s timer is done." % (label if label else "Your"))
    for _ in range(2):
        try:
            _torch("255"); time.sleep(0.25)
            _torch("0"); time.sleep(0.25)
        except OSError:
            break

def mpris_player():
    """First MPRIS media player bus name, or None (playerctl is not installed)."""
    try:
        out = run(["busctl", "--user", "list", "--acquired", "--no-legend"]).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("org.mpris.MediaPlayer2."):
            return parts[0]
    return None

def _get_volume():
    """(volume 0..1, muted) for the default sink via wpctl. Defaults to (0.30, False) on parse fail."""
    out = run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"]).stdout
    m = re.search(r"([0-9]+\.[0-9]+)", out)
    return (float(m.group(1)) if m else 0.30), "MUTED" in out

def _find_my_phone():
    """Loud repeating beep + torch flashing for ~4 s, then restore the prior volume and torch.
    Detached so the spoken ack returns first; caps the alarm at 45 % per hardware safety."""
    prev_vol, prev_muted = _get_volume()
    prev_torch = _read(FLASH, "0")
    try:
        run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"])
        run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "0.45"])
        beep = 0.6 * np.sin(2 * np.pi * 1000 * np.arange(int(0.25 * SR)) / SR)
        one = np.concatenate([beep, np.zeros(int(0.15 * SR))])
        alarm = (np.tile(one, 10) * 32767).clip(-32768, 32767).astype("<i2")   # ~4 s
        with LOCK:
            M._play(alarm, SR)
        for _ in range(10):
            try:
                _torch("255"); time.sleep(0.2)
                _torch("0"); time.sleep(0.2)
            except OSError:
                break
        M.await_playback()
    finally:
        try:
            _torch(prev_torch)
        except OSError:
            pass
        run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "%.2f" % prev_vol])
        run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if prev_muted else "0"])

# ---------- phone info ----------
def _read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default

def cpu_temp_c():
    for tp in glob.glob("/sys/class/thermal/thermal_zone*/type"):
        if _read(tp) == "cpu7-thermal":
            return round(int(_read(tp.replace("type", "temp"), "0")) / 1000)
    return None

def wifi_ssid():
    # Read the active connection WITHOUT triggering a scan (a scan cost phone_info ~4 s).
    try:
        out = run(["nmcli", "-t", "-f", "GENERAL.CONNECTION", "device", "show", WIFI_DEV]).stdout.strip()
        if ":" in out:
            name = out.split(":", 1)[1].strip()
            return name or None
    except Exception:
        pass
    return None

def phone_info():
    import shutil
    cap = _read(BATTERY + "/capacity", "?")
    status = _read(BATTERY + "/status", "?").lower()
    free_gb = round(shutil.disk_usage("/home").free / 1e9, 1)
    memavail = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable"):
            memavail = int(line.split()[1])
    return {"battery_percent": cap, "battery_status": status,
            "storage_free_gb": free_gb, "memory_free_gb": round(memavail / 1e6, 1),
            "cpu_temp_c": cpu_temp_c(), "wifi": wifi_ssid()}

def display_session():
    # systemd --user unit has no seat; resolve the display session for loginctl/busctl brightness.
    try:
        return run(["loginctl", "show-user", str(os.getuid()), "-p", "Display", "--value"]).stdout.strip()
    except Exception:
        return "auto"

# ---------- chat fallback (local LLM: heytux-llm.service on :8771) ----------
LLM_URL = "http://127.0.0.1:8771/v1/chat/completions"
CHAT_FAIL = "I can't answer that right now."

def clean_spoken(t):
    """Make LLM output safe to speak: drop markdown and emoji, collapse space, first 2 sentences.
    Keeps non-ASCII LETTERS (so Czech diacritics survive for the alt voice and lang detection) while
    still dropping non-ASCII symbols/emoji (unicode category S*)."""
    t = re.sub(r"[*_`#>~\[\]]", "", t or "")
    t = "".join(ch for ch in t if ord(ch) < 128 or unicodedata.category(ch)[0] == "L")
    t = re.sub(r"\s+", " ", t).strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:2]).strip()
    return t[:220].strip()

# Reply-language detection for voice selection: the chat model (told to answer in the user's
# language when LANG_AUTO) picks the language, and the spoken voice follows the reply TEXT. A small
# stdlib heuristic - Czech-specific diacritics, or a couple of common Czech function words - is
# enough to pick between the English main voice and the Czech alt voice.
LANG_HINT = "Answer in the same language the user spoke."
_CZ_DIACRITICS = set("ěščřžůťňďý")
# Deliberately NO single letters and no English homographs ("a", "i", "to", "u"...): those appear in
# ordinary English and would misclassify an English reply as Czech. Only multi-letter Czech words
# that are not also English words, so the no-diacritics fallback needs real Czech evidence.
_CZ_STOPWORDS = frozenset("""
je jsem jsi jsme jste jsou byl byla bylo není
na se si že co kdo kde kdy jak proč když jako
ten tato toto můj moje tvůj náš také
ano dobře děkuji prosím dnes teď hodin stupňů baterie telefon
""".split())
_CZ_WORD_RE = re.compile(r"[a-zěščřžýáíéúůťňďó]+")

def is_czech(text):
    """True if the reply text looks Czech: any Czech-only diacritic, or >=2 DISTINCT unambiguous
    Czech words (distinct so a repeated English article can't reach the threshold)."""
    t = (text or "").lower()
    if any(ch in _CZ_DIACRITICS for ch in t):
        return True
    return len(set(_CZ_WORD_RE.findall(t)) & _CZ_STOPWORDS) >= 2

def _live_facts():
    """Cheap sysfs-only facts injected into the LLM prompt so a fallback answers a low-confidence
    real query correctly (e.g. 'what time is it' misrouted to chat). No nmcli / no scans."""
    return "Current date and time: %s. Phone battery: %s percent, %s." % (
        time.strftime("%A, %B %-d, %Y, %H:%M"), _read(BATTERY + "/capacity", "?"),
        _read(BATTERY + "/status", "?").lower())

# Rolling chat memory so "explain it" follows "tell me a joke": (user, answer) pairs of CHAT turns
# only (a tool turn isn't conversation). Only touched via pipeline(), whose every caller (wake,
# /listen, /text) holds LOCK, so no lock of its own.
CHAT_HISTORY = []
chat_last = 0.0                  # when the last remembered chat answer arrived

def chat_messages(sys_prompt, history, transcript):
    """system, then the history pairs oldest first, then the current question."""
    msgs = [{"role": "system", "content": sys_prompt}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": transcript}]

def _chat_history():
    """The pairs to send: none when memory is off or the last answer is older than CHAT_MEMORY_S
    (a new topic, not a follow-up), else the last CHAT_HISTORY_TURNS (it may be lowered live)."""
    n = CFG["CHAT_HISTORY_TURNS"]
    if n <= 0 or time.time() - chat_last > CFG["CHAT_MEMORY_S"]:
        CHAT_HISTORY.clear()
    return CHAT_HISTORY[-n:] if n > 0 else []

def _chat_sys():
    """Stable chat system prompt: LLM_SYS, plus the reply-language instruction when LANG_AUTO. Kept
    free of volatile facts so it (and the tool schemas after it) stay a cacheable prefix in hybrid."""
    return CFG["LLM_SYS"] + (" " + LANG_HINT if CFG["LANG_AUTO"] else "")

def _remember(transcript, answer):
    """Append a (user, answer) turn to the rolling chat memory. Tool turns are recorded too, so a
    follow-up like 'turn it off' after 'turn on the flashlight' has the prior turn as context. Drops
    an expired conversation first (same CHAT_MEMORY_S timeout as _chat_history), so a fast-path tool
    turn that never calls _chat_history can't resurrect stale context for the next chat request."""
    global chat_last
    if not (answer and CFG["CHAT_HISTORY_TURNS"] > 0):
        return
    if time.time() - chat_last > CFG["CHAT_MEMORY_S"]:
        CHAT_HISTORY.clear()
    CHAT_HISTORY.append((transcript, answer))
    del CHAT_HISTORY[:-CFG["CHAT_HISTORY_TURNS"]]
    chat_last = time.time()

def _llm_once(body):
    req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
    # 60 s: a big slow model (Qwen3.5-4B ~3.2 t/s) needs ~19 s for 60 tokens; 10 s timed out valid replies.
    r = json.load(urllib.request.urlopen(req, timeout=60))
    return clean_spoken(r["choices"][0]["message"]["content"])

def chat_llm(transcript, remember=True):
    """Ask the local LLM; return a cleaned spoken answer, or None on any error/timeout. A good
    answer is added to the chat memory unless remember=False (dry /text must not change it). The
    daemon owns the chat server now, so if the request fails ONLY because the process has actually
    died, reload it once (the old unit's Restart=on-failure) and retry; a live-but-slow server is left
    alone (no pointless second wait). Callers already hold LOCK, which ensure_chat needs."""
    body = json.dumps({
        "messages": chat_messages(_chat_sys() + " " + _live_facts(), _chat_history(), transcript),
        "max_tokens": 60, "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        ans = _llm_once(body)
    except Exception as e:
        log("chat_llm error:", repr(e))
        if M is None or M._slot_alive("chat"):
            return None                     # server alive (just slow/other error) - don't reload/retry
        try:
            with LOCK:
                M.ensure_chat()
            ans = _llm_once(body)
        except Exception as e2:
            log("chat_llm retry failed:", repr(e2))
            return None
    if ans and remember:
        _remember(transcript, ans)
    return ans or None

def hybrid_tools():
    """Enabled merged tools offered to the chat model in hybrid mode, minus ask_question (that is the
    router's chat sentinel, not a real tool - the chat model answers directly instead)."""
    return [t for t in MERGED_TOOLS if t["function"]["name"] != "ask_question"]

def chat_tool(transcript, dry=False):
    """Hybrid turn: ask the chat model with the enabled phone commands offered as tools. It either
    calls an enabled, known tool through execute() or answers in words. Explicit LLM calls are
    multilingual, so only the router fast path uses word-overlap grounding.
    Returns (did_tool, said, calls). remember/execute are skipped on a dry run."""
    tools = hybrid_tools()
    enabled = {t["function"]["name"] for t in tools}
    # No _live_facts here: injecting the date/battery preamble into the user turn derails tool
    # selection (weak models echo it or pick phone_info). In hybrid the model has what_time_is_it /
    # battery_status / phone_info tools for those facts instead. The system+tools prefix stays stable,
    # so cache_prompt caches it across turns.
    msgs = chat_messages(_chat_sys(), _chat_history(), transcript)
    try:
        res = M.chat.tool_chat(msgs, tools)
    except Exception as e:
        log("chat_tool error:", repr(e))
        if M is None or M._slot_alive("chat"):
            return False, CHAT_FAIL, []
        try:
            with LOCK:
                M.ensure_chat()
            res = M.chat.tool_chat(msgs, tools)
        except Exception as e2:
            log("chat_tool retry failed:", repr(e2))
            return False, CHAT_FAIL, []
    proposed = res.get("calls") or []
    calls = []
    # Validate the whole response before executing anything: malformed JSON must not become
    # default arguments, and a bad later call must not crash after an earlier hardware action.
    for c in proposed:
        name = c.get("name")
        reason = ("unknown" if TOOL_VOCAB.get(name) is None else
                  "not enabled" if name not in enabled else
                  "bad arguments" if not isinstance(c.get("arguments"), dict) else None)
        if reason:
            log("chat_tool rejected:", name, reason)
        else:
            calls.append(c)
    if calls:
        said = "" if dry else " ".join(a for a in (execute(c.get("name"), c.get("arguments"))
                                                   for c in calls) if a)
        if not dry:
            _remember(transcript, said)
        return True, said, calls
    ans = clean_spoken(res.get("content") or "")
    if not ans:
        return False, CHAT_FAIL, []
    if not dry:
        _remember(transcript, ans)
    return False, ans, []

# ---------- off-topic / grounding gates ----------
# Needle3 ALWAYS emits the closest tool for a full toolset (it says so in `reasoning`, e.g.
# "factual question -> no tool for capital, but flashlight_off is the closest match"), and its
# confidence does not discriminate off-topic. So when the reasoning admits no tool matches we route
# to chat instead of executing. Real commands never contain these phrases. The real fix is a LoRA
# fine-tune with answers:[] examples (Needle guide).
NOMATCH = ("no tool", "no matching", "not a phone", "factual question",
           "closest match", "no function", "there is no")

def _off_topic(reasoning):
    r = (reasoning or "").lower()
    return any(p in r for p in NOMATCH)

# Grounding: Needle3 also emits its closest tool for noise ("hmm" -> flashlight_on 0.88) at
# confidence that doesn't discriminate that ("When you please," -> next_track 1.00). Before acting
# we require one meaningful content word in common between what was said and the tool's own
# vocabulary (name + description); none means the router confabulated it.
_STOPWORDS = frozenset("""
a an the this that these those there here
i me my mine we us our ours you your yours he him his she her hers it its
they them their theirs who whom whose what which when where why how
is am are was were be been being do does did doing have has had having
can could shall should will would may might must
to of in on at by for with and but or if then so as than from up down out
about into onto over again please just like well now okay ok yeah yes no not
uh um hmm ah oh er hm mm anyway
""".split())
_WORD_RE = re.compile(r"[a-z0-9]+")

def _fold(w):
    """Light suffix folding so lights/light and photos/photo match. No stemmer - stdlib only."""
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def _content_words(text):
    t = re.sub(r"[-']", "", (text or "").lower())     # "Wi-Fi"/"don't" -> one token, not split
    return {_fold(w) for w in _WORD_RE.findall(t) if w not in _STOPWORDS and len(w) > 1}

def merged_tools():
    """TOOLS with ~/.config/heytux/tools.json overrides applied (disable a tool, override a
    description). This one merged table feeds the router, the grounding vocab AND the idx name, so
    a disabled/re-described tool can't reach one but not the others."""
    try:
        return JC.merge_tools(TOOLS, JC.load_tools())
    except Exception:
        log("merge_tools failed, using built-in TOOLS:\n" + traceback.format_exc())
        return TOOLS

def _build_tool_vocab(merged):
    """name -> content words, from name + description, so editing a description through the config
    layer updates grounding automatically - no hardcoded per-tool keyword list to rot."""
    out = {}
    for t in merged:
        fn = t["function"]
        out[fn["name"]] = _content_words(fn["name"].replace("_", " ") + " " + fn.get("description", ""))
    return out

MERGED_TOOLS = merged_tools()          # rebuilt in load_config() once tools.json is read at startup
TOOL_VOCAB = _build_tool_vocab(MERGED_TOOLS)

def _grounded(name, transcript):
    """True unless grounding is on, the tool isn't ask_question, and it shares no content word with
    the transcript. Never raises into the pipeline - any error grounds it (falls through to the
    CONF_FLOOR/NOMATCH gates) rather than breaking voice control."""
    if not CFG["GROUNDING_ON"] or name == "ask_question":
        return True
    try:
        vocab = TOOL_VOCAB.get(name)
        if vocab is None:          # confabulated tool name (e.g. "get_phone_phone") - not real
            return False
        return bool(vocab & _content_words(transcript))
    except Exception:
        log("grounding check error for %r/%r:\n" % (name, transcript) + traceback.format_exc())
        return True

# ---------- Needle3 (ctypes, one process-global model) ----------
def needle_idx_path(tools):
    """Idx name derived from a short sha256 of the tool table, so it is rebuilt automatically when
    tools change (the cache goes stale otherwise). Stale heytux*.idx are pruned at startup."""
    h = hashlib.sha256(json.dumps(tools).encode()).hexdigest()[:8]
    return os.path.join(NEEDLE_IDX_DIR, "heytux-%s.idx" % h)

def _prune_stale_idx(keep):
    for p in glob.glob(os.path.join(NEEDLE_IDX_DIR, "heytux*.idx")):
        if p != keep:
            try:
                os.unlink(p)
            except OSError:
                pass

class Needle:
    scored = False              # its generated confidence is not a similarity score
    def __init__(self, tools):
        self.lib = ctypes.CDLL(NEEDLE_DIR + "/libneedle.so")
        self.lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]; self.lib.needle_load.restype = ctypes.c_int
        self.lib.needle_init.argtypes = [ctypes.c_char_p] * 3; self.lib.needle_init.restype = ctypes.c_int
        self.lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.needle_complete.restype = ctypes.c_int
        self.lib.needle_last_error.restype = ctypes.c_char_p
        self.lib.needle_reset.restype = None
        cact = open(NEEDLE_DIR + "/needle3.cact", "rb").read()
        if self.lib.needle_load(cact, len(cact)) < 0:
            raise RuntimeError("needle_load: " + str(self.lib.needle_last_error()))
        idx = needle_idx_path(tools)
        _prune_stale_idx(idx)
        system = "date: %s; locale: en-US; device: phone" % time.strftime("%Y-%m-%d %a %H:%M")
        # Init with the MERGED table (not raw TOOLS): a tool disabled/re-described in tools.json
        # must reach the router, not just the grounding vocab.
        rc = self.lib.needle_init(system.encode(), json.dumps(tools).encode(), idx.encode())
        if rc < 0:
            raise RuntimeError("needle_init: " + str(self.lib.needle_last_error()))
        self.buf = ctypes.create_string_buffer(65536)

    def complete(self, text):
        rc = self.lib.needle_complete(text.encode(), 512, self.buf, len(self.buf))
        try:
            r = json.loads(self.buf.value.decode()) if rc >= 0 else {}
        except Exception:
            r = {}
        self.lib.needle_reset()
        return r

    def close(self):
        pass        # ctypes model with no unload API; dropping the reference is all we can do

# ---------- model registry -> router/rss helpers ----------
def make_router(mid):
    """Build the router backend for a ROUTER_MODEL id. Each returns .complete(transcript) in Needle's
    dict shape (function_calls/confidence/reasoning) and .close(): Needle3 -> the ctypes model; an
    embedding entry -> an embedding-classifier router; any other router-slot entry -> a GGUF
    function-caller. The grounding/CONF_FLOOR/NOMATCH gates in decide_route apply to all three."""
    e = REGISTRY[mid]
    if e.get("runtime") == "needle" or mid == "Needle3":
        # ponytail: Needle3 is a process-global ctypes model with no unload API, so switching away
        # can't free its RAM (close() is a no-op) and /models reports its startup delta, not a live
        # figure. Upgrade path: host it in its own worker process if per-swap RAM reclaim is needed.
        return Needle(MERGED_TOOLS)
    # Both router kinds route over the ENABLED merged tools (a disabled/re-described tool is honored),
    # and the started llama-server is closed if wrapper construction throws, never leaked.
    if e.get("slot") == "embedding":
        srv = HB.LlamaServer(e, embeddings=True, threads=2, ctx=2048)
        try:
            return HB.EmbeddingRouter(srv, MERGED_TOOLS, threshold=HB.EMB_ROUTER_FLOOR)
        except Exception:
            srv.close(); raise
    g = dict(e); g["path"] = HB.resolve_gguf(e["path"], "router")
    srv = HB.LlamaServer(g, threads=2, ctx=2048)
    try:
        return HB.GgufRouter(srv, MERGED_TOOLS)
    except Exception:
        srv.close(); raise

def _daemon_rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None

def _rss_delta(before):
    after = _daemon_rss_mb()
    return None if (before is None or after is None) else round(max(after - before, 0))

def _proc_rss_mb(pid):
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    except OSError:
        pass
    return None

def _obj_proc(obj):
    """The llama-server Popen behind a backend (chat/asr are the object; router wrappers hold it on
    .server), or None for in-process models (Needle, sherpa)."""
    return getattr(obj, "proc", None) or getattr(getattr(obj, "server", None), "proc", None)

def _obj_pid(obj):
    return getattr(obj, "pid", None) or getattr(getattr(obj, "server", None), "pid", None)

def _obj_alive(obj):
    """True if the backend exists and, when it is a subprocess, that process is still running."""
    if obj is None:
        return False
    proc = _obj_proc(obj)
    return (proc.poll() is None) if proc is not None else True

# ---------- models ----------
class Models:
    def __init__(self):
        import sherpa_onnx as so
        self.so = so                 # kept so _vad_build()/build_kws() build sherpa models inside Models
        self.vad = None              # one VAD instance, reset per capture; rebuilt on threshold/model change
        self._vad_threshold = None
        self._vad_id = None
        self._vad_entry = None
        self._vad_warned = False
        self._chat_prewarm_key = None
        self.loaded = {}             # slot -> actually-loaded manifest id (None = not loaded); /models truth
        self.rss_mb = {}             # slot -> best-effort RSS MB (daemon delta for in-process sherpa)
        self.chat = self.router = self.asr = self.tts = self.tts_alt = None
        self.play_proc = None        # last detached ack playback, awaited before the next mic capture
        for slot in ("chat", "router", "asr", "tts", "tts_alt", "vad"):
            self._load_slot(slot, self._resolve(slot))
        self.build_kws()

    def _resolve(self, slot):
        """The id to load for a slot (configured -> default -> first installed). Raises if a required
        slot has nothing installed; vad and tts_alt may resolve to None (disabled)."""
        if slot == "tts_alt":
            want = CFG["TTS_MODEL_ALT"]                 # "" (off), or an id; never falls back/raises
            return want if want in slot_ids("tts") else None
        ids = router_ids() if slot == "router" else slot_ids(slot)
        mid = resolve_id(FIELD_FOR_SLOT[slot], ids)
        if mid is None and slot != "vad":
            raise RuntimeError("no %s model installed for %s" % (slot, FIELD_FOR_SLOT[slot]))
        return mid

    def _vad_build(self, entry, th):
        """A sherpa VoiceActivityDetector for a VAD entry at threshold th (silero or TEN). Called at
        swap time so a broken model fails the swap, and lazily in vad_detector() when th changes."""
        so = self.so
        path = entry["path"]
        onnx = path if (os.path.isfile(path) and path.endswith(".onnx")) \
            else HB._one(path, "*int8.onnx", "*.onnx")
        if "ten" in entry["id"].lower():
            cfg = so.VadModelConfig(ten_vad=so.TenVadModelConfig(
                model=onnx, threshold=th, min_silence_duration=0.1,
                min_speech_duration=0.05, window_size=256), sample_rate=SR, num_threads=1)
        else:
            cfg = so.VadModelConfig(silero_vad=so.SileroVadModelConfig(
                model=onnx, threshold=th, min_silence_duration=0.1,
                min_speech_duration=0.05, window_size=512, max_speech_duration=20),
                sample_rate=SR, num_threads=1)
        return so.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

    # ---------- per-slot load/close (used by __init__ and swap()) ----------
    def _load_slot(self, slot, mid, strict=False):
        """Build the backend for `mid` and publish it (self.<slot>, loaded[slot], rss). Raises on
        failure with nothing published, so loaded[slot] is only set once the model is really up.
        mid None is valid for vad and tts_alt. Optional voice failures are non-fatal except for
        an explicit swap (strict), whose caller must see the error and roll back."""
        t = time.time(); before = _daemon_rss_mb()
        if slot == "vad":
            if mid is None:
                self.vad = self._vad_entry = self._vad_id = self._vad_threshold = None
            else:
                entry = REGISTRY[mid]; th = CFG["VAD_THRESHOLD"]
                self.vad = self._vad_build(entry, th)          # build NOW: a broken model fails the swap
                self._vad_entry = entry; self._vad_id = entry["id"]; self._vad_threshold = th
                self.rss_mb["vad"] = _rss_delta(before)
        elif slot == "chat":
            self._chat_prewarm_key = None       # a new server has no cached prefix, even for the same id
            # The chat LLM, hosted by the daemon on 127.0.0.1:8771 so LLM_URL stays valid (retires
            # heytux-llm.service). -t 2, ctx 2048 per the brief.
            e = dict(REGISTRY[mid]); e["path"] = HB.resolve_gguf(e["path"], "chat")
            self.chat = HB.LlamaServer(e, threads=2, ctx=2048, port=8771)
        elif slot == "router":
            r = make_router(mid)
            try:                                               # probe: reject a router that errors
                r.complete("turn on the flashlight")           # (e.g. xLAM 500'd in run1)
            except Exception as e:
                try: r.close()
                except Exception: pass
                raise RuntimeError("router probe failed: %s" % e)
            self.router = r
            if getattr(self.router, "pid", None) is None:      # in-process (Needle): daemon delta
                self.rss_mb["router"] = _rss_delta(before)
        elif slot in ("asr", "tts"):
            setattr(self, slot, HB.load(REGISTRY[mid], threads=THREADS))
            self.rss_mb[slot] = _rss_delta(before)
        elif slot == "tts_alt":
            if mid:
                try:
                    self.tts_alt = HB.load(REGISTRY[mid], threads=THREADS)
                    self.rss_mb["tts_alt"] = _rss_delta(before)
                except Exception as e:
                    self.tts_alt = None
                    self.loaded["tts_alt"] = None
                    self.rss_mb["tts_alt"] = None
                    if strict:
                        raise
                    log("tts_alt load failed (non-fatal):", repr(e))
                    return
            else:
                self.tts_alt = None                    # disabled: no second voice loaded
        self.loaded[slot] = mid
        log("%s (%s) loaded in %d ms%s" % (slot, mid, 1000 * (time.time() - t),
                                           " on :8771" if slot == "chat" else ""))
        if slot == "chat" and CFG["ROUTE_MODE"] == "hybrid":
            self._prewarm_chat()          # load / swap / crash-reload all funnel through here

    def _prewarm_chat(self):
        """Hybrid only: process the stable system+tools prefix once now, with cache_prompt, so the
        user's first real turn reuses the cache instead of paying the full ~1200-token cold prefill
        (20-60 s on this phone). Non-fatal - a failed prewarm just means the first turn is cold."""
        system, tools = _chat_sys(), hybrid_tools()
        key = (self.loaded.get("chat"), system, json.dumps(tools, sort_keys=True))
        if key == self._chat_prewarm_key:
            return
        try:
            t = time.time()
            self.chat.tool_chat(chat_messages(system, [], "hi"), tools,
                                max_tokens=1, cache_prompt=True, timeout=300)
            self._chat_prewarm_key = key             # failed attempts remain retryable
            log("chat prefix prewarmed in %d ms" % (1000 * (time.time() - t)))
        except Exception as e:
            log("chat prewarm failed (non-fatal):", repr(e))

    def _close_slot(self, slot):
        if slot == "chat":
            self._chat_prewarm_key = None
        if slot == "vad":
            self.vad = self._vad_entry = self._vad_id = self._vad_threshold = None
            return
        obj = getattr(self, slot, None)
        if obj is not None:
            try:
                obj.close()
            except Exception:
                log("closing old %s failed:\n" % slot + traceback.format_exc())
        setattr(self, slot, None)

    def build_kws(self):
        t = time.time()
        self.kws = self.so.KeywordSpotter(
            tokens=KWS_DIR + "/tokens.txt",
            encoder=KWS_DIR + "/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            decoder=KWS_DIR + "/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            joiner=KWS_DIR + "/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            num_threads=1, keywords_file=KWS_KEYWORDS, provider="cpu",
            # host bench, 40 TTS speakers, quiet (peak 0.08): 28/40 at the defaults (4, 0.25, 1.0)
            # -> 35/40 here, near-miss false alarms unchanged; more paths raises false alarms
            max_active_paths=8, keywords_threshold=0.15, keywords_score=1.5)
        log("KWS (zipformer wake word) loaded in %d ms" % (1000 * (time.time() - t)))

    def swap(self, slot, new_id):
        """Load new_id into `slot`, RAM-safe (close the old backend first - 6 GB can't hold both).
        Does NOT touch CFG: the caller publishes CFG and persists only after this returns cleanly. On
        failure, restores whatever was loaded before and re-raises; self.loaded tracks the real state,
        so a slot left down by a doubly-failed swap is loaded=None and the same id can be retried.
        The caller holds LOCK and SWAP_BUSY, which serialise swaps."""
        prev = self.loaded.get(slot)
        self._close_slot(slot)
        self.loaded[slot] = None                 # not loaded until proven
        try:
            self._load_slot(slot, new_id, strict=True)
        except Exception:
            log("swap %s -> %s failed:\n%s" % (slot, new_id, traceback.format_exc()))
            if prev and prev != new_id:
                try:
                    self._load_slot(slot, prev)
                except Exception:
                    log("CRITICAL: could not restore %s=%s:\n%s"
                        % (slot, prev, traceback.format_exc()))
            raise

    def _slot_obj(self, slot):
        return {"chat": self.chat, "router": self.router, "asr": self.asr,
                "tts": self.tts, "tts_alt": self.tts_alt, "vad": self._vad_entry}[slot]

    def _slot_alive(self, slot):
        """True if the slot is loaded AND (for subprocess slots) its process is still running. The one
        liveness definition shared by /models, swap eligibility and the ensure_* recovery."""
        return bool(self.loaded.get(slot) and _obj_alive(self._slot_obj(slot)))

    def ensure_chat(self):
        """Reload the chat server if its process has died (the retired unit had Restart=on-failure).
        Caller holds LOCK."""
        if self._slot_alive("chat"):
            return
        log("chat server not alive, reloading")
        self._close_slot("chat")
        self._load_slot("chat", self.loaded.get("chat") or self._resolve("chat"))

    def ensure_router(self):
        """Reload the router if its backend has died (a dead router subprocess would otherwise raise
        on every turn). Caller holds LOCK."""
        if self._slot_alive("router"):
            return
        log("router not alive, reloading")
        self._close_slot("router")
        self._load_slot("router", self.loaded.get("router") or self._resolve("router"))

    def models_status(self):
        """Per-slot status for GET /models: available ids, current id, honest loaded (a subprocess
        whose process has died reports loaded=false), and best-effort RSS MB (process RSS for a live
        subprocess slot, else the daemon load-delta for the in-process sherpa/Needle models)."""
        specs = [("chat", "CHAT_MODEL", slot_ids("chat")),
                 ("router", "ROUTER_MODEL", router_ids()),
                 ("asr", "ASR_MODEL", slot_ids("asr")),
                 ("tts", "TTS_MODEL", slot_ids("tts")),
                 ("tts_alt", "TTS_MODEL_ALT", slot_ids("tts") + [""]),
                 ("vad", "VAD_MODEL", slot_ids("vad"))]
        out = {}
        for slot, field, ids in specs:
            obj = self._slot_obj(slot); alive = _obj_alive(obj); pid = _obj_pid(obj)
            out[slot] = {"field": field, "available": ids, "current": self.loaded.get(slot),
                         "loaded": bool(alive and self.loaded.get(slot)),
                         "rss_mb": _proc_rss_mb(pid) if (pid and alive) else self.rss_mb.get(slot)}
        return out

    def await_playback(self):
        if self.play_proc and self.play_proc.poll() is None:
            try:
                self.play_proc.wait(timeout=10)
            except Exception:
                pass

    def vad_detector(self):
        """A reset VAD for the selected VAD_MODEL and current VAD_THRESHOLD, or None when no VAD model
        is staged (record() then falls back to the tested energy VAD). The VAD decides speech from
        spectral shape, not loudness, so it hears quiet turns AND rejects the wake cue's own tone
        leaking into the stream. Rebuilt only when the threshold or model changes; one instance is
        reused (reset) per capture, safe because the whole pipeline is serialized under LOCK."""
        entry = self._vad_entry
        if not entry:
            if not self._vad_warned:
                log("no VAD model staged - using energy VAD")
                self._vad_warned = True
            return None
        th = CFG["VAD_THRESHOLD"]
        if self.vad is None or th != self._vad_threshold or entry["id"] != self._vad_id:
            self.vad = self._vad_build(entry, th)      # VAD_THRESHOLD is a live field, so rebuild here
            self._vad_threshold = th; self._vad_id = entry["id"]
        self.vad.reset()
        return self.vad

    def transcribe(self, samples_f32):
        return self.asr.transcribe(samples_f32)

    def _play(self, pcm_i16, sr):
        """Write a mono s16 buffer to a temp wav and play it back detached (non-blocking)."""
        d = sess_env()["XDG_RUNTIME_DIR"]
        for old in glob.glob(d + "/heytux-say-*.wav"):
            try:
                os.unlink(old)          # previous pw-play already opened its file; unlink is safe
            except OSError:
                pass
        path = "%s/heytux-say-%d.wav" % (d, time.time_ns())
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(pcm_i16.tobytes())
        self.play_proc = subprocess.Popen(["pw-play", path], env=sess_env(),
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def speak(self, text):
        """Synthesize (the measured TTS cost) then play back detached so the HTTP response returns
        immediately; playback duration is not on the request path. When a Czech alt voice is loaded
        and the reply text reads as Czech, speak it with that voice (English tool acks read as English
        and stay on the main voice)."""
        tts = self.tts_alt if (self.tts_alt is not None and is_czech(text)) else self.tts
        samples, sr = tts.synth(text, sid=0, speed=1.15)   # 1.15 shaves synth time, still natural
        pcm = (np.asarray(samples, dtype=np.float32) * 32767).clip(-32768, 32767).astype("<i2")
        self._play(pcm, sr)

    def _tone(self, freqs, note_s=0.07, amp=0.35):
        """Concatenate short sine notes into one s16 buffer (each note gets a 5 ms fade to kill clicks)."""
        parts = []
        for f in freqs:
            n = int(note_s * SR)
            t = np.arange(n) / SR
            env = np.ones(n)
            r = min(int(0.005 * SR), n // 2)
            env[:r] = np.linspace(0, 1, r); env[-r:] = np.linspace(1, 0, r)
            parts.append(amp * env * np.sin(2 * np.pi * f * t))
        return (np.concatenate(parts) * 32767).astype("<i2")

    def _cue(self, spec, note_s, haptic_event):
        """Signal the cue per CUE_MODE. Both the haptic and the tone are non-blocking (thread /
        detached Popen): latency is the point, this must never delay the capture that follows it."""
        if CFG["CUE_MODE"] in ("haptic", "both"):
            threading.Thread(target=_haptic, args=(haptic_event,), daemon=True).start()
        if CFG["CUE_MODE"] in ("sound", "both"):
            freqs = JC.parse_cue_freqs(spec)         # a malformed tone spec plays nothing, never fatal
            if freqs:
                self._play(self._tone(freqs, note_s=note_s), SR)

    def play_wake_cue(self):
        """Cue when the wake word fires: haptic buzz by default (mic-silent), or a rising chirp."""
        self._cue(CFG["CUE_WAKE_HZ"], WAKE_CUE_NOTE_S, "button-pressed")

    def play_end_cue(self):
        """Cue when the VAD stops capture: 'stopped listening, thinking now'."""
        self._cue(CFG["CUE_END_HZ"], 0.09, "button-released")

M = None            # Models, loaded in main
LOCK = threading.RLock()   # models + Needle are not thread-safe; serialize the whole pipeline

# ---------- mic capture ----------
NORM_TARGET_DBFS = -6.0          # peak the normalizer aims for before STT
NORM_MAX_GAIN_DB = 24.0          # cap so pure noise / near-silence is never blown up to full scale

def normalize(pcm_f32):
    """Scale an utterance so its peak sits at ~-6 dBFS, boosting by at most +24 dB. The user's
    speech arrives very quiet (peaks 2000-9000 of 32768); Moonshine transcribes a consistent level
    far better. The gain cap means a quiet-but-real turn is lifted while a room-noise floor is not
    amplified to full scale. Returns a new array clipped to [-1, 1]; silence passes through."""
    peak = float(np.max(np.abs(pcm_f32))) if pcm_f32.size else 0.0
    if peak <= 0.0:
        return pcm_f32
    gain = min(10 ** (NORM_TARGET_DBFS / 20) / peak, 10 ** (NORM_MAX_GAIN_DB / 20))
    return (pcm_f32 * gain).clip(-1.0, 1.0)

def _capture_stats(frames):
    """(peak, max 30 ms RMS) of the raw captured s16 frames, both in int16 scale, for the sidecar."""
    if not frames:
        return 0, 0
    x = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float32)
    peak = int(np.abs(x).max())
    nf = len(x) // FRAME
    max_rms = max((math.sqrt(float(np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2)) + 1e-9)
                   for i in range(nf)), default=0.0)
    return peak, int(max_rms)

def drain_stream(src):
    """Discard everything already buffered in the mic pipe so capture starts from NOW.

    The KWS thread holds this pw-record open continuously (~28 % of a little core) and can fall
    behind real time; at a wake hit the pipe may still hold the tail of "hey linux". A fixed skip
    can't know how much: guess low and the wake word lands in the transcript, guess high and it eats
    the command's first word. Draining to empty needs no guess. Returns bytes discarded (for the log)."""
    fd = src.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    n = 0
    try:
        while True:
            try:
                chunk = src.read(65536)
            except (BlockingIOError, InterruptedError):
                break
            if not chunk:          # None = nothing buffered, b"" = EOF
                break
            n += len(chunk)
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
    return n

def _close_rec(p):
    """Terminate a pw-record child exactly once, never raising."""
    if p is None:
        return
    p.terminate()
    try:
        p.wait(timeout=1)
    except Exception:
        p.kill()

def record(src=None, skip_s=0.0, onset_s=0.0):
    """Capture one utterance. Returns (audio_f32, speech, stats).

    Speech detection is the Silero VAD when its model is staged (M.vad_detector()), else the energy
    VAD - a level test that missed the user's quiet turns and false-fired on the wake cue's own tone.
    The onset/offset bookkeeping (SILENCE_STOP_MS, MIN_S, HARD_CAP_S) is identical for both; only the
    per-frame `voiced` decision differs. On speech, audio_f32 is the normalized array STT sees (and
    what save_capture dumps); on no speech it is the raw capture, kept so the miss can be diagnosed.
    stats carries peak/RMS and whether the VAD ever saw speech.

    src=None: open our own pw-record (the /listen path). src=<stdout pipe>: keep reading a stream
    the KWS thread already has open. The mic is never closed across the wake->capture handoff, which
    is what stopped the first word being eaten (closing/respawning pw-record left the mic shut ~1 s
    while the user was still talking). skip_s discards that many seconds up front, by SAMPLE COUNT,
    so blocking cue playback into the room never reaches the VAD/STT. onset_s > 0 (a follow-up turn)
    gives up with no speech if none starts within that much audio, also by sample count; until
    then HARD_CAP_S is not applied, so a late start still gets a full-length utterance."""
    vad = M.vad_detector()      # built/validated BEFORE opening the mic, so a VAD failure can't leak
    p = None                    # the pw-record child (it is only opened once nothing below can raise)
    if src is None:
        M.await_playback()      # don't capture the tail of the previous spoken ack
        p = subprocess.Popen(pw_record_cmd(),
                             env=sess_env(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        src = p.stdout
    frames, speech, silence_ms, voiced_run, vad_speech, start = [], False, 0, 0, False, time.time()
    nbytes = FRAME * 2
    try:
        drop = int(skip_s * SR) * 2
        while drop > 0:
            chunk = src.read(min(drop, nbytes))
            if not chunk:
                break
            drop -= len(chunk)
        while time.time() - start < CFG["HARD_CAP_S"] or (onset_s and not speech):
            buf = src.read(nbytes)
            if len(buf) < nbytes:
                break
            frames.append(buf)
            x = np.frombuffer(buf, dtype="<i2").astype(np.float32)
            if vad is not None:
                vad.accept_waveform(x / 32768.0)
                voiced = vad.is_speech_detected()   # Silero already debounces the onset (min_speech_duration)
            else:
                # energy VAD: hysteresis - a louder onset to start, a quieter floor to keep going.
                rms = math.sqrt(float(np.mean(x * x)) + 1e-9)
                voiced = rms > (CFG["VAD_CONT_RMS"] if speech else CFG["VAD_RMS"])
            vad_speech = vad_speech or voiced
            if onset_s and not speech and len(frames) * FRAME >= onset_s * SR:
                break                       # follow-up window lapsed with no speech onset
            if speech:
                if voiced:
                    silence_ms = 0
                else:
                    silence_ms += 1000 * FRAME / SR
                    if silence_ms >= CFG["SILENCE_STOP_MS"] and (time.time() - start) >= CFG["MIN_S"]:
                        break
            elif voiced:
                if vad is not None:
                    speech = True
                else:
                    voiced_run += 1
                    if voiced_run >= CFG["VAD_ONSET_FRAMES"]:   # sustained -> real speech, not a spike
                        speech = True
                if speech and onset_s:
                    start = time.time()     # HARD_CAP_S/MIN_S count from a follow-up's late onset
            else:
                voiced_run = 0
    finally:
        if p is not None:       # only close what we opened; a handed-over stream stays the caller's
            _close_rec(p)
    peak, max_rms = _capture_stats(frames)
    stats = {"peak": peak, "max_rms": max_rms, "vad_speech": vad_speech}
    raw = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float32) / 32768.0 if frames \
        else np.zeros(0, np.float32)
    if not speech:
        return raw, False, stats     # the spoken "I didn't catch that" is the feedback here
    M.play_end_cue()                 # "stopped listening, thinking now"
    return normalize(raw), True, stats

# ---------- routing + pipeline ----------
def decide_route(r, transcript):
    """Pure route decision from a router result. ask_question -> chat-tool; empty calls / NOMATCH
    reasoning / conf < CONF_FLOOR / ungrounded -> chat-fallback; else tool. CONF_FLOOR applies only to
    routers that report a comparable confidence (Needle3); a self-gated router (embedding floor, GGUF
    tool_choice) sets r["self_gated"] and is trusted past the floor. Grounding applies to all. Returns
    (route, calls, conf, names)."""
    calls = r.get("function_calls") or []
    conf = r.get("confidence", 0.0)
    reasoning = (r.get("reasoning") or "").strip()
    names = [c.get("name") for c in calls]
    is_ask = bool(calls) and all(n == "ask_question" for n in names)
    ungrounded = bool(calls) and not is_ask and not all(_grounded(n, transcript) for n in names)
    if ungrounded:
        log("grounding: rejected %s for %r (no shared content word)" % (names, transcript))
    below_floor = (not r.get("self_gated")) and conf < CFG["CONF_FLOOR"]
    go_chat = is_ask or (not calls) or _off_topic(reasoning) or below_floor or ungrounded
    if go_chat:
        return ("chat-tool" if is_ask else "chat-fallback"), calls, conf, names
    return "tool", calls, conf, names

_NO_ROUTE = {"function_calls": [], "confidence": 0.0, "reasoning": "", "self_gated": True}

def _route_complete(transcript):
    """Route one transcript, surviving a dead/throwing router: reload it once (caller holds LOCK) and
    retry; if it still raises, route to chat rather than failing the whole turn."""
    try:
        return M.router.complete(transcript)
    except Exception:
        log("router.complete failed:\n" + traceback.format_exc())
    try:
        M.ensure_router()
        return M.router.complete(transcript)
    except Exception:
        log("router recovery failed, routing to chat:\n" + traceback.format_exc())
        return dict(_NO_ROUTE)

# Route labels: TOOL_ROUTES ran a phone command (ends any follow-up window); ANSWER_ROUTES spoke a
# chat answer (keeps the follow-up window open). "none" is a no-speech turn.
TOOL_ROUTES = {"tool", "tool-fast", "llm-tool"}
ANSWER_ROUTES = {"chat-tool", "chat-fallback", "llm-answer", "chat-followup"}

def _exec_calls(calls, ms):
    with timed(ms, "act"):
        return " ".join(a for a in (execute(c.get("name"), c.get("arguments")) for c in calls) if a)

def _decide_and_run(transcript, ms, dry, followup):
    """Pick the route for one turn and carry it out (execute a tool, or run the chat model), minus the
    speaking and overlay (pipeline does those). Returns (route, said, calls, conf, names).

    ROUTE_MODE "router" is the original behaviour. "hybrid" keeps the fast router path only for a
    grounded hit at >= HYBRID_FLOOR and sends everything else to the chat model with the enabled
    tools. FOLLOWUP_ROUTE "chat" makes a follow-up turn skip the router entirely (plain chat in
    router mode, chat-with-tools in hybrid)."""
    hybrid = CFG["ROUTE_MODE"] == "hybrid"
    skip = followup and CFG["FOLLOWUP_ROUTE"] == "chat"
    conf, names = 0.0, []
    # Only similarity-scored routers can clear HYBRID_FLOOR. Generative routers' synthetic
    # confidence (e.g. GGUF's constant 1.0) must never bypass the hybrid chat model.
    if not skip and (not hybrid or getattr(M.router, "scored", False)):
        with timed(ms, "route"):
            r = _route_complete(transcript)
        route, calls, conf, names = decide_route(r, transcript)
        if route == "tool" and (not hybrid or conf >= CFG["HYBRID_FLOOR"]):
            said = "" if dry else _exec_calls(calls, ms)
            if not dry:
                _remember(transcript, said)                 # tool turns are context for follow-ups too
            return ("tool-fast" if hybrid else "tool"), said, calls, conf, names
        if not hybrid:                                      # router mode: fall to plain chat (today)
            with timed(ms, "llm"):
                ans = chat_llm(transcript, remember=not dry)   # ORIGINAL transcript, not needle's arg
            return route, (ans or CHAT_FAIL), [], conf, names
    # hybrid (fast path declined or skipped), OR a router-mode follow-up that skips the router:
    if hybrid:
        with timed(ms, "llm"):
            did_tool, said, calls = chat_tool(transcript, dry)
        if did_tool:
            return "llm-tool", said, calls, conf, [c.get("name") for c in calls]
        return ("chat-followup" if skip else "llm-answer"), said, [], conf, []
    with timed(ms, "llm"):                                   # router mode, skip-router follow-up
        ans = chat_llm(transcript, remember=not dry)
    return "chat-followup", (ans or CHAT_FAIL), [], conf, []

def pipeline(transcript, ms, overlay=False, notif_id=0, dry=False, followup=False):
    """transcript already produced; route -> execute/chat -> speak. ms is mutated + returned. overlay
    is set only for audio-triggered turns (never /text). dry (POST /text {"dry":true}) decides the
    route and runs the chat model if chat, but executes no tool and speaks nothing (said=""). followup
    marks a no-wake-word follow-up turn (FOLLOWUP_ROUTE can route those straight to chat)."""
    global last_pipeline_end
    for k in ("route", "llm", "act", "tts"):
        ms.setdefault(k, 0)
    try:
        if not transcript:
            said = "I didn't catch that."
            if overlay and CFG["OVERLAY_ON"]:
                _notify_async("Didn't catch that", "", notif_id, 4000)
            if not dry:
                with timed(ms, "tts"):
                    M.speak(said)
            return {"transcript": "", "calls": [], "said": said, "confidence": 0.0,
                    "route": "none", "ms": ms}
        if overlay and CFG["OVERLAY_ON"]:
            # Bounded (8 s), not never-expiring: needle+act is 130-500 ms so step 3 almost always
            # replaces this first, but a raised exception or notif_id 0 must not leave it forever.
            _notify_async('"%s"' % transcript, "thinking…", notif_id, 8000)
        route, said, calls, conf, names = _decide_and_run(transcript, ms, dry, followup)
        lat = sum(ms.get(k, 0) for k in ("route", "llm", "act"))
        log("route=%s conf=%.2f lat=%dms transcript=%r said=%r" % (route, conf, lat, transcript, said))
        if overlay and CFG["OVERLAY_ON"]:
            # confidence distinguishes a mis-hearing (right tool, wrong words) from a mis-routing.
            action = "→ %s (%.2f)" % (names[0], conf) if route in TOOL_ROUTES and names else "→ chat"
            _notify_async(transcript, "%s\n%s" % (action, said), notif_id, 6000)
        if not dry:
            with timed(ms, "tts"):
                M.speak(said)
        return {"transcript": transcript, "calls": calls, "said": said, "confidence": conf,
                "route": route, "ms": ms}
    finally:
        last_pipeline_end = time.time()

_capture_err_logged = False       # log a capture-save failure once, then just skip silently

def _evict_captures():
    """Delete the oldest wav+json pairs beyond CAPTURE_KEEP. Timestamp filenames sort chronologically."""
    bases = sorted(os.path.splitext(p)[0] for p in glob.glob(os.path.join(CAPTURE_DIR, "*.wav")))
    for base in bases[:-CFG["CAPTURE_KEEP"]]:
        for ext in (".wav", ".json"):
            try:
                os.unlink(base + ext)
            except OSError:
                pass

def save_capture(pcm_f32, result, drained_bytes, stats):
    """Dump the audio plus a sidecar json - the instrument that tells a bad transcript (model) from
    a bad capture (VAD/mic). Runs for every turn, including no-speech ones (route "none"): the wav is
    then the raw capture and stats say whether it was silence, a quiet miss, or the cue leaking in.
    Background thread; logs a failure once."""
    global _capture_err_logged
    if CFG["CAPTURE_KEEP"] <= 0:
        return
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        stamp = "%s-%03d" % (time.strftime("%Y%m%d-%H%M%S"), int(time.time() * 1000) % 1000)
        base = os.path.join(CAPTURE_DIR, stamp)
        pcm_i16 = (pcm_f32 * 32768.0).clip(-32768, 32767).astype("<i2")
        with wave.open(base + ".wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(pcm_i16.tobytes())
        with open(base + ".json", "w") as f:
            json.dump({
                "transcript": result.get("transcript"),
                "route": result.get("route"),
                "calls": result.get("calls") or [],
                "confidence": result.get("confidence"),
                "ms": result.get("ms"),
                "drained_bytes": drained_bytes,
                "peak": stats.get("peak"),
                "max_rms": stats.get("max_rms"),
                "vad_speech": stats.get("vad_speech"),
            }, f, indent=2)
        _evict_captures()
    except Exception as e:
        if not _capture_err_logged:
            log("capture save error (further failures suppressed):", repr(e))
            _capture_err_logged = True

def listen_and_run(src=None, skip_s=0.0, notif_id=0, drained_bytes=0, followup_s=0.0):
    """Record, transcribe, run the pipeline. Shared by /listen and wake hits (which hand over the
    still-open KWS stream via src/skip_s; /listen passes neither). followup_s > 0 is a no-wake-word
    follow-up turn: silence for that long ends it quietly (route "none", nothing spoken or saved -
    it's the normal end of a conversation, not a miss). A raise anywhere here would otherwise leave
    the 'Listening...' banner (expire_ms=0) on screen forever, so clear it and re-raise."""
    ms = {}
    try:
        with timed(ms, "record"):
            pcm, speech, stats = record(src, skip_s, followup_s)
        if not speech and followup_s:
            overlay_clear(notif_id)
            return {"transcript": "", "calls": [], "said": "", "confidence": 0.0,
                    "route": "none", "ms": ms}
        if not speech:
            ms["stt"] = 0
            res = pipeline("", ms, overlay=True, notif_id=notif_id)   # "I didn't catch that"
        else:
            with timed(ms, "stt"):
                tx = M.transcribe(pcm)
            res = pipeline(tx, ms, overlay=True, notif_id=notif_id, followup=bool(followup_s))
        # Save every turn, including no-speech ones, so a "didn't catch that" can be diagnosed later.
        threading.Thread(target=save_capture, args=(pcm, res, drained_bytes, stats), daemon=True).start()
        return res
    except Exception:
        overlay_clear(notif_id)
        raise

# ---------- wake word: KWS owns the mic when idle, hands it to the pipeline on a hit ----------
def pause_kws():
    KWS_RELEASE.set()
    KWS_IDLE.wait(timeout=5)     # block until the KWS thread has let go of the mic

def resume_kws():
    KWS_RELEASE.clear()

def kws_loop():
    global wake_hits
    mic_err = False        # log a dead-mic condition once, then retry quietly every 5 s
    hb_last = [0.0]; hb_frames = [0]; hb_maxabs = [0]
    def heartbeat(where):
        now = time.time()
        if now - hb_last[0] >= 15:
            log("wake heartbeat [%s] frames=%d maxabs=%d hits=%d WAKE_ON=%s RELEASE=%s IDLE=%s target=%r"
                % (where, hb_frames[0], hb_maxabs[0], wake_hits, WAKE_ON.is_set(),
                   KWS_RELEASE.is_set(), KWS_IDLE.is_set(), CFG["MIC_TARGET"] or "default"))
            hb_last[0] = now; hb_frames[0] = 0; hb_maxabs[0] = 0
    while True:
        if not WAKE_ON.is_set() or KWS_RELEASE.is_set():
            KWS_IDLE.set(); heartbeat("paused"); time.sleep(0.15); continue
        M.await_playback()                       # never listen over our own ack/cue
        KWS_IDLE.clear()
        hit = None; frames = 0; rec = None
        try:
            st = M.kws.create_stream()
            rec = subprocess.Popen(pw_record_cmd(), env=sess_env(),
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                while WAKE_ON.is_set() and not KWS_RELEASE.is_set():
                    buf = rec.stdout.read(KWS_CHUNK)
                    if not buf:
                        break                    # pw-record exited (mic/card gone)
                    frames += 1; hb_frames[0] += 1
                    x = np.frombuffer(buf, dtype="<i2")
                    hb_maxabs[0] = max(hb_maxabs[0], int(np.abs(x.astype(np.int32)).max()))
                    st.accept_waveform(SR, x.astype(np.float32) / 32768.0)
                    while M.kws.is_ready(st):
                        M.kws.decode_stream(st)
                        r = M.kws.get_result(st)
                        if r:
                            hit = r; M.kws.reset_stream(st); break
                    heartbeat("capture")
                    if hit:
                        break
            except BaseException:
                _close_rec(rec); rec = None
                raise
        except Exception:
            if not mic_err:
                log("KWS capture error (retrying every 5s):\n" + traceback.format_exc()); mic_err = True
            KWS_IDLE.set(); time.sleep(5); continue
        if hit:
            mic_err = False
            if time.time() - last_pipeline_end < CFG["WAKE_COOLDOWN_S"]:
                _close_rec(rec); KWS_IDLE.set()
                continue                         # cooldown: too soon after the last pipeline
            wake_hits += 1
            log("wake hit '%s' at %s (#%d)" % (hit, time.strftime("%Y-%m-%d %H:%M:%S"), wake_hits))
            try:
                with LOCK:
                    notif_id = overlay_listening()   # overlay step 1, before the cue
                    # Drain BEFORE the cue, not after: this discards only the genuinely stale
                    # pre-hit backlog (the KWS reader behind real time), never what the speaker says
                    # while the cue plays. The mic stays open across the handoff.
                    n = drain_stream(rec.stdout)
                    log("handoff: drained %d bytes (%.2f s) of buffered mic audio"
                        % (n, n / 2.0 / SR))
                    M.play_wake_cue()        # non-blocking in every mode - see Models._cue
                    # skip_s is 0 in haptic/off mode; in sound/both it skips the tone's own length by
                    # SAMPLE COUNT, so not blocking on playback here is harmless.
                    res = listen_and_run(rec.stdout, wake_skip_s(), notif_id, drained_bytes=n)
                    # Follow-up window: after a chat answer, listen again without the wake word on
                    # the SAME open stream (closing it ate the first word). A tool turn, silence for
                    # FOLLOWUP_S, or the wake word being switched off ends it -> back to spotting.
                    while (res["route"] in ANSWER_ROUTES and CFG["FOLLOWUP_S"] > 0
                           and WAKE_ON.is_set()):
                        log("wake pipeline: transcript=%r said=%r" % (res.get("transcript"), res.get("said")))
                        M.await_playback()          # let our spoken answer finish...
                        if not WAKE_ON.is_set():       # switched off while the answer played
                            break
                        n = drain_stream(rec.stdout)   # ...and discard it: the open mic heard it
                        notif_id = overlay_listening()
                        M.play_wake_cue()
                        res = listen_and_run(rec.stdout, wake_skip_s(), notif_id, drained_bytes=n,
                                             followup_s=CFG["FOLLOWUP_S"])
                log("wake pipeline: transcript=%r said=%r" % (res.get("transcript"), res.get("said")))
            except Exception:
                log("wake pipeline error:\n" + traceback.format_exc())
            finally:
                _close_rec(rec)
                KWS_IDLE.set()   # only now is the mic really free (pause_kws waits on this)
            continue
        _close_rec(rec)
        KWS_IDLE.set()
        # no hit: if the mic yielded almost nothing while enabled, the card is gone -> back off 5 s
        if frames < 3 and WAKE_ON.is_set() and not KWS_RELEASE.is_set():
            if not mic_err:
                log("KWS: mic produced no audio, retrying every 5s"); mic_err = True
            time.sleep(5)
        else:
            mic_err = False

# ---------- HTTP ----------
class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionError):
            pass                        # client hung up (e.g. request timed out); nothing to do

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode())
        except Exception:
            return {}

    def log_message(self, *a):
        pass

    def _get_status(self):
        st = phone_info(); st["wake"] = WAKE_ON.is_set(); st["wake_hits"] = wake_hits
        self._send(st)

    def _get_wake(self):
        self._send({"on": WAKE_ON.is_set(), "hits": wake_hits})

    def _get_config(self):
        self._send({"values": live_values(), "defaults": JC.defaults(),
                    "schema": JC.SCHEMA, "pending_restart": pending_restart()})

    def _get_models(self):
        self._send({"busy": SWAP_BUSY.is_set(), "slots": M.models_status() if M else {}})

    def _post_wake(self):
        on = bool(self._body().get("on", True))
        with CFG_LOCK:
            set_wake(on)                        # keeps the Event and config.json in step
            JC.save(JC.CONFIG_PATH, CFG)
        self._send({"on": WAKE_ON.is_set(), "hits": wake_hits})

    def _post_config(self):
        changes = self._body()
        clean, errors = JC.validate(changes)
        if errors:
            self._send({"errors": errors}, 400); return      # nothing applied
        # Model-slot changes hot-swap in place (close old, load new). Any request carrying a model
        # field takes SWAP_LOCK for its whole critical section - eligibility is computed, the swaps
        # run, and CFG is published AND persisted all under the one lock - so two concurrent swaps
        # can't publish out of order and a request can't observe a mid-swap state. A second concurrent
        # model request gets 409, not a queue. A slot is swapped when the id differs, OR the id is the
        # same but the slot is not live (a prior swap that failed, or a crashed subprocess) so it can
        # be retried.
        has_model = any(k in MODEL_FIELDS for k in clean)
        if has_model and not SWAP_LOCK.acquire(blocking=False):
            self._send({"busy": True, "errors": {k: "a model swap is already in progress"
                                                 for k in clean if k in MODEL_FIELDS}}, 409); return
        swap_errors, swapped = {}, {}
        try:
            if has_model:
                SWAP_BUSY.set()
                needs = {k: v for k, v in clean.items() if k in MODEL_FIELDS
                         and (v != CFG.get(k) or not M._slot_alive(MODEL_FIELDS[k]))}
                for k, v in needs.items():
                    try:
                        with LOCK:
                            M.swap(MODEL_FIELDS[k], v)
                        swapped[k] = v                    # publish only after the load succeeded
                    except Exception as e:
                        swap_errors[k] = "could not load %s: %s" % (v, e)
            with CFG_LOCK:
                # Persist swapped model ids and the non-model live fields together, atomically.
                CFG.update(swapped)
                CFG.update({k: v for k, v in clean.items() if k not in MODEL_FIELDS})
                if "WAKE_ON" in clean:
                    set_wake(clean["WAKE_ON"])
                if "KWS_TEXT" in clean:
                    sync_kws_text(CFG["KWS_TEXT"])
                JC.save(JC.CONFIG_PATH, CFG)
            # Swaps see the old CFG. Warm again against the final prefix (mode/system/language),
            # with key-based dedup so an unchanged config never pays another long prefill.
            with LOCK:
                if CFG["ROUTE_MODE"] == "hybrid" and M._slot_alive("chat"):
                    M._prewarm_chat()
        finally:
            if has_model:
                SWAP_BUSY.clear(); SWAP_LOCK.release()
        if swap_errors:
            self._send({"errors": swap_errors}, 500); return
        applied = sorted(set(swapped) | {k for k in clean if k not in MODEL_FIELDS})
        log("config: applied %s" % ", ".join(applied) if applied else "config: no change")
        self._send({"applied": applied, "values": live_values(),
                    "pending_restart": pending_restart()})

    def _post_listen(self):
        with LOCK:
            notif_id = overlay_listening()   # immediate banner, before the KWS-release wait
            pause_kws()                      # take the mic from the KWS thread
            try:
                self._send(listen_and_run(notif_id=notif_id))
            finally:
                resume_kws()

    def _post_text(self):
        body = self._body()
        ms = {"record": 0, "stt": 0}
        with LOCK:
            self._send(pipeline((body.get("text") or "").strip(), ms, dry=bool(body.get("dry"))))

    def _post_say(self):
        txt = (self._body().get("text") or "").strip()
        ms = {}
        with LOCK:
            with timed(ms, "tts"):
                M.speak(txt)
        self._send({"said": txt, "ms": {"tts": ms["tts"]}})

    GET_ROUTES = {"/status": _get_status, "/wake": _get_wake, "/config": _get_config,
                  "/models": _get_models}
    POST_ROUTES = {"/wake": _post_wake, "/config": _post_config, "/listen": _post_listen,
                   "/text": _post_text, "/say": _post_say}

    def do_GET(self):
        fn = self.GET_ROUTES.get(self.path)
        if not fn:
            self._send({"error": "not found"}, 404); return
        try:
            fn(self)
        except Exception as e:
            log("GET error:", repr(e))

    def do_POST(self):
        fn = self.POST_ROUTES.get(self.path)
        if not fn:
            self._send({"error": "not found"}, 404); return
        try:
            fn(self)
        except Exception as e:
            log("request error:", repr(e))
            try:
                self._send({"error": str(e)}, 500)
            except Exception:
                pass

class Server(ThreadingHTTPServer):
    # Threaded so the plasmoid's GET /wake poll and /status don't block behind a 6 s /listen
    # recording. LOCK still serializes the models/pipeline; only the socket handling is concurrent.
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionError)):
            return                      # benign client disconnect; don't dump a traceback
        super().handle_error(request, client_address)

def main():
    global M, CFG
    threading.Thread(target=_overlay_worker, name="overlay", daemon=True).start()
    load_config()
    log("loading models...")
    t0 = time.time()
    try:
        M = Models()
    except Exception:
        # A persisted model id that validated but doesn't actually load must not crash-loop the
        # daemon. Kill any half-started llama-server (so :8771 is free), retry once on defaults; if
        # THAT fails, let it die.
        log("Models() failed with the current config, retrying once on defaults:\n" + traceback.format_exc())
        HB._close_all()
        CFG = JC.defaults()
        set_wake(CFG["WAKE_ON"])
        M = Models()
    log("all models ready in %d ms; serving on 127.0.0.1:%d" % (1000 * (time.time() - t0), CFG["PORT"]))
    threading.Thread(target=kws_loop, name="kws", daemon=True).start()
    log("wake word thread started (wake=%s)" % WAKE_ON.is_set())
    Server(("127.0.0.1", CFG["PORT"]), H).serve_forever()

if __name__ == "__main__":
    main()
