#!/usr/bin/env python3
# heytux.py - offline push-to-talk voice assistant daemon for Pixel 4a / BengalOS.
# Pipeline: mic (Silero VAD, energy fallback) -> normalize -> Parakeet/Moonshine STT -> Needle3 -> execute -> Piper TTS ack.
# stdlib + numpy + sherpa_onnx + ctypes only. Runs as miro's systemd --user service,
# so all subprocesses (pw-record/pw-play/wpctl/gio/busctl/nmcli/bluetoothctl) inherit miro's session.
import contextlib, ctypes, fcntl, glob, hashlib, json, math, os, queue, re, subprocess, sys, \
    threading, time, traceback, urllib.request, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import heytux_config as JC

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
SILERO_VAD = "/home/miro/sherpa/silero_vad.onnx"   # copied to the phone out-of-band; absent => energy VAD fallback

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

def _stt_kind(d):
    """'moonshine' / 'transducer' (NeMo parakeet-tdt) for an STT dir Models() can load, else None."""
    if not os.path.isfile(d + "/tokens.txt"):
        return None
    if os.path.isfile(d + "/encoder_model.ort") and os.path.isfile(d + "/decoder_model_merged.ort"):
        return "moonshine"
    if all(os.path.isfile("%s/%s.int8.onnx" % (d, n)) for n in ("encoder", "decoder", "joiner")):
        return "transducer"
    return None

def discover_models():
    """Fill the enum fields' value lists from what is actually staged on disk. A TTS dir is usable
    only if it has the vits shape Models() builds (one *.onnx + tokens.txt + espeak-ng-data/);
    an STT dir only if _stt_kind() recognizes its export shape - offering any other shape would
    brick the daemon on the next restart."""
    tts = []
    for d in sorted(glob.glob("/home/miro/sherpa/tts/*/")):
        d = d.rstrip("/")
        onnx = glob.glob(d + "/*.onnx")
        if len(onnx) == 1 and os.path.isfile(d + "/tokens.txt") and os.path.isdir(d + "/espeak-ng-data"):
            tts.append(d)
    JC.FIELDS["TTS_DIR"]["values"] = tts
    stt = []
    for d in sorted(glob.glob("/home/miro/sherpa/*/")):
        d = d.rstrip("/")
        if _stt_kind(d):
            stt.append(d)
    JC.FIELDS["STT_DIR"]["values"] = stt

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
    CFG = JC.load(JC.CONFIG_PATH)
    set_wake(CFG["WAKE_ON"])
    sync_kws_text(CFG["KWS_TEXT"])
    MERGED_TOOLS = merged_tools()
    TOOL_VOCAB = _build_tool_vocab(MERGED_TOOLS)
    _STARTED_WITH.update({k: CFG[k] for k, f in JC.FIELDS.items() if f["apply"] in ("restart", "model")})
    log("config loaded from %s" % JC.CONFIG_PATH)

def pending_restart():
    """Keys whose saved value differs from what this process started with. Covers both apply
    classes that need a restart: 'restart' (a new global takes effect only at startup) and
    'model' (KWS_TEXT/TTS_DIR - takes effect only when the model is rebuilt)."""
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

@tool("set_timer", "Start a countdown timer for some minutes, optional label.",
      {"minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
       "label": {"type": "string"}}, ["minutes"])
def _set_timer(args):
    try:
        mins = int(args.get("minutes"))
    except (TypeError, ValueError):
        return "I need a number of minutes."
    label = (args.get("label") or "").strip()
    tmr = threading.Timer(mins * 60, _timer_fire, [label]); tmr.daemon = True; tmr.start()
    return "Timer set for %d minute%s%s." % (mins, "" if mins == 1 else "s",
                                             " for " + label if label else "")

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
    """Make LLM output safe to speak: drop markdown/non-ASCII, collapse space, first 2 sentences."""
    t = re.sub(r"[*_`#>~\[\]]", "", t or "")
    t = "".join(ch for ch in t if ord(ch) < 128)      # strip emoji / non-ASCII symbols
    t = re.sub(r"\s+", " ", t).strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:2]).strip()
    return t[:220].strip()

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

def chat_llm(transcript, remember=True):
    """Ask the local LLM; return a cleaned spoken answer, or None on any error/timeout. A good
    answer is added to the chat memory unless remember=False (dry /text must not change it)."""
    global chat_last
    body = json.dumps({
        "messages": chat_messages(CFG["LLM_SYS"] + " " + _live_facts(), _chat_history(), transcript),
        "max_tokens": 60, "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=10))
        ans = clean_spoken(r["choices"][0]["message"]["content"])
    except Exception as e:
        log("chat_llm error:", repr(e))
        return None
    if ans and remember and CFG["CHAT_HISTORY_TURNS"] > 0:
        CHAT_HISTORY.append((transcript, ans))
        del CHAT_HISTORY[:-CFG["CHAT_HISTORY_TURNS"]]
        chat_last = time.time()
    return ans or None

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

# ---------- models ----------
class Models:
    def __init__(self):
        import sherpa_onnx as so
        self.so = so                 # kept so vad_detector() can build the VAD lazily, still only inside Models
        self.vad = None              # one Silero VAD instance, reset per capture; rebuilt when the threshold changes
        self._vad_threshold = None
        self._vad_warned = False
        t = time.time()
        d, kind = CFG["STT_DIR"], _stt_kind(CFG["STT_DIR"])
        if kind == "transducer":     # parakeet-tdt: ~40 % lower WER than moonshine-tiny on quiet speech
            self.stt = so.OfflineRecognizer.from_transducer(
                encoder=d + "/encoder.int8.onnx", decoder=d + "/decoder.int8.onnx",
                joiner=d + "/joiner.int8.onnx", tokens=d + "/tokens.txt",
                num_threads=THREADS, model_type="nemo_transducer")
        else:
            self.stt = so.OfflineRecognizer.from_moonshine_v2(
                encoder=d + "/encoder_model.ort", decoder=d + "/decoder_model_merged.ort",
                tokens=d + "/tokens.txt", num_threads=THREADS)
        log("STT (%s, %s) loaded in %d ms" % (kind, os.path.basename(d), 1000 * (time.time() - t)))
        t = time.time()
        # discover_models() only lists TTS dirs with exactly one *.onnx, so this always resolves;
        # the onnx filename varies per voice (never assume lessac-low's).
        tts_onnx = glob.glob(CFG["TTS_DIR"] + "/*.onnx")[0]
        self.tts = so.OfflineTts(so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
            vits=so.OfflineTtsVitsModelConfig(model=tts_onnx, lexicon="",
                tokens=CFG["TTS_DIR"] + "/tokens.txt", data_dir=CFG["TTS_DIR"] + "/espeak-ng-data"),
            num_threads=THREADS)))
        log("TTS (piper lessac-low) loaded in %d ms" % (1000 * (time.time() - t)))
        t = time.time()
        self.needle = Needle(MERGED_TOOLS)
        log("Needle3 loaded in %d ms" % (1000 * (time.time() - t)))
        t = time.time()
        self.kws = so.KeywordSpotter(
            tokens=KWS_DIR + "/tokens.txt",
            encoder=KWS_DIR + "/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            decoder=KWS_DIR + "/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            joiner=KWS_DIR + "/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            num_threads=1, keywords_file=KWS_KEYWORDS, provider="cpu",
            # host bench, 40 TTS speakers, quiet (peak 0.08): 28/40 at the defaults (4, 0.25, 1.0)
            # -> 35/40 here, near-miss false alarms unchanged; more paths raises false alarms
            max_active_paths=8, keywords_threshold=0.15, keywords_score=1.5)
        log("KWS (zipformer wake word) loaded in %d ms" % (1000 * (time.time() - t)))
        self.play_proc = None       # last detached ack playback, awaited before the next mic capture

    def await_playback(self):
        if self.play_proc and self.play_proc.poll() is None:
            try:
                self.play_proc.wait(timeout=10)
            except Exception:
                pass

    def vad_detector(self):
        """A reset Silero VAD built for the current VAD_THRESHOLD, or None when the model file is
        absent (record() then falls back to the tested energy VAD). Silero decides speech from
        spectral shape, not loudness, so it hears the user's quiet turns AND rejects the wake cue's
        own tone leaking into the stream - both level-independent, which the RMS gate can't be.
        Rebuilt only when the threshold is changed live; one instance is reused (reset) per capture,
        safe because the whole pipeline is serialized under LOCK."""
        if not os.path.isfile(SILERO_VAD):
            if not self._vad_warned:
                log("silero VAD model %s absent - using energy VAD" % SILERO_VAD)
                self._vad_warned = True
            return None
        th = CFG["VAD_THRESHOLD"]
        if self.vad is None or th != self._vad_threshold:
            so = self.so
            cfg = so.VadModelConfig(silero_vad=so.SileroVadModelConfig(
                model=SILERO_VAD, threshold=th, min_silence_duration=0.1,
                min_speech_duration=0.05, window_size=512, max_speech_duration=20),
                sample_rate=SR, num_threads=1)
            self.vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
            self._vad_threshold = th
        self.vad.reset()
        return self.vad

    def transcribe(self, samples_f32):
        s = self.stt.create_stream()
        s.accept_waveform(SR, samples_f32)
        self.stt.decode_stream(s)
        return s.result.text.strip()

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
        immediately; playback duration is not on the request path."""
        a = self.tts.generate(text, sid=0, speed=1.15)   # 1.15 shaves synth time, still natural
        pcm = (np.asarray(a.samples, dtype=np.float32) * 32767).clip(-32768, 32767).astype("<i2")
        self._play(pcm, a.sample_rate)

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
    p = None
    if src is None:
        M.await_playback()      # don't capture the tail of the previous spoken ack
        p = subprocess.Popen(pw_record_cmd(),
                             env=sess_env(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        src = p.stdout
    vad = M.vad_detector()
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
    """Pure route decision from a Needle3 result. ask_question -> chat-tool; empty calls / NOMATCH
    reasoning / conf < CONF_FLOOR / ungrounded -> chat-fallback; else tool. Returns
    (route, calls, conf, names)."""
    calls = r.get("function_calls") or []
    conf = r.get("confidence", 0.0)
    reasoning = (r.get("reasoning") or "").strip()
    names = [c.get("name") for c in calls]
    is_ask = bool(calls) and all(n == "ask_question" for n in names)
    ungrounded = bool(calls) and not is_ask and not all(_grounded(n, transcript) for n in names)
    if ungrounded:
        log("grounding: rejected %s for %r (no shared content word)" % (names, transcript))
    go_chat = is_ask or (not calls) or _off_topic(reasoning) or conf < CFG["CONF_FLOOR"] or ungrounded
    if go_chat:
        return ("chat-tool" if is_ask else "chat-fallback"), calls, conf, names
    return "tool", calls, conf, names

def pipeline(transcript, ms, overlay=False, notif_id=0, dry=False):
    """transcript already produced; needle -> execute -> speak. ms is mutated + returned. overlay
    is set only for audio-triggered turns (never /text). dry (POST /text {"dry":true}) decides the
    route and runs the chat LLM if chat, but executes no tool and speaks nothing (said="")."""
    global last_pipeline_end
    for k in ("llm", "act", "tts"):
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
        with timed(ms, "needle"):
            r = M.needle.complete(transcript)
        route, calls, conf, names = decide_route(r, transcript)
        if route == "tool":
            if dry:
                said = ""
            else:
                with timed(ms, "act"):
                    said = " ".join(a for a in (execute(c.get("name"), c.get("arguments"))
                                                for c in calls) if a)
        else:
            with timed(ms, "llm"):
                ans = chat_llm(transcript, remember=not dry)   # ORIGINAL transcript, not needle's arg
            said, calls = (ans if ans else CHAT_FAIL), []
        log("route=%s conf=%.2f transcript=%r said=%r" % (route, conf, transcript, said))
        if overlay and CFG["OVERLAY_ON"]:
            # confidence distinguishes a mis-hearing (right tool, wrong words) from a mis-routing.
            action = "→ %s (%.2f)" % (names[0], conf) if route == "tool" and names else "→ chat"
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
            res = pipeline(tx, ms, overlay=True, notif_id=notif_id)
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
                    while (res["route"].startswith("chat") and CFG["FOLLOWUP_S"] > 0
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
        with CFG_LOCK:
            CFG.update(clean)
            if "WAKE_ON" in clean:
                set_wake(clean["WAKE_ON"])
            if "KWS_TEXT" in clean:
                sync_kws_text(CFG["KWS_TEXT"])
            JC.save(JC.CONFIG_PATH, CFG)
        applied = sorted(clean)
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

    GET_ROUTES = {"/status": _get_status, "/wake": _get_wake, "/config": _get_config}
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
        # A persisted TTS_DIR/STT_DIR that validated but doesn't actually load (wrong onnx shape)
        # must not crash-loop the daemon. Retry once on defaults; if THAT fails, let it die.
        log("Models() failed with the current config, retrying once on defaults:\n" + traceback.format_exc())
        CFG = JC.defaults()
        set_wake(CFG["WAKE_ON"])
        M = Models()
    log("all models ready in %d ms; serving on 127.0.0.1:%d" % (1000 * (time.time() - t0), CFG["PORT"]))
    threading.Thread(target=kws_loop, name="kws", daemon=True).start()
    log("wake word thread started (wake=%s)" % WAKE_ON.is_set())
    Server(("127.0.0.1", CFG["PORT"]), H).serve_forever()

if __name__ == "__main__":
    main()
