#!/usr/bin/env python3
# heytux_config.py - schema, validation and persistence for the heytux daemon.
# stdlib ONLY (no numpy, no sherpa, no import of heytux.py) so it runs and tests off-device.
# Field keys deliberately match heytux.py's module-global names: applying config is a global
# rebind, and Python resolves globals at call time, so no use site needs editing.
import json, os, sys

CONFIG_DIR = os.path.expanduser("~/.config/heytux")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
TOOLS_PATH = os.path.join(CONFIG_DIR, "tools.json")

def log(*a):
    print("[config]", *a, file=sys.stderr, flush=True)

def _f(key, type_, default, label, help_, group, apply, **kw):
    d = {"key": key, "type": type_, "default": default, "label": label,
         "help": help_, "group": group, "apply": apply}
    d.update(kw)
    return d

# Display order is the order of this list.
SCHEMA = [
    _f("WAKE_ON", "bool", True, "Listen for \"hey tux\"",
       "Always-on wake word. Costs about 28% of one CPU core while idle.", "wake", "live"),
    _f("VAD_RMS", "float", 600.0, "Voice onset threshold",
       "How loud you have to be to START listening. Lower hears quieter speech but picks up more "
       "room noise. Ambient is 5-30 with spikes to about 1600.", "listening", "live",
       min=100.0, max=3000.0, step=25.0),
    _f("VAD_CONT_RMS", "float", 150.0, "Voice continuation threshold",
       "How quiet you can get, once you're already talking, before it decides you've stopped. "
       "Must be lower than the onset threshold.", "listening", "live",
       min=20.0, max=2000.0, step=10.0),
    _f("VAD_ONSET_FRAMES", "int", 3, "Frames before it counts as speech",
       "Consecutive 30 ms frames above the threshold needed to start. Higher rejects clicks. "
       "Only used by the fallback loudness detector, when the voice model below is not installed.",
       "listening", "live", min=1, max=10, step=1),
    _f("VAD_THRESHOLD", "float", 0.3, "Speech detector sensitivity",
       "How sure the Silero voice model must be that a sound is speech (0 to 1) before it starts "
       "recording. This model listens for the shape of speech, not just loudness, so it hears quiet "
       "talking and ignores the wake beep. Lower catches fainter or less clear speech; higher is "
       "stricter. Takes effect only when the voice model is installed - otherwise the loudness "
       "thresholds above apply instead.", "listening", "live", min=0.1, max=0.9, step=0.05),
    _f("SILENCE_STOP_MS", "int", 900, "Stop after silence",
       "How long you can pause mid-sentence before it stops listening.",
       "listening", "live", min=200, max=2000, step=50),
    _f("HARD_CAP_S", "float", 6.0, "Maximum command length",
       "Recording always stops after this long, even if you are still talking.",
       "listening", "live", min=2.0, max=15.0, step=0.5),
    _f("MIN_S", "float", 1.0, "Minimum command length",
       "Silence before this point does not stop the recording.",
       "listening", "live", min=0.3, max=3.0, step=0.1),
    _f("MIC_TARGET", "text", "", "Microphone",
       "PipeWire source name for pw-record --target. Empty means the default source. The "
       "capture stream is not restarted on change, so this applies from the next recording on, "
       "not to one already in progress.",
       "listening", "live"),
    _f("WAKE_SKIP_S", "float", 0.20, "Wake cue skip margin",
       "Extra silence discarded after the wake cue itself, on top of the cue's own length "
       "(computed automatically, so turning cues off already skips nothing extra). Raise if the "
       "first word is still being eaten, lower if the cue leaks into the transcript.",
       "wake", "live", min=0.0, max=1.0, step=0.01),
    _f("WAKE_COOLDOWN_S", "float", 2.0, "Wake cooldown",
       "A wake hit this soon after the previous command finished is ignored.",
       "wake", "live", min=0.0, max=10.0, step=0.5),
    _f("CUE_MODE", "enum", "haptic", "Wake/end cue",
       "How it signals it started/stopped listening: a haptic buzz (default - the mic can't hear "
       "it, so capture can start immediately), a tone through the speaker, both, or none.",
       "wake", "live", values=["haptic", "sound", "both", "off"]),
    _f("CUE_WAKE_HZ", "text", "660,990", "Wake cue tones",
       "Comma-separated frequencies in Hz for the rising start chirp.", "wake", "live"),
    _f("CUE_END_HZ", "text", "520", "End cue tone",
       "Comma-separated frequencies in Hz for the note that means it stopped listening.",
       "wake", "live"),
    _f("OVERLAY_ON", "bool", True, "Show what it heard",
       "Shows a notification banner with the transcript and what it did (the tool called and its "
       "confidence, or that it fell through to chat), so you can tell a mis-hearing from a "
       "mis-routing at a glance.", "wake", "live"),
    _f("CAPTURE_KEEP", "int", 20, "Save audio captures",
       "Keep this many of the most recent wake/listen recordings (wav + json transcript) under "
       "/home/miro/heytux/captures, for debugging transcription. 0 disables.",
       "wake", "live", min=0, max=200, step=1),
    _f("FOLLOWUP_S", "float", 5.0, "Follow-up window",
       "After a chat answer, keep listening this long for a follow-up question without the wake "
       "word. Ends silently if you say nothing, or after any phone command. 0 disables.",
       "wake", "live", min=0.0, max=15.0, step=0.5),
    _f("CONF_FLOOR", "float", 0.5, "Confidence floor",
       "Below this the Needle router's answer is discarded and the question goes to the chat model. "
       "Applies only to Needle3: the embedding router uses its own similarity floor, and a "
       "function-calling router decides per tool call, so neither is affected by this value.",
       "routing", "live", min=0.0, max=1.0, step=0.05),
    _f("GROUNDING_ON", "bool", True, "Require lexical grounding",
       "Reject a tool call that shares no meaningful word with what you said (the router "
       "confabulating a match) and send it to chat instead.", "routing", "live"),
    _f("LLM_SYS", "text",
       "You are a voice assistant running on a phone. Reply in one or two short spoken "
       "sentences, no lists, no markdown, no emoji.",
       "Chat instructions", "System prompt given to the local chat model.", "routing", "live"),
    _f("CHAT_HISTORY_TURNS", "int", 4, "Chat memory",
       "How many previous question/answer pairs the chat model sees, so \"explain it\" follows "
       "\"tell me a joke\". 0 makes every question stand alone.", "routing", "live",
       min=0, max=8, step=1),
    _f("CHAT_MEMORY_S", "float", 60.0, "Chat memory timeout",
       "Forget the conversation when the last chat answer is older than this.", "routing", "live",
       min=5.0, max=600.0, step=5.0),
    _f("KWS_TEXT", "text", "▁HE Y ▁LI N U X @HEY_LINUX\n", "Wake word",
       "Keyword tokens for the spotter, plain tokens + @label. Per-keyword ':boost #threshold' "
       "is optional; '#' must be a small probability (e.g. #0.15) - the old ':0.25 #1.5' "
       "could never fire. Test before saving.", "wake", "model"),
    # Live per-slot model pickers. Values (manifest ids) are filled at startup from the registry,
    # listing only models actually staged on disk; "live" means POST /config hot-swaps the model in
    # place (close old, load new) with no daemon restart. See heytux.py: discover_models / swap_*.
    _f("CHAT_MODEL", "enum", "Qwen3-0.6B-Q4_K_M", "Chat model",
       "The local LLM that answers anything not handled by a phone command. The daemon hosts it on "
       "127.0.0.1:8771. Bigger models answer better but load slower and use more RAM.",
       "models", "live", values=[]),
    _f("ROUTER_MODEL", "enum", "all-MiniLM-L6-v2-Q8_0", "Router model",
       "Decides which phone command (if any) a request maps to. \"Needle3\" is the tuned baseline; "
       "an embedding model routes by sentence similarity; a function-calling GGUF routes by tool "
       "call. The confidence floor and grounding apply to all three.",
       "models", "live", values=[]),
    _f("ASR_MODEL", "enum", "moonshine-base-en-quantized", "Speech recognition model",
       "Which model turns your voice into text. moonshine-base is a good accuracy/speed balance; "
       "parakeet is more accurate on quiet speech; moonshine-tiny is fastest.",
       "models", "live", values=[]),
    _f("TTS_MODEL", "enum", "vits-piper-en_US-lessac-low-int8", "Voice",
       "Speech synthesis voice used for spoken answers and acknowledgements.",
       "models", "live", values=[]),
    _f("VAD_MODEL", "enum", "silero_vad", "Voice activity detector",
       "The model that decides, frame by frame, when you are speaking. Silero is the baseline; "
       "TEN is an alternative. Takes effect from the next recording on.",
       "models", "live", values=[]),
    _f("PORT", "int", 8770, "HTTP port",
       "Port the daemon listens on. Changing this restarts heytux.",
       "advanced", "restart", min=1024, max=65535, step=1),
]

FIELDS = {f["key"]: f for f in SCHEMA}


def defaults():
    return {f["key"]: f["default"] for f in SCHEMA}


def parse_cue_freqs(spec):
    """Parse a comma-separated cue tone spec (CUE_WAKE_HZ / CUE_END_HZ) into floats.

    All-or-nothing: any bad token means NOTHING is playable, so this returns [] rather than the
    good tokens only. heytux.py's wake_skip_s() and Models._cue() both call this single parser,
    so a malformed spec can never make the mic-stream skip discard audio for a cue that actually
    plays nothing (they used to parse independently and disagree on exactly that).
    """
    try:
        return [float(x) for x in str(spec).split(",") if x.strip()]
    except ValueError:
        return []


def _coerce(f, v):
    """Return v converted to the field's type, or raise TypeError/ValueError."""
    t = f["type"]
    if t == "bool":
        if not isinstance(v, bool):
            raise TypeError("bool")
        return v
    if t == "int":
        # bool is a subclass of int; reject it explicitly.
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            raise TypeError("int")
        return int(v)
    if t == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("float")
        return float(v)
    if not isinstance(v, str):
        raise TypeError("str")
    return v


def validate(changes):
    """Check a proposed change set. Returns (clean, errors).

    The caller MUST apply nothing when errors is non-empty - this function does not enforce
    that, it only reports. Enum fields whose `values` list is empty accept any string, because
    the value list is discovered from disk at runtime.
    """
    clean, errors = {}, {}
    for k, v in changes.items():
        f = FIELDS.get(k)
        if f is None:
            errors[k] = "unknown setting"
            continue
        try:
            v = _coerce(f, v)
        except (TypeError, ValueError):
            errors[k] = "expected %s" % f["type"]
            continue
        if f["type"] in ("int", "float") and not (f["min"] <= v <= f["max"]):
            errors[k] = "must be between %g and %g" % (f["min"], f["max"])
            continue
        if f["type"] == "enum" and f.get("values") and v not in f["values"]:
            errors[k] = "must be one of: %s" % ", ".join(f["values"])
            continue
        clean[k] = v
    return clean, errors


def _read_json(path, want_type):
    """Parse a JSON file of the expected type. Returns None on any problem - never raises."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as e:
        log("ignoring %s (%s)" % (path, e))
        return None
    if not isinstance(data, want_type):
        log("ignoring %s (expected a JSON object)" % path)
        return None
    return data


def overlay(stored):
    """Defaults overlaid with an already-parsed stored dict (validated; bad values logged and
    dropped). Separated from load() so the daemon can migrate the raw dict - e.g. an old STT_DIR/
    TTS_DIR path to a model id - before it is validated against the current schema."""
    vals = defaults()
    clean, errors = validate(stored or {})
    for k, msg in errors.items():
        log("ignoring %s (%s)" % (k, msg))
    vals.update(clean)
    return vals


def load(path=CONFIG_PATH):
    """Defaults overlaid with the stored file. Never raises; bad values fall back to defaults."""
    return overlay(_read_json(path, dict))


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save(path, values):
    """Persist only the non-default values, atomically. Returns what was written."""
    d = defaults()
    keep = {k: v for k, v in values.items() if k in d and v != d[k]}
    _write_json(path, keep)
    return keep


def load_tools(path=TOOLS_PATH):
    """Per-tool overrides: {name: {"enabled": bool, "description": str}}. Never raises."""
    data = _read_json(path, dict)
    if data is None:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def save_tools(path, overrides):
    _write_json(path, overrides)


def merge_tools(builtin, overrides):
    """Apply overrides onto the built-in tool table and return a NEW list.

    `builtin` is the source of truth for which tools exist: an override naming a tool that no
    longer exists in code is ignored, and a tool added in code with no override appears as-is.
    """
    out = []
    for t in builtin:
        fn = t["function"]
        ov = overrides.get(fn["name"]) or {}
        if ov.get("enabled") is False:
            continue
        desc = ov.get("description")
        if desc:
            t = {"type": t["type"], "function": dict(fn, description=desc)}
        out.append(t)
    return out
