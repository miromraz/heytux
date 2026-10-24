#!/usr/bin/env python3
# test_heytux.py - offline unit checks for heytux.py. Plain asserts, no pytest, no phone, no models.
# Imports heytux WITHOUT loading models (sherpa_onnx is only imported inside Models.__init__).
# Run:  python3 test_heytux.py
import io, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import heytux as J
import heytux_config as C

FAILS = []

def check(name, fn):
    try:
        fn()
        print("PASS %s" % name)
    except AssertionError as e:
        FAILS.append(name)
        print("FAIL %s: %s" % (name, e))

def _needle(names, conf=1.0, reasoning=""):
    return {"function_calls": [{"name": n, "arguments": {}} for n in names],
            "confidence": conf, "reasoning": reasoning}

# --- decide_route: one case per gate ---
def t_route_ask_question():
    route, _, _, _ = J.decide_route(_needle(["ask_question"]), "tell me a joke")
    assert route == "chat-tool", route

def t_route_empty_calls():
    route, _, _, _ = J.decide_route(_needle([]), "hmm")
    assert route == "chat-fallback", route

def t_route_nomatch_reasoning():
    # grounded + high conf, but reasoning admits no tool matches -> chat
    route, _, _, _ = J.decide_route(
        _needle(["flashlight_on"], 1.0, "factual question, closest match is flashlight_on"),
        "turn on the flashlight")
    assert route == "chat-fallback", route

def t_route_low_conf():
    route, _, _, _ = J.decide_route(_needle(["flashlight_on"], 0.3), "turn on the flashlight")
    assert route == "chat-fallback", route

def t_route_ungrounded():
    # high conf, no reasoning, but shares no content word with the tool -> confabulated -> chat
    route, _, _, _ = J.decide_route(_needle(["next_track"], 1.0), "when you please")
    assert route == "chat-fallback", route

def t_route_real_tool():
    route, calls, conf, names = J.decide_route(_needle(["flashlight_on"], 1.0), "turn on the flashlight")
    assert route == "tool", route
    assert names == ["flashlight_on"], names

# --- grounding ---
def t_content_words_drops_stopwords():
    assert J._content_words("turn on the flashlight") == {"turn", "flashlight"}, \
        J._content_words("turn on the flashlight")

def t_grounded_real():
    assert J._grounded("flashlight_on", "turn on the flashlight") is True

def t_grounded_no_shared_word():
    assert J._grounded("next_track", "when you please") is False

def t_grounded_confabulated_name():
    # a tool name Needle invented that isn't in the table can never be grounded
    assert J._grounded("get_phone_phone", "give me the full phone status") is False

# --- clean_spoken ---
def t_clean_spoken_markdown_and_sentences():
    out = J.clean_spoken("**Hello** there. Second sentence. Third one.")
    assert out == "Hello there. Second sentence.", repr(out)

def t_clean_spoken_length_cap():
    assert len(J.clean_spoken("word " * 100)) <= 220

# --- normalization before STT ---
def t_normalize_lifts_quiet_to_target():
    # a quiet-but-real signal within the +24 dB budget is lifted to ~-6 dBFS (peak ~0.501)
    quiet = np.full(100, 0.05, dtype=np.float32)      # +24 dB of 0.05 = 0.79 > 0.501, so cap not hit
    out = J.normalize(quiet)
    assert abs(out.max() - 10 ** (J.NORM_TARGET_DBFS / 20)) < 1e-3, out.max()

def t_normalize_gain_is_capped():
    # near-silence/noise must NOT be blown up to full scale: gain caps at +24 dB
    tiny = np.full(100, 0.001, dtype=np.float32)
    out = J.normalize(tiny)
    cap = 10 ** (J.NORM_MAX_GAIN_DB / 20)
    assert abs(float(out[0]) - 0.001 * cap) < 1e-6, float(out[0])
    assert out.max() < 0.1, "capped signal must stay far below full scale, got %r" % out.max()

def t_normalize_silence_is_safe():
    # all-zero input must not divide by zero / blow up
    assert J.normalize(np.zeros(10, dtype=np.float32)).max() == 0.0

def t_normalize_loud_is_attenuated():
    loud = np.full(100, 0.9, dtype=np.float32)
    assert J.normalize(loud).max() <= 1.0

# --- match_app ---
def t_match_app_exact():
    assert J.match_app("calculator") == ("calculator", "org.gnome.Calculator")

def t_match_app_substring():
    assert J.match_app("store") == ("app store", "io.github.kolunmi.Bazaar")

def t_match_app_unknown():
    assert J.match_app("nonsense") == ("nonsense", None)

# --- wake_skip_s invariant (moved from test_endpoints): skip covers exactly what the cue plays,
# plus at most the margin, never less (clipping speech) and never wildly more ---
def t_wake_skip_never_exceeds_what_the_cue_plays():
    note = J.WAKE_CUE_NOTE_S
    margin = J.CFG["WAKE_SKIP_S"]
    saved = (J.CFG["CUE_WAKE_HZ"], J.CFG["CUE_MODE"])
    try:
        for spec, mode in [("660,990", "sound"), ("660,bad", "sound"), ("", "sound"),
                           ("660,990", "both"), ("660,990", "haptic"), ("660,990", "off")]:
            J.CFG["CUE_WAKE_HZ"], J.CFG["CUE_MODE"] = spec, mode
            skip = J.wake_skip_s()
            freqs = C.parse_cue_freqs(spec) if mode in ("sound", "both") else []
            played = len(freqs) * note if freqs else 0.0
            assert played <= skip <= played + margin, \
                "%r/%s: plays %.3fs, skip %.3fs" % (spec, mode, played, skip)
        for mode in ("haptic", "off"):
            J.CFG["CUE_WAKE_HZ"], J.CFG["CUE_MODE"] = "660,990", mode
            assert J.wake_skip_s() == 0.0, "CUE_MODE=%s must skip nothing" % mode
    finally:
        J.CFG["CUE_WAKE_HZ"], J.CFG["CUE_MODE"] = saved

# --- tool registry completeness ---
EXPECTED_TOOLS = [
    "flashlight_on", "flashlight_off", "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
    "set_brightness", "set_volume", "battery_status", "phone_info", "set_timer", "what_time_is_it",
    "open_app", "take_photo", "take_selfie", "lock_screen", "play_pause", "next_track",
    "find_my_phone", "ask_question",
]

def t_tool_order_matches_pre_refactor():
    assert [t["function"]["name"] for t in J.TOOLS] == EXPECTED_TOOLS, \
        [t["function"]["name"] for t in J.TOOLS]

def t_every_spec_has_a_handler():
    names = {t["function"]["name"] for t in J.TOOLS}
    # ask_question is chat-only (decide_route intercepts it), never dispatched via execute()
    assert set(J._HANDLERS) | {"ask_question"} == names, set(J._HANDLERS)
    assert set(J._HANDLERS) <= names, "no orphan handler without a spec"

# --- merged-tools idx name ---
def t_idx_name_changes_when_override_disables_a_tool():
    base = J.needle_idx_path(J.TOOLS)
    disabled = C.merge_tools(J.TOOLS, {"flashlight_on": {"enabled": False}})
    assert J.needle_idx_path(disabled) != base, "disabling a tool must change the idx name"

# --- gdbus GVariant escaping ---
def t_gv_str_escaping():
    assert J._gv_str('hello') == '"hello"'
    assert J._gv_str('a"b') == '"a\\"b"'          # quote -> \"
    assert J._gv_str('a\\b') == '"a\\\\b"'         # backslash -> \\
    assert J._gv_str('-flag')[0] == '"'            # leading '-' can't be read as a CLI flag

# --- dry-run /text: execute() must NOT be called and nothing spoken ---
def t_dry_run_does_not_execute_or_speak():
    class FakeNeedle:
        def complete(self, text):
            return _needle(["flashlight_on"], 1.0)
    class FakeModels:
        def __init__(self): self.router = FakeNeedle(); self.spoke = []
        def speak(self, text): self.spoke.append(text)
    fake = FakeModels()
    called = []
    orig_M, orig_exec = J.M, J.execute
    J.M = fake
    J.execute = lambda name, args: called.append(name) or "SHOULD NOT RUN"
    try:
        res = J.pipeline("turn on the flashlight", {"record": 0, "stt": 0}, dry=True)
    finally:
        J.M, J.execute = orig_M, orig_exec
    assert res["route"] == "tool", res
    assert res["said"] == "", res
    assert res["calls"] == [{"name": "flashlight_on", "arguments": {}}], res["calls"]
    assert called == [], "execute() must not run in a dry route"
    assert fake.spoke == [], "nothing must be spoken in a dry route"

# --- chat memory: chat_llm with urlopen faked; the request body is what the LLM would see ---
def _chat(transcripts, turns=4, memory_s=60.0, remember=True):
    """Run chat_llm over transcripts from an empty memory; returns the sent messages lists."""
    sent = []
    def fake_urlopen(req, timeout=0):
        msgs = json.loads(req.data)["messages"]
        sent.append(msgs)
        return io.BytesIO(json.dumps(
            {"choices": [{"message": {"content": "answer %d." % len(sent)}}]}).encode())
    saved = (urllib.request.urlopen, J.CFG["CHAT_HISTORY_TURNS"], J.CFG["CHAT_MEMORY_S"])
    urllib.request.urlopen = fake_urlopen
    J.CFG["CHAT_HISTORY_TURNS"], J.CFG["CHAT_MEMORY_S"] = turns, memory_s
    J.CHAT_HISTORY.clear()
    try:
        for t in transcripts:
            J.chat_llm(t, remember=remember)
    finally:
        urllib.request.urlopen, J.CFG["CHAT_HISTORY_TURNS"], J.CFG["CHAT_MEMORY_S"] = saved
    return sent

def _pairs(msgs):
    return [(m["role"], m["content"]) for m in msgs[1:]]

def t_chat_messages_order():
    msgs = J.chat_messages("SYS", [("q1", "a1"), ("q2", "a2")], "q3")
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("system", "SYS"), ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"), ("user", "q3")], msgs

def t_chat_history_carries_context():
    sent = _chat(["tell me a joke", "explain it"])
    assert sent[1][0]["role"] == "system"
    assert _pairs(sent[1]) == [("user", "tell me a joke"), ("assistant", "answer 1."),
                               ("user", "explain it")], sent[1]

def t_chat_history_capped():
    sent = _chat(["q1", "q2", "q3", "q4"], turns=2)
    assert _pairs(sent[3]) == [("user", "q2"), ("assistant", "answer 2."),
                               ("user", "q3"), ("assistant", "answer 3."), ("user", "q4")], sent[3]
    assert len(J.CHAT_HISTORY) == 2, J.CHAT_HISTORY

def t_chat_history_zero_is_stateless():
    sent = _chat(["q1", "q2"], turns=0)
    assert _pairs(sent[1]) == [("user", "q2")], sent[1]
    assert J.CHAT_HISTORY == []

def t_chat_history_expires():
    _chat(["q1"])
    J.chat_last -= 61                     # last answer now older than CHAT_MEMORY_S=60
    saved = urllib.request.urlopen
    sent = []
    urllib.request.urlopen = lambda req, timeout=0: (sent.append(json.loads(req.data)["messages"])
        or io.BytesIO(b'{"choices":[{"message":{"content":"x."}}]}'))
    try:
        J.chat_llm("q2")
    finally:
        urllib.request.urlopen = saved
    assert _pairs(sent[0]) == [("user", "q2")], sent[0]

def t_chat_dry_does_not_remember():
    _chat(["q1", "q2"], remember=False)
    assert J.CHAT_HISTORY == []

# --- follow-up onset timeout: silence ends the capture after onset_s of audio, no speech ---
def t_record_onset_timeout_on_silence():
    class FakeModels:
        def vad_detector(self): return None          # energy VAD fallback path
        def play_end_cue(self): raise AssertionError("end cue must not play on no speech")
    orig_M, J.M = J.M, FakeModels()
    try:
        pcm, speech, stats = J.record(io.BytesIO(bytes(J.SR * 2 * 10)), onset_s=1.0)  # 10 s silence
    finally:
        J.M = orig_M
    assert speech is False and stats["vad_speech"] is False, stats
    assert abs(len(pcm) - J.SR) <= J.FRAME, "gave up after %.2f s, want 1.0" % (len(pcm) / J.SR)

def test_all():
    """pytest entry: run every t_* check and fail on the first broken assertion."""
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        fn()


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        check(name[2:], fn)
    print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
