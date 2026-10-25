#!/usr/bin/env python3
# test_heytux.py - offline unit checks for heytux.py. Plain asserts, no pytest, no phone, no models.
# Imports heytux WITHOUT loading models (sherpa_onnx is only imported inside Models.__init__).
# Run:  python3 test_heytux.py
import io, json, os, sys, time, urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
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

def t_remember_drops_expired_history_on_tool_turn():
    # A fast-path tool turn never calls _chat_history(), so _remember itself must expire a stale
    # conversation - else the next chat request sees context that should have timed out.
    saved = (J.CFG["CHAT_HISTORY_TURNS"], J.CFG["CHAT_MEMORY_S"])
    J.CFG["CHAT_HISTORY_TURNS"], J.CFG["CHAT_MEMORY_S"] = 4, 60.0
    J.CHAT_HISTORY[:] = [("tell me a joke", "a joke.")]
    J.chat_last = time.time() - 61                 # older than CHAT_MEMORY_S
    try:
        J._remember("turn on the flashlight", "Flashlight on.")
    finally:
        J.CFG["CHAT_HISTORY_TURNS"], J.CFG["CHAT_MEMORY_S"] = saved
    assert J.CHAT_HISTORY == [("turn on the flashlight", "Flashlight on.")], J.CHAT_HISTORY

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

# --- reply-language heuristic (voice selection) ---
def t_is_czech_diacritics():
    assert J.is_czech("Dnes je úterý a svítí slunce.")           # diacritics -> Czech

def t_is_czech_stopwords_no_diacritics():
    assert J.is_czech("ano to je pravda")                        # >=2 Czech words, no diacritics

def t_is_czech_english_is_false():
    assert not J.is_czech("The capital of France is Paris.")
    assert not J.is_czech("The café serves a soufflé and piñá coládá.")
    assert not J.is_czech("é á ó í ú")
    assert not J.is_czech("Flashlight on.")                      # a tool ack stays English
    assert not J.is_czech("I am a voice assistant.")             # 'i'/'a' must NOT read as Czech
    assert not J.is_czech("a a a the the")                       # repeated articles can't reach 2

def t_is_czech_one_word_not_enough():
    assert not J.is_czech("the ano here")                        # a single Czech word is not enough

def t_clean_spoken_keeps_czech_drops_emoji():
    out = J.clean_spoken("Dobrý den 😀, jak se máš?")
    assert "Dobrý" in out and "máš" in out, out                  # diacritics survive
    assert "😀" not in out, out                                   # emoji dropped

# --- hybrid routing: fast path vs chat model, grounding/enabled gates, execute() ---
class _FakeChat:
    def __init__(self, reply): self.reply = reply
    def tool_chat(self, msgs, tools): return self.reply

class _FakeModels:
    def __init__(self, route_reply=None, chat_reply=None):
        self._route_reply = route_reply
        self.chat = _FakeChat(chat_reply) if chat_reply is not None else None
        self.spoke = []
    class _R:
        scored = True
        def __init__(self, reply): self.reply = reply
        def complete(self, text):
            if self.reply is None: raise AssertionError("router must not be consulted")
            return self.reply
    @property
    def router(self): return self._R(self._route_reply)
    def speak(self, text): self.spoke.append(text)
    def _slot_alive(self, slot): return True

def _hybrid_env(**cfg):
    """Snapshot/patch CFG + globals for a hybrid-routing test; returns a restore() callable."""
    keys = ["ROUTE_MODE", "HYBRID_FLOOR", "FOLLOWUP_ROUTE", "CONF_FLOOR", "GROUNDING_ON",
            "CHAT_HISTORY_TURNS", "LANG_AUTO"]
    saved_cfg = {k: J.CFG[k] for k in keys}
    saved = (J.M, J.execute, list(J.MERGED_TOOLS))
    J.CFG.update({"ROUTE_MODE": "hybrid", "HYBRID_FLOOR": 0.75, "FOLLOWUP_ROUTE": "normal",
                  "CONF_FLOOR": 0.5, "GROUNDING_ON": True, "CHAT_HISTORY_TURNS": 0, "LANG_AUTO": False})
    J.CFG.update(cfg)
    J.CHAT_HISTORY.clear()
    def restore():
        J.CFG.update(saved_cfg)
        J.M, J.execute, J.MERGED_TOOLS = saved
    return restore

def t_hybrid_fast_path_executes_tool():
    restore = _hybrid_env()
    calls = []
    J.M = _FakeModels(route_reply=_needle(["flashlight_on"], 0.9))
    J.execute = lambda n, a: calls.append((n, a)) or "Flashlight on."
    try:
        route, said, c, conf, names = J._decide_and_run("turn on the flashlight", {}, dry=False, followup=False)
    finally:
        restore()
    assert route == "tool-fast", route
    assert calls == [("flashlight_on", {})], calls
    assert said == "Flashlight on.", said

def t_hybrid_low_conf_goes_to_chat_model():
    restore = _hybrid_env()
    J.M = _FakeModels(route_reply=_needle(["flashlight_on"], 0.6),      # >=CONF_FLOOR, <HYBRID_FLOOR
                      chat_reply={"content": "It is sunny.", "calls": []})
    J.execute = lambda n, a: (_ for _ in ()).throw(AssertionError("no tool on a chat answer"))
    try:
        route, said, c, conf, names = J._decide_and_run("turn on the flashlight", {}, dry=False, followup=False)
    finally:
        restore()
    assert route == "llm-answer", route
    assert said == "It is sunny.", said

def t_hybrid_echo_falls_back_without_speaking_or_remembering_echo():
    restore = _hybrid_env()
    try:
        for dry in (False, True):
            for echo in ("**What is today's date?**", "What is todays dates?"):
                J.M = _FakeModels(route_reply=_needle([], 0.0),
                                  chat_reply={"content": echo, "calls": []})
                with patch.object(J, "chat_llm", return_value="Today is October 25.") as fallback, \
                     patch.object(J, "_remember") as remember, patch.object(J, "log") as log:
                    r = J.pipeline("what is today's date", {}, dry=dry)
                fallback.assert_called_once_with("what is today's date", remember=False)
                # remembered once, by chat_tool, with the accepted fallback answer (never the echo)
                assert remember.call_args_list == ([] if dry else
                                                   [call("what is today's date", "Today is October 25.")])
                assert r["said"] == "Today is October 25.", r
                assert J.M.spoke == ([] if dry else [r["said"]]), J.M.spoke
                assert any("chat_tool echo" in str(c) for c in log.call_args_list)
    finally:
        restore()

def t_hybrid_non_echo_answer_is_spoken():
    restore = _hybrid_env()
    J.M = _FakeModels(route_reply=_needle([], 0.0),
                      chat_reply={"content": "Today's date is October 25.", "calls": []})
    try:
        with patch.object(J, "chat_llm") as fallback:
            r = J.pipeline("what is today's date", {}, dry=False)
        fallback.assert_not_called()
        assert J.M.spoke == ["Today's date is October 25."], (r, J.M.spoke)
    finally:
        restore()

def t_hybrid_short_greeting_is_not_an_echo():
    # "hello" -> "Hello!" is a correct reply: one inference, spoken as is.
    for q, a in (("hello", "Hello!"), ("ahoj", "Ahoj!"), ("good morning", "Good morning!")):
        with patch.object(J, "M", _FakeModels(chat_reply={"content": a, "calls": []})), \
             patch.object(J, "chat_llm") as fallback:
            assert J.chat_tool(q, dry=True) == (False, a, [])
        fallback.assert_not_called()

def t_hybrid_echo_fallback_failure_is_safe():
    echo = {"content": "What is today's date?", "calls": []}
    for fb in (None, "What is today's date?", "what_time_is_it"):   # failed, echoed again, bare name
        with patch.object(J, "M", _FakeModels(chat_reply=echo)), \
             patch.object(J, "chat_llm", return_value=fb), patch.object(J, "_remember") as remember:
            assert J.chat_tool("what is today's date", dry=False) == (False, J.CHAT_FAIL, [])
        remember.assert_not_called()

def t_hybrid_bare_tool_name_falls_back():
    restore = _hybrid_env()
    try:
        for text in ("what time is it", "what_time_is_it"):
            J.M = _FakeModels(chat_reply={"content": text, "calls": []})
            with patch.object(J, "chat_llm", return_value="October 25.") as fallback, \
                 patch.object(J, "execute") as execute:
                assert J.chat_tool("what is today's date", dry=True) == (False, "October 25.", [])
            fallback.assert_called_once_with("what is today's date", remember=False)
            execute.assert_not_called()
    finally:
        restore()

def t_hybrid_system_prompt_is_stable_and_scoped():
    with patch.dict(J.CFG, ROUTE_MODE="hybrid", LANG_AUTO=True), \
         patch.object(J, "_live_facts", side_effect=AssertionError("volatile prefix")):
        system = J._chat_sys()
        assert J.HYBRID_HINT in system and J.LANG_HINT in system, system
        assert "time, date, battery" in system and "never guess" in system
        assert "only for explicit phone commands" in system and "singing in words, without tools" in system
        assert "Czech" in system and "English" in system and "Never use another language" in system
        assert J._chat_sys() == system
        # the chat_llm fallback offers no tools but keeps the Czech/English rule
        assert J._chat_sys(tools=False) == J.CFG["LLM_SYS"] + " " + J.LANG_HINT
        J.CFG["LANG_AUTO"] = False
        assert J._chat_sys() == J.CFG["LLM_SYS"] + " " + J.HYBRID_HINT
        J.CFG["ROUTE_MODE"] = "router"
        assert J._chat_sys() == J.CFG["LLM_SYS"]

def t_echo_fallback_uses_facts_without_tool_instructions():
    with patch.dict(J.CFG, ROUTE_MODE="hybrid"), \
         patch.object(J, "_live_facts", return_value="Current date: October 25."), \
         patch.object(J, "_chat_history", return_value=[]), \
         patch.object(J, "_llm_once", return_value="October 25.") as once:
        assert J.chat_llm("date?", remember=False) == "October 25."
        system = json.loads(once.call_args.args[0])["messages"][0]["content"]
        assert "Current date: October 25." in system and J.HYBRID_HINT not in system

def t_czech_bench_uses_daemon_system_prompt():
    from bench import toolcall as B
    be = SimpleNamespace(chat=Mock(return_value={"text": "Česky."}))
    with patch.object(J, "_chat_sys", return_value="daemon prompt") as system:
        rows = B.czech_answers(be, "fake", 90)
    system.assert_called_once_with(tools=False)
    assert len(rows) == 12
    assert all(c.args[0][0]["content"] == "daemon prompt" for c in be.chat.call_args_list)

def t_hybrid_bench_accepts_date_tool_but_rejects_chat_tool_trigger():
    from bench import hybrid_e2e as B
    rows = [{"phrase": phrase, "expected": None, "route": "llm-tool", "name": name,
             "lat_ms": 100} for phrase, name in [("what is today's date", "what_time_is_it"),
                                                  ("tell me a joke", "play_pause")]]
    assert B.summarize(rows)["none_accuracy"] == 0.5

def t_spoken_date_accepts_ordinals_but_not_year_substrings():
    from test_routing import spoken_date_matches
    for day, words in [(1, "first"), (2, "second"), (3, "third"), (12, "twelfth"),
                       (20, "twentieth"), (25, "twenty-fifth"), (30, "thirtieth"), (31, "thirty first")]:
        assert spoken_date_matches("October %s, 2026." % words, "october", day)
        assert spoken_date_matches("October %s, 2026." % day, "october", day)
    assert spoken_date_matches("October 25th, 2026.", "october", 25)
    assert not spoken_date_matches("October 26th, 2025.", "october", 25)
    assert not spoken_date_matches("October 25th, 2026.", "october", 2)
    assert not spoken_date_matches("September twenty-fifth.", "october", 25)
    for said, day in (("October twenty-first.", 1), ("October thirty first.", 1),
                      ("October twenty-second.", 2), ("October twenty-fifth.", 5), ("October 21st.", 1)):
        assert not spoken_date_matches(said, "october", day), (said, day)

def t_hybrid_llm_tool_call_runs_execute():
    restore = _hybrid_env()
    calls = []
    J.M = _FakeModels(route_reply=_needle([], 0.0),                     # router declines -> chat model
                      chat_reply={"content": "", "calls": [{"name": "flashlight_on", "arguments": {}}]})
    J.execute = lambda n, a: calls.append((n, a)) or "Flashlight on."
    try:
        route, said, c, conf, names = J._decide_and_run("turn on the flashlight", {}, dry=False, followup=False)
    finally:
        restore()
    assert route == "llm-tool", route
    assert calls == [("flashlight_on", {})], calls
    assert names == ["flashlight_on"], names

def t_hybrid_disabled_tool_is_rejected():
    # flashlight_on is grounded, but with it removed from the enabled set the model's call must NOT run.
    restore = _hybrid_env()
    calls = []
    J.M = _FakeModels(route_reply=_needle([], 0.0),
                      chat_reply={"content": "", "calls": [{"name": "flashlight_on", "arguments": {}}]})
    J.execute = lambda n, a: calls.append((n, a)) or "ran"
    J.MERGED_TOOLS = [t for t in J.MERGED_TOOLS if t["function"]["name"] != "flashlight_on"]
    try:
        route, said, c, conf, names = J._decide_and_run("turn on the flashlight", {}, dry=False, followup=False)
    finally:
        restore()
    assert calls == [], "a disabled tool must never execute"
    assert route == "llm-answer", route
    assert said == J.CHAT_FAIL, said

def t_hybrid_czech_llm_tool_call_runs_execute():
    restore = _hybrid_env()
    calls = []
    J.M = _FakeModels(route_reply=_needle(["flashlight_on"], 0.99),
                      chat_reply={"content": "", "calls": [{"name": "flashlight_on", "arguments": {}}]})
    J.execute = lambda n, a: calls.append((n, a)) or "Flashlight on."
    try:
        assert not J._grounded("flashlight_on", "rozsviť baterku")
        route, said, c, conf, names = J._decide_and_run("rozsviť baterku", {}, dry=False, followup=False)
    finally:
        restore()
    # High-confidence router call still fails grounding; the explicit LLM call executes.
    assert route == "llm-tool", route
    assert calls == [("flashlight_on", {})], calls
    assert names == ["flashlight_on"], names

def t_hybrid_confabulated_tool_call_is_rejected():
    restore = _hybrid_env()
    calls = []
    logs = []
    saved_log = J.log
    J.log = lambda *args: logs.append(args)
    # Exercise the known-name check independently of the enabled-set check.
    J.MERGED_TOOLS.append({"type": "function", "function": {"name": "get_phone_phone"}})
    J.M = _FakeModels(route_reply=_needle([], 0.0),
                      chat_reply={"content": "", "calls": [{"name": "get_phone_phone", "arguments": {}}]})
    J.execute = lambda n, a: calls.append((n, a)) or "ran"
    try:
        route, said, c, conf, names = J._decide_and_run("what is the capital of france", {}, dry=False, followup=False)
    finally:
        J.log = saved_log
        restore()
    assert calls == [], "a confabulated tool call must never execute"
    assert route == "llm-answer" and said == J.CHAT_FAIL, (route, said)
    assert ("chat_tool rejected:", "get_phone_phone", "unknown") in logs, logs

# --- follow-up routing: FOLLOWUP_ROUTE=chat skips the router on a follow-up turn ---
def t_followup_chat_skips_router_in_router_mode():
    restore = _hybrid_env(ROUTE_MODE="router", FOLLOWUP_ROUTE="chat")
    J.M = _FakeModels(route_reply=None)        # .router.complete would AssertionError if consulted
    saved = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=0: io.BytesIO(
        b'{"choices":[{"message":{"content":"follow up answer."}}]}')
    try:
        route, said, c, conf, names = J._decide_and_run("and the moon?", {}, dry=True, followup=True)
    finally:
        urllib.request.urlopen = saved
        restore()
    assert route == "chat-followup", route
    assert said == "follow up answer.", said

def t_followup_normal_still_uses_router():
    restore = _hybrid_env(ROUTE_MODE="router", FOLLOWUP_ROUTE="normal")
    J.M = _FakeModels(route_reply=_needle(["flashlight_on"], 1.0))
    J.execute = lambda n, a: "Flashlight on."
    try:
        route, said, c, conf, names = J._decide_and_run("turn on the flashlight", {}, dry=True, followup=True)
    finally:
        restore()
    assert route == "tool", route            # router mode: a normal follow-up routes as usual

# --- review regressions: fake backends only; these tests never load a second model ---
def _empty_models():
    m = J.Models.__new__(J.Models)
    m.loaded, m.rss_mb = {}, {}
    m._chat_prewarm_key = None
    m.chat = m.router = m.asr = m.tts = m.tts_alt = m.vad = m._vad_entry = None
    return m

def t_prewarm_dedup_prefix_and_reload():
    m = _empty_models()
    chat = SimpleNamespace(tool_chat=Mock(), close=Mock())
    with patch.dict(J.CFG, ROUTE_MODE="hybrid", LANG_AUTO=False), \
         patch.dict(J.REGISTRY, chat={"path": "/fake.gguf"}), \
         patch.object(J.HB, "LlamaServer", return_value=chat), \
         patch.object(J, "MERGED_TOOLS", list(J.MERGED_TOOLS)):
        m._load_slot("chat", "chat")
        m._prewarm_chat()
        assert chat.tool_chat.call_count == 1
        J.CFG["LLM_SYS"] += " changed"
        m._prewarm_chat()
        J.CFG["LANG_AUTO"] = True
        m._prewarm_chat()
        J.MERGED_TOOLS = J.MERGED_TOOLS[:-2]
        m._prewarm_chat()
        m._prewarm_chat()
        assert chat.tool_chat.call_count == 4
        m._close_slot("chat")
        m._load_slot("chat", "chat")             # same id, new server: cannot reuse the old key
        assert chat.tool_chat.call_count == 5

def t_prewarm_failed_attempt_is_retryable():
    m = _empty_models()
    m.loaded["chat"] = "chat"
    m.chat = SimpleNamespace(tool_chat=Mock(side_effect=[RuntimeError("cold"), {}]))
    m._prewarm_chat()
    assert m._chat_prewarm_key is None
    m._prewarm_chat()
    m._prewarm_chat()
    assert m.chat.tool_chat.call_count == 2

def t_config_prewarms_final_prefix_after_swap_and_live_changes():
    m = _empty_models()
    chat = SimpleNamespace(tool_chat=Mock(), close=Mock())
    h = J.H.__new__(J.H)
    h._send = Mock()
    h._body = lambda: {"CHAT_MODEL": "chat", "ROUTE_MODE": "hybrid",
                       "LLM_SYS": "New system.", "LANG_AUTO": True}
    with patch.dict(J.CFG, ROUTE_MODE="router", CHAT_MODEL="old", LANG_AUTO=False), \
         patch.dict(J.REGISTRY, chat={"path": "/fake.gguf"}), \
         patch.object(J.HB, "LlamaServer", return_value=chat), patch.object(J, "M", m), \
         patch.object(C, "validate", side_effect=lambda changes: (changes, {})), \
         patch.object(C, "save"):
        h._post_config()
        assert chat.tool_chat.call_count == 1
        assert chat.tool_chat.call_args.args[0][0]["content"] == \
            "New system. " + J.HYBRID_HINT + " " + J.LANG_HINT
        h._body = lambda: {"ROUTE_MODE": "hybrid"}
        h._post_config()
        assert chat.tool_chat.call_count == 1, "unchanged mode must not prewarm again"
        for change in ({"LLM_SYS": "Another system."}, {"LANG_AUTO": False}):
            h._body = lambda: change
            h._post_config()
        assert chat.tool_chat.call_count == 3
        assert chat.tool_chat.call_args.args[0][0]["content"] == "Another system. " + J.HYBRID_HINT

def t_tts_alt_startup_and_reload_failure_are_nonfatal():
    load = J.Models._load_slot
    def startup_load(m, slot, mid, strict=False):
        if slot == "tts_alt":
            return load(m, slot, mid, strict=strict)
        m.loaded[slot] = mid
    with patch.dict(sys.modules, sherpa_onnx=SimpleNamespace()), \
         patch.dict(J.REGISTRY, broken={}), \
         patch.object(J.HB, "load", side_effect=RuntimeError("bad voice")), \
         patch.object(J.Models, "_resolve", return_value="broken"), \
         patch.object(J.Models, "_load_slot", startup_load), \
         patch.object(J.Models, "build_kws") as kws:
        m = J.Models()
        assert m.tts_alt is None and m.loaded["tts_alt"] is None
        kws.assert_called_once()
        load(m, "tts_alt", "broken")
        assert m.tts_alt is None and m.loaded["tts_alt"] is None

def t_tts_alt_explicit_swap_reports_error_and_rolls_back():
    m = _empty_models()
    old = SimpleNamespace(close=Mock())
    restored = SimpleNamespace(close=Mock())
    m.tts_alt = old; m.loaded["tts_alt"] = "old"
    with patch.dict(J.REGISTRY, old={}, broken={}), \
         patch.object(J.HB, "load", side_effect=[RuntimeError("bad voice"), restored]):
        try:
            m.swap("tts_alt", "broken")
            assert False, "an explicit bad voice must report failure"
        except RuntimeError as e:
            assert str(e) == "bad voice"
    old.close.assert_called_once()
    assert m.tts_alt is restored and m.loaded["tts_alt"] == "old"
    with patch.dict(J.REGISTRY, old={}, broken={}), \
         patch.object(J.HB, "load", side_effect=RuntimeError("both broken")):
        try:
            m.swap("tts_alt", "broken")
            assert False
        except RuntimeError:
            pass
    assert m.tts_alt is None and m.loaded["tts_alt"] is None

def t_parse_tool_calls_rejects_malformed_arguments():
    for raw in ("{bad", "[]", "1", '"text"', "null", "false", [], [1], 0, 1, False, None):
        calls = J.HB.LlamaServer._parse_tool_calls({"tool_calls": [
            {"function": {"name": "set_volume", "arguments": raw}}]})
        assert calls == [{"name": "set_volume", "arguments": None}], (raw, calls)
    for raw in ({}, "{}", {"percent": 30}, '{"percent":30}'):
        calls = J.HB.LlamaServer._parse_tool_calls({"tool_calls": [
            {"function": {"name": "set_volume", "arguments": raw}}]})
        assert isinstance(calls[0]["arguments"], dict), calls

def t_chat_tool_logs_all_rejections_before_any_execution():
    restore = _hybrid_env()
    events = []
    J.MERGED_TOOLS = [t for t in J.MERGED_TOOLS if t["function"]["name"] != "wifi_off"]
    J.M = _FakeModels(chat_reply={"content": "Here is some text.", "calls": [
        {"name": "flashlight_on", "arguments": {}},
        {"name": "set_volume", "arguments": None},
        {"name": "wifi_off", "arguments": {}},
        {"name": "invented", "arguments": {}}]})
    J.execute = lambda n, a: events.append(("execute", n)) or "Done."
    try:
        with patch.object(J, "log", side_effect=lambda *args: events.append(args)):
            did, said, calls = J.chat_tool("do it")
        assert did and said == "Done."
        assert events == [("chat_tool rejected:", "set_volume", "bad arguments"),
                          ("chat_tool rejected:", "wifi_off", "not enabled"),
                          ("chat_tool rejected:", "invented", "unknown"),
                          ("execute", "flashlight_on")], events
        events.clear()
        J.M.chat.reply["calls"] = [{"name": "set_volume", "arguments": None}]
        with patch.object(J, "log", side_effect=lambda *args: events.append(args)):
            did, said, calls = J.chat_tool("do it")
        assert not did and said == "Here is some text." and calls == []
        assert events == [("chat_tool rejected:", "set_volume", "bad arguments")]
    finally:
        restore()

def t_gguf_router_drops_malformed_arguments():
    srv = SimpleNamespace(route=lambda *a: {"calls": [{"name": "set_volume", "arguments": None},
                                                      {"name": "wifi_off", "arguments": {}}]})
    router = J.HB.GgufRouter(srv, J.TOOLS)
    r = router.complete("volume")
    assert r["function_calls"] == [{"name": "wifi_off", "arguments": {}}] and r["confidence"] == 1.0
    srv.route = lambda *a: {"calls": [{"name": "set_volume", "arguments": None}]}
    r = router.complete("volume")
    assert r["function_calls"] == [] and r["confidence"] == 0.0

def t_hybrid_unscored_router_goes_directly_to_chat():
    assert J.HB.EmbeddingRouter.scored and not J.HB.GgufRouter.scored and not J.Needle.scored
    restore = _hybrid_env()
    J.M = _FakeModels(route_reply=None, chat_reply={"content": "An answer.", "calls": []})
    try:
        with patch.object(_FakeModels._R, "scored", False):
            route, said, calls, conf, names = J._decide_and_run("flashlight", {}, False, False)
        assert route == "llm-answer" and said == "An answer.", (route, said)
    finally:
        restore()

def t_models_alt_slot_maps_to_settings_field():
    m = _empty_models()
    m.tts_alt = SimpleNamespace()
    m.loaded["tts_alt"] = "voice"
    m.rss_mb["tts_alt"] = 42
    s = m.models_status()["tts_alt"]
    assert s["field"] == "TTS_MODEL_ALT" and s["loaded"] and s["rss_mb"] == 42, s


def t_bench_guard_stops_and_waits_for_terminal_state():
    from bench import toolcall as B
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=1 if cmd[0] == "pgrep" else 0, stdout="", stderr="")
    with patch.object(B.subprocess, "run", side_effect=run), \
         patch.object(B, "_daemon_state", side_effect=["activating", "deactivating", "inactive"]), \
         patch.object(B.time, "sleep"), patch.object(B.time, "monotonic", side_effect=[0, 0, 1, 2]):
        B.ensure_daemon_stopped()
    assert commands[0] == ["systemctl", "--user", "stop", "heytux"]
    assert len(commands) == 7, commands

def t_bench_guard_refuses_ambiguous_state_or_orphan_server():
    from bench import toolcall as B
    for state, process, listener, stop_rc, lookup_rc in [
        ("", "", "", 0, 1), ("unknown", "", "", 0, 1),
        ("activating", "", "", 0, 1), ("deactivating", "", "", 0, 1),
        ("inactive", "123 llama-server --port 8771", "", 0, 0),
        ("failed", "123 llama-server --port=8771", "", 0, 0),
        ("inactive", "", "LISTEN 0 128 127.0.0.1:8771", 0, 1),
        ("inactive", "", "", 1, 1), ("inactive", "", "", 0, 2)]:
        def run(cmd, **kwargs):
            rc, out = ((stop_rc, "") if cmd[0] == "systemctl" else
                       (lookup_rc, process) if cmd[0] == "pgrep" else (0, listener))
            return SimpleNamespace(returncode=rc, stdout=out, stderr="lookup/stop failed")
        with patch.object(B.subprocess, "run", side_effect=run), \
             patch.object(B, "_daemon_state", return_value=state), \
             patch.object(B.time, "sleep"), patch.object(B.time, "monotonic", side_effect=[0, 0, 31]):
            try:
                B.ensure_daemon_stopped()
                assert False, (state, process, listener, stop_rc, lookup_rc)
            except SystemExit as e:
                assert "REFUSING" in str(e)
    with patch.object(B.subprocess, "run") as run:
        B.ensure_daemon_stopped(allow=True)
        run.assert_not_called()

def t_bench_guard_failed_lookup_is_not_inactive():
    from bench import toolcall as B
    with patch.object(B.subprocess, "run", return_value=SimpleNamespace(returncode=4, stdout="inactive")):
        assert B._daemon_state() == "unknown"
    with patch.object(B.subprocess, "run", side_effect=OSError("missing systemctl")):
        assert B._daemon_state() == "unknown"

def t_hybrid_bench_restores_after_failed_mutation():
    from bench import hybrid_e2e as B
    before = {"CHAT_MODEL": "original", "ROUTE_MODE": "router"}
    with patch.object(sys, "argv", ["hybrid_e2e.py", "--model", "broken"]), \
         patch.object(B, "_get", return_value={"values": before}), \
         patch.object(B, "_post", side_effect=[RuntimeError("swap failed"), {}]) as post:
        try:
            B.main()
            assert False
        except RuntimeError as e:
            assert str(e) == "swap failed"
        assert post.call_count == 2
        assert post.call_args.args[2] == before


def test_all():
    """pytest entry: run every t_* check and fail on the first broken assertion."""
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        fn()


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        check(name[2:], fn)
    print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
