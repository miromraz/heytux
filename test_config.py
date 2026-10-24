#!/usr/bin/env python3
# test_config.py - offline checks for heytux_config. Plain asserts, no pytest, no phone.
# Run:  python3 test_config.py
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heytux_config as C

FAILS = []

def check(name, fn):
    try:
        fn()
        print("PASS %s" % name)
    except AssertionError as e:
        FAILS.append(name)
        print("FAIL %s: %s" % (name, e))

def tmp():
    d = tempfile.mkdtemp()
    return os.path.join(d, "config.json")

# --- schema integrity ---
def t_defaults_cover_schema():
    d = C.defaults()
    assert set(d) == set(C.FIELDS), "defaults() must have exactly one entry per field"
    for k, f in C.FIELDS.items():
        assert f["apply"] in ("live", "reindex", "model", "restart"), "%s: bad apply class" % k
        assert f["type"] in ("bool", "int", "float", "text", "enum"), "%s: bad type" % k

def t_defaults_are_valid():
    clean, errors = C.validate(C.defaults())
    assert not errors, "defaults must validate cleanly, got %r" % errors

# --- validation ---
def t_accepts_in_range():
    clean, errors = C.validate({"VAD_RMS": 800})
    assert not errors, errors
    assert clean["VAD_RMS"] == 800.0, clean

def t_rejects_out_of_range():
    clean, errors = C.validate({"VAD_RMS": 99999})
    assert "VAD_RMS" in errors, "out-of-range value must be rejected"
    assert not clean, "nothing may be returned as clean"

def t_rejects_wrong_type():
    clean, errors = C.validate({"VAD_RMS": "loud"})
    assert "VAD_RMS" in errors, "string for a float must be rejected"

def t_rejects_bool_as_int():
    # bool is a subclass of int in Python; True must NOT satisfy an int field.
    clean, errors = C.validate({"VAD_ONSET_FRAMES": True})
    assert "VAD_ONSET_FRAMES" in errors, "bool must not pass as int"

def t_rejects_unknown_key():
    clean, errors = C.validate({"NOT_A_SETTING": 1})
    assert "NOT_A_SETTING" in errors, "unknown keys must be reported"

def t_all_or_nothing():
    clean, errors = C.validate({"VAD_RMS": 700, "VAD_ONSET_FRAMES": 999})
    assert errors, "one bad field must produce errors"
    assert "VAD_RMS" in clean, "clean still reports the good field"
    # The CALLER is responsible for applying nothing when errors is non-empty;
    # this test pins the contract that errors is truthy so the caller can check it.

# --- cue tone parsing (I1 round 2: wake_skip_s() and Models._cue() share this parser so they
# can never disagree about what a spec means - any bad token must drop the WHOLE spec, not just
# the bad token, or a skip computed from a partial parse could discard real speech while the
# cue itself (parsed the same all-or-nothing way) plays nothing to justify it) ---
def t_cue_freqs_normal():
    assert C.parse_cue_freqs("660,990") == [660.0, 990.0]

def t_cue_freqs_single():
    assert C.parse_cue_freqs("520") == [520.0]

def t_cue_freqs_malformed_is_all_or_nothing():
    # one bad token must drop the WHOLE spec, not just the bad token - this is the exact bug:
    # a partial-credit parse ("660,bad" -> [660.0]) would let the skip count a note the cue
    # never actually plays.
    assert C.parse_cue_freqs("660,bad") == []

def t_cue_freqs_empty():
    assert C.parse_cue_freqs("") == []

# --- persistence ---
def t_save_writes_only_non_defaults():
    p = tmp()
    v = C.defaults(); v["VAD_RMS"] = 900
    written = C.save(p, v)
    assert written == {"VAD_RMS": 900}, "only non-defaults are stored, got %r" % written
    on_disk = json.load(open(p))
    assert on_disk == {"VAD_RMS": 900}, on_disk

def t_round_trip():
    p = tmp()
    v = C.defaults(); v["VAD_RMS"] = 900; v["SILENCE_STOP_MS"] = 900
    C.save(p, v)
    back = C.load(p)
    assert back["VAD_RMS"] == 900, back
    assert back["SILENCE_STOP_MS"] == 900, back
    assert back["HARD_CAP_S"] == C.defaults()["HARD_CAP_S"], "untouched keys keep defaults"

def t_missing_file_is_defaults():
    assert C.load(tmp()) == C.defaults(), "a missing file must yield defaults"

def t_corrupt_file_is_defaults():
    p = tmp()
    open(p, "w").write("{ this is not json")
    assert C.load(p) == C.defaults(), "a corrupt file must yield defaults, not raise"

def t_non_object_file_is_defaults():
    p = tmp()
    open(p, "w").write("[1, 2, 3]")
    assert C.load(p) == C.defaults(), "a JSON array must be ignored"

def t_load_ignores_unknown_and_invalid():
    p = tmp()
    open(p, "w").write(json.dumps({"NOT_A_SETTING": 1, "VAD_RMS": 99999, "SILENCE_STOP_MS": 800}))
    v = C.load(p)
    assert v["SILENCE_STOP_MS"] == 800, "the valid key still applies"
    assert v["VAD_RMS"] == C.defaults()["VAD_RMS"], "the out-of-range key falls back to default"

# --- tools merge ---
BUILTIN = [
    {"type": "function", "function": {"name": "a", "description": "orig a", "parameters": {}}},
    {"type": "function", "function": {"name": "b", "description": "orig b", "parameters": {}}},
]

def t_merge_no_overrides_is_identity():
    out = C.merge_tools(BUILTIN, {})
    assert [t["function"]["description"] for t in out] == ["orig a", "orig b"], out

def t_merge_applies_description():
    out = C.merge_tools(BUILTIN, {"a": {"description": "new a"}})
    assert out[0]["function"]["description"] == "new a", out
    assert BUILTIN[0]["function"]["description"] == "orig a", "builtin must not be mutated"

def t_merge_drops_disabled():
    out = C.merge_tools(BUILTIN, {"b": {"enabled": False}})
    assert [t["function"]["name"] for t in out] == ["a"], out

def t_merge_keeps_new_builtin_tools():
    # a tool added in code but absent from tools.json must still appear
    out = C.merge_tools(BUILTIN, {"a": {"description": "new a"}})
    assert [t["function"]["name"] for t in out] == ["a", "b"], out

def t_merge_ignores_unknown_override():
    # a tool removed from the code must not resurrect itself from tools.json
    out = C.merge_tools(BUILTIN, {"ghost": {"description": "boo"}})
    assert [t["function"]["name"] for t in out] == ["a", "b"], out

def t_tools_round_trip():
    p = tmp()
    C.save_tools(p, {"a": {"description": "new a"}})
    assert C.load_tools(p) == {"a": {"description": "new a"}}, C.load_tools(p)

def t_tools_corrupt_is_empty():
    p = tmp()
    open(p, "w").write("nope")
    assert C.load_tools(p) == {}, "a corrupt tools file must yield no overrides"

def test_all():
    """pytest entry: run every t_* check and fail on the first broken assertion."""
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        fn()


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        check(name[2:], fn)
    print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
