#!/usr/bin/env python3
# test_models.py - offline pytest checks for the model registry/migration and the embedding
# router's argument extraction. No phone, no models, no network. Run: pytest -q  (or python3 -m pytest)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heytux_backends as HB
import heytux as J


# ---------- embedding-router argument extraction (heytux_backends.extract_args) ----------
def test_extract_brightness_digits():
    assert HB.extract_args("set_brightness", "set the brightness to 40 percent") == {"percent": 40}

def test_extract_brightness_words():
    assert HB.extract_args("set_brightness", "set brightness to forty five percent") == {"percent": 45}

def test_extract_brightness_one_hundred():
    assert HB.extract_args("set_brightness", "brightness one hundred percent") == {"percent": 100}

def test_extract_brightness_without_number():
    assert HB.extract_args("set_brightness", "dim the screen") == {}

def test_extract_volume_set_words():
    assert HB.extract_args("set_volume", "set the volume to thirty percent") == {"action": "set", "percent": 30}

def test_extract_volume_mute_not_unmute():
    assert HB.extract_args("set_volume", "mute the volume") == {"action": "mute"}

def test_extract_volume_unmute():
    assert HB.extract_args("set_volume", "unmute the volume") == {"action": "unmute"}

def test_extract_volume_turn_it_up_no_number():
    assert HB.extract_args("set_volume", "turn it up") == {"action": "set"}

# --- set_timer: spoken numbers, units, label, and ambiguity (item C) ---
def test_extract_timer_word_number():
    assert HB.extract_args("set_timer", "set a timer for five minutes") == {"minutes": 5}

def test_extract_timer_hours():
    assert HB.extract_args("set_timer", "timer for 1 hour") == {"minutes": 60}

def test_extract_timer_fractional_minutes():
    assert HB.extract_args("set_timer", "set a timer for 1.5 minutes") == {"seconds": 90}

def test_extract_timer_seconds():
    assert HB.extract_args("set_timer", "timer for 90 seconds") == {"seconds": 90}
    assert HB.extract_args("set_timer", "Set a timer for 5 seconds.") == {"seconds": 5}
    assert HB.extract_args("set_timer", "set a timer for two minutes") == {"minutes": 2}

def test_extract_timer_half_an_hour():
    assert HB.extract_args("set_timer", "set a timer for half an hour") == {"minutes": 30}

def test_extract_timer_an_hour():
    assert HB.extract_args("set_timer", "set a timer for an hour") == {"minutes": 60}

def test_extract_timer_compound_hour_and_minutes():
    assert HB.extract_args("set_timer", "set a timer for one hour and thirty minutes") == {"minutes": 90}

def test_extract_timer_an_hour_and_a_half():
    assert HB.extract_args("set_timer", "set a timer for an hour and a half") == {"minutes": 90}

def test_extract_timer_one_and_a_half_hours():
    assert HB.extract_args("set_timer", "set a timer for one and a half hours") == {"minutes": 90}

def test_extract_timer_unparseable_is_empty():
    assert HB.extract_args("set_timer", "set a timer for a while") == {}

def test_extract_timer_keeps_label():
    assert HB.extract_args("set_timer", "set a timer for 5 minutes for pasta") == {"minutes": 5, "label": "pasta"}

def test_extract_timer_ambiguous_is_empty():
    assert HB.extract_args("set_timer", "set a timer") == {}    # no value -> handler asks

def test_extract_open_app_strips_filler():
    assert HB.extract_args("open_app", "open the calculator") == {"app": "calculator"}

def test_extract_zero_arg_tool_is_empty():
    assert HB.extract_args("flashlight_on", "turn on the flashlight") == {}


# ---------- config migration (heytux.migrate_config): old STT_DIR/TTS_DIR path -> model id ----------
def test_migrate_maps_paths_to_ids():
    J.REGISTRY = {
        "moonshine-base-en-quantized": {"id": "moonshine-base-en-quantized", "slot": "asr", "path": "/m/moon"},
        "lessac-low": {"id": "lessac-low", "slot": "tts", "path": "/m/voice/"},
    }
    out = J.migrate_config({"STT_DIR": "/m/moon", "TTS_DIR": "/m/voice", "VAD_RMS": 700})
    assert out["ASR_MODEL"] == "moonshine-base-en-quantized"
    assert out["TTS_MODEL"] == "lessac-low"            # trailing-slash difference still matches
    assert "STT_DIR" not in out and "TTS_DIR" not in out
    assert out["VAD_RMS"] == 700                        # unrelated keys pass through untouched

def test_migrate_unknown_path_drops_to_default():
    J.REGISTRY = {}
    out = J.migrate_config({"STT_DIR": "/not/installed"})
    assert "STT_DIR" not in out and "ASR_MODEL" not in out   # field then falls back to its default

def test_migrate_keeps_an_explicit_new_field():
    J.REGISTRY = {"x": {"id": "x", "slot": "asr", "path": "/m/old"}}
    out = J.migrate_config({"STT_DIR": "/m/old", "ASR_MODEL": "chosen"})
    assert out["ASR_MODEL"] == "chosen"                # an already-set ASR_MODEL wins over migration
