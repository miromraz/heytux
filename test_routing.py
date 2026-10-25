#!/usr/bin/env python3
# test_routing.py - routing acceptance check for heytux. Plain asserts, no pytest.
# Drives the LIVE daemon's /text endpoint and checks each phrase reaches the right tool
# (a tool route + expected function) or the chat LLM (an answer route).
# Run ON the phone as miro:  python3 test_routing.py            (defaults to 127.0.0.1:8770)
# Or point elsewhere:        HEYTUX_URL=http://host:8770 python3 test_routing.py
import json, os, sys, urllib.request

BASE = os.environ.get("HEYTUX_URL", "http://127.0.0.1:8770")
# Dry by default: decide the route without executing tools or speaking. Executing here dropped the
# ssh link ("turn off wifi"), sounded find_my_phone's alarm and spoke every answer. Set
# HEYTUX_ROUTING_LIVE=1 to run it for real (executes + speaks).
DRY = os.environ.get("HEYTUX_ROUTING_LIVE") != "1"

# (phrase, expected) - expected is a tool name, or "chat" for anything the chat LLM should field.
CASES = [
    # --- existing radios / toggles ---
    ("turn on the flashlight",            "flashlight_on"),
    ("turn off the flashlight",           "flashlight_off"),
    ("turn on wifi",                      "wifi_on"),
    ("turn off wifi",                     "wifi_off"),
    ("enable bluetooth",                  "bluetooth_on"),
    ("turn off bluetooth",                "bluetooth_off"),
    # --- existing scalar setters ---
    ("set the brightness to 40 percent",  "set_brightness"),
    ("set the volume to 30 percent",      "set_volume"),
    ("mute the volume",                   "set_volume"),
    # --- existing info ---
    ("what is the battery level",         "battery_status"),
    ("give me the full phone status",     "phone_info"),
    ("set a timer for five minutes",      "set_timer"),
    ("what time is it",                   "what_time_is_it"),
    # Date questions miss the router and fall to chat; _live_facts feeds it the real date,
    # so either route is correct. said_has_date() below checks the answer, not the route.
    ("what is today's date",              "chat"),      # hybrid may also call what_time_is_it: see check()
    ("open the calculator",               "open_app"),
    ("open the settings",                 "open_app"),
    ("take a photo",                      "take_photo"),
    # --- new tools ---
    ("take a selfie",                     "take_selfie"),
    ("lock the screen",                   "lock_screen"),
    ("lock the phone",                    "lock_screen"),
    ("pause the music",                   "play_pause"),
    ("play the music",                    "play_pause"),
    ("skip to the next track",            "next_track"),
    ("next song",                         "next_track"),
    ("find my phone",                     "find_my_phone"),
    ("where is my phone",                 "find_my_phone"),
    # --- chatter: must fall to the chat LLM ---
    ("tell me a joke",                    "chat"),
    ("what is the capital of France",     "chat"),
    ("how are you doing today",           "chat"),
    # --- grounding: shares no content word with any tool -> must fall to chat, never fire one.
    # Measured before the grounding fix: "hmm"/"the" -> flashlight_on 0.88-0.89, "okay so anyway"
    # -> flashlight_on 0.85, "When you please," -> next_track 1.00, "Uh, thank you." -> a tool at
    # 0.54, "How are you?" -> phone_info 0.70, "How is the other day?" -> play_pause 0.55. ---
    ("hmm",                               "chat"),
    ("the",                               "chat"),
    ("okay so anyway",                    "chat"),
    ("When you please,",                  "chat"),
    ("Uh, thank you.",                    "chat"),
    ("How are you?",                      "chat"),
    ("How is the other day?",             "chat"),
]

# Documented pre-existing Needle3 collisions (README): informational, never counted.
KNOWN_FAIL = {
    ("who are you", "chat"),
    ("explain how a battery works", "chat"),
    ("good morning", "chat"),          # greetings land on the nearest tool; same ceiling
}

# No screenshot tool: phosh owns org.gnome.Shell.Screenshot but its Screenshot() returns
# false with nothing written, and grim is absent from the read-only /usr. Not wired up.


def route_of(phrase):
    body = json.dumps({"text": phrase, "dry": DRY}).encode()
    req = urllib.request.Request(BASE + "/text", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    calls = r.get("calls") or []
    name = calls[0].get("name") if calls else None
    return r.get("route"), name, r.get("said") or ""


def said_has_date():
    """Whichever way it routes, asking the date must produce today's month and day."""
    import time
    _, name, said = route_of("what is today's date")
    want = (time.strftime("%B").lower(), time.strftime("%-d"))
    # A dry run doesn't execute what_time_is_it (said=""); that handler always speaks the strftime date.
    ok = (DRY and name == "what_time_is_it") or all(w in said.lower() for w in want)
    print("%-4s %-34s want=%-15s got=%r" % ("PASS" if ok else "FAIL", "date is spoken correctly",
                                            " ".join(want), said))
    return ok


def check(phrase, expected):
    route, name, _ = route_of(phrase)
    if phrase == "what is today's date" and name == "what_time_is_it":
        expected = name                   # hybrid: the clock tool also speaks the date, equally right
    if expected == "chat":
        return route in {"chat-tool", "chat-fallback", "llm-answer", "chat-followup"}, "%s/%s" % (route, name)
    return route in {"tool", "tool-fast", "llm-tool"} and name == expected, "%s/%s" % (route, name)


def main():
    passed = 0
    total = 0
    for phrase, expected in CASES:
        total += 1
        ok, got = check(phrase, expected)
        passed += ok
        print("%-4s %-34s want=%-15s got=%s" % ("PASS" if ok else "FAIL", phrase, expected, got))
    total += 1
    passed += said_has_date()
    print("\n--- known pre-existing collisions (not counted) ---")
    for phrase, expected in KNOWN_FAIL:
        ok, got = check(phrase, expected)
        print("%-4s %-34s want=%-15s got=%s" % ("ok" if ok else "xfail", phrase, expected, got))
    print("\nTOTAL %d/%d routed correctly" % (passed, total))
    return 0 if passed == total else 1


# Live test (needs the daemon); runs only as a script so pytest can import it offline.
if __name__ == "__main__":
    sys.exit(main())
