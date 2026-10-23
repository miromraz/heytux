# heytux — offline voice assistant (Pixel 4a / BengalOS)

Fully offline push-to-talk assistant. Runs as miro's `systemd --user` service.
Pipeline: mic (Silero VAD) → normalize → Moonshine STT → Needle3 tool-calling → execute → Piper TTS ack.

## Deploy

    ./deploy.sh

Pushes `heytux.py`, `heytux.service` and the plasmoid to the phone, fixes ownership,
enables/restarts the service and runs a flashlight smoke test. Override the target with
`HEYTUX_SSH="ssh ... root@<ip>" ./deploy.sh`.

Models and Python libs are already staged on the phone under `/home/miro/sherpa`,
`/home/miro/needle-arm64` and `/home/miro/heytux/pylib` — deploy does not touch them.

## Test (no curl on the phone; use python urllib as miro)

    RU="runuser -u miro -- env XDG_RUNTIME_DIR=/run/user/60466"
    $RU python3 -c 'import urllib.request,json;\
      print(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8770/text",\
      data=json.dumps({"text":"what is the battery at"}).encode(),\
      headers={"Content-Type":"application/json"})).read().decode())'

Endpoints (127.0.0.1:8770, threaded — the plasmoid's `GET /wake`/`/status` poll never blocks
behind a recording; `LOCK` still serializes the models/pipeline):

- `POST /listen` — record from mic, run the full pipeline, speak the ack.
- `POST /text {"text": "...", "dry"?: bool}` — same pipeline, no mic (for testing). `dry:true`
  decides the route and runs the chat LLM if it routes to chat, but executes no tool and speaks
  nothing; `said` is `""` for a dry tool route, the response is otherwise identical.
- `POST /say {"text": "..."}` — TTS only.
- `POST /wake {"on": bool}` / `GET /wake` — toggle/read the wake word.
- `GET /config` / `POST /config {...}` — read/change live settings (see the schema).
- `GET  /status` — phone info dict as JSON.

Response: `{"transcript","calls":[...],"said","confidence","route","ms":{record,stt,needle,act,tts}}`.

Offline unit tests (host-runnable, no models, no phone): `python3 test_config.py` and
`python3 test_heytux.py` (route decision, grounding, registry, cue-skip invariant, dry-run).

### Routing acceptance check

`test_routing.py` drives `/text` with a phrase → expected-tool table and exits non-zero on any
regression. It sends `dry:true` by default, so it **decides the route without executing tools or
speaking** — no dropped ssh link (`turn off wifi`), no `find_my_phone` alarm, no TTS. Set
`HEYTUX_ROUTING_LIVE=1` to run it for real (executes + speaks; mute the sink first).

    $RU python3 /home/miro/heytux/test_routing.py

Measured 2026-09-22 on 0.26.0910.14: **28/29**. The one counted failure, `give me the full phone
status` → `battery_status`, is the Needle3 ceiling; the reply still contains the battery, so it is
left alone. Retuning `phone_info`'s description to win it only pushed the collision onto real
chatter (`what is the capital of France` → `phone_info`), so the original wording stands.

Two cases are deliberately not in the table:

- **No screenshot tool.** phosh owns `org.gnome.Shell.Screenshot` but its `Screenshot()` returns
  `(false, '')` and writes nothing (DRM reports `dpms=On`, so this is not a blanked-panel problem),
  and `grim` is absent from the read-only `/usr`. Nothing to wire up until the image carries one.
- **Date questions route to chat, and that is fine.** `what is today's date` misses the router;
  `_live_facts()` feeds the chat model the real calendar date, so the answer is still correct
  ("Today's date is September 22, 2026, at 08:07."). The test asserts the spoken *answer* carries
  today's month and day rather than the route.

## On-screen overlay

`/listen` and a wake hit both show a single D-Bus notification banner that evolves through the
turn (via `org.freedesktop.Notifications` / `gdbus`, replacing itself in place by id, the same
pattern `mpris_player()` uses for other D-Bus calls): `Listening…` on capture start (before the
wake cue), then the transcript in quotes with `thinking…`, then the transcript with the outcome -
`→ <tool> (<confidence>)` or `→ chat`, then the spoken reply on the next line - clearing itself
after ~6 s. No speech captured shows `Didn't catch that` briefly. This is what makes a
mis-hearing (bad transcript, right-looking route) visually distinct from a mis-routing (good
transcript, wrong tool) at a glance, which matters for the STT benchmark this overlay is the
instrument for.

`/text` (used by `test_routing.py`) never triggers it - only audio-triggered turns pass
`overlay=True` into `pipeline()`. Every notify call is wrapped and non-fatal: the daemon runs
headless in tests with no notification server, and a banner must never be able to break voice
control. The first call blocks briefly (~150 ms, short timeout) since its id is needed to update
the banner later; later updates run off a background thread and never touch pipeline latency.
Setting: `OVERLAY_ON` (default on, live-applied, `POST /config`).

## Plasmoid

`org.bengalos.heytux` (Plasma 6.7) — one round mic button; tap → `POST /listen`, shows
`Listening…` / `Thinking…` then the transcript and what was done. Installs to
`~/.local/share/plasma/plasmoids/`. Plasma is not on the phone yet; validated with `qmllint`.

## Tools (15, one action each, act at confidence ≥ 0.5)

- `flashlight_on`, `flashlight_off`
- `wifi_on`, `wifi_off`
- `bluetooth_on`, `bluetooth_off`
- `set_brightness(percent)`
- `set_volume(action=set|mute|unmute, percent?)`
- `battery_status`
- `phone_info`
- `set_timer(minutes, label?)` — speaks and double-flashes the LED when it fires
- `what_time_is_it` (speaks the time and the date)
- `open_app(app)` — free-text name, mapped to browser, camera, calculator, calendar, clock,
  terminal, contacts, settings, music, messages, phone, photos, weather, app store
- `take_photo`, `take_selfie` — both only *open* the camera app: this image has no headless
  still-capture path (no `gst-launch`, no libcamera CLI or python bindings, raw v4l2 STREAMON
  fails on the CSI nodes). Real capture needs a capture binary in the next image.
- `lock_screen`
- `play_pause`, `next_track` — MPRIS over `busctl` (`playerctl` is not installed)
- `find_my_phone` — loud tone plus torch flashes, restores the previous volume and torch state
- `ask_question(question)` — chatter/questions; handled by the chat LLM (see Chat fallback)

On/off radios are split into zero-argument tools on purpose: Needle3 routes those reliably but
dumped a parameterized `set_radio(radio, state)` into `flashlight_off`. That plus `ask_question`
puts the set at 15 tools (over the ≤12 rule of thumb) but every real command routes correctly,
which matters more.

Off-topic / empty tool call / low confidence → answered by the chat LLM (see Chat fallback).
No speech captured → "I didn't catch that".

**Model limitation:** with a full toolset Needle3 always emits the *closest* tool even for
off-topic input (its `reasoning` says so: "factual question -> no tool ... closest match"), and
confidence does not flag it. The daemon refuses when that reasoning admits no tool matches, which
catches factual questions (e.g. "capital of France"). Queries the model rationalizes into a
command ("sing me a song" → volume) still slip through — no fix without a stronger router.

## Speech capture (VAD + normalization)

Speech detection is the **Silero VAD** (`sherpa_onnx.VoiceActivityDetector`), loaded once inside
`Models` at `num_threads=1` from `/home/miro/sherpa/silero_vad.onnx` (copied out-of-band; not in
git — it is a binary). It decides speech from spectral shape, not loudness, which fixes two
measured failures of the old RMS gate at once: the user's real turns arrive quiet (peaks
2300–8800 of 32768, speech-frame RMS 150–900) and were dropped or mis-transcribed, while the wake
cue's own 990 Hz note leaking into the still-open stream was *accepted* as speech and produced a
"You" hallucination. Onset/end-of-speech bookkeeping (`SILENCE_STOP_MS`, `MIN_S`, `HARD_CAP_S`) is
unchanged; only the per-frame speech decision moved from RMS to Silero. **If the model file is
absent** the daemon logs once and falls back to the original energy VAD (`VAD_RMS`/`VAD_CONT_RMS`/
`VAD_ONSET_FRAMES`), which is why those settings remain.

`VAD_THRESHOLD` (default **0.3**, live) is the Silero speech probability required to start
recording — lower catches fainter speech, higher is stricter. Measured on the real "hey linux"
captures: the new VAD detects 5/5 genuine turns as speech at full, ½× and ¼× amplitude, rejects a
3 s pure-noise stretch, and rejects the 2 cue-leak/silence clips the old energy VAD (7/7,
2 of them bogus) turned into "You".

Before STT the utterance is **normalized** to ~−6 dBFS peak with a +24 dB gain cap (`normalize()`),
so a quiet-but-real turn reaches Moonshine at a consistent level while a noise floor is not blown
up. The saved capture is exactly what the model saw.

**Every** turn is now saved to `captures/` (ring `CAPTURE_KEEP`), including no-speech ones: those
get `route:"none"`, empty transcript, and `peak`/`max_rms`/`vad_speech` in the sidecar json, so a
"didn't catch that" can be diagnosed after the fact (silence vs. quiet miss vs. cue leak).

Mic tuning is live via `POST /config`: `VAD_THRESHOLD` (Silero, default 0.3), `SILENCE_STOP_MS`,
`MIN_S`, `HARD_CAP_S`, `MIC_TARGET` (pins the pw-record `--target`; empty = default source), the
energy-fallback thresholds `VAD_RMS`/`VAD_CONT_RMS`/`VAD_ONSET_FRAMES`, and the rest of the schema
(`GET /config` lists it with defaults).

## Wake word ("hey linux")

Always-on keyword spotting via the sherpa-onnx zipformer KWS model. A background thread owns the
mic while idle; on a hit it plays a rising two-note cue (660 -> 990 Hz, no TTS), runs the exact
`/listen` pipeline, then resumes. A single lower note (520 Hz) plays when the VAD stops the
capture, so you can hear that it stopped listening and is thinking.

**The mic is never closed across the wake -> capture handoff.** `kws_loop` hands its still-open
`pw-record` stream to `record(src, skip_s)` instead of terminating and respawning. That is the
root fix for the first word of every command being eaten: closing the mic left it shut for ~1 s
(terminate + blocking cue + a new `pw-record`) while the user was still talking, and no ring
buffer can recover audio that was never captured. Measured before the fix: `And on the
flashlight.` / `Which light?` / `Of this.` / `Uh, sh light.` — 1 of 6 utterances intact.
Because the cue now plays into a stream we are still reading, `record()` discards the cue's own
length plus `WAKE_SKIP_S` (default 0.20, `POST /config`) up front — by **sample count**, not wall
clock, so blocking cue playback cannot desynchronise it. It is paused while `/listen` records and
while an ack plays (so it never hears itself), and ignores a hit within `WAKE_COOLDOWN_S` of the
previous pipeline finishing.

- Keyword file: `hey.txt` = `▁HE Y ▁LI N U X @HEY_LINUX` (deployed to `/home/miro/heytux/`).
  In sherpa-onnx `:` is the boost and `#` the trigger probability, so the old `:0.25 #1.5` could
  never fire (threshold > 1). Spotter: `max_active_paths=8, keywords_threshold=0.15,
  keywords_score=1.5`. Host bench (same int8 model, 40 libritts_r TTS speakers, 100 ms chunks):

  | config | loud (peak 0.5) | quiet (peak 0.08) | false alarms, 5.6 min near-misses |
  |-|-|-|-|
  | "hey tux", defaults | 27/40 | 23/40 | – |
  | "hey linux", defaults | 36/40 | 28/40 | 1 |
  | "hey linux", tuned | 37/40 | 35/40 | 1 |

  More active paths (12/16/32) gain little and raise false alarms to 3/7/18. Do not add a bare
  single-word keyword (false alarms).
- Control: `POST /wake {"on":bool}`; `GET /wake` → `{"on":bool,"hits":n}`; also the `WAKE_ON`
  setting in `POST /config`. `wake`/`wake_hits` are in `GET /status`. Every hit is logged with a
  timestamp. Idle cost ~28 % of one core.
- Plasmoid shows a "hey tux: on/off" label (polls `/wake` every 10 s, tap toggles).

## Chat fallback

Chatter (questions, facts, jokes, greetings, small talk) is answered by a small local LLM —
`heytux-llm.service`, a llama.cpp `llama-server` serving Qwen3-0.6B Q4_K_M on 127.0.0.1:8771
(`-t 2 -c 2048 --log-disable`, Restart=on-failure; 2048 so the chat memory below fits). `heytux.service` has `Wants=`/`After=` it.
`deploy.sh` installs and enables both units.

Routing (Needle3 + a 15th `ask_question` tool):
- `ask_question` selected → `route=chat-tool`.
- empty `function_calls`, NOMATCH reasoning, or confidence < 0.5 → `route=chat-fallback`.
- a real (non-`ask_question`) tool at confidence ≥ 0.5 → `route=tool` (executes).

The 0.5 threshold (not 0.7) avoids sending a real command that scored low — e.g. "what time is
it" measures conf 0.34–0.73 — to chat. When such a query does fall through to chat, live facts in
the system prompt let the LLM still answer correctly: each request appends
"Current date and time: &lt;%A %H:%M&gt;. Phone battery: &lt;N&gt; percent, &lt;status&gt;." (cheap
sysfs reads only, no nmcli).

Chat always sends the **original transcript** (Needle's filled `question` arg is ignored — it is
unreliable). Request: `POST /v1/chat/completions` with the phone-assistant system prompt (+ live
facts), `max_tokens=60`, `temperature=0.7`, `enable_thinking=false` (thinking MUST stay off or it
takes 10 s+). 10 s timeout; on any error/timeout the daemon speaks "I can't answer that right now."
LLM output is sanitized before TTS (strip emoji/non-ASCII, markdown, collapse whitespace, first 2
sentences, ≤220 chars). The response JSON carries `route` and `ms.llm`.

**Chat memory.** The request also carries the last `CHAT_HISTORY_TURNS` (default 4, 0 = stateless)
question/answer pairs of chat turns, so "tell me a joke" → "explain it" works. Tool turns are not
remembered, a failed answer is not remembered, `dry` /text never changes it, and the whole memory
is dropped once the last chat answer is older than `CHAT_MEMORY_S` (default 60).

**Follow-up window.** After a wake turn that went to chat, `kws_loop` waits for the spoken answer
to finish, drains the still-open mic stream (it heard our own reply), plays the wake cue and records
again on the same stream — no wake word needed — for up to `FOLLOWUP_S` (default 5.0, 0 = off) of
audio waiting for speech to start. It repeats while follow-ups keep going to chat; a tool turn,
silence (ends quietly: no "I didn't catch that", no capture saved) or the wake word switched off
returns to keyword spotting. Follow-up turns go through the same routing gates as any other turn.

Known routing limits (Needle3, deterministic): "who are you" → `phone_info` and "explain how a
battery works" → `battery_status` at confidence 1.0 — they collide with those tools and route
`tool`, not chat. Description tuning (negative clauses, stronger `ask_question`) did not move them;
the real fix is the LoRA fine-tune noted below.

## Off-topic handling

Needle3 always emits the closest tool for a full toolset (see NOMATCH heuristic in `heytux.py`).
The heuristic beat every alternative tried (exclusion clauses, a catch-all tool, a forced 2-way
classifier tool, ≤5-tool groups, embedding similarity); the real fix is a LoRA fine-tune with
`answers: []` examples per the Needle guide. NOMATCH reasoning now routes to the chat LLM rather
than refusing (see Chat fallback).
