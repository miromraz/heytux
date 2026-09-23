#!/usr/bin/env python3
# bench_stt.py - STT model benchmark for heytux, scored on END-TO-END COMMAND SUCCESS, not WER.
# See docs/heytux-stt-benchmark task brief. Standalone tool: does not import or modify heytux
# internals beyond reusing its capture-path constants (SR/FRAME/VAD_*/pw_record_cmd), so the
# recorded audio is provably the same as what the live daemon would have heard.
#
# Phase 1 --capture (needs the owner in the room): records each phrase in PHRASES with the
# daemon's own energy-VAD capture path, writes a wav + a JSON manifest under bench/ holding what
# was said and which tool it should trigger.
#
# Phase 2 --score (no owner needed): for each staged model, load it, transcribe every wav, POST
# the transcript to the live daemon's /text (the real Needle3 router - this also EXECUTES the
# tool, see README), compare the fired tool against the manifest's expected tool. --wavdir scores
# an arbitrary wav directory with no manifest instead: transcribe + latency + RAM only, no
# success rate is invented for wavs with no expected tool.
#
# stdlib + numpy + sherpa_onnx only.
import argparse, glob, gc, json, os, re, statistics, subprocess, sys, time, urllib.request, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heytux as J   # reuse only: SR/FRAME/THREADS/pw_record_cmd() constants + the VAD values in
                      # J.CFG (VAD_RMS/VAD_ONSET_FRAMES/SILENCE_STOP_MS/HARD_CAP_S/MIN_S) - the
                      # daemon's own capture path/config, so recordings can't drift from what it hears.

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(HERE, "bench")
MANIFEST_PATH = os.path.join(BENCH_DIR, "manifest.json")
DAEMON_URL = "http://127.0.0.1:8770"
SHERPA_ROOT = "/home/miro/sherpa"

# ---------- Phase 1 phrase list: the ONLY copy (manifest entries are generated from this, never
# hand-duplicated). >=3 reps of every command that matters, incl. the two that failed in the
# brief (flashlight on/off), plus 3 chatter phrases that must route to chat (expected tool=None).
PHRASES = [
    ("turn on the flashlight", "flashlight_on", {}),
    ("turn on the flashlight", "flashlight_on", {}),
    ("turn on the flashlight", "flashlight_on", {}),
    ("turn off the flashlight", "flashlight_off", {}),
    ("turn off the flashlight", "flashlight_off", {}),
    ("turn off the flashlight", "flashlight_off", {}),
    ("what time is it", "what_time_is_it", {}),
    ("what time is it", "what_time_is_it", {}),
    ("what time is it", "what_time_is_it", {}),
    ("what's the battery level", "battery_status", {}),
    ("what's the battery level", "battery_status", {}),
    ("what's the battery level", "battery_status", {}),
    ("turn on wifi", "wifi_on", {}),
    ("turn on wifi", "wifi_on", {}),
    ("turn on wifi", "wifi_on", {}),
    ("turn off wifi", "wifi_off", {}),
    ("turn off wifi", "wifi_off", {}),
    ("turn off wifi", "wifi_off", {}),
    ("turn on bluetooth", "bluetooth_on", {}),
    ("turn on bluetooth", "bluetooth_on", {}),
    ("turn on bluetooth", "bluetooth_on", {}),
    ("turn off bluetooth", "bluetooth_off", {}),
    ("turn off bluetooth", "bluetooth_off", {}),
    ("turn off bluetooth", "bluetooth_off", {}),
    ("lock the screen", "lock_screen", {}),
    ("lock the screen", "lock_screen", {}),
    ("lock the screen", "lock_screen", {}),
    ("take a selfie", "take_selfie", {}),
    ("take a selfie", "take_selfie", {}),
    ("take a selfie", "take_selfie", {}),
    ("next track", "next_track", {}),
    ("next track", "next_track", {}),
    ("next track", "next_track", {}),
    ("set a timer for five minutes", "set_timer", {"minutes": 5}),
    ("set a timer for five minutes", "set_timer", {"minutes": 5}),
    ("set a timer for five minutes", "set_timer", {"minutes": 5}),
    ("tell me a joke", None, {}),
    ("how are you doing today", None, {}),
    ("what do you think about the weather", None, {}),
]

# ---------- models staged on the phone under SHERPA_ROOT; loader kind verified against the
# installed sherpa_onnx API (OfflineRecognizer.from_moonshine_v2/from_nemo_ctc,
# OnlineRecognizer.from_nemo_ctc/from_transducer - see the brief for which is offline vs streaming).
MODELS = {
    "moonshine-tiny":  {"kind": "moonshine",   "dir": "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27"},
    "moonshine-base":  {"kind": "moonshine",   "dir": "sherpa-onnx-moonshine-base-en-quantized-2026-02-27"},
    "parakeet-110m":   {"kind": "nemo_offline", "dir": "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8"},
    "fastconformer-streaming": {"kind": "nemo_online", "dir": "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms-int8"},
    "zipformer-20m-streaming": {"kind": "transducer_online", "dir": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile"},
    "zipformer-kroko-streaming": {"kind": "transducer_online", "dir": "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"},
}

def log(*a):
    print("[bench]", *a, file=sys.stderr, flush=True)

# ---------- manifest ----------
def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def load_manifest(path=MANIFEST_PATH):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def save_manifest(entries, path=MANIFEST_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)

# ---------- wav io ----------
def save_wav(path, pcm_i16_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(J.SR)
        w.writeframes(pcm_i16_bytes)

def read_wav_f32(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != J.SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("%s: expected %d Hz mono s16, got %d Hz %d ch %d byte" % (
                path, J.SR, w.getframerate(), w.getnchannels(), w.getsampwidth()))
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

# ---------- Phase 1: capture (same VAD as heytux.record(), reused not reimplemented-differently) ----------
def daemon_wake(on):
    try:
        req = urllib.request.Request(DAEMON_URL + "/wake",
            data=json.dumps({"on": on}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        log("daemon wake -> %s" % on)
    except Exception as e:
        log("daemon wake toggle failed (continuing anyway): %r" % e)

def capture_utterance():
    """One utterance via pw-record + energy VAD, using heytux.py's own SR/FRAME/pw_record_cmd()
    and the VAD values in J.CFG - the identical constants and command line the daemon's record()
    uses, so this can't silently drift from what the daemon actually hears. Returns raw s16 PCM
    bytes, or None if no speech (mirrors heytux.record()'s contract)."""
    p = subprocess.Popen(J.pw_record_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames, speech, silence_ms, voiced_run, start = [], False, 0, 0, time.time()
    nbytes = J.FRAME * 2
    try:
        while time.time() - start < J.CFG["HARD_CAP_S"]:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frames.append(buf)
            x = np.frombuffer(buf, dtype="<i2").astype(np.float32)
            rms = (float(np.mean(x * x)) + 1e-9) ** 0.5
            if rms > J.CFG["VAD_RMS"]:
                voiced_run += 1
                if voiced_run >= J.CFG["VAD_ONSET_FRAMES"]:
                    speech, silence_ms = True, 0
            else:
                voiced_run = 0
                if speech:
                    silence_ms += 1000 * J.FRAME / J.SR
                    if silence_ms >= J.CFG["SILENCE_STOP_MS"] and (time.time() - start) >= J.CFG["MIN_S"]:
                        break
    finally:
        p.terminate()
        try:
            p.wait(timeout=1)
        except Exception:
            p.kill()
    if not speech:
        return None
    return b"".join(frames)

def cmd_capture(args):
    if args.dry_run:
        print("Phrase list (%d utterances, manifest -> %s):" % (len(PHRASES), MANIFEST_PATH))
        for i, (text, tool, targs) in enumerate(PHRASES):
            print("  %2d. %-40s -> %s %s" % (i, text, tool or "(chat)", targs or ""))
        print("\nManifest entry format (one per recorded utterance):")
        print(json.dumps({"index": 0, "slug": "00-turn-on-the-flashlight",
                           "wav": os.path.join(BENCH_DIR, "00-turn-on-the-flashlight.wav"),
                           "expected_text": "turn on the flashlight",
                           "expected_tool": "flashlight_on", "expected_args": {}}, indent=2))
        return
    os.makedirs(BENCH_DIR, exist_ok=True)
    manifest = load_manifest()
    done = {e["slug"] for e in manifest}
    daemon_wake(False)
    try:
        for i, (text, tool, targs) in enumerate(PHRASES):
            slug = "%02d-%s" % (i, slugify(text))
            if slug in done:
                continue
            while True:
                input("[%d/%d] say: %r  (Enter to record)" % (i + 1, len(PHRASES), text))
                pcm = capture_utterance()
                if pcm is None:
                    print("  no speech detected - try again")
                    continue
                wav_path = os.path.join(BENCH_DIR, slug + ".wav")
                save_wav(wav_path, pcm)
                print("  captured %.1fs -> %s" % (len(pcm) / 2 / J.SR, wav_path))
                ans = input("  keep? [Y/n=redo] ").strip().lower()
                if ans in ("", "y"):
                    break
            manifest.append({"index": i, "slug": slug, "wav": wav_path,
                              "expected_text": text, "expected_tool": tool, "expected_args": targs})
            save_manifest(manifest)
    finally:
        daemon_wake(True)
    print("capture complete: %d/%d utterances -> %s" % (len(manifest), len(PHRASES), MANIFEST_PATH))

# ---------- Phase 2: score ----------
def _pick(d, stem):
    """Prefer an int8-quantized export of `stem`, fall back to fp32. Different zipformer exports
    ship different subsets (int8 encoder+joiner but fp32 decoder is normal), so this is per-file,
    not per-directory."""
    cands = sorted(glob.glob(os.path.join(d, stem + "*int8*.onnx"))) or \
            sorted(glob.glob(os.path.join(d, stem + "*.onnx")))
    if not cands:
        raise FileNotFoundError("%s: no %s*.onnx" % (d, stem))
    return cands[0]

class OfflineWrapper:
    def __init__(self, rec):
        self.rec = rec
    def transcribe(self, pcm_f32):
        s = self.rec.create_stream()
        s.accept_waveform(J.SR, pcm_f32)
        self.rec.decode_stream(s)
        return s.result.text.strip()

class OnlineWrapper:
    def __init__(self, rec):
        self.rec = rec
    def transcribe(self, pcm_f32):
        s = self.rec.create_stream()
        s.accept_waveform(J.SR, pcm_f32)
        s.accept_waveform(J.SR, np.zeros(int(0.5 * J.SR), dtype=np.float32))  # flush the tail
        s.input_finished()
        while self.rec.is_ready(s):
            self.rec.decode_stream(s)
        r = self.rec.get_result(s)
        return (r.text if hasattr(r, "text") else str(r)).strip()

def load_model(name):
    import sherpa_onnx as so
    spec = MODELS[name]
    d = os.path.join(SHERPA_ROOT, spec["dir"])
    kind = spec["kind"]
    if kind == "moonshine":
        rec = so.OfflineRecognizer.from_moonshine_v2(
            encoder=d + "/encoder_model.ort", decoder=d + "/decoder_model_merged.ort",
            tokens=d + "/tokens.txt", num_threads=J.THREADS)
        return OfflineWrapper(rec)
    if kind == "nemo_offline":
        rec = so.OfflineRecognizer.from_nemo_ctc(
            model=d + "/model.int8.onnx", tokens=d + "/tokens.txt", num_threads=J.THREADS)
        return OfflineWrapper(rec)
    if kind == "nemo_online":
        rec = so.OnlineRecognizer.from_nemo_ctc(
            tokens=d + "/tokens.txt", model=d + "/model.int8.onnx", num_threads=J.THREADS)
        return OnlineWrapper(rec)
    if kind == "transducer_online":
        rec = so.OnlineRecognizer.from_transducer(
            tokens=d + "/tokens.txt", encoder=_pick(d, "encoder"), decoder=_pick(d, "decoder"),
            joiner=_pick(d, "joiner"), num_threads=J.THREADS)
        return OnlineWrapper(rec)
    raise ValueError("unknown model kind %r" % kind)

def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0

def call_router(text):
    """POST to the live daemon's /text - the same Needle3 router the daemon uses, and it EXECUTES
    the tool. Only called for manifest entries with an expected_tool (real command phrases)."""
    try:
        req = urllib.request.Request(DAEMON_URL + "/text",
            data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}

def router_matches(r, expected_tool, expected_args):
    if "error" in r:
        return False
    if expected_tool is None:                      # chatter must NOT fire a real tool
        return r.get("route") in ("chat-tool", "chat-fallback")
    calls = r.get("calls") or []
    hit = next((c for c in calls if c.get("name") == expected_tool), None)
    if hit is None:
        return False
    got = hit.get("arguments") or {}
    return all(got.get(k) == v for k, v in (expected_args or {}).items())

def wer(ref, hyp):
    """Word error rate via Levenshtein word-edit distance. Secondary/informational column only -
    the brief explicitly says not to rank models on this."""
    r = re.findall(r"[a-z']+", (ref or "").lower())
    h = re.findall(r"[a-z']+", (hyp or "").lower())
    if not r:
        return None
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)

def load_entries(args):
    if args.wavdir:
        wavs = sorted(glob.glob(os.path.join(args.wavdir, args.glob)))
        labels = {}
        if args.labels and os.path.exists(args.labels):
            for line in open(args.labels):
                parts = line.strip().split("|")
                if len(parts) == 3:
                    base = os.path.splitext(os.path.basename(parts[0]))[0]
                    labels[base] = parts[2]
        entries = []
        for w in wavs:
            base = re.sub(r"_16k$", "", os.path.splitext(os.path.basename(w))[0])
            entries.append({"wav": w, "expected_text": labels.get(base),
                             "expected_tool": None, "expected_args": {}, "score_tool": False})
        return entries
    entries = load_manifest(args.manifest)
    for e in entries:
        e["score_tool"] = True    # manifest entries always carry an intended tool (None = chatter)
    return entries

def score_model(name, entries, args):
    rss0 = rss_kb()
    t0 = time.time()
    try:
        model = load_model(name)
    except Exception as e:
        return {"model": name, "loaded": False, "error": "%s: %s" % (type(e).__name__, e)}
    load_ms = int(1000 * (time.time() - t0))
    rss1 = rss_kb()
    per = []
    for e in entries:
        try:
            pcm = read_wav_f32(e["wav"])
        except Exception as ex:
            per.append({"wav": e["wav"], "transcript": None, "error": str(ex)})
            continue
        t = time.time()
        try:
            tx = model.transcribe(pcm)
        except Exception as ex:
            tx = None
            err = "%s: %s" % (type(ex).__name__, ex)
        else:
            err = None
        lat_ms = int(1000 * (time.time() - t))
        rec = {"wav": e["wav"], "expected_text": e.get("expected_text"),
               "transcript": tx, "latency_ms": lat_ms, "error": err}
        if tx is not None and e.get("expected_text"):
            rec["wer"] = wer(e["expected_text"], tx)
        if e.get("score_tool") and tx is not None:
            r = call_router(tx)
            rec["route"] = r.get("route", r.get("error"))
            rec["fired"] = [c.get("name") for c in (r.get("calls") or [])]
            rec["expected_tool"] = e.get("expected_tool")
            rec["success"] = router_matches(r, e.get("expected_tool"), e.get("expected_args"))
        per.append(rec)
    del model
    gc.collect()
    rss2 = rss_kb()
    lat = sorted(p["latency_ms"] for p in per if p.get("transcript") is not None)
    scored = [p for p in per if "success" in p]
    return {
        "model": name, "loaded": True, "load_ms": load_ms,
        "rss_delta_load_kb": rss1 - rss0, "rss_delta_release_kb": rss1 - rss2,
        "median_ms": statistics.median(lat) if lat else None,
        "p90_ms": lat[int(0.9 * (len(lat) - 1))] if lat else None,
        "success_rate": (sum(1 for p in scored if p["success"]) / len(scored)) if scored else None,
        "per_utterance": per,
    }

def write_report(results, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = ["# heytux STT benchmark - %s" % time.strftime("%Y-%m-%d"), "",
             "Scored on end-to-end command success (transcript -> Needle3 router -> did the "
             "expected tool fire), not WER. WER is a secondary column.", "",
             "| model | loaded | load ms | RSS delta load (kB) | RSS delta release (kB) | "
             "median ms | p90 ms | success rate |",
             "|-|-|-|-|-|-|-|-|"]
    for r in results:
        if not r["loaded"]:
            lines.append("| %s | NO | - | - | - | - | - | - (%s) |" % (r["model"], r["error"]))
            continue
        sr = "n/a" if r["success_rate"] is None else "%.0f%%" % (100 * r["success_rate"])
        lines.append("| %s | yes | %d | %+d | %+d | %s | %s | %s |" % (
            r["model"], r["load_ms"], r["rss_delta_load_kb"], r["rss_delta_release_kb"],
            r["median_ms"], r["p90_ms"], sr))
    lines.append("")
    for r in results:
        if not r["loaded"]:
            continue
        lines.append("## %s" % r["model"])
        for p in r["per_utterance"]:
            if "success" in p:
                mark = "OK" if p["success"] else "FAIL"
                lines.append("- [%s] said=%r heard=%r fired=%r (%d ms)" % (
                    mark, p.get("expected_text"), p["transcript"], p.get("fired"), p["latency_ms"]))
            else:
                w = ("wer=%.2f" % p["wer"]) if p.get("wer") is not None else ""
                lines.append("- heard=%r %s (%d ms)%s" % (
                    p["transcript"], w, p["latency_ms"], (" ERROR: " + p["error"]) if p.get("error") else ""))
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    log("report written to %s" % out_path)

def cmd_score(args):
    entries = load_entries(args)
    if not entries:
        log("no wavs found (check --manifest/--wavdir)"); return
    names = args.models.split(",") if args.models else list(MODELS)
    results = [score_model(n, entries, args) for n in names]
    out_path = args.out or os.path.join(BENCH_DIR, "report-%s.md" % time.strftime("%Y-%m-%d"))
    write_report(results, out_path)
    for r in results:
        if not r["loaded"]:
            print("%-28s FAILED TO LOAD: %s" % (r["model"], r["error"]))
        else:
            sr = "n/a" if r["success_rate"] is None else "%.0f%%" % (100 * r["success_rate"])
            print("%-28s load=%dms median=%sms p90=%sms rss+%dkB success=%s" % (
                r["model"], r["load_ms"], r["median_ms"], r["p90_ms"], r["rss_delta_load_kb"], sr))

# ---------- self-check (ponytail: the smallest thing that fails if the scoring logic breaks) ----------
def _selftest():
    assert wer("turn on the flashlight", "turn on the flashlight") == 0.0
    assert wer("turn off the flashlight", "they're an off flesh light") > 0.5
    r_ok = {"route": "tool", "calls": [{"name": "flashlight_on", "arguments": {}}]}
    assert router_matches(r_ok, "flashlight_on", {}) is True
    assert router_matches(r_ok, "flashlight_off", {}) is False
    r_chat = {"route": "chat-tool", "calls": [{"name": "ask_question", "arguments": {}}]}
    assert router_matches(r_chat, None, {}) is True
    r_wrong_route = {"route": "tool", "calls": [{"name": "bluetooth_on", "arguments": {}}]}
    assert router_matches(r_wrong_route, None, {}) is False
    r_args = {"route": "tool", "calls": [{"name": "set_timer", "arguments": {"minutes": 5}}]}
    assert router_matches(r_args, "set_timer", {"minutes": 5}) is True
    assert router_matches(r_args, "set_timer", {"minutes": 10}) is False
    print("selftest OK")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Phrase list (%d utterances):\n" % len(PHRASES) +
               "\n".join("  %-40s -> %s" % (t, tool or "(chat)") for t, tool, _ in PHRASES) +
               "\n\nManifest entry: {index, slug, wav, expected_text, expected_tool, expected_args}")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Phase 1: record phrases with the owner")
    mode.add_argument("--score", action="store_true", help="Phase 2: score staged models")
    mode.add_argument("--selftest", action="store_true", help="run the offline logic self-check")
    ap.add_argument("--dry-run", action="store_true", help="--capture: print phrases/manifest format only, no recording")
    ap.add_argument("--manifest", default=MANIFEST_PATH, help="--score: manifest.json from --capture")
    ap.add_argument("--wavdir", default=None, help="--score: score an arbitrary wav dir instead (transcribe-only, no success rate)")
    ap.add_argument("--glob", default="*.wav", help="--wavdir: filename glob (default *.wav)")
    ap.add_argument("--labels", default=None, help="--wavdir: optional 'path|POS/NEG|text' label file for WER only")
    ap.add_argument("--models", default=None, help="--score: comma-separated subset of: " + ",".join(MODELS))
    ap.add_argument("--out", default=None, help="--score: markdown report path (default bench/report-<date>.md)")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.capture:
        cmd_capture(args)
    elif args.score:
        cmd_score(args)

if __name__ == "__main__":
    main()
