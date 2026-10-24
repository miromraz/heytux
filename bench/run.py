#!/usr/bin/env python3
# run.py - on-phone model benchmark for the heytux offline voice assistant. One JSON record per
# model under <out>/<slot>/<id>.json; report.py renders them. Resumable (skips models already
# measured), per-model timeout, and a model that crashes/OOMs is recorded failed, not fatal.
#
#   run.py --slot asr --models id1,id2|all --scenario desktop --out results/<ts>/ [--quick]
#
# The orchestrator process (run as root over ssh) drops caches and spawns one worker per model as
# miro (`--one <id>`); the worker loads the backend, measures its slot and writes the record. Crash
# isolation is real: a segfaulting ctypes/llama child takes down the worker, not the whole run.
import argparse, gc, glob, json, math, os, re, signal, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backends
from backends import SR, read_wav_f32, write_wav_f32, log

HEYTUX_DIR = os.environ.get("HEYTUX_DIR", "/home/miro/heytux")
sys.path.insert(0, HEYTUX_DIR)   # for heytux.py (TOOLS/normalize/Needle) and test_routing.CASES

_CUR_WORKER = None    # (Popen, pgid) of the in-flight worker, for signal cleanup
_CUR_PIDFILE = None   # pidfile of the in-flight worker
CAPTURES = os.environ.get("HEYTUX_CAPTURES", HEYTUX_DIR + "/captures")
DEFAULT_MANIFEST = os.environ.get("HEYTUX_MANIFEST", "/home/miro/models/manifest.json")
XDG = os.environ.get("XDG_RUNTIME_DIR", "/run/user/60466")
THREADS = backends.DEFAULT_THREADS

# miro worker env for the runuser re-spawn (lost across runuser); matches the brief's invocation.
WORKER_ENV = {
    "XDG_RUNTIME_DIR": XDG,
    "PYTHONPATH": "%s/pylib" % HEYTUX_DIR,
    "LD_LIBRARY_PATH": "/home/miro/sherpa/sherpa-onnx-v1.13.8-linux-aarch64-shared-cpu/lib",
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": "/home/miro",
}

# ---------- heytux reuse (TOOLS, normalize, Needle, test table) ----------
def _heytux():
    if HEYTUX_DIR not in sys.path:
        sys.path.insert(0, HEYTUX_DIR)
    import heytux  # noqa
    return heytux

def tool_specs():
    """The authoritative 15 tools + ask_question, straight from heytux.TOOLS (no rot)."""
    return list(_heytux().TOOLS)

# ---------- fixtures ----------
CHAT_PROMPTS = [
    "What is the capital of Japan?",
    "How many legs does a spider have?",
    "Tell me a short joke.",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "Give me one tip for better sleep.",
    "Jak se rekne dobry den anglicky?",            # Czech question
    "What is 17 plus 25?",
    "Name a planet in our solar system.",
    "__DATE__",                                    # replaced with a live-facts date question
]
CHAT_SYS = "You are a concise phone voice assistant. Answer in one short sentence."

# Off-topic / chatter that must route to no tool (includes the README known misroutes).
OFFTOPIC = [
    "who are you", "explain how a battery works", "sing me a song", "what is the capital of France",
    "hmm", "tell me a joke", "how are you doing today", "what do you think about the weather",
    "good morning", "the", "okay so anyway", "when you please", "uh thank you", "how are you",
    "what is your favorite color", "say something nice", "i am bored", "what is love",
    "count to three", "whatever",
]

# Compact example phrases per tool for the embedding-classifier router.
# ponytail: hand-written seed set, enough to rank slots; grow it if a tool mis-clusters.
EXAMPLES = {
    "flashlight_on": ["turn on the flashlight", "torch on", "light up the flashlight"],
    "flashlight_off": ["turn off the flashlight", "torch off", "kill the light"],
    "wifi_on": ["turn on wifi", "enable wifi", "connect to wifi"],
    "wifi_off": ["turn off wifi", "disable wifi"],
    "bluetooth_on": ["turn on bluetooth", "enable bluetooth"],
    "bluetooth_off": ["turn off bluetooth", "disable bluetooth"],
    "set_brightness": ["set the brightness to 40 percent", "dim the screen", "make the screen brighter"],
    "set_volume": ["set the volume to 30 percent", "mute the volume", "turn it up"],
    "battery_status": ["what is the battery level", "how much battery is left"],
    "phone_info": ["give me the full phone status", "phone info", "how much storage is free"],
    "set_timer": ["set a timer for five minutes", "start a ten minute timer"],
    "what_time_is_it": ["what time is it", "tell me the time"],
    "open_app": ["open the calculator", "launch the camera app", "open settings"],
    "take_photo": ["take a photo", "photograph this"],
    "take_selfie": ["take a selfie"],
    "lock_screen": ["lock the screen", "lock the phone"],
    "play_pause": ["pause the music", "play the music"],
    "next_track": ["skip to the next track", "next song"],
    "find_my_phone": ["find my phone", "where is my phone"],
}

TTS_SENTENCES = [
    "Turn on the flashlight.",
    "The battery is at eighty percent and charging.",
    "Setting a timer for five minutes.",
    "The capital of Japan is Tokyo.",
    "It is three fifteen in the afternoon.",
]
CZ_SENTENCES = ["Dobry den, jak se mas?", "Baterie je na osmdesat procent.", "Kolik je hodin?"]

# ASR command/chatter phrases (expected tool or None=chatter), reused from the routing table shape.
ASR_PHRASES = [
    ("turn on the flashlight", "flashlight_on"), ("turn off wifi", "wifi_off"),
    ("enable bluetooth", "bluetooth_on"), ("set the brightness to forty percent", "set_brightness"),
    ("mute the volume", "set_volume"), ("what is the battery level", "battery_status"),
    ("set a timer for five minutes", "set_timer"), ("what time is it", "what_time_is_it"),
    ("open the calculator", "open_app"), ("take a selfie", "take_selfie"),
    ("lock the screen", "lock_screen"), ("pause the music", "play_pause"),
    ("skip to the next track", "next_track"), ("find my phone", "find_my_phone"),
    ("give me the full phone status", "phone_info"), ("turn on the flashlight please", "flashlight_on"),
    ("what is the capital of France", None), ("tell me a joke", None),
    ("how are you doing today", None), ("sing me a song", None),
    ("open the weather app", "open_app"), ("turn off the flashlight", "flashlight_off"),
    ("set the volume to thirty percent", "set_volume"), ("next song", "next_track"),
    ("where is my phone", "find_my_phone"),
]
# Properly accented Czech so piper synthesizes correct pronunciation and accented transcripts
# are not scored as errors. Scoring stays accent-sensitive (unicode tokenizer, no stripping).
ASR_CZ_PHRASES = ["rozsviť baterku", "jaký je stav baterie", "kolik je hodin",
                  "ztlum hlasitost", "najdi můj telefon", "zamkni obrazovku",
                  "otevři kalkulačku", "kolik je venku stupňů"]

# ---------- system metrics ----------
def meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "MemFree", "SwapFree", "SwapTotal"):
            out[k] = int(v.split()[0])
    return out

def zram_stats():
    out = []
    for dev in sorted(glob.glob("/sys/block/zram*")):
        try:
            algo = open(dev + "/comp_algorithm").read()
            algo = (re.search(r"\[(\w+)\]", algo) or [None, algo.strip()])[1]
            mm = open(dev + "/mm_stat").read().split()
            out.append({"dev": os.path.basename(dev), "algo": algo,
                        "orig_bytes": int(mm[0]), "compr_bytes": int(mm[1])})
        except (OSError, IndexError):
            pass
    return out

def proc_mem(pid):
    out = {"rss_kb": None, "hwm_kb": None, "pss_kb": None}
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith("VmRSS:"): out["rss_kb"] = int(line.split()[1])
            elif line.startswith("VmHWM:"): out["hwm_kb"] = int(line.split()[1])
    except OSError:
        pass
    try:
        for line in open("/proc/%d/smaps_rollup" % pid):
            if line.startswith("Pss:"): out["pss_kb"] = int(line.split()[1]); break
    except OSError:
        pass
    return out

def backend_mem(be):
    return proc_mem(getattr(be, "pid", os.getpid()))

def sample_mem(rec, be):
    """Record the backend's RSS/PSS AND the system meminfo at the same instant - i.e. while the
    model is still loaded. (meminfo must not be sampled later in run_one, after servers are closed.)"""
    rec["mem"] = backend_mem(be)
    rec["meminfo_during"] = meminfo()

# ---------- WER ----------
_NUMWORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}

def _norm_tokens(s):
    """Lowercase word tokens, strip punctuation, map number words to digits ('forty'->'40') so a
    digit ASR ('40 percent') and a spelled-out label ('forty percent') are not scored as errors.
    Unicode-aware (\\w) so Czech accented words stay whole; applied identically to ref and hyp.
    'flashlight' vs 'flash light' deliberately stays an error (word boundaries are not merged)."""
    return [_NUMWORDS.get(w, w) for w in re.findall(r"[\w']+", (s or "").lower(), re.UNICODE)]

def wer(ref, hyp):
    r = _norm_tokens(ref)
    h = _norm_tokens(hyp)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None

# ---------- ASR audio conditions ----------
def noise_pool(cap_dir):
    """Returns (pool_f32, n_capture_files). n==0 means no real room noise was available."""
    frame, quiet, nfiles = 480, [], 0
    for w in sorted(glob.glob(os.path.join(cap_dir, "*.wav"))):
        x = read_wav_f32(w); nf = len(x) // frame
        if nf < 4: continue
        nfiles += 1
        fr = x[:nf * frame].reshape(nf, frame)
        rms = np.sqrt((fr * fr).mean(1) + 1e-12)
        quiet.append(fr[rms <= np.percentile(rms, 30)].reshape(-1))
    return (np.concatenate(quiet) if quiet else np.zeros(SR, np.float32)), nfiles

NOISE_SNR_DB = 10.0      # noise condition mixes room noise at a fixed, realistic SNR

def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))

def conditions(clean, pool, rng, snr_db=NOISE_SNR_DB):
    """clean -> {clean, quiet(-20 dB), noise(clean + room noise scaled to snr_db, then
    heytux.normalize)}. Noise is scaled to the target SNR so the mix is a realistic hard case, not
    an unbounded floor that buries the speech."""
    quiet = (clean * (10 ** (-20 / 20.0))).astype(np.float32)
    n = len(clean)
    seg = (pool[(i := rng.integers(0, max(1, len(pool) - n))):i + n] if len(pool) > n
           else np.resize(pool, n)).astype(np.float32)
    nr = _rms(seg)
    if nr > 0:
        seg = seg * (_rms(clean) / (nr * 10 ** (snr_db / 20.0)))
    noisy = np.asarray(_heytux().normalize((clean + seg).astype(np.float32)), dtype=np.float32)
    return {"clean": clean, "quiet": quiet, "noise": noisy}

# ---------- per-slot measurement (worker side) ----------
def measure_chat(entry, be, args, rec, outdir):
    # The primary server (be) must be the ONLY model resident during its own measurement: do the
    # server prompts first, then close be BEFORE llama-bench / no-warmup / mmap-kv variants, so a
    # big model (e.g. 4B) never has two copies loaded at once (the earlier 4B OOM/ConnectionRefused).
    hx = _heytux()
    prompts = CHAT_PROMPTS if not args.quick else CHAT_PROMPTS[:2] + [CHAT_PROMPTS[-1]]
    answers, ttfts, tps = [], [], []
    for q in prompts:
        if q == "__DATE__":
            sysp = CHAT_SYS + " " + hx._live_facts(); q = "What is today's date?"
        else:
            sysp = CHAT_SYS
        try:
            r = be.chat([{"role": "user", "content": q}], max_tokens=60, system=sysp)
        except Exception as e:
            answers.append({"q": q, "error": repr(e)[:150]}); continue
        answers.append({"q": q, "a": r["text"], "ttft_s": r["ttft_s"], "tok_s": r["tok_s"],
                        "n": r["n_tokens"]})
        if r["ttft_s"]: ttfts.append(r["ttft_s"])
        if r["tok_s"]: tps.append(r["tok_s"])
    _dump(outdir, entry, "chat_answers.json", answers)
    rec["ttft_s_median"] = round(statistics.median(ttfts), 3) if ttfts else None
    rec["tok_s_median"] = round(statistics.median(tps), 2) if tps else None   # server decode t/s
    sample_mem(rec, be)
    rec["resolved_model"] = os.path.basename(getattr(be, "_model_path", "") or entry["path"])
    be.close()   # free the primary before any further load
    # llama-bench pp64/tg64 (own process)
    rec["llama_bench"] = _llama_bench(entry, args)
    # time-to-health with warmup off, to show how much of load is the warmup decode
    sv = None
    try:
        t = time.time(); sv = backends.load(entry, threads=THREADS, warmup=False)
        rec["load_s_no_warmup"] = round(time.time() - t, 3)
    except Exception as e:
        rec["load_s_no_warmup"] = None; rec["load_s_no_warmup_err"] = repr(e)[:150]
    finally:
        if sv is not None: sv.close()
    # mmap/kv RAM variants (full only); a variant failure is recorded, never fails the record
    if not args.quick:
        rec["mem_variants"] = {}
        for label, kw in (("nommap", {"mmap": False}), ("kv_q8_0", {"kv_type": "q8_0"})):
            sv = None
            try:
                sv = backends.load(entry, threads=THREADS, **kw)
                sv.chat([{"role": "user", "content": "hi"}], max_tokens=8)
                rec["mem_variants"][label] = backend_mem(sv)
            except Exception as e:
                rec["mem_variants"][label] = {"error": repr(e)[:200]}
            finally:
                if sv is not None: sv.close()

def _llama_bench(entry, args):
    model = backends.resolve_gguf(entry["path"], entry.get("slot"))   # dir -> gguf, same as the server
    cmd = [backends.LLAMA_BIN + "/llama-bench", "-m", model, "-t", str(THREADS),
           "-p", "64", "-n", "64", "-r", "2" if args.quick else "3", "-o", "json"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=args.timeout, text=True)
        rows = json.loads(p.stdout)
        out = {}
        for row in rows:
            ts = round(float(row["avg_ts"]), 2)
            out["pp64" if row.get("n_prompt") else "tg64"] = ts
        return out
    except Exception as e:
        return {"error": repr(e)[:200], "stderr": (p.stderr[-300:] if "p" in dir() else "")}

def _router_eval(predict):
    """predict(phrase) -> predicted tool name or 'none'. Returns metrics over the test table."""
    hx_cases = __import__("test_routing").CASES
    truth = [(p, (None if e == "chat" else e)) for p, e in hx_cases] + [(p, None) for p in OFFTOPIC]
    correct = tp = pred_none = truth_none = 0
    lats = []
    for phrase, exp in truth:
        t = time.time(); pred = predict(phrase); lats.append(1000 * (time.time() - t))
        is_none_pred = (pred == "none")
        is_none_truth = (exp is None)
        pred_none += is_none_pred; truth_none += is_none_truth
        if is_none_truth:
            tp += (is_none_pred)
            correct += is_none_pred
        else:
            correct += (pred == exp)
    return {"n": len(truth), "accuracy": round(correct / len(truth), 3),
            "none_recall": round(tp / truth_none, 3) if truth_none else None,
            "none_precision": round(tp / pred_none, 3) if pred_none else None,
            "latency_ms_median": round(statistics.median(lats), 1)}

def measure_router(entry, be, args, rec, outdir):
    tools = tool_specs()
    def predict(phrase):
        r = be.route(phrase, tools)
        names = [n for n in r["names"] if n and n != "ask_question"]
        return names[0] if names else "none"
    rec["gguf_router"] = _router_eval(predict)
    sample_mem(rec, be)
    # Needle3 baseline, through heytux's own class if it stands up standalone.
    rec["needle3_baseline"] = _needle_baseline()

def _needle_baseline():
    try:
        hx = _heytux()
        nd = hx.Needle(hx.MERGED_TOOLS)
    except Exception as e:
        return {"skipped": "heytux.Needle not importable standalone: " + repr(e)[:160]}
    def predict(phrase):
        r = nd.complete(phrase)
        names = [c.get("name") for c in (r.get("function_calls") or [])]
        names = [n for n in names if n and n != "ask_question"]
        return names[0] if names else "none"
    return _router_eval(predict)

def measure_embedding(entry, be, args, rec, outdir):
    # latency per query + RAM
    q = ["what time is it", "turn on the flashlight", "tell me a joke"]
    lats = []
    for s in q:
        t = time.time(); be.embed([s]); lats.append(1000 * (time.time() - t))
    rec["embed_latency_ms_median"] = round(statistics.median(lats), 1)
    sample_mem(rec, be)
    # embedding-classifier router
    labels, texts = [], []
    for tool, exs in EXAMPLES.items():
        for e in exs:
            labels.append(tool); texts.append(e)
    E = be.embed(texts)                       # (n, d), normalized
    thr = float(os.environ.get("HEYTUX_EMB_THRESHOLD", "0.5"))
    def predict(phrase):
        v = be.embed([phrase])[0]
        sims = E @ v
        i = int(np.argmax(sims))
        return labels[i] if sims[i] >= thr else "none"
    r = _router_eval(predict); r["threshold"] = thr
    rec["embedding_router"] = r

def measure_asr(entry, be, args, rec, outdir, manifest):
    rng = np.random.default_rng(1234)
    sample_mem(rec, be)   # sample the ASR model alone, before any TTS voice is loaded in-proc
    rec["resolved_files"] = getattr(be, "files", None)   # shows the actual int8/fp32 files loaded
    pool, n_cap = noise_pool(CAPTURES)
    have_noise = n_cap > 0
    voices = [m for m in manifest if m["slot"] == "tts"]
    en_voices = [v for v in voices if "cs_CZ" not in v["id"]][: (1 if args.quick else 3)]
    phrases = ASR_PHRASES[: (5 if args.quick else len(ASR_PHRASES))]
    per = {"clean": [], "quiet": [], "noise": []}
    samples = {"clean": [], "quiet": [], "noise": []}
    rtf = []
    for v in en_voices:
        tts = backends.load(v, threads=THREADS)
        for text, _tool in phrases:
            x, sr = tts.synth(text, sid=0, speed=1.0)
            x = backends.resample_to_16k(x, sr)
            conds = conditions(x, pool, rng)
            if not have_noise:
                conds.pop("noise")            # no real room noise -> don't fake a noise condition
            for cond, sig in conds.items():
                t = time.time(); hyp = be.transcribe(sig)
                dt = time.time() - t
                rtf.append(dt / max(len(sig) / SR, 1e-6))
                per[cond].append(wer(text, hyp))
                if len(samples[cond]) < 3:
                    samples[cond].append({"ref": text, "hyp": hyp, "wer": round(wer(text, hyp), 3)})
        tts.close(); del tts; gc.collect()
    rec["wer"] = {c: _mean(per[c]) for c in per}
    rec["wer_samples"] = samples          # 3 ref/hyp/wer per condition, for auditing the WER
    rec["noise_snr_db"] = NOISE_SNR_DB if have_noise else None
    rec["noise_provenance"] = ("room noise from %d capture wavs @ %g dB SNR" % (n_cap, NOISE_SNR_DB)
                               if have_noise else "no capture wavs: noise condition skipped")
    rec["rtf_median"] = round(statistics.median(rtf), 3) if rtf else None
    rec["mem_note"] = "RSS/PSS sampled before TTS voices were loaded; VmHWM includes synth peak"
    # Czech (multilingual models only)
    langs = entry.get("languages", [])
    cz_voice = next((v for v in voices if "cs_CZ" in v["id"]), None)
    if cz_voice and any(l in ("cs", "multi") for l in langs):
        tts = backends.load(cz_voice, threads=THREADS); cz = []
        for text in (ASR_CZ_PHRASES[:3] if args.quick else ASR_CZ_PHRASES):
            x, sr = tts.synth(text, sid=0, speed=1.0)
            cz.append(wer(text, be.transcribe(backends.resample_to_16k(x, sr))))
        tts.close(); del tts; gc.collect()
        rec["wer_cs"] = _mean(cz)
    # real captures: transcripts side by side, no ground truth
    caps = sorted(glob.glob(CAPTURES + "/*.wav"))[: (2 if args.quick else 8)]
    rec["captures"] = [{"file": os.path.basename(c), "hyp": be.transcribe(read_wav_f32(c))} for c in caps]

def measure_tts(entry, be, args, rec, outdir):
    dst = os.path.join(outdir, "tts", entry["id"]); os.makedirs(dst, exist_ok=True)
    sents = list(TTS_SENTENCES[: (2 if args.quick else 5)])
    if "cs_CZ" in entry["id"] or any(l in ("cs", "multi") for l in entry.get("languages", [])):
        sents += CZ_SENTENCES[: (1 if args.quick else 3)]
    rtf, ttfa = [], []
    for i, s in enumerate(sents):
        t = time.time(); x, sr = be.synth(s, sid=0, speed=1.0); dt = time.time() - t
        write_wav_f32(os.path.join(dst, "%02d.wav" % i), x, sr)
        dur = len(x) / sr
        rtf.append(dt / max(dur, 1e-6)); ttfa.append(dt)   # offline: no streaming, ttfa == synth time
    rec["rtf_median"] = round(statistics.median(rtf), 3)
    rec["ttfa_s_median"] = round(statistics.median(ttfa), 3)
    rec["ttfa_note"] = "offline synth (no streaming); time to first audio == full synth time"
    rec["wavs_dir"] = dst
    sample_mem(rec, be)

def _test_png(path, w=96, h=96):
    """Write a small RGB PNG (color gradient) with stdlib only, as a vision test image."""
    import struct, zlib
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(((x * 5) % 256, (y * 5) % 256, 160))
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw))))
        f.write(chunk(b"IEND", b""))

def measure_vision(entry, be, args, rec, outdir):
    import base64
    img = os.path.join(outdir, "vision", "test.png"); os.makedirs(os.path.dirname(img), exist_ok=True)
    _test_png(img)
    uri = "data:image/png;base64," + base64.b64encode(open(img, "rb").read()).decode()
    content = [{"type": "text", "text": "Describe this image in one short sentence."},
               {"type": "image_url", "image_url": {"url": uri}}]
    r = be.chat([{"role": "user", "content": content}], max_tokens=64, temperature=0.2)
    rec["answer"] = r["text"]
    rec["ttft_s"] = round(r["ttft_s"], 3) if r["ttft_s"] else None
    rec["tok_s"] = r["tok_s"]
    sample_mem(rec, be)

def measure_vad(entry, be, args, rec, outdir, manifest):
    sample_mem(rec, be)   # sample this VAD alone, before the silero baseline is loaded in-proc
    sil = next((m for m in manifest if m["slot"] == "vad" and "silero" in m["id"]), None)
    baseline = be if (sil and sil["id"] == entry["id"]) else (backends.load(sil) if sil else None)
    files = sorted(glob.glob(CAPTURES + "/*.wav"))[: (3 if args.quick else 12)]
    speed, agree = [], []
    for f in files:
        x = read_wav_f32(f); dur = len(x) / SR
        t = time.time(); segs = be.segments(x); dt = time.time() - t
        speed.append(dur / max(dt, 1e-6))
        if baseline is not None:
            bsegs = baseline.segments(x)
            agree.append(_seg_agreement(segs, bsegs, len(x)))
    rec["speed_x_realtime_median"] = round(statistics.median(speed), 1) if speed else None
    rec["silero_agreement"] = (round(statistics.mean(agree), 3) if agree else None)
    rec["is_silero"] = bool(sil and sil["id"] == entry["id"])
    if baseline is not be and baseline is not None:
        baseline.close()

def _seg_agreement(a, b, n):
    """Frame-level IoU of two segment lists over n samples (1.0 == identical speech mask)."""
    ma = np.zeros(n, bool); mb = np.zeros(n, bool)
    for s, e in a: ma[int(s * SR):int(e * SR)] = True
    for s, e in b: mb[int(s * SR):int(e * SR)] = True
    inter = (ma & mb).sum(); union = (ma | mb).sum()
    return 1.0 if union == 0 else inter / union

SLOT_FN = {"chat": measure_chat, "router": measure_router, "embedding": measure_embedding,
           "tts": measure_tts, "vision": measure_vision}
SLOT_FN_MANIFEST = {"asr": measure_asr, "vad": measure_vad}   # also need the manifest

# ---------- worker ----------
def run_one(entry, args, manifest):
    rec = {k: entry.get(k) for k in ("id", "slot", "family", "params", "quant", "runtime")}
    rec.update({"scenario": args.scenario, "quick": args.quick, "threads": THREADS,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "ok",
                "meminfo_before": meminfo(), "zram": zram_stats()})
    outdir = args.out
    if entry["slot"] == "wake":
        rec["status"] = "skipped"; rec["note"] = "wake slot skipped for now (brief)"
        return rec
    if os.environ.get("HEYTUX_TEST_HANG"):   # test-only: spawn a real server, then hang, to prove kill
        try: backends.load(entry, threads=THREADS)
        except Exception: pass
        time.sleep(99999)
    be = None
    try:
        t = time.time(); be = backends.load(entry, threads=THREADS)
        rec["cold_load_s"] = round(time.time() - t, 3)
        if args.quick:
            rec["warm_load_s"] = rec["cold_load_s"]
        else:
            # drop the first instance fully before loading again, so warm load never holds two models.
            be.close(); be = None; gc.collect()
            t = time.time(); be = backends.load(entry, threads=THREADS)
            rec["warm_load_s"] = round(time.time() - t, 3)
        if entry["slot"] in SLOT_FN_MANIFEST:
            SLOT_FN_MANIFEST[entry["slot"]](entry, be, args, rec, outdir, manifest)
        else:
            SLOT_FN[entry["slot"]](entry, be, args, rec, outdir)
        # meminfo_during is sampled by sample_mem() inside each slot fn, while the model is loaded.
    except NotImplementedError as e:
        rec["status"] = "skipped"; rec["note"] = str(e)
    except Exception as e:
        import traceback
        rec["status"] = "failed"; rec["error"] = repr(e)[:300]
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        # unconditional: close the primary backend plus any other server still registered.
        try:
            if be is not None: be.close()
        except Exception: pass
        backends._close_all()
    return rec

def _dump(outdir, entry, name, obj):
    """Write a per-model artifact (e.g. saved chat answers for grading) atomically, so a crash never
    leaves a truncated json that would break REPORT generation."""
    d = os.path.join(outdir, entry["slot"]); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, entry["id"] + "_" + name); tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def record_path(out, entry):
    return os.path.join(out, entry["slot"], entry["id"] + ".json")

def pidfile_path(out, slot, mid):
    return os.path.join(out, slot, ".pids-" + re.sub(r"[^\w.-]", "_", mid))

def sweep_pidfile(path):
    """Kill any llama-server the worker spawned but didn't reap (hard kill / timeout / segfault),
    by process group. Backstop to the worker's own atexit teardown."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        try:
            if ("llama-server" in open("/proc/%d/cmdline" % pid).read()):
                try: os.killpg(os.getpgid(pid), 9)
                except Exception: os.kill(pid, 9)
        except (OSError, ProcessLookupError):
            pass
    try: os.remove(path)
    except OSError: pass

def write_record(out, entry, rec):
    p = record_path(out, entry); os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)   # atomic: a crashed/killed write never leaves a half record that looks done
    return p

def record_done(p):
    """A record counts as done only if it exists AND parses (a truncated json.dump is not done)."""
    if not os.path.exists(p):
        return False
    try:
        json.load(open(p)); return True
    except Exception:
        return False

# ---------- orchestrator ----------
def select(manifest, slot, models):
    pool = [m for m in manifest if m["slot"] == slot]
    if models == "all":
        return pool
    want = [s.strip() for s in models.split(",") if s.strip()]
    byid = {m["id"]: m for m in manifest}
    out = []
    for w in want:
        if w not in byid:
            raise SystemExit("unknown model id: %s" % w)
        out.append(byid[w])
    return out

def _killpg(p, pgid, grace=3.0):
    """SIGTERM the worker's whole process group, then SIGKILL after `grace` s, on the pgid captured
    at spawn. Never gated on the leader's exit: a dead leader with a live descendant still gets the
    group SIGKILL. subprocess.run(timeout=) only ever SIGKILLs runuser, leaving all of these alive."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if pgid is not None: os.killpg(pgid, sig)
            else: p.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass
        try: p.wait(timeout=grace if sig == signal.SIGTERM else 5)
        except Exception: pass

def spawn_worker(entry, args):
    """Run this file in --one mode, as miro when we are root (proper XDG/libs), in its OWN process
    group with a timeout. On timeout or exit the whole group is killed. Returns (rc, stderr_tail)."""
    base = [sys.executable, os.path.abspath(__file__), "--one", entry["id"], "--slot", args.slot,
            "--scenario", args.scenario, "--out", args.out, "--manifest", args.manifest]
    if args.quick: base.append("--quick")
    base += ["--per-timeout", str(args.timeout)]
    if os.geteuid() == 0:
        wenv = dict(WORKER_ENV)
        for k in ("HEYTUX_TEST_HANG",):   # forward test flags across the runuser env reset
            if os.environ.get(k): wenv[k] = os.environ[k]
        env = ["env"] + ["%s=%s" % (k, v) for k, v in wenv.items()]
        cmd = ["runuser", "-u", "miro", "--"] + env + base
    else:
        cmd = base
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, start_new_session=True)
    try: pgid = os.getpgid(p.pid)      # captured at spawn; used even if the leader exits first
    except Exception: pgid = p.pid
    global _CUR_WORKER
    _CUR_WORKER = (p, pgid)
    rc, err = None, b""
    try:
        err = p.communicate(timeout=args.timeout + 20)[1] or b""
        rc = p.returncode
    except subprocess.TimeoutExpired:
        _killpg(p, pgid)
        try: err = p.communicate(timeout=10)[1] or b""
        except Exception: pass
        rc = "timeout"
    finally:
        _killpg(p, pgid)   # second layer: ensure no straggler survives in the worker's group
        _CUR_WORKER = None
    return rc, err.decode("utf-8", "replace")[-800:]

def _orch_cleanup():
    """Kill the in-flight worker group and sweep its pidfile. Safe to call repeatedly."""
    w = _CUR_WORKER
    if w is not None:
        _killpg(w[0], w[1])
    if _CUR_PIDFILE:
        sweep_pidfile(_CUR_PIDFILE)

def _sig_exit(signum, frame):
    log("signal %d: cleaning up in-flight worker and detached servers" % signum)
    _orch_cleanup()
    sys.exit(128 + signum)

def orchestrate(args):
    global _CUR_PIDFILE
    manifest = json.load(open(args.manifest))
    entries = select(manifest, args.slot, args.models)
    slotdir = os.path.join(args.out, args.slot)
    os.makedirs(slotdir, exist_ok=True)
    if os.geteuid() == 0:
        # worker runs as miro; let it write records/artifacts under this root-created tree.
        for d in (args.out, slotdir):
            try: os.chmod(d, 0o777)
            except OSError: pass
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _sig_exit)
    try:
        for entry in entries:
            p = record_path(args.out, entry)
            if record_done(p) and not args.force:
                log("skip (done): %s" % entry["id"]); continue
            if os.path.exists(p):
                os.remove(p)   # re-measure or stale/truncated: clear so a crash is recorded fresh
            if not args.quick and os.geteuid() == 0:
                try: open("/proc/sys/vm/drop_caches", "w").write("3\n")
                except OSError as e: log("drop_caches failed: %r" % e)
            log("=== %s / %s ===" % (args.slot, entry["id"]))
            _CUR_PIDFILE = pidfile_path(args.out, args.slot, entry["id"])
            try:
                rc, err = spawn_worker(entry, args)
            finally:
                sweep_pidfile(_CUR_PIDFILE); _CUR_PIDFILE = None   # always kill any leaked server
            if not record_done(p):
                # worker crashed/OOM/timeout before a complete record: mark failed, carry on.
                rec = {k: entry.get(k) for k in ("id", "slot", "family", "params", "quant", "runtime")}
                rec.update({"scenario": args.scenario, "status": "failed",
                            "error": ("timeout" if rc == "timeout" else "worker exit rc=%s" % rc),
                            "stderr": err, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                write_record(args.out, entry, rec)
                log("FAILED %s: %s" % (entry["id"], rec["error"]))
            else:
                log("done %s (rc=%s)" % (entry["id"], rc))
    finally:
        _orch_cleanup()
    log("results in %s ; render with report.py %s" % (args.out, args.out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True,
                    choices=["chat", "router", "asr", "tts", "embedding", "vad", "wake", "vision"])
    ap.add_argument("--models", default="all", help="comma-separated ids, or 'all'")
    ap.add_argument("--scenario", default="desktop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="per-model timeout (s)")
    ap.add_argument("--force", action="store_true", help="re-measure even if a record exists")
    ap.add_argument("--one", help="(worker) measure this single id and write its record")
    ap.add_argument("--per-timeout", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.per_timeout:
        args.timeout = args.per_timeout
    if args.one:
        manifest = json.load(open(args.manifest))
        entry = next((m for m in manifest if m["id"] == args.one and m["slot"] == args.slot), None)
        if entry is None:
            raise SystemExit("worker: id %s not in slot %s" % (args.one, args.slot))
        os.makedirs(os.path.join(args.out, args.slot), exist_ok=True)
        os.environ["HEYTUX_PIDFILE"] = pidfile_path(args.out, args.slot, args.one)
        rec = run_one(entry, args, manifest)
        write_record(args.out, entry, rec)
        log("worker wrote %s (status=%s)" % (args.one, rec["status"]))
    else:
        orchestrate(args)

if __name__ == "__main__":
    main()
