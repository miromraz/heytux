#!/usr/bin/env python3
# backends.py - clean per-runtime model adapters for the heytux model benchmark, and (later) for
# the live daemon's model switching. A manifest entry -> load(entry) -> a backend object exposing
# only the method(s) its slot needs (.chat/.route/.embed/.transcribe/.synth/.segments) and .close().
#
# No benchmark code lives here (no timing, no scoring, no report): just the adapters. Stdlib +
# numpy + sherpa_onnx, and the self-contained llama.cpp server/bench binaries on the phone.
import atexit, gc, glob, json, math, os, re, signal, socket, subprocess, sys, time, urllib.request, wave
import numpy as np

_LIVE = set()   # every running LlamaServer; closed unconditionally at interpreter exit (atexit)

@atexit.register
def _close_all():
    for s in list(_LIVE):
        try: s.close()
        except Exception: pass

SR = 16000
LLAMA_BIN = os.environ.get("HEYTUX_LLAMA_BIN", "/home/miro/llm/current/bin")
DEFAULT_THREADS = 2          # only 2 big A76 cores on the 4a; -t 4 measured slower (brief)

def log(*a): print("[backends]", *a, file=sys.stderr, flush=True)

# ---------- wav io (shared; same math as heytux / bench_stt_host) ----------
def read_wav_f32(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        pcm = w.readframes(w.getnframes())
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    if sr != SR:
        x = resample_to_16k(x, sr)
    return x

def write_wav_f32(path, x, sr=SR):
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())

def resample_to_16k(x, sr):
    # ponytail: naive linear interp, fine for ASR/TTS eval; swap for polyphase if artefacts show.
    if sr == SR:
        return x.astype(np.float32)
    n = int(round(len(x) * SR / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

# ---------- llama.cpp server (chat / router / vision / embedding) ----------
def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

class LlamaServer:
    """One llama-server process for a GGUF model. chat()/route() for generative slots, embed() for
    an --embeddings server. .pid is the server pid (sample its /proc for RAM). The daemon can reuse
    this verbatim to host whichever chat/router model is selected live."""
    def __init__(self, entry, threads=DEFAULT_THREADS, ctx=2048, mmap=True, kv_type=None,
                 embeddings=False, warmup=True, host="127.0.0.1", port=None, ready_timeout=180):
        self.host = host
        # A fixed port (the daemon's chat :8771) must be free before we spawn: otherwise a stale
        # server already listening there would answer /health and we'd declare ourselves "ready"
        # while our own child dies on the bind. A random port (_free_port) can't collide.
        if port is not None:
            probe = socket.socket()
            try:
                probe.settimeout(0.5)
                if probe.connect_ex((host, port)) == 0:
                    raise RuntimeError("port %d already in use (stale llama-server?)" % port)
            finally:
                probe.close()
        self.port = port or _free_port()
        self._model_path = entry["path"]
        self.base = "http://%s:%d" % (self.host, self.port)
        cmd = [LLAMA_BIN + "/llama-server", "-m", entry["path"], "-t", str(threads),
               "-c", str(ctx), "--host", host, "--port", str(self.port), "--no-webui"]
        if not warmup:
            cmd.append("--no-warmup")
        if not mmap:
            cmd += ["-lm", "none"]   # this build spells no-mmap as --load-mode none (not --no-mmap)
        if kv_type:
            cmd += ["-ctk", kv_type, "-ctv", kv_type, "-fa", "on"]
        if embeddings:
            cmd += ["--embeddings", "--pooling", "mean"]
        if entry.get("mmproj"):
            cmd += ["--mmproj", entry["mmproj"]]
        self.log_path = "/tmp/llama-server-%d.log" % self.port
        self._fh = open(self.log_path, "wb")
        # own session/process-group so close() can kill the whole thing, never just the leader.
        self.proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.pid = self.proc.pid
        try: self.pgid = os.getpgid(self.pid)   # captured at spawn; new session => pgid == pid
        except Exception: self.pgid = self.pid
        _LIVE.add(self)
        pidfile = os.environ.get("HEYTUX_PIDFILE")   # parent sweeps this on worker timeout/crash
        if pidfile:
            try:
                with open(pidfile, "a") as f: f.write("%d\n" % self.pid)
            except OSError: pass
        try:
            self._wait_ready(ready_timeout)
        except Exception:
            self.close(); raise          # a server that never came up must not leak either

    def _wait_ready(self, timeout):
        # tight poll so the measured load time is ~time-to-health, not a poll-interval artefact.
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited rc=%s, log tail:\n%s"
                                   % (self.proc.returncode, self._log_tail()))
            try:
                r = urllib.request.urlopen(self.base + "/health", timeout=2)
                if json.load(r).get("status") == "ok":
                    return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("llama-server not ready in %ds, log tail:\n%s" % (timeout, self._log_tail()))

    def _log_tail(self, n=1500):
        try:
            return open(self.log_path).read()[-n:]
        except OSError:
            return ""

    def _post(self, path, body, timeout=120, stream=False):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout) if stream \
            else json.load(urllib.request.urlopen(req, timeout=timeout))

    def chat(self, messages, max_tokens=60, temperature=0.7, system=None, think=False):
        """Streamed completion. Returns {text, ttft_s, n_tokens, elapsed_s, tok_s}. ttft is wall time
        to the first content chunk; tok_s is the server's own timings.predicted_per_second when the
        final chunk carries it (decode throughput, excludes TTFT), else n_tokens/(elapsed-ttft)."""
        if system is not None:
            messages = [{"role": "system", "content": system}] + list(messages)
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "stream_options": {"include_usage": True},
                "timings_per_token": True, "chat_template_kwargs": {"enable_thinking": think}}
        t0 = time.time(); ttft = None; out = []; timings = None; usage = None
        finished = False; err = None
        resp = self._post("/v1/chat/completions", body, stream=True)
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True; break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("error"):
                    err = obj["error"]; break
                if obj.get("timings"): timings = obj["timings"]
                if obj.get("usage"): usage = obj["usage"]
                for c in (obj.get("choices") or []):
                    if c.get("finish_reason"): finished = True
                    delta = c.get("delta", {}).get("content")
                    if delta:
                        if ttft is None: ttft = time.time() - t0
                        out.append(delta)
        finally:
            try: resp.close()
            except Exception: pass
        if err is not None or not finished:
            raise RuntimeError("chat stream incomplete: %s" % (err or "no finish marker"))
        elapsed = time.time() - t0
        # token count from the server, never from chunk counting
        n = (usage or {}).get("completion_tokens")
        if n is None and timings: n = timings.get("predicted_n")
        tok_s = round(timings["predicted_per_second"], 2) if (timings and timings.get("predicted_per_second")) else None
        return {"text": "".join(out).strip(), "ttft_s": ttft, "n_tokens": n,
                "elapsed_s": elapsed, "tok_s": tok_s}

    def route(self, transcript, tools, system="You are a function router. Call the single best tool."):
        """OpenAI tool-calling (--jinja is on by default). Returns {names, calls, raw}. Empty names
        == no tool chosen. Decides the route without heytux's grounding/NOMATCH gates (those live in
        the daemon); the bench applies its own none-rule."""
        body = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": transcript}],
                "tools": tools, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256}
        r = self._post("/v1/chat/completions", body, timeout=120)
        msg = r["choices"][0]["message"]
        calls = []
        for c in (msg.get("tool_calls") or []):
            fn = c.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: args = {}
            calls.append({"name": fn.get("name"), "arguments": args or {}})
        return {"names": [c["name"] for c in calls], "calls": calls, "raw": msg}

    def embed(self, texts):
        """L2-normalized embeddings for a list of strings as an (n, d) float32 array."""
        r = self._post("/v1/embeddings", {"input": list(texts)}, timeout=120)
        vecs = np.array([d["embedding"] for d in r["data"]], dtype=np.float32)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True); norm[norm == 0] = 1.0
        return vecs / norm

    def close(self):
        """Idempotent teardown: on the FIRST call, SIGTERM then SIGKILL the server's whole process
        group (pgid captured at spawn), then clear pgid. A second call (e.g. the caller's finally
        after measure_chat already closed it) is a no-op, so we never killpg a pgid that has been
        recycled by an unrelated process."""
        _LIVE.discard(self)
        if getattr(self, "_closed", False):
            return
        self._closed = True
        pgid = self.pgid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if pgid is not None: os.killpg(pgid, sig)
                else: self.proc.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass
            try: self.proc.wait(timeout=5)
            except Exception: pass
        self.pgid = None
        try: self._fh.close()
        except Exception: pass

# ---------- sherpa-onnx: ASR ----------
def _one(d, *globs):
    for g in globs:
        c = sorted(glob.glob(os.path.join(d, g)) or glob.glob(os.path.join(d, "**", g), recursive=True))
        if c:
            return c[0]
    raise FileNotFoundError("%s: none of %s" % (d, globs))

class SherpaASR:
    """Offline recognizer, export shape auto-detected from the dir (moonshine / transducer /
    whisper / nemo-ctc). .transcribe(float32 @16k) -> text."""
    def __init__(self, entry, threads=DEFAULT_THREADS):
        import sherpa_onnx as so
        d = entry["path"]
        self.files = []     # resolved model files actually loaded (so the record shows int8 vs fp32)
        if os.path.isfile(d + "/encoder_model.ort"):
            self.files = [d + "/encoder_model.ort", d + "/decoder_model_merged.ort"]
            self.rec = so.OfflineRecognizer.from_moonshine_v2(
                encoder=d + "/encoder_model.ort", decoder=d + "/decoder_model_merged.ort",
                tokens=d + "/tokens.txt", num_threads=threads)
        elif glob.glob(d + "/*joiner*.onnx"):
            mt = "nemo_transducer" if re.search(r"nemo|parakeet", d) else "transducer"
            self.files = [_one(d, "*encoder*.onnx"), _one(d, "*decoder*.onnx"), _one(d, "*joiner*.onnx")]
            self.rec = so.OfflineRecognizer.from_transducer(
                encoder=self.files[0], decoder=self.files[1], joiner=self.files[2],
                tokens=_one(d, "*tokens.txt"), num_threads=threads, model_type=mt)
        elif glob.glob(d + "/*decoder*.onnx"):
            self.files = [_one(d, "*encoder*.onnx"), _one(d, "*decoder*.onnx")]
            self.rec = so.OfflineRecognizer.from_whisper(
                encoder=self.files[0], decoder=self.files[1],
                tokens=_one(d, "*tokens.txt"), num_threads=threads)
        else:
            self.files = [_one(d, "model.int8.onnx", "model.onnx")]
            self.rec = so.OfflineRecognizer.from_nemo_ctc(
                model=self.files[0], tokens=_one(d, "*tokens.txt"), num_threads=threads)
        self.files = [os.path.basename(f) for f in self.files]

    def transcribe(self, samples_f32):
        s = self.rec.create_stream(); s.accept_waveform(SR, samples_f32)
        self.rec.decode_stream(s)
        return s.result.text.strip()

    def close(self):
        self.rec = None; gc.collect()

# ---------- sherpa-onnx: TTS ----------
class SherpaTTS:
    """Offline TTS, kokoro or piper/vits auto-detected. .synth(text, sid, speed) -> (float32, sr)."""
    def __init__(self, entry, threads=DEFAULT_THREADS):
        import sherpa_onnx as so
        d = entry["path"]
        data_dir = d + "/espeak-ng-data"
        if os.path.isfile(d + "/voices.bin"):
            model = so.OfflineTtsModelConfig(kokoro=so.OfflineTtsKokoroModelConfig(
                model=_one(d, "model.int8.onnx", "model.onnx"), voices=d + "/voices.bin",
                tokens=d + "/tokens.txt", data_dir=data_dir, dict_dir=d + "/dict",
                lexicon=",".join(glob.glob(d + "/lexicon*.txt"))), num_threads=threads)
        else:
            model = so.OfflineTtsModelConfig(vits=so.OfflineTtsVitsModelConfig(
                model=_one(d, "*.onnx"), lexicon="", tokens=d + "/tokens.txt",
                data_dir=data_dir), num_threads=threads)
        self.tts = so.OfflineTts(so.OfflineTtsConfig(model=model))

    def synth(self, text, sid=0, speed=1.0):
        a = self.tts.generate(text, sid=sid, speed=speed)
        return np.asarray(a.samples, dtype=np.float32), a.sample_rate

    def close(self):
        self.tts = None; gc.collect()

# ---------- sherpa-onnx: VAD ----------
class SherpaVAD:
    """Silero or TEN VAD. .segments(float32 @16k) -> list[(start_s, end_s)]."""
    def __init__(self, entry, threads=1, threshold=0.3):
        import sherpa_onnx as so
        d = entry["path"]
        # accept either a directory or a direct .onnx path
        onnx = d if os.path.isfile(d) and d.endswith(".onnx") else None
        if "ten" in entry["id"].lower():
            cfg = so.VadModelConfig(ten_vad=so.TenVadModelConfig(
                model=onnx or _one(d, "*.onnx"), threshold=threshold, min_silence_duration=0.1,
                min_speech_duration=0.05, window_size=256), sample_rate=SR, num_threads=threads)
        else:
            cfg = so.VadModelConfig(silero_vad=so.SileroVadModelConfig(
                model=onnx or _one(d, "*int8.onnx", "*.onnx"), threshold=threshold,
                min_silence_duration=0.1, min_speech_duration=0.05, window_size=512),
                sample_rate=SR, num_threads=threads)
        self.vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    def segments(self, samples_f32):
        self.vad.reset()
        self.vad.accept_waveform(samples_f32)
        self.vad.flush()
        out = []
        while not self.vad.empty():
            seg = self.vad.front
            out.append((seg.start / SR, (seg.start + len(seg.samples)) / SR))
            self.vad.pop()
        return out

    def close(self):
        self.vad = None; gc.collect()

# ---------- routers (shared by the daemon's live router slot and the bench) ----------
# Similarity floor for the embedding router, calibrated on the held-out A-set (bench ASR phrases +
# OFFTOPIC + 30 fresh paraphrases, none of them seeds): 0.40 gave the best held-out tool accuracy
# (0.84) while keeping none-recall high (0.91) - higher floors only trade tool accuracy for none
# recall. Below the floor the router emits no call and the daemon falls to chat. This is the
# embedding router's OWN gate; CONF_FLOOR is Needle's and does not apply (see decide_route self_gated).
EMB_ROUTER_FLOOR = 0.40

# Compact seed phrases per tool for the embedding-classifier router. Hand-written; grow it if a
# tool mis-clusters. Single-sourced here so the daemon's live router and the bench score the exact
# same classifier (run.py's measure_embedding aliases this).
ROUTER_EXAMPLES = {
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

# ---- spoken-number / duration parsing for the embedding router's argument extraction ----
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_UNITS = ("hours", "hour", "minutes", "minute", "seconds", "second")
_UNIT_S = {"hour": 3600.0, "hours": 3600.0, "minute": 60.0, "minutes": 60.0,
           "second": 1.0, "seconds": 1.0}

def _toks(s):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())

def _spoken_number(toks):
    """First number in a token list as a float, from digits ('5', '1.5') or words (zero..ninety-nine,
    compounds like 'forty five', optional 'hundred'). None if there is no number."""
    for i, w in enumerate(toks):
        if re.fullmatch(r"\d+(?:\.\d+)?", w):
            return float(w)
        if w in _ONES or w in _TENS:
            val = _ONES.get(w, _TENS.get(w))
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if w in _TENS and nxt in _ONES and 0 < _ONES[nxt] < 10:   # "forty five" -> 45
                val += _ONES[nxt]; nxt = toks[i + 2] if i + 2 < len(toks) else None
            if nxt == "hundred":
                val = (val or 1) * 100
            return float(val)
        if w == "hundred":
            return 100.0
    return None

def _percent(phrase):
    """A percent value 0..100 from the phrase, or None when none is stated (so the handler can ask)."""
    n = _spoken_number(_toks(phrase))
    if n is None:
        return None
    n = int(round(n))
    return n if 0 <= n <= 100 else None

def _quantity(seg):
    """The numeric quantity in the words before a unit: a number (digits/words/compound), 'a/an'=1,
    and a 'half' modifier (+0.5, or 0.5 on its own as in 'half a(n)'). None if unparseable."""
    s = [t for t in seg if t not in ("and", "for", "please", "the", "in")]
    half = "half" in s
    s = [t for t in s if t != "half"]
    num = _spoken_number(s)
    if num is None:
        if half:
            num = 0.0                                # "half a(n) <unit>" -> 0.5 unit
        elif any(t in ("a", "an") for t in s):
            num = 1.0                                # "a/an <unit>" -> 1 unit
        else:
            return None
    qty = num + (0.5 if half else 0.0)
    return qty if qty > 0 else None

def _duration_seconds(phrase):
    """A timer duration in whole seconds (ceil), or None when unstated/ambiguous (then extract_args
    returns {} and the handler asks). Sums every number+unit pair so compounds work ('one hour and
    thirty minutes'=5400, 'one and a half hours'=5400), handles trailing 'and a half' ('an hour and
    a half'=5400), 'half an hour'=1800. Any pair it can't fully parse -> None."""
    toks = _toks(phrase)
    units = [(i, w) for i, w in enumerate(toks) if w in _UNIT_S]
    if not units:
        return None                                 # no unit -> a bare number is ambiguous
    total = 0.0; start = 0
    for ui, u in units:
        qty = _quantity(toks[start:ui])
        if qty is None:
            return None
        total += qty * _UNIT_S[u]; start = ui + 1
    if "half" in toks[start:]:                       # "... <unit> and a half" -> +0.5 of the last unit
        total += 0.5 * _UNIT_S[units[-1][1]]
    return max(1, math.ceil(total)) if total > 0 else None

def _timer_label(phrase):
    """A timer label when one is spoken ('timer for 5 minutes for pasta' -> 'pasta'): the first
    'for ...' clause that is neither a number nor a duration. None when no label is said."""
    for seg in phrase.lower().split(" for ")[1:]:
        seg = seg.strip()
        st = _toks(seg)
        if seg and not any(u in st for u in _UNITS) and _spoken_number(st) is None:
            return seg
    return None

def extract_args(name, phrase):
    """Fill a tool call's arguments from the raw phrase for the embedding router, which classifies a
    tool NAME only. Mirrors the argument names/types Needle emits for the parameterized tools
    (set_brightness percent:int, set_volume action/percent, set_timer minutes:int or seconds:int, +label, open_app
    app:str); every other tool takes none. Parses spoken numbers and units; when a value is ambiguous
    it returns {} so the handler asks rather than inventing one. Stdlib only, no model."""
    p = phrase.lower()
    if name == "set_brightness":
        pct = _percent(p)
        return {"percent": pct} if pct is not None else {}
    if name == "set_volume":
        if "unmute" in p:
            return {"action": "unmute"}
        if "mute" in p:
            return {"action": "mute"}
        pct = _percent(p)
        return {"action": "set", "percent": pct} if pct is not None else {"action": "set"}
    if name == "set_timer":
        secs = _duration_seconds(p)
        if secs is None:
            return {}
        out = {"minutes": secs // 60} if secs % 60 == 0 else {"seconds": secs}
        label = _timer_label(p)
        if label:
            out["label"] = label
        return out
    if name == "open_app":
        app = re.sub(r"\b(open|launch|start|the|an?|my|app|please)\b", " ", p)
        return {"app": re.sub(r"\s+", " ", app).strip()}
    return {}

class EmbeddingRouter:
    """Embedding-classifier router over a LlamaServer(--embeddings) backend. Candidates are built from
    the ENABLED merged tools only (so a disabled tool can never be chosen and a re-described tool
    routes by its new description): each tool's seeds are ROUTER_EXAMPLES plus its own description, so
    a tool with no examples stays routable. complete(transcript) is self-gated by its own similarity
    floor (CONF_FLOOR is Needle's, not this one's) and returns the Needle-shaped dict decide_route
    consumes: cosine argmax, floor, extract_args() for the parameterized tools."""
    def __init__(self, server, tools, threshold=0.55, examples=ROUTER_EXAMPLES):
        self.server = server
        self.threshold = threshold
        self.labels, texts = [], []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name")
            if not name or name == "ask_question":     # ask_question is the chat sentinel, not a tool
                continue
            seeds = list(examples.get(name, []))
            if fn.get("description"):
                seeds.append(fn["description"])
            for s in seeds:
                self.labels.append(name); texts.append(s)
        self.E = server.embed(texts) if texts else np.zeros((0, 1), np.float32)
        self.pid = getattr(server, "pid", None)

    def complete(self, transcript):
        if not self.labels:
            return {"function_calls": [], "confidence": 0.0, "reasoning": "", "self_gated": True}
        v = self.server.embed([transcript])[0]
        sims = self.E @ v
        i = int(np.argmax(sims))
        score = float(sims[i])
        if score < self.threshold:
            return {"function_calls": [], "confidence": score, "reasoning": "", "self_gated": True}
        name = self.labels[i]
        return {"function_calls": [{"name": name, "arguments": extract_args(name, transcript)}],
                "confidence": score, "reasoning": "", "self_gated": True}

    def close(self):
        self.server.close()

class GgufRouter:
    """GGUF function-calling router over a LlamaServer. complete() adapts LlamaServer.route() to the
    Needle-shaped dict; it is self-gated (the model's own tool_choice decides), so confidence is 1.0
    when a tool is chosen, else 0.0, and CONF_FLOOR does not apply. Calls to a tool that isn't in the
    enabled set are dropped. The daemon's grounding gate still applies."""
    def __init__(self, server, tools):
        self.server = server
        self.tools = tools
        self.enabled = {t["function"]["name"] for t in tools}
        self.pid = getattr(server, "pid", None)

    def complete(self, transcript):
        r = self.server.route(transcript, self.tools)
        calls = [{"name": c.get("name"), "arguments": c.get("arguments") or {}}
                 for c in r["calls"] if c.get("name") in self.enabled]
        return {"function_calls": calls, "confidence": 1.0 if calls else 0.0,
                "reasoning": "", "self_gated": True}

    def close(self):
        self.server.close()

# ---------- dispatch ----------
def resolve_gguf(path, slot=None):
    """A GGUF model's file path. If `path` is a directory, pick its single non-mmproj *.gguf. Shared
    by load() (the server) and run.py's llama-bench so both get the same file (manifest entries like
    Gemma-4-E2B point at a directory, which `llama-server -m`/`llama-bench -m` cannot take)."""
    if os.path.isdir(path):
        ggufs = [f for f in sorted(glob.glob(path + "/*.gguf"))
                 if "mmproj" not in os.path.basename(f).lower()]
        if not ggufs:
            raise NotImplementedError("%s: no base gguf in %s (mmproj pairs with a chat model)"
                                      % (slot or "model", path))
        return ggufs[0]
    if slot == "vision" and "mmproj" in os.path.basename(path).lower():
        raise NotImplementedError("vision: %s is an mmproj projector; benchmark with its base chat gguf" % path)
    return path

def load(entry, **opts):
    """Manifest entry -> backend object, dispatched by slot (the runtime label varies: VAD onnx is
    still loaded through sherpa_onnx). Raises NotImplementedError for a slot with no adapter (wake)
    or a runtime an adapter can't host (ONNX embeddings); callers record that as skipped."""
    slot, rt = entry["slot"], entry.get("runtime")
    if slot in ("chat", "router", "vision"):
        e = dict(entry)
        e["path"] = resolve_gguf(entry["path"], slot)
        return LlamaServer(e, **opts)
    if slot == "embedding":
        if rt == "llama.cpp":
            return LlamaServer(entry, embeddings=True, **opts)
        raise NotImplementedError("embedding runtime %s needs onnxruntime, no adapter (id=%s)"
                                  % (rt, entry["id"]))
    if slot == "asr":
        return SherpaASR(entry, **opts)
    if slot == "tts":
        return SherpaTTS(entry, **opts)
    if slot == "vad":
        return SherpaVAD(entry, **opts)
    raise NotImplementedError("no backend for slot=%s runtime=%s (id=%s)" % (slot, rt, entry["id"]))
