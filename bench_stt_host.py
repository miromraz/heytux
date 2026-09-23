#!/usr/bin/env python3
# bench_stt_host.py - HOST-side STT shootout for heytux, reproducing the QUIET real-capture
# condition the phone actually hears. Standalone: does NOT import heytux (another agent edits it),
# does NOT touch the phone except one batched dry:true routing call per run.
#
# Why a second script (bench_stt.py stays as-is): that one records on the phone and scores against
# the LIVE router by EXECUTING tools. This one runs entirely on the host - same .onnx = same
# transcript as the phone (accuracy is device-independent) - and gets routing from the phone with
# dry:true so nothing fires. Phone latency/RAM for the finalists is a separate --phone pass.
#
# Pipeline (--run):
#   1. reference: transcribe the 7 REAL user captures with a strong host model (REFERENCE, not
#      ground truth - flagged as such in the report).
#   2. synth: TTS the labeled command phrases with >=3 piper voices -> clean 16 kHz.
#   3. conditions per synth wav:
#        clean = TTS as-is (loud, quiet-room-free: the ceiling)
#        sim   = scaled to the measured user peak (~3000/32768) + real room noise mixed in
#        norm  = sim after the daemon's planned peak-normalize (-6 dBFS, +24 dB gain cap)
#   4. transcribe every (model x condition) wav; WER vs the label.
#   5. one ssh call: route every unique transcript through the phone daemon /text {dry:true};
#      command success = routed to the expected tool (or chat).
#   6. markdown report.
#
# stdlib + numpy + sherpa_onnx + soundfile.
import argparse, glob, gc, json, os, re, statistics, subprocess, sys, time, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
THREADS = 4                       # daemon uses 4
TARGET_PEAK = 3000 / 32768.0      # measured real-capture peak (brief)
NORM_PEAK = 10 ** (-6 / 20.0)     # -6 dBFS peak target for the daemon normalizer
NORM_MAX_GAIN = 10 ** (24 / 20.0) # +24 dB gain cap

# phone routing (dry) - reachable via the sunfish ssh alias from the brief
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "UserKnownHostsFile=%s/.ssh/known_hosts.sunfish" % os.path.expanduser("~"),
       "-o", "HostKeyAlias=172.16.42.1", "root@172.16.42.1"]

def log(*a): print("[bench]", *a, file=sys.stderr, flush=True)

# ---------- phrase set: test_routing.CASES + bench_stt.PHRASES, deduped. expected tool, or None
# for chatter/chat (must route to chat, never fire a tool). args only where a slot must match. ----
def phrase_set():
    P = [
        ("turn on the flashlight", "flashlight_on", {}),
        ("turn off the flashlight", "flashlight_off", {}),
        ("turn on wifi", "wifi_on", {}),
        ("turn off wifi", "wifi_off", {}),
        ("enable bluetooth", "bluetooth_on", {}),
        ("turn on bluetooth", "bluetooth_on", {}),
        ("turn off bluetooth", "bluetooth_off", {}),
        ("set the brightness to 40 percent", "set_brightness", {}),
        ("set the volume to 30 percent", "set_volume", {}),
        ("mute the volume", "set_volume", {}),
        ("what is the battery level", "battery_status", {}),
        ("what's the battery level", "battery_status", {}),
        ("give me the full phone status", "phone_info", {}),
        ("set a timer for five minutes", "set_timer", {"minutes": 5}),
        ("what time is it", "what_time_is_it", {}),
        ("open the calculator", "open_app", {}),
        ("open the settings", "open_app", {}),
        ("take a photo", "take_photo", {}),
        ("take a selfie", "take_selfie", {}),
        ("lock the screen", "lock_screen", {}),
        ("lock the phone", "lock_screen", {}),
        ("pause the music", "play_pause", {}),
        ("play the music", "play_pause", {}),
        ("skip to the next track", "next_track", {}),
        ("next song", "next_track", {}),
        ("next track", "next_track", {}),
        ("find my phone", "find_my_phone", {}),
        ("where is my phone", "find_my_phone", {}),
        # chatter / chat-LLM (expected None = must route to chat)
        ("tell me a joke", None, {}),
        ("what is the capital of France", None, {}),
        ("how are you doing today", None, {}),
        ("what do you think about the weather", None, {}),
    ]
    seen, out = set(), []
    for t, tool, a in P:
        if t not in seen:
            seen.add(t); out.append((t, tool, a))
    return out

# ---------- wav io ----------
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

def write_wav_f32(path, x):
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)

def resample_to_16k(x, sr):
    # linear resample. ponytail: naive interp, fine for ASR eval; swap for polyphase if artefacts show.
    n = int(round(len(x) * SR / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

# ---------- noise pool from the quiet stretches of the REAL captures ----------
def build_noise_pool(cap_dir):
    """Collect the lowest-energy 30 ms frames (pure room noise, no speech) across all real wavs."""
    frame = 480
    quiet = []
    for w in sorted(glob.glob(os.path.join(cap_dir, "*.wav"))):
        x = read_wav_f32(w)
        nf = len(x) // frame
        if nf < 4:
            continue
        fr = x[:nf * frame].reshape(nf, frame)
        rms = np.sqrt((fr * fr).mean(1) + 1e-12)
        thr = np.percentile(rms, 30)          # quietest 30% of frames = between/before words
        quiet.append(fr[rms <= thr].reshape(-1))
    pool = np.concatenate(quiet) if quiet else np.zeros(SR, np.float32)
    log("noise pool: %.2fs, rms=%.5f peak=%.5f" % (len(pool) / SR, np.sqrt((pool*pool).mean()), np.abs(pool).max()))
    return pool

def make_conditions(clean, noise_pool, rng):
    """clean -> (sim, norm). sim = scale to TARGET_PEAK + a random real-noise segment at its own
    level (so SNR matches the real quiet captures). norm = daemon peak-normalize applied to sim."""
    pk = np.abs(clean).max() + 1e-9
    sig = clean * (TARGET_PEAK / pk)
    if len(noise_pool) > len(sig):
        i = rng.integers(0, len(noise_pool) - len(sig))
        noise = noise_pool[i:i + len(sig)]
    else:
        noise = np.resize(noise_pool, len(sig))
    sim = sig + noise
    pk2 = np.abs(sim).max() + 1e-9
    norm = sim * min(NORM_PEAK / pk2, NORM_MAX_GAIN)
    return sim.astype(np.float32), norm.astype(np.float32)

# ---------- TTS synth of the labeled set ----------
def synth_labeled(tts_root, phrases, out_dir, noise_pool):
    import sherpa_onnx as so
    voices = sorted(d.rstrip("/") for d in glob.glob(os.path.join(tts_root, "*/")))
    voices = [v for v in voices if glob.glob(v + "/*.onnx") and os.path.isdir(v + "/espeak-ng-data")]
    if len(voices) < 3:
        raise SystemExit("need >=3 piper voices in %s, found %d" % (tts_root, len(voices)))
    log("TTS voices: %s" % [os.path.basename(v) for v in voices])
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(1234)
    entries = []
    for v in voices:
        onnx = glob.glob(v + "/*.onnx")[0]
        tts = so.OfflineTts(so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
            vits=so.OfflineTtsVitsModelConfig(model=onnx, lexicon="",
                tokens=v + "/tokens.txt", data_dir=v + "/espeak-ng-data"), num_threads=THREADS)))
        vn = os.path.basename(v).replace("vits-piper-", "")
        for idx, (text, tool, targs) in enumerate(phrases):
            au = tts.generate(text, sid=0, speed=1.0)
            x = np.asarray(au.samples, dtype=np.float32)
            if au.sample_rate != SR:
                x = resample_to_16k(x, au.sample_rate)
            slug = "%s-%02d" % (vn, idx)
            clean = x
            sim, norm = make_conditions(clean, noise_pool, rng)
            paths = {}
            for cond, sig in (("clean", clean), ("sim", sim), ("norm", norm)):
                p = os.path.join(out_dir, "%s-%s.wav" % (slug, cond))
                write_wav_f32(p, sig); paths[cond] = p
            entries.append({"text": text, "tool": tool, "args": targs, "voice": vn, "paths": paths})
        del tts; gc.collect()
    log("synthesized %d phrases x %d voices x 3 conditions" % (len(phrases), len(voices)))
    return entries

# ---------- model loader (auto-detect export shape by dir contents) ----------
def _int8(d, stem):
    c = sorted(glob.glob(os.path.join(d, stem + "*.int8.onnx"))) or \
        sorted(glob.glob(os.path.join(d, stem + "*.onnx")))
    if not c:
        raise FileNotFoundError("%s: no %s*.onnx" % (d, stem))
    return c[0]

def load_model(d):
    import sherpa_onnx as so
    if os.path.isfile(d + "/encoder_model.ort") and os.path.isfile(d + "/decoder_model_merged.ort"):
        return so.OfflineRecognizer.from_moonshine_v2(encoder=d + "/encoder_model.ort",
            decoder=d + "/decoder_model_merged.ort", tokens=d + "/tokens.txt", num_threads=THREADS)
    enc = glob.glob(d + "/*encoder*.onnx")
    if enc and glob.glob(d + "/*joiner*.onnx"):    # transducer (parakeet tdt / zipformer)
        mt = "nemo_transducer" if re.search(r"nemo|parakeet", d) else "transducer"
        return so.OfflineRecognizer.from_transducer(encoder=_int8(d, os.path.basename(_stem(enc))),
            decoder=_int8(d, "decoder"), joiner=_int8(d, "joiner"), tokens=d + "/tokens.txt",
            num_threads=THREADS, model_type=mt)
    if enc and glob.glob(d + "/*decoder*.onnx"):   # whisper (encoder+decoder, no joiner)
        return so.OfflineRecognizer.from_whisper(encoder=_int8(d, _pfx(d) + "encoder"),
            decoder=_int8(d, _pfx(d) + "decoder"), tokens=glob.glob(d + "/*tokens.txt")[0],
            num_threads=THREADS)
    m = glob.glob(d + "/model.int8.onnx") or glob.glob(d + "/model.onnx")
    if m:                                          # nemo ctc single-model
        return so.OfflineRecognizer.from_nemo_ctc(model=m[0], tokens=d + "/tokens.txt", num_threads=THREADS)
    raise ValueError("unknown export shape in %s" % d)

def _stem(paths):
    return os.path.basename(paths[0]).split("encoder")[0] + "encoder"

def _pfx(d):
    e = glob.glob(d + "/*encoder*.onnx")[0]
    return os.path.basename(e).split("encoder")[0]

def transcribe(rec, x):
    s = rec.create_stream(); s.accept_waveform(SR, x); rec.decode_stream(s)
    return s.result.text.strip()

def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0

# ---------- WER ----------
def wer(ref, hyp):
    r = re.findall(r"[a-z0-9']+", (ref or "").lower())
    h = re.findall(r"[a-z0-9']+", (hyp or "").lower())
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

# ---------- routing via the phone daemon, dry:true, one ssh call ----------
ROUTE_REMOTE = "/home/miro/heytux-bench/route_dry.py"   # stage tools/heytux/route_dry.py here first

def route_batch(transcripts):
    """Route every unique transcript through the phone's /text {dry:true}. Returns {text: result}.
    Calls a staged .py file (not python3 -c) so newlines/parens survive ssh + runuser quoting."""
    uniq = sorted({t for t in transcripts if t and t.strip()})
    if not uniq:
        return {}
    cmd = SSH + ["runuser", "-u", "miro", "--", "python3", ROUTE_REMOTE]
    log("routing %d unique transcripts via phone (dry:true)" % len(uniq))
    p = subprocess.run(cmd, input=json.dumps(uniq).encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600)
    if p.returncode != 0:
        raise SystemExit("routing ssh failed: %s" % p.stderr.decode()[-500:])
    return json.loads(p.stdout.decode())

def routed_ok(res, tool, args):
    if not res or "error" in res:
        return False
    route = res.get("route") or ""
    if tool is None:
        return route.startswith("chat")
    if route != "tool":
        return False
    calls = res.get("calls") or []
    hit = next((c for c in calls if c.get("name") == tool), None)
    if hit is None:
        return False
    got = hit.get("arguments") or {}
    return all(got.get(k) == v for k, v in (args or {}).items())

# ---------- report ----------
def write_report(out_path, models, labeled_rows, ref_rows, cfg):
    L = ["# heytux STT benchmark (host, quiet-capture simulation) - %s" % time.strftime("%Y-%m-%d"), "",
         "Accuracy is measured on the HOST: sherpa-onnx runs the same .onnx the phone would, so a "
         "transcript is byte-identical to the phone's. Phone latency/RSS is a separate on-device pass "
         "(`--phone`) for the finalists only. Command success routes each transcript through the LIVE "
         "phone daemon `/text {dry:true}` (decides route, fires nothing).", "",
         "Reference transcripts for the real captures come from **%s** and are a strong-model "
         "REFERENCE, not human ground truth." % cfg["ref_name"], "",
         "Conditions: **clean** = TTS as-is (ceiling); **sim** = scaled to the measured user peak "
         "(~3000/32768) + real room noise from the quiet stretches of the real captures; **norm** = "
         "sim after the daemon's planned peak-normalize (-6 dBFS, +24 dB cap).", ""]
    # summary table
    L += ["## Accuracy (labeled synthetic set: %d phrases x %d voices)" % (cfg["nphrase"], cfg["nvoice"]), "",
          "WER = word error rate (lower better); CS = command success (routed to expected tool/chat).", "",
          "| model | size MB | WER clean | WER sim | WER norm | CS clean | CS sim | CS norm |",
          "|-|-|-|-|-|-|-|-|"]
    for m in models:
        r = labeled_rows[m["name"]]
        def f(k): return "-" if r[k] is None else ("%.2f" % r[k])
        def pc(k): return "-" if r[k] is None else ("%.0f%%" % (100 * r[k]))
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (m["name"], m["size_mb"],
            f("wer_clean"), f("wer_sim"), f("wer_norm"),
            pc("cs_clean"), pc("cs_sim"), pc("cs_norm")))
    # phone perf table (filled by --phone, merged in)
    if any("median_ms" in m for m in models):
        L += ["", "## Phone latency / RAM (finalists, num_threads=4, on sunfish)", "",
              "| model | median ms | p90 ms | RSS MB |", "|-|-|-|-|"]
        for m in models:
            if "median_ms" in m:
                L.append("| %s | %s | %s | %s |" % (m["name"], m["median_ms"], m["p90_ms"], m["rss_mb"]))
    # real captures
    L += ["", "## Real user captures: reference vs models (7 files)", ""]
    for rr in ref_rows:
        L += ["**%s** - reference (%s): %r" % (rr["file"], cfg["ref_name"], rr["ref"]),
              "  - moonshine-tiny (live daemon .json): %r" % rr.get("moonshine_json", "")]
        for m in models:
            if m.get("reference"):
                continue
            if m["name"] in rr.get("model_tx", {}):
                L.append("  - %s: %r" % (m["name"], rr["model_tx"][m["name"]]))
        L.append("")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    log("report -> %s" % out_path)

# ---------- main run ----------
def cmd_run(args):
    phrases = phrase_set()
    models = json.load(open(args.models))
    noise_pool = build_noise_pool(args.captures)
    # 1. synth labeled set
    synth = synth_labeled(args.tts, phrases, os.path.join(args.work, "synth"), noise_pool)
    # 2. transcribe labeled set + real captures with every model; reference model separately
    real_wavs = sorted(glob.glob(os.path.join(args.captures, "*.wav")))
    labeled_rows, all_tx, real_model_tx = {}, set(), {}
    ref_name = None
    for m in models:
        d = m["dir"]
        t0 = time.time(); rec = load_model(d)
        log("loaded %s in %d ms" % (m["name"], 1000 * (time.time() - t0)))
        # labeled
        per = {"clean": [], "sim": [], "norm": []}
        cs_tx = {"clean": [], "sim": [], "norm": []}
        for e in synth:
            for cond in ("clean", "sim", "norm"):
                tx = transcribe(rec, read_wav_f32(e["paths"][cond]))
                w = wer(e["text"], tx)
                per[cond].append(w)
                all_tx.add(tx)
                cs_tx[cond].append((tx, e["tool"], e["args"]))
        # real captures
        real_model_tx[m["name"]] = {}
        for rw in real_wavs:
            tx = transcribe(rec, read_wav_f32(rw))
            real_model_tx[m["name"]][os.path.basename(rw)] = tx
        labeled_rows[m["name"]] = {"_cs_tx": cs_tx,
            "wer_clean": _mean(per["clean"]), "wer_sim": _mean(per["sim"]), "wer_norm": _mean(per["norm"])}
        del rec; gc.collect()
        if m.get("reference"):
            ref_name = m["name"]
    # reference transcripts for the real captures
    if ref_name is None:
        ref_name = models[0]["name"]
    ref_rows = []
    caps_json = {os.path.basename(w): _json_transcript(w) for w in real_wavs}
    for rw in real_wavs:
        b = os.path.basename(rw)
        ref_rows.append({"file": b, "ref": real_model_tx[ref_name][b],
                         "moonshine_json": caps_json[b],
                         "model_tx": {mn: real_model_tx[mn][b] for mn in real_model_tx}})
    # checkpoint the (expensive) transcription BEFORE routing, so a routing/ssh failure never
    # discards it - resume with --resume <checkpoint>.
    cfg = {"ref_name": ref_name, "nphrase": len(phrases),
           "nvoice": len({e["voice"] for e in synth})}
    ckpt = {"labeled_rows": labeled_rows, "ref_rows": ref_rows, "cfg": cfg, "models": models,
            "route_texts": sorted(all_tx | {e["text"] for e in synth})}
    ckpt_path = os.path.join(args.work, "results_raw.json")
    json.dump(ckpt, open(ckpt_path, "w"), indent=2)
    log("checkpoint -> %s" % ckpt_path)
    if args.no_route:
        log("--no-route: checkpoint only. Route+report with: --resume %s" % ckpt_path)
        return
    _merge_phone_perf(ckpt["models"], args.phone_perf)
    _route_and_report(ckpt, args.out, os.path.join(args.work, "results.json"))

def _route_and_report(ckpt, out_path, results_path):
    routes = route_batch(ckpt["route_texts"])
    for mn, row in ckpt["labeled_rows"].items():
        cs = row.pop("_cs_tx")
        for cond in ("clean", "sim", "norm"):
            oks = [routed_ok(routes.get(tx, {}), tool, a) for tx, tool, a in cs[cond]]
            row["cs_" + cond] = (sum(oks) / len(oks)) if oks else None
    json.dump({"labeled_rows": ckpt["labeled_rows"], "ref_rows": ckpt["ref_rows"],
               "cfg": ckpt["cfg"], "models": ckpt["models"]}, open(results_path, "w"), indent=2)
    write_report(out_path, ckpt["models"], ckpt["labeled_rows"], ckpt["ref_rows"], ckpt["cfg"])

def cmd_resume(args):
    ckpt = json.load(open(args.resume))
    # JSON turned the cs_tx tuples into lists; routed_ok unpacks (tx, tool, args) either way.
    _merge_phone_perf(ckpt["models"], args.phone_perf)
    _route_and_report(ckpt, args.out, args.resume.replace("results_raw", "results"))

def _merge_phone_perf(models, path):
    if not path:
        return
    perf = json.load(open(path))
    for m in models:
        if m["name"] in perf:
            m.update(perf[m["name"]])

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def _json_transcript(wav):
    j = os.path.splitext(wav)[0] + ".json"
    if os.path.exists(j):
        try:
            return json.load(open(j)).get("transcript", "")
        except Exception:
            return ""
    return ""

# ---------- phone latency/RAM finalists ----------
def cmd_bench_phone_local(args):
    """Run ON the phone (invoked over ssh): load one model dir, transcribe wavs, print JSON."""
    wavs = sorted(glob.glob(os.path.join(args.wavdir, "*.wav")))[:args.n]
    rss0 = rss_kb(); rec = load_model(args.dir); rss1 = rss_kb()
    lat = []
    for w in wavs:
        x = read_wav_f32(w); t = time.time(); transcribe(rec, x); lat.append(1000 * (time.time() - t))
    lat.sort()
    print(json.dumps({"median_ms": int(statistics.median(lat)) if lat else None,
        "p90_ms": int(lat[int(0.9 * (len(lat) - 1))]) if lat else None,
        "rss_mb": round((rss1 - rss0) / 1024.0, 1), "n": len(lat)}))

def _selftest():
    assert wer("turn on the flashlight", "turn on the flashlight") == 0.0
    assert wer("turn off the flashlight", "an off flesh light") > 0.5
    assert routed_ok({"route": "tool", "calls": [{"name": "wifi_on", "arguments": {}}]}, "wifi_on", {})
    assert not routed_ok({"route": "tool", "calls": [{"name": "wifi_off", "arguments": {}}]}, "wifi_on", {})
    assert routed_ok({"route": "chat-fallback", "calls": []}, None, {})
    assert not routed_ok({"route": "tool", "calls": [{"name": "wifi_on", "arguments": {}}]}, None, {})
    assert routed_ok({"route": "tool", "calls": [{"name": "set_timer", "arguments": {"minutes": 5}}]}, "set_timer", {"minutes": 5})
    assert not routed_ok({"route": "tool", "calls": [{"name": "set_timer", "arguments": {"minutes": 9}}]}, "set_timer", {"minutes": 5})
    # sim/norm change level as intended
    rng = np.random.default_rng(0)
    clean = (np.sin(np.linspace(0, 200, SR)) * 0.9).astype(np.float32)
    sim, norm = make_conditions(clean, np.zeros(SR, np.float32), rng)
    assert abs(np.abs(sim).max() - TARGET_PEAK) < 1e-3
    assert abs(np.abs(norm).max() - NORM_PEAK) < 1e-2
    print("selftest OK")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--run", action="store_true", help="host: synth + transcribe + route + report")
    sub.add_argument("--resume", metavar="RAW", help="host: route + report from a results_raw.json checkpoint")
    sub.add_argument("--phone-local", action="store_true", help="(runs ON the phone) latency/RAM for one dir")
    sub.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", help="--run: JSON list of {name,dir,size_mb,reference?}")
    ap.add_argument("--captures", help="--run: dir of real *.wav (+ .json)")
    ap.add_argument("--tts", help="--run: dir of piper voice dirs")
    ap.add_argument("--work", help="--run: scratch dir for synth wavs + results.json")
    ap.add_argument("--out", help="--run/--resume: markdown report path")
    ap.add_argument("--phone-perf", help="--run/--resume: phone_perf.json {name:{median_ms,p90_ms,rss_mb}} to merge")
    ap.add_argument("--dir", help="--phone-local: one model dir")
    ap.add_argument("--wavdir", help="--phone-local: dir of wavs to time")
    ap.add_argument("--n", type=int, default=20, help="--phone-local: how many wavs")
    ap.add_argument("--no-route", action="store_true", help="--run: checkpoint transcription only, skip routing")
    args = ap.parse_args()
    if args.selftest: _selftest()
    elif args.run: cmd_run(args)
    elif args.resume: cmd_resume(args)
    elif args.phone_local: cmd_bench_phone_local(args)

if __name__ == "__main__":
    main()
