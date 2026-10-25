#!/usr/bin/env python3
# toolcall.py - on-phone benchmark of chat models as HYBRID tool-callers. Reuses the daemon's exact
# request path (heytux_backends.LlamaServer.tool_chat with the enabled tool specs and cache_prompt),
# so the numbers reflect what ROUTE_MODE=hybrid actually does. Also (with --czech) generates Czech
# chat answers for later human judging. Run ON the phone with the heytux daemon STOPPED (RAM/CPU).
#   python3 toolcall.py --models all --out results/run1/toolcall
#   python3 toolcall.py --models Qwen3.5-2B-Q4_0,LFM2.5-1.2B-Instruct-Q4_K_M --czech
import argparse, gc, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # /home/miro/heytux (heytux*.py live here)
import heytux_backends as HB
import heytux as J

MANIFEST = os.environ.get("HEYTUX_MANIFEST", os.path.join(HERE, "manifest.json"))

# The chat models the brief asks us to compare (manifest ids). Smallest first so a run that is cut
# short still covers the cheap candidates.
MODELS = ["Qwen3-0.6B-Q4_K_M", "Qwen3.5-0.8B-Q4_0", "Qwen3.5-0.8B-Q4_K_M",
          "Granite-4.0-1B-Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M", "LFM2.5-1.2B-Instruct-Q4_K_M",
          "Qwen3-1.7B-Q4_K_M", "Qwen3.5-2B-Q4_0", "Gemma-3n-E2B-it-Q4_K_M", "Qwen3.5-4B-Q4_K_M"]

# ~15 Czech commands/questions, proper diacritics. expected = English tool name, or None for a chat
# question (the hybrid model must map Czech -> the English tool, or answer).
CZECH_PHRASES = [
    ("rozsviť baterku", "flashlight_on"), ("zhasni baterku", "flashlight_off"),
    ("zapni wifi", "wifi_on"), ("vypni wifi", "wifi_off"), ("zapni bluetooth", "bluetooth_on"),
    ("nastav jas na čtyřicet procent", "set_brightness"), ("ztlum hlasitost", "set_volume"),
    ("jaký je stav baterie", "battery_status"), ("kolik je hodin", "what_time_is_it"),
    ("nastav časovač na pět minut", "set_timer"), ("otevři kalkulačku", "open_app"),
    ("zamkni obrazovku", "lock_screen"), ("najdi můj telefon", "find_my_phone"),
    ("jaké je hlavní město Francie", None), ("pověz mi vtip", None), ("jak se dnes máš", None),
]

# 12 Czech chat questions for quality judging: general knowledge, a follow-up PAIR (indices 5-6 share
# context), and a phone-related one. Answers saved verbatim; the orchestrator judges them.
CZECH_QA = [
    "Jaké je hlavní město České republiky?",
    "Kdo napsal knihu Osudy dobrého vojáka Švejka?",
    "Kolik je sedm krát osm?",
    "Pověz mi krátký vtip.",
    "Co je to fotosyntéza, vysvětli to jednoduše?",
    "Jaká je největší planeta sluneční soustavy?",     # 5: follow-up pair, first turn
    "A která je druhá největší?",                       # 6: follow-up, uses the previous answer
    "Vyjmenuj tři největší města v Česku.",
    "Jak se řekne anglicky dobrý den?",
    "Kolik mám volného místa v telefonu?",              # phone-related
    "Proč je obloha modrá?",
    "Dej mi tip, jak ušetřit baterii na telefonu.",
]


def heldout():
    """The ~75 held-out phrases: bench ASR phrases + OFFTOPIC + the routing table, deduped. expected
    is an English tool name, or None for a phrase that must be answered (chat), never routed."""
    import run, test_routing
    seen, out = set(), []
    def add(p, exp):
        k = p.lower().strip()
        if k and k not in seen:
            seen.add(k); out.append((p, exp))
    for p, e in run.ASR_PHRASES:
        add(p, e)
    for p in run.OFFTOPIC:
        add(p, None)
    for p, e in test_routing.CASES:
        add(p, None if e == "chat" else e)
    return out


def _daemon_state():
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "heytux"],
                           capture_output=True, text=True, timeout=10)
        state = r.stdout.strip()
        # is-active returns 3 for a stopped/failed unit. Lookup errors, unknown units and empty
        # output are not evidence that it is safe to start another multi-GB model.
        return state if r.returncode in (0, 3) else "unknown"
    except Exception:
        return "unknown"

def ensure_daemon_stopped(allow=False):
    """Stop even during auto-restart, then prove the unit AND its chat server are down. A failed
    stop/lookup must fail closed: two chat models can OOM the phone before either is listening."""
    if allow:
        print("daemon guard bypassed (--allow-daemon)", flush=True)
        return
    try:
        r = subprocess.run(["systemctl", "--user", "stop", "heytux"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode:
            raise RuntimeError("systemctl stop failed: " + r.stderr.strip())
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            st = _daemon_state()
            # Check the process too: an orphan server may still be loading and not listening yet.
            procs = subprocess.run(["pgrep", "-af", "llama-server"],
                                   capture_output=True, text=True, timeout=5)
            listeners = subprocess.run(["ss", "-H", "-ltn", "sport = :8771"],
                                       capture_output=True, text=True, timeout=5)
            if procs.returncode not in (0, 1) or listeners.returncode:
                raise RuntimeError("could not check for the chat server on :8771")
            chat = re.search(r"(?:--port(?:=|\s+)|-p\s+)8771(?:\s|$)", procs.stdout)
            if st in ("inactive", "failed") and not chat and not listeners.stdout.strip():
                print("daemon state: %s; :8771 clear" % st, flush=True)
                return
            time.sleep(1)
        raise RuntimeError("daemon state %s or chat server on :8771 did not stop" % st)
    except Exception as e:
        raise SystemExit("REFUSING to bench: %s" % e)

def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return None

def rss_mb(pid):
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    except OSError:
        pass
    return None


def resolve_entry(mid, manifest):
    for e in manifest:
        if e["id"] == mid:
            e = dict(e); e["path"] = HB.resolve_gguf(e["path"], e.get("slot"))
            return e
    raise KeyError(mid)


def predict(be, tools, enabled, phrase, max_tokens):
    """One hybrid tool-calling turn via the daemon's path. Returns (pred, latency_s, timings).
    pred is the first enabled tool name, or 'none' when the model answered instead."""
    msgs = J.chat_messages(J._chat_sys(), [], phrase)      # daemon hybrid path: no facts preamble
    t = time.time()
    r = be.tool_chat(msgs, tools, max_tokens=max_tokens)
    dt = time.time() - t
    names = [c.get("name") for c in r["calls"]       # same acceptance as heytux.chat_tool
             if c.get("name") in enabled and isinstance(c.get("arguments"), dict)]
    return (names[0] if names else "none"), dt, (r.get("timings") or {})


def metrics(rows):
    """Tool accuracy over tool-expected phrases + none precision/recall over chat-expected phrases."""
    tool_total = tool_ok = pred_none = truth_none = none_ok = 0
    for r in rows:
        exp, pred = r["expected"], r["pred"]
        if exp is None:
            # Same rule as hybrid_e2e/test_routing: the clock tool also answers the date question.
            truth_none += 1
            none_ok += (pred == "none" or (r["phrase"] == "what is today's date" and pred == "what_time_is_it"))
        else:
            tool_total += 1; tool_ok += (pred == exp)
        pred_none += (pred == "none")
    return {"tool_accuracy": round(tool_ok / tool_total, 3) if tool_total else None,
            "tool_n": tool_total,
            "none_recall": round(none_ok / truth_none, 3) if truth_none else None,
            "none_precision": round(none_ok / pred_none, 3) if pred_none else None,
            "none_n": truth_none}


def latency(rows):
    """Latency over the post-prewarm turns. A cold prefill was already paid (and the cache warmed)
    before this loop, so every row here reuses the system+tools prefix; the 'first' row is the user's
    first real turn after the prewarm. Failed/empty requests (timings absent, or an ERROR pred) are
    excluded so a failed turn is never reported as 0 ms latency."""
    valid = [r for r in rows if r.get("timings") and not str(r.get("pred")).startswith("ERROR")]
    if not valid:
        return {"note": "no valid turns"}
    first, rest = valid[0], valid[1:]
    def pm(r):  # prefill ms ~ time-to-first-token for a non-streamed call
        return (r["timings"] or {}).get("prompt_ms")
    cached_pm = [pm(r) for r in rest if pm(r) is not None]
    return {"first_turn_total_s": round(first["latency_s"], 2),
            "cached_turn_total_s_median": round(statistics.median(r["latency_s"] for r in rest), 2)
            if rest else None,
            "first_turn_prompt_ms": round(pm(first), 1) if pm(first) is not None else None,
            "cached_turn_prompt_ms_median": round(statistics.median(cached_pm), 1) if cached_pm else None,
            "prompt_tokens_first": (first["timings"] or {}).get("prompt_n"),
            "n_valid": len(valid), "n_error": len(rows) - len(valid)}


def czech_answers(be, model_id, max_tokens):
    """Generate the 12 Czech answers (one follow-up pair keeps history). Returns a list of
    {model, question, answer} dicts; the orchestrator judges them."""
    sys_prompt = J._chat_sys(tools=False)       # text-only language eval: no tools offered here
    out, history = [], []
    for i, q in enumerate(CZECH_QA):
        use_hist = history if i == 6 else []        # Q6 is the follow-up to Q5
        msgs = J.chat_messages(sys_prompt, use_hist, q)
        try:
            ans = be.chat(msgs, max_tokens=max_tokens, temperature=0.7).get("text", "")
        except Exception as e:
            ans = "ERROR: %r" % e
        out.append({"model": model_id, "question": q, "answer": ans})
        if i == 5:
            history = [(q, ans)]
    return out


def run_model(mid, manifest, args):
    entry = resolve_entry(mid, manifest)
    tools = J.hybrid_tools()
    enabled = {t["function"]["name"] for t in tools}
    phrases = heldout() + CZECH_PHRASES
    rec = {"id": mid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_phrases": len(phrases),
           "max_tokens": args.max_tokens, "n_tools": len(tools)}
    avail = mem_available_gb()
    rec["mem_available_gb"] = round(avail, 2) if avail is not None else None
    if args.min_avail_gb and avail is not None and avail < args.min_avail_gb:
        rec["status"] = "skipped"
        rec["note"] = "MemAvailable %.2f GB < --min-avail-gb %.2f" % (avail, args.min_avail_gb)
        return rec
    be = None; t0 = time.time()
    try:
        be = HB.LlamaServer(entry, threads=args.threads, ctx=2048)
        rec["load_s"] = round(time.time() - t0, 2)
        # Cold prefill of the system+tools prefix: what the user's very first hybrid turn costs
        # WITHOUT the daemon's prewarm (~1200 tokens, up to ~60 s here). Long timeout so a slow model
        # doesn't fail it. This also warms the slot cache, so every phrase below is a post-prewarm
        # turn - matching the daemon, which prewarms the prefix on chat load/swap.
        try:
            ct = time.time()
            cr = be.tool_chat(J.chat_messages(J._chat_sys(), [], "hi"), tools, max_tokens=1, timeout=300)
            rec["cold_prefill_s"] = round(time.time() - ct, 2)
            rec["cold_prefill_prompt_ms"] = (cr.get("timings") or {}).get("prompt_ms")
        except Exception as e:
            rec["cold_prefill_error"] = repr(e)[:200]
        n_heldout = len(heldout())
        rows = []
        tstart = time.time()
        for i, (phrase, exp) in enumerate(phrases):
            try:
                pred, dt, tm = predict(be, tools, enabled, phrase, args.max_tokens)
            except Exception as e:
                pred, dt, tm = "ERROR:%r" % e, 0.0, {}
            rows.append({"phrase": phrase, "expected": exp, "pred": pred,
                         "group": "heldout" if i < n_heldout else "czech",
                         "latency_s": round(dt, 2), "timings": tm})
            if args.per_timeout and time.time() - tstart > args.per_timeout:
                rec["truncated_after"] = len(rows); break
        rec["rss_mb"] = rss_mb(be.pid)
        rec["metrics_heldout"] = metrics([r for r in rows if r["group"] == "heldout"])
        rec["metrics_czech"] = metrics([r for r in rows if r["group"] == "czech"])
        rec["metrics_all"] = metrics(rows)
        rec["latency"] = latency(rows)
        rec["errors"] = [{"phrase": r["phrase"], "pred": r["pred"]}
                         for r in rows if str(r["pred"]).startswith("ERROR")]
        rec["rows"] = rows
        if args.czech:
            rec["czech_answers"] = czech_answers(be, mid, args.answer_tokens)
    except Exception as e:
        import traceback
        rec["status"] = "failed"; rec["error"] = repr(e)[:300]
        rec["traceback"] = traceback.format_exc()[-1200:]
    finally:
        if be is not None:
            be.close()
        gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="comma-separated manifest ids, or 'all'")
    ap.add_argument("--out", default=os.path.join(HERE, "results/run1/toolcall"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64, dest="max_tokens",
                    help="cap for a tool-call turn (keeps slow models bounded)")
    ap.add_argument("--answer-tokens", type=int, default=90, dest="answer_tokens")
    ap.add_argument("--per-timeout", type=int, default=0, dest="per_timeout",
                    help="stop a model's phrase loop after this many seconds (0 = no cap)")
    ap.add_argument("--czech", action="store_true", help="also generate Czech chat answers")
    ap.add_argument("--czech-only", action="store_true", dest="czech_only",
                    help="only generate Czech answers (don't run or overwrite the tool-call JSONs)")
    ap.add_argument("--min-avail-gb", type=float, default=0.0, dest="min_avail_gb",
                    help="skip a model if MemAvailable is below this before loading (0 = no guard)")
    ap.add_argument("--allow-daemon", action="store_true", dest="allow_daemon",
                    help="DANGER: bench even while the heytux daemon is active (can OOM the phone)")
    args = ap.parse_args()

    J.load_config()
    J.CFG["ROUTE_MODE"] = "hybrid"              # benchmark mode only; never persist it
    manifest = json.load(open(MANIFEST))
    ids = MODELS if args.models == "all" else args.models.split(",")
    os.makedirs(args.out, exist_ok=True)
    cz_all = []
    for mid in ids:
        ensure_daemon_stopped(args.allow_daemon)      # re-check before every model (OOM incident)
        print("=== %s ===" % mid, flush=True)
        if args.czech_only:
            avail = mem_available_gb()
            print("  MemAvailable: %s GB" % avail, flush=True)
            if args.min_avail_gb and (avail is None or avail < args.min_avail_gb):
                print("  skipped: MemAvailable %s GB < required %.2f GB (or unavailable)" %
                      (avail, args.min_avail_gb), flush=True)
                continue
            be = None
            try:
                entry = resolve_entry(mid, manifest)
                entry.pop("mmproj", None)       # Czech answers only need text; save vision RAM.
                be = HB.LlamaServer(entry, threads=args.threads, ctx=2048)
                cz_all += czech_answers(be, mid, args.answer_tokens)
                print("  czech answers generated", flush=True)
            except Exception as e:
                print("  czech FAILED: %r" % e, flush=True)
            finally:
                if be is not None:
                    be.close()
                gc.collect()
            continue
        rec = run_model(mid, manifest, args)
        with open(os.path.join(args.out, mid + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        m = rec.get("metrics_all", {})
        print("  tool_acc=%s none_p=%s none_r=%s rss=%sMB lat=%s status=%s" % (
            m.get("tool_accuracy"), m.get("none_precision"), m.get("none_recall"),
            rec.get("rss_mb"), rec.get("latency", {}).get("cached_turn_total_s_median"),
            rec.get("status", "ok")), flush=True)
        cz_all += rec.get("czech_answers", [])
    if cz_all:
        path = os.path.join(os.path.dirname(args.out), "czech_answers.json")
        existing = []
        if os.path.exists(path):
            try: existing = json.load(open(path))
            except Exception: existing = []
        json.dump(existing + cz_all, open(path, "w"), ensure_ascii=False, indent=1)
        print("czech answers -> %s (+%d)" % (path, len(cz_all)))


if __name__ == "__main__":
    main()
