#!/usr/bin/env python3
# hybrid_e2e.py - drive the LIVE heytux daemon in ROUTE_MODE=hybrid over the held-out + Czech phrases
# and report end-to-end accuracy, latency and how many turns took the router fast path. Uses dry /text
# so no hardware is toggled and nothing is spoken (the route + timing is still real). The daemon must
# be running; this swaps CHAT_MODEL and ROUTE_MODE via /config, then restores them.
#   python3 hybrid_e2e.py --model Qwen3.5-2B-Q4_0
import argparse, json, os, statistics, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolcall                                  # phrase sets (heldout + CZECH_PHRASES)

TOOL_ROUTES = {"tool", "tool-fast", "llm-tool"}
ANSWER_ROUTES = {"chat-tool", "chat-fallback", "llm-answer", "chat-followup"}


def _post(base, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _get(base, path, timeout=10):
    return json.load(urllib.request.urlopen(base + path, timeout=timeout))


def run(base, model, phrases):
    rows = []
    for phrase, exp in phrases:
        t = time.time()
        try:
            r = _post(base, "/text", {"text": phrase, "dry": True})
        except Exception as e:
            rows.append({"phrase": phrase, "expected": exp, "route": "ERROR", "name": None,
                         "lat_ms": None, "error": repr(e)[:150]}); continue
        ms = r.get("ms") or {}
        calls = r.get("calls") or []
        rows.append({"phrase": phrase, "expected": exp, "route": r.get("route"),
                     "name": calls[0]["name"] if calls else None,
                     "lat_ms": sum(ms.get(k, 0) for k in ("route", "llm", "act"))})
    return rows


def summarize(rows):
    ok = tool_total = tool_ok = none_total = none_ok = fast = 0
    lats = [r["lat_ms"] for r in rows if r.get("lat_ms") is not None]
    for r in rows:
        exp, route, name = r["expected"], r["route"], r["name"]
        if route == "tool-fast":
            fast += 1
        if exp is None:
            none_total += 1
            # Like test_routing.check(): the date question accepts either a factual answer or
            # the clock tool (which also speaks the date), not a false positive phone action.
            if route in ANSWER_ROUTES or (r["phrase"] == "what is today's date" and
                    route in TOOL_ROUTES and name == "what_time_is_it"):
                none_ok += 1; ok += 1
        else:
            tool_total += 1
            if route in TOOL_ROUTES and name == exp:
                tool_ok += 1; ok += 1
    return {"n": len(rows), "accuracy": round(ok / len(rows), 3) if rows else None,
            "tool_accuracy": round(tool_ok / tool_total, 3) if tool_total else None,
            "none_accuracy": round(none_ok / none_total, 3) if none_total else None,
            "fast_path": fast, "fast_path_pct": round(100 * fast / len(rows)) if rows else None,
            "latency_ms_median": round(statistics.median(lats)) if lats else None,
            "latency_ms_p90": round(statistics.quantiles(lats, n=10)[-1]) if len(lats) > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CHAT_MODEL id to swap in")
    ap.add_argument("--base", default="http://127.0.0.1:8770")
    ap.add_argument("--out", default=os.path.join(HERE, "results/run1/toolcall"))
    args = ap.parse_args()

    before = _get(args.base, "/config")["values"]
    prev_model, prev_mode = before["CHAT_MODEL"], before["ROUTE_MODE"]
    print("swapping CHAT_MODEL=%s ROUTE_MODE=hybrid (was %s/%s)" % (args.model, prev_model, prev_mode))
    try:
        print(_post(args.base, "/config", {"CHAT_MODEL": args.model, "ROUTE_MODE": "hybrid"}, timeout=300))
        phrases = toolcall.heldout() + toolcall.CZECH_PHRASES
        rows = run(args.base, args.model, phrases)
        rec = {"id": args.model, "mode": "hybrid-e2e", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "summary": summarize(rows), "rows": rows}
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, args.model + "_hybrid_e2e.json")
        json.dump(rec, open(path, "w"), ensure_ascii=False, indent=1)
        print(json.dumps(rec["summary"], indent=1)); print("->", path)
    finally:
        # The first POST may have timed out while the daemon is still loading/prewarming (it then
        # answers 409 to a second swap), so retry for up to 15 min and verify the restore took.
        print("restoring CHAT_MODEL=%s ROUTE_MODE=%s" % (prev_model, prev_mode))
        want = {"CHAT_MODEL": prev_model, "ROUTE_MODE": prev_mode}
        for _ in range(30):
            try:
                _post(args.base, "/config", want, timeout=300)
            except Exception as e:
                print("  restore not yet applied: %r" % e)
            try:
                v = _get(args.base, "/config")["values"]
                if all(v.get(k) == x for k, x in want.items()):
                    break
            except Exception:
                pass
            time.sleep(30)
        else:
            raise SystemExit("RESTORE FAILED: daemon config is not back to %r" % want)


if __name__ == "__main__":
    main()
