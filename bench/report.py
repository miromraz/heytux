#!/usr/bin/env python3
# report.py - render the JSON records under a run dir into one markdown table per slot.
#   report.py <run_dir>          # writes <run_dir>/REPORT.md
import glob, json, os, sys

def mb(kb): return "-" if kb is None else "%.0f" % (kb / 1024.0)
def num(v, f="%.3f"): return "-" if v is None else (f % v if isinstance(v, (int, float)) else str(v))
def pct(v): return "-" if v is None else "%.0f%%" % (100 * v)

def _read_json(p):
    """Parse a json file, or warn and return None (a truncated artifact must not crash REPORT)."""
    try:
        return json.load(open(p))
    except Exception as e:
        print("WARN: skipping malformed %s: %r" % (p, e), file=sys.stderr)
        return None

def load(run_dir, slot):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, slot, "*.json"))):
        r = _read_json(p)
        if isinstance(r, dict) and r.get("slot"):   # skip artifacts (saved answers etc.)
            rows.append(r)
    return rows

def common_mem(r):
    m = r.get("mem") or {}
    return mb(m.get("rss_kb")), mb(m.get("pss_kb"))

def tbl(header, sep, rows):
    return [header, sep] + rows

def render_slot(run_dir, slot):
    rows = load(run_dir, slot)
    if not rows:
        return []
    L = ["", "## %s" % slot, ""]
    bad = [r for r in rows if r.get("status") != "ok"]
    ok = [r for r in rows if r.get("status") == "ok"]
    if slot == "chat":
        body = []
        for r in ok:
            lb = r.get("llama_bench") or {}
            rss, pss = common_mem(r)
            body.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], num(r.get("cold_load_s"), "%.1f"), num(r.get("load_s_no_warmup"), "%.1f"),
                num(lb.get("pp64"), "%.1f"), num(lb.get("tg64"), "%.1f"),
                num(r.get("ttft_s_median"), "%.2f"), num(r.get("tok_s_median"), "%.1f"), rss, pss,
                r.get("quant", "-")))
        L += tbl("| model | load s | no-warmup s | pp64 t/s | tg64 t/s | TTFT s | decode t/s | "
                 "RSS MB | PSS MB | quant |", "|-|-|-|-|-|-|-|-|-|-|", body)
        L += ["", "load s = time-to-health (warmup on); decode t/s = server timings predicted_per_second."]
    elif slot == "router":
        body = []
        for r in ok:
            g = r.get("gguf_router") or {}
            rss, _ = common_mem(r)
            body.append("| %s | %s | %s | %s | %s | %s |" % (
                r["id"], pct(g.get("accuracy")), pct(g.get("none_precision")),
                pct(g.get("none_recall")), num(g.get("latency_ms_median"), "%.0f"), rss))
        L += tbl("| model | tool acc | none prec | none recall | latency ms | RSS MB |",
                 "|-|-|-|-|-|-|", body)
        nb = next((r.get("needle3_baseline") for r in ok if r.get("needle3_baseline")), None)
        if nb:
            L += ["", "Needle3 baseline: " + (nb["skipped"] if "skipped" in nb else
                  "acc %s, none P/R %s/%s, %s ms" % (pct(nb.get("accuracy")),
                   pct(nb.get("none_precision")), pct(nb.get("none_recall")),
                   num(nb.get("latency_ms_median"), "%.0f")))]
    elif slot == "embedding":
        body = []
        for r in ok:
            er = r.get("embedding_router") or {}
            rss, _ = common_mem(r)
            body.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], num(r.get("embed_latency_ms_median"), "%.0f"), pct(er.get("accuracy")),
                pct(er.get("none_precision")), pct(er.get("none_recall")),
                num(er.get("threshold"), "%.2f"), rss))
        L += tbl("| model | embed ms | router acc | none prec | none recall | thr | RSS MB |",
                 "|-|-|-|-|-|-|-|", body)
    elif slot == "asr":
        body = []
        snr = next((r.get("noise_snr_db") for r in ok if r.get("noise_snr_db") is not None), "?")
        for r in ok:
            w = r.get("wer") or {}
            rss, _ = common_mem(r)
            body.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], num(r.get("rtf_median")), pct(w.get("clean")), pct(w.get("quiet")),
                pct(w.get("noise")), pct(r.get("wer_cs")), rss))
        L += tbl("| model | RTF | WER clean | WER quiet(-20dB) | WER noise(%s dB SNR) | WER cs | RSS MB |"
                 % snr, "|-|-|-|-|-|-|-|", body)
        for r in ok:
            s = r.get("wer_samples") or {}
            if s:
                L += ["", "**%s** ref/hyp samples:" % r["id"]]
                for cond in ("clean", "quiet", "noise"):
                    for e in s.get(cond, []):
                        L.append("  - [%s] ref=%r hyp=%r wer=%s" % (cond, e["ref"], e["hyp"], e["wer"]))
        for r in ok:
            caps = r.get("captures") or []
            if caps:
                L += ["", "**%s** real captures (no ground truth):" % r["id"]]
                L += ["  - %s: %r" % (c["file"], c["hyp"]) for c in caps]
    elif slot == "tts":
        body = []
        for r in ok:
            rss, _ = common_mem(r)
            body.append("| %s | %s | %s | %s | %s |" % (
                r["id"], num(r.get("rtf_median")), num(r.get("ttfa_s_median")),
                rss, r.get("wavs_dir", "-")))
        L += tbl("| model | RTF | TTFA s | RSS MB | wavs |", "|-|-|-|-|-|", body)
        L += ["", "TTFA is full offline synth time (no streaming)."]
    elif slot == "vad":
        body = []
        for r in ok:
            rss, _ = common_mem(r)
            body.append("| %s | %s | %s | %s | %s |" % (
                r["id"], num(r.get("speed_x_realtime_median"), "%.0f"),
                num(r.get("silero_agreement")), "yes" if r.get("is_silero") else "no", rss))
        L += tbl("| model | xRT | silero IoU | is silero | RSS MB |", "|-|-|-|-|-|", body)
    elif slot == "vision":
        body = []
        for r in ok:
            rss, _ = common_mem(r)
            body.append("| %s | %s | %s | %s | %s |" % (
                r["id"], num(r.get("cold_load_s"), "%.1f"), num(r.get("ttft_s"), "%.2f"),
                num(r.get("tok_s"), "%.1f"), rss))
        L += tbl("| model | load s | TTFT s | tok/s | RSS MB |", "|-|-|-|-|-|", body)
        for r in ok:
            if r.get("answer"):
                L += ["  - %s: %r" % (r["id"], r["answer"][:160])]
    else:
        L += ["| model | status |", "|-|-|"] + ["| %s | %s |" % (r["id"], r.get("status")) for r in ok]
    if bad:
        L += ["", "### %s: failed / skipped" % slot, "", "| model | status | detail |", "|-|-|-|"]
        for r in bad:
            L.append("| %s | %s | %s |" % (r["id"], r.get("status"),
                     (r.get("error") or r.get("note") or "")[:120]))
    return L

def render_zram(run_dir):
    """zram_tune writes a LIST of per-(config,model) records per file (not slot dicts)."""
    files = sorted(glob.glob(os.path.join(run_dir, "zram", "*.json")))
    if not files:
        return []
    L = ["", "## zram / swap tuning", "",
         "| config | model | load s | tg tok/s | TTFT s | MemAvail MB | swap MB | pswpout | zram ratio | OOM |",
         "|-|-|-|-|-|-|-|-|-|-|"]
    skipped, failed = [], []
    for p in files:
        recs = _read_json(p)
        if recs is None:
            continue
        for r in (recs if isinstance(recs, list) else [recs]):
            if r.get("status") == "skipped":
                skipped.append("%s (%s)" % (r.get("label"), (r.get("reason") or "")[:60])); continue
            if r.get("status") not in ("ok", None):
                failed.append((r.get("label"), r.get("model"), (r.get("error") or "")[:80])); continue
            z = r.get("zram") or {}
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("label"), r.get("model"), num(r.get("load_s"), "%.1f"),
                num(r.get("tg_tok_s"), "%.2f"), num(r.get("ttft_s"), "%.2f"),
                num(r.get("mem_avail_mb"), "%.0f"), num(r.get("swap_used_mb"), "%.0f"),
                num(r.get("pswpout_delta"), "%.0f"), num(z.get("ratio"), "%.1f"),
                ("yes" if r.get("oom") else "no")))
    if failed:
        L += ["", "### zram: failed workloads", "", "| config | model | error |", "|-|-|-|"]
        L += ["| %s | %s | %s |" % f for f in failed]
    if skipped:
        L += ["", "Skipped configs: " + "; ".join(skipped)]
    return L

def render_toolcall(run_dir):
    """toolcall/ holds one record per chat model benched as a hybrid tool-caller (toolcall.py);
    its schema differs from the slot records, so it gets its own table."""
    files = sorted(glob.glob(os.path.join(run_dir, "toolcall", "*.json")))
    if not files:
        return []
    L = ["", "## toolcall (chat models as hybrid tool-callers)", "",
         "| model | tool acc | none prec | none recall | cz tool acc | cached turn s | "
         "cold prefill s | prefill ms warm1/cached | RSS MB | status |", "|-|-|-|-|-|-|-|-|-|-|"]
    for p in sorted(files):
        r = _read_json(p)
        if not isinstance(r, dict) or r.get("mode") == "hybrid-e2e":   # e2e records rendered below
            continue
        h = r.get("metrics_heldout") or {}; cz = r.get("metrics_czech") or {}
        lat = r.get("latency") or {}
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s |" % (
            r.get("id"), pct(h.get("tool_accuracy")), pct(h.get("none_precision")),
            pct(h.get("none_recall")), pct(cz.get("tool_accuracy")),
            num(lat.get("cached_turn_total_s_median"), "%.1f"),
            num(r.get("cold_prefill_s"), "%.1f"),
            num(lat.get("first_turn_prompt_ms"), "%.0f"),
            num(lat.get("cached_turn_prompt_ms_median"), "%.0f"),
            num(r.get("rss_mb"), "%.0f"), r.get("status", "ok")))
    L += ["", "cold prefill s = first ever turn (no prewarm, full ~1200-token tools prefix). The "
          "daemon prewarms this on chat load in hybrid mode, so warm1 (first turn after prewarm) and "
          "cached are what a user actually sees; warm1/cached prefill ms near-equal means the "
          "system+tools prefix stayed cached (cache_prompt held)."]
    # hybrid end-to-end records (through the real daemon)
    e2e = [r for r in (_read_json(p) for p in sorted(files))
           if isinstance(r, dict) and r.get("mode") == "hybrid-e2e"]
    if e2e:
        L += ["", "### hybrid end-to-end (live daemon, dry /text)", "",
              "| model | accuracy | tool acc | none acc | fast-path | median ms | p90 ms |",
              "|-|-|-|-|-|-|-|"]
        for r in e2e:
            s = r.get("summary") or {}
            L.append("| %s | %s | %s | %s | %s (%s%%) | %s | %s |" % (
                r.get("id"), pct(s.get("accuracy")), pct(s.get("tool_accuracy")),
                pct(s.get("none_accuracy")), num(s.get("fast_path"), "%d"),
                num(s.get("fast_path_pct"), "%d"), num(s.get("latency_ms_median"), "%d"),
                num(s.get("latency_ms_p90"), "%d")))
    return L

def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: report.py <run_dir>")
    run_dir = sys.argv[1]
    slots = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(run_dir, "*")))
             if os.path.isdir(d)]
    L = ["# heytux model benchmark - %s" % os.path.basename(run_dir.rstrip("/")), "",
         "One record per model; see the per-slot JSON under this dir for full detail.", ""]
    for slot in ["chat", "router", "embedding", "asr", "tts", "vad", "vision", "wake"]:
        if slot in slots:
            L += render_slot(run_dir, slot)
    if "toolcall" in slots:
        L += render_toolcall(run_dir)
    if "zram" in slots:
        L += render_zram(run_dir)
    out = os.path.join(run_dir, "REPORT.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
