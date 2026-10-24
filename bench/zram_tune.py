#!/usr/bin/env python3
# zram_tune.py - on-phone zram/swap tuning sweep for heytux (run as ROOT). For each config it
# reconfigures zram (algo + disksize), sets vm.swappiness / vm.page-cluster, drops caches, loads
# Qwen3.5-4B then Gemma-4-E2B via llama-server (-t 2 -c 2048) and records load time, tg tok/s, TTFT,
# MemAvailable, pswpin/pswpout delta, zram orig/compr, and any OOM kill. The ORIGINAL config is
# captured at start and restored in a finally, always. Results -> <out>/zram/<label>.json.
#
# Changing the algorithm needs swapoff + reset; we refuse swapoff unless MemAvailable covers the
# swap in use (pages come back to RAM). Nothing is persisted: the on-disk config file that WOULD
# persist this is noted in the report, not written.
import gc, json, os, signal, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends

ZRAM = "/sys/block/zram0"
SWAPDEV = "/dev/zram0"
MODELS = [
    {"id": "Qwen3.5-4B-Q4_K_M", "slot": "chat", "runtime": "llama.cpp",
     "path": "/home/miro/models/chat/Qwen3.5-4B-Q4_K_M/Qwen3.5-4B-Q4_K_M.gguf"},
    {"id": "Gemma-4-E2B-it-Q4_K_M", "slot": "chat", "runtime": "llama.cpp",
     "path": "/home/miro/models/chat/Gemma-4-E2B-it-Q4_K_M/gemma-4-E2B-it-Q4_K_M.gguf"},
]
G4 = 4 * 1024**3
G27 = 2891448320
# Only algorithms the kernel already exposes can be used (no modprobe permitted). lz4/lz4hc are kept
# in the list so the report records them as unavailable rather than silently omitting them.
CONFIGS = [
    {"label": "zram-zstd-2.7g-s60-pc3",  "algo": "zstd",  "size": G27, "sw": 60,  "pc": 3},
    {"label": "zram-zstd-2.7g-s100-pc3", "algo": "zstd",  "size": G27, "sw": 100, "pc": 3},
    {"label": "zram-zstd-4g-s100-pc3",   "algo": "zstd",  "size": G4,  "sw": 100, "pc": 3},
    {"label": "zram-zstd-4g-s180-pc0",   "algo": "zstd",  "size": G4,  "sw": 180, "pc": 0},
    {"label": "zram-zstd-4g-s60-pc0",    "algo": "zstd",  "size": G4,  "sw": 60,  "pc": 0},
    {"label": "zram-lz4-4g-s100-pc3",    "algo": "lz4",   "size": G4,  "sw": 100, "pc": 3},
]

class _Stop(BaseException): pass   # BaseException so `except Exception` cannot swallow it; raised by a signal handler so restore runs exactly once in finally

def log(*a): print("[zram]", *a, file=sys.stderr, flush=True)
def rd(p):
    with open(p) as f: return f.read().strip()
def wr(p, v):
    with open(p, "w") as f: f.write(str(v))   # explicit close: sysfs commits on close, no EBUSY at GC
def sh(cmd): return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()
def shc(argv):
    """Run an argv list, raising on nonzero (mkswap/swapon/swapoff must not fail silently)."""
    subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def done_file(path):
    """A config result counts as done only if it exists AND parses (not a truncated write)."""
    if not os.path.exists(path):
        return False
    try:
        json.load(open(path)); return True
    except Exception:
        return False

def meminfo():
    o = {}
    for ln in open("/proc/meminfo"):
        k, _, v = ln.partition(":")
        if k in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"):
            o[k] = int(v.split()[0])
    return o

def vmstat():
    o = {}
    for ln in open("/proc/vmstat"):
        k, _, v = ln.partition(" ")
        if k in ("pswpin", "pswpout"):
            o[k] = int(v)
    return o

def zram_mm():
    mm = rd(ZRAM + "/mm_stat").split()
    return {"orig_bytes": int(mm[0]), "compr_bytes": int(mm[1]),
            "ratio": round(int(mm[0]) / max(int(mm[1]), 1), 2)}

def _swap_active():
    return any(ln.startswith(SWAPDEV) for ln in open("/proc/swaps").read().splitlines()[1:])

def current_config():
    algo = rd(ZRAM + "/comp_algorithm")
    import re
    algo = (re.search(r"\[(\w+)\]", algo) or [None, algo])[1]
    return {"algo": algo, "size": int(rd(ZRAM + "/disksize")),
            "sw": int(rd("/proc/sys/vm/swappiness")), "pc": int(rd("/proc/sys/vm/page-cluster")),
            "prio": _swap_prio(), "active": _swap_active()}

def _swap_prio():
    for ln in open("/proc/swaps").read().splitlines()[1:]:
        if ln.startswith(SWAPDEV):
            return int(ln.split()[-1])
    return 100

def available_algos():
    return [a.strip("[]") for a in rd(ZRAM + "/comp_algorithm").split()]

def apply_config(cfg, prio=100, activate=True):
    """Reconfigure zram and vm settings. mkswap/swapon/swapoff are checked. If activate is False the
    swap device is left off (used by restore when the original device was inactive). Verifies the
    effective settings before returning so a measurement never runs on a half-applied config."""
    if cfg["algo"] not in available_algos():
        raise RuntimeError("algo %s not supported (have %s)" % (cfg["algo"], available_algos()))
    used_kb = meminfo()["SwapTotal"] - meminfo()["SwapFree"]
    avail_kb = meminfo()["MemAvailable"]
    if _swap_active() and used_kb > 0 and avail_kb < used_kb + 300_000:
        raise RuntimeError("refuse swapoff: swap used %dMB > MemAvailable %dMB" %
                           (used_kb // 1024, avail_kb // 1024))
    if _swap_active():
        shc(["swapoff", SWAPDEV])
    wr(ZRAM + "/reset", 1)
    wr(ZRAM + "/comp_algorithm", cfg["algo"])
    wr(ZRAM + "/disksize", str(cfg["size"]))
    wr("/proc/sys/vm/swappiness", str(cfg["sw"]))
    wr("/proc/sys/vm/page-cluster", str(cfg["pc"]))
    if activate:
        shc(["mkswap", SWAPDEV])
        shc(["swapon", "-p", str(prio), SWAPDEV])
    eff = current_config()
    bad = [k for k in ("algo", "size", "sw", "pc") if eff[k] != cfg[k]]
    if bad or (activate and not eff["active"]):
        raise RuntimeError("verify failed for %s: want %s got %s (bad=%s)" %
                           (cfg.get("label"), cfg, eff, bad))
    log("applied %s: %s" % (cfg.get("label"), eff))

def oom_since(t0):
    out = sh("journalctl -k --since @%d 2>/dev/null | grep -iE 'out of memory|killed process|oom-kill' | tail -5" % int(t0))
    return out.strip() or None

def workload(label, scenario_out):
    recs = []
    for m in MODELS:
        t0 = time.time()
        sh("sync; echo 3 > /proc/sys/vm/drop_caches")
        vm0 = vmstat()
        rec = {"label": label, "model": m["id"]}
        try:
            t = time.time()
            # root loads the server; llama-server itself is self-contained (no sherpa env needed).
            be = backends.LlamaServer(m, threads=2, ctx=2048)
            rec["load_s"] = round(time.time() - t, 2)
            r = be.chat([{"role": "user", "content": "Explain gravity in two sentences."}],
                        max_tokens=64, temperature=0.7)
            rec["ttft_s"] = round(r["ttft_s"], 3) if r["ttft_s"] else None
            rec["tg_tok_s"] = r["tok_s"]
            mi = meminfo()
            rec["mem_avail_mb"] = round(mi["MemAvailable"] / 1024)
            rec["swap_used_mb"] = round((mi["SwapTotal"] - mi["SwapFree"]) / 1024)
            vm1 = vmstat()
            rec["pswpin_delta"] = vm1["pswpin"] - vm0["pswpin"]
            rec["pswpout_delta"] = vm1["pswpout"] - vm0["pswpout"]
            rec["zram"] = zram_mm()
            be.close(); del be; gc.collect()
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "failed"; rec["error"] = repr(e)[:200]
            backends._close_all()
        rec["oom"] = oom_since(t0)
        recs.append(rec)
        log("%s / %s: %s" % (label, m["id"], {k: rec.get(k) for k in
            ("load_s", "tg_tok_s", "ttft_s", "mem_avail_mb", "swap_used_mb", "pswpout_delta", "status")}))
    out = os.path.join(scenario_out, "zram")
    os.makedirs(out, exist_ok=True)
    write_json_atomic(os.path.join(out, label + ".json"), recs)
    return recs

def restore(orig):
    """Restore the original config, always. Close any loaded server first (swapoff refuses while a
    model is resident), restore swappiness/page-cluster even if the zram reset fails, and return True
    only if every original setting is back."""
    import backends as _b
    _b._close_all()
    log("restoring ORIGINAL %s" % orig)
    try:
        apply_config({"label": "restore", **orig}, prio=orig["prio"], activate=orig.get("active", True))
    except Exception as e:
        log("zram restore via apply failed: %r; forcing swappiness/page-cluster" % e)
        try: wr("/proc/sys/vm/swappiness", str(orig["sw"]))
        except Exception as e2: log("swappiness restore failed: %r" % e2)
        try: wr("/proc/sys/vm/page-cluster", str(orig["pc"]))
        except Exception as e2: log("page-cluster restore failed: %r" % e2)
    eff = current_config()
    ok = all(eff[k] == orig[k] for k in ("algo", "size", "sw", "pc")) and eff["active"] == orig.get("active", True)
    log("restore %s -> effective %s" % ("OK" if ok else "INCOMPLETE", eff))
    return ok

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/miro/heytux/bench/results/zram"
    orig = current_config()
    log("ORIGINAL config: %s" % orig)
    json.dump(orig, open("/tmp/zram_orig.json", "w"))
    def _sig(signum, frame):
        log("signal %d: stopping, will restore once in finally" % signum)
        raise _Stop()
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _sig)
    only = sys.argv[2] if len(sys.argv) > 2 else None   # optional single-config label for a smoke check
    stopped = False
    try:
        for cfg in CONFIGS:
            if only and cfg["label"] != only:
                continue
            lbl = os.path.join(out, "zram", cfg["label"] + ".json")
            if done_file(lbl):
                log("skip (done): %s" % cfg["label"]); continue
            try:
                apply_config(cfg, prio=orig["prio"])
            except Exception as e:
                log("apply skipped %s: %r" % (cfg["label"], e))
                d = os.path.join(out, "zram"); os.makedirs(d, exist_ok=True)
                write_json_atomic(os.path.join(d, cfg["label"] + ".json"),
                                  [{"label": cfg["label"], "status": "skipped", "reason": repr(e)[:200]}])
                continue
            workload(cfg["label"], out)
    except _Stop:
        stopped = True
    finally:
        # restore EXACTLY once, with signals ignored so a second signal can't re-enter it
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        restored = restore(orig)
        log("done. Persist file (NOT written): /etc/systemd/zram-generator.conf or the zram udev rule.")
    if not restored:
        sys.exit(2)       # restore incomplete
    if stopped:
        sys.exit(130)     # stopped by signal, but cleanly restored

if __name__ == "__main__":
    main()
