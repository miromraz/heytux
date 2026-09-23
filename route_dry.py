#!/usr/bin/env python3
# Runs ON the phone. Reads a JSON list of transcripts on stdin, routes each through the LIVE
# heytux daemon /text with dry:true (decides route, fires nothing), prints {transcript: result}.
import sys, json, urllib.request
out = {}
for t in json.load(sys.stdin):
    body = json.dumps({"text": t, "dry": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:8770/text", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=20).read())
        out[t] = {"route": r.get("route"),
                  "calls": [{"name": c.get("name"), "arguments": c.get("arguments")}
                            for c in (r.get("calls") or [])]}
    except Exception as e:
        out[t] = {"error": "%s: %s" % (type(e).__name__, e)}
print(json.dumps(out))
