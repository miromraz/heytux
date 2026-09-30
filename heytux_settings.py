#!/usr/bin/env python3
# heytux_settings.py - GTK4/libadwaita settings app for the heytux voice daemon.
# Talks to the daemon over HTTP (HEYTUX_URL, default http://127.0.0.1:8770). Fully schema-driven:
# GET /config hands us the field list, so new settings appear with no code change here.
# stdlib + gi only. All network I/O runs on a worker thread; results marshal back via GLib.idle_add.
import os, sys, json, glob, time, threading, subprocess
import urllib.request, urllib.error

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GLib, Gio  # noqa: E402

BASE = os.environ.get("HEYTUX_URL", "http://127.0.0.1:8770").rstrip("/")
CAPTURE_DIR = os.environ.get("HEYTUX_CAPTURES", os.path.expanduser("~/heytux/captures"))

# Friendly group titles + display order. A group the schema uses but we don't list here still
# renders (appended, titled by its raw name) - the UI never hides a field it doesn't recognise.
GROUPS = [("wake", "Wake word"), ("listening", "Listening"),
          ("routing", "Routing"), ("voice", "Voice and models"), ("advanced", "Advanced")]


def http(method, path, data=None, timeout=12):
    """(status, parsed-json). Raises on a transport error; an HTTP 4xx/5xx comes back as a tuple."""
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def digits_for(step):
    s = repr(float(step))
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def enum_label(v):
    """Show the basename of a path-like enum value; leave a plain word alone."""
    return os.path.basename(v.rstrip("/")) if isinstance(v, str) and "/" in v else v


class Ctl:
    """One settings row: the widget, a blocked setter, a getter, and its reset button."""
    def __init__(self, field, row, get, set_blocked, reset_btn=None, enum_values=None):
        self.field, self.row, self.get, self.set_blocked = field, row, get, set_blocked
        self.reset_btn, self.enum_values = reset_btn, enum_values


class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Hey Tux")
        self.set_default_size(360, 740)
        self.values, self.defaults, self.pending = {}, {}, []
        self.ctl = {}
        self._built = False
        self._debounce = {}

        self.toasts = Adw.ToastOverlay()
        self.set_content(self.toasts)

        tv = Adw.ToolbarView()
        self.toasts.set_child(tv)
        header = Adw.HeaderBar()
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Refresh")
        refresh.connect("clicked", lambda *_: self.refresh())
        header.pack_start(refresh)
        tv.add_top_bar(header)

        self.banner = Adw.Banner(button_label="Restart")
        self.banner.connect("button-clicked", self._on_restart)
        tv.add_top_bar(self.banner)

        self.stack = Gtk.Stack()
        tv.set_content(self.stack)

        # --- daemon-down page ---
        self.down = Adw.StatusPage(icon_name="network-offline-symbolic",
                                   title="Hey Tux is not running",
                                   description="It does not start by itself")
        self.start_btn = Gtk.Button(label="Start", halign=Gtk.Align.CENTER,
                                    css_classes=["pill", "suggested-action"])
        self.start_btn.connect("clicked", self._on_start)
        self.down.set_child(self.start_btn)
        self.stack.add_named(self.down, "down")

        # --- main preferences page ---
        self.page = Adw.PreferencesPage()
        self.stack.add_named(self.page, "main")
        self.stack.set_visible_child_name("down")

        self.refresh()

    # ---------- background call plumbing ----------
    def _bg(self, fn, cb):
        def run():
            try:
                res = fn()
            except Exception as e:
                res = e
            GLib.idle_add(cb, res)
        threading.Thread(target=run, daemon=True).start()

    def toast(self, text):
        self.toasts.add_toast(Adw.Toast(title=text, timeout=4))

    # ---------- refresh ----------
    def refresh(self):
        self._bg(lambda: (http("GET", "/config"), http("GET", "/status")), self._on_refresh)

    def _on_refresh(self, res):
        if isinstance(res, Exception) or res[0][0] != 200:
            self.stack.set_visible_child_name("down")
            return
        (_, cfg), (scode, status) = res
        self.values = cfg.get("values", {})
        self.defaults = cfg.get("defaults", {})
        self.pending = cfg.get("pending_restart", [])
        schema = cfg.get("schema", [])
        st = status if scode == 200 else {}
        if not self._built:
            self._build(schema)
            self._built = True
        self._sync()
        self._update_status(st)
        self._update_banner()
        self._load_recent()
        self.stack.set_visible_child_name("main")

    # ---------- build the schema-driven page (once) ----------
    def _build(self, schema):
        # status group (wake switch + hit count) at the top
        g = Adw.PreferencesGroup(title="Status")
        self.wake_row = Adw.SwitchRow(title="Wake word",
                                      subtitle="Listen for “hey tux”")
        self._wake_h = self.wake_row.connect("notify::active", self._on_wake)
        g.add(self.wake_row)
        self.hits_row = Adw.ActionRow(title="Wake hits", subtitle="–")
        g.add(self.hits_row)
        run_row = Adw.ActionRow(title="Running", subtitle="Stop the assistant and its chat model")
        stop = Gtk.Button(label="Stop", valign=Gtk.Align.CENTER, css_classes=["destructive-action"])
        stop.connect("clicked", self._on_stop)
        run_row.add_suffix(stop)
        g.add(run_row)
        self.page.add(g)

        # Try it
        tg = Adw.PreferencesGroup(title="Try it")
        self.try_row = Adw.ActionRow(title="Run a voice turn",
                                     subtitle="Speak a command after you press the button")
        self.try_spinner = Gtk.Spinner()
        self.try_btn = Gtk.Button(label="Listen", valign=Gtk.Align.CENTER)
        self.try_btn.connect("clicked", self._on_try)
        self.try_row.add_suffix(self.try_spinner)
        self.try_row.add_suffix(self.try_btn)
        tg.add(self.try_row)
        self.try_result = Adw.ActionRow(title="", subtitle="", visible=False)
        self.try_result.add_css_class("property")
        tg.add(self.try_result)
        self.page.add(tg)

        # schema groups, ordered
        order = [k for k, _ in GROUPS]
        by_group = {}
        for f in schema:
            if f.get("key") == "WAKE_ON":
                continue  # shown as the top status switch (POST /wake), not duplicated here
            by_group.setdefault(f.get("group", "advanced"), []).append(f)
        seen = set()
        for key, title in GROUPS + [(k, k.title()) for k in by_group if k not in order]:
            if key in seen or key not in by_group:
                continue
            seen.add(key)
            grp = Adw.PreferencesGroup(title=title)
            for f in by_group[key]:
                self._add_row(grp, f)
            self.page.add(grp)

        # Recent
        self.recent_group = Adw.PreferencesGroup(title="Recent")
        self.page.add(self.recent_group)

    def _add_row(self, grp, f):
        key, t = f.get("key"), f.get("type")
        subtitle = f.get("help", "")
        ap = f.get("apply")
        if ap == "restart":
            subtitle += "\n(applies after restarting heytux)"
        elif ap == "model":
            subtitle += "\n(applies after the model reloads)"
        try:
            if t == "bool":
                ctl = self._row_bool(grp, f, subtitle)
            elif t in ("int", "float"):
                ctl = self._row_num(grp, f, subtitle)
            elif t == "enum":
                ctl = self._row_enum(grp, f, subtitle)
            elif t == "text":
                ctl = self._row_text(grp, f, subtitle)
            else:
                ctl = self._row_unknown(grp, f, subtitle)
        except Exception as e:            # never let one odd field break the whole page
            ctl = self._row_unknown(grp, f, subtitle + "\n[unrenderable: %r]" % e)
        self.ctl[key] = ctl

    def _reset_btn(self, key, row):
        btn = Gtk.Button(icon_name="edit-undo-symbolic", valign=Gtk.Align.CENTER,
                         tooltip_text="Reset to default", visible=False,
                         css_classes=["flat"])
        btn.connect("clicked", lambda *_: self.commit(key, self.defaults.get(key)))
        row.add_suffix(btn)
        return btn

    def _row_bool(self, grp, f, sub):
        row = Adw.SwitchRow(title=f.get("label", f["key"]), subtitle=sub)
        rb = self._reset_btn(f["key"], row)
        h = row.connect("notify::active", lambda *_: self.commit(f["key"], row.get_active()))

        def setb(v):
            row.handler_block(h); row.set_active(bool(v)); row.handler_unblock(h)
        grp.add(row)
        return Ctl(f, row, row.get_active, setb, rb)

    def _row_num(self, grp, f, sub):
        d = 0 if f["type"] == "int" else digits_for(f.get("step", 1))
        adj = Gtk.Adjustment(lower=f["min"], upper=f["max"],
                             step_increment=f.get("step", 1),
                             page_increment=f.get("step", 1) * 10)
        row = Adw.SpinRow(title=f.get("label", f["key"]), subtitle=sub, adjustment=adj, digits=d)
        rb = self._reset_btn(f["key"], row)
        h = adj.connect("value-changed", lambda *_: self._debounced(f["key"], adj.get_value()))

        def setn(v):
            adj.handler_block(h); adj.set_value(float(v)); adj.handler_unblock(h)
        grp.add(row)
        get = (lambda: int(adj.get_value())) if f["type"] == "int" else adj.get_value
        return Ctl(f, row, get, setn, rb)

    def _row_enum(self, grp, f, sub):
        vals = list(f.get("values") or [])
        cur = self.values.get(f["key"], f.get("default"))
        if cur is not None and cur not in vals:
            vals.append(cur)
        model = Gtk.StringList()
        for v in vals:
            model.append(str(enum_label(v)))
        row = Adw.ComboRow(title=f.get("label", f["key"]), subtitle=sub, model=model)
        rb = self._reset_btn(f["key"], row)

        def on_sel(*_):
            i = row.get_selected()
            if 0 <= i < len(vals):
                self.commit(f["key"], vals[i])
        h = row.connect("notify::selected", on_sel)

        def sete(v):
            row.handler_block(h)
            if v in vals:
                row.set_selected(vals.index(v))
            row.handler_unblock(h)
        grp.add(row)
        return Ctl(f, row, lambda: vals[row.get_selected()], sete, rb, enum_values=vals)

    def _row_text(self, grp, f, sub):
        row = Adw.EntryRow(title=f.get("label", f["key"]), show_apply_button=True)
        row.set_tooltip_text(sub)
        rb = self._reset_btn(f["key"], row)
        h = row.connect("apply", lambda *_: self.commit(f["key"], row.get_text()))

        def sett(v):
            row.handler_block(h); row.set_text("" if v is None else str(v)); row.handler_unblock(h)
        grp.add(row)
        return Ctl(f, row, row.get_text, sett, rb)

    def _row_unknown(self, grp, f, sub):
        row = Adw.ActionRow(title=f.get("label", f.get("key", "?")), subtitle=sub)
        val = Gtk.Label(label=str(self.values.get(f.get("key"), "")), valign=Gtk.Align.CENTER,
                        css_classes=["dim-label"])
        row.add_suffix(val)
        grp.add(row)
        return Ctl(f, row, lambda: None, lambda v: val.set_text(str(v)))

    # ---------- sync widgets from self.values ----------
    def _sync(self):
        for key, c in self.ctl.items():
            if key in self.values:
                c.set_blocked(self.values[key])
            if c.reset_btn is not None:
                c.reset_btn.set_visible(key in self.defaults
                                        and self.values.get(key) != self.defaults.get(key))

    def _update_status(self, st):
        on = self.values.get("WAKE_ON", st.get("wake", False))
        self.wake_row.handler_block(self._wake_h)
        self.wake_row.set_active(bool(on))
        self.wake_row.handler_unblock(self._wake_h)
        hits = st.get("wake_hits")
        self.hits_row.set_subtitle("–" if hits is None else str(hits))

    def _update_banner(self):
        if self.pending:
            labels = [self.ctl[k].field.get("label", k) if k in self.ctl else k
                      for k in self.pending]
            self.banner.set_title("Restart heytux to apply: " + ", ".join(labels))
            self.banner.set_revealed(True)
        else:
            self.banner.set_revealed(False)

    # ---------- apply a change ----------
    def _debounced(self, key, value):
        src = self._debounce.pop(key, None)
        if src:
            GLib.source_remove(src)
        self._debounce[key] = GLib.timeout_add(400, self._fire, key, value)

    def _fire(self, key, value):
        self._debounce.pop(key, None)
        self.commit(key, value)
        return False

    def commit(self, key, value):
        self._bg(lambda: http("POST", "/config", {key: value}),
                 lambda res: self._on_commit(key, res))

    def _on_commit(self, key, res):
        if isinstance(res, Exception):
            self.toast("Could not reach heytux")
            self._sync()
            return
        status, body = res
        if status == 200:
            self.values.update(body.get("values", {}))
            self.pending = body.get("pending_restart", self.pending)
            self._sync()
            self._update_banner()
        else:
            msg = (body.get("errors", {}) or {}).get(key) or "invalid value"
            self.toast(self.ctl[key].field.get("label", key) + ": " + msg)
            self._sync()  # revert the row to the daemon's value

    def _on_wake(self, *_):
        on = self.wake_row.get_active()
        self._bg(lambda: http("POST", "/wake", {"on": on}), lambda res: None)

    # ---------- try it ----------
    def _on_try(self, *_):
        self.try_btn.set_sensitive(False)
        self.try_spinner.start()
        self.try_result.set_visible(False)
        self._bg(lambda: http("POST", "/listen", {}, timeout=30), self._on_try_done)

    def _on_try_done(self, res):
        self.try_btn.set_sensitive(True)
        self.try_spinner.stop()
        if isinstance(res, Exception) or res[0] != 200:
            self.toast("Voice turn failed")
            return
        r = res[1]
        tx = (r.get("transcript") or "").strip()
        route = r.get("route") or "?"
        calls = r.get("calls") or []
        tool = calls[0].get("name") if calls and isinstance(calls[0], dict) else None
        conf = r.get("confidence")
        said = r.get("said") or ""
        detail = "route: %s%s" % (route, ("  •  " + tool) if tool else "")
        if conf is not None:
            detail += "  •  conf %.2f" % conf
        if said:
            detail += "\nsaid: " + said
        self.try_result.set_title(tx or "(no speech)")
        self.try_result.set_subtitle(detail)
        self.try_result.set_visible(True)
        self._load_recent()

    # ---------- recent captures (local JSON files) ----------
    def _load_recent(self):
        rows = getattr(self, "_recent_rows", [])
        for r in rows:
            self.recent_group.remove(r)
        self._recent_rows = []
        files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.json")))[-10:][::-1]
        if not files:
            row = Adw.ActionRow(title="No recent captures", css_classes=["dim-label"])
            self.recent_group.add(row)
            self._recent_rows.append(row)
            return
        for path in files:
            try:
                with open(path) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            tx = (d.get("transcript") or "").strip() or "(no speech)"
            calls = d.get("calls") or []
            tool = calls[0].get("name") if calls and isinstance(calls[0], dict) else None
            sub = tool or (d.get("route") or "")
            conf = d.get("confidence")
            if conf is not None:
                sub = ("%s  •  %.2f" % (sub, conf)).strip("  •  ")
            row = Adw.ActionRow(title=tx, subtitle=sub)
            self.recent_group.add(row)
            self._recent_rows.append(row)

    # ---------- restart ----------
    def _on_restart(self, *_):
        self.banner.set_title("Restarting heytux…")
        self.banner.set_button_label("")

        def work():
            subprocess.run(["systemctl", "--user", "restart", "heytux"], timeout=20)
            return wait_up()
        self._bg(work, self._on_restarted)

    def _on_restarted(self, ok):
        self.banner.set_button_label("Restart")
        self.toast("Hey Tux restarted" if ok is True else "Restart timed out")
        self.refresh()

    # ---------- start / stop (the units are not enabled, so nothing starts at login) ----------
    def _on_start(self, *_):
        self.start_btn.set_sensitive(False)
        self.start_btn.set_label("Starting…")

        def work():
            # heytux.service Wants= heytux-llm.service, so this brings up both
            subprocess.run(["systemctl", "--user", "start", "heytux"], timeout=20)
            return wait_up()
        self._bg(work, self._on_started)

    def _on_started(self, ok):
        self.start_btn.set_sensitive(True)
        self.start_btn.set_label("Start")
        if ok is not True:
            self.toast("Hey Tux did not start")
        self.refresh()

    def _on_stop(self, *_):
        self._bg(lambda: subprocess.run(["systemctl", "--user", "stop", "heytux", "heytux-llm"],
                                        timeout=30),
                 lambda res: self.refresh())


def wait_up(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if http("GET", "/config", timeout=3)[0] == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


class HeyTuxApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="org.bengalos.HeyTuxSettings",
                         flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.win = None

    def do_activate(self):
        if not self.win:
            self.win = Window(self)
        self.win.present()


def main():
    return HeyTuxApp().run(None)


if __name__ == "__main__":
    sys.exit(main())
