"""dialogs.py - secondary windows for FontStudio (Tkinter)."""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

import fontcore as fc
import photoimport as pi


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def num(var: tk.Variable, default: float, cast: Callable[[Any], Any] = float) -> Any:
    try:
        return cast(var.get())
    except (tk.TclError, ValueError):
        return default


def spin(parent: tk.Misc, var: tk.Variable, lo: float, hi: float, inc: float = 1.0,
         width: int = 7, fmt: Optional[str] = None) -> ttk.Spinbox:
    kw: Dict[str, Any] = {}
    if fmt:
        kw["format"] = fmt
    return ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, textvariable=var, width=width, **kw)


def ui_font(size: int) -> Any:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def center_on(win: tk.Toplevel, parent: tk.Misc, w: int, h: int) -> None:
    parent.update_idletasks()
    x = parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)
    y = parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 2)
    win.geometry("%dx%d+%d+%d" % (w, h, x, y))


def run_task(parent: tk.Misc, title: str, work: Callable[[Callable[[int, int], None]], Any],
             on_done: Callable[[Any, Optional[BaseException]], None]) -> None:
    """Run `work(progress)` in a thread behind a small progress window."""
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.resizable(False, False)
    dlg.transient(parent)  # type: ignore[arg-type]
    ttk.Label(dlg, text=title + " ...", padding=(16, 14, 16, 4)).pack()
    pb = ttk.Progressbar(dlg, mode="determinate", length=340, maximum=100)
    pb.pack(padx=16, pady=(0, 16))
    center_on(dlg, parent, 380, 100)
    dlg.protocol("WM_DELETE_WINDOW", lambda: None)
    try:
        dlg.grab_set()
    except tk.TclError:
        pass
    q: "queue.Queue[tuple]" = queue.Queue()

    def progress(i: int, n: int) -> None:
        q.put(("p", i, n))

    def target() -> None:
        try:
            res = work(progress)
            q.put(("done", res, None))
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()
            q.put(("done", None, e))

    threading.Thread(target=target, daemon=True).start()

    def poll() -> None:
        try:
            while True:
                item = q.get_nowait()
                if item[0] == "p":
                    pb["value"] = 100.0 * item[1] / max(1, item[2])
                else:
                    dlg.grab_release()
                    dlg.destroy()
                    on_done(item[1], item[2])
                    return
        except queue.Empty:
            pass
        dlg.after(50, poll)

    poll()


# --------------------------------------------------------------------------
# Photo import dialog
# --------------------------------------------------------------------------
MODE_NAMES = {
    "Template sheet (printed from this program)": "template",
    "Plain grid (rows x columns, one character per cell)": "grid",
    "Auto-detect characters (free layout / handwriting lines)": "auto",
    "Single character photo": "single",
}
PLACEMENT_NAMES = {
    "Uniform scale, baseline per line (keeps your proportions)": "uniform",
    "Fit height by letter type (normalises sloppy sizes)": "class",
    "Fit every shape to the glyph box": "fit",
}
THRESH_NAMES = {"Automatic (Otsu)": "otsu", "Adaptive (uneven light)": "adaptive",
                "Manual": "manual"}


def _inv(d: Dict[str, str], value: str) -> str:
    for k, v in d.items():
        if v == value:
            return k
    return next(iter(d))


class PhotoDialog(tk.Toplevel):
    def __init__(self, app: Any, mode: str = "auto", chars: str = "",
                 placement: str = "uniform") -> None:
        super().__init__(app)
        self.app = app
        self.title("Import glyphs from a photo")
        self.geometry("1180x780")
        self.minsize(980, 640)
        self.transient(app)
        self.det: Optional[pi.Detection] = None
        self.src: Optional[Any] = None
        self.src_path = ""
        self._after: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._scale = 1.0
        self._off = (0, 0)
        self.result_masks: Dict[int, np.ndarray] = {}
        dp = pi.PhotoParams()
        self.v_mode = tk.StringVar(value=_inv(MODE_NAMES, mode))
        self.v_place = tk.StringVar(value=_inv(PLACEMENT_NAMES, placement))
        self.v_rows = tk.IntVar(value=app.tpl_rows)
        self.v_cols = tk.IntVar(value=app.tpl_cols)
        self.v_page = tk.IntVar(value=1)
        self.v_thm = tk.StringVar(value=_inv(THRESH_NAMES, dp.threshold_method))
        self.v_thr = tk.IntVar(value=dp.threshold)
        self.v_ab = tk.IntVar(value=dp.adaptive_block)
        self.v_ac = tk.IntVar(value=dp.adaptive_c)
        self.v_inv = tk.BooleanVar(value=dp.invert)
        self.v_norm = tk.BooleanVar(value=dp.normalize_light)
        self.v_desk = tk.BooleanVar(value=dp.deskew)
        self.v_blur = tk.DoubleVar(value=dp.blur)
        self.v_min = tk.IntVar(value=dp.min_area)
        self.v_close = tk.IntVar(value=dp.close_px)
        self.v_dots = tk.BooleanVar(value=dp.merge_dots)
        self.v_gap = tk.IntVar(value=dp.merge_gap)
        self.v_margin = tk.DoubleVar(value=dp.cell_margin_pct)
        self.v_crop = tk.BooleanVar(value=dp.crop_to_ink)
        self.v_over = tk.BooleanVar(value=True)
        self._build(chars)
        self.after(50, self._grab)
        self._mode_changed()

    def _grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:
            pass

    # ---- UI
    def _build(self, chars: str) -> None:
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)
        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)
        self.view = tk.Canvas(left, bg="#303030", highlightthickness=0, cursor="hand2")
        self.view.pack(fill="both", expand=True)
        self.view.bind("<Configure>", lambda e: self._redraw())
        self.view.bind("<Button-1>", self._click)
        self.view.bind("<Button-3>", self._rclick)
        self.msg = tk.StringVar(value="Choose a photo to begin.")
        ttk.Label(left, textvariable=self.msg, wraplength=700, padding=(2, 6)).pack(fill="x")
        ttk.Label(left, foreground="#555555", wraplength=700,
                  text="Auto/grid review: left-click a box to discard it (noise, smudges); right-click a "
                       "box to merge it with the next shape.").pack(fill="x")

        right = ttk.Frame(root, padding=(8, 0, 0, 0))
        right.pack(side="right", fill="y")
        f = ttk.LabelFrame(right, text="1. Photo", padding=6)
        f.pack(fill="x")
        ttk.Button(f, text="Choose photo...", command=self.choose).pack(side="left")
        self.lbl_file = ttk.Label(f, text="  (none)", width=28)
        self.lbl_file.pack(side="left")

        f = ttk.LabelFrame(right, text="2. What kind of picture is it?", padding=6)
        f.pack(fill="x", pady=4)
        cb = ttk.Combobox(f, textvariable=self.v_mode, values=list(MODE_NAMES), state="readonly", width=52)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self._mode_changed())
        g = ttk.Frame(f)
        g.pack(fill="x", pady=(4, 0))
        ttk.Label(g, text="Rows").grid(row=0, column=0, sticky="w")
        self.sp_rows = spin(g, self.v_rows, 1, 30, width=5)
        self.sp_rows.grid(row=0, column=1, padx=4)
        ttk.Label(g, text="Columns").grid(row=0, column=2, sticky="w")
        self.sp_cols = spin(g, self.v_cols, 1, 30, width=5)
        self.sp_cols.grid(row=0, column=3, padx=4)
        ttk.Label(g, text="Template page").grid(row=0, column=4, sticky="w")
        self.sp_page = spin(g, self.v_page, 1, 20, width=4)
        self.sp_page.grid(row=0, column=5, padx=4)
        ttk.Label(f, text="Size / placement of the shapes:").pack(anchor="w", pady=(6, 0))
        self.cb_place = ttk.Combobox(f, textvariable=self.v_place, values=list(PLACEMENT_NAMES),
                                     state="readonly", width=52)
        self.cb_place.pack(fill="x")

        f = ttk.LabelFrame(right, text="3. Which characters are in the photo? (in reading order)", padding=6)
        f.pack(fill="x", pady=4)
        self.txt = tk.Text(f, height=5, width=48, wrap="char", font=("Consolas", 11))
        self.txt.pack(fill="x")
        self.txt.insert("1.0", chars)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        self.cb_pre = ttk.Combobox(row, values=list(fc.CHARSET_PRESETS), state="readonly", width=34)
        self.cb_pre.pack(side="left")
        self.cb_pre.set("Insert a preset...")
        self.cb_pre.bind("<<ComboboxSelected>>", self._preset)
        ttk.Button(row, text="Update", command=self.schedule).pack(side="right")
        ttk.Label(f, foreground="#555555", wraplength=420,
                  text="Press Enter for a new line when the photo has several lines of writing - the "
                       "program then matches line by line, which is far more robust.").pack(anchor="w")

        f = ttk.LabelFrame(right, text="4. Clean-up", padding=6)
        f.pack(fill="x", pady=4)
        g = ttk.Frame(f)
        g.pack(fill="x")
        ttk.Label(g, text="Ink detection").grid(row=0, column=0, sticky="w")
        cbt = ttk.Combobox(g, textvariable=self.v_thm, values=list(THRESH_NAMES), state="readonly", width=22)
        cbt.grid(row=0, column=1, columnspan=3, sticky="w", padx=4)
        cbt.bind("<<ComboboxSelected>>", lambda e: self.schedule())
        ttk.Label(g, text="Manual level").grid(row=1, column=0, sticky="w")
        spin(g, self.v_thr, 20, 240, 5, 5).grid(row=1, column=1, padx=4, sticky="w")
        ttk.Label(g, text="Adaptive size / offset").grid(row=1, column=2, sticky="w")
        spin(g, self.v_ab, 11, 301, 10, 5).grid(row=1, column=3, padx=2)
        spin(g, self.v_ac, -20, 60, 1, 4).grid(row=1, column=4, padx=2)
        ttk.Label(g, text="Blur").grid(row=2, column=0, sticky="w")
        spin(g, self.v_blur, 0, 5, 0.2, 5).grid(row=2, column=1, padx=4, sticky="w")
        ttk.Label(g, text="Ignore specks < px").grid(row=2, column=2, sticky="w")
        spin(g, self.v_min, 0, 2000, 10, 5).grid(row=2, column=3, padx=2)
        ttk.Label(g, text="Join broken strokes px").grid(row=3, column=0, sticky="w")
        spin(g, self.v_close, 0, 12, 1, 5).grid(row=3, column=1, padx=4, sticky="w")
        ttk.Label(g, text="Merge shapes closer than px").grid(row=3, column=2, sticky="w")
        spin(g, self.v_gap, 0, 80, 1, 5).grid(row=3, column=3, padx=2)
        ttk.Label(g, text="Cell edge margin %").grid(row=4, column=0, sticky="w")
        spin(g, self.v_margin, 0, 15, 0.5, 5).grid(row=4, column=1, padx=4, sticky="w")
        checks = ttk.Frame(f)
        checks.pack(fill="x", pady=(4, 0))
        for text, var in (("Even out lighting / shadows", self.v_norm), ("Straighten (deskew)", self.v_desk),
                          ("Photo is light-on-dark (invert)", self.v_inv),
                          ("Merge i-dots, accents, ! ? : ;", self.v_dots),
                          ("Grid: crop to the writing first", self.v_crop)):
            ttk.Checkbutton(checks, text=text, variable=var, command=self.schedule).pack(anchor="w")
        for v in (self.v_thr, self.v_ab, self.v_ac, self.v_blur, self.v_min, self.v_close, self.v_gap,
                  self.v_margin, self.v_rows, self.v_cols, self.v_page):
            v.trace_add("write", lambda *a: self.schedule())
        self.v_place.trace_add("write", lambda *a: self._refresh_only())

        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(bar, text="Replace glyphs that already exist", variable=self.v_over).pack(anchor="w")
        b2 = ttk.Frame(bar)
        b2.pack(fill="x", pady=4)
        ttk.Button(b2, text="Import these glyphs", command=self.do_import).pack(side="left")
        ttk.Button(b2, text="Cancel", command=self.destroy).pack(side="right")

    # ---- state helpers
    def params(self) -> pi.PhotoParams:
        return pi.PhotoParams(
            mode=MODE_NAMES.get(self.v_mode.get(), "auto"),
            placement=PLACEMENT_NAMES.get(self.v_place.get(), "uniform"),
            rows=max(1, num(self.v_rows, 9, int)), cols=max(1, num(self.v_cols, 8, int)),
            page=max(1, num(self.v_page, 1, int)),
            threshold_method=THRESH_NAMES.get(self.v_thm.get(), "otsu"),
            threshold=num(self.v_thr, 140, int), adaptive_block=num(self.v_ab, 41, int),
            adaptive_c=num(self.v_ac, 12, int), invert=bool(self.v_inv.get()),
            normalize_light=bool(self.v_norm.get()), deskew=bool(self.v_desk.get()),
            blur=num(self.v_blur, 0.8), min_area=num(self.v_min, 40, int),
            close_px=num(self.v_close, 0, int), merge_dots=bool(self.v_dots.get()),
            merge_gap=num(self.v_gap, 0, int), cell_margin_pct=num(self.v_margin, 3.0),
            crop_to_ink=bool(self.v_crop.get()))

    def chars_text(self) -> str:
        return self.txt.get("1.0", "end").rstrip("\n")

    def _mode_changed(self) -> None:
        mode = MODE_NAMES.get(self.v_mode.get(), "auto")
        grid = mode in ("template", "grid")
        for w in (self.sp_rows, self.sp_cols):
            w.state(["!disabled"] if grid else ["disabled"])
        self.sp_page.state(["!disabled"] if mode == "template" else ["disabled"])
        self.cb_place.state(["disabled"] if mode == "template" else ["!disabled"])
        self.schedule()

    def _preset(self, _e: Any = None) -> None:
        name = self.cb_pre.get()
        if name in fc.CHARSET_PRESETS:
            if self.txt.get("1.0", "end").strip():
                self.txt.insert("end", "\n")
            self.txt.insert("end", fc.CHARSET_PRESETS[name])
            self.cb_pre.set("Insert a preset...")
            self.schedule()

    def choose(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Choose a photo or scan",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.src = Image.open(path)
            self.src.load()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot open image", str(e), parent=self)
            return
        self.src_path = path
        self.lbl_file.configure(text="  " + os.path.basename(path)[:34])
        self.analyse()

    def schedule(self) -> None:
        if self.src is None:
            return
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(700, self.analyse)

    def analyse(self) -> None:
        self._after = None
        if self.src is None:
            return
        p = self.params()
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.det = pi.analyse(self.src, self.chars_text(), p, self.app.proj.s)
            self.msg.set(self.det.message)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.det = None
            self.msg.set("Analysis failed: %s" % e)
        finally:
            self.config(cursor="")
        self._redraw()

    def _refresh_only(self) -> None:
        pass

    # ---- preview drawing
    def _redraw(self) -> None:
        self.view.delete("all")
        if self.det is None or self.det.debug is None:
            return
        rgb = cv2.cvtColor(self.det.debug, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        fnt = ui_font(max(14, im.size[1] // 40))
        for it in self.det.items:
            if it.cp is not None and not it.discarded:
                d.text((it.bbox[0] + 2, it.bbox[1] + 1), chr(it.cp), font=fnt, fill=(210, 0, 90))
        cw = max(100, self.view.winfo_width())
        ch = max(100, self.view.winfo_height())
        sc = min(cw / im.size[0], ch / im.size[1], 2.0)
        self._scale = sc
        im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))), Image.Resampling.LANCZOS)
        self._off = ((cw - im.size[0]) // 2, (ch - im.size[1]) // 2)
        self._photo = ImageTk.PhotoImage(im)
        self.view.create_image(self._off[0], self._off[1], anchor="nw", image=self._photo)

    def _hit(self, ev: Any) -> Optional[int]:
        if self.det is None or not self.det.items:
            return None
        x = (ev.x - self._off[0]) / self._scale
        y = (ev.y - self._off[1]) / self._scale
        best, area = None, 1e18
        for i, it in enumerate(self.det.items):
            x0, y0, x1, y1 = it.bbox
            if x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3:
                a = (x1 - x0) * (y1 - y0)
                if a < area:
                    best, area = i, a
        return best

    def _reassign(self) -> None:
        if self.det is None or self.det.prep is None:
            return
        p = self.params()
        done, total = pi.assign_chars(self.det.items, self.chars_text(), positional=(p.mode == "grid"))
        active = len([i for i in self.det.items if not i.discarded])
        self.msg.set("%d shapes; %d characters typed; %d assigned." % (active, total, done))
        self.det.debug = pi.overlay(self.det.prep, self.det.items)
        self._redraw()

    def _click(self, ev: Any) -> None:
        i = self._hit(ev)
        if i is None or self.det is None:
            return
        self.det.items[i].discarded = not self.det.items[i].discarded
        self._reassign()

    def _rclick(self, ev: Any) -> None:
        i = self._hit(ev)
        if i is None or self.det is None or self.det.prep is None:
            return
        items = self.det.items
        if i + 1 >= len(items):
            return
        merged = pi.merge_items(self.det.prep, items[i], items[i + 1])
        items[i:i + 2] = [merged]
        self._reassign()

    # ---- result
    def do_import(self) -> None:
        if self.det is None:
            messagebox.showinfo("Nothing to import", "Choose a photo first.", parent=self)
            return
        p = self.params()
        try:
            masks = pi.finish(self.det, p, self.app.proj.s)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Import failed", str(e), parent=self)
            return
        if not masks:
            messagebox.showwarning("Nothing found", "No glyphs could be extracted with these settings.",
                                   parent=self)
            return
        self.app.apply_masks(masks, bool(self.v_over.get()))
        self.destroy()


# --------------------------------------------------------------------------
# Template sheet maker
# --------------------------------------------------------------------------
class TemplateDialog(tk.Toplevel):
    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.app = app
        self.title("Make a printable template sheet")
        self.transient(app)
        center_on(self, app, 640, 560)
        self.v_cols = tk.IntVar(value=app.tpl_cols)
        self.v_rows = tk.IntVar(value=app.tpl_rows)
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, wraplength=600, text=(
            "1) Choose the characters.  2) Save and print the pages (A4, 300 dpi).  3) Write each "
            "character in its box with a dark pen, keeping the black frame fully in view.  4) Photograph "
            "or scan the page and use Import > Photo to glyphs with 'Template sheet'.")).pack(anchor="w")
        ttk.Label(f, text="Characters on the template:").pack(anchor="w", pady=(10, 2))
        self.txt = tk.Text(f, height=6, width=60, wrap="char", font=("Consolas", 11))
        self.txt.pack(fill="x")
        self.txt.insert("1.0", "".join(chr(c) for c in app.proj.chars if c != 32))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        self.cb = ttk.Combobox(row, values=list(fc.CHARSET_PRESETS), state="readonly", width=40)
        self.cb.pack(side="left")
        self.cb.set("Replace with a preset...")
        self.cb.bind("<<ComboboxSelected>>", self._preset)
        g = ttk.Frame(f)
        g.pack(fill="x", pady=6)
        ttk.Label(g, text="Boxes per row").pack(side="left")
        spin(g, self.v_cols, 4, 12, 1, 4).pack(side="left", padx=6)
        ttk.Label(g, text="Rows per page").pack(side="left")
        spin(g, self.v_rows, 3, 16, 1, 4).pack(side="left", padx=6)
        self.lbl = ttk.Label(f, text="")
        self.lbl.pack(anchor="w")
        for v in (self.v_cols, self.v_rows):
            v.trace_add("write", lambda *a: self._count())
        self.txt.bind("<KeyRelease>", lambda e: self._count())
        self.prev = tk.Canvas(f, height=170, bg="#e8e8e8", highlightthickness=0)
        self.prev.pack(fill="x", pady=6)
        self._ph: Optional[ImageTk.PhotoImage] = None
        b = ttk.Frame(f)
        b.pack(fill="x")
        ttk.Button(b, text="Save template pages...", command=self.save).pack(side="left")
        ttk.Button(b, text="Close", command=self.destroy).pack(side="right")
        self._count()

    def _preset(self, _e: Any = None) -> None:
        name = self.cb.get()
        if name in fc.CHARSET_PRESETS:
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", fc.CHARSET_PRESETS[name])
            self._count()

    def _text(self) -> str:
        return "".join(c for c in self.txt.get("1.0", "end") if not c.isspace())

    def _count(self) -> None:
        cols = max(1, num(self.v_cols, 8, int))
        rows = max(1, num(self.v_rows, 9, int))
        pages = pi.template_pages(self._text(), cols, rows)
        self.lbl.configure(text="%d characters -> %d page(s) with %d boxes each." %
                           (len(self._text()), pages, cols * rows))
        try:
            im = pi.make_template(self._text(), cols, rows, self.app.proj.s, 1)
            im = im.resize((int(im.size[0] * 170 / im.size[1] * 1.0), 170), Image.Resampling.LANCZOS)
            self._ph = ImageTk.PhotoImage(im)
            self.prev.delete("all")
            self.prev.create_image(self.prev.winfo_width() // 2 or 300, 0, anchor="n", image=self._ph)
        except Exception:  # noqa: BLE001
            pass

    def save(self) -> None:
        text = self._text()
        if not text:
            return
        cols = max(1, num(self.v_cols, 8, int))
        rows = max(1, num(self.v_rows, 9, int))
        path = filedialog.asksaveasfilename(
            parent=self, title="Save template", defaultextension=".pdf",
            filetypes=[("PDF (all pages in one file)", "*.pdf"), ("PNG images (one per page)", "*.png")],
            initialfile="FontStudio-template.pdf")
        if not path:
            return
        pages = pi.template_pages(text, cols, rows)
        imgs = [pi.make_template(text, cols, rows, self.app.proj.s, n + 1) for n in range(pages)]
        try:
            if path.lower().endswith(".pdf"):
                imgs[0].convert("RGB").save(path, "PDF", resolution=300.0, save_all=True,
                                            append_images=[i.convert("RGB") for i in imgs[1:]])
            else:
                base, ext = os.path.splitext(path)
                for n, im in enumerate(imgs):
                    im.save("%s_page%d%s" % (base, n + 1, ext or ".png"), dpi=(300, 300))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not save", str(e), parent=self)
            return
        self.app.tpl_cols, self.app.tpl_rows = cols, rows
        self.app.tpl_chars = text
        messagebox.showinfo("Template saved", "Saved %d page(s).\n\nWhen importing, choose 'Template sheet', "
                            "use %d columns x %d rows, paste the same characters, and pick the page number."
                            % (pages, cols, rows), parent=self)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            pass


# --------------------------------------------------------------------------
# Export dialog
# --------------------------------------------------------------------------
class ExportDialog(tk.Toplevel):
    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.app = app
        self.title("Export font files")
        self.transient(app)
        self.resizable(False, False)
        center_on(self, app, 560, 360)
        s = app.proj.s
        safe = "".join(c for c in s.family if c.isalnum() or c in " -_").strip() or "MyFont"
        self.v_name = tk.StringVar(value=safe.replace(" ", ""))
        self.v_dir = tk.StringVar(value=app.last_export_dir)
        self.v_fmt: Dict[str, tk.BooleanVar] = {}
        self.v_install = tk.BooleanVar(value=False)
        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Font family name:").grid(row=0, column=0, sticky="w")
        ttk.Label(f, text=s.family + "  (" + s.style + ")", font=("", 10, "bold")).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="File name:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.v_name, width=34).grid(row=1, column=1, sticky="w")
        ttk.Label(f, text="Save into folder:").grid(row=2, column=0, sticky="w")
        e = ttk.Entry(f, textvariable=self.v_dir, width=34)
        e.grid(row=2, column=1, sticky="w")
        ttk.Button(f, text="Browse...", command=self.browse).grid(row=2, column=2, padx=4)
        fm = ttk.LabelFrame(f, text="Formats", padding=6)
        fm.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        avail = fc.available_formats()
        for k, label in fc.FORMATS.items():
            var = tk.BooleanVar(value=(k == "ttf"))
            self.v_fmt[k] = var
            cb = ttk.Checkbutton(fm, text=label + ("" if k in avail else "   (needs the 'brotli' package)"),
                                 variable=var)
            cb.pack(anchor="w")
            if k not in avail:
                cb.state(["disabled"])
                var.set(False)
        if sys.platform == "win32":
            ttk.Checkbutton(f, text="Install the .ttf (or .otf) for my Windows user after exporting",
                            variable=self.v_install).grid(row=4, column=0, columnspan=3, sticky="w")
        b = ttk.Frame(f)
        b.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        ttk.Button(b, text="Export", command=self.go).pack(side="left")
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right")

    def browse(self) -> None:
        d = filedialog.askdirectory(parent=self, initialdir=self.v_dir.get() or None)
        if d:
            self.v_dir.set(d)

    def go(self) -> None:
        kinds = [k for k, v in self.v_fmt.items() if v.get()]
        folder = self.v_dir.get().strip()
        name = self.v_name.get().strip() or "MyFont"
        if not kinds:
            messagebox.showinfo("Choose a format", "Tick at least one format.", parent=self)
            return
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Folder", "Please choose an existing folder.", parent=self)
            return
        install = bool(self.v_install.get())
        self.app.last_export_dir = folder
        self.destroy()
        self.app.do_export(kinds, folder, name, install)


# --------------------------------------------------------------------------
# Small dialogs
# --------------------------------------------------------------------------
class TransformDialog(tk.Toplevel):
    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.app = app
        self.title("Transform glyph")
        self.transient(app)
        self.resizable(False, False)
        center_on(self, app, 340, 240)
        self.vars = {k: tk.DoubleVar(value=v) for k, v in
                     (("sx", 100.0), ("sy", 100.0), ("rot", 0.0), ("shear", 0.0))}
        f = ttk.Frame(self, padding=12)
        f.pack(fill="both")
        for i, (key, label, lo, hi, inc) in enumerate((
                ("sx", "Width %", 10, 400, 1), ("sy", "Height %", 10, 400, 1),
                ("rot", "Rotate (degrees)", -180, 180, 1), ("shear", "Slant (degrees)", -45, 45, 1))):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", pady=3)
            spin(f, self.vars[key], lo, hi, inc, 8).grid(row=i, column=1, padx=8)
        ttk.Label(f, foreground="#555555", text="Scaling happens around the baseline centre.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=4)
        b = ttk.Frame(f)
        b.grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(b, text="Apply", command=self.apply).pack(side="left")
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right")

    def apply(self) -> None:
        g = lambda k, d: num(self.vars[k], d)  # noqa: E731
        self.app.transform_glyph(g("sx", 100) / 100.0, g("sy", 100) / 100.0, g("rot", 0), g("shear", 0))
        self.destroy()


class FontPickerDialog(tk.Toplevel):
    """Pick a reference font from the system fonts or a file."""

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.app = app
        self.title("Reference font")
        self.transient(app)
        center_on(self, app, 460, 520)
        self.fonts = fc.list_system_fonts()
        self.v_q = tk.StringVar()
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, wraplength=420, text="A reference font is shown faintly behind your drawing (onion "
                  "skin) and can be stamped into the canvas as a starting point.").pack(anchor="w")
        e = ttk.Entry(f, textvariable=self.v_q)
        e.pack(fill="x", pady=6)
        e.focus_set()
        self.v_q.trace_add("write", lambda *a: self.fill())
        box = ttk.Frame(f)
        box.pack(fill="both", expand=True)
        self.lb = tk.Listbox(box, activestyle="dotbox", exportselection=False)
        sb = ttk.Scrollbar(box, command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        self.lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.lb.bind("<Double-Button-1>", lambda e: self.use())
        b = ttk.Frame(f)
        b.pack(fill="x", pady=(8, 0))
        ttk.Button(b, text="Use selected", command=self.use).pack(side="left")
        ttk.Button(b, text="Browse for file...", command=self.browse).pack(side="left", padx=6)
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right")
        self.names: List[str] = []
        self.fill()

    def fill(self) -> None:
        q = self.v_q.get().lower()
        self.names = [n for n in self.fonts if q in n.lower()][:2000]
        self.lb.delete(0, "end")
        for n in self.names:
            self.lb.insert("end", n)

    def use(self) -> None:
        sel = self.lb.curselection()
        if not sel:
            return
        self.app.set_reference(self.fonts[self.names[sel[0]]])
        self.destroy()

    def browse(self) -> None:
        p = filedialog.askopenfilename(parent=self, filetypes=[("Fonts", "*.ttf *.otf *.ttc"), ("All", "*.*")])
        if p:
            self.app.set_reference(p)
            self.destroy()
