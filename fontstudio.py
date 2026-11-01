"""
FontStudio - make real, installable fonts from your handwriting, drawings and photos.

Run:  python fontstudio.py            (or build FontStudio.exe with build_exe.bat)
"""
from __future__ import annotations

import ctypes
import dataclasses
import math
import os
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

import fontcore as fc
import photoimport as pi
from dialogs import (ExportDialog, FontPickerDialog, PhotoDialog, TemplateDialog, TransformDialog,
                     center_on, run_task, spin)

APP_NAME = "FontStudio"
APP_VERSION = "1.0"
CW, CH = fc.CW, fc.CH

# --------------------------------------------------------------------------
# settings panel specification
# --------------------------------------------------------------------------
# (field, label, kind, lo, hi, step)      kind: f=float spin, i=int spin, b=check, s=entry, c=combo
STYLE_CHOICES = ["Regular", "Bold", "Italic", "Bold Italic"]
CASE_CHOICES = {"Off - only the glyphs I drew": "off",
                "Use my capitals for missing lowercase": "upper_for_lower",
                "Use my lowercase for missing capitals": "lower_for_upper"}

TABS: List[Tuple[str, List[tuple]]] = [
    ("Font", [
        ("family", "Font family name", "s"), ("style", "Style", "c", STYLE_CHOICES),
        ("designer", "Designer", "s"), ("copyright", "Copyright", "s"),
        ("license", "License text", "s"), ("version", "Version", "s"),
    ]),
    ("Metrics", [
        ("ascender", "Ascender (em)", "f", 0.4, 1.2, 0.01), ("descender", "Descender (em)", "f", 0.0, 0.6, 0.01),
        ("cap_height", "Capital height (em)", "f", 0.3, 1.0, 0.01), ("x_height", "x-height (em)", "f", 0.2, 0.9, 0.01),
        ("line_gap", "Extra line gap (em)", "f", 0.0, 0.6, 0.01),
        ("upm", "Units per em", "c", ["1000", "1024", "2000", "2048"]),
    ]),
    ("Spacing", [
        ("lsb", "Left space (em)", "f", -0.2, 0.5, 0.005), ("rsb", "Right space (em)", "f", -0.2, 0.5, 0.005),
        ("tracking", "Extra tracking (em)", "f", -0.3, 0.5, 0.005), ("space_width", "Space width (em)", "f", 0.05, 1.0, 0.01),
        ("min_advance", "Minimum glyph width (em)", "f", 0.0, 0.8, 0.01),
        ("monospace", "Monospaced font", "b"), ("mono_width", "Monospace width (em)", "f", 0.3, 1.2, 0.01),
    ]),
    ("Tracing", [
        ("upscale", "Detail (upscale x)", "i", 1, 4, 1), ("smooth", "Smoothing", "f", 0.0, 4.0, 0.1),
        ("precision", "Curve tightness (px, low = exact)", "f", 0.2, 6.0, 0.1),
        ("corner_angle", "Sharp corner angle (deg)", "f", 15, 180, 5),
        ("min_speck", "Ignore specks smaller than", "i", 0, 2000, 10),
        ("straight_lines", "Snap straight edges to lines", "b"),
        ("roughness", "Hand-made wobble", "f", 0.0, 6.0, 0.1), ("seed", "Wobble random seed", "i", 0, 9999, 1),
    ]),
    ("Style", [
        ("weight", "Weight: thicken / thin (px)", "f", -40, 60, 1), ("slant", "Slant (deg, + = right)", "f", -30, 30, 1),
        ("width_pct", "Width %", "f", 40, 200, 1), ("scale_pct", "Overall size %", "f", 40, 200, 1),
        ("baseline_shift", "Baseline shift (px)", "f", -120, 120, 1),
        ("case_fallback", "Missing letters", "c", list(CASE_CHOICES)),
        ("add_nbsp", "Also map non-breaking space", "b"),
    ]),
]

STYLE_PRESETS: Dict[str, dict] = {
    "Natural handwriting": dict(upscale=2, smooth=1.0, precision=1.2, corner_angle=55, straight_lines=False,
                                roughness=0.0, weight=0, slant=0),
    "Clean & geometric": dict(upscale=3, smooth=1.6, precision=1.6, corner_angle=45, straight_lines=True,
                              roughness=0.0, weight=0, slant=0),
    "Smooth marker": dict(upscale=2, smooth=2.4, precision=2.2, corner_angle=80, straight_lines=False,
                          roughness=0.0, weight=4, slant=0),
    "Rough / hand-cut": dict(upscale=2, smooth=0.6, precision=1.0, corner_angle=40, straight_lines=True,
                             roughness=2.2, weight=0, slant=0),
    "Pixel / blocky": dict(upscale=1, smooth=0.0, precision=0.4, corner_angle=30, straight_lines=True,
                           roughness=0.0, weight=0, slant=0),
    "Italic version of my drawing": dict(slant=12),
    "Bold version of my drawing": dict(weight=12, style="Bold"),
    "Light version of my drawing": dict(weight=-6),
}

SAMPLE = ("Hamburgefonstiv 0123456789\nThe quick brown fox jumps over the lazy dog.\n"
          "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS!")

TOOLS = [("pen", "Pen"), ("eraser", "Eraser"), ("line", "Line"), ("rect", "Rectangle"),
         ("rectf", "Box (filled)"), ("ellipse", "Ellipse"), ("ellipsef", "Disc (filled)"),
         ("fill", "Fill"), ("move", "Move")]
NIBS = ["Round", "Square", "Flat pen (calligraphy)", "Slim flat pen"]

INK = (28, 28, 34)
ONION = (176, 208, 244)
UNDERLAY = (226, 214, 190)
GUIDE_COLORS = {"baseline": (220, 40, 40), "xh": (60, 150, 60), "cap": (60, 90, 220),
                "asc": (170, 100, 200), "desc": (210, 130, 40)}


def nib_points(shape: str, size: float, angle: float) -> np.ndarray:
    r = max(0.5, size / 2.0)
    if shape == "Round":
        t = np.linspace(0, 2 * math.pi, 28, endpoint=False)
        pts = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    elif shape == "Square":
        pts = np.array([[-r, -r], [r, -r], [r, r], [-r, r]])
    else:
        thick = 0.16 if shape.startswith("Flat") else 0.07
        t = np.linspace(0, 2 * math.pi, 28, endpoint=False)
        pts = np.stack([r * np.cos(t), max(0.8, 2 * r * thick) * np.sin(t)], 1)
    if shape not in ("Round",):
        a = math.radians(-angle if shape != "Square" else -angle)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts = pts @ R.T
    return pts


def stroke(mask: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float], nib: np.ndarray,
           value: int) -> None:
    pts = np.vstack([nib + np.array(p0), nib + np.array(p1)]).astype(np.float32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, np.round(hull.reshape(-1, 2) * 16).astype(np.int32), int(value),
                       lineType=cv2.LINE_8, shift=4)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1500x900")
        self.minsize(1180, 700)
        self.proj = fc.Project()
        self.path: Optional[str] = None
        self.dirty = False
        self.cp: int = ord("A")
        self.mask = np.zeros((CH, CW), np.uint8)
        self.undo: Dict[int, List[np.ndarray]] = {}
        self.redo: Dict[int, List[np.ndarray]] = {}
        self.clip: Optional[np.ndarray] = None
        self.ref_path: Optional[str] = None
        self.ref_cache: Dict[int, np.ndarray] = {}
        self.underlay: Optional[np.ndarray] = None
        self.tpl_cols, self.tpl_rows, self.tpl_chars = 8, 9, ""
        self.last_export_dir = os.path.join(os.path.expanduser("~"), "Documents")
        if not os.path.isdir(self.last_export_dir):
            self.last_export_dir = os.path.expanduser("~")
        # drawing state
        self.v_tool = tk.StringVar(value="pen")
        self.v_size = tk.DoubleVar(value=22.0)
        self.v_nib = tk.StringVar(value=NIBS[0])
        self.v_angle = tk.DoubleVar(value=35.0)
        self.v_stab = tk.DoubleVar(value=0.0)
        self.v_mirror = tk.BooleanVar(value=False)
        self.v_guides = tk.BooleanVar(value=True)
        self.v_grid = tk.BooleanVar(value=False)
        self.v_outline = tk.BooleanVar(value=False)
        self.v_onion = tk.BooleanVar(value=True)
        self.v_under = tk.BooleanVar(value=True)
        self.v_hide_done = tk.BooleanVar(value=False)
        self.v_bl = tk.DoubleVar(value=0.0)
        self.v_br = tk.DoubleVar(value=0.0)
        self._drag: Dict[str, Any] = {}
        self._view = {"zoom": 1.0, "ox": 0, "oy": 0}
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._paint_job: Optional[str] = None
        self._thumbs: Dict[int, Tuple[int, ImageTk.PhotoImage]] = {}
        self._pal_job: Optional[str] = None
        self._pv_job: Optional[str] = None
        self._pv_running = False
        self._pv_pending = False
        self._pv_done = False
        self._pv_result: Tuple[Any, Any] = (None, None)
        self._pv_photo: Optional[ImageTk.PhotoImage] = None
        self._loading = False
        self.svars: Dict[str, tk.Variable] = {}
        self._build_menu()
        self._build_ui()
        self._load_glyph(self.cp)
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.after(200, self._first_run)

    # ======================================================================
    # menus
    # ======================================================================
    def _build_menu(self) -> None:
        m = tk.Menu(self)
        self.config(menu=m)

        f = tk.Menu(m, tearoff=False)
        m.add_cascade(label="File", menu=f)
        f.add_command(label="New project", accelerator="Ctrl+N", command=self.new_project)
        f.add_command(label="Open project...", accelerator="Ctrl+O", command=self.open_project)
        f.add_command(label="Save project", accelerator="Ctrl+S", command=self.save_project)
        f.add_command(label="Save project as...", command=lambda: self.save_project(True))
        f.add_separator()
        f.add_command(label="Export font files...", accelerator="Ctrl+E", command=self.export_dialog)
        f.add_command(label="Save the preview as picture...", command=self.save_preview_png)
        f.add_separator()
        f.add_command(label="Exit", command=self.quit_app)

        e = tk.Menu(m, tearoff=False)
        m.add_cascade(label="Edit", menu=e)
        e.add_command(label="Undo", accelerator="Ctrl+Z", command=self.do_undo)
        e.add_command(label="Redo", accelerator="Ctrl+Y", command=self.do_redo)
        e.add_separator()
        e.add_command(label="Copy glyph", accelerator="Ctrl+C", command=self.copy_glyph)
        e.add_command(label="Paste glyph", accelerator="Ctrl+V", command=self.paste_glyph)
        e.add_command(label="Copy from another character...", command=self.copy_from_char)
        e.add_command(label="Clear glyph", accelerator="Del", command=self.clear_glyph)

        g = tk.Menu(m, tearoff=False)
        m.add_cascade(label="Glyph", menu=g)
        g.add_command(label="Centre horizontally", command=self.center_glyph)
        g.add_command(label="Transform (size / rotate / slant)...", command=lambda: TransformDialog(self))
        g.add_command(label="Flip left <-> right", command=lambda: self.transform_glyph(-1, 1, 0, 0))
        g.add_command(label="Flip upside down", command=lambda: self.transform_glyph(1, -1, 0, 0))
        g.add_separator()
        g.add_command(label="Thicken strokes", command=lambda: self.morph("dilate"))
        g.add_command(label="Thin strokes", command=lambda: self.morph("erode"))
        g.add_command(label="Smooth edges", command=lambda: self.morph("smooth"))
        g.add_separator()
        g.add_command(label="Move left (Shift+Left)", command=lambda: self.nudge(-8, 0))
        g.add_command(label="Move right (Shift+Right)", command=lambda: self.nudge(8, 0))
        g.add_command(label="Move up (Shift+Up)", command=lambda: self.nudge(0, -8))
        g.add_command(label="Move down (Shift+Down)", command=lambda: self.nudge(0, 8))
        g.add_separator()
        g.add_command(label="Stamp reference font into this glyph", command=self.stamp_reference)
        g.add_separator()
        g.add_command(label="Previous glyph", accelerator="Left", command=lambda: self.step(-1))
        g.add_command(label="Next glyph", accelerator="Right", command=lambda: self.step(1))
        g.add_command(label="Next empty glyph", accelerator="Ctrl+Right", command=self.next_empty)

        i = tk.Menu(m, tearoff=False)
        m.add_cascade(label="Import", menu=i)
        i.add_command(label="Photo -> glyphs (auto-detect writing)...", accelerator="Ctrl+I",
                      command=lambda: self.open_photo("auto"))
        i.add_command(label="Photo of a printed template sheet...", command=lambda: self.open_photo("template"))
        i.add_command(label="Photo of a plain grid...", command=lambda: self.open_photo("grid"))
        i.add_command(label="Photo of one character (into the selected glyph)...",
                      command=lambda: self.open_photo("single"))
        i.add_separator()
        i.add_command(label="Make a printable template sheet...", command=lambda: TemplateDialog(self))
        i.add_separator()
        i.add_command(label="Underlay picture to trace by hand...", command=self.load_underlay)
        i.add_command(label="Remove underlay", command=self.clear_underlay)
        i.add_separator()
        i.add_command(label="Reference font (onion skin)...", command=lambda: FontPickerDialog(self))
        i.add_command(label="Remove reference font", command=lambda: self.set_reference(None))
        i.add_command(label="Fill empty glyphs from the reference font", command=self.fill_from_reference)

        c = tk.Menu(m, tearoff=False)
        m.add_cascade(label="Characters", menu=c)
        c.add_command(label="Add characters...", command=self.add_chars)
        c.add_separator()
        for name in fc.CHARSET_PRESETS:
            c.add_command(label="Add set: " + name, command=lambda n=name: self.add_set(n))
        c.add_separator()
        c.add_command(label="Remove empty glyph slots...", command=self.remove_empty_slots)

        v = tk.Menu(m, tearoff=False)
        m.add_cascade(label="View", menu=v)
        for label, var in (("Guide lines", self.v_guides), ("Pixel grid", self.v_grid),
                           ("Vector outline preview", self.v_outline), ("Reference font (onion skin)", self.v_onion),
                           ("Underlay picture", self.v_under)):
            v.add_checkbutton(label=label, variable=var, command=self.paint)

        h = tk.Menu(m, tearoff=False)
        m.add_cascade(label="Help", menu=h)
        h.add_command(label="Quick start", command=self.quick_start)
        h.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "%s %s\n\nMake real TrueType / OpenType / WOFF fonts from drawings, handwriting and "
            "photographs." % (APP_NAME, APP_VERSION), parent=self))

    # ======================================================================
    # layout
    # ======================================================================
    def _build_ui(self) -> None:
        self.status = tk.StringVar(value="")
        sb = ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(6, 2))
        sb.pack(side="bottom", fill="x")

        outer = ttk.PanedWindow(self, orient="vertical")
        outer.pack(fill="both", expand=True)
        top = ttk.PanedWindow(outer, orient="horizontal")
        outer.add(top, weight=5)

        # ---- palette
        pal = ttk.Frame(top, padding=4)
        top.add(pal, weight=0)
        hdr = ttk.Frame(pal)
        hdr.pack(fill="x")
        self.lbl_done = ttk.Label(hdr, text="")
        self.lbl_done.pack(side="left")
        ttk.Button(hdr, text="Add...", width=6, command=self.add_chars).pack(side="right")
        ttk.Checkbutton(pal, text="Hide finished glyphs", variable=self.v_hide_done,
                        command=self.refresh_palette).pack(anchor="w")
        box = ttk.Frame(pal)
        box.pack(fill="both", expand=True)
        self.pal = tk.Canvas(box, width=292, bg="#f4f4f4", highlightthickness=0)
        ps = ttk.Scrollbar(box, orient="vertical", command=self.pal.yview)
        self.pal.configure(yscrollcommand=ps.set)
        self.pal.pack(side="left", fill="both", expand=True)
        ps.pack(side="right", fill="y")
        self.pal.bind("<Button-1>", self._pal_click)
        self.pal.bind("<MouseWheel>", lambda e: self.pal.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.pal.bind("<Button-4>", lambda e: self.pal.yview_scroll(-1, "units"))
        self.pal.bind("<Button-5>", lambda e: self.pal.yview_scroll(1, "units"))

        # ---- editor
        ed = ttk.Frame(top, padding=4)
        top.add(ed, weight=4)
        self._build_toolbar(ed)
        mid = ttk.Frame(ed)
        mid.pack(fill="both", expand=True)
        self.cv = tk.Canvas(mid, bg="#8a8f98", highlightthickness=0, cursor="none")
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<Configure>", lambda e: self.paint())
        self.cv.bind("<ButtonPress-1>", self._press)
        self.cv.bind("<B1-Motion>", self._motion)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self.cv.bind("<Motion>", self._hover)
        self.cv.bind("<Leave>", lambda e: self.cv.delete("cursor"))
        self.cv.bind("<ButtonPress-3>", self._press_erase)
        self.cv.bind("<B3-Motion>", self._motion)
        self.cv.bind("<ButtonRelease-3>", self._release)
        bot = ttk.Frame(ed)
        bot.pack(fill="x", pady=(4, 0))
        self.lbl_glyph = ttk.Label(bot, text="", font=("Segoe UI", 11, "bold"))
        self.lbl_glyph.pack(side="left")
        ttk.Button(bot, text="Reset spacing", command=self.reset_bearings).pack(side="right")
        sr = ttk.Frame(bot)
        sr.pack(side="right", padx=8)
        ttk.Label(sr, text="Space left").pack(side="left")
        ttk.Scale(sr, from_=-250, to=250, variable=self.v_bl, length=130,
                  command=lambda v: self.bearing_changed()).pack(side="left", padx=4)
        ttk.Label(sr, text="Space right").pack(side="left")
        ttk.Scale(sr, from_=-250, to=250, variable=self.v_br, length=130,
                  command=lambda v: self.bearing_changed()).pack(side="left", padx=4)

        # ---- settings
        right = ttk.Frame(top, padding=(4, 4, 4, 0))
        top.add(right, weight=0)
        self._build_settings(right)

        # ---- preview
        pv = ttk.Frame(outer, padding=4)
        outer.add(pv, weight=2)
        row = ttk.Frame(pv)
        row.pack(fill="x")
        ttk.Label(row, text="Live preview of the real font:").pack(side="left")
        self.v_pv_text = tk.StringVar(value="")
        self.pv_entry = tk.Text(pv, height=3, width=40, wrap="word", font=("Segoe UI", 10))
        self.v_pv_size = tk.IntVar(value=64)
        ttk.Label(row, text="  Size").pack(side="left")
        spin(row, self.v_pv_size, 12, 240, 4, 5).pack(side="left", padx=4)
        ttk.Button(row, text="Refresh", command=lambda: self.schedule_preview(0)).pack(side="left", padx=6)
        self.lbl_pv = ttk.Label(row, text="")
        self.lbl_pv.pack(side="left", padx=8)
        body = ttk.Frame(pv)
        body.pack(fill="both", expand=True, pady=(4, 0))
        self.pv_entry.pack(in_=body, side="left", fill="y")
        self.pv_entry.insert("1.0", SAMPLE)
        self.pv_entry.bind("<KeyRelease>", lambda e: self.schedule_preview())
        self.pv_cv = tk.Canvas(body, bg="white", highlightthickness=1, highlightbackground="#999")
        pvs = ttk.Scrollbar(body, orient="vertical", command=self.pv_cv.yview)
        self.pv_cv.configure(yscrollcommand=pvs.set)
        pvs.pack(side="right", fill="y")
        self.pv_cv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.pv_cv.bind("<Configure>", lambda e: self.schedule_preview(250))
        self.v_pv_size.trace_add("write", lambda *a: self.schedule_preview())

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x")
        for key, label in TOOLS:
            ttk.Radiobutton(bar, text=label, value=key, variable=self.v_tool, style="Toolbutton",
                            command=self._tool_changed).pack(side="left", padx=1, pady=1)
        bar2 = ttk.Frame(parent)
        bar2.pack(fill="x", pady=(2, 0))
        ttk.Label(bar2, text="Size").pack(side="left")
        ttk.Scale(bar2, from_=1, to=140, variable=self.v_size, length=120,
                  command=lambda v: self._tool_changed()).pack(side="left", padx=4)
        self.lbl_size = ttk.Label(bar2, text="22", width=4)
        self.lbl_size.pack(side="left")
        ttk.Label(bar2, text="Nib").pack(side="left", padx=(6, 0))
        cb = ttk.Combobox(bar2, textvariable=self.v_nib, values=NIBS, state="readonly", width=20)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._tool_changed())
        ttk.Label(bar2, text="Angle").pack(side="left")
        ttk.Scale(bar2, from_=0, to=180, variable=self.v_angle, length=80,
                  command=lambda v: self._tool_changed()).pack(side="left", padx=4)
        ttk.Label(bar2, text="Steady hand").pack(side="left", padx=(6, 0))
        ttk.Scale(bar2, from_=0, to=0.92, variable=self.v_stab, length=80).pack(side="left", padx=4)
        bar3 = ttk.Frame(parent)
        bar3.pack(fill="x", pady=(2, 4))
        for text, var in (("Mirror strokes", self.v_mirror), ("Guides", self.v_guides), ("Grid", self.v_grid),
                          ("Vector outline", self.v_outline), ("Reference", self.v_onion),
                          ("Underlay", self.v_under)):
            ttk.Checkbutton(bar3, text=text, variable=var, command=self.paint).pack(side="left", padx=3)
        ttk.Button(bar3, text="Undo", width=6, command=self.do_undo).pack(side="right", padx=1)
        ttk.Button(bar3, text="Redo", width=6, command=self.do_redo).pack(side="right", padx=1)
        ttk.Button(bar3, text="Clear", width=6, command=self.clear_glyph).pack(side="right", padx=1)
        ttk.Button(bar3, text="Photo...", width=8, command=lambda: self.open_photo("auto")).pack(side="right", padx=6)

    def _build_settings(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill="x")
        ttk.Label(top, text="Style presets").pack(anchor="w")
        cb = ttk.Combobox(top, values=list(STYLE_PRESETS), state="readonly", width=34)
        cb.pack(fill="x", pady=(0, 6))
        cb.set("Apply a preset...")
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(cb.get()))
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        s0 = self.proj.s
        for title, rows in TABS:
            fr = ttk.Frame(nb, padding=8)
            nb.add(fr, text=title)
            fr.columnconfigure(1, weight=1)
            for r, spec in enumerate(rows):
                field, label, kind = spec[0], spec[1], spec[2]
                ttk.Label(fr, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
                cur = getattr(s0, field)
                if kind == "b":
                    var: tk.Variable = tk.BooleanVar(value=bool(cur))
                    ttk.Checkbutton(fr, variable=var).grid(row=r, column=1, sticky="w")
                elif kind == "s":
                    var = tk.StringVar(value=str(cur))
                    ttk.Entry(fr, textvariable=var, width=24).grid(row=r, column=1, sticky="ew")
                elif kind == "c":
                    vals = spec[3]
                    if field == "case_fallback":
                        var = tk.StringVar(value=next(k for k, v in CASE_CHOICES.items() if v == cur))
                    else:
                        var = tk.StringVar(value=str(cur))
                    ttk.Combobox(fr, textvariable=var, values=vals, state="readonly",
                                 width=30 if field == "case_fallback" else 12).grid(row=r, column=1, sticky="w")
                elif kind == "i":
                    var = tk.IntVar(value=int(cur))
                    spin(fr, var, spec[3], spec[4], spec[5], 8).grid(row=r, column=1, sticky="w")
                else:
                    var = tk.DoubleVar(value=float(cur))
                    spin(fr, var, spec[3], spec[4], spec[5], 8, "%.3f" if spec[5] < 0.05 else "%.2f"
                         ).grid(row=r, column=1, sticky="w")
                self.svars[field] = var
                var.trace_add("write", lambda *a: self.settings_changed())
        ttk.Label(parent, foreground="#555555", wraplength=330,
                  text="Tip: spacing and tracing changes are heard instantly in the live preview below.",
                  ).pack(fill="x", pady=(6, 0))

    # ======================================================================
    # keyboard
    # ======================================================================
    def _bind_keys(self) -> None:
        def typing() -> bool:
            w = self.focus_get()
            return w is not None and w.winfo_class() in ("TEntry", "Entry", "Text", "TSpinbox", "Spinbox",
                                                         "TCombobox")

        def guard(fn: Any) -> Any:
            def h(e: Any) -> Optional[str]:
                if typing():
                    return None
                fn()
                return "break"
            return h

        b = self.bind_all
        b("<Control-z>", lambda e: self.do_undo())
        b("<Control-y>", lambda e: self.do_redo())
        b("<Control-s>", lambda e: self.save_project())
        b("<Control-o>", lambda e: self.open_project())
        b("<Control-n>", lambda e: self.new_project())
        b("<Control-e>", lambda e: self.export_dialog())
        b("<Control-i>", lambda e: self.open_photo("auto"))
        b("<Control-c>", guard(self.copy_glyph))
        b("<Control-v>", guard(self.paste_glyph))
        b("<Delete>", guard(self.clear_glyph))
        b("<Left>", guard(lambda: self.step(-1)))
        b("<Right>", guard(lambda: self.step(1)))
        b("<Control-Right>", guard(self.next_empty))
        b("<Shift-Left>", guard(lambda: self.nudge(-8, 0)))
        b("<Shift-Right>", guard(lambda: self.nudge(8, 0)))
        b("<Shift-Up>", guard(lambda: self.nudge(0, -8)))
        b("<Shift-Down>", guard(lambda: self.nudge(0, 8)))
        b("<bracketleft>", guard(lambda: self.v_size.set(max(1, self.v_size.get() - 3)) or self._tool_changed()))
        b("<bracketright>", guard(lambda: self.v_size.set(min(140, self.v_size.get() + 3)) or self._tool_changed()))
        for key, tool in (("p", "pen"), ("e", "eraser"), ("l", "line"), ("r", "rect"), ("o", "ellipse"),
                          ("f", "fill"), ("m", "move")):
            b("<KeyPress-%s>" % key, guard(lambda t=tool: (self.v_tool.set(t), self._tool_changed())))

    # ======================================================================
    # helpers
    # ======================================================================
    def info(self, msg: str) -> None:
        self.status.set(msg)

    def mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = os.path.basename(self.path) if self.path else "untitled"
        self.title("%s%s - %s" % (name, " *" if self.dirty else "", APP_NAME))

    def _first_run(self) -> None:
        self.refresh_palette()
        self.schedule_preview(100)
        self._update_title()
        self.info("Draw a letter on the big canvas, or use Import > Photo to bring in your handwriting. "
                  "Help > Quick start explains the workflow.")

    def quick_start(self) -> None:
        messagebox.showinfo("Quick start", (
            "FROM A PHOTO\n"
            "  1. Import > Make a printable template sheet, print it and write one letter per box.\n"
            "  2. Photograph or scan it, then Import > Photo of a printed template sheet.\n"
            "  (or use 'Photo -> glyphs' on any picture of handwriting; type what it says, in order)\n\n"
            "BY DRAWING\n"
            "  Pick a letter on the left, draw with the pen (right mouse button erases).\n"
            "  Faint guide lines show baseline, x-height, capitals and ascender.\n"
            "  Use Import > Reference font to trace over an existing typeface.\n\n"
            "FINISH\n"
            "  Spacing/Tracing/Style tabs shape the font; the preview shows the real font live.\n"
            "  File > Export font files writes .ttf / .otf / .woff and can install it for you."),
            parent=self)

    # ======================================================================
    # glyph selection / state
    # ======================================================================
    def _load_glyph(self, cp: int) -> None:
        self.cp = cp
        self.mask = self.proj.get_mask(cp).copy()
        bl, br = self.proj.bearings.get(cp, [0.0, 0.0])
        self._loading = True
        self.v_bl.set(bl)
        self.v_br.set(br)
        self._loading = False
        ch = chr(cp)
        self.lbl_glyph.configure(text="U+%04X   %s" % (cp, ch if ch.isprintable() else ""))
        self.paint()
        self.refresh_palette()

    def select(self, cp: int) -> None:
        if cp == self.cp:
            return
        self.cp = cp
        self._load_glyph(cp)

    def step(self, d: int) -> None:
        chars = self._visible_chars() or self.proj.chars
        if not chars:
            return
        try:
            i = chars.index(self.cp)
        except ValueError:
            i = -1 if d > 0 else 0
        self.select(chars[(i + d) % len(chars)])

    def next_empty(self) -> None:
        chars = self.proj.chars
        if not chars:
            return
        start = chars.index(self.cp) if self.cp in chars else -1
        for k in range(1, len(chars) + 1):
            cp = chars[(start + k) % len(chars)]
            if not self.proj.filled(cp):
                self.select(cp)
                return
        self.info("Every glyph in the list has a drawing.")

    def commit(self, schedule: bool = True) -> None:
        self.proj.set_mask(self.cp, self.mask)
        self.mark_dirty()
        self._thumbs.pop(self.cp, None)
        self.refresh_palette()
        if schedule:
            self.schedule_preview()

    # ---- undo / redo
    def push_undo(self) -> None:
        st = self.undo.setdefault(self.cp, [])
        st.append(self.mask.copy())
        if len(st) > 60:
            del st[0]
        self.redo[self.cp] = []

    def do_undo(self) -> None:
        st = self.undo.get(self.cp)
        if not st:
            return
        self.redo.setdefault(self.cp, []).append(self.mask.copy())
        self.mask = st.pop()
        self.commit()
        self.paint()

    def do_redo(self) -> None:
        st = self.redo.get(self.cp)
        if not st:
            return
        self.undo.setdefault(self.cp, []).append(self.mask.copy())
        self.mask = st.pop()
        self.commit()
        self.paint()

    # ---- glyph operations
    def clear_glyph(self) -> None:
        if not self.mask.any():
            return
        self.push_undo()
        self.mask[:] = 0
        self.commit()
        self.paint()

    def copy_glyph(self) -> None:
        self.clip = self.mask.copy()
        self.info("Glyph copied. Select another character and press Ctrl+V.")

    def paste_glyph(self) -> None:
        if self.clip is None:
            return
        self.push_undo()
        self.mask = self.clip.copy()
        self.commit()
        self.paint()

    def copy_from_char(self) -> None:
        s = simpledialog.askstring("Copy from character", "Type the character to copy into this glyph "
                                   "(for example copy O into 0):", parent=self)
        if not s:
            return
        src = ord(s[0])
        if not self.proj.filled(src):
            messagebox.showinfo("Nothing to copy", "'%s' has no drawing yet." % s[0], parent=self)
            return
        self.push_undo()
        self.mask = self.proj.get_mask(src).copy()
        self.commit()
        self.paint()

    def center_glyph(self) -> None:
        if not self.mask.any():
            return
        self.push_undo()
        self.mask = fc.auto_center(self.mask)
        self.commit()
        self.paint()

    def nudge(self, dx: int, dy: int) -> None:
        if not self.mask.any():
            return
        self.push_undo()
        self.mask = fc.shift_mask(self.mask, dx, dy)
        self.commit()
        self.paint()

    def transform_glyph(self, sx: float, sy: float, rot: float, shear: float) -> None:
        if not self.mask.any():
            return
        self.push_undo()
        self.mask = fc.affine_mask(self.mask, sx, sy, rot, shear)
        self.commit()
        self.paint()

    def morph(self, kind: str) -> None:
        if not self.mask.any():
            return
        self.push_undo()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if kind == "dilate":
            self.mask = cv2.dilate(self.mask, k)
        elif kind == "erode":
            self.mask = cv2.erode(self.mask, k)
        else:
            b = cv2.GaussianBlur(self.mask.astype(np.float32), (0, 0), 2.2)
            self.mask = np.where(b > 127, 255, 0).astype(np.uint8)
        self.commit()
        self.paint()

    def bearing_changed(self) -> None:
        if self._loading:
            return
        bl, br = round(self.v_bl.get()), round(self.v_br.get())
        if bl == 0 and br == 0:
            self.proj.bearings.pop(self.cp, None)
        else:
            self.proj.bearings[self.cp] = [float(bl), float(br)]
        self.mark_dirty()
        self.schedule_preview(250)

    def reset_bearings(self) -> None:
        self.proj.bearings.pop(self.cp, None)
        self._loading = True
        self.v_bl.set(0)
        self.v_br.set(0)
        self._loading = False
        self.mark_dirty()
        self.schedule_preview()

    # ======================================================================
    # reference font / underlay
    # ======================================================================
    def set_reference(self, path: Optional[str]) -> None:
        self.ref_path = path
        self.ref_cache.clear()
        if path:
            self.v_onion.set(True)
            self.info("Reference font: %s" % os.path.basename(path))
        self.paint()

    def ref_mask(self, cp: int) -> Optional[np.ndarray]:
        if not self.ref_path:
            return None
        m = self.ref_cache.get(cp)
        if m is None:
            try:
                m = fc.reference_mask(chr(cp), self.ref_path)
            except Exception:  # noqa: BLE001
                m = np.zeros((CH, CW), np.uint8)
            self.ref_cache[cp] = m
        return m

    def stamp_reference(self) -> None:
        m = self.ref_mask(self.cp)
        if m is None:
            messagebox.showinfo("No reference font", "Choose one with Import > Reference font first.", parent=self)
            return
        self.push_undo()
        self.mask = np.maximum(self.mask, m)
        self.commit()
        self.paint()

    def fill_from_reference(self) -> None:
        if not self.ref_path:
            messagebox.showinfo("No reference font", "Choose one with Import > Reference font first.", parent=self)
            return
        n = 0
        for cp in self.proj.chars:
            if not self.proj.filled(cp) and chr(cp).strip():
                m = self.ref_mask(cp)
                if m is not None and m.any():
                    self.proj.set_mask(cp, m)
                    n += 1
        self._load_glyph(self.cp)
        self.mark_dirty()
        self.schedule_preview()
        self.info("Filled %d empty glyphs from the reference font - now edit them into your own style." % n)

    def load_underlay(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Choose a picture to trace over",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                                                     ("All files", "*.*")])
        if not path:
            return
        try:
            g = pi.load_gray(path, 2000)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot open image", str(e), parent=self)
            return
        h, w = g.shape
        sc = min(CW / w, CH / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((CH, CW), 255, np.uint8)
        canvas[(CH - nh) // 2:(CH - nh) // 2 + nh, (CW - nw) // 2:(CW - nw) // 2 + nw] = g
        self.underlay = canvas
        self.v_under.set(True)
        self.paint()
        self.info("Underlay loaded - trace over it with the pen. (View > Underlay picture hides it.)")

    def clear_underlay(self) -> None:
        self.underlay = None
        self.paint()

    # ======================================================================
    # drawing canvas
    # ======================================================================
    def _tool_changed(self) -> None:
        self.lbl_size.configure(text="%d" % round(self.v_size.get()))

    def _nib(self) -> np.ndarray:
        shape = self.v_nib.get()
        return nib_points(shape, float(self.v_size.get()), float(self.v_angle.get()))

    def _to_img(self, ev: Any) -> Tuple[float, float]:
        z = self._view["zoom"]
        return (ev.x - self._view["ox"]) / z, (ev.y - self._view["oy"]) / z

    def _hover(self, ev: Any) -> None:
        self.cv.delete("cursor")
        z = self._view["zoom"]
        tool = self.v_tool.get()
        if tool in ("pen", "eraser"):
            pts = self._nib() * z + np.array([ev.x, ev.y])
            self.cv.create_polygon(*pts.reshape(-1).tolist(), outline="#e0245e", fill="", tags="cursor")
        else:
            self.cv.create_line(ev.x - 8, ev.y, ev.x + 8, ev.y, fill="#e0245e", tags="cursor")
            self.cv.create_line(ev.x, ev.y - 8, ev.x, ev.y + 8, fill="#e0245e", tags="cursor")
        x, y = self._to_img(ev)
        em = fc.PX_EM
        self.info("x %.0f   y %.0f   (%.2f em above baseline)" % (x, y, (fc.BASELINE_Y - y) / em))

    def _press_erase(self, ev: Any) -> None:
        self._press(ev, erase=True)

    def _press(self, ev: Any, erase: bool = False) -> None:
        self.cv.focus_set()
        x, y = self._to_img(ev)
        tool = self.v_tool.get()
        self._drag = {"tool": tool, "erase": erase, "start": (x, y), "last": (x, y), "sm": (x, y),
                      "base": None}
        if tool == "fill":
            self._flood(x, y, erase)
            self._drag = {}
            return
        self.push_undo()
        if tool in ("pen", "eraser") or erase:
            self._drag["tool"] = "eraser" if erase else tool
            self._dab(x, y, x, y, erase or tool == "eraser")
        elif tool == "move":
            self._drag["base"] = self.mask.copy()
        else:
            self._drag["base"] = self.mask.copy()
        self.paint(fast=True)

    def _dab(self, x0: float, y0: float, x1: float, y1: float, erase: bool) -> None:
        nib = self._nib()
        val = 0 if erase else 255
        stroke(self.mask, (x0, y0), (x1, y1), nib, val)
        if self.v_mirror.get():
            nib_m = nib * np.array([-1.0, 1.0])
            stroke(self.mask, (CW - x0, y0), (CW - x1, y1), nib_m, val)

    def _motion(self, ev: Any) -> None:
        d = self._drag
        if not d:
            return
        self._hover(ev)
        x, y = self._to_img(ev)
        tool = d["tool"]
        if tool in ("pen", "eraser"):
            s = float(self.v_stab.get())
            sx, sy = d["sm"]
            nx, ny = sx + (x - sx) * (1 - s), sy + (y - sy) * (1 - s)
            lx, ly = d["last"]
            self._dab(lx, ly, nx, ny, tool == "eraser")
            d["last"] = (nx, ny)
            d["sm"] = (nx, ny)
        elif tool == "move":
            sx, sy = d["start"]
            self.mask = fc.shift_mask(d["base"], int(round(x - sx)), int(round(y - sy)))
        else:
            self.mask = d["base"].copy()
            self._shape(tool, d["start"], (x, y), d["erase"], ev)
        self.paint(fast=True)

    def _shape(self, tool: str, a: Tuple[float, float], b: Tuple[float, float], erase: bool, ev: Any) -> None:
        val = 0 if erase else 255
        size = max(1, int(round(self.v_size.get())))
        x0, y0 = a
        x1, y1 = b
        if ev.state & 0x0001:       # Shift: straight/square/circle
            if tool == "line":
                ang = round(math.atan2(y1 - y0, x1 - x0) / (math.pi / 4)) * (math.pi / 4)
                ln = math.hypot(x1 - x0, y1 - y0)
                x1, y1 = x0 + ln * math.cos(ang), y0 + ln * math.sin(ang)
            else:
                s = max(abs(x1 - x0), abs(y1 - y0))
                x1 = x0 + math.copysign(s, x1 - x0 or 1)
                y1 = y0 + math.copysign(s, y1 - y0 or 1)
        if tool == "line":
            self._dab(x0, y0, x1, y1, erase)
            return
        fill = tool.endswith("f")
        t = -1 if fill else size
        shapes = [((x0, y0), (x1, y1))]
        if self.v_mirror.get():
            shapes.append(((CW - x0, y0), (CW - x1, y1)))
        for (p0, p1) in shapes:
            if tool.startswith("rect"):
                cv2.rectangle(self.mask, (int(round(p0[0])), int(round(p0[1]))),
                              (int(round(p1[0])), int(round(p1[1]))), val, t, cv2.LINE_8)
            else:
                c = (int(round((p0[0] + p1[0]) / 2)), int(round((p0[1] + p1[1]) / 2)))
                ax = (max(1, int(abs(p1[0] - p0[0]) / 2)), max(1, int(abs(p1[1] - p0[1]) / 2)))
                cv2.ellipse(self.mask, c, ax, 0, 0, 360, val, t, cv2.LINE_8)

    def _release(self, ev: Any) -> None:
        d = self._drag
        if not d:
            return
        x, y = self._to_img(ev)
        if d["tool"] in ("pen", "eraser"):
            lx, ly = d["last"]
            self._dab(lx, ly, x, y, d["tool"] == "eraser")
        self._drag = {}
        changed = True
        if self.undo.get(self.cp) and np.array_equal(self.undo[self.cp][-1], self.mask):
            self.undo[self.cp].pop()
            changed = False
        if changed:
            self.commit()
        self.paint()

    def _flood(self, x: float, y: float, erase: bool) -> None:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < CW and 0 <= yi < CH):
            return
        self.push_undo()
        h, w = self.mask.shape
        ff = np.zeros((h + 2, w + 2), np.uint8)
        src = self.mask.copy()
        target = 0 if src[yi, xi] > 127 else 255
        cv2.floodFill(src, ff, (xi, yi), int(target), loDiff=0, upDiff=0, flags=4 | cv2.FLOODFILL_FIXED_RANGE)
        if target == 255 and src[0, 0] == 255 and ff[1, 1] == 1:
            if not messagebox.askyesno("Fill the whole page?",
                                       "This area is not enclosed, so filling would flood the whole canvas. "
                                       "Close the shape first, or continue anyway?", parent=self):
                self.undo[self.cp].pop()
                return
        self.mask = src
        self.commit()
        self.paint()

    # ---- painting
    def paint(self, fast: bool = False) -> None:
        if fast:
            if self._paint_job is None:
                self._paint_job = self.after_idle(self._paint_now)
            return
        self._paint_now()

    def _paint_now(self) -> None:
        self._paint_job = None
        cw, ch = max(200, self.cv.winfo_width()), max(200, self.cv.winfo_height())
        z = min((cw - 16) / CW, (ch - 16) / CH)
        z = max(0.3, z)
        W, H = int(CW * z), int(CH * z)
        ox, oy = (cw - W) // 2, (ch - H) // 2
        self._view.update(zoom=z, ox=ox, oy=oy)
        img = np.full((CH, CW, 3), 255, np.uint8)
        if self.underlay is not None and self.v_under.get():
            u = (self.underlay.astype(np.float32) / 255.0)[..., None]
            img = (np.array(UNDERLAY, np.float32)[None, None, :] * (1 - u) + 255 * u).astype(np.uint8)
        if self.v_onion.get():
            r = self.ref_mask(self.cp)
            if r is not None:
                img[r > 127] = ONION
        ink = self.mask > 127
        img[ink] = INK
        interp = cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR
        disp = cv2.resize(img, (W, H), interpolation=interp)
        if self.v_grid.get():
            step = 32
            for gx in range(0, CW, step):
                cv2.line(disp, (int(gx * z), 0), (int(gx * z), H), (232, 232, 238), 1)
            for gy in range(0, CH, step):
                cv2.line(disp, (0, int(gy * z)), (W, int(gy * z)), (232, 232, 238), 1)
        if self.v_guides.get():
            self._guides(disp, z)
        if self.v_outline.get() and self.mask.any():
            self._outline(disp, z)
        self._photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self.cv.delete("img")
        self.cv.create_rectangle(ox - 1, oy - 1, ox + W, oy + H, outline="#444", tags="img")
        self.cv.create_image(ox, oy, anchor="nw", image=self._photo, tags="img")
        self.cv.tag_lower("img")

    def _guides(self, disp: np.ndarray, z: float) -> None:
        s = self.proj.s
        base = fc.BASELINE_Y
        W = disp.shape[1]
        items = [("baseline", 0.0, 2), ("xh", s.x_height, 1), ("cap", s.cap_height, 1),
                 ("asc", s.ascender, 1), ("desc", -s.descender, 1)]
        for key, em, th in items:
            y = int(round((base - em * fc.PX_EM) * z))
            if 0 <= y < disp.shape[0]:
                col = GUIDE_COLORS[key]
                cv2.line(disp, (0, y), (W, y), col, th, cv2.LINE_AA)
        x = int(CW / 2 * z)
        cv2.line(disp, (x, 0), (x, disp.shape[0]), (225, 225, 235), 1)

    def _outline(self, disp: np.ndarray, z: float) -> None:
        try:
            contours = fc.traced_contours(self.proj, self.cp, self.mask)
        except Exception:  # noqa: BLE001
            return
        for c in contours:
            pts: List[Tuple[int, int]] = []
            nodes: List[Tuple[int, int]] = []
            for seg in c["segs"]:
                if len(seg) == 2:
                    pts.append((int(seg[0][0] * z), int(seg[0][1] * z)))
                else:
                    t = np.linspace(0, 1, 12, endpoint=False)
                    for q in fc._bez(np.array(seg), t):
                        pts.append((int(q[0] * z), int(q[1] * z)))
                nodes.append((int(seg[0][0] * z), int(seg[0][1] * z)))
            if len(pts) > 1:
                cv2.polylines(disp, [np.array(pts, np.int32)], True, (230, 30, 90), 1, cv2.LINE_AA)
            for nx, ny in nodes:
                cv2.rectangle(disp, (nx - 2, ny - 2), (nx + 2, ny + 2), (20, 90, 220), -1)

    # ======================================================================
    # palette
    # ======================================================================
    CELL_W, CELL_H = 58, 72

    def _visible_chars(self) -> List[int]:
        if self.v_hide_done.get():
            return [cp for cp in self.proj.chars if not self.proj.filled(cp) or cp == self.cp]
        return list(self.proj.chars)

    def refresh_palette(self) -> None:
        if self._pal_job is None:
            self._pal_job = self.after(30, self._palette_now)

    def _thumb(self, cp: int) -> Optional[ImageTk.PhotoImage]:
        m = self.proj.glyphs.get(cp)
        if m is None or not m.any():
            return None
        hit = self._thumbs.get(cp)
        if hit and hit[0] == id(m):
            return hit[1]
        sm = cv2.resize(m, (36, 45), interpolation=cv2.INTER_AREA)
        arr = (255 - (sm.astype(np.float32) * 0.88)).astype(np.uint8)
        ph = ImageTk.PhotoImage(Image.fromarray(arr))
        self._thumbs[cp] = (id(m), ph)
        return ph

    def _palette_now(self) -> None:
        self._pal_job = None
        chars = self._visible_chars()
        cols = max(1, int(self.pal.winfo_width() // self.CELL_W) or 5)
        self.pal.delete("all")
        for n, cp in enumerate(chars):
            r, c = divmod(n, cols)
            x0, y0 = c * self.CELL_W + 2, r * self.CELL_H + 2
            sel = cp == self.cp
            done = self.proj.filled(cp)
            self.pal.create_rectangle(x0, y0, x0 + self.CELL_W - 4, y0 + self.CELL_H - 4,
                                      fill="#d9ecff" if sel else ("#ffffff" if done else "#f4f4f4"),
                                      outline="#2a7de1" if sel else "#c8c8c8", width=2 if sel else 1)
            ph = self._thumb(cp)
            if ph is not None:
                self.pal.create_image(x0 + (self.CELL_W - 4) // 2, y0 + 4, anchor="n", image=ph)
            label = chr(cp) if chr(cp).strip() else "SP"
            self.pal.create_text(x0 + (self.CELL_W - 4) // 2, y0 + self.CELL_H - 14, text=label,
                                 font=("Segoe UI", 10, "bold"), fill="#1a7f37" if done else "#888")
        rows = (len(chars) + cols - 1) // cols
        self.pal.configure(scrollregion=(0, 0, cols * self.CELL_W, max(rows * self.CELL_H, 10)))
        self._pal_cols = cols
        self._pal_chars = chars
        total = len(self.proj.chars)
        done = sum(1 for cp in self.proj.chars if self.proj.filled(cp))
        self.lbl_done.configure(text="%d of %d glyphs drawn" % (done, total))

    def _pal_click(self, ev: Any) -> None:
        chars = getattr(self, "_pal_chars", [])
        cols = getattr(self, "_pal_cols", 5)
        x = self.pal.canvasx(ev.x)
        y = self.pal.canvasy(ev.y)
        c, r = int(x // self.CELL_W), int(y // self.CELL_H)
        if c >= cols:
            return
        n = r * cols + c
        if 0 <= n < len(chars):
            self.select(chars[n])

    def add_chars(self) -> None:
        s = simpledialog.askstring("Add characters", "Type or paste the characters to add:", parent=self)
        if not s:
            return
        added = 0
        for cp in fc.text_to_cps(s):
            if cp not in self.proj.chars:
                self.proj.chars.append(cp)
                added += 1
        self.mark_dirty()
        self.refresh_palette()
        self.info("Added %d characters." % added)

    def add_set(self, name: str) -> None:
        added = 0
        for cp in fc.text_to_cps(fc.CHARSET_PRESETS[name]):
            if cp not in self.proj.chars:
                self.proj.chars.append(cp)
                added += 1
        self.mark_dirty()
        self.refresh_palette()
        self.info("Added %d characters from '%s'." % (added, name))

    def remove_empty_slots(self) -> None:
        empty = [cp for cp in self.proj.chars if not self.proj.filled(cp)]
        if not empty:
            return
        if not messagebox.askyesno("Remove empty slots", "Remove %d characters that have no drawing from the "
                                   "list? (You can add them again any time.)" % len(empty), parent=self):
            return
        self.proj.chars = [cp for cp in self.proj.chars if self.proj.filled(cp)] or fc.default_charset()[:1]
        if self.cp not in self.proj.chars:
            self.select(self.proj.chars[0])
        self.mark_dirty()
        self.refresh_palette()

    # ======================================================================
    # settings
    # ======================================================================
    def settings_changed(self) -> None:
        if self._loading:
            return
        s = self.proj.s
        for field, var in self.svars.items():
            try:
                raw = var.get()
            except (tk.TclError, ValueError):
                continue
            if field == "case_fallback":
                raw = CASE_CHOICES.get(str(raw), "off")
            cur = getattr(s, field)
            try:
                setattr(s, field, type(cur)(raw))
            except (TypeError, ValueError):
                pass
        self.mark_dirty()
        self.paint(fast=True)
        self.schedule_preview(350)

    def push_settings_to_ui(self) -> None:
        self._loading = True
        s = self.proj.s
        for field, var in self.svars.items():
            val = getattr(s, field)
            if field == "case_fallback":
                val = next((k for k, v in CASE_CHOICES.items() if v == val), list(CASE_CHOICES)[0])
            try:
                var.set(val)
            except tk.TclError:
                pass
        self._loading = False

    def apply_preset(self, name: str) -> None:
        d = STYLE_PRESETS.get(name)
        if not d:
            return
        for k, v in d.items():
            setattr(self.proj.s, k, type(getattr(self.proj.s, k))(v))
        self.push_settings_to_ui()
        self.settings_changed()
        self.info("Applied style preset '%s'." % name)

    # ======================================================================
    # live preview
    # ======================================================================
    def schedule_preview(self, delay: int = 450) -> None:
        if self._pv_job:
            self.after_cancel(self._pv_job)
        self._pv_job = self.after(delay, self._start_preview)

    def _start_preview(self) -> None:
        self._pv_job = None
        if self._pv_running:
            self._pv_pending = True
            return
        if self.proj.count_filled() == 0:
            self.pv_cv.delete("all")
            self.pv_cv.create_text(20, 20, anchor="nw", fill="#777", font=("Segoe UI", 11),
                                   text="Draw or import at least one glyph to see the font here.")
            self.lbl_pv.configure(text="")
            return
        snap = self.proj.snapshot()
        text = self.pv_entry.get("1.0", "end").rstrip("\n") or SAMPLE
        size = max(8, num_int(self.v_pv_size, 64))
        width = max(300, self.pv_cv.winfo_width() - 4)
        self._pv_running = True
        self._pv_done = False
        self.lbl_pv.configure(text="updating...")

        def work() -> None:
            try:
                data = fc.font_bytes(snap, "ttf")
                img = fc.render_text(snap, text, size, width, data=data)
                self._pv_result = (img, None)
            except BaseException as e:  # noqa: BLE001
                traceback.print_exc()
                self._pv_result = (None, e)
            self._pv_done = True        # polled from the Tk thread (Tk is not thread safe)

        threading.Thread(target=work, daemon=True).start()
        self.after(60, self._poll_preview)

    def _poll_preview(self) -> None:
        if self._pv_done:
            self._finish_preview()
        else:
            self.after(60, self._poll_preview)

    def _finish_preview(self) -> None:
        self._pv_running = False
        img, err = getattr(self, "_pv_result", (None, None))
        self.pv_cv.delete("all")
        if err is not None or img is None:
            self.lbl_pv.configure(text="preview failed: %s" % err)
        else:
            self._pv_photo = ImageTk.PhotoImage(img)
            self.pv_cv.create_image(2, 2, anchor="nw", image=self._pv_photo)
            self.pv_cv.configure(scrollregion=(0, 0, img.size[0], img.size[1] + 4))
            self.lbl_pv.configure(text="")
        if self._pv_pending:
            self._pv_pending = False
            self.schedule_preview(50)

    def save_preview_png(self) -> None:
        if self.proj.count_filled() == 0:
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".png",
                                            filetypes=[("PNG picture", "*.png")], initialfile="font-preview.png")
        if not path:
            return
        text = self.pv_entry.get("1.0", "end").rstrip("\n") or SAMPLE
        try:
            fc.render_text(self.proj.snapshot(), text, max(8, num_int(self.v_pv_size, 64)), 1600).save(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not save", str(e), parent=self)

    # ======================================================================
    # photo import
    # ======================================================================
    def open_photo(self, mode: str) -> None:
        if mode == "single":
            chars, place = chr(self.cp), "fit"
        elif mode == "template":
            chars = self.tpl_chars or "".join(chr(c) for c in self.proj.chars if c != 32)
            place = "uniform"
        else:
            chars = "".join(chr(c) for c in self.proj.chars if c != 32)
            place = "uniform"
        PhotoDialog(self, mode, chars, place)

    def apply_masks(self, masks: Dict[int, np.ndarray], overwrite: bool = True) -> None:
        n = 0
        for cp, m in masks.items():
            if not overwrite and self.proj.filled(cp):
                continue
            old = self.proj.glyphs.get(cp)
            if old is not None and old.any():
                st = self.undo.setdefault(cp, [])
                st.append(old.copy())
                self.redo[cp] = []
            self.proj.set_mask(cp, m)
            self._thumbs.pop(cp, None)
            n += 1
        self._load_glyph(self.cp)
        self.mark_dirty()
        self.schedule_preview(100)
        self.info("Imported %d glyphs. Fine-tune them on the canvas (use Glyph > Centre / Transform), then tune "
                  "spacing and style on the right." % n)

    # ======================================================================
    # project files / export
    # ======================================================================
    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = messagebox.askyesnocancel("Unsaved changes", "Save changes to the project first?", parent=self)
        if r is None:
            return False
        if r:
            return self.save_project()
        return True

    def new_project(self) -> None:
        if not self.confirm_discard():
            return
        self.proj = fc.Project()
        self.path = None
        self.undo.clear()
        self.redo.clear()
        self._thumbs.clear()
        self.cp = ord("A")
        self.push_settings_to_ui()
        self.dirty = False
        self._load_glyph(self.cp)
        self._update_title()
        self.schedule_preview(50)

    def open_project(self) -> None:
        if not self.confirm_discard():
            return
        path = filedialog.askopenfilename(parent=self, title="Open project",
                                          filetypes=[("FontStudio project", "*.fsproj"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.proj = fc.Project.load(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot open project", str(e), parent=self)
            return
        self.path = path
        self.undo.clear()
        self.redo.clear()
        self._thumbs.clear()
        self.cp = self.proj.chars[0] if self.proj.chars else ord("A")
        self.push_settings_to_ui()
        self.dirty = False
        self._load_glyph(self.cp)
        self._update_title()
        self.schedule_preview(50)
        self.info("Opened %s" % path)

    def save_project(self, as_new: bool = False) -> bool:
        path = self.path
        if as_new or not path:
            path = filedialog.asksaveasfilename(
                parent=self, title="Save project", defaultextension=".fsproj",
                filetypes=[("FontStudio project", "*.fsproj")],
                initialfile=(self.proj.s.family.replace(" ", "") or "MyFont") + ".fsproj")
            if not path:
                return False
        try:
            self.proj.save(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not save", str(e), parent=self)
            return False
        self.path = path
        self.dirty = False
        self._update_title()
        self.info("Saved %s" % path)
        return True

    def export_dialog(self) -> None:
        if self.proj.count_filled() == 0:
            messagebox.showinfo("Nothing to export", "Draw or import at least one glyph first.", parent=self)
            return
        ExportDialog(self)

    def do_export(self, kinds: List[str], folder: str, name: str, install: bool) -> None:
        snap = self.proj.snapshot()

        def work(progress: Any) -> List[str]:
            out = []
            for k, kind in enumerate(kinds):
                path = os.path.join(folder, "%s.%s" % (name, kind))
                fc.export_font(snap, path, kind, lambda i, n, k=k: progress(
                    int((k + i / max(1, n)) * 100 / len(kinds)), 100))
                out.append(path)
            if install and sys.platform == "win32":
                pick = next((p for p in out if p.endswith((".ttf", ".otf"))), None)
                if pick:
                    fc.install_font_windows(pick, snap.s.family)
                    out.append("(installed for your Windows user: %s)" % snap.s.family)
            return out

        def done(res: Any, err: Optional[BaseException]) -> None:
            if err is not None:
                messagebox.showerror("Export failed", "%s: %s" % (type(err).__name__, err), parent=self)
                return
            msg = "Created:\n\n" + "\n".join(res)
            if sys.platform == "win32" and not install:
                msg += "\n\nTo use it: right-click the .ttf file and choose Install."
            msg += "\n\nOpen the folder now?"
            if messagebox.askyesno("Font exported", msg, parent=self):
                try:
                    if sys.platform == "win32":
                        os.startfile(folder)  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        os.system('open "%s"' % folder)
                    else:
                        os.system('xdg-open "%s" >/dev/null 2>&1 &' % folder)
                except OSError:
                    pass

        run_task(self, "Building font", work, done)

    def quit_app(self) -> None:
        if self.confirm_discard():
            self.destroy()


def num_int(var: tk.Variable, default: int) -> int:
    try:
        return int(var.get())
    except (tk.TclError, ValueError):
        return default


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            try:
                ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
    app = App()
    st = ttk.Style(app)
    for th in ("vista", "clam"):
        if th in st.theme_names():
            st.theme_use(th)
            break
    st.configure("Toolbutton", padding=(8, 3))
    app.mainloop()


if __name__ == "__main__":
    main()
