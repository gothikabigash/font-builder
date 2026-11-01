"""Tk-free tests of drawing helpers (Tk is stubbed so this runs anywhere)."""
import os, sys, types
from unittest import mock
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
try:
    import tkinter  # noqa
except ImportError:
    tk = types.ModuleType("tkinter"); tk.Tk = type("Tk", (), {}); tk.Toplevel = type("Toplevel", (), {})
    tk.Variable = tk.Misc = tk.Menu = tk.Canvas = tk.Text = object
    tk.TclError = Exception
    for n in ("StringVar", "IntVar", "DoubleVar", "BooleanVar"): setattr(tk, n, mock.MagicMock)
    ttk = mock.MagicMock(); fd = mock.MagicMock(); sd = mock.MagicMock()
    ttk.Spinbox = ttk.Frame = object
    tk.ttk, tk.filedialog, tk.messagebox, tk.simpledialog = ttk, fd, mock.MagicMock(), sd
    sys.modules.update({"tkinter": tk, "tkinter.ttk": ttk, "tkinter.filedialog": fd,
                        "tkinter.messagebox": tk.messagebox, "tkinter.simpledialog": sd})
    sys.modules["PIL.ImageTk"] = mock.MagicMock()
    import PIL; PIL.ImageTk = sys.modules["PIL.ImageTk"]
import numpy as np
import fontstudio as fs
import fontcore as fc

def test_nibs_and_strokes():
    for shape in fs.NIBS:
        m = np.zeros((fc.CH, fc.CW), np.uint8)
        fs.stroke(m, (100, 100), (300, 200), fs.nib_points(shape, 30, 40), 255)
        assert m.sum() > 0, shape
        fs.stroke(m, (100, 100), (300, 200), fs.nib_points("Round", 60, 0), 0)
        assert m[150, 200] == 0
    r = np.zeros((fc.CH, fc.CW), np.uint8); fs.stroke(r, (200, 300), (200, 300), fs.nib_points("Round", 40, 0), 255)
    assert 1100 < (r > 0).sum() < 1400      # ~ pi*20^2
    flat = np.zeros_like(r); fs.stroke(flat, (200, 300), (200, 300), fs.nib_points("Flat pen (calligraphy)", 60, 0), 255)
    ys, xs = np.nonzero(flat); assert (xs.max() - xs.min()) > 2.5 * (ys.max() - ys.min())

def test_presets_valid():
    s = fc.FontSettings()
    for name, d in fs.STYLE_PRESETS.items():
        for k in d: assert hasattr(s, k), (name, k)
    for _t, rows in fs.TABS:
        for spec in rows: assert hasattr(s, spec[0]), spec[0]

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
