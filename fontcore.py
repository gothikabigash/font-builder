"""
fontcore.py - drawing/photo bitmap -> vector outlines -> real font files.

Everything in this module is GUI independent so it can be tested headlessly.

Glyph bitmaps live on a fixed canvas (CW x CH pixels).  One em is PX_EM
pixels tall and the baseline sits BASELINE_Y pixels from the top.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import zipfile
import zlib
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Canvas geometry
# --------------------------------------------------------------------------
CW, CH = 512, 640          # glyph canvas size in pixels
PX_EM = 512.0              # pixels per em
BASELINE_Y = 450           # baseline row on the canvas

# --------------------------------------------------------------------------
# Character sets
# --------------------------------------------------------------------------
ASCII_PRINTABLE = "".join(chr(c) for c in range(33, 127))

CHARSET_PRESETS: Dict[str, str] = {
    "Basic Latin (A-Z a-z 0-9 punctuation)": ASCII_PRINTABLE,
    "Uppercase + digits + punctuation": "".join(
        c for c in ASCII_PRINTABLE if not c.islower()),
    "Lowercase + digits + punctuation": "".join(
        c for c in ASCII_PRINTABLE if not c.isupper()),
    "Letters only (A-Z a-z)": "".join(c for c in ASCII_PRINTABLE if c.isalpha()),
    "Digits only": "0123456789",
    "Latin-1 accented letters":
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ",
    "Extra punctuation & symbols": "¡¿«»“”‘’–—…€£¥©®°±×÷§¶•™",
    "Greek": "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω",
    "Cyrillic": "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя",
}


def default_charset() -> List[int]:
    return [ord(c) for c in ASCII_PRINTABLE]


def text_to_cps(text: str) -> List[int]:
    """Unique code points of a string, whitespace removed, order kept."""
    seen = set()
    out = []
    for ch in text:
        if ch.isspace() or ch in "\ufeff\u200b":
            continue
        cp = ord(ch)
        if cp not in seen:
            seen.add(cp)
            out.append(cp)
    return out


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class FontSettings:
    # names
    family: str = "My Handwriting"
    style: str = "Regular"            # Regular / Bold / Italic / Bold Italic
    designer: str = ""
    copyright: str = ""
    license: str = ""
    version: str = "1.000"
    # vertical metrics (fractions of an em; also drive the guide lines)
    upm: int = 1000
    ascender: float = 0.78
    descender: float = 0.22
    cap_height: float = 0.70
    x_height: float = 0.50
    line_gap: float = 0.10
    # spacing (fractions of an em)
    lsb: float = 0.05
    rsb: float = 0.05
    tracking: float = 0.0
    space_width: float = 0.30
    min_advance: float = 0.20
    monospace: bool = False
    mono_width: float = 0.60
    # tracing
    upscale: int = 2
    smooth: float = 1.0
    precision: float = 1.2            # curve fit tolerance in canvas px
    corner_angle: float = 50.0        # degrees; 180 = no sharp corners
    min_speck: int = 20               # drop islands smaller than this (px^2)
    straight_lines: bool = True
    roughness: float = 0.0            # hand-made wobble
    seed: int = 1
    # style transforms
    weight: float = 0.0               # px of stroke thickness added (+) / removed (-)
    slant: float = 0.0                # degrees, + leans right
    width_pct: float = 100.0
    scale_pct: float = 100.0
    baseline_shift: float = 0.0       # px, + moves glyphs up
    # extras
    case_fallback: str = "off"        # off / upper_for_lower / lower_for_upper
    add_nbsp: bool = True
    format_ttf_max_err: float = 1.0   # cubic->quadratic error (font units)


def settings_from_dict(d: dict) -> FontSettings:
    s = FontSettings()
    names = {f.name: f for f in fields(FontSettings)}
    for k, v in d.items():
        if k in names:
            try:
                setattr(s, k, type(getattr(s, k))(v))
            except (TypeError, ValueError):
                pass
    return s


# --------------------------------------------------------------------------
# Project (glyph bitmaps + settings) with .fsproj save/load
# --------------------------------------------------------------------------
class Project:
    def __init__(self) -> None:
        self.s = FontSettings()
        self.glyphs: Dict[int, np.ndarray] = {}
        self.bearings: Dict[int, List[float]] = {}   # cp -> [extra_left, extra_right] (1/1000 em)
        self.chars: List[int] = default_charset()
        self._trace_cache: Dict[tuple, list] = {}
        self.revision = 0

    # ---- glyph access
    def filled(self, cp: int) -> bool:
        m = self.glyphs.get(cp)
        return m is not None and bool(m.any())

    def get_mask(self, cp: int) -> np.ndarray:
        m = self.glyphs.get(cp)
        if m is None:
            return np.zeros((CH, CW), np.uint8)
        return m

    def set_mask(self, cp: int, mask: np.ndarray) -> None:
        mask = np.asarray(mask, np.uint8)
        if mask.shape != (CH, CW):
            raise ValueError("mask must be %dx%d" % (CW, CH))
        self.glyphs[cp] = np.where(mask > 127, 255, 0).astype(np.uint8)
        if cp not in self.chars:
            self.chars.append(cp)
        self.revision += 1

    def clear_mask(self, cp: int) -> None:
        self.glyphs.pop(cp, None)
        self.revision += 1

    def snapshot(self) -> "Project":
        """Cheap copy for background work (glyph arrays are never mutated in place)."""
        import dataclasses
        q = Project.__new__(Project)
        q.s = dataclasses.replace(self.s)
        q.glyphs = dict(self.glyphs)
        q.bearings = {k: list(v) for k, v in self.bearings.items()}
        q.chars = list(self.chars)
        q._trace_cache = self._trace_cache
        q.revision = self.revision
        return q

    def count_filled(self) -> int:
        return sum(1 for cp in self.glyphs if self.filled(cp))

    # ---- persistence
    def save(self, path: str) -> None:
        meta = {
            "format": 1,
            "settings": asdict(self.s),
            "chars": self.chars,
            "bearings": {str(k): v for k, v in self.bearings.items()},
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps(meta, indent=1))
            for cp, m in self.glyphs.items():
                if not m.any():
                    continue
                buf = io.BytesIO()
                Image.fromarray(m).save(buf, "PNG", optimize=True)
                z.writestr("glyphs/%06X.png" % cp, buf.getvalue())

    @staticmethod
    def load(path: str) -> "Project":
        p = Project()
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("project.json").decode("utf-8"))
            p.s = settings_from_dict(meta.get("settings", {}))
            p.chars = [int(c) for c in meta.get("chars", [])] or default_charset()
            p.bearings = {int(k): [float(x) for x in v]
                          for k, v in meta.get("bearings", {}).items()}
            for name in z.namelist():
                m = re.match(r"glyphs/([0-9A-Fa-f]+)\.png$", name)
                if not m:
                    continue
                img = Image.open(io.BytesIO(z.read(name))).convert("L")
                if img.size != (CW, CH):
                    img = img.resize((CW, CH))
                arr = np.array(img, np.uint8)
                p.glyphs[int(m.group(1), 16)] = np.where(arr > 127, 255, 0).astype(np.uint8)
        return p


# --------------------------------------------------------------------------
# Bezier curve fitting (Schneider, "An Algorithm for Automatically Fitting
# Digitized Curves", Graphics Gems)
# --------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return v / n


def _bez(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    return (ctrl[0] * (mt ** 3)[:, None] + ctrl[1] * (3 * mt * mt * t)[:, None]
            + ctrl[2] * (3 * mt * t * t)[:, None] + ctrl[3] * (t ** 3)[:, None])


def _bez_d1(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    return (3 * (ctrl[1] - ctrl[0]) * (mt * mt)[:, None]
            + 6 * (ctrl[2] - ctrl[1]) * (mt * t)[:, None]
            + 3 * (ctrl[3] - ctrl[2]) * (t * t)[:, None])


def _bez_d2(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    return (6 * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) * mt[:, None]
            + 6 * (ctrl[3] - 2 * ctrl[2] + ctrl[1]) * t[:, None])


def _chord_param(P: np.ndarray) -> np.ndarray:
    d = np.hypot(*(P[1:] - P[:-1]).T)
    u = np.concatenate([[0.0], np.cumsum(d)])
    if u[-1] <= 1e-12:
        return np.linspace(0, 1, len(P))
    return u / u[-1]


def _gen_bezier(P, u, t1, t2) -> np.ndarray:
    p0, p3 = P[0], P[-1]
    b0 = (1 - u) ** 3
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u * u * (1 - u)
    b3 = u ** 3
    A1 = b1[:, None] * t1
    A2 = b2[:, None] * t2
    c00 = float((A1 * A1).sum())
    c01 = float((A1 * A2).sum())
    c11 = float((A2 * A2).sum())
    tmp = P - (p0 * (b0 + b1)[:, None] + p3 * (b2 + b3)[:, None])
    x0 = float((A1 * tmp).sum())
    x1 = float((A2 * tmp).sum())
    det = c00 * c11 - c01 * c01
    seg = float(np.hypot(*(p3 - p0)))
    if abs(det) > 1e-12:
        al = (x0 * c11 - x1 * c01) / det
        ar = (c00 * x1 - c01 * x0) / det
    else:
        al = ar = 0.0
    eps = 1e-6 * seg
    if al < eps or ar < eps or seg < 1e-9:
        al = ar = seg / 3.0
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _max_error(P, ctrl, u) -> Tuple[float, int]:
    d = P - _bez(ctrl, u)
    e = (d * d).sum(1)
    if len(e) <= 2:
        return 0.0, len(P) // 2
    i = int(np.argmax(e[1:-1])) + 1
    return float(e[i]), i


def _reparam(P, u, ctrl) -> np.ndarray:
    q = _bez(ctrl, u) - P
    d1 = _bez_d1(ctrl, u)
    d2 = _bez_d2(ctrl, u)
    num = (q * d1).sum(1)
    den = (d1 * d1).sum(1) + (q * d2).sum(1)
    out = u.copy()
    ok = np.abs(den) > 1e-12
    out[ok] = u[ok] - num[ok] / den[ok]
    out = np.clip(out, 0.0, 1.0)
    # keep monotone
    return np.maximum.accumulate(out)


def _fit_curve(P: np.ndarray, t1: np.ndarray, t2: np.ndarray, err2: float,
               out: list, depth: int = 0) -> None:
    n = len(P)
    if n == 2:
        d = float(np.hypot(*(P[1] - P[0]))) / 3.0
        out.append(np.array([P[0], P[0] + t1 * d, P[1] + t2 * d, P[1]]))
        return
    u = _chord_param(P)
    ctrl = _gen_bezier(P, u, t1, t2)
    err, split = _max_error(P, ctrl, u)
    if err < err2:
        out.append(ctrl)
        return
    if err < err2 * 36:
        for _ in range(5):
            u = _reparam(P, u, ctrl)
            ctrl = _gen_bezier(P, u, t1, t2)
            err, split = _max_error(P, ctrl, u)
            if err < err2:
                out.append(ctrl)
                return
    if depth > 24:
        out.append(ctrl)
        return
    split = min(max(split, 1), n - 2)
    tc = _unit(P[split - 1] - P[split + 1])
    _fit_curve(P[:split + 1], t1, tc, err2, out, depth + 1)
    _fit_curve(P[split:], -tc, t2, err2, out, depth + 1)


# --------------------------------------------------------------------------
# Contour processing
# --------------------------------------------------------------------------
def _gauss1d(sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _smooth_closed(P: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.3 or len(P) < 8:
        return P
    k = _gauss1d(sigma)
    r = (len(k) - 1) // 2
    pad = np.pad(P, ((r, r), (0, 0)), mode="wrap")
    out = np.empty_like(P)
    for d in (0, 1):
        out[:, d] = np.convolve(pad[:, d], k, mode="valid")
    return out


def _dedupe(P: np.ndarray) -> np.ndarray:
    keep = np.ones(len(P), bool)
    keep[1:] = np.any(P[1:] != P[:-1], axis=1)
    P = P[keep]
    if len(P) > 1 and np.all(P[0] == P[-1]):
        P = P[:-1]
    return P


def _find_corners(P: np.ndarray, k: int, angle_deg: float) -> List[int]:
    n = len(P)
    if n < 4 * k or angle_deg >= 175:
        return []
    prev = np.roll(P, k, axis=0)
    nxt = np.roll(P, -k, axis=0)
    v1 = P - prev
    v2 = nxt - P
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(1)
    ang = np.degrees(np.abs(np.arctan2(cross, dot)))
    idx = np.where(ang >= angle_deg)[0]
    if len(idx) == 0:
        return []
    runs: List[List[int]] = [[int(idx[0])]]
    for i in idx[1:]:
        if i - runs[-1][-1] <= k:
            runs[-1].append(int(i))
        else:
            runs.append([int(i)])
    if len(runs) > 1 and (runs[0][0] + n - runs[-1][-1]) <= k:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    corners = []
    for r in runs:
        corners.append(max(r, key=lambda j: ang[j]))
    return sorted(set(corners))


def _tan_fwd(Q: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(Q) - 1))
    return _unit(Q[k] - Q[0])


def _tan_bwd(Q: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(Q) - 1))
    return _unit(Q[-1 - k] - Q[-1])


def _straight_run(Q: np.ndarray, tol: float) -> bool:
    d = Q[-1] - Q[0]
    L = float(np.hypot(*d))
    if L < 1e-9:
        return False
    nrm = np.array([-d[1], d[0]]) / L
    return float(np.abs((Q - Q[0]) @ nrm).max()) <= tol


def _line_fit(Q: np.ndarray, ref_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = Q.mean(0)
    _u, _s, vt = np.linalg.svd(Q - c, full_matrices=False)
    d = vt[0]
    if float(d @ ref_dir) < 0:
        d = -d
    return c, d


def _intersect(c1, d1, c2, d2) -> Optional[np.ndarray]:
    cr = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cr) < 0.2:
        return None
    w = c2 - c1
    t = (w[0] * d2[1] - w[1] * d2[0]) / cr
    return c1 + d1 * t


def _fit_closed(P: np.ndarray, corners: List[int], err2: float, kt: int,
                straight_tol: float = 0.0, trim: int = 6) -> list:
    n = len(P)
    beziers: list = []
    if not corners:
        h = n // 2
        ta = _unit(P[(0 + kt) % n] - P[(0 - kt) % n])
        tb = _unit(P[(h + kt) % n] - P[(h - kt) % n])
        _fit_curve(P[np.arange(0, h + 1) % n], ta, -tb, err2, beziers)
        _fit_curve(P[np.arange(h, n + 1) % n], tb, -ta, err2, beziers)
        return beziers

    m = len(corners)
    runs = []          # (Q, line or None)
    for a, b in zip(corners, corners[1:] + [corners[0] + n]):
        Q = P[np.arange(a, b + 1) % n]
        line = None
        if straight_tol > 0 and len(Q) >= 3:
            t = min(trim, max(0, (len(Q) - 3) // 2))
            inner = Q[t:len(Q) - t]
            if len(inner) >= 3 and _straight_run(inner, straight_tol):
                line = _line_fit(inner, Q[-1] - Q[0])
        runs.append((Q, line))

    pts = [P[c].copy() for c in corners]
    for i in range(m):
        l1 = runs[i - 1][1]
        l2 = runs[i][1]
        if l1 is not None and l2 is not None:
            X = _intersect(l1[0], l1[1], l2[0], l2[1])
            if X is not None and float(np.hypot(*(X - pts[i]))) <= 3.0 * trim:
                pts[i] = X
    for i, (Q, line) in enumerate(runs):
        p0, p3 = pts[i], pts[(i + 1) % m]
        if line is not None:
            beziers.append(np.array([p0, p0 + (p3 - p0) / 3.0, p0 + (p3 - p0) * 2.0 / 3.0, p3]))
            continue
        if len(Q) < 2:
            continue
        t1, t2 = _tan_fwd(Q, kt), _tan_bwd(Q, kt)
        Q2 = Q.copy()
        Q2[0], Q2[-1] = p0, p3
        _fit_curve(Q2, t1, t2, err2, beziers)
    return beziers


def _is_line(c: np.ndarray, tol: float) -> bool:
    d = c[3] - c[0]
    L = float(np.hypot(*d))
    if L < 1e-9:
        return False
    nrm = np.array([-d[1], d[0]]) / L
    return abs(float((c[1] - c[0]) @ nrm)) <= tol and abs(float((c[2] - c[0]) @ nrm)) <= tol


# A traced contour:  {"segs": [seg, ...], "hole": bool}
# seg = ((x,y),(x,y))                       straight line  p0 -> p1
#     | ((x,y),(x,y),(x,y),(x,y))           cubic          p0, c1, c2, p3
Contour = dict


def trace_mask(mask: np.ndarray, *, upscale: int = 2, smooth: float = 1.0,
               precision: float = 1.2, corner_angle: float = 50.0,
               min_speck: int = 20, straight: bool = True,
               roughness: float = 0.0, weight: float = 0.0,
               seed: int = 1) -> List[Contour]:
    """Turn a 0/255 bitmap into closed Bezier contours (pixel coordinates)."""
    m = (np.asarray(mask) > 127).astype(np.uint8)
    if not m.any():
        return []
    U = int(max(1, min(4, upscale)))
    margin = 10 + int(abs(weight)) + 2
    ys, xs = np.nonzero(m)
    y0 = int(ys.min()) - margin
    y1 = int(ys.max()) + margin + 1
    x0 = int(xs.min()) - margin
    x1 = int(xs.max()) + margin + 1
    sub = np.zeros((y1 - y0, x1 - x0), np.float32)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, m.shape[0]), min(x1, m.shape[1])
    sub[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = m[sy0:sy1, sx0:sx1]

    if smooth > 0:
        sub = cv2.GaussianBlur(sub, (0, 0), sigmaX=max(0.55, 0.7 * smooth))
    big = cv2.resize(sub, None, fx=U, fy=U, interpolation=cv2.INTER_CUBIC)
    big = (big >= 0.5).astype(np.uint8)

    if abs(weight) > 1e-6:
        r = abs(weight) * U / 2.0
        if weight > 0:
            dist = cv2.distanceTransform((big == 0).astype(np.uint8), cv2.DIST_L2, 5)
            big = ((dist <= r) | (big > 0)).astype(np.uint8)
        else:
            dist = cv2.distanceTransform(big, cv2.DIST_L2, 5)
            big = (dist > r).astype(np.uint8)
        if not big.any():
            return []

    found = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hier = found[-2], found[-1]
    if hier is None:
        return []
    hier = hier[0]
    out: List[Contour] = []
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    err2 = (max(0.05, precision) * U) ** 2
    kcorner = max(3, int(round(2.5 * U)))
    for ci, c in enumerate(contours):
        if abs(cv2.contourArea(c)) < max(4, min_speck) * U * U:
            continue
        P = c.reshape(-1, 2).astype(np.float64)
        P = _dedupe(P)
        if len(P) < 8:
            continue
        if roughness > 0:
            noise = rng.normal(size=P.shape)
            noise = _smooth_closed(noise, 7.0 * U)
            sd = float(noise.std())
            if sd > 1e-9:
                P = P + noise / sd * (0.9 * roughness * U)
        P = _smooth_closed(P, max(0.0, 1.0 * U * smooth))
        corners = _find_corners(P, kcorner, corner_angle)
        beziers = _fit_closed(P, corners, err2, kcorner,
                              0.55 * precision * U if straight else 0.0,
                              trim=int(round(2.0 * U * max(smooth, 0.5))) + 2)
        if not beziers:
            continue
        segs = []
        for b in beziers:
            pts = [((float(p[0]) + 0.5) / U + x0, (float(p[1]) + 0.5) / U + y0) for p in b]
            if straight and _is_line(b, 0.02 * U):
                segs.append((pts[0], pts[3]))
            else:
                segs.append(tuple(pts))
        out.append({"segs": segs, "hole": hier[ci][3] != -1})
    return out


# --------------------------------------------------------------------------
# Geometry helpers on traced contours
# --------------------------------------------------------------------------
def _map_contour(c: Contour, fn: Callable[[float, float], Tuple[float, float]]) -> Contour:
    return {"segs": [tuple(fn(*p) for p in s) for s in c["segs"]], "hole": c["hole"]}


def _signed_area(c: Contour) -> float:
    pts = []
    for s in c["segs"]:
        if len(s) == 2:
            pts.append(s[0])
        else:
            # sample the bezier so the sign is reliable
            ctrl = np.array(s)
            t = np.linspace(0, 1, 6, endpoint=False)
            for q in _bez(ctrl, t):
                pts.append((float(q[0]), float(q[1])))
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def _reverse(c: Contour) -> Contour:
    return {"segs": [tuple(reversed(s)) for s in reversed(c["segs"])], "hole": c["hole"]}


def _draw(contours: List[Contour], pen, ttf: bool) -> None:
    """Draw contours on a fontTools pen with proper winding direction.

    TrueType: outer contours clockwise.  PostScript/CFF: counter-clockwise.
    Coordinates must be y-up.
    """
    for c in contours:
        area = _signed_area(c)               # >0 = counter-clockwise (y up)
        want_ccw = (not ttf)
        if c["hole"]:
            want_ccw = not want_ccw
        if (area > 0) != want_ccw:
            c = _reverse(c)
        segs = c["segs"]
        start = segs[0][0]
        pen.moveTo((round(start[0], 2), round(start[1], 2)))
        for i, s in enumerate(segs):
            last = i == len(segs) - 1
            if len(s) == 2:
                if last:
                    continue                 # closePath draws the closing line
                pen.lineTo((round(s[1][0], 2), round(s[1][1], 2)))
            else:
                pen.curveTo(tuple(np.round(s[1], 2)), tuple(np.round(s[2], 2)),
                            tuple(np.round(s[3], 2)))
        pen.closePath()


def _bounds(contours: List[Contour]) -> Optional[Tuple[float, float, float, float]]:
    from fontTools.pens.boundsPen import BoundsPen
    bp = BoundsPen(None)
    _draw(contours, bp, True)
    return bp.bounds


# --------------------------------------------------------------------------
# Glyph preparation (bitmap -> font units + metrics)
# --------------------------------------------------------------------------
def _trace_key(s: FontSettings) -> tuple:
    return (s.upscale, round(s.smooth, 3), round(s.precision, 3), round(s.corner_angle, 2),
            s.min_speck, s.straight_lines, round(s.roughness, 3), s.seed, round(s.weight, 3))


def traced_contours(proj: Project, cp: int, mask: np.ndarray) -> List[Contour]:
    s = proj.s
    key = (cp, zlib.crc32(mask.tobytes()), _trace_key(s))
    hit = proj._trace_cache.get(key)
    if hit is not None:
        return hit
    res = trace_mask(mask, upscale=s.upscale, smooth=s.smooth, precision=s.precision,
                     corner_angle=s.corner_angle, min_speck=s.min_speck,
                     straight=s.straight_lines, roughness=s.roughness,
                     weight=s.weight, seed=s.seed + cp)
    if len(proj._trace_cache) > 4000:
        proj._trace_cache.clear()
    proj._trace_cache[key] = res
    return res


@dataclass
class BuiltGlyph:
    cp: int
    name: str
    contours: List[Contour]       # font units, y-up, positioned (lsb applied)
    advance: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def _glyph_name(cp: int) -> str:
    from fontTools.agl import UV2AGL
    if cp == 0x20:
        return "space"
    name = UV2AGL.get(cp)
    if name:
        return name
    return ("uni%04X" % cp) if cp <= 0xFFFF else ("u%X" % cp)


def effective_masks(proj: Project) -> Dict[int, np.ndarray]:
    """Glyph masks that will go into the font (incl. case fallback)."""
    out = {cp: m for cp, m in proj.glyphs.items() if m.any() and cp != 0x20}
    mode = proj.s.case_fallback
    if mode in ("upper_for_lower", "lower_for_upper"):
        for cp in list(proj.glyphs.keys()):
            if not proj.filled(cp) or cp > 0xFFFF:
                continue
            ch = chr(cp)
            if mode == "upper_for_lower":
                if not ch.isupper():
                    continue
                tgt = ch.lower()
            else:
                if not ch.islower():
                    continue
                tgt = ch.upper()
            if len(tgt) == 1 and tgt != ch and ord(tgt) not in out:
                out[ord(tgt)] = proj.glyphs[cp]
    return out


def build_glyphs(proj: Project, progress: Optional[Callable[[int, int], None]] = None
                 ) -> Dict[int, BuiltGlyph]:
    s = proj.s
    upm = float(s.upm)
    k = upm / PX_EM
    masks = effective_masks(proj)
    tan_sl = math.tan(math.radians(s.slant))
    sc = s.scale_pct / 100.0
    wsc = s.width_pct / 100.0
    result: Dict[int, BuiltGlyph] = {}
    items = sorted(masks.items())
    for n, (cp, mask) in enumerate(items):
        if progress:
            progress(n, len(items))
        contours = traced_contours(proj, cp, mask)
        if not contours:
            continue

        def fn(x: float, y: float) -> Tuple[float, float]:
            fx = (x - CW / 2.0) * k
            fy = (BASELINE_Y - y + s.baseline_shift) * k
            fx *= sc
            fy *= sc
            fx += fy * tan_sl
            fx *= wsc
            return fx, fy

        cs = [_map_contour(c, fn) for c in contours]
        b = _bounds(cs)
        if b is None:
            continue
        xmin, ymin, xmax, ymax = b
        dl, dr = proj.bearings.get(cp, [0.0, 0.0])
        lsb = s.lsb * upm + dl * upm / 1000.0
        rsb = s.rsb * upm + dr * upm / 1000.0
        width = xmax - xmin
        if s.monospace:
            adv = s.mono_width * upm
            dx = (adv - width) / 2.0 - xmin + (dl - dr) * upm / 2000.0
        else:
            adv = width + lsb + rsb + s.tracking * upm
            dx = lsb - xmin
            if adv < s.min_advance * upm:
                dx += (s.min_advance * upm - adv) / 2.0
                adv = s.min_advance * upm
        cs = [_map_contour(c, lambda x, y: (x + dx, y)) for c in cs]
        result[cp] = BuiltGlyph(cp, _glyph_name(cp), cs, int(round(max(adv, 1))),
                                xmin + dx, ymin, xmax + dx, ymax)
    if progress:
        progress(len(items), len(items))
    return result


# --------------------------------------------------------------------------
# Font file creation
# --------------------------------------------------------------------------
FORMATS = {
    "ttf": "TrueType (.ttf)",
    "otf": "OpenType / CFF (.otf)",
    "woff": "WOFF web font (.woff)",
    "woff2": "WOFF2 web font (.woff2)",
}


def available_formats() -> List[str]:
    out = ["ttf", "otf", "woff"]
    try:
        import brotli  # noqa: F401
        out.append("woff2")
    except Exception:
        pass
    return out


def _ps_name(family: str, style: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]", "", family) or "Font"
    st = re.sub(r"[^A-Za-z0-9]", "", style) or "Regular"
    return (base + "-" + st)[:63]


def build_ttfont(proj: Project, kind: str = "ttf",
                 progress: Optional[Callable[[int, int], None]] = None):
    """Return a fontTools TTFont built from the project."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    s = proj.s
    upm = int(s.upm)
    glyphs = build_glyphs(proj, progress)
    is_ttf = kind in ("ttf", "woff", "woff2")

    order = [".notdef", "space"]
    cmap: Dict[int, str] = {0x20: "space"}
    if s.add_nbsp:
        cmap[0xA0] = "space"
    for cp in sorted(glyphs):
        g = glyphs[cp]
        if g.name in order:
            continue
        order.append(g.name)
        cmap[cp] = g.name

    # .notdef : a hollow box
    nd_adv = int(0.55 * upm)
    pad = int(0.08 * upm)
    top = int(max(s.cap_height, 0.5) * upm)
    box_out = {"segs": [((pad, 0), (pad, top)), ((pad, top), (nd_adv - pad, top)),
                        ((nd_adv - pad, top), (nd_adv - pad, 0)), ((nd_adv - pad, 0), (pad, 0))],
               "hole": False}
    inn = int(0.04 * upm)
    box_in = {"segs": [((pad + inn, inn), (pad + inn, top - inn)),
                       ((pad + inn, top - inn), (nd_adv - pad - inn, top - inn)),
                       ((nd_adv - pad - inn, top - inn), (nd_adv - pad - inn, inn)),
                       ((nd_adv - pad - inn, inn), (pad + inn, inn))],
              "hole": True}
    notdef = [box_out, box_in]
    space_adv = int(round(s.space_width * upm))
    if s.monospace:
        space_adv = int(round(s.mono_width * upm))

    advances: Dict[str, int] = {".notdef": nd_adv, "space": space_adv}
    bounds: Dict[str, Tuple[float, float, float, float]] = {}
    for g in glyphs.values():
        advances[g.name] = g.advance
        bounds[g.name] = (g.xmin, g.ymin, g.xmax, g.ymax)
    bounds[".notdef"] = (pad, 0, nd_adv - pad, top)

    fb = FontBuilder(upm, isTTF=is_ttf)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)

    contour_map: Dict[str, List[Contour]] = {".notdef": notdef, "space": []}
    for g in glyphs.values():
        contour_map[g.name] = g.contours

    if is_ttf:
        table = {}
        for name in order:
            pen = TTGlyphPen(None)
            if contour_map[name]:
                _draw(contour_map[name], Cu2QuPen(pen, max(0.5, s.format_ttf_max_err),
                                                  reverse_direction=False), True)
            table[name] = pen.glyph()
        fb.setupGlyf(table)
        metrics = {}
        for name in order:
            gl = table[name]
            lsb = int(getattr(gl, "xMin", 0)) if contour_map[name] else 0
            metrics[name] = (advances[name], lsb)
    else:
        charstrings = {}
        for name in order:
            pen = T2CharStringPen(advances[name], None)
            if contour_map[name]:
                _draw(contour_map[name], pen, False)
            charstrings[name] = pen.getCharString()
        ps = _ps_name(s.family, s.style)
        full = s.family if s.style == "Regular" else "%s %s" % (s.family, s.style)
        fb.setupCFF(ps, {"FullName": full, "FamilyName": s.family, "Weight": s.style},
                    charstrings, {})
        metrics = {}
        for name in order:
            b = bounds.get(name)
            metrics[name] = (advances[name], int(round(b[0])) if b else 0)
    fb.setupHorizontalMetrics(metrics)

    ymax = max([b[3] for b in bounds.values()] + [s.ascender * upm])
    ymin = min([b[1] for b in bounds.values()] + [-s.descender * upm])
    asc = int(round(s.ascender * upm))
    desc = -int(round(s.descender * upm))
    gap = int(round(s.line_gap * upm))
    fb.setupHorizontalHeader(ascent=asc, descent=desc, lineGap=gap)

    bold = "Bold" in s.style
    italic = "Italic" in s.style or "Oblique" in s.style or abs(s.slant) > 0.01
    fs_sel = (0x20 if bold else 0) | (0x01 if italic else 0)
    if not bold and not italic:
        fs_sel = 0x40
    from fontTools.misc.timeTools import timestampNow
    now = timestampNow()
    fb.setupHead(unitsPerEm=upm, macStyle=(1 if bold else 0) | (2 if italic else 0),
                 created=now, modified=now)
    fb.setupOS2(sTypoAscender=asc, sTypoDescender=desc, sTypoLineGap=gap,
                usWinAscent=int(math.ceil(ymax)), usWinDescent=int(math.ceil(-ymin)),
                sxHeight=int(round(s.x_height * upm)), sCapHeight=int(round(s.cap_height * upm)),
                usWeightClass=700 if bold else 400, fsSelection=fs_sel,
                achVendID="FSTD", fsType=0)
    ps = _ps_name(s.family, s.style)
    full = s.family if s.style == "Regular" else "%s %s" % (s.family, s.style)
    names = {"familyName": s.family, "styleName": s.style, "uniqueFontIdentifier":
             "%s;%s;FontStudio" % (s.version, ps), "fullName": full,
             "psName": ps, "version": "Version %s" % s.version}
    if s.copyright:
        names["copyright"] = s.copyright
    if s.designer:
        names["designer"] = s.designer
    if s.license:
        names["licenseDescription"] = s.license
    fb.setupNameTable(names)
    fb.setupPost(italicAngle=-s.slant if italic else 0, isFixedPitch=1 if s.monospace else 0,
                 underlinePosition=-int(0.1 * upm), underlineThickness=int(0.05 * upm))
    font = fb.font
    if kind == "woff":
        font.flavor = "woff"
    elif kind == "woff2":
        font.flavor = "woff2"
    return font


def font_bytes(proj: Project, kind: str = "ttf",
               progress: Optional[Callable[[int, int], None]] = None) -> bytes:
    font = build_ttfont(proj, kind, progress)
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def export_font(proj: Project, path: str, kind: Optional[str] = None,
                progress: Optional[Callable[[int, int], None]] = None) -> str:
    if kind is None:
        kind = os.path.splitext(path)[1].lstrip(".").lower() or "ttf"
    data = font_bytes(proj, kind, progress)
    with open(path, "wb") as f:
        f.write(data)
    return path


def install_font_windows(path: str, family: str) -> str:
    """Install a font for the current user (Windows 10 1809+ / 11)."""
    if sys.platform != "win32":
        raise RuntimeError("Font installation is only available on Windows.")
    import ctypes
    import shutil
    import winreg  # type: ignore[import-not-found]
    fonts_dir = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "Windows", "Fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    dest = os.path.join(fonts_dir, os.path.basename(path))
    shutil.copy2(path, dest)
    suffix = "(OpenType)" if dest.lower().endswith(".otf") else "(TrueType)"
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows NT\CurrentVersion\Fonts")
    winreg.SetValueEx(key, "%s %s" % (family, suffix), 0, winreg.REG_SZ, dest)
    winreg.CloseKey(key)
    ctypes.windll.gdi32.AddFontResourceW(dest)                 # type: ignore[attr-defined]
    ctypes.windll.user32.SendMessageTimeoutW(                  # type: ignore[attr-defined]
        0xFFFF, 0x001D, 0, 0, 0x0002, 1000, None)
    return dest


# --------------------------------------------------------------------------
# Preview rendering (uses the real built font through PIL/FreeType)
# --------------------------------------------------------------------------
def render_text(proj: Project, text: str, size: int = 64, width: int = 900,
                fg: int = 20, bg: int = 255, kind: str = "ttf",
                data: Optional[bytes] = None) -> Image.Image:
    if data is None:
        data = font_bytes(proj, kind)
    font = ImageFont.truetype(io.BytesIO(data), size)
    lines: List[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = (cur + " " + word) if cur else word
            if font.getlength(trial) <= width - 20 or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    lh = int(size * 1.35)
    img = Image.new("L", (width, max(lh * len(lines) + 20, 60)), bg)
    d = ImageDraw.Draw(img)
    y = 10
    for ln in lines:
        d.text((10, y), ln, font=font, fill=fg)
        y += lh
    return img


# --------------------------------------------------------------------------
# Reference fonts (stamp an existing font glyph as a starting point)
# --------------------------------------------------------------------------
def system_font_dirs() -> List[str]:
    dirs = []
    if sys.platform == "win32":
        dirs.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        la = os.environ.get("LOCALAPPDATA")
        if la:
            dirs.append(os.path.join(la, "Microsoft", "Windows", "Fonts"))
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    else:
        dirs += ["/usr/share/fonts", os.path.expanduser("~/.fonts")]
    return [d for d in dirs if os.path.isdir(d)]


def list_system_fonts() -> Dict[str, str]:
    found: Dict[str, str] = {}
    for d in system_font_dirs():
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith((".ttf", ".otf")):
                    found.setdefault(os.path.splitext(fn)[0], os.path.join(root, fn))
            if len(found) > 3000:
                break
    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


def reference_mask(ch: str, font_path: str, scale: float = 1.0) -> np.ndarray:
    """Render one character from a font file onto the glyph canvas."""
    font = ImageFont.truetype(font_path, max(8, int(PX_EM * scale)))
    img = Image.new("L", (CW, CH), 0)
    d = ImageDraw.Draw(img)
    d.text((CW / 2.0, BASELINE_Y), ch, font=font, fill=255, anchor="ms")
    return np.where(np.array(img) > 127, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Bitmap transforms used by the editor
# --------------------------------------------------------------------------
def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(mask)
    ys0, ys1 = max(0, dy), min(CH, CH + dy)
    xs0, xs1 = max(0, dx), min(CW, CW + dx)
    out[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def affine_mask(mask: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0,
                rotate: float = 0.0, shear: float = 0.0) -> np.ndarray:
    """Transform about the baseline/centre point of the canvas."""
    cx, cy = CW / 2.0, float(BASELINE_Y)
    a = math.radians(rotate)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    S = np.array([[scale_x, 0], [0, scale_y]])
    H = np.array([[1, math.tan(math.radians(shear))], [0, 1]])
    A = R @ H @ S
    t = np.array([cx, cy]) - A @ np.array([cx, cy])
    M = np.hstack([A, t[:, None]]).astype(np.float32)
    out = cv2.warpAffine(mask.astype(np.float32), M, (CW, CH), flags=cv2.INTER_CUBIC)
    return np.where(out > 127, 255, 0).astype(np.uint8)


def auto_center(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    dx = int(round(CW / 2.0 - (xs.min() + xs.max() + 1) / 2.0))
    return shift_mask(mask, dx, 0)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
