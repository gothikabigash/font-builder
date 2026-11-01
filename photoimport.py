"""
photoimport.py - turn photos / scans into glyph bitmaps.

Four import modes
  template : printed template sheet (see make_template) - the sheet frame is
             found, perspective corrected and every cell mapped exactly.
  grid     : a plain sheet with N rows x M columns, one character per cell.
  auto     : free layout; shapes are detected, grouped into lines, merged
             (i-dots, accents ...) and assigned to the characters you type.
  single   : the whole photo is one character.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fontcore import BASELINE_Y, CH, CW, PX_EM, FontSettings

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass
class PhotoParams:
    mode: str = "auto"               # template / grid / auto / single
    placement: str = "uniform"       # uniform / class / fit   (not used by template)
    rows: int = 9
    cols: int = 8
    page: int = 1                    # template page (1-based)
    threshold_method: str = "otsu"   # otsu / adaptive / manual
    threshold: int = 140             # manual: pixels darker than this are ink
    adaptive_block: int = 41
    adaptive_c: int = 12
    invert: bool = False
    normalize_light: bool = True
    deskew: bool = True
    blur: float = 0.8
    min_area: int = 40
    close_px: int = 0
    merge_dots: bool = True
    merge_gap: int = 0
    cell_margin_pct: float = 3.0
    crop_to_ink: bool = True
    max_side: int = 2400


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------
def load_gray(path_or_img, max_side: int = 2400) -> np.ndarray:
    if isinstance(path_or_img, Image.Image):
        img = path_or_img
    else:
        img = Image.open(path_or_img)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = np.array(img.convert("L"), np.uint8)
    h, w = g.shape
    if max(h, w) > max_side:
        f = max_side / float(max(h, w))
        g = cv2.resize(g, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    return g


def normalize_lighting(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimate of the paper brightness (kills shadows/gradients)."""
    h, w = gray.shape
    sc = 600.0 / max(h, w)
    small = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else gray.copy()
    k = max(7, int(0.04 * max(small.shape)) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Prepared:
    gray: np.ndarray       # normalised grey image (paper ~255)
    soft: np.ndarray       # float32 ink strength 0..255
    binary: np.ndarray     # uint8 0/255
    thresh: float          # threshold on `soft`


def binarize(gray: np.ndarray, p: PhotoParams) -> Prepared:
    g = gray.astype(np.float32)
    if p.invert:
        g = 255.0 - g
    if p.blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), p.blur)
    gu = np.clip(g, 0, 255).astype(np.uint8)
    if p.threshold_method == "adaptive":
        bs = max(3, int(p.adaptive_block) | 1)
        b = cv2.adaptiveThreshold(gu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                  bs, int(p.adaptive_c))
        return Prepared(gu, b.astype(np.float32), b, 127.5)
    if p.threshold_method == "manual":
        tn = float(p.threshold)
    else:
        tmp = np.where(gu > 205, 255, gu).astype(np.uint8)
        tn, _ = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tn = float(min(tn, 200.0))
    soft = 255.0 - g
    T = 255.0 - tn
    binary = np.where(soft >= T, 255, 0).astype(np.uint8)
    return Prepared(gu, soft.astype(np.float32), binary, T)


def remove_specks(prep: Prepared, min_area: int) -> None:
    if min_area <= 1:
        return
    n, lab, stats, _ = cv2.connectedComponentsWithStats(prep.binary, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < min_area
    kill = small[lab]
    prep.binary[kill] = 0
    prep.soft[kill] = 0


def estimate_skew(binary: np.ndarray, max_angle: float = 10.0) -> float:
    h, w = binary.shape
    sc = min(1.0, 700.0 / max(h, w))
    small = cv2.resize(binary, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32)
    if small.sum() < 50:
        return 0.0
    cx, cy = small.shape[1] / 2.0, small.shape[0] / 2.0

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        r = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]), flags=cv2.INTER_LINEAR)
        prof = r.sum(1)
        return float(np.var(prof))

    best = 0.0
    bs = score(0.0)
    for a in np.arange(-max_angle, max_angle + 1e-9, 1.0):
        s = score(float(a))
        if s > bs:
            bs, best = s, float(a)
    for a in np.arange(best - 1.0, best + 1.0 + 1e-9, 0.2):
        s = score(float(a))
        if s > bs:
            bs, best = s, float(a)
    return best


def rotate_image(img: np.ndarray, angle: float, border: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


def prepare(gray: np.ndarray, p: PhotoParams, allow_deskew: bool = True) -> Prepared:
    g = normalize_lighting(gray) if p.normalize_light else gray.copy()
    prep = binarize(g, p)
    if allow_deskew and p.deskew:
        remove_specks(prep, max(p.min_area, 20))
        ang = estimate_skew(prep.binary)
        if abs(ang) > 0.15:
            g = rotate_image(g, ang, 255)
            prep = binarize(g, p)
    remove_specks(prep, p.min_area)
    if p.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.close_px + 1, 2 * p.close_px + 1))
        prep.binary = cv2.morphologyEx(prep.binary, cv2.MORPH_CLOSE, k)
        prep.soft = np.maximum(prep.soft, prep.binary.astype(np.float32))
    return prep


# --------------------------------------------------------------------------
# Items (detected shapes)
# --------------------------------------------------------------------------
@dataclass
class Item:
    bbox: Tuple[int, int, int, int]          # x0, y0, x1, y1 (exclusive)
    mask: np.ndarray                          # uint8 0/255, cropped to bbox
    soft: np.ndarray                          # float32, cropped to bbox
    line: int = 0
    cp: Optional[int] = None
    discarded: bool = False

    @property
    def w(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> int:
        return self.bbox[3] - self.bbox[1]


def _make_item(prep: Prepared, mask_full: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               line: int = 0) -> Item:
    m = mask_full[y0:y1, x0:x1]
    dil = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2) > 0
    soft = prep.soft[y0:y1, x0:x1] * dil
    return Item((x0, y0, x1, y1), m.copy(), soft.astype(np.float32), line)


def _tight(binary: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def detect_items(prep: Prepared, p: PhotoParams) -> List[Item]:
    """Auto mode: connected components -> merged groups -> lines -> reading order."""
    b = prep.binary
    src = b
    if p.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.close_px + 1,) * 2)
        src = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a >= max(p.min_area, 4):
            comps.append({"id": i, "x0": x, "y0": y, "x1": x + w, "y1": y + h, "area": a})
    if not comps:
        return []
    parent = list(range(len(comps)))
    areas = np.array([c["area"] for c in comps], float)
    big_h = [c["y1"] - c["y0"] for c in comps if c["area"] >= 0.4 * float(np.median(areas))]
    hmed = float(np.median(big_h)) if big_h else float(max(c["y1"] - c["y0"] for c in comps))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(comps)):
        a = comps[i]
        for j in range(i + 1, len(comps)):
            c = comps[j]
            big, small = (a, c) if a["area"] >= c["area"] else (c, a)
            xov = min(a["x1"], c["x1"]) - max(a["x0"], c["x0"])
            gap_y = max(a["y0"], c["y0"]) - min(a["y1"], c["y1"])
            gap_x = max(a["x0"], c["x0"]) - min(a["x1"], c["x1"])
            sw = small["x1"] - small["x0"]
            bh = big["y1"] - big["y0"]
            if p.merge_dots:
                stacked = xov >= 0.5 * min(sw, big["x1"] - big["x0"])
                # a dot / accent above or below a body  (i j ! ? ; é ñ ...)
                if stacked and small["area"] < 0.7 * big["area"] and gap_y <= 0.5 * bh \
                        and (small["y1"] - small["y0"]) < 0.6 * bh:
                    union(i, j)
                    continue
                # two small marks stacked (: = ; ")
                if stacked and gap_y <= 0.8 * max(a["y1"] - a["y0"], c["y1"] - c["y0"]) \
                        and max(a["y1"] - a["y0"], c["y1"] - c["y0"]) < 0.55 * hmed:
                    union(i, j)
                    continue
            if p.merge_gap > 0 and gap_x <= p.merge_gap and gap_y <= p.merge_gap:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(comps)):
        groups[find(i)].append(i)
    items: List[Item] = []
    for members in groups.values():
        ids = [comps[m]["id"] for m in members]
        x0 = min(comps[m]["x0"] for m in members)
        y0 = min(comps[m]["y0"] for m in members)
        x1 = max(comps[m]["x1"] for m in members)
        y1 = max(comps[m]["y1"] for m in members)
        sel = np.isin(labels[y0:y1, x0:x1], ids) & (b[y0:y1, x0:x1] > 0)
        full = np.zeros_like(b)
        full[y0:y1, x0:x1] = sel.astype(np.uint8) * 255
        tb = _tight(sel)
        if tb is None:
            continue
        tx0, ty0, tx1, ty1 = tb
        items.append(_make_item(prep, full, x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1))
    assign_lines(items)
    return items


def assign_lines(items: List[Item]) -> None:
    """Group items into text lines and sort them into reading order (in place)."""
    if not items:
        return
    hs = np.array([it.h for it in items], float)
    hmed = float(np.median(hs[hs >= 0.5 * np.median(hs)]))
    order = sorted(items, key=lambda it: (it.bbox[1] + it.bbox[3]) / 2.0)
    lines: List[List[Item]] = []
    refs: List[float] = []
    for it in order:
        cy = (it.bbox[1] + it.bbox[3]) / 2.0
        placed = False
        for li, ref in enumerate(refs):
            if abs(cy - ref) <= 0.6 * hmed:
                lines[li].append(it)
                tall = [m for m in lines[li] if m.h >= 0.5 * hmed]
                refs[li] = float(np.median([(m.bbox[1] + m.bbox[3]) / 2.0 for m in tall])) if tall else cy
                placed = True
                break
        if not placed:
            lines.append([it])
            refs.append(cy)
    idx = sorted(range(len(lines)), key=lambda i: refs[i])
    items.clear()
    for ln, li in enumerate(idx):
        for it in sorted(lines[li], key=lambda m: m.bbox[0]):
            it.line = ln
            items.append(it)


def grid_items(prep: Prepared, p: PhotoParams) -> List[Item]:
    """Grid mode: split into rows x cols cells, all ink of a cell = one glyph."""
    b = prep.binary
    H, W = b.shape
    x0, y0, x1, y1 = 0, 0, W, H
    if p.crop_to_ink:
        tb = _tight(b)
        if tb:
            x0, y0, x1, y1 = tb
    cw = (x1 - x0) / float(p.cols)
    chh = (y1 - y0) / float(p.rows)
    mx = int(cw * p.cell_margin_pct / 100.0)
    my = int(chh * p.cell_margin_pct / 100.0)
    items: List[Item] = []
    for r in range(p.rows):
        for c in range(p.cols):
            cx0 = int(round(x0 + c * cw)) + mx
            cx1 = int(round(x0 + (c + 1) * cw)) - mx
            cy0 = int(round(y0 + r * chh)) + my
            cy1 = int(round(y0 + (r + 1) * chh)) - my
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            sub = np.zeros_like(b)
            sub[cy0:cy1, cx0:cx1] = b[cy0:cy1, cx0:cx1]
            tb = _tight(sub)
            if tb is None:
                it = Item((cx0, cy0, cx0 + 1, cy0 + 1), np.zeros((1, 1), np.uint8),
                          np.zeros((1, 1), np.float32), r)
                it.discarded = True
                items.append(it)
                continue
            items.append(_make_item(prep, sub, tb[0], tb[1], tb[2], tb[3], r))
    return items


def single_item(prep: Prepared) -> List[Item]:
    tb = _tight(prep.binary)
    if tb is None:
        return []
    return [_make_item(prep, prep.binary, *tb, 0)]


def merge_items(prep: Prepared, a: Item, b: Item) -> Item:
    """Join two detected shapes into one (used by the review window)."""
    x0 = min(a.bbox[0], b.bbox[0]); y0 = min(a.bbox[1], b.bbox[1])
    x1 = max(a.bbox[2], b.bbox[2]); y1 = max(a.bbox[3], b.bbox[3])
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    soft = np.zeros((y1 - y0, x1 - x0), np.float32)
    for it in (a, b):
        ox, oy = it.bbox[0] - x0, it.bbox[1] - y0
        mask[oy:oy + it.h, ox:ox + it.w] |= it.mask
        soft[oy:oy + it.h, ox:ox + it.w] = np.maximum(soft[oy:oy + it.h, ox:ox + it.w], it.soft)
    return Item((x0, y0, x1, y1), mask, soft, a.line)


def assign_chars(items: List[Item], text: str, positional: bool = False) -> Tuple[int, int]:
    """Assign characters to items (grid mode keeps empty cells as placeholders).

    A newline in `text` starts a new line; if the number of lines matches the
    detected lines the characters are matched line by line.
    Returns (items_assigned, characters_available)."""
    active = list(items) if positional else [it for it in items if not it.discarded]
    for it in items:
        it.cp = None
    rows = [[ord(c) for c in ln if not c.isspace()] for ln in text.split("\n")]
    rows = [r for r in rows if r]
    nlines = (max((it.line for it in active), default=-1) + 1)
    total = sum(len(r) for r in rows)
    done = 0
    if len(rows) > 1 and len(rows) == nlines and not positional:
        by_line: Dict[int, List[Item]] = defaultdict(list)
        for it in active:
            by_line[it.line].append(it)
        for li, chars in enumerate(rows):
            lst = by_line.get(li, [])
            for it, cp in zip(lst, chars):
                it.cp = cp
                done += 1
    else:
        flat = [c for r in rows for c in r]
        for it, cp in zip(active, flat):
            it.cp = cp
            done += 1
    return done, total


# --------------------------------------------------------------------------
# Placement of items onto the glyph canvas
# --------------------------------------------------------------------------
CAPS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
XH_CHARS = set("acemnorsuvwxz")
ASC_CHARS = set("bdfhiklt")
DESC_CHARS = set("gpqy")
BASELINE_CHARS = set("ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789abcdefhiklmnorstuvwxz")


def _class_extent(ch: str, s: FontSettings) -> Optional[Tuple[float, float]]:
    """(top, bottom) in em above baseline for letters whose height is known."""
    if ch in CAPS:
        return s.cap_height, 0.0
    if ch in XH_CHARS:
        return s.x_height, 0.0
    if ch == "t":
        return s.x_height + 0.55 * (s.ascender - s.x_height), 0.0
    if ch in ASC_CHARS:
        return s.ascender, 0.0
    if ch in DESC_CHARS:
        return s.x_height, -s.descender
    if ch == "j":
        return s.ascender, -s.descender
    return None


def _blit(soft: np.ndarray, scale: float, tx: float, ty: float, thresh: float) -> np.ndarray:
    """Draw the soft ink crop onto the glyph canvas (scale + translate)."""
    src = soft
    s = scale
    if s < 0.7:
        src = cv2.resize(soft, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        s = 1.0
        # AREA resize maps pixel centres consistently, keep sub-pixel shift in tx/ty
    M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    out = cv2.warpAffine(src, M, (CW, CH), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(out >= thresh, 255, 0).astype(np.uint8)


def place_items(items: List[Item], s: FontSettings, mode: str,
                thresh: float) -> Dict[int, np.ndarray]:
    its = [it for it in items if it.cp is not None and not it.discarded and it.w > 0 and it.h > 0]
    if not its:
        return {}
    cap_px, xh_px, asc_px = s.cap_height * PX_EM, s.x_height * PX_EM, s.ascender * PX_EM
    # baselines per line
    by_line: Dict[int, List[Item]] = defaultdict(list)
    for it in its:
        by_line[it.line].append(it)
    baseline: Dict[int, float] = {}
    for ln, lst in by_line.items():
        bs = [it.bbox[3] for it in lst if chr(it.cp or 0) in BASELINE_CHARS]
        if not bs:
            bs = [it.bbox[3] for it in lst]
        baseline[ln] = float(np.median(bs))

    def med_h(chars: set) -> Optional[float]:
        hs = [it.h for it in its if chr(it.cp or 0) in chars]
        return float(np.median(hs)) if hs else None

    s_uni: Optional[float] = None
    for chars, target in ((CAPS, cap_px), (XH_CHARS, xh_px), (ASC_CHARS - {"t", "i"}, asc_px)):
        h = med_h(chars)
        if h:
            s_uni = target / h
            break
    if mode in ("uniform", "class") and s_uni is None and mode == "uniform":
        mode = "fit"

    # a typical scale for punctuation in class mode
    if mode == "class":
        ss = []
        for it in its:
            ext = _class_extent(chr(it.cp or 0), s)
            if ext:
                ss.append((ext[0] - ext[1]) * PX_EM / it.h)
        s_uni = float(np.median(ss)) if ss else s_uni
        if s_uni is None:
            mode = "fit"

    out: Dict[int, np.ndarray] = {}
    for it in its:
        ch = chr(it.cp or 0)
        w, h = float(it.w), float(it.h)
        ext = _class_extent(ch, s)
        if mode == "uniform" and s_uni:
            sc = s_uni
            ty = BASELINE_Y + (it.bbox[1] - baseline[it.line]) * sc
        elif mode == "class" and s_uni:
            if ext:
                top, bot = ext
                sc = (top - bot) * PX_EM / h
                ty = BASELINE_Y - top * PX_EM
            else:
                sc = s_uni
                ty = BASELINE_Y + (it.bbox[1] - baseline[it.line]) * sc
        else:  # fit to the em box
            box_h = (s.ascender + s.descender) * PX_EM * 0.92
            box_top = BASELINE_Y - s.ascender * PX_EM + 0.04 * (s.ascender + s.descender) * PX_EM
            sc = min(0.86 * CW / w, box_h / h)
            ty = box_top + (box_h - h * sc) / 2.0
        tx = CW / 2.0 - w * sc / 2.0
        m = _blit(it.soft, sc, tx, ty, thresh)
        if m.any():
            out[it.cp or 0] = m
    return out


# --------------------------------------------------------------------------
# Template sheets
# --------------------------------------------------------------------------
PAGE_W, PAGE_H = 2480, 3508       # A4 at 300 dpi
FRAME = 14                        # thickness of the printed frame
TOP_STRIP = 0.16                  # fraction of a cell used for the label
GUIDE_GRAY = 225


def template_layout(cols: int, rows: int, s: FontSettings) -> dict:
    side, top, bottom = 120, 230, 120
    gw = PAGE_W - 2 * side - 2 * FRAME
    gh = PAGE_H - top - bottom - 2 * FRAME
    cw = gw // cols
    chh = gh // rows
    em = min(0.785 * chh, 0.92 * cw)
    strip = TOP_STRIP * chh
    yb = strip + (0.04 + s.ascender) * em
    return {"cw": cw, "ch": chh, "em": em, "yb": yb, "strip": strip,
            "gw": cw * cols, "gh": chh * rows, "gx0": side + FRAME, "gy0": top + FRAME,
            "cols": cols, "rows": rows}


def template_capacity(cols: int, rows: int) -> int:
    return cols * rows


def template_page_chars(chars: str, cols: int, rows: int, page: int) -> str:
    flat = "".join(c for c in chars if not c.isspace())
    cap = template_capacity(cols, rows)
    return flat[(page - 1) * cap: page * cap]


def template_pages(chars: str, cols: int, rows: int) -> int:
    flat = "".join(c for c in chars if not c.isspace())
    return max(1, -(-len(flat) // template_capacity(cols, rows)))


def _label_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_template(chars: str, cols: int, rows: int, s: FontSettings, page: int = 1,
                  title: str = "") -> Image.Image:
    """Printable sheet: write one character per box with a dark pen, photograph
    or scan it, then import it with mode 'template'."""
    L = template_layout(cols, rows, s)
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    d = ImageDraw.Draw(img)
    page_chars = template_page_chars(chars, cols, rows, page)
    npages = template_pages(chars, cols, rows)
    gx0, gy0, cw, chh = L["gx0"], L["gy0"], L["cw"], L["ch"]
    d.rectangle([gx0 - FRAME, gy0 - FRAME, gx0 + L["gw"] + FRAME - 1, gy0 + L["gh"] + FRAME - 1],
                outline=0, width=FRAME)
    tfont = _label_font(54)
    d.text((gx0, 60), (title or "FontStudio template") +
           "   -   page %d of %d   -   %d x %d boxes" % (page, npages, cols, rows),
           font=tfont, fill=60)
    d.text((gx0, 130), "Write each character inside its box with a dark pen. Sit letters on "
           "the baseline (solid line). Keep the black frame fully visible in your photo.",
           font=_label_font(34), fill=90)
    lab = _label_font(int(0.10 * chh))
    em = L["em"]
    for idx in range(cols * rows):
        r, c = divmod(idx, cols)
        x0 = gx0 + c * cw
        y0 = gy0 + r * chh
        d.rectangle([x0, y0, x0 + cw - 1, y0 + chh - 1], outline=GUIDE_GRAY, width=3)
        yb = y0 + L["yb"]
        for ratio, wd in ((0.0, 4), (s.x_height, 2), (s.cap_height, 2), (s.ascender, 2),
                          (-s.descender, 2)):
            y = int(round(yb - ratio * em))
            d.line([(x0 + 4, y), (x0 + cw - 5, y)], fill=GUIDE_GRAY if ratio != 0 else 200, width=wd)
        d.line([(x0, y0 + L["strip"]), (x0 + cw - 1, y0 + L["strip"])], fill=GUIDE_GRAY, width=3)
        if idx < len(page_chars):
            d.text((x0 + 10, y0 + 4), page_chars[idx], font=lab, fill=120)
    return img


def find_frame(gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = gray.shape
    sc = min(1.0, 1100.0 / max(h, w))
    sm = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else gray
    sm = cv2.GaussianBlur(sm, (5, 5), 0)
    sm = normalize_lighting(sm)
    _, bw = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    found = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = found[-2]
    hh, ww = sm.shape
    best, best_area = None, 0.0
    fallback, fb_area = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.10 * hh * ww:
            continue
        x, y, cw_, ch_ = cv2.boundingRect(c)
        touches = (x <= 2) + (y <= 2) + (x + cw_ >= ww - 2) + (y + ch_ >= hh - 2)
        if touches >= 3:
            continue
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap) and a > best_area:
            best, best_area = ap.reshape(4, 2).astype(np.float32), a
        if a > fb_area:
            fallback, fb_area = c, a
    if best is None and fallback is not None:
        best = cv2.boxPoints(cv2.minAreaRect(fallback)).astype(np.float32)
    if best is None:
        return None
    s = best.sum(1)
    d = best[:, 1] - best[:, 0]
    quad = np.array([best[np.argmin(s)], best[np.argmin(d)], best[np.argmax(s)], best[np.argmax(d)]],
                    np.float32)
    return quad / sc


def import_template(img, chars: str, p: PhotoParams, s: FontSettings
                    ) -> Tuple[Dict[int, np.ndarray], str, np.ndarray]:
    """Returns (glyph masks, info message, debug image of the straightened sheet)."""
    L = template_layout(p.cols, p.rows, s)
    gray = load_gray(img, p.max_side * 2)
    quad = find_frame(gray)
    msg = ""
    h, w = gray.shape
    if quad is None:
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        msg = "Frame not found - assuming the photo is already cropped to the frame. "
    Wd = L["gw"] + 2 * FRAME
    Hd = L["gh"] + 2 * FRAME
    dst = np.array([[0, 0], [Wd - 1, 0], [Wd - 1, Hd - 1], [0, Hd - 1]], np.float32)
    Hm = cv2.getPerspectiveTransform(quad, dst)
    sheet = cv2.warpPerspective(gray, Hm, (Wd, Hd), flags=cv2.INTER_CUBIC, borderValue=255)
    if p.normalize_light:
        sheet = normalize_lighting(sheet)
    prep = binarize(sheet, p)
    page_chars = template_page_chars(chars, p.cols, p.rows, p.page)
    k = PX_EM / L["em"]
    out: Dict[int, np.ndarray] = {}
    dbg = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    cw, chh = L["cw"], L["ch"]
    inset = int(p.cell_margin_pct / 100.0 * cw) + 2
    found = 0
    for idx, ch in enumerate(page_chars):
        r, c = divmod(idx, p.cols)
        x0 = FRAME + c * cw
        y0 = FRAME + r * chh
        cx0, cx1 = x0 + inset, x0 + cw - inset
        cy0, cy1 = int(y0 + L["strip"]) + inset // 2, y0 + chh - inset
        cv2.rectangle(dbg, (cx0, cy0), (cx1, cy1), (0, 160, 0), 2)
        binc = prep.binary[cy0:cy1, cx0:cx1].copy()
        if p.min_area > 1:
            n, lab, st, _ = cv2.connectedComponentsWithStats(binc, connectivity=8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < p.min_area:
                    binc[lab == i] = 0
        if not binc.any():
            continue
        dil = cv2.dilate(binc, np.ones((3, 3), np.uint8), iterations=2) > 0
        soft = prep.soft[cy0:cy1, cx0:cx1] * dil
        tx = CW / 2.0 + (cx0 - (x0 + cw / 2.0)) * k
        ty = BASELINE_Y + (cy0 - (y0 + L["yb"])) * k
        m = _blit(soft.astype(np.float32), k, tx, ty, prep.thresh)
        if m.any():
            out[ord(ch)] = m
            found += 1
    msg += "Found ink in %d of %d boxes." % (found, len(page_chars))
    return out, msg, dbg


# --------------------------------------------------------------------------
# Top level entry points used by the GUI and the tests
# --------------------------------------------------------------------------
@dataclass
class Detection:
    prep: Optional[Prepared]
    items: List[Item]
    message: str = ""
    debug: Optional[np.ndarray] = None
    masks: Dict[int, np.ndarray] = field(default_factory=dict)


def analyse(img, chars: str, p: PhotoParams, s: FontSettings) -> Detection:
    """Run segmentation (no glyph placement yet except for template mode)."""
    if p.mode == "template":
        masks, msg, dbg = import_template(img, chars, p, s)
        return Detection(None, [], msg, dbg, masks)
    gray = load_gray(img, p.max_side)
    prep = prepare(gray, p, allow_deskew=p.deskew)
    if p.mode == "grid":
        items = grid_items(prep, p)
    elif p.mode == "single":
        items = single_item(prep)
    else:
        items = detect_items(prep, p)
    done, total = assign_chars(items, chars, positional=(p.mode == "grid"))
    active = len([it for it in items if not it.discarded])
    msg = "Detected %d shapes; %d characters typed; %d assigned." % (active, total, done)
    if p.mode == "single":
        msg = "Single character photo."
    elif active != total:
        msg += "  The counts differ - click shapes to discard noise, or fix the character list."
    return Detection(prep, items, msg, overlay(prep, items))


def finish(det: Detection, p: PhotoParams, s: FontSettings) -> Dict[int, np.ndarray]:
    """Convert a Detection into glyph masks (canvas sized)."""
    if p.mode == "template":
        return det.masks
    assert det.prep is not None
    return place_items(det.items, s, p.placement, det.prep.thresh)


def overlay(prep: Prepared, items: List[Item]) -> np.ndarray:
    """BGR picture with numbered boxes for the review window."""
    vis = cv2.cvtColor(np.clip(255 - prep.soft, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for n, it in enumerate(items):
        x0, y0, x1, y1 = it.bbox
        col = (0, 0, 220) if it.discarded else (0, 150, 0)
        cv2.rectangle(vis, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), col, 2)
        cv2.putText(vis, "%d" % (n + 1), (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                    cv2.LINE_AA)
    return vis
