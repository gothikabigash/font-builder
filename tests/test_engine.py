"""Headless tests for the FontStudio engine.   Run:  python tests/test_engine.py"""
import io, os, sys, tempfile, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontcore import *
from photoimport import *
from synth import FONT, degrade, text_photo

REF = FONT
def ink(img): return int((np.array(img) < 128).sum())

def glyph_project(chars="AHOgQ8%&ae.i"):
    P = Project()
    for ch in chars: P.set_mask(ord(ch), reference_mask(ch, REF))
    return P

def test_trace_shapes():
    m = np.zeros((CH, CW), np.uint8); m[100:300, 100:300] = 255; m[150:250, 150:250] = 0
    cs = trace_mask(m)
    assert len(cs) == 2 and sorted(c["hole"] for c in cs) == [False, True]
    assert all(len(c["segs"]) == 4 and all(len(s) == 2 for s in c["segs"]) for c in cs), "square -> 4 straight lines"
    c = np.zeros((CH, CW), np.uint8); cv2.circle(c, (256, 300), 120, 255, -1)
    cs = trace_mask(c)
    assert len(cs) == 1 and len(cs[0]["segs"]) <= 12, len(cs[0]["segs"])

def test_formats_and_names():
    P = glyph_project(); P.s.family = "Test Hand"; P.s.designer = "Me"
    for kind in ("ttf", "otf", "woff"):
        data = font_bytes(P, kind)
        f = TTFont(io.BytesIO(data))
        assert 0x41 in f.getBestCmap() and f["name"].getDebugName(1) == "Test Hand"
        img = render_text(P, "AHOg", 80, 500, data=data)
        assert ink(img) > 2000, kind
    f = TTFont(io.BytesIO(font_bytes(P, "ttf")))
    assert f["glyf"]["O"].numberOfContours == 2
    assert f["hhea"].ascent == 780 and f["OS/2"].usWinAscent >= 780

def test_winding_ttf():
    P = glyph_project("O"); f = TTFont(io.BytesIO(font_bytes(P, "ttf")))
    g = f["glyf"]["O"]; ends = g.endPtsOfContours; co = list(g.coordinates)
    def area(pts): return sum(pts[i][0]*pts[(i+1)%len(pts)][1]-pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))/2
    a0 = area(co[:ends[0]+1]); a1 = area(co[ends[0]+1:ends[1]+1])
    assert a0 < 0 < a1, (a0, a1)   # outer clockwise, counter inside counter-clockwise

def test_style_options():
    P = glyph_project("Hl"); base = ink(render_text(P, "H", 120, 300))
    P.s.weight = 14; bold = ink(render_text(P, "H", 120, 300))
    P.s.weight = -10; thin = ink(render_text(P, "H", 120, 300))
    assert thin < base < bold, (thin, base, bold)
    P.s.weight = 0; P.s.monospace = True
    f = TTFont(io.BytesIO(font_bytes(P, "ttf")))
    assert f["hmtx"]["H"][0] == f["hmtx"]["l"][0] == 600
    P.s.monospace = False; P.s.slant = 15
    f = TTFont(io.BytesIO(font_bytes(P, "ttf")))
    assert f["post"].italicAngle == -15 and f["OS/2"].fsSelection & 1
    P.s.slant = 0; P.s.roughness = 1.5
    assert ink(render_text(P, "H", 120, 300)) > 1000

def test_case_fallback():
    P = glyph_project("AB"); P.s.case_fallback = "upper_for_lower"
    cm = TTFont(io.BytesIO(font_bytes(P, "ttf"))).getBestCmap()
    assert 0x61 in cm and 0x62 in cm
    P.s.case_fallback = "off"
    assert 0x61 not in TTFont(io.BytesIO(font_bytes(P, "ttf"))).getBestCmap()

def test_project_roundtrip():
    P = glyph_project("Ag"); P.bearings[ord("A")] = [10, 20]; P.s.family = "RT"; P.s.weight = 3.5
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.fsproj"); P.save(path); Q = Project.load(path)
    assert Q.s.family == "RT" and Q.s.weight == 3.5 and Q.bearings[65] == [10, 20]
    assert np.array_equal(Q.glyphs[65], P.glyphs[65])

def test_transforms():
    m = reference_mask("H", REF)
    b = mask_bbox(shift_mask(m, 20, -10)); b0 = mask_bbox(m)
    assert (b[0]-b0[0], b[1]-b0[1]) == (20, -10)
    r = affine_mask(m, rotate=8); assert 0.9*m.sum() < r.sum() < 1.1*m.sum() and not np.array_equal(r, m)
    c = mask_bbox(auto_center(shift_mask(m, 50, 0))); assert abs((c[0]+c[2])/2 - CW/2) <= 1

LINES = ["ABCDEFGH", "abcdefgh", "ijklmnop", "qrstuvwx", "yz0123!?"]
def test_auto_photo():
    s = FontSettings(); img = Image.fromarray(text_photo(LINES))
    for placement in ("uniform", "class", "fit"):
        p = PhotoParams(mode="auto", placement=placement)
        det = analyse(img, "\n".join(LINES), p, s)
        assert len([i for i in det.items if not i.discarded]) == 40, det.message
        masks = finish(det, p, s); assert len(masks) == 40
        if placement != "fit":
            bb = mask_bbox(masks[ord("H")]); assert abs(bb[3]-BASELINE_Y) < 6, bb      # sits on baseline
            bg = mask_bbox(masks[ord("g")]); assert bg[3] > BASELINE_Y + 60            # descender hangs below
            ba = mask_bbox(masks[ord("a")]); xh = BASELINE_Y - ba[1]
            assert (abs(xh - s.x_height*PX_EM) < 18) if placement == "class" else (250 < xh < 300), (placement, xh)
    # discarding a shape shifts nothing else when counts are corrected
    det = analyse(img, "".join(LINES), PhotoParams(mode="auto"), s); assert det.items[0].cp == ord("A")

def test_grid_and_single():
    s = FontSettings(); f = ImageFont.truetype(FONT, 150)
    im = Image.new("L", (1600, 900), 245); d = ImageDraw.Draw(im)
    text = "ABCDEFGHIJKL"
    for i, ch in enumerate(text):
        r, c = divmod(i, 6)
        d.rectangle([40+c*250, 40+r*270, 40+c*250+250, 40+r*270+270], outline=200, width=2)
        d.text((40+c*250+125, 40+r*270+190), ch, font=f, fill=20, anchor="ms")
    img = Image.fromarray(degrade(np.array(im), angle=-1.0, seed=5))
    p = PhotoParams(mode="grid", rows=2, cols=6, placement="uniform", crop_to_ink=False)
    det = analyse(img, text, p, s); masks = finish(det, p, s)
    assert set(masks) == set(ord(c) for c in text), (len(masks), det.message)
    assert ink(Image.fromarray(255-masks[ord("B")])) > 3000
    single = Image.fromarray(255 - reference_mask("G", REF)[100:520, 40:480])
    p = PhotoParams(mode="single", placement="fit")
    det = analyse(single, "G", p, s); m = finish(det, p, s)[ord("G")]
    bb = mask_bbox(m); assert bb and bb[2]-bb[0] > 200

def test_template_roundtrip():
    s = FontSettings(); cols, rows = 8, 9; chars = ASCII_PRINTABLE
    tpl = make_template(chars, cols, rows, s, page=2); L = template_layout(cols, rows, s)
    assert template_pages(chars, cols, rows) == 2 and len(template_page_chars(chars, cols, rows, 2)) == 22
    d = ImageDraw.Draw(tpl); font = ImageFont.truetype(FONT, int(L["em"]))
    for idx, ch in enumerate(template_page_chars(chars, cols, rows, 2)):
        r, c = divmod(idx, cols)
        d.text((L["gx0"]+c*L["cw"]+L["cw"]/2, L["gy0"]+r*L["ch"]+L["yb"]), ch, font=font, fill=30, anchor="ms")
    sim = degrade(np.array(tpl), persp=True, seed=9); sim = cv2.resize(sim, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    p = PhotoParams(mode="template", rows=rows, cols=cols, page=2)
    det = analyse(Image.fromarray(sim), chars, p, s); masks = finish(det, p, s)
    want = template_page_chars(chars, cols, rows, 2)
    assert len(masks) == len(want), det.message
    for ch in "tuvw":   # check exact baseline mapping
        if ch in want:
            ref = reference_mask(ch, REF)
            inter = np.logical_and(masks[ord(ch)] > 0, ref > 0).sum() / max(1, (ref > 0).sum())
            assert inter > 0.85, (ch, inter)

def test_full_pipeline_font_from_photo():
    s = FontSettings(); img = Image.fromarray(text_photo(LINES)); p = PhotoParams(mode="auto")
    det = analyse(img, "\n".join(LINES), p, s); P = Project()
    for cp, m in finish(det, p, s).items(): P.set_mask(cp, m)
    data = font_bytes(P, "ttf"); out = render_text(P, "ABC abc xyz 012", 90, 800, data=data)
    assert ink(out) > 6000
    with tempfile.TemporaryDirectory() as d:
        for k in ("ttf", "otf", "woff"):
            path = os.path.join(d, "t." + k); export_font(P, path); assert os.path.getsize(path) > 2000

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn(); print("PASS", name)
            except Exception: fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
