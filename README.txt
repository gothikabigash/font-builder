FONTSTUDIO  -  make real fonts from photos and drawings
=========================================================

GET THE .EXE
  Option A (2 minutes, on your Windows PC):
    1. Install Python 3.10+ from python.org (tick "Add to PATH" and "tcl/tk").
    2. Double-click  build_exe.bat   ->  dist\FontStudio.exe
  Option B: push this folder to GitHub; the included workflow builds the exe for you
    (Actions tab -> Build FontStudio.exe -> download the artifact).
  Just want to try it?  Double-click run_from_source.bat.

WORKFLOWS
  Photo of handwriting
    Import > Make a printable template sheet -> print -> write one letter per box ->
    photo/scan -> Import > Photo of a printed template sheet.
    Best accuracy: perspective, shadows and rotation are corrected automatically.
  Any picture of writing
    Import > Photo -> glyphs. Type the characters in reading order (Enter = new line).
    Click a detected box to discard noise; right-click to merge it with the next shape.
  Drawing
    Pick a letter, draw (right mouse = erase). Tools: pen, eraser, line, rectangle, box,
    ellipse, disc, fill, move; nibs: round / square / calligraphy; mirror strokes;
    steady-hand smoothing; guide lines; reference font and underlay picture to trace over.
  Output
    File > Export: .ttf  .otf  .woff  .woff2  (and optional one-click install on Windows).

OPTIONS (right-hand panel)
  Font name/style/designer/license - ascender, descender, cap & x-height, units per em -
  side bearings, tracking, space width, monospace - tracing detail, smoothing, curve
  tightness, corner angle, straight-edge snapping, hand-made wobble - weight, slant,
  width, size, baseline shift, missing-letter fallback - and one-click style presets.
  Per-glyph left/right spacing sliders sit under the canvas.

Shortcuts: P pen, E eraser, L line, R rect, O ellipse, F fill, M move, [ ] size,
Ctrl+Z/Y undo/redo, Ctrl+S save, Ctrl+E export, Ctrl+I photo import, arrows = prev/next glyph,
Shift+arrows nudge, Ctrl+Right next empty glyph.

Fonts you make are yours. Only trace handwriting/typefaces you have the right to use.
