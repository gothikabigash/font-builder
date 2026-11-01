"""Synthetic 'photos' used by the tests."""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

import os
def _find_font():
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', 'arialbd.ttf'),
              os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', 'segoeuib.ttf'),
              '/Library/Fonts/Arial Bold.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'):
        if os.path.exists(p):
            return p
    raise SystemExit('no bold system font found for the self-test')
FONT = _find_font()

def degrade(gray, angle=0.0, persp=False, seed=0, noise=6, shade=True):
    rng = np.random.default_rng(seed)
    h, w = gray.shape
    img = gray.astype(np.float32)
    if shade:
        yy, xx = np.mgrid[0:h, 0:w]
        grad = 0.78 + 0.22 * (xx / w) + 0.1 * (yy / h)
        img = img * np.clip(grad, 0, 1)
    img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    if abs(angle) > 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=150)
    if persp:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[60, 40], [w - 30, 10], [w - 80, h - 50], [20, h - 20]])
        # shrink into a larger dark "desk" canvas
        big = np.full((h + 200, w + 200), 70, np.uint8)
        M = cv2.getPerspectiveTransform(src, dst + 100)
        img = cv2.warpPerspective(img, M, (w + 200, h + 200), dst=big, borderMode=cv2.BORDER_TRANSPARENT)
        img = big
    return img

def text_photo(lines, size=110, angle=1.5, seed=1):
    f = ImageFont.truetype(FONT, size)
    W = 1800
    H = int(size * 1.9 * len(lines) + 160)
    im = Image.new('L', (W, H), 240)
    d = ImageDraw.Draw(im)
    y = 100
    for ln in lines:
        x = 80
        for ch in ln:
            d.text((x, y), ch, font=f, fill=25, anchor='ls' if False else 'la')
            x += int(size * 0.95)
        y += int(size * 1.9)
    return degrade(np.array(im), angle=angle, seed=seed)
