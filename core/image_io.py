"""
Image loading / saving that also works with non-English characters in file
paths (cv2.imread / cv2.imwrite fail on those on Windows).
"""
import os

import cv2
import numpy as np

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def load_image(path):
    """Return a BGR image, or None if the file can't be read."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_TYPES:
        path, ext = path + ".png", ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except OSError:
        return False


def fit_max(img, max_side):
    """Shrink so the longest side is <= max_side. Returns (image, scale)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    s = max_side / longest
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA), s
