"""
COLOR PICK — color identification and HSV color detection.

Main ideas
----------
1. Pixel sampling : read the color under a point (median of a small window,
                    which removes camera noise).
2. Color naming   : nearest of 140 CSS color names, compared in CIE L*a*b*.
3. HSV range      : the color to detect is described by six trackbar values
                        H min / H max   (hue,        0..179)
                        S min / S max   (saturation, 0..255)
                        V min / V max   (value,      0..255)
                    cv2.inRange() marks every pixel inside that range -> mask.
                    If H min > H max the range wraps around red (e.g. 170 -> 8),
                    because red sits at both ends of the hue circle.

OpenCV stores hue as degrees / 2, so 0..179 instead of 0..359.
"""
from dataclasses import dataclass, asdict, replace
import colorsys

import cv2
import numpy as np

from core.color_names import nearest_color_name

# (family, hue_low, hue_high) in OpenCV hue units (0..179)
HUE_FAMILIES = [
    ("Red", 0, 8),
    ("Orange", 9, 20),
    ("Yellow", 21, 34),
    ("Green", 35, 85),
    ("Cyan", 86, 99),
    ("Blue", 100, 128),
    ("Purple", 129, 148),
    ("Pink", 149, 169),
    ("Red", 170, 179),  # red wraps around the hue circle
]

# Achromatic thresholds
BLACK_MAX_V = 50
GRAY_MAX_S = 40
WHITE_MIN_V = 190


# --------------------------------------------------------------------------- #
#  Describing a single color
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColorInfo:
    bgr: tuple          # (B, G, R) 0..255
    hsv: tuple          # OpenCV HSV (H 0..179, S 0..255, V 0..255)
    name: str           # nearest CSS name, e.g. "Tomato"
    family: str         # simple family, e.g. "Red"

    @property
    def rgb(self):
        b, g, r = self.bgr
        return r, g, b

    @property
    def hex(self):
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    @property
    def rgb_text(self):
        return "{}, {}, {}".format(*self.rgb)

    @property
    def hsv_text(self):
        """OpenCV values, the same numbers the trackbars use."""
        return "{}, {}, {}".format(*self.hsv)

    @property
    def text_color(self):
        """Black or white, whichever is readable on top of this color."""
        r, g, b = self.rgb
        return "#111111" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#FFFFFF"


def color_family(h: int, s: int, v: int) -> str:
    """Classify an OpenCV-HSV color into a simple family name."""
    if v < BLACK_MAX_V:
        return "Black"
    if s < GRAY_MAX_S:
        return "White" if v >= WHITE_MIN_V else "Gray"
    family = "Red"
    for name, lo, hi in HUE_FAMILIES:
        if lo <= h <= hi:
            family = name
            break
    if family == "Orange" and v < 170:
        return "Brown"
    return family


def describe_bgr(bgr) -> ColorInfo:
    b, g, r = (int(np.clip(round(float(c)), 0, 255)) for c in bgr)
    h, s, v = (int(x) for x in cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0])
    return ColorInfo((b, g, r), (h, s, v), nearest_color_name((r, g, b)), color_family(h, s, v))


def sample_color(img: np.ndarray, x: int, y: int, radius: int = 2) -> ColorInfo:
    """Median color of a (2r+1)x(2r+1) window centred on (x, y)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = img[y0:y1, x0:x1].reshape(-1, 3)
    return describe_bgr(np.median(patch, axis=0))


# --------------------------------------------------------------------------- #
#  HSV range (what the six trackbars control)
# --------------------------------------------------------------------------- #
RANGE_KEYS = ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max")
RANGE_LIMITS = {"h_min": 179, "h_max": 179, "s_min": 255, "s_max": 255, "v_min": 255, "v_max": 255}


@dataclass(frozen=True)
class HSVRange:
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255

    @property
    def wraps(self):
        """True when the hue range goes through red (179 -> 0)."""
        return self.h_min > self.h_max

    @property
    def valid(self):
        return self.s_min <= self.s_max and self.v_min <= self.v_max

    def to_dict(self):
        return asdict(self)

    def with_values(self, **values):
        return replace(self, **values)

    def text(self):
        hue = f"H {self.h_min}–{self.h_max}" + (" (wraps)" if self.wraps else "")
        return f"{hue}  ·  S {self.s_min}–{self.s_max}  ·  V {self.v_min}–{self.v_max}"


PRESETS = {
    "Red": HSVRange(170, 8, 70, 255, 50, 255),
    "Orange": HSVRange(9, 20, 70, 255, 50, 255),
    "Yellow": HSVRange(21, 34, 70, 255, 50, 255),
    "Green": HSVRange(35, 85, 70, 255, 50, 255),
    "Cyan": HSVRange(86, 99, 70, 255, 50, 255),
    "Blue": HSVRange(100, 128, 70, 255, 50, 255),
    "Purple": HSVRange(129, 148, 70, 255, 50, 255),
    "Pink": HSVRange(149, 169, 70, 255, 50, 255),
    "White": HSVRange(0, 179, 0, 45, 190, 255),
    "Gray": HSVRange(0, 179, 0, 40, 50, 190),
    "Black": HSVRange(0, 179, 0, 255, 0, 50),
}


def range_around(info: ColorInfo, tol: int = 10) -> HSVRange:
    """
    A sensible HSV range around a picked color.
    tol = how many hue steps either side (also widens S/V for gray/black/white).
    """
    h, s, v = info.hsv
    if v < BLACK_MAX_V:                                    # black
        return HSVRange(0, 179, 0, 255, 0, min(255, v + 25 + tol * 2))
    if s < GRAY_MAX_S:                                     # white / gray
        spread = 25 + tol * 2
        return HSVRange(0, 179, 0, min(255, s + 20 + tol), max(0, v - spread), min(255, v + spread))
    if tol * 2 + 1 >= 180:
        h_lo, h_hi = 0, 179
    else:
        h_lo, h_hi = (h - tol) % 180, (h + tol) % 180       # % 180 handles the wrap-around
    return HSVRange(h_lo, h_hi, max(25, int(s * 0.55)), 255, max(25, int(v * 0.55)), 255)


def range_mask(img: np.ndarray, rng: HSVRange, blur: int = 5) -> np.ndarray:
    """Binary mask (0/255): white where the pixel's HSV is inside the range."""
    if blur >= 3:
        k = blur if blur % 2 else blur + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo_sv, hi_sv = (rng.s_min, rng.v_min), (rng.s_max, rng.v_max)
    if rng.wraps:   # e.g. red 170..8  =  170..179  OR  0..8
        return cv2.bitwise_or(cv2.inRange(hsv, (rng.h_min, *lo_sv), (179, *hi_sv)),
                              cv2.inRange(hsv, (0, *lo_sv), (rng.h_max, *hi_sv)))
    return cv2.inRange(hsv, (rng.h_min, *lo_sv), (rng.h_max, *hi_sv))


def hue_strip(width: int, height: int, rng: HSVRange) -> np.ndarray:
    """RGB image of the hue spectrum with the parts outside the range darkened."""
    width, height = max(1, width), max(1, height)
    hues = np.linspace(0, 179, width).round().astype(np.uint8)
    hsv = np.dstack([hues, np.full(width, 255, np.uint8), np.full(width, 255, np.uint8)])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]
    inside = (hues >= rng.h_min) | (hues <= rng.h_max) if rng.wraps else \
             (hues >= rng.h_min) & (hues <= rng.h_max)
    rgb[~inside] = (rgb[~inside] * 0.18).astype(np.uint8)
    return np.repeat(rgb[None, :, :], height, axis=0)


def distinct_color(i):
    """Visually distinct BGR color for index i (golden-ratio hue stepping)."""
    h = (i * 0.618033988749895 + 0.08) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.70, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)
