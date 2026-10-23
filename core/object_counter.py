"""
ObjectCount — counting the pixels and the objects of the selected color.

Pipeline
--------
  1. Color mask   : pixels inside the HSV range           (cv2.inRange)
  2. Pixel count  : number of white pixels in the mask      (cv2.countNonZero)
  3. Clean-up     : morphological opening / closing, hole filling
  4. Separation   : distance transform splits objects that touch
  5. Objects      : cv2.findContours -> filter by area -> measure each one
                    (area, color pixels, perimeter, centroid, roundness)
"""
from dataclasses import dataclass, field
import csv
import math

import cv2
import numpy as np

from core.color_detector import HSVRange, distinct_color, range_mask


@dataclass
class CountParams:
    blur: int = 5               # Gaussian blur before color detection (0 = off)
    morph: int = 3              # noise removal kernel (0 = off)
    fill_holes: bool = True     # fill reflections / holes inside objects
    separate: bool = True       # split touching objects
    split: int = 50             # split strength 20..80 (%)
    min_area: int = 100         # ignore objects smaller than this (px)
    max_area: int = 0           # 0 = no limit
    exclude_border: bool = False


@dataclass
class DetectedObject:
    id: int
    contour: np.ndarray
    area: float                 # pixels inside the object outline
    perimeter: float
    center: tuple
    bbox: tuple                 # x, y, w, h
    circularity: float          # 1.0 = perfect circle
    color_pixels: int = 0       # pixels inside the object that have the selected color


@dataclass
class CountResult:
    objects: list = field(default_factory=list)
    mask: np.ndarray = None     # every pixel of the selected color (what is counted)
    clean: np.ndarray = None    # mask after clean-up (used to find objects)
    pixel_count: int = 0
    total_pixels: int = 0
    mean_bgr: tuple = None      # average color of the matching pixels

    @property
    def count(self):
        return len(self.objects)

    @property
    def percent(self):
        return 100.0 * self.pixel_count / self.total_pixels if self.total_pixels else 0.0


# --------------------------------------------------------------------------- #
#  Mask helpers
# --------------------------------------------------------------------------- #
def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def clean_mask(mask, p: CountParams):
    out = mask
    if p.morph >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph, p.morph))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)     # removes specks
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)    # closes small gaps
    if p.fill_holes:
        out = fill_holes(out)
    return out


def split_touching(mask, strength=0.5):
    """
    Separate touching objects using the distance transform.

    Every white blob gets a "core" = pixels whose distance to the background is
    at least `strength` x the blob's maximum distance. A blob made of two
    touching candies therefore gets two separate cores (one per candy). Each
    pixel is then given to its nearest core, and a cut is made wherever two
    different cores meet (a marker-based, watershed-style split).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cores = np.zeros_like(mask)
    big_blobs = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        big_blobs += 1
        roi_lab = labels[y:y + h, x:x + w] == i
        roi_dist = dist[y:y + h, x:x + w]
        peak = roi_dist[roi_lab].max()
        cores[y:y + h, x:x + w][roi_lab & (roi_dist >= strength * peak)] = 255

    n_cores, _ = cv2.connectedComponents(cores)
    if n_cores - 1 <= big_blobs:        # nothing to split
        return mask

    # Give every pixel the label of its NEAREST core; cut where labels change.
    _, lab = cv2.distanceTransformWithLabels(255 - cores, cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_CCOMP)
    inside = mask > 0
    cut = np.zeros_like(inside)
    h_diff = (lab[:, :-1] != lab[:, 1:]) & inside[:, :-1] & inside[:, 1:]
    v_diff = (lab[:-1, :] != lab[1:, :]) & inside[:-1, :] & inside[1:, :]
    cut[:, :-1] |= h_diff
    cut[:, 1:] |= h_diff
    cut[:-1, :] |= v_diff
    cut[1:, :] |= v_diff

    out = mask.copy()
    out[cut] = 0
    return out


def find_objects(mask, p: CountParams, color_mask=None):
    """Contours of `mask` -> filtered, measured, numbered objects."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < max(1, p.min_area) or (p.max_area > 0 and area > p.max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if p.exclude_border and (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1):
            continue
        m = cv2.moments(c)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)
        perim = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (perim * perim) if perim else 0.0
        found.append((c, area, perim, (int(round(cx)), int(round(cy))), (x, y, w, h), min(circ, 1.0)))

    # Number objects in reading order: top-to-bottom rows, left-to-right
    if found:
        row_h = max(1.0, float(np.median([f[4][3] for f in found])))
        found.sort(key=lambda f: (int(f[3][1] // row_h), f[3][0]))
    objects = [DetectedObject(i + 1, *f) for i, f in enumerate(found)]

    # Measure each object in real pixels: its filled area, and how many of
    # those pixels have the selected color.
    for o in objects:
        x, y, w, h = o.bbox
        roi = np.zeros((h, w), np.uint8)
        cv2.drawContours(roi, [o.contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        o.area = float(cv2.countNonZero(roi))
        if color_mask is not None:
            o.color_pixels = cv2.countNonZero(cv2.bitwise_and(roi, color_mask[y:y + h, x:x + w]))
    return objects


# --------------------------------------------------------------------------- #
#  Main API
# --------------------------------------------------------------------------- #
def count_color(img, rng: HSVRange, p: CountParams) -> CountResult:
    """Count the pixels and the objects whose color is inside `rng`."""
    mask = range_mask(img, rng, p.blur)
    pixels = cv2.countNonZero(mask)
    mean = tuple(cv2.mean(img, mask=mask)[:3]) if pixels else None

    clean = clean_mask(mask, p)
    work = split_touching(clean, p.split / 100.0) if p.separate else clean
    objects = find_objects(work, p, color_mask=mask)
    h, w = img.shape[:2]
    return CountResult(objects, mask, clean, pixels, h * w, mean)


def annotate(img, result: CountResult, dim_others=True, fill=False, outline=True, labels=True,
             boxes=False, highlight=None, banner=False):
    """
    Draw the result on a copy of the image.
    dim_others : show the matching pixels in color and everything else dimmed gray
    """
    if dim_others and result.mask is not None:
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out = np.where(result.mask[..., None] > 0, img, (gray * 0.35).astype(np.uint8))
    else:
        out = img.copy()
    objs = result.objects
    H, W = img.shape[:2]
    base = max(H, W)
    thick = max(1, round(base / 700))

    if fill and objs:
        overlay = out.copy()
        for o in objs:
            cv2.drawContours(overlay, [o.contour], -1, distinct_color(o.id), cv2.FILLED)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for o in objs:
        color = distinct_color(o.id)
        if outline:
            cv2.drawContours(out, [o.contour], -1, color, thick + 1, cv2.LINE_AA)
        if boxes:
            x, y, w, h = o.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

    if highlight is not None:
        for o in objs:
            if o.id == highlight:
                x, y, w, h = o.bbox
                cv2.drawContours(out, [o.contour], -1, (0, 0, 0), thick + 4, cv2.LINE_AA)
                cv2.drawContours(out, [o.contour], -1, (0, 255, 255), thick + 2, cv2.LINE_AA)
                cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 255), 1, cv2.LINE_AA)

    if labels:
        max_fs = min(1.1, max(0.45, base / 1400))
        for o in objs:
            fs = float(np.clip(min(o.bbox[2], o.bbox[3]) / 40.0, 0.35, max_fs))
            _draw_label(out, str(o.id), o.center, fs)

    if banner:
        text = f"Objects: {result.count}   Pixels: {result.pixel_count:,} ({result.percent:.2f}%)"
        fs = max(0.6, base / 1300)
        t = max(1, round(fs * 2))
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        pad = int(10 * fs)
        cv2.rectangle(out, (0, 0), (tw + pad * 2, th + bl + pad * 2), (20, 20, 20), cv2.FILLED)
        cv2.putText(out, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), t, cv2.LINE_AA)
    return out


def _draw_label(img, text, center, fs):
    t = max(1, round(fs * 2))
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
    cx, cy = center
    pad = max(2, int(4 * fs))
    cv2.rectangle(img, (cx - tw // 2 - pad, cy - th // 2 - pad),
                  (cx + tw // 2 + pad, cy + th // 2 + pad + bl // 2), (25, 25, 25), cv2.FILLED)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), t, cv2.LINE_AA)


def export_csv(result: CountResult, rng: HSVRange, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "color_pixels", "area_px", "perimeter_px", "circularity",
                    "center_x", "center_y", "box_x", "box_y", "box_w", "box_h"])
        for o in result.objects:
            w.writerow([o.id, o.color_pixels, round(o.area, 1), round(o.perimeter, 1),
                        round(o.circularity, 3), *o.center, *o.bbox])
        w.writerow([])
        w.writerow(["total_color_pixels", result.pixel_count])
        w.writerow(["image_pixels", result.total_pixels])
        w.writerow(["percent_of_image", round(result.percent, 3)])
        w.writerow(["objects", result.count])
        for key, value in rng.to_dict().items():
            w.writerow([key, value])


def object_at(result: CountResult, x, y):
    """Return the object whose contour contains point (x, y), or None."""
    for o in result.objects:
        bx, by, bw, bh = o.bbox
        if bx <= x <= bx + bw and by <= y <= by + bh:
            if cv2.pointPolygonTest(o.contour, (float(x), float(y)), False) >= 0:
                return o
    return None
