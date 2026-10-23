"""
COLOR PICK + ObjectCount  —  Android version (Kivy + OpenCV, pure Python)
=======================================================================
  * COLOR PICK  : A color detection application using image processing
  * ObjectCount : A simple object counting application using computer vision

Tap a color (or pick a preset / move the HSV trackbars) and the app counts
how many PIXELS have that color and how many separate OBJECTS have it.

Runs on Android (built with Buildozer) and on a desktop for testing:
    pip install kivy opencv-python numpy
    python main.py
"""
import os
import time
from collections import deque

from kivy.app import App
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, platform

import cv2
import numpy as np

from core.color_detector import (PRESETS, RANGE_KEYS, RANGE_LIMITS, HSVRange, describe_bgr,
                                 hue_strip, range_around, sample_color)
from core.image_io import fit_max, load_image, save_image
from core.object_counter import CountParams, CountResult, annotate, count_color, export_csv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(APP_DIR, "samples", "candies_30.png")
MAX_SIDE = 1000            # photos are shrunk to this for fast processing on a phone
LIVE_SIDE = 640            # camera frames are processed at this size
LIVE_FPS = 10
DEFAULT_TOL = 6
MONO = "RobotoMono-Regular"
ANDROID = platform == "android"

def selftest_requested():
    """True when the app is started for automated testing (never for normal users)."""
    if os.environ.get("COLORCOUNT_SELFTEST"):
        return True
    if ANDROID:
        try:
            from jnius import autoclass
            intent = autoclass("org.kivy.android.PythonActivity").mActivity.getIntent()
            return intent is not None and intent.getStringExtra("selftest") is not None
        except Exception:
            return False
    return False


# --------------------------------------------------------------------------- #
#  Theme
# --------------------------------------------------------------------------- #
C = {
    "bg": "#0f1117", "card": "#181c25", "card2": "#20252f", "button": "#252b37",
    "button_down": "#323a4b", "border": "#2a303d", "canvas": "#0a0c10",
    "text": "#e7e9ee", "muted": "#8b93a7", "subtle": "#5b6375",
    "accent": "#6c8cff", "accent_down": "#5370e0", "accent_dim": "#2c3a6e",
    "success": "#34d399", "warning": "#fbbf24", "danger": "#f87171",
}


def rgba(hex_code, alpha=1.0):
    r, g, b, _ = get_color_from_hex(hex_code)
    return [r, g, b, alpha]


def paint(widget, color, radius=dp(14), border=None):
    """Rounded background (and optional border) that follows the widget."""
    with widget.canvas.before:
        fill = Color(*rgba(color))
        rect = RoundedRectangle(pos=widget.pos, size=widget.size, radius=[radius])
        if border:
            Color(*rgba(border))
            line = Line(rounded_rectangle=(*widget.pos, *widget.size, radius), width=dp(1))
    def update(*_):
        rect.pos, rect.size = widget.pos, widget.size
        if border:
            line.rounded_rectangle = (*widget.pos, *widget.size, radius)
    widget.bind(pos=update, size=update)
    return fill


def text(txt, size=13, color="text", bold=False, halign="left", mono=False, **kw):
    """Label that wraps inside its width."""
    lbl = Label(text=txt, font_size=sp(size), color=rgba(C.get(color, color)), bold=bold,
                halign=halign, valign="middle", **kw)
    if mono:
        lbl.font_name = MONO
    lbl.bind(size=lambda l, s: setattr(l, "text_size", (s[0], None)))
    return lbl


def auto_height(label, extra=0):
    """Make a wrapping label exactly as tall as its text."""
    label.size_hint_y = None
    label.bind(texture_size=lambda l, ts: setattr(l, "height", ts[1] + extra))
    return label


# --------------------------------------------------------------------------- #
#  Small widgets
# --------------------------------------------------------------------------- #
class FlatButton(ButtonBehavior, Label):
    def __init__(self, label, accent=False, fs=14, **kw):
        kw.setdefault("color", rgba("#ffffff" if accent else C["text"]))
        super().__init__(text=label, font_size=sp(fs), bold=True, halign="center", valign="middle", **kw)
        self._accent = accent
        self._fill = paint(self, C["accent"] if accent else C["button"], radius=dp(12))
        self.bind(state=lambda *_: self._repaint())

    def set_accent(self, on):
        self._accent = on
        self.color = rgba("#ffffff" if on else C["text"])
        self._repaint()

    def _repaint(self):
        down = self.state == "down"
        if self._accent:
            self._fill.rgba = rgba(C["accent_down"] if down else C["accent"])
        else:
            self._fill.rgba = rgba(C["button_down"] if down else C["button"])


class Segmented(BoxLayout):
    """Row of buttons where exactly one is selected."""

    def __init__(self, options, value, on_change, fs=12, **kw):
        super().__init__(spacing=dp(4), **kw)
        self.value, self.on_change, self.buttons = value, on_change, {}
        for key, label in options:
            b = FlatButton(label, accent=key == value, fs=fs)
            b.bind(on_release=lambda _b, k=key: self.select(k))
            self.buttons[key] = b
            self.add_widget(b)

    def select(self, key, notify=True):
        self.value = key
        for k, b in self.buttons.items():
            b.set_accent(k == key)
        if notify:
            self.on_change(key)


class ValueSlider(BoxLayout):
    """Trackbar: a name, the current value, and a slider."""

    def __init__(self, name, lo, hi, value, command=None, step=1, fmt=str, **kw):
        super().__init__(orientation="vertical", size_hint_y=None, height=dp(62), **kw)
        self.command, self.fmt, self._silent = command, fmt, False
        top = BoxLayout(size_hint_y=None, height=dp(22))
        top.add_widget(text(name, 13))
        self.value_lbl = text(fmt(value), 13, "accent", bold=True, halign="right", mono=True,
                              size_hint_x=None, width=dp(96))
        top.add_widget(self.value_lbl)
        self.add_widget(top)
        self.slider = Slider(min=lo, max=hi, value=value, step=step, value_track=True,
                             value_track_color=rgba(C["accent"]), value_track_width=dp(4),
                             cursor_size=(dp(26), dp(26)), padding=dp(14))
        self.slider.bind(value=self._changed)
        self.add_widget(self.slider)
        self._value = int(value)

    def _changed(self, _s, v):
        v = int(round(v))
        self.value_lbl.text = self.fmt(v)
        if v != self._value:
            self._value = v
            if self.command and not self._silent:
                self.command(v)

    def get(self):
        return self._value

    def set(self, v):
        self._silent = True
        self._value = int(v)
        self.slider.value = int(v)
        self.value_lbl.text = self.fmt(int(v))
        self._silent = False


class ToggleRow(ButtonBehavior, BoxLayout):
    """A setting that is ON or OFF (tap anywhere on the row)."""

    def __init__(self, name, value, command, **kw):
        super().__init__(size_hint_y=None, height=dp(44), spacing=dp(10), **kw)
        self.value, self.command = value, command
        self.add_widget(text(name, 13))
        self.pill = Label(text="", font_size=sp(12), bold=True, size_hint=(None, None),
                          size=(dp(60), dp(30)), pos_hint={"center_y": 0.5})
        self._pill_bg = paint(self.pill, C["button"], radius=dp(15))
        self.add_widget(self.pill)
        self.bind(on_release=lambda *_: self.set(not self.value, notify=True))
        self.set(value)

    def set(self, value, notify=False):
        self.value = value
        self.pill.text = "ON" if value else "OFF"
        self.pill.color = rgba("#ffffff" if value else C["muted"])
        self._pill_bg.rgba = rgba(C["accent"] if value else C["button"])
        if notify:
            self.command(value)


class Swatch(Widget):
    def __init__(self, diameter=dp(30), **kw):
        super().__init__(size_hint=(None, None), size=(diameter, diameter), **kw)
        with self.canvas:
            self._ring = Color(*rgba(C["border"]))
            self._outer = Ellipse()
            self._fill = Color(*rgba(C["button"]))
            self._inner = Ellipse()
        self.bind(pos=self._draw, size=self._draw)

    def _draw(self, *_):
        self._outer.pos, self._outer.size = self.pos, self.size
        b = dp(2)
        self._inner.pos = (self.x + b, self.y + b)
        self._inner.size = (self.width - 2 * b, self.height - 2 * b)

    def set(self, hex_code):
        self._fill.rgba = rgba(hex_code or C["button"])


class PresetChip(ButtonBehavior, BoxLayout):
    def __init__(self, name, rng, command, **kw):
        super().__init__(size_hint_y=None, height=dp(40), padding=(dp(8), 0), spacing=dp(8), **kw)
        self.name = name
        mid = rng.h_min if not rng.wraps else 0
        if name in ("White", "Gray", "Black"):
            hex_code = {"White": "#f2f2f2", "Gray": "#8a8a8a", "Black": "#111111"}[name]
        else:
            h = (rng.h_min + ((rng.h_max - rng.h_min) % 180) // 2) % 180
            b, g, r = cv2.cvtColor(np.uint8([[[h, 220, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
            hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
        self._bg = paint(self, C["button"], radius=dp(12))
        sw = Swatch(dp(18), pos_hint={"center_y": 0.5})
        sw.set(hex_code)
        self.add_widget(sw)
        self.lbl = text(name, 12)
        self.add_widget(self.lbl)
        self.bind(on_release=lambda *_: command(name))

    def set_active(self, on):
        self._bg.rgba = rgba(C["accent_dim"] if on else C["button"])
        self.lbl.bold = on


class Card(BoxLayout):
    def __init__(self, **kw):
        kw.setdefault("orientation", "vertical")
        kw.setdefault("padding", dp(12))
        kw.setdefault("spacing", dp(6))
        super().__init__(**kw)
        paint(self, C["card"], radius=dp(16), border=C["border"])


class HueBar(Widget):
    """The hue spectrum with the part outside the selected range darkened."""

    def __init__(self, **kw):
        super().__init__(size_hint_y=None, height=dp(18), **kw)
        self.rng = HSVRange()
        with self.canvas:
            Color(1, 1, 1, 1)
            self._rect = Rectangle()
        self.bind(pos=self.redraw, size=self.redraw)

    def set_range(self, rng):
        self.rng = rng
        self.redraw()

    def redraw(self, *_):
        w, h = int(self.width), int(self.height)
        if w < 4 or h < 2:
            return
        strip = cv2.cvtColor(hue_strip(w, h, self.rng), cv2.COLOR_RGB2RGBA)
        tex = Texture.create(size=(w, h), colorfmt="rgba")
        tex.blit_buffer(strip.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self._rect.texture = tex
        self._rect.pos, self._rect.size = self.pos, self.size


# --------------------------------------------------------------------------- #
#  Image view (tap to pick a color)
# --------------------------------------------------------------------------- #
class ImageView(FloatLayout):
    def __init__(self, on_preview, on_pick, **kw):
        super().__init__(**kw)
        self.on_preview, self.on_pick = on_preview, on_pick
        self.texture, self.img_size, self.marker = None, None, None
        self._fit = (0, 0, 1.0)      # x0, y0, scale
        paint(self, C["canvas"], radius=dp(16), border=C["border"])
        with self.canvas:
            Color(1, 1, 1, 1)
            self._rect = Rectangle()
            self._mk_shadow = Color(0, 0, 0, 0)
            self._mk_line1 = Line(circle=(0, 0, 1), width=dp(3))
            self._mk_color = Color(1, 1, 1, 0)
            self._mk_line2 = Line(circle=(0, 0, 1), width=dp(1.6))
        self.placeholder = text("No image yet\n\nTap  Gallery,  Camera  or  Sample  below,\n"
                                "then tap a color in the picture to count it.",
                                13, "muted", halign="center", pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.add_widget(self.placeholder)
        self.bind(pos=self._layout, size=self._layout)

    def show(self, img):
        if img is None:
            return
        rgba_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA if img.ndim == 2 else cv2.COLOR_BGR2RGBA)
        h, w = rgba_img.shape[:2]
        if self.texture is None or self.texture.size != (w, h):
            self.texture = Texture.create(size=(w, h), colorfmt="rgba")
            self.texture.flip_vertical()           # numpy row 0 = top of the picture
            self._rect.texture = self.texture
        self.texture.blit_buffer(rgba_img.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.img_size = (w, h)
        self.placeholder.opacity = 0
        self._layout()
        self.canvas.ask_update()

    def set_marker(self, pos):
        self.marker = pos
        self._layout()

    def _layout(self, *_):
        pad = dp(4)
        if not self.img_size:
            self._rect.size = (0, 0)
            return
        w, h = self.img_size
        bw, bh = self.width - 2 * pad, self.height - 2 * pad
        s = max(1e-6, min(bw / w, bh / h))
        dw, dh = w * s, h * s
        x0, y0 = self.x + pad + (bw - dw) / 2, self.y + pad + (bh - dh) / 2
        self._rect.pos, self._rect.size = (x0, y0), (dw, dh)
        self._fit = (x0, y0, s)
        if self.marker:
            mx = x0 + (self.marker[0] + 0.5) * s
            my = y0 + dh - (self.marker[1] + 0.5) * s
            r = dp(11)
            self._mk_line1.circle = (mx, my, r)
            self._mk_line2.circle = (mx, my, r)
            self._mk_shadow.a, self._mk_color.a = 0.8, 1
        else:
            self._mk_shadow.a = self._mk_color.a = 0

    def to_image(self, tx, ty):
        if not self.img_size:
            return None
        x0, y0, s = self._fit
        w, h = self.img_size
        ix, iy = int((tx - x0) / s), int((y0 + h * s - ty) / s)
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):       # overlay buttons first
            return True
        if self.collide_point(*touch.pos) and self.img_size:
            touch.grab(self)
            pos = self.to_image(*touch.pos)
            if pos:
                self.on_preview(*pos)
            return True
        return False

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            pos = self.to_image(*touch.pos)
            if pos:
                self.on_preview(*pos)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            pos = self.to_image(*touch.pos)
            if pos:
                self.on_pick(*pos)
            return True
        return super().on_touch_up(touch)


# --------------------------------------------------------------------------- #
#  The app
# --------------------------------------------------------------------------- #
class ColorCountApp(App):
    title = "ColorCount"

    def build(self):
        Window.clearcolor = rgba(C["bg"])
        self.image = None
        self.source_name = ""
        self.result = CountResult()
        self.rng = PRESETS["Red"]
        self.preset = "Red"
        self.picked = None
        self.pick_pos = None
        self.view_mode = "highlight"
        self.highlight_id = None
        self.cam = None
        self.cam_event = None
        self.rotation = 90 if ANDROID else 0
        self._recent = deque(maxlen=8)
        self._refresh_ev = None
        self._objects_dirty = True
        self._last_list = 0.0
        self._chooser = None
        self._frames = 0

        root = FloatLayout()
        main = BoxLayout(orientation="vertical", padding=(dp(12), dp(10), dp(12), dp(10)), spacing=dp(10))
        root.add_widget(main)
        main.add_widget(self._build_header())

        self.view = ImageView(self._on_preview, self._on_pick, size_hint_y=1.0)
        main.add_widget(self.view)
        self.view_seg = Segmented([("highlight", "Highlight"), ("objects", "Objects"),
                                   ("mask", "Mask"), ("original", "Original")], "highlight",
                                  self._on_view, fs=11, size_hint_y=None, height=dp(32))
        main.add_widget(self.view_seg)

        main.add_widget(self._build_actions())
        main.add_widget(self._build_stats())
        main.add_widget(self._build_tabs())

        self.toast_lbl = Label(text="", font_size=sp(13), color=rgba("#ffffff"), size_hint=(None, None),
                               halign="center", valign="middle", opacity=0, padding=(dp(14), dp(8)))
        self.toast_lbl.bind(texture_size=lambda l, ts: setattr(l, "size", ts))
        self.toast_lbl.bind(size=lambda *_: self._place_toast())   # re-centre when the text changes
        self._toast_bg = paint(self.toast_lbl, "#2a3142", radius=dp(18))
        root.add_widget(self.toast_lbl)
        root.bind(size=lambda *_: self._place_toast())

        self._apply_range(self.rng, preset="Red")
        return root

    # ================================================================ layout
    def _build_header(self):
        bar = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
        logo = Widget(size_hint=(None, None), size=(dp(36), dp(36)), pos_hint={"center_y": 0.5})
        with logo.canvas:
            rings = []
            for col in ("#ff5c5c", "#4ade80", "#5b8cff"):
                Color(*rgba(col))
                rings.append(Line(circle=(0, 0, 1), width=dp(1.6)))
        def draw_logo(*_):
            s, r = logo.width, logo.width * 0.24
            for ln, (fx, fy) in zip(rings, ((0.36, 0.62), (0.64, 0.62), (0.5, 0.37))):
                ln.circle = (logo.x + s * fx, logo.y + s * fy, r)
        logo.bind(pos=draw_logo, size=draw_logo)
        bar.add_widget(logo)
        titles = BoxLayout(orientation="vertical")
        titles.add_widget(text("COLOR PICK + ObjectCount", 18, bold=True))
        titles.add_widget(text("Color detection & object counting · OpenCV", 11, "muted"))
        bar.add_widget(titles)
        return bar

    def _build_actions(self):
        row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))
        self.btn_gallery = FlatButton("Gallery", accent=True)
        self.btn_camera = FlatButton("Camera")
        self.btn_sample = FlatButton("Sample")
        self.btn_save = FlatButton("Save")
        self.btn_gallery.bind(on_release=lambda *_: self.open_gallery())
        self.btn_camera.bind(on_release=lambda *_: self.camera_button())
        self.btn_sample.bind(on_release=lambda *_: self.load_path(SAMPLE, "Sample: candies"))
        self.btn_save.bind(on_release=lambda *_: self.save_result())
        for b in (self.btn_gallery, self.btn_camera, self.btn_sample, self.btn_save):
            row.add_widget(b)
        return row

    def _build_stats(self):
        card = Card(orientation="horizontal", size_hint_y=None, height=dp(96), spacing=dp(10),
                    padding=(dp(14), dp(10)))
        col1 = BoxLayout(orientation="vertical", size_hint_x=0.26)
        col1.add_widget(text("OBJECTS", 10, "muted", bold=True, size_hint_y=None, height=dp(16)))
        self.obj_lbl = text("0", 34, bold=True)
        col1.add_widget(self.obj_lbl)
        card.add_widget(col1)

        col2 = BoxLayout(orientation="vertical", size_hint_x=0.38)
        col2.add_widget(text("PIXELS", 10, "muted", bold=True, size_hint_y=None, height=dp(16)))
        self.px_lbl = text("0", 22, "accent", bold=True)
        col2.add_widget(self.px_lbl)
        self.pct_lbl = text("—", 11, "muted", size_hint_y=None, height=dp(18))
        col2.add_widget(self.pct_lbl)
        card.add_widget(col2)

        col3 = BoxLayout(orientation="vertical", size_hint_x=0.36, spacing=dp(2))
        col3.add_widget(text("COLOR", 10, "muted", bold=True, size_hint_y=None, height=dp(16)))
        r = BoxLayout(spacing=dp(8))
        self.color_sw = Swatch(dp(30), pos_hint={"center_y": 0.5})
        r.add_widget(self.color_sw)
        names = BoxLayout(orientation="vertical")
        self.color_name = text("Red preset", 12, bold=True, shorten=True, shorten_from="right")
        self.color_hex = text("tap the image", 10, "muted", mono=True)
        names.add_widget(self.color_name)
        names.add_widget(self.color_hex)
        r.add_widget(names)
        col3.add_widget(r)
        card.add_widget(col3)
        return card

    def _build_tabs(self):
        box = BoxLayout(orientation="vertical", size_hint_y=1.15, spacing=dp(8))
        self.tab_seg = Segmented([("colors", "Colors"), ("trackbars", "Trackbars"),
                                  ("settings", "Settings"), ("objects", "Objects")], "colors",
                                 self._show_tab, fs=12, size_hint_y=None, height=dp(38))
        box.add_widget(self.tab_seg)
        self.scroll = ScrollView(do_scroll_x=False, bar_width=dp(4), scroll_type=["bars", "content"])
        box.add_widget(self.scroll)
        self.tabs = {"colors": self._tab_colors(), "trackbars": self._tab_trackbars(),
                     "settings": self._tab_settings(), "objects": self._tab_objects()}
        self._show_tab("colors")
        return box

    def _column(self):
        g = GridLayout(cols=1, size_hint_y=None, spacing=dp(10), padding=(0, 0, dp(6), dp(12)))
        g.bind(minimum_height=g.setter("height"))
        return g

    def _tab_colors(self):
        col = self._column()
        card = Card(size_hint_y=None)
        card.bind(minimum_height=card.setter("height"))
        card.add_widget(auto_height(text("Tap the picture to pick any color, or choose a preset:", 12, "muted"),
                                    dp(4)))
        grid = GridLayout(cols=3, spacing=dp(6), size_hint_y=None)
        grid.bind(minimum_height=grid.setter("height"))
        self.chips = {}
        for name, rng in PRESETS.items():
            chip = PresetChip(name, rng, self.choose_preset)
            self.chips[name] = chip
            grid.add_widget(chip)
        card.add_widget(grid)
        col.add_widget(card)

        card2 = Card(size_hint_y=None)
        card2.bind(minimum_height=card2.setter("height"))
        card2.add_widget(auto_height(text("PICKED COLOR", 10, "muted", bold=True)))
        self.pick_info = auto_height(text("Nothing picked yet.", 13, mono=True), dp(4))
        card2.add_widget(self.pick_info)
        self.tol_sl = ValueSlider("Pick tolerance (± hue)", 1, 40, DEFAULT_TOL, self._on_tolerance)
        card2.add_widget(self.tol_sl)
        col.add_widget(card2)
        return col

    def _tab_trackbars(self):
        col = self._column()
        card = Card(size_hint_y=None)
        card.bind(minimum_height=card.setter("height"))
        card.add_widget(auto_height(text("HSV TRACKBARS", 10, "muted", bold=True)))
        self.hue_bar = HueBar()
        card.add_widget(self.hue_bar)
        self.range_lbl = auto_height(text("", 11, "muted", mono=True), dp(4))
        card.add_widget(self.range_lbl)
        names = {"h_min": "Hue min", "h_max": "Hue max", "s_min": "Saturation min",
                 "s_max": "Saturation max", "v_min": "Value min", "v_max": "Value max"}
        self.range_sliders = {}
        for key in RANGE_KEYS:
            sl = ValueSlider(names[key], 0, RANGE_LIMITS[key], 0, lambda v, k=key: self._on_range_slider(k, v))
            self.range_sliders[key] = sl
            card.add_widget(sl)
        card.add_widget(auto_height(text("OpenCV hue is 0–179 (degrees ÷ 2). If Hue min is larger than "
                                         "Hue max, the range wraps around red (e.g. 170 → 8).", 11, "muted"),
                                    dp(4)))
        col.add_widget(card)
        return col

    def _tab_settings(self):
        col = self._column()
        d = CountParams()
        card = Card(size_hint_y=None)
        card.bind(minimum_height=card.setter("height"))
        card.add_widget(auto_height(text("OBJECT COUNTING", 10, "muted", bold=True)))
        r = self.schedule_refresh
        self.min_sl = ValueSlider("Minimum object area (px)", 0, 5000, d.min_area, r, step=10)
        self.max_sl = ValueSlider("Maximum object area (px)", 0, 200000, 0, r, step=500,
                                  fmt=lambda v: "No limit" if v == 0 else f"{v:,}")
        self.blur_sl = ValueSlider("Blur (smooths noise)", 0, 15, d.blur, r)
        self.morph_sl = ValueSlider("Noise removal", 0, 15, d.morph, r)
        self.split_sl = ValueSlider("Split strength (touching objects)", 20, 80, d.split, r,
                                    fmt=lambda v: f"{v}%")
        for w in (self.min_sl, self.max_sl, self.blur_sl, self.morph_sl):
            card.add_widget(w)
        self.fill_tg = ToggleRow("Fill holes / reflections", d.fill_holes, lambda v: r())
        self.sep_tg = ToggleRow("Separate touching objects", d.separate, lambda v: r())
        self.edge_tg = ToggleRow("Ignore objects cut by the edge", d.exclude_border, lambda v: r())
        card.add_widget(self.fill_tg)
        card.add_widget(self.sep_tg)
        card.add_widget(self.split_sl)
        card.add_widget(self.edge_tg)
        col.add_widget(card)

        card2 = Card(size_hint_y=None)
        card2.bind(minimum_height=card2.setter("height"))
        card2.add_widget(auto_height(text("CAMERA ROTATION", 10, "muted", bold=True)))
        card2.add_widget(auto_height(text("If the live camera picture is sideways, change this.", 11, "muted")))
        self.rot_seg = Segmented([(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")], self.rotation,
                                 self._on_rotation, fs=12, size_hint_y=None, height=dp(36))
        card2.add_widget(self.rot_seg)
        col.add_widget(card2)

        reset = FlatButton("Reset all settings", size_hint_y=None, height=dp(44))
        reset.bind(on_release=lambda *_: self.reset_defaults())
        col.add_widget(reset)
        return col

    def _tab_objects(self):
        col = self._column()
        card = Card(size_hint_y=None)
        card.bind(minimum_height=card.setter("height"))
        head = BoxLayout(size_hint_y=None, height=dp(34))
        head.add_widget(text("OBJECTS FOUND", 10, "muted", bold=True))
        csv_btn = FlatButton("Save CSV", fs=12, size_hint=(None, None), size=(dp(96), dp(32)))
        csv_btn.bind(on_release=lambda *_: self.save_csv())
        head.add_widget(csv_btn)
        card.add_widget(head)
        hdr = BoxLayout(size_hint_y=None, height=dp(24))
        for t, w in (("#", 0.12), ("Color px", 0.3), ("Area", 0.3), ("Round", 0.28)):
            hdr.add_widget(text(t, 11, "muted", bold=True, size_hint_x=w,
                                halign="left" if t == "#" else "right"))
        card.add_widget(hdr)
        self.obj_list = GridLayout(cols=1, size_hint_y=None, spacing=dp(2))
        self.obj_list.bind(minimum_height=self.obj_list.setter("height"))
        card.add_widget(self.obj_list)
        card.add_widget(auto_height(text("Color px = pixels of the selected color inside the object. "
                                         "Tap a row to highlight it.", 11, "muted"), dp(6)))
        col.add_widget(card)
        return col

    def _show_tab(self, key):
        self.scroll.clear_widgets()
        self.scroll.add_widget(self.tabs[key])
        self.scroll.scroll_y = 1
        if key == "objects":
            self._fill_objects()

    def _place_toast(self):
        root = self.root
        if root:
            self.toast_lbl.center_x = root.center_x
            self.toast_lbl.y = root.y + dp(80)

    # ================================================================ sources
    def load_path(self, path, name=None):
        img = load_image(path)
        if img is None:
            self.toast("That file is not an image")
            return
        self.stop_camera()
        self.image, _ = fit_max(img, MAX_SIDE)
        self.source_name = name or os.path.basename(path)
        self.pick_pos = None
        self.highlight_id = None
        self.view.set_marker(None)
        self.refresh()
        h, w = self.image.shape[:2]
        self.toast(f"{self.source_name}  ({w}×{h})")

    def open_gallery(self):
        if ANDROID:
            try:
                from androidstorage4kivy import Chooser
                if self._chooser is None:
                    self._chooser = Chooser(self._chooser_done)
                self._chooser.choose_content("image/*")
            except Exception as exc:
                self.toast(f"Gallery not available: {exc}")
            return
        try:                                   # desktop testing
            import tkinter as tk
            from tkinter import filedialog
            r = tk.Tk()
            r.withdraw()
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
            r.destroy()
            if path:
                self.load_path(path)
        except Exception:
            self.load_path(SAMPLE, "Sample: candies")

    def _chooser_done(self, uri_list):
        try:
            from androidstorage4kivy import SharedStorage
            for uri in uri_list:
                path = SharedStorage().copy_from_shared(uri)
                if path:
                    self._load_from_thread(path)
                    break
        except Exception as exc:
            self._toast_from_thread(f"Could not open the picture: {exc}")

    @mainthread
    def _load_from_thread(self, path):
        self.load_path(path)

    @mainthread
    def _toast_from_thread(self, msg):
        self.toast(msg)

    # ---- camera ---------------------------------------------------------------
    def camera_button(self):
        if self.cam is not None:          # live -> freeze this frame
            self.stop_camera()
            self.source_name = "Captured frame"
            self.toast("Frame captured")
            self.refresh()
            return
        self._with_camera_permission(self.start_camera)

    def _with_camera_permission(self, then):
        if not ANDROID:
            then()
            return
        try:
            from android.permissions import Permission, check_permission, request_permissions
        except ImportError:
            then()
            return
        if check_permission(Permission.CAMERA):
            then()
            return

        def done(_perms, grants):
            if grants and all(grants):
                Clock.schedule_once(lambda dt: then())
            else:
                self._toast_from_thread("Camera permission is needed for the live camera")
        request_permissions([Permission.CAMERA], done)

    def start_camera(self):
        try:
            from kivy.core.camera import Camera as CoreCamera
            if CoreCamera is None:
                raise RuntimeError("no camera provider")
            self.cam = CoreCamera(index=0, resolution=(640, 480), stopped=True)
            self.cam.start()
        except Exception as exc:
            self.cam = None
            self.toast(f"Camera could not start ({exc.__class__.__name__})")
            return
        self.source_name = "Live camera"
        self.pick_pos = None
        self.view.set_marker(None)
        self._recent.clear()
        self.btn_camera.text = "Capture"
        self.btn_camera.set_accent(True)
        self.btn_gallery.set_accent(False)
        self.cam_event = Clock.schedule_interval(self._camera_tick, 1.0 / LIVE_FPS)
        self.toast("Live camera: tap Capture to freeze the picture")

    def stop_camera(self):
        if self.cam_event is not None:
            self.cam_event.cancel()
            self.cam_event = None
        if self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
            self.cam = None
        self.btn_camera.text = "Camera"
        self.btn_camera.set_accent(False)
        self.btn_gallery.set_accent(True)

    def _read_frame(self):
        cam = self.cam
        frame = None
        if hasattr(cam, "grab_frame"):            # Android: raw NV21 preview buffer
            try:
                buf = cam.grab_frame()
                if buf:
                    w, h = cam._resolution
                    arr = np.frombuffer(bytes(buf), np.uint8)
                    n = w * h * 3 // 2
                    if arr.size >= n:
                        frame = cv2.cvtColor(arr[:n].reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
            except Exception:
                frame = None
        if frame is None and cam.texture is not None:  # desktop providers
            tex = cam.texture
            w, h = tex.size
            arr = np.frombuffer(tex.pixels, np.uint8)
            if arr.size == w * h * 4:
                frame = cv2.cvtColor(cv2.flip(arr.reshape(h, w, 4), 0), cv2.COLOR_RGBA2BGR)
        if frame is not None and self.rotation:
            frame = cv2.rotate(frame, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotation])
        return frame

    def _camera_tick(self, _dt):
        if self.cam is None:
            return
        frame = self._read_frame()
        if frame is None:
            return
        self._frames += 1
        self.image, _ = fit_max(frame, LIVE_SIDE)
        self.refresh(live=True)

    def _on_rotation(self, value):
        self.rotation = value

    # ================================================================ picking
    def _on_preview(self, x, y):
        if self.image is None:
            return
        info = sample_color(self.image, x, y, 2)
        self.view.set_marker((x, y))
        self.color_sw.set(info.hex)
        self.color_name.text = info.name
        self.color_hex.text = info.hex

    def _on_pick(self, x, y):
        if self.image is None:
            return
        info = sample_color(self.image, x, y, 2)
        self.picked, self.pick_pos = info, (x, y)
        self.view.set_marker((x, y))
        self._apply_range(range_around(info, self.tol_sl.get()), preset=None)
        self.toast(f"{info.name}  {info.hex}  ({info.family})")

    def _on_tolerance(self, _v):
        if self.picked is not None:
            self._apply_range(range_around(self.picked, self.tol_sl.get()), preset=None)

    def choose_preset(self, name):
        self.picked = None
        self.pick_pos = None
        self.view.set_marker(None)
        self._apply_range(PRESETS[name], preset=name)

    def _on_range_slider(self, key, value):
        self.picked = None
        self._apply_range(self.rng.with_values(**{key: value}), preset=None, move_sliders=False)

    def _apply_range(self, rng, preset=None, move_sliders=True):
        self.rng, self.preset = rng, preset
        if move_sliders:
            for key, v in rng.to_dict().items():
                self.range_sliders[key].set(v)
        for name, chip in self.chips.items():
            chip.set_active(name == preset)
        self.hue_bar.set_range(rng)
        self.range_lbl.text = rng.text() if rng.valid else "Saturation/Value: min is above max — nothing matches."
        self.range_lbl.color = rgba(C["muted"] if rng.valid else C["warning"])
        self._show_color_info()
        self.schedule_refresh()

    def _show_color_info(self):
        if self.picked is not None:
            p = self.picked
            self.color_sw.set(p.hex)
            self.color_name.text = p.name
            self.color_hex.text = p.hex
            self.pick_info.text = (f"{p.name}  ({p.family})\nHEX  {p.hex}\nRGB  {p.rgb_text}\n"
                                   f"HSV  {p.hsv_text}")
        else:
            label = f"{self.preset} preset" if self.preset else "Custom range"
            self.color_name.text = label
            self.color_hex.text = "trackbars" if not self.preset else "tap the image"
            mean = self.result.mean_bgr if self.result.pixel_count else None
            self.color_sw.set(describe_bgr(mean).hex if mean is not None else None)
            self.pick_info.text = "Nothing picked yet: tap the picture."

    # ============================================================== settings
    def params(self):
        return CountParams(blur=self.blur_sl.get(), morph=self.morph_sl.get(),
                           fill_holes=self.fill_tg.value, separate=self.sep_tg.value,
                           split=self.split_sl.get(), min_area=self.min_sl.get(),
                           max_area=self.max_sl.get(), exclude_border=self.edge_tg.value)

    def reset_defaults(self):
        d = CountParams()
        for sl, v in ((self.min_sl, d.min_area), (self.max_sl, d.max_area), (self.blur_sl, d.blur),
                      (self.morph_sl, d.morph), (self.split_sl, d.split), (self.tol_sl, DEFAULT_TOL)):
            sl.set(v)
        self.fill_tg.set(d.fill_holes)
        self.sep_tg.set(d.separate)
        self.edge_tg.set(d.exclude_border)
        self.view_seg.select("highlight", notify=False)
        self.view_mode = "highlight"
        if self.picked is not None:
            self._apply_range(range_around(self.picked, DEFAULT_TOL), preset=None)
        else:
            self._apply_range(PRESETS["Red"], preset="Red")
        self.toast("Settings reset")

    def _on_view(self, mode):
        self.view_mode = mode
        self._render()

    # ============================================================ processing
    def schedule_refresh(self, *_):
        if self.cam is not None or self.image is None:
            return
        if self._refresh_ev is not None:
            self._refresh_ev.cancel()
        self._refresh_ev = Clock.schedule_once(lambda dt: self.refresh(), 0.05)

    def refresh(self, live=False):
        self._refresh_ev = None
        if self.image is None:
            return
        self.result = count_color(self.image, self.rng, self.params())
        if self.highlight_id is not None and all(o.id != self.highlight_id for o in self.result.objects):
            self.highlight_id = None
        self._render()
        self._update_stats(live)
        self._objects_dirty = True
        now = time.monotonic()
        if self.tab_seg.value == "objects" and (not live or now - self._last_list > 0.6):
            self._fill_objects()

    def _render(self):
        if self.image is None:
            return
        mode = self.view_mode
        if mode == "mask" and self.result.mask is not None:
            out = self.result.mask
        elif mode == "original":
            out = self.image
        else:
            out = annotate(self.image, self.result, dim_others=mode == "highlight",
                           highlight=self.highlight_id)
        self.view.show(out)

    def _update_stats(self, live):
        r = self.result
        if live:
            self._recent.append(r.count)
            n = int(round(float(np.median(self._recent))))
        else:
            n = r.count
        self.obj_lbl.text = str(n)
        self.px_lbl.text = f"{r.pixel_count:,}"
        self.pct_lbl.text = f"{r.percent:.2f}% of {r.total_pixels:,} px"
        if self.picked is None:
            self._show_color_info()

    def _fill_objects(self):
        self._last_list = time.monotonic()
        if not self._objects_dirty and self.obj_list.children:
            return
        self._objects_dirty = False
        self.obj_list.clear_widgets()
        objs = self.result.objects
        if not objs:
            self.obj_list.add_widget(auto_height(text(
                "No objects. Pick a color, or lower 'Minimum object area' in Settings.", 12, "muted"), dp(8)))
            return
        for o in objs[:150]:
            row = ObjectRow(o, o.id == self.highlight_id, self._select_object)
            self.obj_list.add_widget(row)
        if len(objs) > 150:
            self.obj_list.add_widget(auto_height(text(f"… and {len(objs) - 150} more", 11, "muted")))

    def _select_object(self, oid):
        self.highlight_id = None if self.highlight_id == oid else oid
        self._objects_dirty = True
        self._fill_objects()
        self._render()
        for o in self.result.objects:
            if o.id == oid and self.highlight_id:
                self.toast(f"Object #{o.id}: {o.color_pixels:,} color px, roundness {o.circularity:.2f}")

    # ================================================================== save
    def _output_dir(self):
        if ANDROID:
            from androidstorage4kivy import SharedStorage
            return SharedStorage().get_cache_dir()
        d = os.path.join(os.path.expanduser("~"), "Pictures", "ColorCount")
        os.makedirs(d, exist_ok=True)
        return d

    def _share(self, path, kind):
        if ANDROID:
            self._legacy_storage_permission()
            from androidstorage4kivy import SharedStorage
            SharedStorage().copy_to_shared(path)
            where = "Pictures" if kind == "image" else "Documents"
            return f"Saved to {where}/{self.title}"
        return f"Saved to {path}"

    def _legacy_storage_permission(self):
        """Android 9 and older need WRITE_EXTERNAL_STORAGE to save into Pictures."""
        try:
            from android import api_version
            from android.permissions import Permission, check_permission, request_permissions
            if api_version < 29 and not check_permission(Permission.WRITE_EXTERNAL_STORAGE):
                request_permissions([Permission.WRITE_EXTERNAL_STORAGE])
        except Exception:
            pass

    def save_result(self):
        if self.image is None:
            self.toast("Nothing to save yet")
            return
        try:
            img = annotate(self.image, self.result, dim_others=self.view_mode == "highlight",
                           banner=True) if self.view_mode in ("highlight", "objects") else (
                self.result.mask if self.view_mode == "mask" else self.image)
            path = os.path.join(self._output_dir(), time.strftime("colorcount_%Y%m%d_%H%M%S.png"))
            if not save_image(path, img):
                raise OSError("could not write the image")
            self.toast(self._share(path, "image"))
        except Exception as exc:
            self.toast(f"Save failed: {exc}")

    def save_csv(self):
        if self.image is None:
            self.toast("Nothing to save yet")
            return
        try:
            path = os.path.join(self._output_dir(), time.strftime("colorcount_%Y%m%d_%H%M%S.csv"))
            export_csv(self.result, self.rng, path)
            self.toast(self._share(path, "csv"))
        except Exception as exc:
            self.toast(f"Save failed: {exc}")

    # ================================================================= misc
    def toast(self, msg, seconds=2.6):
        lbl = self.toast_lbl
        Animation.cancel_all(lbl)
        lbl.text = msg
        lbl.text_size = (min(Window.width - dp(48), dp(420)), None)
        self._place_toast()
        lbl.opacity = 1
        anim = Animation(opacity=1, d=seconds) + Animation(opacity=0, d=0.4)
        anim.start(lbl)

    def on_start(self):
        if selftest_requested():
            from selftest import SelfTest
            Clock.schedule_once(lambda dt: SelfTest(self).start(), 4)

    def on_pause(self):
        self.stop_camera()
        return True

    def on_stop(self):
        self.stop_camera()


class ObjectRow(ButtonBehavior, BoxLayout):
    def __init__(self, obj, selected, command, **kw):
        super().__init__(size_hint_y=None, height=dp(34), padding=(dp(6), 0), **kw)
        paint(self, C["accent_dim"] if selected else C["card2"], radius=dp(8))
        for t, w, align in ((str(obj.id), 0.12, "left"), (f"{obj.color_pixels:,}", 0.3, "right"),
                            (f"{obj.area:,.0f}", 0.3, "right"), (f"{obj.circularity:.2f}", 0.28, "right")):
            self.add_widget(text(t, 12, mono=True, size_hint_x=w, halign=align))
        self.bind(on_release=lambda *_: command(obj.id))


if __name__ == "__main__":
    ColorCountApp().run()
