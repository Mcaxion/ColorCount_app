"""
Automated self-test, used only by the GitHub emulator test.
Started with:  adb shell am start ... --es selftest 1
It presses every feature in order and prints results to the Android log
("SELFTEST STEP n ...") so the test script can take a screenshot after each step.
It also prints screen coordinates ("TAPCOORD name x y") for real finger taps.
"""
import traceback

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window

import main as M


def log(msg):
    print(msg, flush=True)


class SelfTest:
    def __init__(self, app):
        self.a = app

    # ---------------------------------------------------------------- helpers
    def status_top(self):
        if not M.ANDROID:
            return 0
        try:
            from jnius import autoclass
            Rect = autoclass("android.graphics.Rect")
            act = autoclass("org.kivy.android.PythonActivity").mActivity
            r = Rect()
            act.getWindow().getDecorView().getWindowVisibleDisplayFrame(r)
            log(f"SELFTEST INFO visible-frame top={r.top} bottom={r.bottom} left={r.left} right={r.right}")
            return int(r.top)
        except Exception as exc:
            log(f"SELFTEST INFO status_top failed: {exc}")
            return 0

    def screen(self, kx, ky):
        """Kivy window coordinates (origin bottom-left) -> Android screen pixels."""
        return int(kx), int(self.top + Window.height - ky)

    def tapcoord(self, name, kx, ky):
        x, y = self.screen(kx, ky)
        log(f"TAPCOORD {name} {x} {y}")

    def widget(self, name, w):
        self.tapcoord(name, *w.to_window(*w.center))

    def blue_pixel(self):
        hsv = cv2.cvtColor(self.a.image, cv2.COLOR_BGR2HSV)
        ys, xs = np.where((hsv[..., 0] > 100) & (hsv[..., 0] < 115) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150))
        i = len(xs) // 2
        return int(xs[i]), int(ys[i])

    def image_point(self, name, ix, iy):
        v = self.a.view
        x0, y0, s = v._fit
        h = v.img_size[1]
        self.tapcoord(name, x0 + (ix + 0.5) * s, y0 + h * s - (iy + 0.5) * s)

    def result(self):
        r = self.a.result
        return f"objects={r.count} pixels={r.pixel_count} ({r.percent:.2f}%)"

    # ------------------------------------------------------------------ steps
    def s_layout(self):
        self.top = self.status_top()
        return f"window={tuple(Window.size)} density={Window.dpi} status_top={self.top}"

    def s_sample(self):
        self.a.load_path(M.SAMPLE, "Sample: candies")
        return "red preset " + self.result()

    def s_pick(self):
        x, y = self.blue_pixel()
        self.a._on_preview(x, y)
        self.a._on_pick(x, y)
        self.a.refresh()
        return f"picked={self.a.picked.hex} {self.a.picked.name} {self.result()} range=[{self.a.rng.text()}]"

    def s_trackbars(self):
        self.a.tab_seg.select("trackbars")
        return "trackbars tab shown"

    def s_slider_coords(self):
        sl = self.a.range_sliders["h_max"].slider
        vx, vy = sl.to_window(*sl.value_pos)
        rx, ry = sl.to_window(sl.right - sl.padding, sl.center_y)
        self.tapcoord("hmax_from", vx, vy)
        self.tapcoord("hmax_to", rx, ry)
        sl.value = 179
        self.a.refresh()
        return "h_max=179 -> " + self.result()

    def s_mask(self):
        self.a.view_seg.select("mask")
        return "mask view"

    def s_objects(self):
        self.a.view_seg.select("highlight")
        self.a.choose_preset("Yellow")
        self.a.refresh()
        self.a.tab_seg.select("objects")
        self.a._select_object(2)
        return f"yellow {self.result()} rows={len(self.a.obj_list.children)}"

    def s_save(self):
        self.a.save_result()
        img_msg = self.a.toast_lbl.text
        self.a.save_csv()
        return f"image: {img_msg} | csv: {self.a.toast_lbl.text}"

    def s_settings(self):
        self.a.tab_seg.select("settings")
        return "settings tab shown"

    def s_camera(self):
        self.a.camera_button()
        return f"camera started={self.a.cam is not None} msg={self.a.toast_lbl.text}"

    def s_camera_live(self):
        img = None if self.a.image is None else self.a.image.shape
        return f"frames={self.a._frames} image={img} {self.result()}"

    def s_capture(self):
        if self.a.cam is not None:
            self.a.camera_button()
        return f"source={self.a.source_name} camera_off={self.a.cam is None} {self.result()}"

    def s_gallery(self):
        if not M.ANDROID:
            return "skipped on desktop"
        self.a.open_gallery()
        return "gallery picker requested"

    def s_final(self):
        self.a.tab_seg.select("colors")
        self.a.reset_defaults()
        self.a.load_path(M.SAMPLE, "Sample: candies")
        a = self.a
        self.widget("btn_sample", a.btn_sample)
        self.widget("btn_camera", a.btn_camera)
        self.widget("tab_trackbars", a.tab_seg.buttons["trackbars"])
        self.widget("tab_objects", a.tab_seg.buttons["objects"])
        self.widget("tab_colors", a.tab_seg.buttons["colors"])
        self.widget("view_mask", a.view_seg.buttons["mask"])
        self.widget("view_highlight", a.view_seg.buttons["highlight"])
        self.widget("chip_green", a.chips["Green"])
        x, y = self.blue_pixel()
        self.image_point("candy", x, y)
        return "coordinates printed; " + self.result()

    # ------------------------------------------------------------------- run
    def start(self):
        self.top = 0
        self.steps = [
            ("layout", self.s_layout, 3), ("sample", self.s_sample, 4), ("pick", self.s_pick, 4),
            ("trackbars", self.s_trackbars, 4), ("slider", self.s_slider_coords, 4), ("mask", self.s_mask, 4),
            ("objects", self.s_objects, 4), ("save", self.s_save, 4), ("settings", self.s_settings, 4),
            ("camera", self.s_camera, 7), ("camera_live", self.s_camera_live, 4), ("capture", self.s_capture, 4),
            ("gallery", self.s_gallery, 9), ("final", self.s_final, 3),
        ]
        log("SELFTEST BEGIN")
        self.run(0)

    def run(self, i):
        if i >= len(self.steps):
            log("SELFTEST DONE")
            return
        name, fn, wait = self.steps[i]
        try:
            info = fn()
            log(f"SELFTEST STEP {i + 1} {name} OK | {info}")
        except Exception:
            log(f"SELFTEST STEP {i + 1} {name} FAIL | " + traceback.format_exc().replace("\n", " || "))
        Clock.schedule_once(lambda dt: self.run(i + 1), wait)
