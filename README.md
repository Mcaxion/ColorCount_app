# ColorCount: COLOR PICK + ObjectCount (Android)

An Android app written in **Python** with **OpenCV** and **Kivy** that combines two computer-vision projects:

| Part | Title |
|---|---|
| **COLOR PICK** | A color detection application using image processing |
| **ObjectCount** | A simple object counting application using computer vision |

Tap a color in a photo or the live camera (or choose a preset, or move the HSV trackbars). The app counts **how many pixels** have that color and **how many separate objects** of that color are in the picture.

## How it works
```
image → blur → BGR→HSV → cv2.inRange(H,S,V min/max) → mask → pixel count (cv2.countNonZero)
      → opening/closing → fill holes → split touching objects (distance transform) → contours → object count
```

## Get the APK
Every push builds the APK automatically. When it's done, download **ColorCount.apk** from the **[Releases](../../releases/latest)** page (right side of the repository page).

See [BUILD_APK.md](BUILD_APK.md) for installing it on a phone and other ways to build it.

## Run on a computer
```bash
pip install -r requirements-desktop.txt
python main.py
```
