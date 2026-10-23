[app]

# ---- name shown on the phone ---------------------------------------------------
title = ColorCount
package.name = colorcount
package.domain = org.visiontoolkit
version = 1.0

# ---- what goes into the APK ----------------------------------------------------
source.dir = .
source.include_exts = py,png,jpg
source.exclude_dirs = bin, .buildozer, venv, __pycache__, .git

# Python + Kivy (UI) + NumPy + OpenCV (image processing) + Android file access
requirements = python3,kivy,numpy,opencv,androidstorage4kivy

# ---- look ------------------------------------------------------------------------
icon.filename = %(source.dir)s/assets/icon.png
presplash.filename = %(source.dir)s/assets/presplash.png
android.presplash_color = #0f1117
orientation = portrait
fullscreen = 0

# ---- Android ---------------------------------------------------------------------
# CAMERA for the live camera. Storage is only needed on Android 9 and older to save
# pictures; newer phones save through the system without a permission.
android.permissions = CAMERA, (name=android.permission.WRITE_EXTERNAL_STORAGE;maxSdkVersion=28)
android.api = 33
android.minapi = 24
android.ndk = 25b
# 64-bit ARM covers practically every phone from the last 6+ years.
# Add ", armeabi-v7a" for very old 32-bit phones (doubles the build time).
android.archs = arm64-v8a
android.accept_sdk_license = True
android.allow_backup = True

# A fixed python-for-android release, so the build is the same every time
p4a.branch = v2024.01.21


[buildozer]
log_level = 2
# Colab and WSL often run as root; don't stop to ask about it
warn_on_root = 0
