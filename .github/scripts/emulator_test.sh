#!/usr/bin/env bash
# Runs inside the Android emulator: install the APK, run the app's self-test,
# then tap through the app like a person. Screenshots + logs go to test-results/.
set -u
OUT=test-results
mkdir -p "$OUT"
PKG=org.visiontoolkit.colorcount
APK=$(ls apk/*.apk | head -n 1)

shot() { adb exec-out screencap -p > "$OUT/$1.png"; echo "screenshot $1"; }
pylog() { adb logcat -d -s python:* 2>/dev/null; }
wait_for() {  # wait_for "<text>" <max seconds>
  for _ in $(seq "$2"); do
    pylog | grep -q "$1" && return 0
    sleep 1
  done
  return 1
}
coords() { pylog | grep -o "TAPCOORD $1 [0-9]* [0-9]*" | tail -n 1 | awk '{print $3, $4}'; }
tap() {  # tap <name> <screenshot>
  local xy; xy=$(coords "$1")
  if [ -n "$xy" ]; then adb shell input tap $xy; sleep 4; shot "$2"; else echo "no coordinates for $1"; fi
}
foreground() { adb shell dumpsys activity activities | grep -m1 -E "topResumedActivity|mResumedActivity"; }

{
  echo "ABI: $(adb shell getprop ro.product.cpu.abilist)"
  echo "Android: $(adb shell getprop ro.build.version.release) (API $(adb shell getprop ro.build.version.sdk))"
  adb shell wm size; adb shell wm density
} > "$OUT/device.txt"
cat "$OUT/device.txt"

adb install -r -g "$APK" > "$OUT/install.txt" 2>&1
cat "$OUT/install.txt"

adb logcat -c
adb shell am start -n "$PKG/org.kivy.android.PythonActivity" --es selftest 1
sleep 8;  shot 00_launch
if wait_for "SELFTEST STEP 1 " 150; then sleep 2; shot 01_layout; else shot 01_TIMEOUT; fi

names=(x x sample pick trackbars slider mask objects save settings camera camera_live capture gallery final)
for n in $(seq 2 14); do
  if ! wait_for "SELFTEST STEP $n " 45; then echo "step $n never arrived"; shot "$(printf '%02d' "$n")_MISSING"; break; fi
  sleep 2
  shot "$(printf '%02d' "$n")_${names[$n]}"
  if [ "${names[$n]}" = "gallery" ]; then
    sleep 4; shot 13b_gallery_picker
    foreground | tee "$OUT/gallery_foreground.txt"
    if ! foreground | grep -q "$PKG"; then adb shell input keyevent 4; sleep 3; fi   # close the picker
  fi
done
wait_for "SELFTEST DONE" 30 && echo "self-test finished"
sleep 2; shot 15_after_selftest

# ---- real finger taps, like a person using the app ----
tap btn_sample      20_tap_sample
tap candy           21_tap_candy
tap tab_trackbars   22_tap_trackbars_tab
from=$(coords hmax_from); to=$(coords hmax_to)
if [ -n "$from" ] && [ -n "$to" ]; then
  set -- $from; fx=$1; fy=$2; set -- $to; tx=$1
  adb shell input swipe "$fx" "$fy" "$tx" "$fy" 600; sleep 4; shot 23_drag_hue_max
fi
tap view_mask       24_tap_mask_view
tap view_highlight  25_tap_highlight_view
tap tab_objects     26_tap_objects_tab
tap btn_camera      27_tap_camera
sleep 4; shot 28_camera_live
tap btn_camera      29_tap_capture

foreground > "$OUT/final_foreground.txt"
pylog > "$OUT/python_log.txt"
adb logcat -d '*:E' > "$OUT/errors_log.txt"
adb logcat -d | tail -n 4000 > "$OUT/full_log_tail.txt"
grep -E "SELFTEST|TAPCOORD|Traceback|Error" "$OUT/python_log.txt" > "$OUT/SUMMARY.txt" || true
echo "=== SUMMARY ==="; cat "$OUT/SUMMARY.txt"
exit 0
