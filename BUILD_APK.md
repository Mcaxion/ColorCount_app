# ColorCount for Android: how to build and install the APK

The Android app is **pure Python**: `main.py` (Kivy interface) plus `core/` (the same OpenCV code as the desktop version). An APK can only be built on **Linux**. On Windows, the easiest free way is **Google Colab**, which runs in your browser.

```
ColorCountAndroid/
├── main.py                 # the whole Android app (Python + Kivy)
├── core/                   # OpenCV logic: color detection, pixel count, object count
├── samples/                # demo pictures bundled inside the app
├── assets/                 # app icon + splash screen
├── buildozer.spec          # APK settings (name, icon, permissions, libraries)
├── build_apk_colab.ipynb   # one-click build notebook for Google Colab
└── requirements-desktop.txt
```

---

## Option A: Google Colab (recommended, nothing to install)

1. Go to **https://colab.research.google.com** and sign in with a Google account.
2. **File → Upload notebook**, then choose `build_apk_colab.ipynb`.
3. Run the cells **in order** (click ▶ and wait for each to finish):

   | Cell | What it does | Time |
   |---|---|---|
   | 1 | Installs the build tools | ~5 min |
   | 2 | Click **Choose Files** and upload **`ColorCountAndroid.zip`** | ~10 s |
   | 3 | Builds the APK (compiles Python, NumPy, OpenCV and Kivy for Android) | **40–60 min** the first time |
   | 4 | Downloads `colorcount-1.0-arm64-v8a-debug.apk` | ~10 s |

4. Keep the tab open during the build. If Colab disconnects, run all the cells again.

If cell 4 says the build failed, copy the lines it prints and send them to Claude.

---

## Option B: Your own laptop with WSL (Linux inside Windows)

```powershell
# PowerShell as Administrator, then restart the laptop
wsl --install -d Ubuntu-22.04
```

Then in the **Ubuntu** window:

```bash
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf automake libtool \
     pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev libltdl-dev
pip3 install --user buildozer "cython==0.29.37"
export PATH=$PATH:~/.local/bin

# copy the project into Linux (building from /mnt/c is very slow)
cp -r /mnt/c/Users/<you>/Documents/ColorCountAndroid ~/
cd ~/ColorCountAndroid
buildozer -v android debug          # first build: 40-60 min, needs ~10 GB free
cp bin/*.apk /mnt/c/Users/<you>/Desktop/
```

---

## Install it on the phone

1. Copy the `.apk` to the phone (USB cable, Google Drive, or Messenger/email to yourself).
2. Tap the file. Android asks to allow **Install unknown apps** for that app (Files, Drive, etc.). Allow it.
3. Tap **Install**. If Play Protect warns about an unknown developer, choose **Install anyway**. This happens with every self-built app.
4. Open **ColorCount**. The first time you tap **Camera**, allow camera access.

Works on Android 7.0 or newer with a 64-bit phone (practically all phones from 2017 onwards).

---

## Presentation demo (2 minutes)

1. **Sample**: loads the candy picture. The **Red** preset already counts **5 objects** and shows the pixel count.
2. **Tap a blue candy**. The trackbars jump to blue and the count changes to 5 blue objects with their pixel count.
3. **Mask** view: "these white pixels are exactly what is counted."
4. **Trackbars** tab: drag *Hue max* and watch the pixel and object counts change live.
5. **Objects** tab: tap a row to highlight that object.
6. **Camera**: point it at real colored objects (candies, bottle caps, Lego), tap one to pick its color, then **Capture**.
7. **Save**: the result picture goes to the phone's *Pictures/ColorCount* folder.

**Tip for real objects:** use a plain background (white paper or a dark table) and even light. If similar colors mix (orange and yellow, for example), lower *Pick tolerance* or narrow the Hue trackbars.

---

## Test on your computer first (optional)

```bash
pip install -r requirements-desktop.txt
python main.py
```

The same app opens in a window, which is handy for checking changes before a long APK build.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| The live camera picture is sideways | **Settings → Camera rotation**: try 270° or 0° |
| "Camera permission is needed" | Phone Settings → Apps → ColorCount → Permissions → Camera → Allow |
| The app closes right after the splash screen | Rebuild once more. If it still happens, connect the phone with a USB cable and run `adb logcat -s python` to see the error |
| The build stops with "No space left" | Colab: Runtime → Disconnect and delete runtime, then start again. WSL: free ~10 GB |
| Want to support very old 32-bit phones | In `buildozer.spec` set `android.archs = arm64-v8a, armeabi-v7a` (the build takes twice as long) |
