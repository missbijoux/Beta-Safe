# BetaSafe

BetaSafe is a **desktop screen overlay** that runs **outside the browser**. It watches your display(s), looks for **image-like regions** (photos, video thumbnails, embedded pictures), and can **mosaic or blur** them in real time. You control it from the **system tray** (menu next to the clock).

It is meant for people who want an extra layer of visual privacy on their own machine—for example when browsing, streaming, or screen sharing.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Windows 10/11** or **macOS** | Multi-monitor setups are supported. |
| **Python 3.10+** | Only if you run from source (see below). |
| **Administrator / special permissions** | Usually **not** required for normal use. macOS may ask for **Screen Recording** so the app can see the screen to protect it. |

---

## Quick start (from source)

These steps are for **developers** or anyone comfortable with a terminal.

### 1. Get the code

Clone this repository (or download and unzip it), then open a terminal **in the project folder** (the one that contains `requirements.txt` and `src/`).

### 2. Create a virtual environment (recommended)

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run BetaSafe

**macOS / Linux:**

```bash
python3 -m src
```

**Windows:**

```powershell
python -m src
```

You should see a **tray icon**. Use **right‑click** (or equivalent) on the icon for:

- **Pause / resume** (if you enabled optional hotkeys, **F10** may also toggle pause)
- **Mosaic** or **Blur** mode
- **Click-through** (whether mouse clicks pass through the overlay)
- **Quit**

If you do not see the tray icon, your desktop environment may hide “background” apps—check the **^** / chevron area on the Windows taskbar, or the menu bar extras on macOS.

---

## First-time tips

1. **Start with pause or low impact**  
   While tuning performance, you can start with the overlay logic off and still use the tray:

   ```bash
   export BETASAFE_START_PAUSED=1   # macOS/Linux — omit on Windows or use set BETASAFE_START_PAUSED=1
   python3 -m src
   ```

   On Windows PowerShell, use:

   ```powershell
   $env:BETASAFE_START_PAUSED="1"
   python -m src
   ```

2. **macOS permissions**  
   If capture fails or the screen stays unchanged, open **System Settings → Privacy & Security → Screen Recording** and allow **Terminal** (if you run from Terminal) or **Python** / **BetaSafe** (if you run a built app).

3. **Performance**  
   Lower **FPS** and detection rate if the machine feels slow. See **Settings** below.

4. **Optional “adult content” gate**  
   Without extra ML packages, BetaSafe can still mosaic **heuristic** “looks like a photo” regions. To only mosaic regions that pass an **NSFW classifier**, install the optional dependencies in `requirements.txt` (comment lines for `transformers` / `torch` or `onnxruntime`) and set the matching `BETASAFE_*` environment variables. This is heavier on CPU/GPU and download size.

---

## Settings (environment variables)

Most behavior is tuned with variables whose names start with `BETASAFE_`. The full list and defaults are documented in code in [`src/config.py`](src/config.py) (long docstring at the top).

Common ones:

| Variable | What it does (short) |
|----------|----------------------|
| `BETASAFE_TARGET_FPS` | How often the pipeline runs (default 12). Lower = less CPU. |
| `BETASAFE_DETECT_EVERY` | Run detection every N frames (default 3). Higher = cheaper. |
| `BETASAFE_MAX_PROCESS_WIDTH` | Downscale width for processing (default 720). Smaller = faster. |
| `BETASAFE_MAX_CENSOR_AREA` | Ignore rectangles larger than this fraction of the screen (default 0.92) to avoid accidental full-screen mosaic. |
| `BETASAFE_HIDE_DELAY_MS` | Delay before hiding empty overlay layers (reduces flicker). |
| `BETASAFE_ADULT_HF_MODEL` | Hugging Face model id for optional NSFW gating (requires extra installs). |
| `BETASAFE_ADULT_ONNX_PATH` | Path to an ONNX NSFW classifier (alternative to HF). |
| `BETASAFE_START_PAUSED` | If set truthy, starts without drawing overlays (useful for testing). |
| `BETASAFE_AUTO_QUIT_MS` | Quit automatically after N milliseconds (debug / CI). |

**Windows only:** `BETASAFE_USE_DXGI` (default on) uses fast capture when `dxcam` works; otherwise the app falls back to `mss`.

---

## Building a downloadable app (installers)

If you want a **`.exe` + folder`**, a **Windows installer**, or a **macOS `.app` / DMG`**, follow the step-by-step guide in:

**[`packaging/BUILD.txt`](packaging/BUILD.txt)**

That file covers PyInstaller, Inno Setup (Windows), DMG layout, and optional macOS signing/notarization.

**Branding assets** (app icon, installer wizard art, DMG background) live under `packaging/icons/` and `packaging/assets/`.

---

## GitHub Actions (optional)

A ready-made workflow file is kept at **`packaging/ci/build-pyinstaller.yml`**. To run builds on GitHub:

1. Copy it to `.github/workflows/build-pyinstaller.yml` in your clone.  
2. Commit and push using credentials that are allowed to update workflows (some tokens need the **workflow** scope).

---

## Troubleshooting

| Problem | Things to try |
|---------|----------------|
| Tray says “no icon” or icon is blank | Known quirk on some setups; the menu still works. You can add a proper icon in Qt later. |
| High CPU / fans | Lower `BETASAFE_TARGET_FPS`, raise `BETASAFE_DETECT_EVERY`, lower `BETASAFE_MAX_PROCESS_WIDTH`. |
| Random small mosaics | Tighten adult threshold or disable extra tile candidates; see past tuning notes in `src/detect.py` / `src/config.py`. |
| macOS freezes or compositor issues | Try lowering FPS and `BETASAFE_HIDE_DELAY_MS`; avoid toggling huge fullscreen translucent layers every frame. |

---

## Project layout (short)

| Path | Role |
|------|------|
| `src/main.py` | Starts the Qt app. |
| `src/overlay.py` | Tray, overlays, frame coordinator. |
| `src/config.py` | Environment-driven settings. |
| `src/detect.py` | Heuristic “image-like” regions. |
| `src/mosaic.py` | Mosaic / blur. |
| `packaging/` | Icons, installer assets, PyInstaller spec, scripts. |

---

## License

Add a `LICENSE` file to this repository if you want to specify how others may use the code. Until then, assume **all rights reserved** unless you state otherwise.

---

## Disclaimer

This tool processes **your screen contents** on your machine for the purpose you configure. It is **not** a guarantee of safety, compliance, or age verification. You are responsible for how you use it and for obeying local laws and platform terms.

If something feels wrong, use **Pause** or **Quit** from the tray immediately.
