# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BetaSafe (onedir). Build from repo root:

    pip install -r requirements.txt -r requirements-build.txt
    pyinstaller -y --clean packaging/pyinstaller/betasafe.spec

Windows: dist/BetaSafe/BetaSafe.exe
macOS:   dist/BetaSafe.app
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

block_cipher = None

spec_dir = Path(SPECPATH).resolve()
project_root = spec_dir.parents[1]
src_main = project_root / "src" / "__main__.py"
icon_win = project_root / "packaging" / "icons" / "app.ico"
icon_mac = project_root / "packaging" / "icons" / "app.icns"

hiddenimports = [
    "src",
    "src.__main__",
    "src.main",
    "src.overlay",
    "src.config",
    "src.capture",
    "src.detect",
    "src.mosaic",
    "src.dxg_capture",
    "src.adult_worker",
    "src.pipeline_thread",
    "src.nsfw_regions",
    "src.nsfw_heuristic",
    "src.nsfw_hf",
    "src.nsfw_onnx",
    "src.yolov8_onnx",
]
if sys.platform == "win32":
    hiddenimports.append("dxcam")

a = Analysis(
    [str(src_main)],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

win_icon = str(icon_win) if icon_win.is_file() else None
mac_icon = str(icon_mac) if icon_mac.is_file() else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BetaSafe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=platform.system() == "Darwin",
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=win_icon if platform.system() == "Windows" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="BetaSafe",
)

if platform.system() == "Darwin":
    app = BUNDLE(
        coll,
        name="BetaSafe.app",
        icon=mac_icon,
        bundle_identifier="com.missbijoux.betasafe",
        info_plist={
            "CFBundleName": "BetaSafe",
            "CFBundleDisplayName": "BetaSafe",
            "NSHighResolutionCapable": True,
            "LSUIElement": True,
        },
    )
