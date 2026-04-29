"""Entry: ``python -m src`` or ``python -m src.main`` (from repo root)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .overlay import run_app


def main() -> int:
    # Some PySide6 builds expose this enum on Qt, not QApplication.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
