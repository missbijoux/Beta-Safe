"""Allow ``python -m src`` and PyInstaller entry matching ``python -m src.main``."""

from __future__ import annotations

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
