"""
OCR utilities for the Clinical AI Portal.

Engine priority:
  1. EasyOCR  – pure Python, no system install needed (preferred)
  2. Tesseract – system binary via pytesseract (fallback)

Both are optional. If neither is available the function raises a clear
RuntimeError so the caller can show a helpful message in the UI.
"""

import io
import os
from PIL import Image

# ─── EasyOCR (lazy-loaded so startup is not slowed) ────────────────────────
_easyocr_reader = None
_easyocr_available = None          # None = not yet probed


def _try_easyocr(img_array) -> str | None:
    """Return OCR text via EasyOCR, or None if unavailable/failed."""
    global _easyocr_reader, _easyocr_available

    # Fast-fail if we already know it's not installed
    if _easyocr_available is False:
        return None

    try:
        import easyocr  # noqa: PLC0415
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _easyocr_available = True
        results = _easyocr_reader.readtext(img_array, detail=0, paragraph=True)
        return "\n".join(results)
    except ImportError:
        _easyocr_available = False
        return None
    except Exception:
        return None


# ─── Tesseract (optional system binary) ────────────────────────────────────
_tesseract_available = None        # None = not yet probed


def set_tesseract_path() -> bool:
    """Attempt to locate the Tesseract executable on Windows."""
    global _tesseract_available
    try:
        import pytesseract  # noqa: PLC0415
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\Dishant\AppData\Local\Tesseract-OCR\tesseract.exe",
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                _tesseract_available = True
                return True
    except ImportError:
        pass
    _tesseract_available = False
    return False


def _try_tesseract(img: Image.Image) -> str | None:
    """Return OCR text via Tesseract, or None if unavailable/failed."""
    try:
        import pytesseract  # noqa: PLC0415
        set_tesseract_path()
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception:
        return None


# ─── Public API ─────────────────────────────────────────────────────────────
def image_to_text(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return plain text.

    Tries EasyOCR first (pure Python), then Tesseract.

    Raises:
        RuntimeError: with a user-friendly message if no engine works.
    """
    import numpy as np  # noqa: PLC0415

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # 1. EasyOCR
    result = _try_easyocr(img_array)
    if result is not None:
        return result

    # 2. Tesseract
    result = _try_tesseract(img)
    if result is not None:
        return result

    # 3. Neither available
    raise RuntimeError(
        "no_ocr_engine"          # sentinel string checked in the UI
    )
