import os

from flash_gradio.component import FlashGradio

__all__ = ["FlashGradio"]

_PACKAGE_ROOT = os.path.dirname(__file__)
TEMPLATES_ROOT = os.path.join(_PACKAGE_ROOT, "templates")
