"""SEAL Watermark Experiment Package"""

from .config import Config, load_config
from .seal import SEAL
from .detector import Detector
from .transforms import Transforms

__version__ = "1.0.0"
__author__ = "Kasra Arabi et al."
__paper__ = "SEAL: Semantic Aware Image Watermarking (ICCV 2024)"

__all__ = [
    "Config",
    "load_config",
    "SEAL",
    "Detector",
    "Transforms",
]
