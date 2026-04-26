"""Model architectures for super-resolution."""

from src.models.edsr import EDSR
from src.models.common import ResBlock

__all__ = ["EDSR", "ResBlock"]
