"""Punjab GeoGuardian - AI-based monitoring of urban expansion and farmland loss.

AIRI Team PITB - AI Internship Task 3. Author: Hamza Nadeem.
"""
from .config import AOI_PRESETS, CLASS_NAMES, ProjectConfig

__version__ = "1.0.0"
__all__ = ["ProjectConfig", "AOI_PRESETS", "CLASS_NAMES", "GeoGuardian"]


def __getattr__(name):  # lazy import keeps `import geoguardian` light
    if name == "GeoGuardian":
        from .pipeline import GeoGuardian
        return GeoGuardian
    raise AttributeError(name)
