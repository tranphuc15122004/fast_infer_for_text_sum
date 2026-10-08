"""Training-free Context-Adaptive DFlash inference adapter.

Keep this module lightweight so CPU-only preparation/report phases do not
import Torch or initialize CUDA.
"""

from .config import AdaptiveConfig
from .types import Action, GenerationResult, PromptLayout, RoundState

__all__ = ["Action", "AdaptiveConfig", "GenerationResult", "PromptLayout", "RoundState"]
