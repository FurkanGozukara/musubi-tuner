"""Optimizers bundled with Musubi Tuner."""

from .automagic import Automagic
from .automagic2 import Automagic2
from .automagic3 import Automagic3
from .automagic_experiment import AutomagicEXPERIMENT

__all__ = ["Automagic", "Automagic2", "Automagic3", "AutomagicEXPERIMENT"]
