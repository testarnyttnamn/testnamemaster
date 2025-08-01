# -*- coding: utf-8 -*-

"""COSMOSIS INTEGRATION

Integration of CLOE with CosmoSIS.

"""

# Explicitly import all subpackages so Sphinx autodoc can discover them
from . import (
    cosmosis_interface,
    cosmosis_with_cobaya_interface,
    camb_interface,
)

# Define public API
__all__ = [
    "cosmosis_interface",
    "cosmosis_with_cobaya_interface",
    "camb_interface",
]
