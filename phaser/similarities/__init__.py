"""
The :mod:`phaser.similarities` module includes various ...
"""

from ._distances import (
    find_inter_samplesize,
    IntraDistance,
    InterDistance
)

__all__ = [
    "IntraDistance",
    "InterDistance",
    "find_inter_samplesize"
]