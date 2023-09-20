"""
The :mod:`phaser.hashing` module includes various utilities.

H1 inside module contents
=========================
Add some important text here to describe or link to examples.

"""

from ._algorithms import (
    PHASH,
    ComputeHashes,
    ColourHash,
    WaveHash,
    PdqHash
    )

# Include names of private functions to autodoc
__all__ = [
    "PHASH",
    "ComputeHashes",
    "ColourHash",
    "WaveHash",
    "PdqHash"
    ]