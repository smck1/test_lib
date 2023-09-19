import imagehash
import numpy as np

# Cross module import exampl from https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/metrics/_classification.py
# from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils import bool2binstring

# HASHING FUNCTIONS
class PHASH():
    def __init__(self, hash_size=8, highfreq_factor=4):
        self.hash_size=hash_size
        self.highfreq_factor = highfreq_factor
        
    def fit(self, img):
        """Apply hashing algorithm to provided Pillow image

        Args:
            img (PIL.image): Pillow image

        Returns:
            str: hash-digest
        """
        hash = imagehash.phash(
            image=img, 
            hash_size=self.hash_size,
            highfreq_factor=self.highfreq_factor
            ).hash
        
        hash = np.array(hash).flatten()
        
        # Convert bool array to a string
        hash = bool2binstring(hash)
        return hash