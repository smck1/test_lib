import imagehash
import pdqhash
import numpy as np
from copy import deepcopy
import pandas as pd
from joblib import Parallel, delayed

# Local imports from ..utils
from ..utils import ImageLoader, bool2binstring

class PHASH():
    def __init__(self, hash_size=8, highfreq_factor=4):
        self.hash_size=hash_size
        self.highfreq_factor = highfreq_factor
        
    def fit(self, img):
        hash = imagehash.phash(
            image=img, 
            hash_size=self.hash_size,
            highfreq_factor=self.highfreq_factor
            ).hash
        
        hash = np.array(hash).flatten()
        
        # Convert bool array to a string
        binary_hash = bool2binstring(hash)
        return binary_hash
        
class ColourHash():
    def __init__(self, binbits=3) -> None:
        self.binbits = binbits

    def fit(self, img) -> str:
        hash = imagehash.colorhash(
            image=img,
            binbits=self.binbits).hash
        
        flat_hash = np.concatenate(hash).flatten()
        
        binary_hash = bool2binstring(flat_hash)
        return binary_hash
    
class WaveHash():
    def __init__(
            self, 
            hash_size = 8,
            image_scale = None,
            mode = 'haar',
            remove_max_haar_ll = True
            ) -> None:
        
        self.hash_size = hash_size
        self.image_scale = image_scale
        self.mode = mode
        self.remove_max_haar_ll = remove_max_haar_ll
        
    def fit(self, img)  -> str:
        hash = imagehash.whash(
            image=img,
            hash_size = self.hash_size,
            image_scale = self.image_scale,
            mode = self.mode,
            remove_max_haar_ll = self.remove_max_haar_ll).hash
        
        flat_hash = np.concatenate(hash).flatten()
        binary_hash = bool2binstring(flat_hash)
        return binary_hash

class PdqHash():
    def __init__(self) -> None:
        pass

    def fit(self, img):
        # https://github.com/faustomorales/pdqhash-python
        # pip install pdqhash
        # pdq expects the images as a numpy array. Convert accordingly
        # https://stackoverflow.com/questions/384759/how-do-i-convert-a-pil-image-into-a-numpy-array
        # np.bool_(pdqhash.compute(np.asarray(self.image))[0])
        flat_hash = pdqhash.compute(np.asarray(img))[0].astype(bool)
        #hex_hash = bool2hex(flat_hash)
        binary_hash = bool2binstring(flat_hash)
        return binary_hash

        