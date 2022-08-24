'''
Adaptado de https://github.com/yalaudah/facies_classification_benchmark
'''

import numpy as np
from PIL import Image

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):

        img, mask = Image.fromarray(img, mode=None), Image.fromarray(mask, mode='L')
        assert img.size == mask.size
        
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)

    
class HorizontallyFlip(object):
    def __call__(self, img, mask):

        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

   
