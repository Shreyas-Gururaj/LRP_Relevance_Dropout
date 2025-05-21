from __future__ import absolute_import

from torchvision.transforms import * # This is a bit broad, consider specific imports

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, mean, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.log_aspect_ratio = (math.log(self.r1), math.log(1/self.r1)) # Consistent aspect ratio sampling
       
    def __call__(self, img): # img is expected to be a Tensor (C, H, W)

        if random.uniform(0, 1) >= self.probability: # Changed to >= to match RelDrop logic (no erase if random >= prob)
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            # aspect_ratio = random.uniform(self.r1, 1/self.r1) # Original
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))


            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w > 0 and h > 0 and w < img.size()[2] and h < img.size()[1]: # Ensure w,h > 0
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    # img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    # img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    img[0, x1:x1+h, y1:y1+w] = 255
                    img[1, x1:x1+h, y1:y1+w] = 255
                    img[2, x1:x1+h, y1:y1+w] = 255
                elif img.size()[0] == 1: # Grayscale
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0] if isinstance(self.mean, (list, tuple)) else self.mean
                else:
                    # Handle other channel numbers if necessary, or raise error
                    pass # Or raise error for unsupported channel size
                return img # Return after one successful erase attempt
        return img
