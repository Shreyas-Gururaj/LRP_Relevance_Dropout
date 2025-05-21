from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RelevantErasing_Block(object):
    ''' 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, mean, xy_rel_center, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.xy_rel_center = xy_rel_center
        
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):

            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                size_rank = self.xy_rel_center.shape[0]
                rand_k = random.randint(0, size_rank-1)
                x1 = self.xy_rel_center[rand_k][0].item()
                y1 = self.xy_rel_center[rand_k][1].item()
                
                x_lh = int(round((x1-(h/2))))
                if x_lh < 0:
                    x_lh = 0
            
                x_rh = int(round((x1+(h/2))))
                if x_rh > img.size()[1]:
                    x_rh = img.size()[1]
                        
                y_down = int(round((y1-(w/2))))
                if y_down < 0:
                    y_down = 0
                    
                y_up = int(round((y1+(w/2))))
                if y_up > img.size()[2]:
                    y_down = img.size()[2]
                
                if img.size()[0] == 3:
                    img[0, x_lh:x_rh, y_down:y_up] = self.mean[0]
                    img[1, x_lh:x_rh, y_down:y_up] = self.mean[1]
                    img[2, x_lh:x_rh, y_down:y_up] = self.mean[2]
                else:
                    img[0, int(round(x1-h/2)):int(round(x1+h/2)), int(round(y1-w/2)):int(round(y1+w/2))] = self.mean[0]
                return img
        return img
