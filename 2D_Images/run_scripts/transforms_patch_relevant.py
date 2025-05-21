from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

# class RelevantErasing_Patch(object):
#     '''
#     Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
#     -------------------------------------------------------------------------------------
#     probability: The probability that the operation will be performed.
#     sl: min erasing area
#     sh: max erasing area
#     r1: min aspect ratio
#     mean: erasing value
#     -------------------------------------------------------------------------------------
#     '''
#     def __init__(self, xy_rel_center, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, h_new=4, w_new=4, mean=[0.4914, 0.4822, 0.4465]):
#         self.probability = probability
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#         self.xy_rel_center = xy_rel_center
#         self.h_new = h_new
#         self.w_new = w_new
        
#     def __call__(self, img):

#         if random.uniform(0, 1) > self.probability:
#             return img

#         # for attempt in range(100):
#         #     area = img.size()[1] * img.size()[2]
       
#         #     target_area = random.uniform(self.sl, self.sh) * area
#         #     aspect_ratio = random.uniform(self.r1, 1/self.r1)

#         #     h = int(round(math.sqrt(target_area * aspect_ratio)))
#         #     w = int(round(math.sqrt(target_area / aspect_ratio)))

#         for x_y in self.xy_rel_center:
#             # if w < img.size()[2] and h < img.size()[1]:
#             x1 = x_y[0].item()
#             y1 = x_y[1].item()
#             x_lh = int(math.ceil((x1-self.h_new)/2))
#             # if x_lh < 0:
#             #     x_lh = 0
            
#             x_rh = int(math.ceil((x1+self.h_new)/2))
#             # if x_rh > 31:
#             #     x_rh = 31
                        
#             y_down = int(math.ceil((y1-self.w_new)/2))
#             # if y_down < 0:
#             #     y_down = 0
                    
#             y_up = int(math.ceil((y1+self.w_new)/2))
#             # if y_up > 31:
#             #     y_down = 31
                        
#             # print(f'x_lh is {x_lh}, x_rh is {x_rh}, y_down is {y_down}, y_up is {y_up}')
#             if img.size()[0] == 3:
#                 img[0, x_lh:x_rh, y_down:y_up] = self.mean[0]
#                 img[1, x_lh:x_rh, y_down:y_up] = self.mean[1]
#                 img[2, x_lh:x_rh, y_down:y_up] = self.mean[2]
#             else:
#                 # print('Entered loop 2')
#                 img[0, int(math.ceil(x1-self.h_new/2)):int(math.ceil(x1+self.h_new/2)), int(math.ceil(y1-self.w_new/2)):int(math.ceil(y1+self.w_new/2))] = self.mean[0]
#             # return img
#         return img

class RelevantErasing_Patch(object):
    '''
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, mean, xy_rel_center, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, r2=None, min_count=1, max_count=None):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.xy_rel_center = xy_rel_center
        r2 = r2 or 1 / r1
        self.log_aspect_ratio = (math.log(r1), math.log(r2))
        self.min_count = min_count
        self.max_count = max_count or min_count
 
        
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        
        area = img.size()[1] * img.size()[2]
        count = self.min_count if self.min_count == self.max_count else \
            self.max_count

        for _ in range(count):
            for attempt in range(100):
        
                target_area = random.uniform(self.sl, self.sh) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))

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

