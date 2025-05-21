from __future__ import absolute_import

import torch
import random
import math
import numpy as np
from zennit.attribution import Gradient
from zennit.core import Stabilizer
# Assuming BatchNormalize is defined elsewhere or passed if needed by composite_creator_fn
# For now, composite_creator_fn is defined in the main script and passed in.

class BatchNormalize: # Definition from cifar_imagenet.py for standalone use if needed
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

class BatchRelevantErasing_Base(object):
    def __init__(self, model, composite_creator_fn, num_classes, mean, probability, sl, sh, r1):
        self.model = model
        self.composite_creator_fn = composite_creator_fn # Expected signature: fn(device, input_batch_shape) -> LRP Composite
        self.num_classes = num_classes
        self.mean = mean # list or tuple of means
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.log_aspect_ratio = (math.log(self.r1), math.log(1/self.r1))


    def _get_attributions(self, inputs_batch, targets_batch):
        original_training_mode = self.model.training
        self.model.eval() # LRP should be done in eval mode

        composite = self.composite_creator_fn(inputs_batch.device, inputs_batch.shape)
        
        if targets_batch.ndim == 1 or (targets_batch.ndim == 2 and targets_batch.shape[-1] == 1):
            targets_one_hot = torch.eye(self.num_classes, device=inputs_batch.device)[targets_batch.squeeze().long()]
        else: 
            targets_one_hot = targets_batch.to(device=inputs_batch.device, dtype=torch.float)

        with Gradient(model=self.model, composite=composite) as attributor:
            inputs_for_attr = inputs_batch.clone().requires_grad_(True)
            _, attribution = attributor(inputs_for_attr, targets_one_hot)
        
        if original_training_mode:
            self.model.train() 
        return attribution

class BatchRelevantErasing_Block(BatchRelevantErasing_Base):
    def __init__(self, model, composite_creator_fn, num_classes, mean, probability, sl, sh, r1, k_centers_to_consider):
        super().__init__(model, composite_creator_fn, num_classes, mean, probability, sl, sh, r1)
        self.k_centers_to_consider = k_centers_to_consider # Number of top relevant pixels to choose one from

    def __call__(self, inputs_batch, targets_batch):
        if random.uniform(0, 1) >= self.probability:
            return inputs_batch

        attribution = self._get_attributions(inputs_batch, targets_batch)
        relevance = torch.squeeze(attribution.sum(1)) # Sum over channels: (B, H, W)
        relevance_flattened = relevance.view(relevance.shape[0], -1) # (B, H*W)

        # Get top k candidate centers for each image in the batch
        # Ensure k is not larger than the number of pixels
        actual_k_centers = min(self.k_centers_to_consider, relevance_flattened.shape[1])
        if actual_k_centers <=0: return inputs_batch # Should not happen with valid k

        _, top_k_flat_indices = torch.topk(relevance_flattened, largest=True, k=actual_k_centers, dim=1)

        outputs_batch = inputs_batch.clone()
        img_h, img_w = relevance.shape[1], relevance.shape[2]

        for i in range(inputs_batch.shape[0]): # Iterate over batch
            top_k_flat_indices_img = top_k_flat_indices[i]
            
            centers_x = torch.div(top_k_flat_indices_img, img_w, rounding_mode='trunc')
            centers_y = top_k_flat_indices_img % img_w
            xy_rel_centers_for_img = torch.stack([centers_x, centers_y], dim=-1) # Shape: (actual_k_centers, 2)

            img_single = outputs_batch[i] # C, H, W
            applied_erase = False
            for _attempt in range(100): 
                area = img_single.size()[1] * img_single.size()[2]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                
                h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
                w_erase = int(round(math.sqrt(target_area / aspect_ratio)))

                if w_erase > 0 and h_erase > 0 and w_erase < img_single.size()[2] and h_erase < img_single.size()[1]:
                    # Randomly pick one of the top-k relevant pixels as the center
                    rand_k_idx = random.randint(0, xy_rel_centers_for_img.shape[0] - 1)
                    center_x_coord = xy_rel_centers_for_img[rand_k_idx, 0].item() # row index (h)
                    center_y_coord = xy_rel_centers_for_img[rand_k_idx, 1].item() # col index (w)

                    x1 = int(round(center_x_coord - h_erase / 2))
                    y1 = int(round(center_y_coord - w_erase / 2))
                    
                    x1_clamped = max(0, x1)
                    y1_clamped = max(0, y1)
                    x2_clamped = min(img_single.size()[1], x1 + h_erase) # Exclusive endpoint
                    y2_clamped = min(img_single.size()[2], y1 + w_erase) # Exclusive endpoint
                    
                    if (x2_clamped > x1_clamped) and (y2_clamped > y1_clamped):
                        if img_single.size()[0] == 3:
                            img_single[0, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[0]
                            img_single[1, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[1]
                            img_single[2, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[2]
                        else: # Grayscale
                            img_single[0, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[0]
                        applied_erase = True
                        break # Break from attempt loop, one block erased for this image
            # outputs_batch[i] is already modified
        return outputs_batch

class BatchRelevantErasing_Patch(BatchRelevantErasing_Base):
    def __init__(self, model, composite_creator_fn, num_classes, mean, probability, sl, sh, r1, k_patches_to_erase):
        super().__init__(model, composite_creator_fn, num_classes, mean, probability, sl, sh, r1)
        # k_patches_to_erase is the 'count' from original RelevantErasing_Patch.
        # It defines the number of top relevant pixels to consider AND scales the target_area.
        # At most ONE patch (scaled by this count) will be applied due to early return logic.
        self.k_patches_to_erase = max(1, k_patches_to_erase) # Ensure count is at least 1

    def __call__(self, inputs_batch, targets_batch):
        if random.uniform(0, 1) >= self.probability:
            return inputs_batch

        attribution = self._get_attributions(inputs_batch, targets_batch)
        relevance = torch.squeeze(attribution.sum(1)) # (B, H, W)
        relevance_flattened = relevance.view(relevance.shape[0], -1) # (B, H*W)

        # Number of candidate centers is k_patches_to_erase
        actual_k_candidate_centers = min(self.k_patches_to_erase, relevance_flattened.shape[1])
        if actual_k_candidate_centers <= 0: return inputs_batch

        _, top_k_flat_indices = torch.topk(relevance_flattened, largest=True, k=actual_k_candidate_centers, dim=1)

        outputs_batch = inputs_batch.clone()
        img_h, img_w = relevance.shape[1], relevance.shape[2]

        for i in range(inputs_batch.shape[0]): # Iterate over batch
            top_k_flat_indices_img = top_k_flat_indices[i]
            
            centers_x = torch.div(top_k_flat_indices_img, img_w, rounding_mode='trunc')
            centers_y = top_k_flat_indices_img % img_w
            # xy_candidate_centers shape: (actual_k_candidate_centers, 2)
            xy_candidate_centers = torch.stack([centers_x, centers_y], dim=-1)

            img_single = outputs_batch[i] # C, H, W
            
            # Replicating original RelevantErasing_Patch logic:
            # It tries 'count' (self.k_patches_to_erase) times to place a patch.
            # If any attempt (within 100 tries for that "count" iteration) is successful, it returns.
            # This means at most one patch is applied, scaled by 'count'.
            
            applied_one_patch_for_image = False
            # This outer loop corresponds to `for _ in range(count):` in original RelevantErasing_Patch
            for _patch_iteration_num in range(self.k_patches_to_erase): 
                if applied_one_patch_for_image: # If a patch was already applied to this image
                    break

                # Inner loop for 100 attempts for the current _patch_iteration_num
                for _attempt in range(100):
                    area = img_single.size()[1] * img_single.size()[2]
                    # Target area is scaled by k_patches_to_erase (the 'count')
                    target_area_scaled = random.uniform(self.sl, self.sh) * area / self.k_patches_to_erase
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))

                    h_erase = int(round(math.sqrt(target_area_scaled * aspect_ratio)))
                    w_erase = int(round(math.sqrt(target_area_scaled / aspect_ratio)))

                    if w_erase > 0 and h_erase > 0 and w_erase < img_single.size()[2] and h_erase < img_single.size()[1]:
                        # Randomly pick one of the top-k relevant pixels as the center
                        rand_k_idx = random.randint(0, xy_candidate_centers.shape[0] - 1)
                        center_x_coord = xy_candidate_centers[rand_k_idx, 0].item()
                        center_y_coord = xy_candidate_centers[rand_k_idx, 1].item()

                        x1 = int(round(center_x_coord - h_erase / 2))
                        y1 = int(round(center_y_coord - w_erase / 2))

                        x1_clamped = max(0, x1)
                        y1_clamped = max(0, y1)
                        x2_clamped = min(img_single.size()[1], x1 + h_erase)
                        y2_clamped = min(img_single.size()[2], y1 + w_erase)

                        if (x2_clamped > x1_clamped) and (y2_clamped > y1_clamped):
                            if img_single.size()[0] == 3:
                                img_single[0, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[0]
                                img_single[1, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[1]
                                img_single[2, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[2]
                            else:
                                img_single[0, x1_clamped:x2_clamped, y1_clamped:y2_clamped] = self.mean[0]
                            
                            applied_one_patch_for_image = True
                            break # Break from attempt loop (inner)
                
                if applied_one_patch_for_image:
                    break # Break from _patch_iteration_num loop (outer)
            # outputs_batch[i] is already modified
        return outputs_batch

class BatchRelevantPixelDropout(BatchRelevantErasing_Base):
    def __init__(self, model, composite_creator_fn, num_classes, mean, probability, alpha_dropout_rel, beta_dropout_rel):
        super().__init__(model, composite_creator_fn, num_classes, mean, probability, 0.0, 0.0, 0.0) 
        self.alpha_dropout_rel = alpha_dropout_rel
        self.beta_dropout_rel = beta_dropout_rel

    def __call__(self, inputs_batch, targets_batch):
        outputs_batch = inputs_batch.clone()
        
        attribution = self._get_attributions(inputs_batch, targets_batch)
        relevance_batch = torch.squeeze(attribution.sum(1)) # (B, H, W)

        for i in range(inputs_batch.shape[0]):
            if random.random() < self.probability: 
                img_single = outputs_batch[i] 
                relevance_single = relevance_batch[i]

                min_rel = torch.min(relevance_single)
                max_rel = torch.max(relevance_single)
                if (max_rel - min_rel).abs() > 1e-6: # Avoid division by zero/small number
                    relevance_single_norm = (relevance_single - min_rel) / (max_rel - min_rel)
                else:
                    relevance_single_norm = torch.zeros_like(relevance_single)
                
                # Ensure random_tensor is on the same device as relevance_single_norm
                random_tensor = torch.rand_like(relevance_single_norm) 

                cond = ((self.alpha_dropout_rel * random_tensor) + \
                       ((1.0 - self.alpha_dropout_rel) * relevance_single_norm)) >= (1.0 - self.beta_dropout_rel)
                
                if img_single.size()[0] == 3:
                    img_single[0][cond] = self.mean[0]
                    img_single[1][cond] = self.mean[1]
                    img_single[2][cond] = self.mean[2]
                else: 
                    img_single[0][cond] = self.mean[0]
        
        return outputs_batch
