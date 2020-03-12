from math import ceil
import torch
import torch.nn as nn
# import numpy as np
import torch.nn.functional as functional
from functools import reduce

# from src import network
from src.models import Model
from src.utils import build_block


# only calculate with pixels which have value
# pool size=stride=4 or 2
# unless specified, the default Gaussian kernel size is 15 and sigma is 4
# the first threshold is the average density of pooling ground truth
# the next threshold is the average pooling density of the area remaining after excluding areas with a density below the previous threshold
# different pooling density maps are inconsistent in size
dataset_density_level = dict()
dataset_density_level['shtA1_train_4_2'] = (3.206821266542162, 1.7318443746056567)
dataset_density_level['shtA1_train_8_4'] = (8.464520906327428, 4.9102145881524950)


class CrowdCount(nn.Module):
    def __init__(self):
        super(CrowdCount, self).__init__()
        self.features = Model()
        self.my_loss = None
        self.this_dataset_density_level = dataset_density_level['shtA1_train_8_4']

    @property
    def loss(self):
        return self.my_loss

    def forward(self, im_data, roi, ground_truth=None):
        estimate_map, foreground_mask, visual_dict = self.features(im_data.cuda(), roi.cuda())

        if self.training:
            self.my_loss, loss_dict = self.build_loss(ground_truth.cuda(), estimate_map, foreground_mask)
        else:
            loss_dict = None

        return estimate_map, loss_dict, visual_dict

    def build_loss(self, ground_truth_map, estimate_map, foreground_mask):
        if ground_truth_map.shape != estimate_map.shape:
            raise Exception('shapes of ground_truth_map and estimate_map are mismatch')
        if ground_truth_map.shape != foreground_mask.shape:
            raise Exception('shapes of ground_truth_map and foreground_mask are mismatch')

        ground_truth_map = ground_truth_map * foreground_mask
        estimate_map = estimate_map * foreground_mask

        pool8_loss_map = self.pooling_loss_map(ground_truth_map, estimate_map, 8)
        pool4_loss_map = self.pooling_loss_map(ground_truth_map, estimate_map, 4)

        foreground_active_for_pool8 = functional.interpolate(foreground_mask, scale_factor=1 / 8.0, mode='nearest')
        foreground_active_for_pool4 = functional.interpolate(foreground_mask, scale_factor=1 / 4.0, mode='nearest')

        pool8_deactive = build_block(ground_truth_map, 8)
        pool8_deactive[pool8_deactive < self.this_dataset_density_level[0]] = 0.0
        pool8_deactive[pool8_deactive > 0] = 1.0
        pool8_active = 1 - pool8_deactive

        pool8_deactive_for_pool4 = functional.interpolate(pool8_deactive, scale_factor=2.0, mode='nearest')
        pool4_active = torch.ones_like(pool4_loss_map)
        if pool8_deactive_for_pool4.shape != pool4_active.shape:
            raise Exception('active map mismatch')
        pool4_active = pool4_active * pool8_deactive_for_pool4

        pool8_active = pool8_active * foreground_active_for_pool8
        pool4_active = pool4_active * foreground_active_for_pool4

        pool8_active_sum = torch.sum(pool8_active)
        pool4_active_sum = torch.sum(pool4_active)

        pool8_loss = torch.sum(pool8_loss_map * pool8_active) / pool8_active_sum if pool8_active_sum > 0 else torch.sum(pool8_loss_map * pool8_active)
        pool4_loss = torch.sum(pool4_loss_map * pool4_active) / pool4_active_sum if pool4_active_sum > 0 else torch.sum(pool4_loss_map * pool4_active)
        total_loss = pool8_loss * 4 + pool4_loss

        loss_dict = dict()
        loss_dict['pool8'] = pool8_loss
        loss_dict['pool4'] = pool4_loss
        loss_dict['total'] = total_loss
        return total_loss, loss_dict

    @staticmethod
    def pooling_loss_map(ground_truth, estimate, block_size=4):
        square_error = (ground_truth - estimate) ** 2
        element_amount = reduce(lambda x, y: x * y, square_error.shape)
        block_square_error = build_block(square_error / element_amount, block_size)
        block_ground_truth = build_block(ground_truth, block_size)
        block_loss = block_square_error / (block_ground_truth + 1)
        return block_loss
