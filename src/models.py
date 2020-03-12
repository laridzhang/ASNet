import torch
import torch.nn as nn
import torch.nn.functional as functional
import cv2
import numpy as np

from src.network import Conv2d, ConvTranspose2d
from src.utils import ndarray_to_tensor
from src.data_multithread_preload import DOWNSAMPLE


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.prior = nn.Sequential(Conv2d(3, 64, 3, same_padding=True),
                                   Conv2d(64, 64, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(64, 128, 3, same_padding=True),
                                   Conv2d(128, 128, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(128, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(256, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 256, 1, same_padding=True),
                                   ConvTranspose2d(256, 128, 2, stride=2, padding=0),
                                   Conv2d(128, 128, 3, same_padding=True),
                                   Conv2d(128, 3, 1, same_padding=True))

        self.vgg16 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True),
                                   Conv2d(64, 64, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(64, 128, 3, same_padding=True),
                                   Conv2d(128, 128, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(128, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(256, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 256, 1, same_padding=True),
                                   ConvTranspose2d(256, 128, 2, stride=2, padding=0))

        self.map = nn.Sequential(Conv2d(128, 128, 3, same_padding=True),
                                 Conv2d(128, 2, 1, same_padding=True))

        self.scale = nn.Sequential(Conv2d(128, 128, 3, same_padding=True),
                                   Conv2d(128, 2, 1, same_padding=True, relu=False),
                                   nn.AdaptiveAvgPool2d(1),
                                   nn.Hardtanh(-1.0, 1.0))

    def forward(self, im_data, roi=None):
        with torch.no_grad():
            x_prior = self.prior(im_data)
            flag = torch.argmax(x_prior, dim=1, keepdim=True)

            background_mask = (flag == 0).to(torch.float32)
            foreground_mask = 1 - background_mask
            resized_foreground_mask = functional.interpolate(1 - background_mask, scale_factor=8.0, mode='nearest')

            # masks of foreground classes
            masks = None
            for i in range(1, x_prior.shape[1]):
                if masks is None:
                    masks = (flag == i).to(torch.float32)
                else:
                    masks = torch.cat((masks, (flag == i).to(torch.float32)), dim=1)

            dilate_size = 4
            if dilate_size > 1:
                _, number_of_classes, _, _ = masks.shape
                # pad mask for same size output
                if dilate_size % 2 == 0:
                    pad_size = (dilate_size / 2, dilate_size / 2 - 1, dilate_size / 2, dilate_size / 2 - 1)
                    pad_size = *(int(i) for i in pad_size),
                else:
                    pad_size = int((dilate_size - 1) / 2)
                    pad_size = (pad_size, pad_size, pad_size, pad_size)
                padded_mask = functional.pad(masks, pad_size, mode='constant', value=0)
                # dilate mask using convolution function
                padded_mask_list = torch.chunk(padded_mask, number_of_classes, dim=1)
                dilated_masks = None
                filters = torch.ones(1, 1, dilate_size, dilate_size).cuda()
                for i in range(number_of_classes):
                    if dilated_masks is None:
                        dilated_masks = torch.clamp(functional.conv2d(padded_mask_list[i], filters), 0, 1) * foreground_mask
                    else:
                        dilated_masks = torch.cat((dilated_masks, torch.clamp(functional.conv2d(padded_mask_list[i], filters), 0, 1) * foreground_mask), dim=1)
            else:
                dilated_masks = masks

            dilated_masks = torch.round(dilated_masks).to(torch.float32)

        x1 = self.vgg16(im_data * resized_foreground_mask)
        maps = self.map(x1)
        scales = self.scale(x1) + 1

        if dilated_masks.shape != maps.shape:
            raise Exception('mask and map mismatch')
        if dilated_masks.shape[1] != scales.shape[1]:
            raise Exception('mask and scale mismatch')

        flag = torch.sum(dilated_masks, 1, keepdim=True) + background_mask
        if torch.min(flag) < 1:  # there should not be any zeros in flag
            raise Exception('invalid dilated masks')

        scaled_maps = maps * dilated_masks * scales

        scaled_map = torch.sum(scaled_maps, 1, keepdim=True) / flag
        density_map = torch.sum(scaled_map, 1, keepdim=True)

        resized_roi = functional.interpolate(roi, scale_factor=1 / DOWNSAMPLE, mode='nearest')
        density_map = density_map * resized_roi

        visual_dict = dict()
        visual_dict['density'] = density_map
        visual_dict['raw_maps'] = maps
        visual_dict['scaled_maps'] = scaled_maps
        visual_dict['masks'] = dilated_masks

        return density_map, foreground_mask, visual_dict