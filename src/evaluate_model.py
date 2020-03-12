import numpy as np
import torch
import torch.nn as nn
import math
import os

from src.crowd_count import CrowdCount
from src import network
from src.utils import ndarray_to_tensor
from src.psnr import build_psnr
from src.ssim import SSIM


def evaluate_model(model_path, data):
    net = CrowdCount()
    network.load_net(model_path, net)
    net.cuda()
    net.eval()

    build_ssim = SSIM(window_size=11)

    game = GridAverageMeanAbsoluteError()
    
    mae = 0.0
    mse = 0.0
    psnr = 0.0
    ssim = 0.0
    game_0 = 0.0
    game_1 = 0.0
    game_2 = 0.0
    game_3 = 0.0
    index = 0

    for blob in data:
        image_data = blob['image']
        ground_truth_data = blob['density']
        roi = blob['roi']
        # filename = blob['filename']

        if image_data.shape[0] != 1:
            raise Exception('invalid image batch size (%d) for evaluation' % image_data.shape[0])

        with torch.no_grad():
            estimate_map, _, _ = net(image_data, roi)

        ground_truth_data = ground_truth_data.data.cpu().numpy()
        density_map = estimate_map.data.cpu().numpy()

        ground_truth_count = np.sum(ground_truth_data)
        estimate_count = np.sum(density_map)

        mae += abs(ground_truth_count - estimate_count)
        mse += (ground_truth_count - estimate_count) ** 2
        psnr += build_psnr(ground_truth_data, density_map)
        ssim += build_ssim(ndarray_to_tensor(ground_truth_data), ndarray_to_tensor(density_map)).item()
        game_0 += game.calculate_error(ground_truth_data, density_map, 0)
        game_1 += game.calculate_error(ground_truth_data, density_map, 1)
        game_2 += game.calculate_error(ground_truth_data, density_map, 2)
        game_3 += game.calculate_error(ground_truth_data, density_map, 3)
        index += 1

    result_dict = dict()
    result_dict['name'] = os.path.basename(model_path)
    result_dict['number'] = int(index)
    result_dict['mae'] = float(mae / index)
    result_dict['mse'] = float(np.sqrt(mse / index))
    result_dict['psnr'] = float(psnr / index)
    result_dict['ssim'] = float(ssim / index)
    result_dict['game_0'] = float(game_0 / index)
    result_dict['game_1'] = float(game_1 / index)
    result_dict['game_2'] = float(game_2 / index)
    result_dict['game_3'] = float(game_3 / index)

    return result_dict


class GridAverageMeanAbsoluteError:
    @staticmethod
    def calculate_error(ground_truth, estimate, L=0):
        # grid average mean absolute error
        # ground_truth Tensor: shape=(1, 1, h, w)
        # estimate Tensor: same shape of ground_truth
        ground_truth = ndarray_to_tensor(ground_truth, is_cuda=True)
        estimate = ndarray_to_tensor(estimate, is_cuda=True)
        height = ground_truth.shape[2]
        width = ground_truth.shape[3]
        times = math.sqrt(math.pow(4, L))
        padding_height = int(math.ceil(height / times) * times - height)
        padding_width = int(math.ceil(width / times) * times - width)
        if padding_height != 0 or padding_width != 0:
            m = nn.ZeroPad2d((0, padding_width, 0, padding_height))
            ground_truth = m(ground_truth)
            estimate = m(estimate)
            height = ground_truth.shape[2]
            width = ground_truth.shape[3]
        m = nn.AdaptiveAvgPool2d(int(times))
        ground_truth = m(ground_truth) * (height / times) * (width / times)
        estimate = m(estimate) * (height / times) * (width / times)
        game = torch.sum(torch.abs(ground_truth - estimate))
        return game.item()

    # @staticmethod
    # def calculate_error_not_include_pad(ground_truth, estimate, L=0):
    #     # grid average mean absolute error
    #     # ground_truth Tensor: shape=(1, 1, h, w)
    #     # estimate Tensor: same shape of ground_truth
    #     ground_truth = ndarray_to_tensor(ground_truth, is_cuda=True)
    #     estimate = ndarray_to_tensor(estimate, is_cuda=True)
    #     height = ground_truth.shape[2]
    #     width = ground_truth.shape[3]
    #     times = math.sqrt(math.pow(4, L))
    #     grid_height = int(math.ceil(height / times))
    #     grid_width = int(math.ceil(width / times))
    #     padding_height = int(math.ceil((grid_height * times - height) / 2))
    #     padding_width = int(math.ceil((grid_width * times - width) / 2))
    #     m = nn.AvgPool2d((grid_height, grid_width), stride=(grid_height, grid_width), padding=(padding_height, padding_width), count_include_pad=False)
    #     ground_truth = m(ground_truth) * (height / times) * (width / times)
    #     estimate = m(estimate) * (height / times) * (width / times)
    #     game = torch.sum(torch.abs(ground_truth - estimate))
    #     return game.item()
