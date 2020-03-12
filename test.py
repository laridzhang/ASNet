import numpy as np
import time

from src.utils import calculate_game
from src.crowd_count import CrowdCount
from src.data_multithread_preload import multithread_dataloader
from src import network


test_flag = dict()
test_flag['preload'] = False
test_flag['label'] = False
test_flag['mask'] = False

test_model_path = r'./final_model/shtechA.h5'
# original_dataset_name = 'shtechA'
test_data_config = dict()
test_data_config['shtA1_test'] = test_flag.copy()

# load data
all_data = multithread_dataloader(test_data_config)

net = CrowdCount()

network.load_net(test_model_path, net)

net.cuda()
net.eval()

total_forward_time = 0.0

# calculate error on the test dataset
for data_name in test_data_config:
    data = all_data[data_name]['data']

    mae = 0.0
    mse = 0.0
    game_0 = 0.0
    game_1 = 0.0
    game_2 = 0.0
    game_3 = 0.0
    index = 0
    for blob in data:
        image_data = blob['image']
        ground_truth_data = blob['density']
        roi = blob['roi']
        image_name = blob['image_name'][0]

        start_time = time.perf_counter()
        estimate_map, _, visual_dict = net(image_data, roi=roi)
        total_forward_time += time.perf_counter() - start_time

        ground_truth_map = ground_truth_data.data.cpu().numpy()
        estimate_map = estimate_map.data.cpu().numpy()

        ground_truth_count = np.sum(ground_truth_map)
        estimate_count = np.sum(estimate_map)

        mae += np.abs(ground_truth_count - estimate_count)
        mse += (ground_truth_count - estimate_count) ** 2
        game_0 += calculate_game(ground_truth_map, estimate_map, 0)
        game_1 += calculate_game(ground_truth_map, estimate_map, 1)
        game_2 += calculate_game(ground_truth_map, estimate_map, 2)
        game_3 += calculate_game(ground_truth_map, estimate_map, 3)
        index += 1

    mae = mae / index
    mse = np.sqrt(mse / index)
    game_0 = game_0 / index
    game_1 = game_1 / index
    game_2 = game_2 / index
    game_3 = game_3 / index
    print('mae: %.2f mse: %.2f game: %.2f %.2f %.2f %.2f' % (mae, mse, game_0, game_1, game_2, game_3))

print('total forward time is %f seconds of %d samples.' % (total_forward_time, index))

