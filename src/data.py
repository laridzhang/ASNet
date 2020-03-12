import numpy as np
import os
import cv2
import random
import pandas
import scipy.io as scio
import torch
import torch.nn as nn
import psutil
import warnings
from os.path import join

from src.utils import ndarray_to_tensor
from src.data_path import DataPath


DOWNSAMPLE = 8  # The output of the network needs to be downsampled to one of DOWNSAMPLE


class LoadData:
    def __init__(self, image_path, density_map_path, roi_path=None, shuffle=False, random_seed=10, pre_load=False, is_label=False):
        # image_path: path of all image file
        # density_map_path: path of all density map file
        # roi_path: path of all region of interest file
        # shuffle: shuffle the order of images
        # random_seed: random seed of shuffle
        # pre_load

        self.image_path = image_path
        self.density_map_path = density_map_path
        self.roi_path = roi_path
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.pre_load = pre_load
        self.is_label = is_label

        self.image_channel = 3
        self.number_of_labels = 5
        self.batch_size = 1
        self.min_available_memory = 2 * 1024 * 1024 * 1024  # 2GB

        # get all image file name
        self.image_filename_list = [filename for filename in os.listdir(self.image_path) if os.path.isfile(join(self.image_path, filename))]
        self.image_filename_list.sort()

        if self.shuffle:
            random.seed(self.random_seed)

        self.number_of_samples = len(self.image_filename_list)
        self.blob_dict = dict()

        # preload data
        if self.pre_load is None:
            print('Pre-loading data.')
            index = 0
            for filename in self.image_filename_list:
                if psutil.virtual_memory().available > self.min_available_memory:
                    self.blob_dict[filename] = self.get_blob(filename)
                    index += 1
                else:
                    self.blob_dict[filename] = None
                    warnings.warn('not enough free memory. some of the data will not preload.', ResourceWarning)
                if index % 100 == 0:
                    print('Loaded %6d of %d files.' % (index, self.number_of_samples))
            print('Completed loading %6d of %d files. Others are not preloaded.' % (index, self.number_of_samples))
        elif self.pre_load:
            print('Pre-loading data.')
            index = 0
            for filename in self.image_filename_list:
                self.blob_dict[filename] = self.get_blob(filename)
                index += 1
                if index % 100 == 0:
                    print('Loaded %6d of %d files.' % (index, self.number_of_samples))
            print('Completed loading %6d files.' % index)
        else:
            for filename in self.image_filename_list:
                self.blob_dict[filename] = None

        # compute label weights for bce loss function
        if self.is_label:
            label_histogram = np.zeros(self.number_of_labels)
            index = 0
            for filename in self.image_filename_list:
                blob = self.blob_dict[filename]
                if blob is None:
                    blob = self.get_blob(filename)
                label_histogram[torch.argmax(blob['label'])] += 1
                index += 1
                if index % 100 == 0:
                    print('Built %6d of %d labels.' % (index, self.number_of_samples))
            print('Completed building %d labels. Label histogram is %s' % (index, ' '.join([str(i) for i in label_histogram])))
            self.label_weights = 1 - label_histogram / sum(label_histogram)
            self.label_weights = self.label_weights / sum(self.label_weights)
        return

    def __call__(self, new_batch_size):
        self.batch_size = new_batch_size
        return self

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.image_filename_list)

        batch = dict()
        index = 0

        for filename in self.image_filename_list:
            blob = self.blob_dict[filename]
            if blob is None:
                blob = self.get_blob(filename)

            if self.pre_load is None:
                if psutil.virtual_memory().available > self.min_available_memory:
                    self.blob_dict[filename] = blob
                else:
                    self.blob_dict[filename] = None

            if index == 0:
                batch['image_name'] = blob['image_name']
                batch['image'] = blob['image']
                batch['density'] = blob['density']
                batch['roi'] = blob['roi']
                batch['label'] = blob['label']
            else:
                batch['image_name'] = 'batched_images'
                batch['image'] = torch.cat((batch['image'], blob['image']), dim=0)
                batch['density'] = torch.cat((batch['density'], blob['density']), dim=0)
                roi_map = blob['roi']
                if roi_map is not None:
                    batch['roi'] = torch.cat((batch['roi'], roi_map), dim=0)
                else:
                    batch['roi'] = None
                if self.is_label:
                    batch['label'] = torch.cat((batch['label'], blob['label']), dim=0)
                else:
                    batch['label'] = None

            index += 1
            if index >= self.batch_size:
                yield batch
                index = 0

    def get_blob(self, filename):
        image_name, _ = os.path.splitext(filename)
        blob = dict()
        blob['image_name'] = image_name

        if self.image_channel == 1:
            image = cv2.imread(join(self.image_path, filename), 0)
        elif self.image_channel == 3:
            image = cv2.imread(join(self.image_path, filename), 1)
        else:
            raise Exception('invalid number of image channels')

        density_map = pandas.read_csv(join(self.density_map_path, image_name + '.csv'), sep=',', header=None).values

        if image.shape[0] != density_map.shape[0] or image.shape[1] != density_map.shape[1]:
            raise Exception('density map size mismatch.')

        density_map = self.downsample(density_map, DOWNSAMPLE)

        if self.roi_path is not None:
            if DOWNSAMPLE == 1:
                roi = self.load_roi(join(self.roi_path, image_name + '_roi.mat'))
            elif DOWNSAMPLE == 4:
                roi = self.load_roi(join(self.roi_path, image_name + '_roi_fourth_size.mat'))
            elif DOWNSAMPLE == 8:
                roi = self.load_roi(join(self.roi_path, image_name + '_roi_eighth_size.mat'))
            else:
                raise Exception('no suitable RoI file available')
        else:
            roi = None

        image = self.reshape_data(image)
        density_map = self.reshape_data(density_map)
        if roi is not None:
            roi = self.reshape_data(roi)
            if roi.shape[2] != density_map.shape[2] or roi.shape[3] != density_map.shape[3]:
                raise Exception('RoI size mismatch')

        blob['image'] = ndarray_to_tensor(image, is_cuda=False)
        blob['density'] = ndarray_to_tensor(density_map, is_cuda=False)
        if roi is not None:
            blob['roi'] = ndarray_to_tensor(roi, is_cuda=False)
        else:
            blob['roi'] = None

        if self.is_label:
            blob['label'] = ndarray_to_tensor(self.get_label(blob['density']), is_cuda=False)
        else:
            blob['label'] = None
        return blob

    def compute_label(self, count):
        if count == 0:
            return 0
        # label = int(min(max(np.floor(np.log2(density * 3200 / DOWNSAMPLE ** 2)), 0), self.number_of_labels - 1))
        label = int(min(max(np.floor(np.log2(count / 10)), 0), self.number_of_labels - 1))
        return label

    def get_label(self, density_map):
        # density_map torch.Tensor shape=(1, 1, h, w)
        if density_map.shape[0] != 1:
            raise Exception('invalid density map shape')
        # average_density = torch.mean(density_map)
        count = torch.sum(density_map)
        label = np.zeros(self.number_of_labels, dtype=np.int)
        label[self.compute_label(count)] = 1
        return label.reshape(1, -1)

    def get_label_weights(self):
        return self.label_weights

    def get_number_of_samples(self):
        return self.number_of_samples

    @staticmethod
    def downsample(density_map, downsample_value=1):
        # height and width of output density map are about 1/[downsample_value] times that of original density map
        import torch
        import torch.nn.functional as functional

        if density_map.shape[0] % downsample_value != 0 or density_map.shape[1] % downsample_value != 0:
            raise Exception('density map size is not suitable for downsample value')

        density_map = density_map.reshape((1, 1, density_map.shape[0], density_map.shape[1]))
        if downsample_value > 1:
            density_map_tensor = torch.tensor(density_map, dtype=torch.float32)
            density_map_tensor = functional.avg_pool2d(density_map_tensor, downsample_value, stride=downsample_value)
            density_map = density_map_tensor.data.cpu().numpy()
            density_map = density_map * downsample_value * downsample_value
        density_map = density_map.reshape((density_map.shape[2], density_map.shape[3]))

        return density_map

    @staticmethod
    def load_roi(path):
        roi_mat = scio.loadmat(path)
        roi = roi_mat['roi']
        raw_mask = roi['mask']
        mask = raw_mask[0, 0]
        mask = mask.astype(np.float32, copy=False)
        return mask

    @staticmethod
    def reshape_data(data):
        # data numpy.ndarray shape=(x, y) or (x, y, 3)
        # return numpy.ndarray shape=(1, 1, x, y) or (1, 3, x, y)
        data = data.astype(np.float32, copy=False)
        height = data.shape[0]
        width = data.shape[1]
        if len(data.shape) == 3 and data.shape[2] == 3:
            image_r = data[:, :, 0]
            image_g = data[:, :, 1]
            image_b = data[:, :, 2]
            image = np.zeros((3, height, width), dtype=np.float32)
            image[0] = image_r
            image[1] = image_g
            image[2] = image_b
            reshaped_data = image.reshape((1, 3, height, width))
        elif len(data.shape) == 2:
            reshaped_data = data.reshape((1, 1, height, width))
        else:
            raise Exception('Invalid data shape.')

        return reshaped_data


class MaskLabel:
    def __init__(self, data, size=9):
        if not isinstance(size, int):
            raise Exception('size should be an integer')

        if size % 2 == 0:
            raise Exception('size should be an odd integer')

        self.average_pool = nn.AvgPool2d(size, stride=1, padding=int((size - 1) / 2), count_include_pad=False)

        index = 0
        number_of_samples = data.get_number_of_samples()
        density_list = list()

        print('Building mask label.')
        for blob in data:
            index += 1

            ground_truth_data = blob['density_map']

            if np.sum(ground_truth_data) == 0:
                continue

            ground_truth = ndarray_to_tensor(ground_truth_data)
            pooled_ground_truth = self.average_pool(ground_truth)
            pooled_ground_truth = pooled_ground_truth.data.cpu().numpy()

            density_list.extend(pooled_ground_truth[pooled_ground_truth > 0])

            if index % 100 == 0:
                print('Built %6d / %d mask labels.' % (index, number_of_samples))

        self.density_flag = np.percentile(density_list, 75)

        print('Finished building mask.')

    def get_mask(self, ground_truth_map):
        # ground_truth numpy.ndarray shape=(1, 1, h, w)
        # return mask numpy.ndarray shape=(1, 3, h, w)

        foreground_mask = np.zeros_like(ground_truth_map)
        foreground_mask[ground_truth_map > 0] = 1.0

        ground_truth = ndarray_to_tensor(ground_truth_map)
        pooled_ground_truth = self.average_pool(ground_truth)
        pooled_ground_truth = pooled_ground_truth.data.cpu().numpy()

        # start_density = 0
        # end_density = self.low_density_flag
        # zeros = np.zeros_like(pooled_ground_truth)
        # zeros[np.logical_and(pooled_ground_truth > start_density, pooled_ground_truth <= end_density)] = 1
        # mask = zeros * foreground_mask
        #
        # start_density = self.low_density_flag
        # end_density = self.high_density_flag
        # zeros = np.zeros_like(pooled_ground_truth)
        # zeros[np.logical_and(pooled_ground_truth > start_density, pooled_ground_truth <= end_density)] = 1
        # mask = np.concatenate((mask, zeros * foreground_mask), axis=1)
        #
        # start_density = self.high_density_flag
        # end_density = sys.maxsize
        # zeros = np.zeros_like(pooled_ground_truth)
        # zeros[np.logical_and(pooled_ground_truth > start_density, pooled_ground_truth <= end_density)] = 1
        # mask = np.concatenate((mask, zeros * foreground_mask), axis=1)

        zeros = np.zeros_like(pooled_ground_truth)
        zeros[pooled_ground_truth < self.density_flag] = 1
        mask = zeros * foreground_mask

        zeros = np.zeros_like(pooled_ground_truth)
        zeros[pooled_ground_truth >= self.density_flag] = 1
        mask = np.concatenate((mask, zeros * foreground_mask), axis=1)

        return mask


class Data:
    def __init__(self, data_config):
        # data_config: dict, a dictionay contains several datasets info,
        #              key is dataset name,
        #              value is a dict which contains is_preload and is_label and is_mask
        data_path = DataPath()

        self.data = dict()

        for name in data_config:
            this_dataset_flag = data_config[name]
            is_preload = this_dataset_flag['preload']
            if 'label' in this_dataset_flag:
                is_label = this_dataset_flag['label']
            else:
                is_label = False
            if 'mask' in this_dataset_flag:
                is_mask = this_dataset_flag['mask']
            else:
                is_mask = False
            if 'shuffle' in this_dataset_flag:
                is_shuffle = this_dataset_flag['shuffle']
            else:
                is_shuffle = False
            if 'seed' in this_dataset_flag:
                random_seed = this_dataset_flag['seed']
            else:
                random_seed = None

            path = data_path.get_path(name)
            loaded_data = LoadData(path['image'], path['gt'], roi_path=path['roi'],
                                   shuffle=is_shuffle, random_seed=random_seed, pre_load=is_preload, is_label=is_label)

            this_dataset_dict = dict()
            this_dataset_dict['data'] = loaded_data
            this_dataset_dict['mask'] = MaskLabel(loaded_data) if is_mask else None

            self.data[name] = this_dataset_dict

    def get(self):
        return self.data
