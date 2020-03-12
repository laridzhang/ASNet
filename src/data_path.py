import os


class DataPath:
    def __init__(self):
        self.base_path_list = list()
        self.base_path_list.append(r'/home/antec/PycharmProjects/')
        self.base_path_list.append(r'/media/antec/storage/PycharmProjects')
        
        self.data_path = dict()

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_64_64/train'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_64_64/train_den'
        path['roi'] = None
        self.data_path['shtA0RandomOverturn_64_64_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_128/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_128/train_den'
        path['roi'] = None
        self.data_path['shtA0RandFlip_128_128_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_128/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_128/train_den'
        path['roi'] = None
        self.data_path['shtA0RandFlip_128_128_trainAsVali'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256/train'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256/train_den'
        path['roi'] = None
        self.data_path['shtA0RandomOverturn_128_256_train'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256_more_than_one_pedestrain/train'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256_more_than_one_pedestrain/train_den'
        path['roi'] = None
        self.data_path['shtA0RandomOverturn_128_256_more1_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_256_more_than_ten_pedestrian/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_256_more_than_ten_pedestrian/train_den'
        path['roi'] = None
        self.data_path['shtA0RandFlip_128_256_more10_train'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test_den'
        path['roi'] = None
        self.data_path['shtA1Resize1_test'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/train'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/train_den'
        path['roi'] = None
        self.data_path['shtA1Resize1_train'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb_times32/test'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb_times32/test_den'
        path['roi'] = None
        self.data_path['shtA1Resize1Times32_test'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/train_den'
        path['roi'] = None
        self.data_path['shtA1_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/test'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/test_den'
        path['roi'] = None
        self.data_path['shtA1_test'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_den'
        path['roi'] = None
        self.data_path['shtA9RandomOverturn_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_without_validation'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_without_validation_den'
        path['roi'] = None
        self.data_path['shtA9RandFlip_trainNoVali'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/validation'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/validation_den'
        path['roi'] = None
        self.data_path['shtA9RandFlip_vali'] = path

        path = dict()
        path['image'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb_times32/train'
        path['gt'] = 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb_times32/train_den'
        path['roi'] = None
        self.data_path['shtA9RandomOverturnTimes32_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_0_random_flip_128_128_rgb/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_0_random_flip_128_128_rgb/train_den'
        path['roi'] = None
        self.data_path['shtB0RandFlip_128_128_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_0_random_flip_more1pedestrian_128_128_rgb/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_0_random_flip_more1pedestrian_128_128_rgb/train_den'
        path['roi'] = None
        self.data_path['shtB0RandFlipMore1_128_128_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_1_rgb/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_1_rgb/train_den'
        path['roi'] = None
        self.data_path['shtB1_train'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_1_rgb/test'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_1_rgb/test_den'
        path['roi'] = None
        self.data_path['shtB1_test'] = path

        path = dict()
        path['image'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_flip_rgb/train'
        path['gt'] = r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_flip_rgb/train_den'
        path['roi'] = None
        self.data_path['shtB9RandFlip_train'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/1/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/1/train_den'
        path['roi'] = None
        self.data_path['ucf0RandFlip_128_128_train1'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/2/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/2/train_den'
        path['roi'] = None
        self.data_path['ucf0RandFlip_128_128_train2'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/3/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/3/train_den'
        path['roi'] = None
        self.data_path['ucf0RandFlip_128_128_train3'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/4/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/4/train_den'
        path['roi'] = None
        self.data_path['ucf0RandFlip_128_128_train4'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/5/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_0_random_rgb_flip_128_128/5/train_den'
        path['roi'] = None
        self.data_path['ucf0RandFlip_128_128_train5'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/1/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/1/train_den'
        path['roi'] = None
        self.data_path['ucf1_train1'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/2/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/2/train_den'
        path['roi'] = None
        self.data_path['ucf1_train2'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/3/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/3/train_den'
        path['roi'] = None
        self.data_path['ucf1_train3'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/4/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/4/train_den'
        path['roi'] = None
        self.data_path['ucf1_train4'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/5/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/5/train_den'
        path['roi'] = None
        self.data_path['ucf1_train5'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/1/val'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/1/val_den'
        path['roi'] = None
        self.data_path['ucf1_test1'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/2/val'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/2/val_den'
        path['roi'] = None
        self.data_path['ucf1_test2'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/3/val'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/3/val_den'
        path['roi'] = None
        self.data_path['ucf1_test3'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/4/val'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/4/val_den'
        path['roi'] = None
        self.data_path['ucf1_test4'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/5/val'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/5/val_den'
        path['roi'] = None
        self.data_path['ucf1_test5'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/1/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/1/train_den'
        path['roi'] = None
        self.data_path['ucf9RandFlip_train1'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/2/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/2/train_den'
        path['roi'] = None
        self.data_path['ucf9RandFlip_train2'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/3/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/3/train_den'
        path['roi'] = None
        self.data_path['ucf9RandFlip_train3'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/4/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/4/train_den'
        path['roi'] = None
        self.data_path['ucf9RandFlip_train4'] = path

        path = dict()
        path['image'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/5/train'
        path['gt'] = r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/5/train_den'
        path['roi'] = None
        self.data_path['ucf9RandFlip_train5'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train_den'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train_roi'
        self.data_path['we1Flip_train'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_den'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_roi'
        self.data_path['we1_train'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/1'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/1'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all'
        self.data_path['we1_test1'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/2'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/2'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all'
        self.data_path['we1_test2'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/3'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/3'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all'
        self.data_path['we1_test3'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/4'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/4'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all'
        self.data_path['we1_test4'] = path

        path = dict()
        path['image'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/5'
        path['gt'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/5'
        path['roi'] = r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all'
        self.data_path['we1_test5'] = path

        path = dict()
        path['image'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn'
        path['gt'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn_den'
        path['roi'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn_roi'
        self.data_path['tran1FlipResize1_trainAllValiFlip'] = path

        path = dict()
        path['image'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val'
        path['gt'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val_den'
        path['roi'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val_roi'
        self.data_path['tran1Resize1_Vali'] = path

        path = dict()
        path['image'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test'
        path['gt'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test_den'
        path['roi'] = r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test_roi'
        self.data_path['tran1Resize1_test'] = path

        path = dict()
        path['image'] = r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train'
        path['gt'] = r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train_den'
        path['roi'] = r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train_roi'
        self.data_path['mall1Resize05_train'] = path

        path = dict()
        path['image'] = r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val'
        path['gt'] = r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val_den'
        path['roi'] = r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val_roi'
        self.data_path['mall1Resize05_val'] = path

        path = dict()
        path['image'] = r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train'
        path['gt'] = r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_den'
        path['roi'] = r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_roi'
        self.data_path['air1_train'] = path

        path = dict()
        path['image'] = r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test'
        path['gt'] = r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_den'
        path['roi'] = r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_roi'
        self.data_path['air1_test'] = path

        path = dict()
        path['image'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_0_random_flip_rgb_128_128_resize1024/train'
        path['gt'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_0_random_flip_rgb_128_128_resize1024/train_den'
        path['roi'] = None
        self.data_path['ucfQnrf0RandFlip_128_128_resize1024_train'] = path

        path = dict()
        path['image'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_0_random_flip_rgb_128_256_more1_resize1024/train'
        path['gt'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_0_random_flip_rgb_128_256_more1_resize1024/train_den'
        path['roi'] = None
        self.data_path['ucfQnrf0RandFlip_128_256_more1Resize1024_train'] = path

        path = dict()
        path['image'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/train'
        path['gt'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/train_den'
        path['roi'] = None
        self.data_path['ucfQnrf1Resize1024_train'] = path

        path = dict()
        path['image'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/test'
        path['gt'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/test_den'
        path['roi'] = None
        self.data_path['ucfQnrf1Resize1024_test'] = path

        path = dict()
        path['image'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_9_random_flip_rgb_resize1024/train'
        path['gt'] = r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_9_random_flip_rgb_resize1024/train_den'
        path['roi'] = None
        self.data_path['ucfQnrf9RandFlipResize1024_train'] = path

    def get_path(self, name):
        data_path_dict = self.data_path[name]
        abs_path_dict = dict()

        is_dir = False

        for base_path in self.base_path_list:
            for key in data_path_dict:
                if data_path_dict[key] is not None:
                    this_abs_path = os.path.join(base_path, data_path_dict[key])
                    if os.path.isdir(this_abs_path):
                        is_dir = True
                        abs_path_dict[key] = this_abs_path
                    else:
                        break
                else:
                    abs_path_dict[key] = None

            if is_dir:
                break

        for key in data_path_dict:
            if not key in abs_path_dict:
                raise Exception('invalid key in absolute data path dict')

        return abs_path_dict
