clc; clear all;
% seed = 95461354;
% rng(seed);

% number of patches
% set to 0 to get adaptive number of patches
number_of_patches = 0;

% size of patches
height_of_patches = 128;
width_of_patches = 128;

% resize image and annotation to fit max_height_width
is_resize = false;
max_height_width = 1024;

%if true, will get gray images(one channel)
%if false, will get RGB images(three channel)
is_gray_image = false;

%if true, will apply ROI to every image
%if true, must give a ROI(roi.mask)
is_use_roi = false;

%if true, will save roi info for every image
%not support N=16 now
%if true, must give a ROI(roi.mask)
%if true, is_use_roi must be true
is_save_roi = false;

%if true, will get a smallest patch outside ROI of every image. then get patches from this patch
%if true, must give a ROI(roi.mask)
%if true, is_use_roi must be true
is_circumscribed_rectangle = false;

% try to get patches in ROI
is_all_in_roi = false;

% try to get patches which have more than some pedestrians in each of them
is_more_than_pedestrian = true;
pedstrian_number = 1;

% try to get patches which have zero pedestrians and more than one pedestrian in each of them
is_forbid_zero_to_one_pedestrian = false;

%if true, will get original patchs and overturned patchs
is_overturn = true;

% get fewer patches when number_of_patches == 0
is_fewer_samples = false;

is_shanghaitech = false;
is_ucf_cc_50 = false;
is_worldexpo = false;
is_airport = false;
is_ucf_qnrf = false;
is_gcc = true;
if is_shanghaitech
    dataset = 'B';
    dataset_name = ['shanghaitech_part_' dataset '_patches_' num2str(number_of_patches)];
    output_path = ['D:\Dataset\ShanghaiTech\formatted_trainval\'];
    img_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\train_data\images\'];
    gt_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\train_data\ground_truth\'];
    train_path_img = strcat(output_path, dataset_name,'\train\');
    train_path_den = strcat(output_path, dataset_name,'\train_den\');
%     img_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\test_data\images\'];
%     gt_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\test_data\ground_truth\'];
%     train_path_img = strcat(output_path, dataset_name,'\test\');
%     train_path_den = strcat(output_path, dataset_name,'\test_den\');
elseif is_ucf_cc_50
    dataset_name = ['ucf_cc_50_patches_' num2str(number_of_patches)];
    img_path = 'D:\Dataset\UCF_CC_50\original\image\';
    gt_path = 'D:\Dataset\UCF_CC_50\original\gt\';
    output_path = 'D:\Dataset\UCF_CC_50\formatted_trainval\';
    train_path_img = strcat(output_path, dataset_name,'\train\');
    train_path_den = strcat(output_path, dataset_name,'\train_den\');
elseif is_worldexpo
    dataset_name = ['worldexpo_patches_' num2str(number_of_patches)];
    output_path = 'D:\Dataset\WorldExpo10\formatted_trainval\';
    img_path = 'D:\Dataset\WorldExpo10\train_frame\';
    gt_path = 'D:\Dataset\WorldExpo10\train_label\';
    train_path_img = strcat(output_path, dataset_name,'\train\');
    train_path_den = strcat(output_path, dataset_name,'\train_den\');
    train_path_roi = strcat(output_path, dataset_name,'\train_roi\');
%     img_path = 'D:\Dataset\WorldExpo10\test_frame\';
%     gt_path = 'D:\Dataset\WorldExpo10\test_label\';
%     train_path_img = strcat(output_path, dataset_name,'\test\');
%     train_path_den = strcat(output_path, dataset_name,'\test_den\');
%     train_path_roi = strcat(output_path, dataset_name,'\test_roi\');
elseif is_airport
    dataset_name = ['airport_patches_' num2str(number_of_patches)];
    output_path = 'D:\Dataset\airport\formatted_trainval\';
    img_path = 'D:\Dataset\airport\img\';
    gt_path = 'D:\Dataset\airport\gt\';
    roi_root_path = 'D:\Dataset\airport\roi\';
    train_path_img = strcat(output_path, dataset_name,'\train\');
    train_path_den = strcat(output_path, dataset_name,'\train_den\');
    train_path_roi = strcat(output_path, dataset_name,'\train_roi\');
elseif is_ucf_qnrf
    dataset_name = ['ucf_qnrf_patches_' num2str(number_of_patches)];
    img_path = 'D:\Dataset\UCF-QNRF\original\Train\';
    gt_path = 'D:\Dataset\UCF-QNRF\original\Train\';
%     img_path = 'D:\Dataset\UCF-QNRF\original\Test\';
%     gt_path = 'D:\Dataset\UCF-QNRF\original\Test\';
    output_path = 'D:\Dataset\UCF-QNRF\kernel\';
    train_path_img = strcat(output_path, dataset_name,'\train\');
    train_path_den = strcat(output_path, dataset_name,'\train_den\');
elseif is_gcc
    dataset_name = ['gcc_patches_' num2str(number_of_patches)];
    output_path = ['D:\Dataset\GCC\all\formatted_trainval\'];
    img_path = ['D:\Dataset\GCC\all\train\jpgs\'];
    gt_path = ['D:\Dataset\GCC\all\train\mats\'];
    train_path_img = strcat(output_path, dataset_name,'\train\');
    train_path_den = strcat(output_path, dataset_name,'\train_den\');
end

mkdir(output_path);
mkdir(train_path_img);
mkdir(train_path_den);
if is_save_roi
    mkdir(train_path_roi);
end

dir_output = dir(fullfile(img_path,'*.jpg'));
img_name_list = {dir_output.name};
if is_use_roi
    roi_container = containers.Map;
    roi_fourth_size_container = containers.Map;
    roi_eighth_size_container = containers.Map;
end

num_images = numel(img_name_list);
for index = 1:num_images
    [~, img_name, ~] = fileparts(img_name_list{index});
    if is_worldexpo
        scene_number = img_name(1:6);
    end
    if is_airport
        scene_number = img_name(1:2);
    end
    
    if (mod(index, 10)==0)
        fprintf(1,'Processing %3d/%d files\n', index, num_images);
    end
    
    if is_shanghaitech
        load(strcat(gt_path, 'GT_', img_name, '.mat')) ;
        input_img_name = strcat(img_path, img_name, '.jpg');
    elseif is_ucf_cc_50
        load(strcat(gt_path, img_name, '_ann.mat')) ;
        input_img_name = strcat(img_path, img_name, '.jpg');
    elseif is_worldexpo
        load(strcat(gt_path, scene_number, '\', img_name, '.mat'));
        input_img_name = strcat(img_path, img_name_list{index});
        if is_use_roi
            roi_path = strcat(gt_path, scene_number, '\roi.mat');
        end
    elseif is_airport
        load(strcat(gt_path, img_name, '.mat'));
        input_img_name = strcat(img_path, img_name_list{index});
        if is_use_roi
            roi_path = strcat(roi_root_path, 'roi-', scene_number, '.mat');
        end
    elseif is_ucf_qnrf
        load(strcat(gt_path, img_name, '_ann.mat')) ;
        input_img_name = strcat(img_path, img_name, '.jpg');
    elseif is_gcc
        load(strcat(gt_path, img_name, '.mat')) ;
        input_img_name = strcat(img_path, img_name, '.jpg');
    end
    
    im = imread(input_img_name);
    
    [height, width, channel] = size(im);
    if is_gray_image
        if (channel == 3)
            im = rgb2gray(im);
        elseif (channel == 1)
            im = im;
        end
    else
        if (channel == 3)
            im = im;
        elseif (channel == 1)
            im_original = im;
            im = uint8(zeros(height, width, 3));
            im(:, :, 1) = im_original;
            im(:, :, 2) = im_original;
            im(:, :, 3) = im_original;
        end
    end
    
    if is_shanghaitech
        annPoints = image_info{1}.location;
    elseif is_ucf_cc_50
        % nothing need to do here
    elseif is_worldexpo
        annPoints = point_position;
    elseif is_airport
        annPoints = image_info.location;
    elseif is_ucf_qnrf
        % nothing need to do here
    elseif is_gcc
        annPoints = image_info.location;
    end
    
    if is_resize
        [height, width, ~] = size(im);
        if height > width
            resized_height = max_height_width;
            resized_width = round(width / height * resized_height);
        else
            resized_width = max_height_width;
            resized_height = round(height / width * resized_width);
        end
        im = imresize(im, [resized_height, resized_width], 'bilinear');
        annPoints(:, 1) = annPoints(:, 1) / width * resized_width;
        annPoints(:, 2) = annPoints(:, 2) / height * resized_height;
        if is_use_roi
            error('roi is not supported');
        end
    end
    
%     im_density_1 = get_density_map_gaussian(im, annPoints, 9, 4);
%     im_density_2 = get_density_map_gaussian(im, annPoints, 15, 4);
%     im_density_3 = get_density_map_gaussian(im, annPoints, 21, 4);
%     im_density = (im_density_1 + im_density_2 + im_density_3) / 3.0;

%     im_density_1 = get_density_map_gaussian(im, annPoints, 9, 4);
%     im_density_2 = get_density_map_gaussian(im, annPoints, 15, 4);
%     im_density = (im_density_1 + im_density_2) / 2.0;

    im_density = get_density_map_gaussian(im, annPoints, 15, 4);
    
%     view_density_map(im_density);
    
%     fid=fopen('D:\count.txt','a');
%     fprintf(fid, '%s\t%f\n', img_name, sum(im_density(:)));
%     fclose(fid);
    
    if is_worldexpo
        if is_use_roi
            if roi_container.isKey(scene_number)
                roi_original = roi_container(scene_number);
                roi_fourth_size = roi_fourth_size_container(scene_number);
                roi_eighth_size = roi_eighth_size_container(scene_number);
            else
                roi_original = get_mask_map_worldexpo(roi_path);
                roi_container(scene_number) = roi_original;
                roi_fourth_size = get_mask_map_worldexpo(roi_path, 'fourth');
                roi_fourth_size_container(scene_number) = roi_fourth_size;
                roi_eighth_size = get_mask_map_worldexpo(roi_path, 'eighth');
                roi_eighth_size_container(scene_number) = roi_eighth_size;
            end
            im_density = im_density .* roi_original.mask;
        end
    end
    
    if is_airport
        if is_use_roi
            if roi_container.isKey(scene_number)
                roi_original = roi_container(scene_number);
                roi_fourth_size = roi_fourth_size_container(scene_number);
                roi_eighth_size = roi_eighth_size_container(scene_number);
            else
                roi_original = get_mask_map_airport(roi_path);
                roi_container(scene_number) = roi_original;
                roi_fourth_size = get_mask_map_airport(roi_path, 'fourth');
                roi_fourth_size_container(scene_number) = roi_fourth_size;
                roi_eighth_size = get_mask_map_airport(roi_path, 'eighth');
                roi_eighth_size_container(scene_number) = roi_eighth_size;
            end
            im_density = im_density .* roi_original.matrix;
        end
    end
    
    if is_circumscribed_rectangle
        [top, bottom, left, right] = get_circumscribed_rectangle_roi(roi_original.mask(:,:));
        rectangle_height = 8 * floor((bottom - top) / 8);
        rectangle_width = 8 * floor((right - left) / 8);
        bottom = top + rectangle_height - 1;
        right = left + rectangle_width - 1;
        top_fourth_size = max(round(top / 4), 1);
        left_fourth_size = max(round(left / 4), 1);
        bottom_fourth_size = top_fourth_size + rectangle_height / 4 - 1;
        right_fourth_size = left_fourth_size + rectangle_width / 4 - 1;
        top_eighth_size = max(round(top / 8), 1);
        left_eighth_size = max(round(left / 8), 1);
        bottom_eighth_size = top_eighth_size + rectangle_height / 8 - 1;
        right_eighth_size = left_eighth_size + rectangle_width / 8 - 1;
        
        if is_gray_image
            im = im(top:bottom, left:right);
        else
            im = im(top:bottom, left:right, :);
        end
        im_density = im_density(top:bottom, left:right);
        roi_original.mask = roi_original.mask(top:bottom, left:right);
        roi_fourth_size.mask = roi_fourth_size.mask(top_fourth_size:bottom_fourth_size, left_fourth_size:right_fourth_size);
        roi_eighth_size.mask = roi_eighth_size.mask(top_eighth_size:bottom_eighth_size, left_eighth_size:right_eighth_size);
        roi_original.matrix = roi_original.matrix(top:bottom, left:right);
        roi_fourth_size.matrix = roi_fourth_size.matrix(top_fourth_size:bottom_fourth_size, left_fourth_size:right_fourth_size);
        roi_eighth_size.matrix = roi_eighth_size.matrix(top_eighth_size:bottom_eighth_size, left_eighth_size:right_eighth_size);
    end
    
    [height, width, ~] = size(im);
    half_width_of_patches = floor(width_of_patches / 2);
    half_height_of_patches = floor(height_of_patches / 2);
    half_width_of_patches_fourth_size = half_width_of_patches / 4;
    half_height_of_patches_fourth_size = half_height_of_patches / 4;
    half_width_of_patches_eighth_size = half_width_of_patches / 8;
    half_height_of_patches_eighth_size = half_height_of_patches / 8;
    
    start_width = half_width_of_patches + 1;
    end_width = width - half_width_of_patches;
    start_height = half_height_of_patches + 1;
    end_height = height - half_height_of_patches;
    
    if number_of_patches == 0
        if is_fewer_samples
            N = ceil(height * width / height_of_patches / width_of_patches);
        else
            N = 4 * ceil(height * width / height_of_patches / width_of_patches);
        end
    else
        N = number_of_patches;
    end
    
    i = 0;
    patience = 0;
    while(i <= N)
        x_position = rand;
        y_position = rand;
        
        x = floor((end_width - start_width) * x_position + start_width);
        y = floor((end_height - start_height) * y_position + start_height);
        x1 = x - half_width_of_patches;
        y1 = y - half_height_of_patches;
        x2 = x + half_width_of_patches-1;
        y2 = y + half_height_of_patches-1;
        
        x_fourth_size = max(floor(x / 4), 1) + 1;
        y_fourth_size = max(floor(y / 4), 1) + 1;
        x1_fourth_size = x_fourth_size - half_width_of_patches_fourth_size;
        y1_fourth_size = y_fourth_size - half_height_of_patches_fourth_size;
        x2_fourth_size = x_fourth_size + half_width_of_patches_fourth_size - 1;
        y2_fourth_size = y_fourth_size + half_height_of_patches_fourth_size - 1;
        
        x_eighth_size = max(floor(x / 8), 1) + 1;
        y_eighth_size = max(floor(y / 8), 1) + 1;
        x1_eighth_size = x_eighth_size - half_width_of_patches_eighth_size;
        y1_eighth_size = y_eighth_size - half_height_of_patches_eighth_size;
        x2_eighth_size = x_eighth_size + half_width_of_patches_eighth_size - 1;
        y2_eighth_size = y_eighth_size + half_height_of_patches_eighth_size - 1;
        
        if is_gray_image
            im_sampled = im(y1:y2, x1:x2);
        else
            im_sampled = im(y1:y2, x1:x2, :);
        end
        im_density_sampled = im_density(y1:y2, x1:x2);
        if is_use_roi
            roi_sampled.mask = roi_original.mask(y1:y2,x1:x2);
            roi_fourth_size_sampled.mask = roi_fourth_size.mask(y1_fourth_size:y2_fourth_size, x1_fourth_size:x2_fourth_size);
            roi_eighth_size_sampled.mask = roi_eighth_size.mask(y1_eighth_size:y2_eighth_size, x1_eighth_size:x2_eighth_size);
            roi_sampled.matrix = roi_original.matrix(y1:y2,x1:x2);
            roi_fourth_size_sampled.matrix = roi_fourth_size.matrix(y1_fourth_size:y2_fourth_size, x1_fourth_size:x2_fourth_size);
            roi_eighth_size_sampled.matrix = roi_eighth_size.matrix(y1_eighth_size:y2_eighth_size, x1_eighth_size:x2_eighth_size);
        end
        
        if is_all_in_roi && sum(roi_sampled.matrix(:)) ~= width_of_patches * height_of_patches
            patience = patience + 1;
            if patience < N
                continue
            end
        elseif is_more_than_pedestrian && sum(im_density_sampled(:)) < pedstrian_number
            patience = patience + 1;
            if patience < N
                continue
            end
        elseif is_forbid_zero_to_one_pedestrian && sum(im_density_sampled(:)) > 0.0 && sum(im_density_sampled(:)) < 1.0
            patience = patience + 1;
            if patience < N
                continue
            end
        end
        patience = 0;
        i = i + 1;
        
        save_name = strcat(img_name, '_', num2str(i));
        
        imwrite(im_sampled, [train_path_img save_name '.jpg']);
        csvwrite([train_path_den save_name '.csv'], im_density_sampled);
        if is_save_roi
            roi = roi_sampled;
            save([train_path_roi save_name '_roi.mat'], 'roi');
            roi = roi_fourth_size_sampled;
            save([train_path_roi save_name '_roi_fourth_size.mat'], 'roi');
            roi = roi_eighth_size_sampled;
            save([train_path_roi save_name '_roi_eighth_size.mat'], 'roi');
        end
        
        if is_overturn
            im_sampled_overturn = fliplr(im_sampled);
            im_density_sampled_overturn = fliplr(im_density_sampled);
            if is_use_roi
                roi_sampled_overturn.mask = fliplr(roi_sampled.mask);
                roi_fourth_size_sampled_overturn.mask = fliplr(roi_fourth_size_sampled.mask);
                roi_eighth_size_sampled_overturn.mask = fliplr(roi_eighth_size_sampled.mask);
                roi_sampled_overturn.matrix = fliplr(roi_sampled.matrix);
                roi_fourth_size_sampled_overturn.matrix = fliplr(roi_fourth_size_sampled.matrix);
                roi_eighth_size_sampled_overturn.matrix = fliplr(roi_eighth_size_sampled.matrix);
            end
            imwrite(im_sampled_overturn, [train_path_img save_name '_overturn.jpg']);
            csvwrite([train_path_den save_name '_overturn.csv'], im_density_sampled_overturn);
            if is_save_roi
                roi = roi_sampled_overturn;
                save([train_path_roi save_name '_overturn_roi.mat'], 'roi');
                roi = roi_fourth_size_sampled_overturn;
                save([train_path_roi save_name '_overturn_roi_fourth_size.mat'], 'roi');
                roi = roi_eighth_size_sampled_overturn;
                save([train_path_roi save_name '_overturn_roi_eighth_size.mat'], 'roi');
            end
        end
    end
end

