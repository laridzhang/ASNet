clc; clear all;
seed = 95461354;
rng(seed);

% size of patchs will be number times TIMES
% at least 4
% default 16
TIMES = 16;

%if true, will get gray images(one channel)
%if false, will get RGB images(three channel)
is_gray_image = false;

is_random_position = true;% default true

% if true, will get four patches which are fixed position out of nine
is_fixed_position = false;

%if true, will get a smallest patch outside ROI of every image
%if true, N must be 1
%if true, must give a ROI(roi.mask)
%if true, is_inscribed_rectangle must be false
%if true, is_half_bottom must be false
%if true, is_use_roi must be true
is_circumscribed_rectangle = false;

%if true, will get original patchs and overturned patchs
is_overturn = true;

%if true, will apply ROI to every image
%if true, must give a ROI(roi.mask)
is_use_roi = false;

%if true, will save roi info for every image
%not support N=16 now
%if true, must give a ROI(roi.mask)
%if true, is_use_roi must be true
is_save_roi = false;

% try to get patches in ROI
is_all_in_roi = false;

% try to get patches which have more than one pedestrain in each of them
is_more_than_one_pedestrain = false;

% resize image and annotation to fit max_height_width
is_resize_to_max_height_width = false;
max_height_width = 1024;

is_shanghaitech = false;
is_ucf_cc_50 = false;
is_ucsd = false;
is_worldexpo = false;
is_airport = false;
is_ucf_qnrf = false;
is_gcc = true;
if is_shanghaitech
    dataset = 'B';
    dataset_name = ['shanghaitech_part_' dataset '_patches_9'];
    output_path = ['D:\Dataset\ShanghaiTech\formatted_trainval\'];
    img_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\train_data\images\'];
    gt_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\train_data\ground_truth\'];
%     img_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\test_data\images\'];
%     gt_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\test_data\ground_truth\'];
elseif is_ucf_cc_50
    dataset_name = ['ucf_cc_50_patches_9'];
    img_path = 'D:\Dataset\UCF_CC_50\original\image\';
    gt_path = 'D:\Dataset\UCF_CC_50\original\gt\';
    output_path = 'D:\Dataset\UCF_CC_50\formatted_trainval\';
elseif is_worldexpo
    dataset_name = ['worldexpo_patches_9'];
    output_path = 'D:\Dataset\WorldExpo10\formatted_trainval\';
    img_path = 'D:\Dataset\WorldExpo10\train_frame\';
    gt_path = 'D:\Dataset\WorldExpo10\train_label\';
%     img_path = 'D:\Dataset\WorldExpo10\test_frame\';
%     gt_path = 'D:\Dataset\WorldExpo10\test_label\';
elseif is_airport
    dataset_name = ['airport_patches_9'];
    output_path = 'D:\Dataset\airport\formatted_trainval\';
    img_path = 'D:\Dataset\airport\img\';
    gt_path = 'D:\Dataset\airport\gt\';
    roi_root_path = 'D:\Dataset\airport\roi\';
elseif is_ucf_qnrf
    dataset_name = ['ucf_qnrf_patches_1'];
    output_path = 'D:\Dataset\UCF-QNRF\kernel\';
    img_path = 'D:\Dataset\UCF-QNRF\original\Train\';
    gt_path = 'D:\Dataset\UCF-QNRF\original\Train\';
%     img_path = 'D:\Dataset\UCF-QNRF\original\Test\';
%     gt_path = 'D:\Dataset\UCF-QNRF\original\Test\';
elseif is_gcc
    dataset_name = 'gcc_patches_9';
    output_path = ['D:\Dataset\GCC\ours\formatted_trainval\'];
    img_path = ['D:\Dataset\GCC\ours\original\train\jpgs\'];
    gt_path = ['D:\Dataset\GCC\ours\original\train\mats\'];
end

output_path_img = strcat(output_path, dataset_name,'\img\');
output_path_den = strcat(output_path, dataset_name,'\den\');
if is_save_roi
    output_path_roi = strcat(output_path, dataset_name,'\roi\');
end

mkdir(output_path);
mkdir(output_path_img);
mkdir(output_path_den);
if is_save_roi
    mkdir(output_path_roi);
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
        load(strcat(gt_path, 'GT_', img_name, '.mat'));
        im = imread(strcat(img_path, img_name, '.jpg'));
    elseif is_ucf_cc_50
        load(strcat(gt_path, img_name, '_ann.mat')) ;
        im = imread(strcat(img_path, img_name, '.jpg'));
    elseif is_worldexpo
        load(strcat(gt_path, scene_number, '\', img_name, '.mat'));
        im = imread(strcat(img_path, img_name, '.jpg'));
        if is_use_roi
            roi_path = strcat(gt_path, scene_number, '\roi.mat');
        end
    elseif is_airport
        load(strcat(gt_path, img_name, '.mat'));
        im = imread(strcat(img_path, img_name, '.jpg'));
        if is_use_roi
            roi_path = strcat(roi_root_path, 'roi-', scene_number, '.mat');
        end
    elseif is_ucf_qnrf
        load(strcat(gt_path, img_name, '_ann.mat')) ;
        im = imread(strcat(img_path, img_name, '.jpg'));
    elseif is_gcc
        load(strcat(gt_path, img_name, '.mat')) ;
        im = imread(strcat(img_path, img_name, '.jpg'));
    end
    
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
    elseif is_gcc
        annPoints = image_info.location;
    end
    
    if is_resize_to_max_height_width
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
                roi_original = get_mask_map_airport(roi_path, TIMES);
                roi_container(scene_number) = roi_original;
                roi_fourth_size = get_mask_map_airport(roi_path, TIMES, 'fourth');
                roi_fourth_size_container(scene_number) = roi_fourth_size;
                roi_eighth_size = get_mask_map_airport(roi_path, TIMES, 'eighth');
                roi_eighth_size_container(scene_number) = roi_eighth_size;
            end
            im_density = im_density .* roi_original.matrix;
        end
    end
    
    if is_circumscribed_rectangle
        [top, bottom, left, right] = get_circumscribed_rectangle_roi(roi_original.mask(:,:));
        if is_offset
            offset_height = min(top, height - bottom) * (rand - 0.5) * 2;
            offset_width = min(left, width - right) * (rand - 0.5) * 2;
            top = max(floor(top + offset_height), 1);
            bottom = min(floor(bottom + offset_height), height);
            left = max(floor(left + offset_width), 1);
            right = min(floor(right + offset_width), width);
        end
        rectangle_height = TIMES * floor((bottom - top) / TIMES);
        rectangle_width = TIMES * floor((right - left) / TIMES);
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
    half_width_of_patches = floor(width/4);
    half_height_of_patches = floor(height/4);
    half_width_of_patches = TIMES * floor(half_width_of_patches / TIMES);
    half_height_of_patches = TIMES * floor(half_height_of_patches / TIMES);
    half_width_of_patches_fourth_size = half_width_of_patches / 4;
    half_height_of_patches_fourth_size = half_height_of_patches / 4;
    half_width_of_patches_eighth_size = half_width_of_patches / 8;
    half_height_of_patches_eighth_size = half_height_of_patches / 8;
    
    a_width = half_width_of_patches+1;
    b_width = width - half_width_of_patches;
    a_height = half_height_of_patches+1;
    b_height = height - half_height_of_patches;
    
    j = 0;
    patience = 0;
    while(j < 9)
        x_position = -1;
        y_position = -1;
        
        if is_fixed_position && j < 4
            position = [0 1];
            x_position = position(mod(j, 2) + 1);
            y_position = position(mod(j, 2) + 1);
        else
            if is_random_position
                x_position = rand;
                y_position = rand;
            else
                position = [0 0.5 1];
                x_position = position(mod(j, 3) + 1);
                y_position = position(ceil(j / 3));
            end
        end
        
        x = floor((b_width - a_width) * x_position + a_width);
        y = floor((b_height - a_height) * y_position + a_height);
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
        
        if is_fixed_position && j < 4
            % nothing to do here
        else
            if is_all_in_roi && sum(roi_sampled.matrix(:)) ~= (y2 - y1 + 1) * (x2 - x1 + 1)
                patience = patience + 1;
                if patience < 81
                    continue
                end
            elseif is_more_than_one_pedestrain && sum(im_density_sampled(:)) < 1.0
                patience = patience + 1;
                if patience < 81
                    continue
                end
            end
        end
        patience = 0;
        j = j + 1;
        
        if is_worldexpo || is_airport
            img_idx = strcat(scene_number, '_', num2str(index), '_',num2str(j));
        else
            img_idx = strcat(num2str(index), '_',num2str(j));
        end
        
        imwrite(im_sampled, [output_path_img img_name '_' num2str(j) '.jpg']);
        csvwrite([output_path_den img_name '_' num2str(j) '.csv'], im_density_sampled);
        if is_save_roi
            roi = roi_sampled;
            save([output_path_roi img_name '_' num2str(j) '_roi.mat'], 'roi');
            roi = roi_fourth_size_sampled;
            save([output_path_roi img_name '_' num2str(j) '_roi_fourth_size.mat'], 'roi');
            roi = roi_eighth_size_sampled;
            save([output_path_roi img_name '_' num2str(j) '_roi_eighth_size.mat'], 'roi');
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
            imwrite(im_sampled_overturn, [output_path_img img_name '_' num2str(j) '_overturn.jpg']);
            csvwrite([output_path_den img_name '_' num2str(j) '_overturn.csv'], im_density_sampled_overturn);
            if is_save_roi
                roi = roi_sampled_overturn;
                save([output_path_roi img_name '_' num2str(j) '_overturn_roi.mat'], 'roi');
                roi = roi_fourth_size_sampled_overturn;
                save([output_path_roi img_name '_' num2str(j) '_overturn_roi_fourth_size.mat'], 'roi');
                roi = roi_eighth_size_sampled_overturn;
                save([output_path_roi img_name '_' num2str(j) '_overturn_roi_eighth_size.mat'], 'roi');
            end
        end
    end
end
