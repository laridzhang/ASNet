% create one patch from every image
clc; clear all;
seed = 95461354;
rng(seed);

% default 16
TIMES = 16;

%if true, will get gray images(one channel)
%if false, will get RGB images(three channel)
is_gray_image = false;

%if true, will apply ROI to every image
%if true, must give a ROI(roi.mask)
is_use_roi = false;

%if true, will save roi info for every image
%if true, must give a ROI(roi.mask)
%if true, is_use_roi must be true
is_save_roi = false;

%if true, is_use_roi must be true
is_apply_roi_to_image = false;

%if true, will get a smallest patch outside ROI of every image. then get patches from this patch
%if true, must give a ROI(roi.mask)
%if true, is_use_roi must be true
is_circumscribed_rectangle = false;

%if true, will get original patchs and overturned patchs
is_overturn = false;

%if true, will resize image and density map and roi to fit [TIMES]
%if false, will crop image and density map and roi to fit [TIMES]
%default true
is_resize = false;

%this has nothing to do with [is_resize]
%use this to resize image and density map and roi
%use for ucsd
%default 1.0
resize_level = 1.0;

% resize image and annotation to fit max_height_width
is_resize_to_max_height_width = false;
max_height_width = 1024;

is_shanghaitech = false;
is_ucf_cc_50 = false;
is_worldexpo = false;
is_airport = false;
is_ucsd = false;
is_trancos = false;
is_mall = false;
is_ucf_qnrf = false;
is_gcc = true;
if is_shanghaitech
    dataset = 'B';
    dataset_name = ['shanghaitech_part_' dataset '_patches_1'];
    root_save_path = ['D:\Dataset\ShanghaiTech\formatted_trainval\'];
%     img_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\train_data\images\'];
%     gt_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\train_data\ground_truth\'];
    img_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\test_data\images\'];
    gt_path = ['D:\Dataset\ShanghaiTech\original\shanghaitech\part_' dataset '_final\test_data\ground_truth\'];
elseif is_ucf_cc_50
    dataset_name = 'ucf_cc_50_patches_1';
    img_path = 'D:\Dataset\UCF_CC_50\original\image\';
    gt_path = 'D:\Dataset\UCF_CC_50\original\gt\';
    root_save_path = 'D:\Dataset\UCF_CC_50\formatted_trainval\';
elseif is_worldexpo
    dataset_name = 'worldexpo_patches_1';
    root_save_path = 'D:\Dataset\WorldExpo10\formatted_trainval\';
    img_path = 'D:\Dataset\WorldExpo10\original\train_frame\';
    gt_path = 'D:\Dataset\WorldExpo10\original\train_label\';
    roi_save_path = strcat(root_save_path, dataset_name,'\roi\');
%     img_path = 'D:\Dataset\WorldExpo10\original\test_frame\';
%     gt_path = 'D:\Dataset\WorldExpo10\original\test_label\';
%     roi_save_path = strcat(root_save_path, dataset_name,'\roi\');
elseif is_airport
    dataset_name = 'airport_patches_1';
    root_save_path = 'D:\Dataset\airport\formatted_trainval\';
    img_path = 'D:\Dataset\airport\img\';
    gt_path = 'D:\Dataset\airport\gt\';
    roi_root_path = 'D:\Dataset\airport\roi\';
    roi_save_path = strcat(root_save_path, dataset_name,'\roi\');
elseif is_ucsd
    dataset_name = 'ucsd_patches_1';
    root_save_path = 'D:\Dataset\UCSD\formatted_trainval\';
    img_path = 'D:\Dataset\UCSD\original\img\';
    gt_path = 'D:\Dataset\UCSD\original\gt\';
    roi_root_path = 'D:\Dataset\UCSD\original\roi\';
    roi_save_path = strcat(root_save_path, dataset_name,'\roi\');
elseif is_trancos
    dataset_name = 'trancos_patches_1';
    root_save_path = 'D:\Dataset\TRANCOS\original\formatted_trainval\';
    img_path = 'D:\Dataset\TRANCOS\original\images\';
    gt_path = 'D:\Dataset\TRANCOS\original\images\';
    roi_root_path = 'D:\Dataset\TRANCOS\original\images\';
    roi_save_path = strcat(root_save_path, dataset_name,'\roi\');
elseif is_mall
    dataset_name = 'mall_patches_1';
    root_save_path = 'D:\Dataset\mall\formatted_trainval\';
    img_path = 'D:\Dataset\mall\original\frames\';
    gt_path = 'D:\Dataset\mall\original\';
    roi_root_path = 'D:\Dataset\mall\original\';
    roi_save_path = strcat(root_save_path, dataset_name,'\roi\');
elseif is_ucf_qnrf
    dataset_name = ['ucf_qnrf_patches_1'];
    root_save_path = 'D:\Dataset\UCF-QNRF\kernel\';
%     img_path = 'D:\Dataset\UCF-QNRF\original\Train\';
%     gt_path = 'D:\Dataset\UCF-QNRF\original\Train\';
    img_path = 'D:\Dataset\UCF-QNRF\original\Test\';
    gt_path = 'D:\Dataset\UCF-QNRF\original\Test\';
elseif is_gcc
    dataset_name = 'gcc_patches_1';
    root_save_path = ['D:\Dataset\GCC\ours\formatted_trainval\'];
%     img_path = ['D:\Dataset\GCC\ours\original\train\jpgs\'];
%     gt_path = ['D:\Dataset\GCC\ours\original\train\mats\'];
    img_path = ['D:\Dataset\GCC\ours\original\val\jpgs\'];
    gt_path = ['D:\Dataset\GCC\ours\original\val\mats\'];
end
img_save_path = strcat(root_save_path, dataset_name,'\img\');
den_save_path = strcat(root_save_path, dataset_name,'\den\');

mkdir(root_save_path);
mkdir(img_save_path);
mkdir(den_save_path);
if is_save_roi
    mkdir(roi_save_path);
end

if is_ucsd
    dir_output = dir(fullfile(img_path,'*.png'));
else
    dir_output = dir(fullfile(img_path,'*.jpg'));
end
img_name_list = {dir_output.name};

if is_use_roi
    roi_container = containers.Map;
    roi_fourth_size_container = containers.Map;
    roi_eighth_size_container = containers.Map;
end

num_images = numel(img_name_list);
for idx = 1:num_images
    [~, img_name, ~] = fileparts(img_name_list{idx});
    
    if is_worldexpo
        scene_number = img_name(1:6);
    end
    if is_airport
        scene_number = img_name(1:2);
    end
    if is_ucsd
        scene_number = img_name(1:12);
    end
    if is_mall
        scene_number = img_name(5:10);
    end
    
    if (mod(idx, 10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    
    if is_shanghaitech
        load(strcat(gt_path, 'GT_', img_name, '.mat'));
        annPoints = image_info{1}.location;
        im = imread(strcat(img_path, img_name, '.jpg'));
    elseif is_ucf_cc_50
        load(strcat(gt_path, num2str(idx), '_ann.mat')) ;
        im = imread(strcat(img_path, img_name, '.jpg'));
    elseif is_worldexpo
        load(strcat(gt_path, scene_number, '\', img_name, '.mat'));
        annPoints = point_position;
        im = imread(strcat(img_path, img_name, '.jpg'));
        if is_use_roi
            load(strcat(gt_path, scene_number, '\roi.mat'));
            roi_raw.x = maskVerticesXCoordinates;
            roi_raw.y = maskVerticesYCoordinates;
        end
    elseif is_airport
        load(strcat(gt_path, img_name, '.mat'));
        annPoints = image_info.location;
        im = imread(strcat(img_path, img_name, '.jpg'));
        if is_use_roi
            load(strcat(roi_root_path, 'roi-', scene_number, '.mat'));
            roi_raw.x = roi_coordinate(:, 1);
            roi_raw.y = roi_coordinate(:, 2);
        end
    elseif is_ucsd
        load(strcat(gt_path, scene_number, '_frame_full', '.mat'));
        annPoints = fgt.frame{1, str2num(img_name(15:17))}.loc(:, 1:2);
        im = imread(strcat(img_path, img_name, '.png'));
        if is_use_roi
            load(strcat(roi_root_path, 'vidf1_33_roi_mainwalkway.mat'));
            roi_raw.x = roi.xi;
            roi_raw.y = roi.yi;
        end
    elseif is_trancos
        annPoints = load(strcat(gt_path, img_name, '.txt'));
        im = imread(strcat(img_path, img_name, '.jpg'));
        if is_use_roi
            load(strcat(roi_root_path, img_name, 'mask.mat'));
            roi_raw_map = BW * 1.0;
        end
    elseif is_mall
        load(strcat(gt_path, 'mall_gt.mat'));
        annPoints = frame{1, str2double(scene_number)}.loc;
        im = imread(strcat(img_path, img_name, '.jpg'));
        if is_use_roi
            load(strcat(roi_root_path, 'perspective_roi.mat'));
            roi_raw_map = roi.mask * 1.0;
        end
    elseif is_ucf_qnrf
        load(strcat(gt_path, img_name, '_ann.mat')) ;
        im = imread(strcat(img_path, img_name, '.jpg'));
    elseif is_gcc
        load(strcat(gt_path, img_name, '.mat')) ;
        annPoints = image_info.location;
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
    
    if resize_level ~= 1.0
        im = imresize(im, resize_level, 'bilinear');
        annPoints = annPoints * resize_level;
        if is_use_roi
            if is_worldexpo || is_airport || is_ucsd
                roi_raw.x = roi_raw.x * resize_level;
                roi_raw.y = roi_raw.y * resize_level;
            elseif is_trancos || is_mall
                roi_raw_map = round(imresize(roi_raw_map, resize_level, 'bilinear'));
            end
        end
    end
    
    if is_resize
        [height, width, ~] = size(im);
        height_1 = TIMES * round(height / TIMES);
        width_1 = TIMES * round(width / TIMES);
        im = imresize(im, [height_1, width_1], 'bilinear');
        annPoints(:, 1) = annPoints(:, 1) / width * width_1;
        annPoints(:, 2) = annPoints(:, 2) / height * height_1;
        if is_use_roi
            if is_worldexpo || is_airport || is_ucsd
                roi_raw.x = roi_raw.x / width * width_1;
                roi_raw.y = roi_raw.y / height * height_1;
            elseif is_trancos || is_mall
                roi_raw_map = round(imresize(roi_raw_map, [height_1, width_1], 'bilinear'));
            end
        end
    end
    if is_use_roi
        if is_worldexpo || is_airport || is_ucsd
            [height, width, ~] = size(im);
            roi_raw.width = width;
            roi_raw.height = height;
        elseif is_trancos || is_mall
            % nothing need to do here
        end
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

    im_density = get_density_map_gaussian(im, annPoints, 15, 15);
    
%     im_gray = rgb2gray(im);
%     max_density = max(im_density(:));
%     imshow(imlincomb(0.5, im_gray, 0.5, uint8(im_density / max_density * 256)));
    
    if is_use_roi
        if is_worldexpo || is_airport || is_ucsd
            if is_worldexpo || is_airport
                roi_key = scene_number;
            elseif is_ucsd
                roi_key = 'all';
            end
            if roi_container.isKey(roi_key)
                roi_original = roi_container(roi_key);
                roi_fourth_size = roi_fourth_size_container(roi_key);
                roi_eighth_size = roi_eighth_size_container(roi_key);
            else
                roi_original = get_mask_map(roi_raw);
                roi_fourth_size = get_mask_map(roi_raw, 'fourth');
                roi_eighth_size = get_mask_map(roi_raw, 'eighth');
                roi_container(roi_key) = roi_original;
                roi_fourth_size_container(roi_key) = roi_fourth_size;
                roi_eighth_size_container(roi_key) = roi_eighth_size;
            end
        elseif is_trancos || is_mall
            roi_original.matrix = roi_raw_map;
            roi_original.mask = roi_raw_map == 1;
            roi_fourth_size.matrix = round(imresize(roi_raw_map, 0.25, 'bilinear'));
            roi_fourth_size.mask = roi_fourth_size.matrix == 1;
            roi_eighth_size.matrix = round(imresize(roi_raw_map, 0.125, 'bilinear'));
            roi_eighth_size.mask = roi_eighth_size.matrix == 1;
        end
        im_density = im_density .* roi_original.matrix;
        if is_apply_roi_to_image
            if is_gray_image
                im = im .* uint8(roi_original.matrix);
            else
                im(:, :, 1) = im(:, :, 1) .* uint8(roi_original.matrix);
                im(:, :, 2) = im(:, :, 2) .* uint8(roi_original.matrix);
                im(:, :, 3) = im(:, :, 3) .* uint8(roi_original.matrix);
            end
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
    
    if is_resize
        % nothing need to do here
    else
        [height, width, ~] = size(im);
        height_1 = TIMES * floor(height / TIMES);
        width_1 = TIMES * floor(width / TIMES);
        height_1_fourth_size = height_1 / 4;
        width_1_fourth_size = width_1 / 4;
        height_1_eighth_size = height_1 / 8;
        width_1_eighth_size = width_1 / 8;
        if is_gray_image
            im = im(1:height_1, 1:width_1);
        else
            im = im(1:height_1, 1:width_1, :);
        end
        im_density = im_density(1:height_1, 1:width_1);
        if is_use_roi
            roi_original.mask = roi_original.mask(1:height_1, 1:width_1);
            roi_fourth_size.mask = roi_fourth_size.mask(1:height_1_fourth_size, 1:width_1_fourth_size);
            roi_eighth_size.mask = roi_eighth_size.mask(1:height_1_eighth_size, 1:width_1_eighth_size);
            roi_original.matrix = roi_original.matrix(1:height_1, 1:width_1);
            roi_fourth_size.matrix = roi_fourth_size.matrix(1:height_1_fourth_size, 1:width_1_fourth_size);
            roi_eighth_size.matrix = roi_eighth_size.matrix(1:height_1_eighth_size, 1:width_1_eighth_size);
        end
    end

    imwrite(im, [img_save_path img_name '.jpg']);
    csvwrite([den_save_path img_name '.csv'], im_density);
    if is_save_roi
        roi = roi_original;
        save([roi_save_path img_name '_roi.mat'], 'roi');
        roi = roi_fourth_size;
        save([roi_save_path img_name '_roi_fourth_size.mat'], 'roi');
        roi = roi_eighth_size;
        save([roi_save_path img_name '_roi_eighth_size.mat'], 'roi');
    end
    if is_overturn
        im_overturn = fliplr(im);
        im_density_overturn = fliplr(im_density);
        if is_use_roi
            roi_original_overturn.mask = fliplr(roi_original.mask);
            roi_fourth_size_overturn.mask = fliplr(roi_fourth_size.mask);
            roi_eighth_size_overturn.mask = fliplr(roi_eighth_size.mask);
            roi_original_overturn.matrix = fliplr(roi_original.matrix);
            roi_fourth_size_overturn.matrix = fliplr(roi_fourth_size.matrix);
            roi_eighth_size_overturn.matrix = fliplr(roi_eighth_size.matrix);
        end
        imwrite(im_overturn, [img_save_path img_name '_overturn.jpg']);
        csvwrite([den_save_path img_name '_overturn.csv'], im_density_overturn);
        if is_save_roi
            roi = roi_original_overturn;
            save([roi_save_path img_name '_overturn_roi.mat'], 'roi');
            roi = roi_fourth_size_overturn;
            save([roi_save_path img_name '_overturn_roi_fourth_size.mat'], 'roi');
            roi = roi_eighth_size_overturn;
            save([roi_save_path img_name '_overturn_roi_eighth_size.mat'], 'roi');
        end
    end
end

