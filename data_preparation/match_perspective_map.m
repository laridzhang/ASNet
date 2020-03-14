clc;
clear all;

% target_perspective_path = 'D:\Dataset\WorldExpo10\original\test_perspective\104207.mat';
% target_perspective_path = 'D:\Dataset\WorldExpo10\original\test_perspective\200608.mat';
% target_perspective_path = 'D:\Dataset\WorldExpo10\original\test_perspective\200702.mat';
% target_perspective_path = 'D:\Dataset\WorldExpo10\original\test_perspective\202201.mat';
target_perspective_path = 'D:\Dataset\WorldExpo10\original\test_perspective\500717.mat';

train_perspective_path = 'D:\Dataset\WorldExpo10\original\train_perspective';
train_perspective_dir = dir(fullfile(train_perspective_path,'*.mat'));
train_perspective_name_list = {train_perspective_dir.name};

load(target_perspective_path);
target_perspective_map = pMap;

absolute_error_list = zeros(103, 1);

for i = 1:103
    load([train_perspective_path '\' train_perspective_name_list{i}]);
    train_perspective_map = pMap;
    absolute_error_map = abs(target_perspective_map - train_perspective_map);
    absolute_error = sum(absolute_error_map(:));
    absolute_error_list(i) = absolute_error;
end

[~, sort_idx] = sort(absolute_error_list, 'ascend');
for i = 1:4
    k = sort_idx(i);
    disp(train_perspective_name_list{k});
    disp(absolute_error_list(k));
end