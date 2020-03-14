function roi = get_mask_map_worldexpo(roi_data, mode)
%if is_quarter is true, will get quarter size ROI (same size of density map)
is_fourth = false;
is_eighth = false;

if nargin < 2
    is_fourth = false;
    is_eighth = false;
else
    if strcmp(mode, 'fourth')
        is_fourth = true;
    elseif strcmp(mode, 'eighth')
        is_eighth = true;
    end
end

%104207
%200608
%200702
%202201
%500717
%roi_path = 'D:\Dataset\WorldExpo10\test_label\500717\roi.mat';

X = 720;
Y = 576;

if is_fourth
    X = X / 4;
    Y = Y / 4;
    roi_data.x = roi_data.x / 4;
    roi_data.y = roi_data.y / 4;
end

if is_eighth
    X = X / 8;
    Y = Y / 8;
    roi_data.x = roi_data.x / 8;
    roi_data.y = roi_data.y / 8;
end

roi.mask = false(Y, X);
roi.matrix= zeros(Y, X);
for y_ = 1:Y
    for x_ = 1:X
        IN = inpolygon(x_, y_, roi_data.x, roi_data.y);
        roi.mask(y_,x_) = (IN==1);
        roi.matrix(y_, x_) = IN;
    end
end
%view_density_map(ma);
end