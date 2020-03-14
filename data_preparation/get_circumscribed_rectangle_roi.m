function [top, bottom, left, right] = get_circumscribed_rectangle_roi(mask)
%get the smallest circumscribed rectangle in ROI
%input mask is a m*n matrix

whole_size = size(mask);
outside_top = 0;
outside_bottom = whole_size(1);
outside_left = 0;
outside_right = whole_size(2);

mark = false;
for i = 1:whole_size(1)
    for j = 1:whole_size(2)
        if mask(i, j) == 1
            mark = true;
        end
    end
    if mark
        outside_top = i;
        mark = false;
        break;
    end
end

mark = false;
for i = whole_size(1):-1:1
    for j = 1:whole_size(2)
        if mask(i, j) == 1
            mark = true;
        end
    end
    if mark
        outside_bottom = i;
        mark = false;
        break;
    end
end

mark = false;
for j = 1:whole_size(2)
    for i = 1:whole_size(1)
        if mask(i, j) == 1
            mark = true;
        end
    end
    if mark
        outside_left = j;
        mark = false;
        break;
    end
end

mark = false;
for j = whole_size(2):-1:1
    for i = 1:whole_size(1)
        if mask(i, j) == 1
            mark = true;
        end
    end
    if mark
        outside_right = j;
        mark = false;
        break;
    end
end

top = outside_top;
bottom = outside_bottom;
left = outside_left;
right = outside_right;
end