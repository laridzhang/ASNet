function [top, bottom, left, right] = get_inscribed_rectangle_roi(mask)
%get the largest inscribed rectangle in ROI
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

inside_top = round((outside_top + outside_bottom) / 2) - 1;
inside_bottom = round((outside_top + outside_bottom) / 2) + 1;
inside_left = round((outside_left + outside_right) / 2) - 1;
inside_right = round((outside_left + outside_right) / 2) + 1;
raw_inside_left = inside_left;
raw_inside_right = inside_right;

proportion = abs(outside_right - outside_left) / abs(outside_bottom - outside_top);

mark_top = true;
mark_bottom = true;
mark_left = true;
mark_right = true;

while mark_top || mark_bottom || mark_left || mark_right
    if mark_top
        for i = inside_left:inside_right
            if mask(inside_top, i) == 0
                mark_top = false;
            end
        end
        if mark_top
            inside_top = inside_top - 1;
        end
    end
    
    if mark_bottom
        for i = inside_left:inside_right
            if mask(inside_bottom, i) == 0
                mark_bottom = false;
            end
        end
        if mark_bottom
            inside_bottom = inside_bottom + 1;
        end
    end
    
    if mark_left
        for i = inside_top:inside_bottom
            if mask(i, inside_left) == 0
                mark_left = false;
            end
        end
        if mark_left
            raw_inside_left =  raw_inside_left - proportion;
            inside_left = round(raw_inside_left);
        end
    end
    
    if mark_right
        for i = inside_top:inside_bottom
            if mask(i, inside_right) == 0
                mark_right = false;
            end
        end
        if mark_right
            raw_inside_right = raw_inside_right + proportion;
            inside_right = round(raw_inside_right);
        end
    end
end

top = inside_top;
bottom = inside_bottom;
left = inside_left;
right = inside_right;

end