function im_density = get_density_map_gaussian(im, points, gaussian_size, sigma)
if nargin == 2
    gaussian_size = 15;
    sigma = 4;
elseif nargin ~= 4
    error('No size or sigma provided.')
end

[h, w, ~] = size(im);
im_density = zeros(h, w); 

if(isempty(points))
    return;
end
%{
if(length(points(:,1))==1)
    x1 = max(1,min(w,round(points(1,1))));
    y1 = max(1,min(h,round(points(1,2))));
    im_density(y1,x1) = 255;
    return;
end
%}
for j = 1:length(points(:,1))
    H = fspecial('Gaussian', gaussian_size, sigma);
    
    x = floor(points(j,1));
    y = floor(points(j,2));
    if(x > w || y > h || x < 1 || y < 1)
        continue;
    end
    x = min(w,max(1,abs(int32(x)))); 
    y = min(h,max(1,abs(int32(y))));
    
    x1 = x - int32(floor(gaussian_size / 2)); y1 = y - int32(floor(gaussian_size / 2));
    x2 = x + int32(floor(gaussian_size / 2)); y2 = y + int32(floor(gaussian_size / 2));
    dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0;
    change_H = false;
    if(x1 < 1)
        dfx1 = abs(x1)+1;
        x1 = 1;
        change_H = true;
    end
    if(y1 < 1)
        dfy1 = abs(y1)+1;
        y1 = 1;
        change_H = true;
    end
    if(x2 > w)
        dfx2 = x2 - w;
        x2 = w;
        change_H = true;
    end
    if(y2 > h)
        dfy2 = y2 - h;
        y2 = h;
        change_H = true;
    end
    x1h = 1+dfx1; y1h = 1+dfy1; x2h = gaussian_size - dfx2; y2h = gaussian_size - dfy2;
    if (change_H == true)
        H =  fspecial('Gaussian',[double(y2h-y1h+1), double(x2h-x1h+1)],sigma);
    end
    im_density(y1:y2, x1:x2) = im_density(y1:y2, x1:x2) + H;
     
end

end