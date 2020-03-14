%laridzhang 显示密度图
%d_map 二维数据
function view_density_map(d_map)
figure;
s = surf(d_map);
% s.EdgeColor = 'none';%不显示网格
set(s,'edgecolor','none')%不显示网格
view(0,-90);%旋转视角
axis equal off;%XY轴等比例 不显示XY轴
end