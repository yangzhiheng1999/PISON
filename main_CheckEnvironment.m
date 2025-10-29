% 查看Environment.mat场文件

clc; clear; close all;

load("../datas/Environment_0_2.mat")

find_index = 1314;

index_R_t = 0;
for i = 1: length(Wind)
    index_L_t = index_R_t;
    index_R_t = index_L_t + size(Wind{i}, 1);

    if find_index>index_L_t & find_index<=index_R_t
        cell_i = i;
        vec_i = find_index - index_L_t;
        break
    end
end

% 绘图坐标点
space_step = 0.2;
% space_step = 0.1;
x=105.3: space_step: 116; y=12:space_step:23;
[X,Y] = meshgrid(x, y);

var_field = Wind{cell_i}(vec_i, :);
var_field_2d = reshape(var_field, size(X));

% 风场
u = real(var_field_2d);
v = imag(var_field_2d);
% u = reshape(real(WindSpeed_t(:,99)), size(X));
% v = reshape(imag(WindSpeed_t(:,99)), size(X));
Var_temp = sqrt(u.^2 + v.^2);

% 画图
fig = figure('Color',[1 1 1]);
hold on
m_proj('mercator','long',[x(1),x(end)],'lat',[y(1),y(end)]);
s = m_pcolor(X, Y, Var_temp);
m_quiver(X, Y, u, -v);
colormap(f_ColorMap([1,1,1,0; ...
        0.9,0.9,.2, .5
        0.8,0,0, 1], 300))
hc=colorbar;
hc.FontSize = 11;
clim([0 20])
colorbar
hold on
% m_gshhs_i('patch',[.85 .85 .85])
% 画箭头
m_grid('box','fancy','tickdir','in','gridlines',[.25 .25 .25],'fontsize',12,'tickstyle','dd')


%% 对比原风场
UWindList = dir("F:/20241105TimeDifferenceOfWindsTidesAndWaves/datas/u_wind/*.mat");
VWindList = dir("F:/20241105TimeDifferenceOfWindsTidesAndWaves/datas/v_wind/*.mat");

load(strcat(UWindList(cell_i).folder,'\',UWindList(cell_i).name))
uwind = var_mat(:, vec_i);
load(strcat(VWindList(cell_i).folder,'\',VWindList(cell_i).name));
vwind = var_mat(:, vec_i);

% 绘图坐标点
% 读取fvcom网格
[node, ~] = f_getNODEandELE(1,"F:/20241105TimeDifferenceOfWindsTidesAndWaves/datas/fort.grd");
space_step = 0.2;
x=105.3: space_step: 116; y=12:space_step:23;
[X,Y] = meshgrid(x, y);
% 网格1维化
LonRegion = reshape(X, [length(x)*length(y),1]);
LatRegion = reshape(Y, [length(x)*length(y),1]);
% 网格点
Position = [LonRegion, LatRegion];
% 查找索引
PositionIndex = f_findPosition(Position, node(:,2), node(:,3));

uwind_2d = reshape(uwind(PositionIndex), size(X));
vwind_2d = reshape(vwind(PositionIndex), size(X));

% 画图
fig = figure('Color',[1 1 1]);
hold on
m_proj('mercator','long',[x(1),x(end)],'lat',[y(1),y(end)]);
s = m_pcolor(X, Y, sqrt(uwind_2d.^2 + vwind_2d.^2));
m_quiver(X, Y, uwind_2d, vwind_2d);
colormap(f_ColorMap([1,1,1,0; ...
        0.9,0.9,.2, .5
        0.8,0,0, 1], 300))
hc=colorbar;
hc.FontSize = 11;
clim([0 20])
colorbar
hold on
m_gshhs_i('patch',[.85 .85 .85])
% 画箭头
m_grid('box','fancy','tickdir','in','gridlines',[.25 .25 .25],'fontsize',12,'tickstyle','dd')