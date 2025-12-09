function fig = f_plotMap(X, Y, Var, carray, options)
%f_plotMap 绘制变量地图
%   x,y为线性经纬度
%   X,Y为二维经纬度
%   Var为绘图二维变量
%   carray为数值范围
%   seg为数值分段
%   rate为一个段内颜色数目
%   label_s为标签

arguments
    X 
    Y 
    Var 
    carray 
    options.Var_contour
    options.seg  = 10
    options.rate = 4
    options.label_s 
    options.colmap = othercolor('RdBu5') % f_ColorMap([0.8,0,0,0; 1,1,1,1/2; 0,0,0.8,1], 300)
    options.plot_land {mustBeMember(options.plot_land, {'no','yes','shoreline'})} = 'yes'
    options.color_bar_ticks
end
seg = options.seg;
rate = options.rate;
colmap = f_InterceptColorbar(options.colmap, rate*seg);

col_level = linspace(carray(1), carray(2), rate*seg+1);
col_tick = linspace(carray(1), carray(2), seg+1);

fig = figure('Color',[1,1,1]);
clf
m_proj('mercator','long',[min(X,[],'all'),max(X,[],'all')],'lat',[min(Y,[],'all'),max(Y,[],'all')]);     % ERA5范围

% 画图块
m_contourf(X, Y, Var,col_level,'edgecolor','none');
if isfield(options, 'Var_contour')
    hold on
    m_contour(X, Y, options.Var_contour, 10, 'k',"ShowText",true)
end
% col_map = f_InterceptColorbar(flipud(othercolor('RdBu4')), rate*seg);
col_map = colmap;
colormap(col_map)
% m_pcolor(X, Y, Var);
% colormap(flipud(othercolor('Spectral10')))
hc=colorbar();
if isfield(options, 'color_bar_ticks')
    hc.Ticks = options.color_bar_ticks{1};
    hc.TickLabels = options.color_bar_ticks{2};
else
    hc.Ticks = col_tick;
end
if isfield(options, 'label_s')
    hc.Label.String = options.label_s;
end
% hc.Label.Interpreter = 'latex';
hc.Label.FontName = '等线';
hc.FontSize = 11;
% 画边框
hold on
if options.plot_land == "yes"
    m_gshhs_i('patch',[.85 .85 .85])
elseif options.plot_land == "shoreline"
    m_gshhs_i('Color','k','Linewidth',1)
end
m_grid('box','fancy','tickdir','in','gridlines',[.25 .25 .25],'fontsize',12,'tickstyle','dd')
% 画箭头
m_northarrow(107.7,20.3,  0.8)
clim(carray)

set(gca, 'FontName', '等线', 'FontSize', 12)

end