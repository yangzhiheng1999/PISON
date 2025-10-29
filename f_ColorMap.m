function color_map = f_ColorMap(ColorMat, ColorNum)
%生成color map
%   ColorMap为N*4矩阵，分别为R、G、B、位置（0-1）
%   ColorNum为标量，指示colormap共有多少颜色

position = ColorMat(:,4);
position_index = floor((ColorNum-1)*position + 1);

segment_length = position_index(2: end) - position_index(1: end-1)+1;

R = ColorMat(:,1); G = ColorMat(:,2); B = ColorMat(:,3);

for i = 1: length(position_index)-1
    colormap_r(position_index(i):position_index(i+1)) = linspace(R(i),R(i+1),segment_length(i));
    colormap_g(position_index(i):position_index(i+1)) = linspace(G(i),G(i+1),segment_length(i));
    colormap_b(position_index(i):position_index(i+1)) = linspace(B(i),B(i+1),segment_length(i));
end

color_map = [colormap_r', colormap_g', colormap_b'];

end