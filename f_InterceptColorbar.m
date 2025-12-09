function [col_new] = f_InterceptColorbar(col, seg)
% 将colorbar均分成seg段，返回新的colorbar

col_length = size(col, 1);  % colorbar大小

k = (col_length-1)/(seg-1);     % 插值斜率
b = 1-k;                    % 插值截距

xx = 1: 1: seg;
col_index = round(k*xx+b);

col_new = col(col_index,:);

end