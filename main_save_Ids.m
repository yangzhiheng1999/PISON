% 提取Environment对应的Id


clc
clear
close all


%% mat文件
ZetaList = dir("F:/20241105TimeDifferenceOfWindsTidesAndWaves/datas/zeta/*.mat");
load("../datas/Environment_0_2.mat")
clear Wave Current Wind

% 编号
Id = [];

% 预设阈值
for tc_i = 1: length(ZetaList)
    % TC场次
    tc_id = str2double(ZetaList(tc_i).name(1:6));
    fprintf("%6d\n", tc_id)

    Id = [Id; repmat(tc_id, size(Surge{tc_i}, 1), 1)];

    disp('================================')
end

save("../datas/Id.mat", 'Id')
