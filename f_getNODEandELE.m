function [node, tri] = f_getNODEandELE(mode_id, file_path, ele_path)
% 本函数用于提取网格的节点和坐标数据
%   输入参数为调用模式：
%   1：直接读取sms的*.grd（或fort.14）
%   2：分别读取*.node和*.ele文件
%       默认为模式1, 返回的node中含有水深数据（模式2中不含）
%
%   返回值为[node, tri]
%   mode1中node为三列，分别为lon、lat、dep
%   mode2中node为两列，分别为lon、lat
%   tri为三列，包含三角形三个顶点

if nargin == 0
    mode_id = 1;
end

switch mode_id
    case 1
        switch nargin
            case 1
                [file, path] = uigetfile({'*.grd; *.14'}, "选择*.grd文件");
                loc = strcat(path, file);
                [node, tri] = read_grd(loc);
            case 2
                [node, tri] = read_grd(file_path);
            otherwise
                error('读取grd（fort.14)文件，f_getNODEandELE函数输入多');
        end

    case 2
        switch nargin
            case 1
                [node_file, node_path] = uigetfile('*.node', "选择*.node文件");
                [ele_file, ele_path] = uigetfile('*.ele', "选择*.ele文件");
                n_loc = strcat(node_path, node_file);
                e_loc = strcat(ele_path, ele_file);
                [node, tri] = read_nodeandele(n_loc, e_loc);
            case 3
                [node, tri] = read_nodeandele(file_loc, ele_loc);
            otherwise
                error('读取node和ele文件，f_getNODEandELE函数输入不满足要求')
        end
end



    
    function [node, tri] = read_grd(grd_loc)                    % 读取grd文件
        fileID = fopen(grd_loc, 'r');
        info = fscanf(fileID, '%i', [1, 2]);                    % 读取开头信息
        tri_num = info(1);                                      % 读取面数
        node_num = info(2);                                     % 读取节点数
        node = fscanf(fileID, '%f', [4, node_num])';
        tri = fscanf(fileID, '%f', [5, tri_num])';
        fclose(fileID);
    end
    
    function [node, tri] = read_nodeandele(node_loc, ele_loc)   % 读取node和ele文件
        fid_n = fopen(node_loc);                                % load TRIANGLE vertex based connectivity file
        [nnode] = fscanf(fid_n,'%i',[1 4]);                     % get number of nodes
        ncol = 3+nnode(3)+nnode(4);                             % specify number of columns in nodefile
        data = fscanf(fid_n,'%f',[ncol nnode(1)])';             % get data
        node=data(:,2:3);                                       % get coordinates
        fclose(fid_n);

        fid_e = fopen(ele_loc);                                 % load TRIANGLE element based connectivity file
        [nelem] = fscanf(fid_e,'%i',[1 3]);                     % get number of triangles
        ncol = 4+nelem(3);                                      % specify number of columns in elefile
        tri = fscanf(fid_e,'%i',[ncol nelem(1)])';              % get connectivity table
        fclose(fid_e);
    end

end













