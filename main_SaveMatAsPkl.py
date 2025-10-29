# -*- coding: utf-8 -*-
# 自建函数库
from main_python_MyFunctions import f_ReadCellMat
from main_python_MyFunctions import f_PileMatrix
from main_python_MyFunctions import c_PickleHandler

if __name__=="__main__":
    
    # 读取mat文件
    print('**reading mat files ...')
    # typhoon_info = f_ReadCellMat('../datas/typhoon_info.mat', 'typhoon_info')
    
    # 数据文件句柄
    # typhoon_handler = c_PickleHandler('../datas/typhoon_piled.pkl')
    surge_handler = c_PickleHandler('../datas/surge_2d.pkl')
    current_handler = c_PickleHandler('../datas/current_2d.pkl')
    wave_handler = c_PickleHandler('../datas/wave_2d.pkl')
    wind_handler = c_PickleHandler('../datas/wind_2d.pkl')

    
    # 不需要每次都读取环境数据
    # 读取环境数据
    Surge = f_ReadCellMat('../datas/Environment_0_2.mat', 'Surge')
    Current = f_ReadCellMat('../datas/Environment_0_2.mat', 'Current')
    Wave = f_ReadCellMat('../datas/Environment_0_2.mat', 'Wave')
    Wind = f_ReadCellMat('../datas/Environment_0_2.mat', 'Wind')
    # 将读取出的数据堆叠起来
    print('piling data sets ...')
    # typhoon_piled = f_PileMatrix(typhoon_info)
    surge_piled = f_PileMatrix(Surge)
    current_piled = f_PileMatrix(Current)
    wave_piled = f_PileMatrix(Wave)
    wind_piled = f_PileMatrix(Wind)

    # 将堆叠数据展成二维矩阵
    N_time = surge_piled.shape[0]
    surge_2d = surge_piled.reshape(N_time, 54, 56).transpose(0, 2, 1)
    current_2d = current_piled.reshape(N_time, 54, 56).transpose(0, 2, 1)
    wave_2d = wave_piled.reshape(N_time, 54, 56).transpose(0, 2, 1)
    wind_2d = wind_piled.reshape(N_time, 54, 56).transpose(0, 2, 1)

    # 保存数据
    print('**saving data sets ...')
    # typhoon_handler.save_data(typhoon_piled)
    surge_handler.save_data(surge_2d)
    current_handler.save_data(current_2d)
    wave_handler.save_data(wave_2d)
    wind_handler.save_data(wind_2d)
    
    '''# 读取数据
    print('**reading data sets ...')
    # typhoon_piled = typhoon_handler.load_data()
    surge_piled = surge_handler.load_data()
    current_piled = current_handler.load_data()
    wave_piled = wave_handler.load_data()
    wind_piled = wind_handler.load_data()'''