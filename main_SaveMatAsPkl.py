# -*- coding: utf-8 -*-
# 自建函数库
from main_python_MyFunctions import f_ReadCellMat
from main_python_MyFunctions import f_PileMatrix
from main_python_MyFunctions import c_PickleHandler
import gc  # 关键：手动垃圾回收

if __name__=="__main__":

    # 数据文件句柄
    # typhoon_handler = c_PickleHandler('../datas/typhoon_piled.pkl')
    surge_handler = c_PickleHandler('../datas/surge_2d.pkl')
    current_handler = c_PickleHandler('../datas/current_2d.pkl')
    wave_handler = c_PickleHandler('../datas/wave_2d.pkl')
    wind_handler = c_PickleHandler('../datas/wind_2d.pkl')

    # 将读取出的数据堆叠起来
    print('**proccessing surge data')
    Surge = f_ReadCellMat('../datas/Environment_64.mat', 'Surge')
    print('piling surge ...')
    surge_piled = f_PileMatrix(Surge)
    print('reshaping piled surge data ...')
    N_time = surge_piled.shape[0]
    surge_2d = surge_piled.reshape(N_time, 64, 64).transpose(0, 2, 1)
    print('saving surge data ...')
    surge_handler.save_data(surge_2d)
    # 释放内存
    del surge_2d
    del surge_piled
    del Surge
    gc.collect()

    print('**proccessing current data')
    Current = f_ReadCellMat('../datas/Environment_64.mat', 'Current')
    print('piling current ...')
    current_piled = f_PileMatrix(Current)
    print('reshaping piled current data ...')
    current_2d = current_piled.reshape(N_time, 64, 64).transpose(0, 2, 1)
    print('saving current data ...')
    current_handler.save_data(current_2d)
    # 释放内存
    del current_2d
    del current_piled
    del Current
    gc.collect()


    print('**proccessing wave data')
    Wave = f_ReadCellMat('../datas/Environment_64.mat', 'Wave')
    print('piling wave ...')
    wave_piled = f_PileMatrix(Wave)
    print('reshaping piled wave data ...')
    wave_2d = wave_piled.reshape(N_time, 64, 64).transpose(0, 2, 1)
    print('saving wave data ...')
    wave_handler.save_data(wave_2d)
    # 释放内存
    del wave_2d
    del wave_piled
    del Wave
    gc.collect()


    print('**proccessing wind data')
    Wind = f_ReadCellMat('../datas/Environment_64.mat', 'Wind')
    print('piling wind ...')
    wind_piled = f_PileMatrix(Wind)
    print('reshaping piled wind data ...')
    wind_2d = wind_piled.reshape(N_time, 64, 64).transpose(0, 2, 1)
    print('saving wind data ...')
    wind_handler.save_data(wind_2d)
    # 释放内存
    del wind_2d
    del wind_piled
    del Wind
    gc.collect()