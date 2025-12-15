import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from numpy.random import choice
import pandas as pd
from scipy.stats import gaussian_kde
import datetime
import os

def read_matlab_mat(file_path, var_name):
    """
    读取MATLAB v7.3版本的MAT文件中的指定变量。

    参数:
    file_path (str): MAT文件的路径。
    var_name (str): 要读取的变量名。

    返回:
    numpy.ndarray: 读取到的变量数据。

    抛出:
    ValueError: 如果指定的变量名不存在于文件中。
    RuntimeError: 如果在读取文件时发生其他错误。
    """
    import h5py

    try:
        with h5py.File(file_path, 'r') as file:
            dataset = file[var_name]
            data = dataset[()]
        return data
    except KeyError:
        raise ValueError(f"Variable '{var_name}' not found in the file.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

def timedelta_to_hours(timedelta):
    return timedelta.total_seconds() / 3600

def fetch_chabst_track(file_path, time_interval=6.0):
    """
    读取文件，提取纬度和经度数据
    补充计算移动速度和最大风半径
    :param file_path: 文件路径
    """

    # 打开文件并读取数据
    #从文件中读取日期，并转化为python日期格式
    # date = datetime.datetime.strptime(date_str, '%Y%m%d%H%M')
    dates = []
    # 初始化列表
    lats = []
    lons = []
    air_pressures = []
    wind_speeds = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 跳过首行（标题行）
        for line in lines[1:]:
            # 去除换行符和可能的空格
            line = line.strip()
            # 提取日期字符串
            date_str = line[0:10]
            # 提取纬度和经度字符串
            lat_str = line[13:16]  # 索引13到15（三个字符）
            lon_str = line[17:21]  # 索引17到20（四个字符）
            # 提取其他数据
            air_pressure_str = line[22:27]
            wind_speed_str = line[32:35]
            
            # 转换为浮点数
            try:
                dates.append(datetime.datetime.strptime(date_str, '%Y%m%d%H'))
                lat = float(f"{lat_str[:2]}.{lat_str[2]}")
                lon = float(f"{lon_str[:3]}.{lon_str[3]}")
                air_pressure = float(air_pressure_str)
                wind_speed = float(wind_speed_str)
                lats.append(lat)
                lons.append(lon)
                air_pressures.append(air_pressure)
                wind_speeds.append(wind_speed)
            except ValueError:
                # 处理可能的无效数据
                print(f"无法转换的数据行：{line}")

    # 根据经纬度计算移动速度
    moving_speeds = []
    for i in range(1, len(lats)):
        # 计算经纬度差值
        lat_diff = lats[i] - lats[i - 1]
        lon_diff = lons[i] - lons[i - 1]
        # 计算移动速度（假设每行数据间隔1小时）
        moving_speed = (lat_diff**2 + lon_diff**2)**0.5 * 111 / timedelta_to_hours(dates[i]-dates[i-1])  # 每度约等于111公里
        moving_speeds.append(moving_speed)
        
    moving_speeds.append(moving_speed)  # 最后一个点的移动速度与前一个点相同

    # 计算最大风半径
    max_wind_radius = []
    max_wind_radius = [(-37.82 + 0.11 * lat) * np.log((2.86 - 0.0029 * lat) * (1013 - air_pressure)**0.7) + 178.2
                       for lat, air_pressure in zip(lats, air_pressures)]

    # 输出结果
    return dates, lats, lons, air_pressures, wind_speeds, moving_speeds, max_wind_radius

def interpolate_track(typhoon_track):
    # 对台风轨迹进行插值，使其每小时都有数据点
    """    对台风轨迹进行插值，使其每小时都有数据点。
    参数:   
        typhoon_track (tuple): 包含日期、经度、纬度、气压、风速、移动速度和最大风半径的元组。
    返回:
        tuple: 插值后的日期、经度、纬度、气压、风速、移动速度和最大风半径。
    """
    dates, lats, lons, air_pressures, wind_speeds, moving_speeds, max_wind_radius = typhoon_track
    # 创建新的日期列表，间隔为1小时
    new_dates = []
    for i in range(len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1]
        # 计算时间间隔
        time_diff = timedelta_to_hours(end_date - start_date)  # 转换为小时
        # 插入每小时的日期
        for j in range(int(time_diff)):
            new_dates.append(start_date + datetime.timedelta(hours=j))
    new_dates.append(end_date)  # 添加最后一个日期
    # 插值经度和纬度
    new_lats = np.interp(
        [date.timestamp() for date in new_dates],
        [date.timestamp() for date in dates],
        lats
    )
    new_lons = np.interp(
        [date.timestamp() for date in new_dates],
        [date.timestamp() for date in dates],
        lons
    )
    # 插值气压、风速、移动速度和最大风半径
    new_air_pressures = np.interp(
        [date.timestamp() for date in new_dates],
        [date.timestamp() for date in dates],
        air_pressures
    )
    new_wind_speeds = np.interp(
        [date.timestamp() for date in new_dates],
        [date.timestamp() for date in dates],
        wind_speeds
    )
    new_moving_speeds = np.interp(
        [date.timestamp() for date in new_dates],
        [date.timestamp() for date in dates],
        moving_speeds
    )
    new_max_wind_radius = np.interp(
        [date.timestamp() for date in new_dates],
        [date.timestamp() for date in dates],
        max_wind_radius
    )
    # 返回插值后的结果
    return {
        'dates': new_dates,
        'lons': new_lons.tolist(),  
        'lats': new_lats.tolist(), 
        'air_pressures': new_air_pressures.tolist(), 
        'wind_speeds': new_wind_speeds.tolist(), 
        'moving_speeds': new_moving_speeds.tolist(), 
        'max_wind_radius': new_max_wind_radius.tolist()
    }


def fetch_observe_value(file_path):
    """
    读取文件，提取观测数据
    :param file_path: 文件路径
    """
    # 打开文件并读取数据
    #从文件中读取日期，并转化为python日期格式
    # date = datetime.datetime.strptime(date_str, '%Y%m%d%H%M')
    dates = []
    # 初始化列表
    data = []

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 去除换行符和可能的空格
                line = line.strip()
                # 提取日期字符串
                date_str = line.split('\t')[0]
                # 提取数据字符串
                data_str = line.split('\t')[1]
                
                # 转换为浮点数
                try:
                    dates.append(datetime.datetime.strptime(date_str, '%Y/%m/%d %H:%M'))
                    data.append(float(data_str))
                except ValueError:
                    # 处理可能的无效数据
                    print(f"无法转换的数据行：{line}")
    else:
        print(f"文件 {file_path} 不存在。")
        dates = []
        data = []

    # 输出结果
    return dates, data

# 读取轨迹数据并分割为历史轨迹和预报轨迹
def read_and_split_track(file_path, history_length):
    """
    读取轨迹数据并分割为历史轨迹和预报轨迹。
    参数:
        file_path (str): 轨迹数据文件路径。
        history_length (int): 历史轨迹的长度。
    返回:
        tuple: 历史经度列表, 历史纬度列表, 预报经度列表, 预报纬度列表。
    异常:
        ValueError: 如果轨迹长度不足以分割历史和预报轨迹。
    """
    # 从文件中提取轨迹数据
    dates, lats, lons, air_pressures, wind_speeds, moving_speeds, max_wind_radius = fetch_chabst_track(file_path)
    # 检查轨迹长度是否足够
    if len(lats) <= history_length:
        raise ValueError("轨迹长度不足，无法分割历史和预报轨迹。")
    # 分割历史轨迹
    history_track = {
        'dates': dates[:history_length + 1],
        'lons': lons[:history_length + 1], 
        'lats': lats[:history_length + 1],
        'air_pressures': air_pressures[:history_length + 1],
        'wind_speeds': wind_speeds[:history_length + 1],
        'moving_speeds': moving_speeds[:history_length + 1],
        'max_wind_radius': max_wind_radius[:history_length + 1]
    }
    analysis_track = {
        'dates': dates[history_length: history_length + 72 // 6 + 1: 24 // 6],
        'lons': lons[history_length: history_length + 72 // 6 + 1: 24 // 6], 
        'lats': lats[history_length: history_length + 72 // 6 + 1: 24 // 6],
        'air_pressures': air_pressures[history_length: history_length + 72 // 6 + 1: 24 // 6],
        'wind_speeds': wind_speeds[history_length: history_length + 72 // 6 + 1: 24 // 6],
        'moving_speeds': moving_speeds[history_length: history_length + 72 // 6 + 1: 24 // 6],
        'max_wind_radius': max_wind_radius[history_length: history_length + 72 // 6 + 1: 24 // 6]
    }
    # 分割预报轨迹，间隔为24小时
    
    return history_track, analysis_track

# 绘制地图函数
def create_map(history_lons, history_lats, forecast_lons, forecast_lats, radius, sample=None, save_path=None):
    # 创建地图和投影
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 设置地图显示范围
    ax.set_extent([104, 130, 10, 27], crs=ccrs.PlateCarree())
    
    # 添加地图特征
    ax.add_feature(cfeature.LAND, facecolor='lightgray')  # 陆地
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')  # 海洋
    ax.add_feature(cfeature.COASTLINE)  # 海岸线
    
    # 绘制预报轨迹
    ax.plot(
        forecast_lons, forecast_lats, label='Forecast track', marker='o',
        color='red', linewidth=2, markersize=5, transform=ccrs.PlateCarree()
    )
    
    # 绘制历史轨迹
    ax.plot(
        history_lons, history_lats, label='History track', marker='o',
        color='green', linewidth=2, markersize=5, transform=ccrs.PlateCarree()
    )

    # 根据样本在图上绘制散点
    if sample is not None:
        ax.scatter(
            sample['lons'], sample['lats'], color='orange', s=10, alpha=0.5,
            label='Sample points', transform=ccrs.PlateCarree()
        )
    
    # 根据半径绘制概率圆
    for i in range(len(radius)):
        circle = plt.Circle(
            (forecast_lons[i + 1], forecast_lats[i + 1]), radius[i],
            label='Probability circle', color='blue', alpha=0.5,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(circle)

    # 添加网格线
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    gl.xlocator = plt.MultipleLocator(2)  # 经度间隔
    gl.ylocator = plt.MultipleLocator(2)  # 纬度间隔
    # 设置经纬度标签样式
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    # 添加地图外边框
    gl.top_labels = False
    gl.right_labels = False
    
    # 设置标题
    ax.set_title('Ensemble Forecast Map', fontsize=16, fontweight='bold')
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[i] for i in [0, 1, 2, 5]], [labels[i] for i in [0, 1, 2, 5]], loc='upper right', fontsize=15,
        frameon=True, facecolor='white', edgecolor='black'
    )

    plt.tight_layout()
    
    # 显示地图
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    # 保存地图

def save_list_to_csv(data_list, filename):
  """将一维列表保存为CSV文件。

  Args:
    data_list: 要保存的列表。
    filename: CSV文件名。
  """
  import csv
  
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data_list)

def save2txt(file_path, data, format='%.6f'):
    """
    保存预测结果到txt文件。
    参数:
        YPred_history (numpy.ndarray): 历史预测结果。
        YPred_forecast (list): 未来预测结果列表。
    """
    # 保存预测结果：将data保存为指定目录下的纯数字txt文件
    np.savetxt(file_path, data, fmt=format)
    print(f"预测结果已保存到: {file_path}")

def plot_forecast_line(dates_line, results_line, 
                       dates_scatter, results_scatter, 
                       observe_date=None, observe_data=None, 
                       probability=None, y_label="Data", y_range=None, threshold=None, bw_method=1, save_path=None):
    """
    绘制预报结果的折线图和小提琴图。
    参数:
        dates_line (list): 历史日期列表。
        results_line (list): 历史预报结果列表。
        dates_scatter (list): 预测日期列表。
        results_scatter (list): 预测结果列表。
        scatter_size (int): 默认值10，未在图中使用。
        save_path (str): 保存路径，默认为None表示不保存。
    """
    # 转换所有日期为matplotlib数值格式
    dates_line_num = mdates.date2num(dates_line)
    dates_scatter_num = mdates.date2num(dates_scatter)
    dates_ob_line_num = mdates.date2num(observe_date)

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 准备小提琴图数据
    dates_flat = []
    values_flat = []
    weights_flat = []
    new_results = resample_with_extension(results_scatter, probability, n_samples_per_date=50000, bandwidth_scale=bw_method)

    # violinplot使用的数据方法
    for date, row in zip(dates_scatter_num, new_results):
        for x in row:
            dates_flat.append(date)
            values_flat.append(x.item())
    '''# kdeplot使用的数据方法
    for x,p in zip(results_scatter[0], probability[0]):
        values_flat.append(x.item())
        weights_flat.append(p.item())'''

    date_data = pd.DataFrame({
        'date': dates_flat-min(dates_scatter_num),
        'value': values_flat
    })

    # 先绘制小提琴图，设置透明度
    sns.violinplot(
        x='date',
        y='value',
        data=date_data,
        inner="box",
        cut=0,
        color="tomato",
        linecolor="#2d3436",
        fill=True,
        density_norm='width',
        saturation=0.8,
        bw_method=bw_method,
        width=0.7,      # 减小宽度避免重叠
        linewidth=1.0,    # 增加边框线宽
        label='Ensemble forecast',
        alpha=0.9,     # 设置透明度
        zorder=1,
        ax=ax
    )
    '''sns.kdeplot(
        data=date_data,
        # x='date',
        y='value',
        weights='weights',
        fill=True,
        vertical=False,
        color="tomato",
        alpha=0.9,     # 设置透明度
        bw_method=bw_method,
        label='Ensemble forecast',
        zorder=1,
        ax=ax
    )'''
    '''sns.scatterplot(
        x='date',
        y='value',
        data=date_data,
        label='Ensemble forecast',
        zorder=1,
        ax=ax
    )'''

    # 再绘制后报，确保在顶层
    sns.lineplot(
        x=dates_line_num - min(dates_scatter_num),
        y=[arr.item() for arr in results_line],
        marker='o',
        linestyle='-',
        color='green',
        markersize=10,
        linewidth=0,
        label='Hindcast',
        zorder=3,
        ax=ax
    )

    if observe_data:
        # 绘制观测数据的折线图
        sns.lineplot(
            x=dates_ob_line_num - min(dates_scatter_num),
            y=observe_data,
            marker=None,
            linestyle='-',
            color='#0984e3',
            markersize=8,
            linewidth=2,
            label='Reference value',
            zorder=3,
            ax=ax
        )

    if threshold:
        sns.lineplot(
            x=(dates_scatter_num - min(dates_scatter_num))*1.5-0.5,
            y=threshold,
            linestyle='--',
            color='orange',
            linewidth=3,
            label='Threshold',
            zorder=2,
            ax=ax
        )

    # 设置 x 轴为日期格式
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # 设置y轴范围
    if y_range:
        ax.set_ylim(y_range)
    # 获取x轴的刻度位置
    all_date = ax.get_xticks()
    ax.set_xticks(all_date)
    ax.set_xticklabels([
            mdates.num2date(x).strftime('%Y-%m-%d %H:%M') 
            for x in all_date+min(dates_scatter_num)], 
            rotation=15, fontsize=20
    )
    # 设置y轴刻度字号
    ax.tick_params(axis='y', labelsize=22)

    # 美化图形
    ax.set_xlabel('')
    ax.set_ylabel(y_label, labelpad=15, fontsize=24)
    ax.grid(True, linestyle='--', alpha=0.7)
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    if observe_data:
        ax.legend(
            [handles[i] for i in [-4, -3,-2,-1]], [labels[i] for i in [-4, -3,-2,-1]], loc='upper left', fontsize=24,
            frameon=True, facecolor='white', edgecolor='black'
        )
    else:
        ax.legend(
            [handles[i] for i in [-2,-1]], [labels[i] for i in [-2,-1]], loc='upper left', fontsize=24,
            frameon=True, facecolor='white', edgecolor='black'
        )
    plt.tight_layout()

    # 保存图像（如果提供路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print('图像已保存到:', save_path)
    
    plt.show()

def resample_with_extension(results_scatter, probability, n_samples_per_date=500, bandwidth_scale=1):
    """
    对每个日期行的数据进行重采样并添加噪声以扩展范围。
    
    参数:
        results_scatter (np.ndarray): 二维数组，每行代表日期的结果集合。
        probability (np.ndarray): 对应结果的概率值数组。
        n_samples_per_date (int): 每个日期生成的新样本数。
        noise_scale_factor (float): 控制噪声幅度的因子，基于数据标准差。
    
    返回:
        np.ndarray: 重采样后的二维数组，形状为(n_dates, n_samples_per_date)。
    """
    # 转换为NumPy数组并进行维度校验
    results_scatter = np.asarray([[x.item() for x in row] for row in results_scatter])
    probability = np.asarray(probability)
    
    # 校验维度一致性
    assert results_scatter.shape == probability.shape, "输入数组维度必须一致"
    assert len(results_scatter.shape) == 2, "需要二维输入数组"

    n_dates, n_elements = results_scatter.shape
    resampled_data = np.zeros((n_dates, n_samples_per_date))
    
    for i in range(n_dates):
        # 获取当前日期数据
        data = results_scatter[i]
        weights = probability[i]        
        
        # 处理零概率和归一化
        weights = weights / weights.sum()
        
        # 创建加权KDE估计器
        kde = gaussian_kde(data, weights=weights)
        
        # 调整带宽扩大采样范围
        original_bw = kde.factor
        # print('KDE带宽{%.4f}', original_bw * bandwidth_scale)
        kde.set_bandwidth(bw_method = original_bw * bandwidth_scale)
        
        # 生成新样本
        samples = kde.resample(n_samples_per_date)
        resampled_data[i] = samples.flatten()
    
    return resampled_data

def plot_exceedance_probability(
    sample_num, exceed_prob,
    y_label='超越概率',
    x_label='样本数量',
    save_path=None,
    line_color='blue',
):
    """
    绘制超越概率随样本数量变化的折线图。
    
    参数:
        sample_num (list): 样本数量列表。
        exceed_prob (list): 超越概率列表。
        y_label (str): y轴标签，默认为'超越概率'。
        x_label (str): x轴标签，默认为'样本数量'。
        save_path (str): 保存路径，默认为None表示不保存。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sample_num, exceed_prob, marker='o', linestyle='-', color=line_color)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像（如果提供路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print('图像已保存到:', save_path)
    
    plt.show()












