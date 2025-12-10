import h5py
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# 读取以cell格式存储的mat文件
def f_ReadCellMat(mat_path:str, mat_name:str):
    # 打开mat文件
    with h5py.File(mat_path, 'r') as f:
        # 列出文件中的所有组和数据集
        # print(list(f.keys()))
        # 获取指定数据集
        data_f = f[mat_name]
        # 创建一个空列表，用于存储转换后的数据
        data = []
        # 遍历每个HDF5对象引用
        for ref in data_f:
            # 使用获取引用指向的数据
            data_t = f[ref[0]]
            # 将数据转换为NumPy数组，并添加到列表中
            data.append(np.array(data_t))
    # 返回变量
    return data


# 合并矩阵
'''
将输入多维 (本案例中为3维, 第1维为台风id) data_mat列表降维转化为二维竖向堆叠的矩阵data
注意三维列表的第二位(索引1)的位置为特征, 第二位有多少数代表有多少特征
'''
def f_PileMatrix(data_mat: list):
    # 获取最大特征数量
    max_charactor_length = max(len(item) for item in data_mat)

    # 初始化二维列表，每一列对应一个特征
    data = [[] for _ in range(max_charactor_length)]

    for item in data_mat:
        for i, value in enumerate(item):
            # 如果当前特征数量小于最大特征数量，则填充空值（可以根据实际情况替换为其他填充值）
            if i < len(data):
                data[i].extend(value)
            else:
                data[i].extend(None)  # 或其他填充值

    data_array = np.array(data)
    data_transpose = data_array.transpose(1,0)
    return data_transpose

# 创建一个对象，读写数据文件
'''
文件格式为pickle, 因此需要pickle标准库
'''
class c_PickleHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_data(self, data):
        with open(self.file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            return data

# 创建一个对象，整理数据格式
class c_SuitDataFormat:
    def __init__(self, data):
        self.origin_data = data
        self.scale = MinMaxScaler()
    
    def f_extract_column(self, column_i):
        data = [row[column_i] for row in self.origin_data]
        data = np.array(data)
        self.column_data = data.reshape(-1, 1)
        return self.column_data
    
    def f_normalize(self, data):
        self.normalized = self.scale.fit_transform(data)
        return self.normalized
    
    def f_inverse_normalize(self, data):
        if len(data.shape) ==1:
            data = data.reshape(-1,1)
        self.inversed_normalized = self.scale.inverse_transform(data)
        return self.inversed_normalized
    












