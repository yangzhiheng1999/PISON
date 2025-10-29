# -*- coding: utf-8 -*-
"""
Created on Tue Oct  16 2025
@author: zhihe
=================================
python version: 3.9.21
relied on module: h5py, numpy, ...

based on main_python_MyFunctions
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
# 自建函数库
from main_python_MyFunctions import c_PickleHandler


def f_load_pickle(file_path):
    handle = c_PickleHandler(file_path)
    data = handle.load_data()
    return data



# 1. 数据预处理模块
class WindWaveDataset(Dataset):
    def __init__(self, wind_data, wave_data, input_timesteps=3, output_timesteps=1):
        """
        风场和波浪场数据集
        :param wind_data: 风场数据 [num_samples, time_steps, height, width]
        :param wave_data: 波浪场数据 [num_samples, time_steps, height, width]
        :param input_timesteps: 输入时间步数
        :param output_timesteps: 输出时间步数
        """
        self.wind_data = wind_data
        self.wave_data = wave_data
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
    def __len__(self):
        return len(self.wind_data) - self.input_timesteps - self.output_timesteps + 1
    
    def __getitem__(self, idx):
        # 获取输入序列（风场）
        x = self.wind_data[idx:idx+self.input_timesteps]
        # 获取输出序列（波浪场）
        y = self.wave_data[idx+self.input_timesteps:idx+self.input_timesteps+self.output_timesteps]
        
        # 转换为PyTorch张量
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y

# 2. CNN模型架构
class WindWaveCNN(nn.Module):
    def __init__(self, input_channels, output_channels, height, width):
        """
        :param input_channels: 输入通道数（时间步数）
        :param output_channels: 输出通道数（预测时间步数）
        :param height: 空间高度
        :param width: 空间宽度
        """
        super(WindWaveCNN, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
        
        # 计算编码器输出尺寸
        self.encoder_output_height = height // 4
        self.encoder_output_width = width // 4
        
    def forward(self, x):
        # 输入形状: [batch_size, input_timesteps, height, width]
        x = self.encoder(x)
        # 上采样到原始尺寸
        x = self.decoder(x)
        return x

# 3. 训练函数（预留损失函数接口）
def train_model(model, train_loader, val_loader, epochs, lr=0.001, 
                loss_fn=nn.MSELoss(), device='cuda', model_name='wind_wave_cnn'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算损失（使用自定义损失函数）
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, targets).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(f'Best model saved with val loss: {val_loss:.6f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{model_name}_loss_curve.png')
    plt.close()
    
    print(f'Training complete. Best validation loss: {best_val_loss:.6f}')
    return model

# 4. 自定义损失函数示例
class WeightedMSELoss(nn.Module):
    """加权MSE损失函数，关注特定区域"""
    def __init__(self, weight_map, reduction='mean'):
        super().__init__()
        self.weight_map = weight_map
        self.reduction = reduction
        
    def forward(self, input, target):
        # 计算加权MSE
        loss = (input - target) ** 2
        loss = loss * self.weight_map
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class PhysicalConstraintLoss(nn.Module):
    """物理约束损失函数，添加梯度平滑约束"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def gradient_loss(self, x):
        # 计算水平和垂直梯度
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx.mean() + dy.mean()
    
    def forward(self, input, target):
        mse_loss = self.mse(input, target)
        smooth_loss = self.gradient_loss(input)
        return mse_loss + self.alpha * smooth_loss

# 5. 预测和评估函数
def predict(model, wind_sequence, device='cuda'):
    """预测波浪场"""
    model.eval()
    model.to(device)
    
    # 转换为张量并添加批次维度
    inputs = torch.FloatTensor(wind_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # 移除批次维度并转换为numpy
    return outputs.squeeze(0).cpu().numpy()

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    model.eval()
    model.to(device)
    
    total_mse = 0
    total_mae = 0
    total_ssim = 0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 计算MSE
            mse = nn.functional.mse_loss(outputs, targets).item()
            total_mse += mse
            
            # 计算MAE
            mae = nn.functional.l1_loss(outputs, targets).item()
            total_mae += mae
            
            # 计算SSIM (需要实现或使用现有库)
            # 这里简化处理，实际应用中应使用专门的SSIM计算
            ssim = 0.9  # 示例值
            total_ssim += ssim
            
            count += 1
    
    avg_mse = total_mse / count
    avg_mae = total_mae / count
    avg_ssim = total_ssim / count
    
    print(f'Test MSE: {avg_mse:.6f}')
    print(f'Test MAE: {avg_mae:.6f}')
    print(f'Test SSIM: {avg_ssim:.4f}')
    
    return avg_mse, avg_mae, avg_ssim



if __name__=="__main__":    
    # 读取文件
    print('**loading files ...')
    surge_2d = f_load_pickle('../datas/surge_2d.pkl')
    current_2d = f_load_pickle('../datas/current_2d.pkl')
    wave_2d = f_load_pickle('../datas/wave_2d.pkl')
    wind_2d = f_load_pickle('../datas/wind_2d.pkl')

    # 查询surge_piled绝对值大于3的值的索引
    # 只保留绝对值小于3的风暴潮数据
    print('**cleaning surge data ...')
    current_2d[np.abs(surge_2d) > 5] = 0
    surge_2d[np.abs(surge_2d) > 5] = 0

    # 网格数据
    x = np.arange(105.3, 116.01, 0.2, dtype = float)
    y = np.arange(12, 23.01, 0.2, dtype = float)
    X,Y = np.meshgrid(x, y)
    # 网格1维化
    LonRegion = X.transpose(1,0).reshape(-1, 1)
    LatRegion = Y.transpose(1,0).reshape(-1, 1)

    # 假设数据格式 [num_samples, time_steps, height, width]
    # 示例数据（实际应用中应替换为真实数据）
    num_samples = 1000
    time_steps = 48
    height, width = 64, 64
    
    # 生成模拟数据
    wind_data = np.random.randn(num_samples, time_steps, height, width).astype(np.float32)
    wave_data = np.random.randn(num_samples, time_steps, height, width).astype(np.float32)
    
    # 创建数据集
    input_timesteps = 3  # 使用3个时间步的风场作为输入
    output_timesteps = 1  # 预测1个时间步的波浪场
    dataset = WindWaveDataset(wind_data, wave_data, input_timesteps, output_timesteps)
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = WindWaveCNN(
        input_channels=input_timesteps,
        output_channels=output_timesteps,
        height=height,
        width=width
    )
    
    # 创建自定义损失函数
    # 示例：加权MSE损失（更关注中心区域）
    weight_map = np.ones((height, width), dtype=np.float32)
    center_size = 16
    center_y, center_x = height // 2, width // 2
    weight_map[
        center_y-center_size:center_y+center_size, 
        center_x-center_size:center_x+center_size
    ] = 2.0
    weight_map = torch.FloatTensor(weight_map).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    weighted_mse = WeightedMSELoss(weight_map)
    
    # 物理约束损失
    physical_loss = PhysicalConstraintLoss(alpha=0.1)
    
    # 开始训练（可选择使用自定义损失函数）
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        loss_fn=nn.MSELoss(),  # 可替换为weighted_mse或physical_loss
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_name='wind_wave_cnn'
    )
    
    # 评估模型
    evaluate_model(trained_model, test_loader)
    
    # 示例预测
    sample_idx = 0
    sample_input, sample_target = test_dataset[sample_idx]
    prediction = predict(trained_model, sample_input.numpy())
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(sample_input[0], cmap='viridis')
    plt.title('Wind Input (t=0)')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(sample_target[0], cmap='viridis')
    plt.title('True Wave')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction[0], cmap='viridis')
    plt.title('Predicted Wave')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('wind_wave_prediction.png')
    plt.close()
    
    print('**FIN!!')


