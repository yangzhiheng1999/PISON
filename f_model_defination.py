import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for spatial attention in U-Net skip connections.
    Computes attention coefficients for the skip connection features.
    Adapted from Attention U-Net paper (Oktay et al., 2018).
    """
    def __init__(self, F_g, F_l, F_int):  # F_g: gating signal channels (from up), F_l: skip channels, F_int: intermediate channels
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.GELU()   # 修正：使用 GELU 替代 ReLU，不使用 inplace=True

    def forward(self, g, x):  # g: gating signal (low-res from up), x: skip (high-res)
        g1 = self.W_g(g)  # Downsample g to match x size? No, we upsample g via conv, but actually resize g to x's spatial size
        # Resize g to match x's spatial dimensions (upsample if needed)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # Additive attention
        psi = self.psi(psi)
        # Resize attention map back if needed, but since we apply to x, it's already matching
        return x * psi  # Element-wise multiplication for attended skip

class DoubleConv(nn.Module):
    """双卷积块：不变"""
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),     # 修正：使用 GELU 替代 ReLU，不使用 inplace=True
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),     # 修正：使用 GELU 替代 ReLU，不使用 inplace=True
            nn.Dropout2d(p=dropout_p)  # 2D Dropout，针对空间场
        )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    """
    Residual Block = DoubleConv + identity shortcut
    - GELU (non-inplace)
    - Dropout2d
    - 当 stride 或通道不匹配时使用 1×1 投影
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.2):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.final_gelu = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out = out + identity
        out = self.final_gelu(out)
        return out

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, stride=1, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res_block(x)      # 高分辨率 skip
        pooled = self.pool(skip)      # 低分辨率传给下一层
        return pooled, skip

class Up(nn.Module):
    """Up: 上采样 → Attention → cat → ResidualBlock → Dropout"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.res_block = ResidualBlock(out_channels * 2, out_channels, dropout_p=dropout_p)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 精确尺寸对齐
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        if diffY != 0 or diffX != 0:
            pad_top = max(0, diffY // 2)
            pad_bottom = max(0, diffY - diffY // 2)
            pad_left = max(0, diffX // 2)
            pad_right = max(0, diffX - diffX // 2)
            if diffY < 0 or diffX < 0:
                # crop x1
                h_start = max(0, -diffY // 2)
                w_start = max(0, -diffX // 2)
                x1 = x1[:, :, h_start:h_start + x2.size(2), w_start:w_start + x2.size(3)]
            else:
                x1 = F.pad(x1, (pad_left, pad_right, pad_top, pad_bottom))

        x2_att = self.attention(x1, x2)
        x = torch.cat([x2_att, x1], dim=1)
        x = self.res_block(x)
        return self.dropout(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[16, 32, 64, 128], dropout_p=0.2, env_type=[]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.env_type = env_type
        # print(f'UNet env_type: {self.env_type}')

        in_ch = in_channels
        for f in features:
            self.downs.append(Down(in_ch, f, dropout_p=dropout_p))
            in_ch = f

        # 瓶颈使用 ResidualBlock
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2, dropout_p=dropout_p)

        # 上采样路径
        for f in reversed(features):
            self.ups.append(Up(f * 2, f, dropout_p=dropout_p * 0.5))  # 上采样 dropout 可减半

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.final_norm = nn.BatchNorm2d(out_channels)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # GELU 初始化使用 ReLU 的 Kaiming（官方推荐）
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)

        x = self.final_conv(x)
        x = self.final_norm(x)
        if self.env_type == 'wave':
            return torch.sigmoid(x)  # 或 return x （若后续有归一化）
        else:
            return torch.tanh(x)



# 在U-Net模型定义后，训练循环前
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

scaler = GradScaler()  # 混合精度缩放器

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast():  # 关键！
            output = model(data)
            # 训练循环中，加权大值区域
            loss = criterion(output, target)

        # NaN 监控
        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}! Check gradients/data.")
            break
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        '''loss.backward()
        # 梯度裁剪（防爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()'''
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else float('inf')

# 验证函数
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 训练循环中，加权大值区域
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                print("Warning: NaN val loss!")
                return float('inf')
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else float('inf')

class UNetBoosting(nn.Module):
    def __init__(self, num_models=5, in_channels=2, out_channels=1, learning_rate=1e-3, shrinkage=0.1, features=[16, 32, 64, 128], device=[], env_type=[]):
        super(UNetBoosting, self).__init__()
        self.num_models = num_models
        self.shrinkage = shrinkage  # 收缩因子，控制后续模型贡献
        self.models = nn.ModuleList()  # 存储多个U-Net
        self.optimizers = []  # 每个U-Net独立优化器
        self.env_type = env_type
        self.device = device
        for i in range(num_models):
            model = UNet(in_channels=in_channels, out_channels=out_channels, features=features, env_type=self.env_type).to(self.device)
            # print(f'Env type for model {i}: {self.env_type}')
            model.apply(model.init_weights)  # 初始化
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 后续模型lr不缩小
            self.models.append(model)
            self.optimizers.append(optimizer)
    
    def forward(self, x):
        # 预测：sum所有U-Net输出（第一个完整，后续乘shrinkage）
        pred = self.models[0](x)
        for model in self.models[1:]:
            pred += self.shrinkage * model(x)
        return pred
    
    def train_ensemble(self, train_loader, val_loader, criterion_1, scheduler_func, num_epochs=100, patience=7, device='cuda'):
        train_losses = [[] for _ in range(self.num_models)]
        val_losses = [[] for _ in range(self.num_models)]

        # === 1. 提取原始数据（只做一次）===
        train_subset = train_loader.dataset
        val_subset = val_loader.dataset
        full_dataset = train_subset.dataset  # 原始 NumpyDataset

        train_indices = train_subset.indices
        val_indices = val_subset.indices

        # 原始 numpy → Tensor（只转一次）
        wind_tensor = torch.from_numpy(full_dataset.wind_data.astype(np.float32)).to(device)
        # 根据维度判断是否需要 unsqueeze
        output_data_np = full_dataset.output_data.astype(np.float32)
        if output_data_np.ndim == 3: # 如果是 (N, H, W) -> (N, 1, H, W)
            output_tensor = torch.from_numpy(output_data_np).unsqueeze(1).to(device)
        else: # 如果已经是 (N, C, H, W) -> 保持原样
            output_tensor = torch.from_numpy(output_data_np).to(device)

        train_wind = wind_tensor[train_indices]
        val_wind = wind_tensor[val_indices]
        train_targets = output_tensor[train_indices]
        val_targets = output_tensor[val_indices]

        # === 2. 残差训练循环 ===
        current_train_res = train_targets.clone()
        current_val_res = val_targets.clone()

        for i in range(self.num_models):
            print(f"\nTraining U-Net {i+1}/{self.num_models}")
            counter = 0
            best_val_loss = float('inf')    # 源代码在循环外重置

            # 训练损失函数
            if i == 0:
                criterion = criterion_1  # 第一个模型使用指定损失函数
            else:
                print(f'env type: {self.env_type}')
                if self.env_type == 'current':
                    criterion = nn.MSELoss()  # 后续模型使用 MSE
                elif self.env_type == 'surge':
                    criterion = nn.L1Loss()  # 后续模型使用 MSE
                elif self.env_type == 'wave':
                    criterion = nn.L1Loss()   # 后续模型使用 MAE

            model = self.models[i]
            optimizer = self.optimizers[i]
            scheduler = scheduler_func(optimizer)

            # 使用当前残差创建临时 DataLoader（不修改原始 train_loader）
            if i == 0:
                train_res_dataset = TensorDataset(train_wind, current_train_res)
                val_res_dataset = TensorDataset(val_wind, current_val_res)
            else:
                train_res_dataset = TensorDataset(train_wind, current_train_res / self.shrinkage)
                val_res_dataset = TensorDataset(val_wind, current_val_res / self.shrinkage)

            train_loader_res = DataLoader(train_res_dataset, batch_size=train_loader.batch_size, shuffle=True)
            val_loader_res = DataLoader(val_res_dataset, batch_size=val_loader.batch_size, shuffle=False)

            # 训练当前 U-Net
            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader_res, criterion, optimizer, device)
                val_loss = val_epoch(model, val_loader_res, criterion, device)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 5e-5:  # 小阈值防止频繁保存
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(self.state_dict(), f'best_boosting_model_{i}.pth')
                else:
                    counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

                print(f'U-Net {i+1} Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}')

                train_losses[i].append(train_loss)
                val_losses[i].append(val_loss)

            # === 3. 更新残差（为下一个 U-Net 准备）===
            if i < self.num_models:
                pred_train = self._predict_single_model(model, train_wind, train_loader.batch_size, device)
                pred_val = self._predict_single_model(model, val_wind, val_loader.batch_size, device)
                if i > 0:
                    pred_train = pred_train * self.shrinkage
                    pred_val = pred_val * self.shrinkage

                current_train_res = current_train_res - pred_train
                current_val_res = current_val_res - pred_val

        return train_losses, val_losses
    
    def _predict_single_model(self, model, wind_tensor, batch_size, device):
        model.eval()
        preds = []
        total = len(wind_tensor)
        with torch.no_grad():
            for i in range(0, total, batch_size):
                # 直接切片，不产生额外的大型数据拷贝
                batch_x = wind_tensor[i : i + batch_size].to(device)
                preds.append(model(batch_x))
        return torch.cat(preds, dim=0)